"""
VAL-XSP-001: Cross-Species Kendall's Tau Conservation
======================================================
Tests whether opcode (primary_function) frequency distributions are
conserved across species, correlating pairwise Kendall's tau with
evolutionary divergence time.

Method:
  1. Load vocabulary files for all 6 species
  2. Compute opcode (primary_function) frequency distribution per species
  3. Compute pairwise Kendall's tau for all species pairs on shared opcodes
  4. Correlate tau values with evolutionary divergence times
  5. Plot tau vs divergence and opcode frequency heatmap

Input: server/data/{species}/vocabulary.csv (6 species)
       server/data/species_registry.json
Output: validation/VAL-XSP-001_cross_species_tau.json
        validation/VAL-XSP-001_summary.md
        validation/VAL-XSP-001_tau_vs_divergence.png
"""

import os, json, csv, hashlib, time
from collections import Counter
from datetime import datetime, timezone

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "server", "data")
REGISTRY_PATH = os.path.join(DATA_DIR, "species_registry.json")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42

DIVERGENCE_MYA = {
    "human": 0, "mouse": 90, "zebrafish": 450,
    "celegans": 600, "fly": 700, "yeast": 1000,
    "arabidopsis": 1500, "ecoli": 2000, "halobacterium": 3500,
}


def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def file_rows(path):
    with open(path) as f:
        return sum(1 for _ in f) - 1


def load_vocab(species_id):
    path = os.path.join(DATA_DIR, species_id, "vocabulary.csv")
    words = []
    if not os.path.exists(path):
        return words, path
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            words.append({
                "hex": row.get("word_hex", ""),
                "function": row.get("primary_function", "Unclassified").strip(),
                "enrichment": float(row.get("token_enrichment", 0)),
            })
    return words, path


def opcode_freq_distribution(words):
    counts = Counter(w["function"] for w in words if w["function"] != "Unclassified")
    total = sum(counts.values())
    if total == 0:
        return {}
    return {f: c / total for f, c in counts.items()}


def pairwise_tau(freq_a, freq_b):
    shared = sorted(set(freq_a.keys()) & set(freq_b.keys()))
    if len(shared) < 2:
        return float("nan"), float("nan"), len(shared)
    vals_a = [freq_a[f] for f in shared]
    vals_b = [freq_b[f] for f in shared]
    try:
        tau, p = stats.kendalltau(vals_a, vals_b)
    except Exception:
        return float("nan"), float("nan"), len(shared)
    return tau, p, len(shared)


def main():
    print("VAL-XSP-001: Cross-Species Kendall's Tau Conservation")
    print("=" * 60)

    t0 = time.time()

    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    species_list = [s["id"] for s in registry["species"]]

    all_species = {}
    species_hashes = {}
    species_row_counts = {}
    species_word_counts = {}

    for sid in species_list:
        words, path = load_vocab(sid)
        all_species[sid] = words
        if os.path.exists(path):
            species_hashes[sid] = file_hash(path)
            species_row_counts[sid] = file_rows(path)
        species_word_counts[sid] = len(words)
        print(f"  {sid}: {len(words)} words")

    freq_dists = {}
    for sid in species_list:
        freq_dists[sid] = opcode_freq_distribution(all_species[sid])
        print(f"  {sid}: {len(freq_dists[sid])} classified opcodes")

    valid_species = [sid for sid in species_list if len(freq_dists[sid]) > 0]
    print(f"\nSpecies with opcodes: {valid_species} ({len(valid_species)}/{len(species_list)})")
    excluded = [sid for sid in species_list if len(freq_dists[sid]) == 0]
    if excluded:
        print(f"  Excluded (no classified opcodes): {excluded}")

    pairs = []
    taus = []
    divergences = []
    for i, s1 in enumerate(valid_species):
        for s2 in valid_species[i+1:]:
            tau, p, n_shared = pairwise_tau(freq_dists[s1], freq_dists[s2])
            div = abs(DIVERGENCE_MYA.get(s1, 0) - DIVERGENCE_MYA.get(s2, 0))
            if div == 0:
                div = max(DIVERGENCE_MYA.get(s1, 0), DIVERGENCE_MYA.get(s2, 0))
            pairs.append({
                "species_a": s1, "species_b": s2,
                "tau": round(tau, 4) if not np.isnan(tau) else None,
                "p_value": float(f"{p:.6e}") if not np.isnan(p) else None,
                "n_shared_opcodes": n_shared,
                "divergence_mya": div,
            })
            if not np.isnan(tau):
                taus.append(tau)
                divergences.append(div)
            print(f"  {s1} vs {s2}: tau={tau:.4f}, p={p:.2e}, shared={n_shared}, div={div} Mya")

    if len(taus) >= 3:
        corr_tau, corr_p = stats.kendalltau(divergences, taus)
        spear_r, spear_p = stats.spearmanr(divergences, taus)
    else:
        corr_tau, corr_p = float("nan"), float("nan")
        spear_r, spear_p = float("nan"), float("nan")

    all_opcodes = sorted(set().union(*[set(fd.keys()) for fd in freq_dists.values()]))
    n_op = len(all_opcodes)

    rng = np.random.RandomState(SEED)
    null_corrs = []
    N_PERMS = 1000
    if len(taus) >= 3:
        print(f"\nRunning {N_PERMS} permutations for tau-divergence correlation null...")
        for perm in range(N_PERMS):
            shuffled_taus = list(taus)
            rng.shuffle(shuffled_taus)
            null_tau, _ = stats.kendalltau(divergences, shuffled_taus)
            if not np.isnan(null_tau):
                null_corrs.append(null_tau)
        null_corrs = np.array(null_corrs)
        if not np.isnan(corr_tau):
            if corr_tau < 0:
                p_perm = (np.sum(null_corrs <= corr_tau) + 1) / (len(null_corrs) + 1)
            else:
                p_perm = (np.sum(null_corrs >= corr_tau) + 1) / (len(null_corrs) + 1)
        else:
            p_perm = 1.0
    else:
        null_corrs = np.array([])
        p_perm = 1.0

    elapsed = time.time() - t0

    sig = not np.isnan(corr_tau) and p_perm < 0.05
    conclusion = (
        f"{'SIGNIFICANT' if sig else 'NOT SIGNIFICANT'}: "
        f"Tau-divergence correlation: Kendall's tau={corr_tau:.4f} "
        f"(Spearman r={spear_r:.4f}, p_analytic={corr_p:.2e}, p_permutation={p_perm:.4e}). "
        f"Tested {len(pairs)} species pairs across {len(valid_species)} species. "
        f"Mean pairwise tau={np.mean(taus):.4f} ({len(all_opcodes)} unique opcodes). "
        f"{'Negative correlation indicates opcode usage diverges with evolutionary distance.' if corr_tau < 0 else 'Positive correlation indicates opcode conservation.'}"
    )

    results = {
        "test_id": "VAL-XSP-001",
        "test_name": "Cross-Species Kendall's Tau Conservation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "n_species": len(valid_species),
            "species_list": valid_species,
            "n_pairs": len(pairs),
            "seed": SEED,
            "n_permutations": N_PERMS,
            "metric": "opcode (primary_function) frequency distribution",
            "divergence_source": "established evolutionary divergence times (Mya)",
        },
        "provenance": {
            "vocabulary_files": {sid: {
                "path": os.path.join(DATA_DIR, sid, "vocabulary.csv"),
                "md5": species_hashes.get(sid, "N/A"),
                "rows": species_row_counts.get(sid, 0),
                "words": species_word_counts.get(sid, 0),
            } for sid in species_list},
            "registry_file": REGISTRY_PATH,
            "registry_md5": file_hash(REGISTRY_PATH),
        },
        "results": {
            "pairwise_tau": pairs,
            "tau_divergence_correlation": {
                "kendall_tau": round(corr_tau, 4) if not np.isnan(corr_tau) else None,
                "kendall_p_analytic": float(f"{corr_p:.6e}") if not np.isnan(corr_p) else None,
                "spearman_r": round(spear_r, 4) if not np.isnan(spear_r) else None,
                "spearman_p": float(f"{spear_p:.6e}") if not np.isnan(spear_p) else None,
                "p_permutation": float(f"{p_perm:.6e}"),
            },
            "summary_stats": {
                "mean_tau": round(float(np.mean(taus)), 4) if taus else None,
                "std_tau": round(float(np.std(taus)), 4) if taus else None,
                "min_tau": round(float(np.min(taus)), 4) if taus else None,
                "max_tau": round(float(np.max(taus)), 4) if taus else None,
                "n_unique_opcodes": n_op,
            },
            "opcode_frequencies": {sid: freq_dists[sid] for sid in valid_species},
        },
        "conclusion": conclusion,
        "elapsed_seconds": round(elapsed, 1),
    }

    json_path = os.path.join(OUT_DIR, "VAL-XSP-001_cross_species_tau.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("VAL-XSP-001: Cross-Species Opcode Conservation", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    if taus and divergences:
        ax.scatter(divergences, taus, s=60, c="#1565C0", alpha=0.8, edgecolors="white", linewidth=0.5, zorder=5)
        for pair in pairs:
            if pair["tau"] is not None:
                ax.annotate(f"{pair['species_a'][:3]}-{pair['species_b'][:3]}",
                           (pair["divergence_mya"], pair["tau"]),
                           fontsize=6, alpha=0.7, textcoords="offset points", xytext=(5, 5))
        if len(taus) >= 3:
            z = np.polyfit(divergences, taus, 1)
            p_fit = np.poly1d(z)
            x_fit = np.linspace(min(divergences), max(divergences), 100)
            ax.plot(x_fit, p_fit(x_fit), "r--", alpha=0.5, label=f"Linear fit")
    ax.set_xlabel("Divergence time (Mya)")
    ax.set_ylabel("Kendall's tau")
    ax.set_title(f"Tau vs Divergence (r={corr_tau:.3f}, p_perm={p_perm:.3e})")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
    ax.legend()

    ax = axes[0, 1]
    if len(null_corrs) > 0:
        ax.hist(null_corrs, bins=50, color="#90CAF9", edgecolor="white", alpha=0.8, label="Null distribution")
        if not np.isnan(corr_tau):
            ax.axvline(x=corr_tau, color="#D32F2F", linewidth=2, linestyle="--",
                       label=f"Observed ({corr_tau:.4f})")
        ax.set_xlabel("Kendall's tau (divergence vs tau)")
        ax.set_ylabel("Count")
        ax.set_title(f"Correlation Null Model (p_perm={p_perm:.2e})")
        ax.legend()

    ax = axes[1, 0]
    n_sp = len(valid_species)
    if n_sp > 0 and n_op > 0:
        display_ops = all_opcodes
        matrix = np.zeros((n_sp, len(display_ops)))
        for i, sid in enumerate(valid_species):
            for j, op in enumerate(display_ops):
                matrix[i, j] = freq_dists[sid].get(op, 0)
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(n_sp))
        ax.set_yticklabels(valid_species, fontsize=8)
        ax.set_xticks(range(len(display_ops)))
        fs = 6 if len(display_ops) <= 30 else 4
        ax.set_xticklabels([op[:12] for op in display_ops], rotation=90, fontsize=fs)
        ax.set_title(f"Opcode Frequency Heatmap (all {n_op} opcodes)")
        plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1, 1]
    if n_sp > 1:
        tau_matrix = np.ones((n_sp, n_sp))
        for pair in pairs:
            if pair["tau"] is not None:
                i = valid_species.index(pair["species_a"])
                j = valid_species.index(pair["species_b"])
                tau_matrix[i, j] = pair["tau"]
                tau_matrix[j, i] = pair["tau"]
        im = ax.imshow(tau_matrix, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_xticks(range(n_sp))
        ax.set_xticklabels(valid_species, rotation=45, fontsize=8)
        ax.set_yticks(range(n_sp))
        ax.set_yticklabels(valid_species, fontsize=8)
        for i in range(n_sp):
            for j in range(n_sp):
                ax.text(j, i, f"{tau_matrix[i,j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title("Pairwise Kendall's Tau Matrix")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, "VAL-XSP-001_tau_vs_divergence.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Plot saved to {png_path}")

    md = f"""# VAL-XSP-001: Cross-Species Kendall's Tau Conservation

## Purpose
Test whether opcode (primary_function) frequency distributions are conserved
across species and whether conservation correlates with evolutionary divergence
time. Under the biological hypothesis, closely related species should share
more similar opcode usage patterns.

## Method
1. Load vocabulary files for {len(species_list)} species
2. Compute opcode (primary_function) frequency distribution per species
3. Compute pairwise Kendall's tau for all species pairs on shared opcodes
4. Correlate tau values with evolutionary divergence times (Mya)
5. Permutation test ({N_PERMS} rounds): shuffle tau-divergence pairs

## Materials
| Species | Words | Opcodes | Divergence (Mya) | File MD5 |
|---------|-------|---------|-------------------|----------|
"""
    for sid in species_list:
        md += f"| {sid} | {species_word_counts.get(sid, 0)} | {len(freq_dists.get(sid, {}))} | {DIVERGENCE_MYA.get(sid, '?')} | `{species_hashes.get(sid, 'N/A')[:12]}...` |\n"

    md += f"""
## Results

### Pairwise Tau
| Species A | Species B | Tau | p-value | Shared Opcodes | Divergence |
|-----------|-----------|-----|---------|----------------|------------|
"""
    for pair in pairs:
        tau_str = f"{pair['tau']:.4f}" if pair["tau"] is not None else "N/A"
        p_str = f"{pair['p_value']:.2e}" if pair["p_value"] is not None else "N/A"
        md += f"| {pair['species_a']} | {pair['species_b']} | {tau_str} | {p_str} | {pair['n_shared_opcodes']} | {pair['divergence_mya']} Mya |\n"

    md += f"""
### Tau-Divergence Correlation
| Metric | Value |
|--------|-------|
| Kendall's tau | {f"{corr_tau:.4f}" if not np.isnan(corr_tau) else "N/A"} |
| Spearman r | {f"{spear_r:.4f}" if not np.isnan(spear_r) else "N/A"} |
| p (analytic) | {f"{corr_p:.2e}" if not np.isnan(corr_p) else "N/A"} |
| p (permutation) | {p_perm:.2e} |
| Mean pairwise tau | {f"{np.mean(taus):.4f}" if taus else "N/A"} |

## Interpretation
{conclusion}

## Graphs
![Tau vs Divergence](VAL-XSP-001_tau_vs_divergence.png)

## Provenance
| Claim | Source |
|-------|--------|
| {len(species_list)} species | `server/data/species_registry.json` (MD5: `{file_hash(REGISTRY_PATH)}`) |
| {n_op} opcodes | Union of primary_function across species vocabularies |
| Divergence times | Established evolutionary estimates (human=0, mouse=90, zebrafish=450, fly=700, yeast=1000, ecoli=2000 Mya) |

## Runtime
{elapsed:.1f} seconds
"""
    md_path = os.path.join(OUT_DIR, "VAL-XSP-001_summary.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Summary saved to {md_path}")
    print(f"\n{conclusion}")


if __name__ == "__main__":
    main()
