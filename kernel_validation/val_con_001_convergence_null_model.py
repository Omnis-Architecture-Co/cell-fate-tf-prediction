"""
VAL-CON-001: Vocabulary Convergence Null Model
================================================
Tests whether the functional labels assigned to VALDICT vocabulary tokens
produce meaningful protein-level overlap — i.e., whether proteins' tokens
converge on consistent functions under the real VALDICT mapping vs random
label assignment.

Overlap metric: For each protein, compute the fraction of token-pair
comparisons where both tokens share the same VALDICT function label.
This is the per-protein functional overlap rate (mathematically equivalent
to HHI). Average across all proteins to get the mean functional overlap.

Null model: Shuffle primary_function labels across all 55,641 VALDICT
words (1,000 permutations), breaking the real token→function mapping.
Under random assignment, proteins' tokens get diverse random functions
→ low overlap. Under real labels, biologically coherent mapping → high
overlap.

Data: valdict_extended (55,641 words), protein_tokens_v2 (1.85M rows,
      93,465 proteins, 125,468 distinct tokens)
      55,641 VALDICT tokens are 100% contained in protein_tokens_v2.
"""

import os, json, time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import psycopg2
import numpy as np
from scipy.sparse import coo_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DB_URL = os.environ.get("BETA_DATABASE_URL") or os.environ.get("DATABASE_URL")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_PERMUTATIONS = 1000
SEED = 42


def load_data():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("SELECT token_hex, primary_function FROM valdict_extended")
    valdict_rows = cur.fetchall()
    cur.execute("SELECT uniprot_id, token_hex FROM protein_tokens_v2")
    protein_rows = cur.fetchall()
    cur.execute("SELECT COUNT(*) FROM valdict_extended")
    valdict_total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT token_hex) FROM protein_tokens_v2")
    pt_distinct = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT uniprot_id) FROM protein_tokens_v2")
    n_proteins_db = cur.fetchone()[0]
    conn.close()
    return valdict_rows, protein_rows, valdict_total, pt_distinct, n_proteins_db


def build_matrix(func_indices_arr, protein_valdict_indices, protein_ids_arr, n_funcs, n_proteins):
    mapped = func_indices_arr[protein_valdict_indices]
    mat = coo_matrix((np.ones(len(mapped), dtype=np.int32), (protein_ids_arr, mapped)),
                     shape=(n_proteins, n_funcs))
    return mat.toarray()


def compute_overlap(counts_matrix):
    totals = counts_matrix.sum(axis=1)
    valid = totals >= 2
    vc = counts_matrix[valid].astype(np.float64)
    vt = totals[valid].astype(np.float64)
    fracs = vc / vt[:, None]
    overlaps = np.sum(fracs ** 2, axis=1)
    return float(overlaps.mean()) if len(overlaps) > 0 else 0.0, int(valid.sum()), overlaps


def main():
    print("VAL-CON-001: Vocabulary Convergence Null Model")
    print("=" * 60)
    t0 = time.time()

    valdict_rows, protein_rows, valdict_total, pt_distinct, n_proteins_db = load_data()
    print(f"Loaded: {len(valdict_rows)} VALDICT words, {len(protein_rows)} protein-token rows")

    token_hexes = [r[0] for r in valdict_rows]
    functions = [r[1] or "Unclassified" for r in valdict_rows]
    n_words = len(valdict_rows)

    all_func_labels = sorted(set(functions))
    func_to_idx = {f: i for i, f in enumerate(all_func_labels)}
    n_funcs = len(all_func_labels)
    func_indices = np.array([func_to_idx[f] for f in functions], dtype=np.int32)

    token_to_valdict_idx = {token_hexes[i]: i for i in range(n_words)}

    protein_tokens_by_id = defaultdict(list)
    for uniprot_id, token_hex in protein_rows:
        vidx = token_to_valdict_idx.get(token_hex, -1)
        if vidx >= 0:
            protein_tokens_by_id[uniprot_id].append(vidx)

    protein_ids = sorted(protein_tokens_by_id.keys())
    pid_to_int = {pid: i for i, pid in enumerate(protein_ids)}
    n_proteins = len(protein_ids)
    print(f"Proteins with VALDICT-matched tokens: {n_proteins}")

    all_valdict_indices = []
    all_protein_int_ids = []
    for pid in protein_ids:
        indices = protein_tokens_by_id[pid]
        all_valdict_indices.extend(indices)
        all_protein_int_ids.extend([pid_to_int[pid]] * len(indices))
    pvi = np.array(all_valdict_indices, dtype=np.int32)
    pia = np.array(all_protein_int_ids, dtype=np.int32)
    mean_tokens = len(all_valdict_indices) / n_proteins if n_proteins > 0 else 0
    print(f"Total matched token-slots: {len(all_valdict_indices)} (mean {mean_tokens:.1f}/protein)")

    print("\nComputing observed functional overlap...")
    obs_matrix = build_matrix(func_indices, pvi, pia, n_funcs, n_proteins)
    obs_overlap, obs_n_valid, obs_overlaps = compute_overlap(obs_matrix)
    obs_median = float(np.median(obs_overlaps)) if len(obs_overlaps) > 0 else 0.0
    obs_high = int(np.sum(obs_overlaps > 0.5))
    print(f"  Observed functional overlap: {obs_overlap:.6f}")
    print(f"  Median overlap: {obs_median:.6f}")
    print(f"  Proteins with overlap > 50%: {obs_high}/{obs_n_valid}")

    rng = np.random.RandomState(SEED)
    null_overlaps = np.empty(N_PERMUTATIONS)

    print(f"\nRunning {N_PERMUTATIONS} permutations (randomize labels across {n_words} token hashes)...")
    for perm in range(N_PERMUTATIONS):
        if perm % 200 == 0:
            print(f"  Permutation {perm}... ({time.time() - t0:.0f}s)")

        perm_func = func_indices.copy()
        rng.shuffle(perm_func)

        perm_matrix = build_matrix(perm_func, pvi, pia, n_funcs, n_proteins)
        null_overlaps[perm], _, _ = compute_overlap(perm_matrix)

    p_value = (np.sum(null_overlaps >= obs_overlap) + 1) / (N_PERMUTATIONS + 1)
    z_score = (obs_overlap - null_overlaps.mean()) / (null_overlaps.std() + 1e-10)

    elapsed = time.time() - t0

    sig = p_value < 0.05
    conclusion = (
        f"{'SIGNIFICANT' if sig else 'NOT SIGNIFICANT'}: "
        f"Observed functional overlap={obs_overlap:.6f} "
        f"(null mean={null_overlaps.mean():.6f} +/- {null_overlaps.std():.6f}, "
        f"z={z_score:.2f}, p={p_value:.4e}). "
        f"Randomly assigning vocabulary labels to the {n_words} token hashes and "
        f"measuring overlap across {n_proteins} proteins ({pt_distinct} distinct tokens). "
        f"The real VALDICT labeling produces {obs_overlap/null_overlaps.mean():.1f}x higher "
        f"functional overlap than random label assignment."
    )
    print(f"\n{conclusion}")

    results = {
        "test_id": "VAL-CON-001",
        "test_name": "Vocabulary Convergence Null Model",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "n_permutations": N_PERMUTATIONS,
            "seed": SEED,
            "n_valdict_words": n_words,
            "n_protein_tokens_distinct": pt_distinct,
            "n_proteins": n_proteins,
            "n_functions": n_funcs,
            "null_model": "randomly assign vocabulary labels to token hashes (shuffle primary_function across 55,641 VALDICT words), recompute functional overlap across 93K proteins",
        },
        "provenance": {
            "valdict_table": "valdict_extended",
            "valdict_rows": valdict_total,
            "protein_tokens_table": "protein_tokens_v2",
            "protein_tokens_distinct": pt_distinct,
            "n_proteins_db": n_proteins_db,
            "database": "BETA_DATABASE_URL",
            "valdict_token_coverage": "100% of VALDICT tokens appear in protein_tokens_v2",
        },
        "results": {
            "observed": {
                "functional_overlap": round(obs_overlap, 6),
                "median_overlap": round(obs_median, 6),
                "high_overlap_count": obs_high,
                "proteins_tested": obs_n_valid,
            },
            "null_distribution": {
                "overlap_mean": round(float(null_overlaps.mean()), 6),
                "overlap_std": round(float(null_overlaps.std()), 6),
                "overlap_max": round(float(null_overlaps.max()), 6),
            },
            "p_value": float(f"{p_value:.6e}"),
            "z_score": round(z_score, 2),
            "enrichment_ratio": round(obs_overlap / null_overlaps.mean(), 2),
        },
        "conclusion": conclusion,
        "elapsed_seconds": round(elapsed, 1),
    }

    json_path = os.path.join(OUT_DIR, "VAL-CON-001_convergence_null_model.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON saved to {json_path}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"VAL-CON-001: Vocabulary Convergence — Functional Overlap\n"
        f"Random assignment of vocabulary labels to {n_words} token hashes, "
        f"{n_proteins} proteins, {pt_distinct} tokens",
        fontsize=13, fontweight="bold"
    )

    ax = axes[0, 0]
    ax.hist(null_overlaps, bins=50, color="#90CAF9", edgecolor="white", alpha=0.8, label="Null distribution")
    ax.axvline(x=obs_overlap, color="#D32F2F", linewidth=2, linestyle="--",
               label=f"Observed ({obs_overlap:.4f})")
    ax.set_xlabel("Mean functional overlap")
    ax.set_ylabel("Count")
    ax.set_title(f"Observed Overlap vs Null (p={p_value:.2e}, z={z_score:.1f})")
    ax.legend()
    ax.annotate(f"p = {p_value:.4e}\nz = {z_score:.1f}\n{obs_overlap/null_overlaps.mean():.1f}x enrichment",
                xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax = axes[0, 1]
    ax.hist(obs_overlaps, bins=50, color="#A5D6A7", edgecolor="white", alpha=0.8)
    ax.axvline(x=obs_overlap, color="#D32F2F", linewidth=2, linestyle="--",
               label=f"Mean ({obs_overlap:.4f})")
    ax.set_xlabel("Per-protein functional overlap")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Observed Overlap (n={len(obs_overlaps)})")
    ax.legend()

    ax = axes[1, 0]
    func_freq = Counter(functions)
    sorted_funcs = func_freq.most_common(20)
    fn_names = [f[0][:15] for f in sorted_funcs]
    fn_vals = [f[1] for f in sorted_funcs]
    ax.barh(range(len(fn_names)), fn_vals, color="#1565C0", alpha=0.7)
    ax.set_yticks(range(len(fn_names)))
    ax.set_yticklabels(fn_names, fontsize=7)
    ax.set_xlabel("VALDICT word count")
    ax.set_title(f"Vocabulary Label Distribution ({n_funcs} labels)")
    ax.invert_yaxis()

    ax = axes[1, 1]
    valid_sizes = []
    totals = obs_matrix.sum(axis=1)
    for i in range(n_proteins):
        if totals[i] >= 2:
            valid_sizes.append(totals[i])
    valid_sizes = np.array(valid_sizes)
    sample_n = min(5000, len(obs_overlaps))
    ax.scatter(valid_sizes[:sample_n], obs_overlaps[:sample_n], s=5, alpha=0.15, c="#1565C0")
    ax.set_xlabel("Classified tokens per protein")
    ax.set_ylabel("Functional overlap")
    ax.set_title(f"Overlap vs Protein Size (first {sample_n})")

    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, "VAL-CON-001_convergence_heatmap.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Plot saved to {png_path}")

    md = f"""# VAL-CON-001: Vocabulary Convergence Null Model

## Purpose
Test whether the functional labels assigned to VALDICT vocabulary tokens
produce meaningful functional overlap across proteins. Under the real
VALDICT mapping, proteins' tokens should converge on consistent functions.
Under random label assignment, overlap should be low.

## Method
1. Load all {n_words} VALDICT words from `valdict_extended` (token_hex -> primary_function).
   All {n_words} VALDICT token hashes are present in protein_tokens_v2 (100% coverage).
2. Load protein-token assignments from `protein_tokens_v2` ({n_proteins} proteins,
   {pt_distinct} distinct tokens, ~{mean_tokens:.0f} tokens/protein).
3. For each protein, look up all its tokens in VALDICT and compute the functional
   overlap: the probability that two randomly chosen tokens from the same protein
   share the same vocabulary label (function). High overlap = consistent labeling.
4. Null model ({N_PERMUTATIONS} permutations): randomly assign vocabulary labels
   to the {n_words} token hashes (shuffle primary_function across all VALDICT words),
   recompute mean functional overlap across all proteins.
5. Empirical p-value: fraction of null overlaps >= observed.

## Materials
| Item | Value |
|------|-------|
| VALDICT table | valdict_extended |
| VALDICT words (token hashes) | {n_words} |
| Protein table | protein_tokens_v2 |
| Proteins | {n_proteins} |
| Distinct protein tokens | {pt_distinct} |
| Vocabulary labels (functions) | {n_funcs} |
| Permutations | {N_PERMUTATIONS} |
| Seed | {SEED} |

## Results

### Functional Overlap: Observed vs Null
| Metric | Observed | Null Mean +/- SD | z-score | p-value |
|--------|----------|------------------|---------|---------|
| Mean overlap | {obs_overlap:.6f} | {null_overlaps.mean():.6f} +/- {null_overlaps.std():.6f} | {z_score:.2f} | {p_value:.2e} |
| Median overlap | {obs_median:.6f} | -- | -- | -- |
| Overlap > 50% | {obs_high}/{obs_n_valid} | -- | -- | -- |
| Enrichment | {obs_overlap/null_overlaps.mean():.1f}x | -- | -- | -- |

## Interpretation
{conclusion}

## Graph
![Convergence Heatmap](VAL-CON-001_convergence_heatmap.png)

## Provenance
| Claim | Source |
|-------|--------|
| {n_words} VALDICT token hashes | valdict_extended, BETA_DATABASE_URL ({valdict_total} rows) |
| {n_proteins} proteins | protein_tokens_v2, BETA_DATABASE_URL ({n_proteins_db} total) |
| {pt_distinct} distinct tokens | protein_tokens_v2 distinct token_hex |
| Overlap={obs_overlap:.6f} | P(2 random tokens from same protein share label) |
| p={p_value:.2e} | {N_PERMUTATIONS} permutations, seed={SEED} |

## Runtime
{elapsed:.1f} seconds
"""
    md_path = os.path.join(OUT_DIR, "VAL-CON-001_summary.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Summary saved to {md_path}")


if __name__ == "__main__":
    main()
