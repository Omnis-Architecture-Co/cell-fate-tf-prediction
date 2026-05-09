"""
VAL-PRM-001: Primitive Recurrence Permutation Test
===================================================
Tests whether genome primitive recurrence patterns exceed what is expected
when function sequences are shuffled across chromosomes.

Method:
  1. Load programs from exports/programs_annotated.csv (4,936 programs)
  2. Extract function_sequence per program and chromosome assignment
  3. Null model: shuffle chromosome assignments of programs while preserving
     per-chromosome program counts, then recount recurrence (how many times
     each unique function_sequence appears)
  4. Compare observed recurrence distribution to null (1,000 permutations)

Input: exports/programs_annotated.csv (4,936 programs)
       exports/primitive_annotations_complete.csv (116 primitives)
Output: validation/VAL-PRM-001_primitive_recurrence.json
        validation/VAL-PRM-001_summary.md
        validation/VAL-PRM-001_recurrence_distribution.png
"""

import os, json, csv, random, hashlib, time
from collections import Counter
from datetime import datetime, timezone

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROGRAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "exports", "programs_annotated.csv")
PRIMITIVES_PATH = os.path.join(os.path.dirname(__file__), "..", "exports", "primitive_annotations_complete.csv")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_PERMUTATIONS = 1000
SEED = 42


def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def file_rows(path):
    with open(path) as f:
        return sum(1 for _ in f) - 1


def load_programs():
    programs = []
    with open(PROGRAMS_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            programs.append({
                "chromosome": row.get("chromosome", ""),
                "function_sequence": row.get("function_sequence", ""),
                "recurrence": int(row.get("recurrence", 0)),
                "length_bytes": int(row.get("length_bytes", 0)),
            })
    return programs


def load_primitives():
    primitives = []
    with open(PRIMITIVES_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            primitives.append({
                "rank": int(row.get("rank", 0)),
                "function_sequence": row.get("function_sequence", ""),
                "recurrence": int(row.get("recurrence", 0)),
                "n_chromosomes": int(row.get("n_chromosomes", 0)),
                "classification": row.get("final_classification", ""),
            })
    return primitives


def compute_recurrence_stats(programs):
    func_seq_counts = Counter(p["function_sequence"] for p in programs)
    recurrences = list(func_seq_counts.values())
    return {
        "unique_sequences": len(func_seq_counts),
        "max_recurrence": max(recurrences) if recurrences else 0,
        "mean_recurrence": float(np.mean(recurrences)) if recurrences else 0,
        "high_recurrence_count": sum(1 for r in recurrences if r >= 5),
        "multi_chrom": 0,
    }


def compute_multi_chrom(programs):
    func_chroms = {}
    for p in programs:
        fs = p["function_sequence"]
        if fs not in func_chroms:
            func_chroms[fs] = set()
        func_chroms[fs].add(p["chromosome"])
    return sum(1 for chroms in func_chroms.values() if len(chroms) >= 3)


def main():
    print("VAL-PRM-001: Primitive Recurrence Permutation Test")
    print("=" * 60)

    t0 = time.time()

    programs = load_programs()
    programs_hash = file_hash(PROGRAMS_PATH)
    programs_rows = file_rows(PROGRAMS_PATH)
    print(f"Loaded {len(programs)} programs (md5={programs_hash}, {programs_rows} rows)")

    primitives = load_primitives()
    primitives_hash = file_hash(PRIMITIVES_PATH)
    primitives_rows = file_rows(PRIMITIVES_PATH)
    print(f"Loaded {len(primitives)} primitives (md5={primitives_hash}, {primitives_rows} rows)")

    chrom_counts = Counter(p["chromosome"] for p in programs)
    print(f"  Chromosomes: {len(chrom_counts)}")
    print(f"  Programs per chromosome: {dict(sorted(chrom_counts.items())[:5])}...")

    observed_stats = compute_recurrence_stats(programs)
    observed_multi_chrom = compute_multi_chrom(programs)
    observed_stats["multi_chrom"] = observed_multi_chrom

    func_seq_recurrence = Counter(p["function_sequence"] for p in programs)
    observed_recurrences = sorted(func_seq_recurrence.values(), reverse=True)

    prim_recurrences = [p["recurrence"] for p in primitives]
    prim_n_chroms = [p["n_chromosomes"] for p in primitives]
    spearman_r, spearman_p = stats.spearmanr(prim_recurrences, prim_n_chroms)

    print(f"\n  Observed unique function sequences: {observed_stats['unique_sequences']}")
    print(f"  Observed max recurrence: {observed_stats['max_recurrence']}")
    print(f"  Observed high-recurrence (≥5): {observed_stats['high_recurrence_count']}")
    print(f"  Observed multi-chromosome (≥3): {observed_multi_chrom}")
    print(f"  Spearman(recurrence, n_chromosomes): r={spearman_r:.4f}, p={spearman_p:.2e}")

    chrom_list = list(chrom_counts.keys())
    chrom_sizes = [chrom_counts[c] for c in chrom_list]

    func_sequences = [p["function_sequence"] for p in programs]
    chromosomes_list = [p["chromosome"] for p in programs]

    rng = random.Random(SEED)
    null_multi_chroms = []
    null_max_per_chrom_recs = []
    null_concentration = []

    def compute_per_chrom_max(progs):
        chrom_seq_counts = {}
        for p in progs:
            c = p["chromosome"]
            fs = p["function_sequence"]
            if c not in chrom_seq_counts:
                chrom_seq_counts[c] = Counter()
            chrom_seq_counts[c][fs] += 1
        max_per_chrom = max(
            (max(counts.values()) for counts in chrom_seq_counts.values() if counts),
            default=0,
        )
        return max_per_chrom

    def compute_concentration(progs):
        func_chrom_dist = {}
        for p in progs:
            fs = p["function_sequence"]
            c = p["chromosome"]
            if fs not in func_chrom_dist:
                func_chrom_dist[fs] = Counter()
            func_chrom_dist[fs][c] += 1
        hhi_sum = 0.0
        n = 0
        for fs, chrom_counts_local in func_chrom_dist.items():
            total = sum(chrom_counts_local.values())
            if total >= 2:
                hhi = sum((c / total) ** 2 for c in chrom_counts_local.values())
                hhi_sum += hhi
                n += 1
        return hhi_sum / n if n > 0 else 0.0

    observed_per_chrom_max = compute_per_chrom_max(programs)
    observed_concentration = compute_concentration(programs)
    print(f"  Observed per-chromosome max recurrence: {observed_per_chrom_max}")
    print(f"  Observed chromosome concentration (mean HHI): {observed_concentration:.4f}")

    print(f"\nRunning {N_PERMUTATIONS} permutations (shuffle function sequences across chromosomes, preserve per-chromosome counts)...")
    for perm in range(N_PERMUTATIONS):
        if perm % 200 == 0 and perm > 0:
            print(f"  Permutation {perm}...")

        shuffled_seqs = list(func_sequences)
        rng.shuffle(shuffled_seqs)

        perm_programs = [{"chromosome": chromosomes_list[i], "function_sequence": shuffled_seqs[i]}
                         for i in range(len(programs))]

        perm_multi = compute_multi_chrom(perm_programs)
        perm_per_chrom_max = compute_per_chrom_max(perm_programs)
        perm_conc = compute_concentration(perm_programs)

        null_multi_chroms.append(perm_multi)
        null_max_per_chrom_recs.append(perm_per_chrom_max)
        null_concentration.append(perm_conc)

    null_multi_chroms = np.array(null_multi_chroms)
    null_max_per_chrom_recs = np.array(null_max_per_chrom_recs)
    null_concentration = np.array(null_concentration)

    p_multi_lower = (np.sum(null_multi_chroms <= observed_multi_chrom) + 1) / (N_PERMUTATIONS + 1)
    p_multi_upper = (np.sum(null_multi_chroms >= observed_multi_chrom) + 1) / (N_PERMUTATIONS + 1)
    p_multi = 2 * min(p_multi_lower, p_multi_upper)

    p_per_chrom = (np.sum(null_max_per_chrom_recs >= observed_per_chrom_max) + 1) / (N_PERMUTATIONS + 1)

    p_conc_upper = (np.sum(null_concentration >= observed_concentration) + 1) / (N_PERMUTATIONS + 1)
    p_conc_lower = (np.sum(null_concentration <= observed_concentration) + 1) / (N_PERMUTATIONS + 1)
    p_conc = 2 * min(p_conc_upper, p_conc_lower)

    z_multi = (observed_multi_chrom - null_multi_chroms.mean()) / (null_multi_chroms.std() + 1e-10)
    z_per_chrom = (observed_per_chrom_max - null_max_per_chrom_recs.mean()) / (null_max_per_chrom_recs.std() + 1e-10)
    z_conc = (observed_concentration - null_concentration.mean()) / (null_concentration.std() + 1e-10)

    classification_counts = Counter(p["classification"] for p in primitives)

    elapsed = time.time() - t0

    sig = p_multi < 0.05 or p_per_chrom < 0.05 or p_conc < 0.05
    conclusion = (
        f"{'SIGNIFICANT' if sig else 'NOT SIGNIFICANT'}: "
        f"Chromosome-shuffle null preserving per-chromosome program counts. "
        f"Multi-chromosome spread (≥3 chroms): {observed_multi_chrom} observed vs "
        f"{null_multi_chroms.mean():.1f} null (z={z_multi:.2f}, p_2sided={p_multi:.4e}). "
        f"Per-chromosome max recurrence: {observed_per_chrom_max} observed vs "
        f"{null_max_per_chrom_recs.mean():.1f} null (z={z_per_chrom:.2f}, p={p_per_chrom:.4e}). "
        f"Chromosome concentration (mean HHI): {observed_concentration:.4f} vs "
        f"{null_concentration.mean():.4f} null (z={z_conc:.2f}, p_2sided={p_conc:.4e}). "
        f"Spearman(recurrence, n_chromosomes): r={spearman_r:.3f} (p={spearman_p:.2e}). "
        f"From {len(programs)} programs and {len(primitives)} annotated primitives."
    )

    results = {
        "test_id": "VAL-PRM-001",
        "test_name": "Primitive Recurrence Permutation Test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "n_programs": len(programs),
            "n_primitives": len(primitives),
            "n_chromosomes": len(chrom_counts),
            "n_permutations": N_PERMUTATIONS,
            "seed": SEED,
            "null_model": "shuffle existing function_sequence assignments across chromosomes preserving per-chromosome program counts, recompute per-chromosome recurrence and multi-chromosome spread",
        },
        "provenance": {
            "programs_file": PROGRAMS_PATH,
            "programs_md5": programs_hash,
            "programs_rows": programs_rows,
            "primitives_file": PRIMITIVES_PATH,
            "primitives_md5": primitives_hash,
            "primitives_rows": primitives_rows,
        },
        "results": {
            "observed": {
                "unique_function_sequences": observed_stats["unique_sequences"],
                "max_recurrence_global": observed_stats["max_recurrence"],
                "mean_recurrence": round(observed_stats["mean_recurrence"], 2),
                "high_recurrence_count_ge5": observed_stats["high_recurrence_count"],
                "multi_chromosome_count_ge3": observed_multi_chrom,
                "per_chromosome_max_recurrence": observed_per_chrom_max,
                "chromosome_concentration_hhi": round(observed_concentration, 4),
                "top_recurrent": [{"sequence": fs, "count": c}
                                  for fs, c in func_seq_recurrence.most_common(10)],
            },
            "null_distribution": {
                "multi_chrom_mean": round(float(null_multi_chroms.mean()), 2),
                "multi_chrom_std": round(float(null_multi_chroms.std()), 2),
                "per_chrom_max_mean": round(float(null_max_per_chrom_recs.mean()), 2),
                "per_chrom_max_std": round(float(null_max_per_chrom_recs.std()), 2),
                "concentration_mean": round(float(null_concentration.mean()), 4),
                "concentration_std": round(float(null_concentration.std()), 4),
            },
            "p_value_multi_chromosome": float(f"{p_multi:.6e}"),
            "p_value_per_chrom_max": float(f"{p_per_chrom:.6e}"),
            "p_value_concentration": float(f"{p_conc:.6e}"),
            "z_score_multi": round(z_multi, 2),
            "z_score_per_chrom": round(z_per_chrom, 2),
            "z_score_concentration": round(z_conc, 2),
            "spearman_recurrence_vs_chromosomes": {
                "r": round(spearman_r, 4),
                "p_value": float(f"{spearman_p:.6e}"),
            },
            "classification_distribution": dict(classification_counts.most_common()),
            "chromosome_program_counts": dict(sorted(chrom_counts.items())),
        },
        "conclusion": conclusion,
        "elapsed_seconds": round(elapsed, 1),
    }

    json_path = os.path.join(OUT_DIR, "VAL-PRM-001_primitive_recurrence.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"VAL-PRM-001: Primitive Recurrence Permutation Test\n"
        f"{len(programs)} programs, {len(primitives)} primitives, "
        f"mean recurrence={observed_stats['mean_recurrence']:.1f}",
        fontsize=14, fontweight="bold"
    )

    ax = axes[0, 0]
    rec_vals = sorted(func_seq_recurrence.values(), reverse=True)
    ax.bar(range(min(50, len(rec_vals))), rec_vals[:50], color="#1565C0", alpha=0.7)
    ax.axhline(y=observed_stats["mean_recurrence"], color="#FF8F00", linestyle="--",
               label=f"Mean recurrence ({observed_stats['mean_recurrence']:.1f})")
    ax.set_xlabel("Function sequence rank")
    ax.set_ylabel("Recurrence count")
    ax.set_title(f"Top 50 Recurrent Sequences (max={observed_stats['max_recurrence']})")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    all_recs = list(func_seq_recurrence.values())
    ax.hist(all_recs, bins=min(50, max(10, max(all_recs))), color="#1565C0", edgecolor="white", alpha=0.7)
    ax.axvline(x=observed_stats["mean_recurrence"], color="#D32F2F", linewidth=2, linestyle="--",
               label=f"Mean ({observed_stats['mean_recurrence']:.1f})")
    ax.set_xlabel("Recurrence count")
    ax.set_ylabel("Number of sequences")
    ax.set_title(f"Recurrence Distribution ({len(primitives)} primitives, {observed_stats['unique_sequences']} unique)")
    ax.legend(fontsize=7)

    ax = axes[0, 2]
    n_bins_pc = min(50, max(10, len(set(null_max_per_chrom_recs))))
    ax.hist(null_max_per_chrom_recs, bins=n_bins_pc, color="#90CAF9", edgecolor="white", alpha=0.8, label="Null distribution")
    ax.axvline(x=observed_per_chrom_max, color="#D32F2F", linewidth=2, linestyle="--",
               label=f"Observed ({observed_per_chrom_max})")
    ax.set_xlabel("Per-chromosome max recurrence")
    ax.set_ylabel("Count")
    ax.set_title(f"Per-Chrom Max vs Null (p={p_per_chrom:.2e}, z={z_per_chrom:.1f})")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    n_bins_mc = min(50, max(10, len(set(null_multi_chroms))))
    ax.hist(null_multi_chroms, bins=n_bins_mc, color="#A5D6A7", edgecolor="white", alpha=0.8, label="Null distribution")
    ax.axvline(x=observed_multi_chrom, color="#D32F2F", linewidth=2, linestyle="--",
               label=f"Observed ({observed_multi_chrom})")
    ax.set_xlabel("Count spanning >=3 chromosomes")
    ax.set_ylabel("Count")
    ax.set_title(f"Multi-Chromosome Spread (p_2sided={p_multi:.2e}, z={z_multi:.1f})")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    n_bins_conc = min(50, max(10, len(set(np.round(null_concentration, 6)))))
    ax.hist(null_concentration, bins=n_bins_conc, color="#FFCC80", edgecolor="white", alpha=0.8, label="Null distribution")
    ax.axvline(x=observed_concentration, color="#D32F2F", linewidth=2, linestyle="--",
               label=f"Observed ({observed_concentration:.4f})")
    ax.set_xlabel("Chromosome concentration (HHI)")
    ax.set_ylabel("Count")
    ax.set_title(f"Concentration vs Null (p_2sided={p_conc:.2e}, z={z_conc:.1f})")
    ax.legend(fontsize=7)

    ax = axes[1, 2]
    ax.scatter(prim_n_chroms, prim_recurrences, s=30, alpha=0.5, c="#1565C0", edgecolors="white", linewidth=0.5)
    ax.set_xlabel("Number of chromosomes")
    ax.set_ylabel("Recurrence count")
    ax.set_title(f"Recurrence vs Chromosome Span (ρ={spearman_r:.3f}, p={spearman_p:.1e})")
    if len(prim_n_chroms) > 1:
        z = np.polyfit(prim_n_chroms, prim_recurrences, 1)
        p_fit = np.poly1d(z)
        x_fit = np.linspace(min(prim_n_chroms), max(prim_n_chroms), 100)
        ax.plot(x_fit, p_fit(x_fit), "r--", alpha=0.5)

    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, "VAL-PRM-001_recurrence_distribution.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Plot saved to {png_path}")

    md = f"""# VAL-PRM-001: Primitive Recurrence Permutation Test

## Purpose
Test whether genome primitive recurrence patterns exceed what is expected when
function sequences are shuffled across chromosomes. This validates that the pipeline
discovers genuinely repeated functional motifs rather than chromosome-biased artifacts.

## Method
1. Load {len(programs)} programs from `programs_annotated.csv`
2. Extract function_sequence per program and its chromosome assignment
3. Null model ({N_PERMUTATIONS} permutations): shuffle existing function_sequence
   assignments across all programs while preserving per-chromosome program counts.
   Sequences stay intact but are randomly reassigned to chromosomes. This tests
   whether the chromosome-specific distribution of primitives is non-random.
4. Compare observed per-chromosome recurrence and multi-chromosome spread to null

## Materials
| Item | Value |
|------|-------|
| Programs | {len(programs)} |
| Programs file MD5 | `{programs_hash}` |
| Programs rows | {programs_rows} |
| Primitives | {len(primitives)} |
| Primitives file MD5 | `{primitives_hash}` |
| Primitives rows | {primitives_rows} |
| Chromosomes | {len(chrom_counts)} |
| Permutations | {N_PERMUTATIONS} |
| Seed | {SEED} |

## Results

### Observed Primitive Recurrence Summary
| Metric | Value |
|--------|-------|
| Annotated primitives | {len(primitives)} |
| Unique function sequences | {observed_stats['unique_sequences']} |
| Global max recurrence | {observed_stats['max_recurrence']} |
| Mean recurrence | {observed_stats['mean_recurrence']:.1f} |
| High recurrence (>=5) | {observed_stats['high_recurrence_count']} |
| Multi-chromosome (>=3) | {observed_multi_chrom} |

### Chromosome-Shuffle Recurrence Statistics
| Metric | Observed | Null Mean +/- SD | z-score | p-value |
|--------|----------|------------------|---------|---------|
| Per-chrom max rec | {observed_per_chrom_max} | {null_max_per_chrom_recs.mean():.1f} +/- {null_max_per_chrom_recs.std():.1f} | {z_per_chrom:.2f} | {p_per_chrom:.2e} |
| Multi-chrom (>=3) | {observed_multi_chrom} | {null_multi_chroms.mean():.1f} +/- {null_multi_chroms.std():.1f} | {z_multi:.2f} | {p_multi:.2e} (2-sided) |
| Concentration (HHI) | {observed_concentration:.4f} | {null_concentration.mean():.4f} +/- {null_concentration.std():.4f} | {z_conc:.2f} | {p_conc:.2e} (2-sided) |
| Global max recurrence | {observed_stats['max_recurrence']} | invariant | — | — |

### Recurrence-Chromosome Correlation (primitives)
| Metric | Value |
|--------|-------|
| Spearman r | {spearman_r:.4f} |
| p-value | {spearman_p:.2e} |

### Classification Distribution
| Classification | Count |
|----------------|-------|
"""
    for cls, cnt in classification_counts.most_common():
        md += f"| {cls} | {cnt} |\n"

    md += f"""
## Interpretation
{conclusion}

## Graph
![Recurrence Distribution](VAL-PRM-001_recurrence_distribution.png)

## Provenance
| Claim | Source |
|-------|--------|
| {len(programs)} programs | `exports/programs_annotated.csv` (MD5: `{programs_hash}`, {programs_rows} rows) |
| {len(primitives)} primitives | `exports/primitive_annotations_complete.csv` (MD5: `{primitives_hash}`, {primitives_rows} rows) |
| max recurrence={observed_stats['max_recurrence']} | Computed from function_sequence counts |

## Runtime
{elapsed:.1f} seconds
"""
    md_path = os.path.join(OUT_DIR, "VAL-PRM-001_summary.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Summary saved to {md_path}")
    print(f"\n{conclusion}")


if __name__ == "__main__":
    main()
