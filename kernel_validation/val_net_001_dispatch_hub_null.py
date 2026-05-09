"""
VAL-NET-001: Dispatch Hub Null Model
=====================================
Tests whether chrM and chr19 are statistically significant hubs in the
cross-chromosome dispatch network using hub dominance metrics (Gini
coefficient, outbound/inbound ratio) and degree-preserving edge-swap
randomization.

Method:
  1. Load cross_chrom_matrix from DB (625 edges, 25 chromosomes)
  2. Compute observed hub metrics: Gini coefficient of total outbound edges,
     outbound/inbound ratio per chromosome
  3. Degree-preserving edge-swap null (1,000 permutations): randomly swap
     pairs of edges while preserving each node's total degree
  4. Compare observed Gini and hub ratios to null distribution

Input: cross_chrom_matrix table (BETA_DATABASE_URL)
Output: validation/VAL-NET-001_dispatch_hub.json
        validation/VAL-NET-001_summary.md
        validation/VAL-NET-001_hub_analysis.png
"""

import os, json, hashlib, time
from collections import defaultdict
from datetime import datetime, timezone

import psycopg2
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DB_URL = os.environ.get("BETA_DATABASE_URL") or os.environ.get("DATABASE_URL")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_PERMUTATIONS = 1000
N_SWAPS_PER_PERM = 5000
SEED = 42


def gini_coefficient(values):
    vals = np.sort(np.array(values, dtype=float))
    n = len(vals)
    if n == 0 or vals.sum() == 0:
        return 0
    index = np.arange(1, n + 1)
    return (2.0 * np.sum(index * vals) - (n + 1) * vals.sum()) / (n * vals.sum())


def load_matrix():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("SELECT source_chrom, target_chrom, ocm_edges, combined_weight FROM cross_chrom_matrix")
    rows = cur.fetchall()
    cur.execute("SELECT COUNT(*) FROM cross_chrom_matrix")
    total_rows = cur.fetchone()[0]
    conn.close()

    edges = []
    for src, tgt, ocm, weight in rows:
        src = src.lower()
        tgt = tgt.lower()
        if src != tgt:
            edges.append({"source": src, "target": tgt, "weight": weight or ocm or 0})
    return edges, total_rows


def compute_hub_metrics(edges, chromosomes):
    outbound = defaultdict(float)
    inbound = defaultdict(float)
    for e in edges:
        outbound[e["source"]] += e["weight"]
        inbound[e["target"]] += e["weight"]

    out_values = [outbound.get(c, 0) for c in chromosomes]
    in_values = [inbound.get(c, 0) for c in chromosomes]

    g = gini_coefficient(out_values)

    ratios = {}
    for c in chromosomes:
        o = outbound.get(c, 0)
        i = inbound.get(c, 0)
        ratios[c] = o / i if i > 0 else float("inf") if o > 0 else 1.0

    max_ratio_chrom = max(ratios, key=ratios.get)
    max_ratio = ratios[max_ratio_chrom]

    return {
        "gini": g,
        "outbound": dict(outbound),
        "inbound": dict(inbound),
        "ratios": ratios,
        "max_ratio_chrom": max_ratio_chrom,
        "max_ratio": max_ratio,
        "out_values": out_values,
        "in_values": in_values,
    }


def edge_swap_null(edges, n_swaps, rng):
    swapped = [dict(e) for e in edges]
    n = len(swapped)
    if n < 2:
        return swapped
    for _ in range(n_swaps):
        i, j = rng.sample(range(n), 2)
        if rng.random() < 0.5:
            swapped[i]["target"], swapped[j]["target"] = swapped[j]["target"], swapped[i]["target"]
        else:
            swapped[i]["source"], swapped[j]["source"] = swapped[j]["source"], swapped[i]["source"]
        if swapped[i]["source"] == swapped[i]["target"] or swapped[j]["source"] == swapped[j]["target"]:
            if rng.random() < 0.5:
                swapped[i]["target"], swapped[j]["target"] = swapped[j]["target"], swapped[i]["target"]
            else:
                swapped[i]["source"], swapped[j]["source"] = swapped[j]["source"], swapped[i]["source"]
    return swapped


def main():
    print("VAL-NET-001: Dispatch Hub Null Model")
    print("=" * 60)

    t0 = time.time()

    edges, total_rows = load_matrix()
    print(f"Loaded {len(edges)} inter-chromosome edges ({total_rows} total DB rows)")

    chromosomes = sorted(set(e["source"] for e in edges) | set(e["target"] for e in edges))
    print(f"Chromosomes: {len(chromosomes)}")

    observed = compute_hub_metrics(edges, chromosomes)
    print(f"\n  Observed Gini (outbound): {observed['gini']:.6f}")
    print(f"  Top outbound: {sorted(observed['outbound'].items(), key=lambda x: -x[1])[:5]}")
    print(f"  chrm outbound: {observed['outbound'].get('chrm', 0):.0f}")
    print(f"  chr19 outbound: {observed['outbound'].get('chr19', 0):.0f}")
    print(f"  chrm ratio (out/in): {observed['ratios'].get('chrm', 0):.4f}")
    print(f"  chr19 ratio (out/in): {observed['ratios'].get('chr19', 0):.4f}")

    rng = np.random.RandomState(SEED)

    class RngAdapter:
        def __init__(self, rs):
            self.rs = rs
        def sample(self, pop, k):
            return list(self.rs.choice(len(pop), size=k, replace=False))
        def random(self):
            return self.rs.random()

    rng_adapter = RngAdapter(rng)

    null_ginis = []
    null_chrm_ratios = []
    null_chr19_ratios = []
    null_max_ratios = []

    print(f"\nRunning {N_PERMUTATIONS} edge-swap permutations ({N_SWAPS_PER_PERM} swaps each)...")
    for perm in range(N_PERMUTATIONS):
        if perm % 200 == 0 and perm > 0:
            print(f"  Permutation {perm}...")

        swapped = edge_swap_null(edges, N_SWAPS_PER_PERM, rng_adapter)
        metrics = compute_hub_metrics(swapped, chromosomes)
        null_ginis.append(metrics["gini"])
        null_chrm_ratios.append(metrics["ratios"].get("chrm", 1.0))
        null_chr19_ratios.append(metrics["ratios"].get("chr19", 1.0))
        null_max_ratios.append(metrics["max_ratio"])

    null_ginis = np.array(null_ginis)
    null_chrm_ratios = np.array(null_chrm_ratios)
    null_chr19_ratios = np.array(null_chr19_ratios)
    null_max_ratios = np.array(null_max_ratios)

    obs_chrm_ratio = observed["ratios"].get("chrm", 1.0)
    obs_chr19_ratio = observed["ratios"].get("chr19", 1.0)

    p_gini = (np.sum(null_ginis >= observed["gini"]) + 1) / (N_PERMUTATIONS + 1)
    p_chrm = (np.sum(null_chrm_ratios >= obs_chrm_ratio) + 1) / (N_PERMUTATIONS + 1)
    p_chr19 = (np.sum(null_chr19_ratios >= obs_chr19_ratio) + 1) / (N_PERMUTATIONS + 1)

    z_gini = (observed["gini"] - null_ginis.mean()) / (null_ginis.std() + 1e-10)
    z_chrm = (obs_chrm_ratio - null_chrm_ratios.mean()) / (null_chrm_ratios.std() + 1e-10)
    z_chr19 = (obs_chr19_ratio - null_chr19_ratios.mean()) / (null_chr19_ratios.std() + 1e-10)

    elapsed = time.time() - t0

    sig = p_gini < 0.05 or p_chrm < 0.05 or p_chr19 < 0.05
    conclusion = (
        f"{'SIGNIFICANT' if sig else 'NOT SIGNIFICANT'}: "
        f"Gini={observed['gini']:.4f} (null mean={null_ginis.mean():.4f}, z={z_gini:.2f}, p={p_gini:.4e}). "
        f"chrM out/in ratio={obs_chrm_ratio:.4f} (null mean={null_chrm_ratios.mean():.4f}, z={z_chrm:.2f}, p={p_chrm:.4e}). "
        f"chr19 out/in ratio={obs_chr19_ratio:.4f} (null mean={null_chr19_ratios.mean():.4f}, z={z_chr19:.2f}, p={p_chr19:.4e}). "
        f"Network: {len(edges)} edges, {len(chromosomes)} chromosomes, {N_SWAPS_PER_PERM} edge swaps per permutation."
    )

    results = {
        "test_id": "VAL-NET-001",
        "test_name": "Dispatch Hub Null Model",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "n_edges": len(edges),
            "n_chromosomes": len(chromosomes),
            "n_permutations": N_PERMUTATIONS,
            "n_swaps_per_permutation": N_SWAPS_PER_PERM,
            "seed": SEED,
            "null_model": "edge-swap degree-preserving randomization",
        },
        "provenance": {
            "database": "BETA_DATABASE_URL",
            "table": "cross_chrom_matrix",
            "total_db_rows": total_rows,
            "inter_chromosome_edges": len(edges),
        },
        "results": {
            "observed": {
                "gini_outbound": round(observed["gini"], 6),
                "chrm_outbound": round(observed["outbound"].get("chrm", 0), 0),
                "chrm_inbound": round(observed["inbound"].get("chrm", 0), 0),
                "chrm_ratio": round(obs_chrm_ratio, 4),
                "chr19_outbound": round(observed["outbound"].get("chr19", 0), 0),
                "chr19_inbound": round(observed["inbound"].get("chr19", 0), 0),
                "chr19_ratio": round(obs_chr19_ratio, 4),
                "max_ratio_chromosome": observed["max_ratio_chrom"],
                "max_ratio": round(observed["max_ratio"], 4),
                "top5_outbound": {c: round(v, 0) for c, v in sorted(observed["outbound"].items(), key=lambda x: -x[1])[:5]},
                "top5_inbound": {c: round(v, 0) for c, v in sorted(observed["inbound"].items(), key=lambda x: -x[1])[:5]},
            },
            "null_distribution": {
                "gini_mean": round(float(null_ginis.mean()), 6),
                "gini_std": round(float(null_ginis.std()), 6),
                "chrm_ratio_mean": round(float(null_chrm_ratios.mean()), 4),
                "chrm_ratio_std": round(float(null_chrm_ratios.std()), 4),
                "chr19_ratio_mean": round(float(null_chr19_ratios.mean()), 4),
                "chr19_ratio_std": round(float(null_chr19_ratios.std()), 4),
            },
            "p_value_gini": float(f"{p_gini:.6e}"),
            "p_value_chrm_ratio": float(f"{p_chrm:.6e}"),
            "p_value_chr19_ratio": float(f"{p_chr19:.6e}"),
            "z_score_gini": round(z_gini, 2),
            "z_score_chrm": round(z_chrm, 2),
            "z_score_chr19": round(z_chr19, 2),
        },
        "conclusion": conclusion,
        "elapsed_seconds": round(elapsed, 1),
    }

    json_path = os.path.join(OUT_DIR, "VAL-NET-001_dispatch_hub.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("VAL-NET-001: Dispatch Hub Null Model\nEdge-swap degree-preserving randomization", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    out_sorted = sorted(observed["outbound"].items(), key=lambda x: -x[1])
    chroms_sorted = [c[0] for c in out_sorted]
    out_vals = [c[1] for c in out_sorted]
    in_vals = [observed["inbound"].get(c, 0) for c in chroms_sorted]
    x = np.arange(len(chroms_sorted))
    ax.bar(x - 0.2, out_vals, 0.4, label="Outbound", color="#1565C0", alpha=0.7)
    ax.bar(x + 0.2, in_vals, 0.4, label="Inbound", color="#FF8F00", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(chroms_sorted, rotation=45, fontsize=7)
    ax.set_ylabel("Edge weight")
    ax.set_title("Outbound vs Inbound per Chromosome")
    ax.legend()

    ax = axes[0, 1]
    ax.hist(null_ginis, bins=50, color="#90CAF9", edgecolor="white", alpha=0.8, label="Null Gini")
    ax.axvline(x=observed["gini"], color="#D32F2F", linewidth=2, linestyle="--",
               label=f"Observed ({observed['gini']:.4f})")
    ax.set_xlabel("Gini coefficient (outbound)")
    ax.set_ylabel("Count")
    ax.set_title(f"Gini: Observed vs Null (p={p_gini:.2e}, z={z_gini:.1f})")
    ax.legend()

    ax = axes[1, 0]
    ax.hist(null_chrm_ratios, bins=50, color="#A5D6A7", edgecolor="white", alpha=0.8, label="Null chrM ratio")
    ax.axvline(x=obs_chrm_ratio, color="#D32F2F", linewidth=2, linestyle="--",
               label=f"Observed ({obs_chrm_ratio:.4f})")
    ax.set_xlabel("chrM outbound/inbound ratio")
    ax.set_ylabel("Count")
    ax.set_title(f"chrM Hub Ratio (p={p_chrm:.2e}, z={z_chrm:.1f})")
    ax.legend()

    ax = axes[1, 1]
    ax.hist(null_chr19_ratios, bins=50, color="#CE93D8", edgecolor="white", alpha=0.8, label="Null chr19 ratio")
    ax.axvline(x=obs_chr19_ratio, color="#D32F2F", linewidth=2, linestyle="--",
               label=f"Observed ({obs_chr19_ratio:.4f})")
    ax.set_xlabel("chr19 outbound/inbound ratio")
    ax.set_ylabel("Count")
    ax.set_title(f"chr19 Hub Ratio (p={p_chr19:.2e}, z={z_chr19:.1f})")
    ax.legend()

    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, "VAL-NET-001_hub_analysis.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Plot saved to {png_path}")

    r = results["results"]
    md = f"""# VAL-NET-001: Dispatch Hub Null Model

## Purpose
Test whether chrM (KERNEL) and chr19 (RELAY hub) are statistically significant
hub chromosomes in the OBS cross-chromosome dispatch network using the Gini
coefficient of outbound edge weights and outbound/inbound ratios, validated
against degree-preserving edge-swap randomization.

## Method
1. Load {len(edges)} inter-chromosome edges from `cross_chrom_matrix` table
2. Compute observed metrics:
   - Gini coefficient of per-chromosome outbound edge weights
   - Outbound/inbound ratio per chromosome (hub dominance indicator)
3. Null model ({N_PERMUTATIONS} rounds, {N_SWAPS_PER_PERM} edge swaps each):
   degree-preserving edge-swap randomization — randomly select two edges and
   swap their targets (or sources), rejecting self-loops. This preserves
   each node's approximate degree while randomizing topology.
4. Compare observed Gini and chrM/chr19 ratios to null distributions

## Materials
| Item | Value |
|------|-------|
| Database table | cross_chrom_matrix |
| Total DB rows | {total_rows} |
| Inter-chromosome edges | {len(edges)} |
| Chromosomes | {len(chromosomes)} |
| Permutations | {N_PERMUTATIONS} |
| Edge swaps per perm | {N_SWAPS_PER_PERM} |
| Seed | {SEED} |

## Results

### Hub Dominance
| Metric | Observed | Null Mean +/- SD | z-score | p-value |
|--------|----------|----------------|---------|---------|
| Gini (outbound) | {observed['gini']:.4f} | {null_ginis.mean():.4f} +/- {null_ginis.std():.4f} | {z_gini:.2f} | {p_gini:.2e} |
| chrM ratio | {obs_chrm_ratio:.4f} | {null_chrm_ratios.mean():.4f} +/- {null_chrm_ratios.std():.4f} | {z_chrm:.2f} | {p_chrm:.2e} |
| chr19 ratio | {obs_chr19_ratio:.4f} | {null_chr19_ratios.mean():.4f} +/- {null_chr19_ratios.std():.4f} | {z_chr19:.2f} | {p_chr19:.2e} |

### Key Chromosomes
| Chromosome | Outbound | Inbound | Out/In Ratio |
|------------|----------|---------|--------------|
| chrm | {observed['outbound'].get('chrm', 0):.0f} | {observed['inbound'].get('chrm', 0):.0f} | {obs_chrm_ratio:.4f} |
| chr19 | {observed['outbound'].get('chr19', 0):.0f} | {observed['inbound'].get('chr19', 0):.0f} | {obs_chr19_ratio:.4f} |

## Interpretation
{conclusion}

## Graph
![Hub Analysis](VAL-NET-001_hub_analysis.png)

## Provenance
| Claim | Source |
|-------|--------|
| {len(edges)} edges | cross_chrom_matrix, BETA_DATABASE_URL ({total_rows} rows) |
| Gini={observed['gini']:.4f} | Computed from {len(chromosomes)} chromosome outbound sums |
| chrM ratio={obs_chrm_ratio:.4f} | chrM outbound / chrM inbound |

## Runtime
{elapsed:.1f} seconds
"""
    md_path = os.path.join(OUT_DIR, "VAL-NET-001_summary.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Summary saved to {md_path}")
    print(f"\n{conclusion}")


if __name__ == "__main__":
    main()
