#!/usr/bin/env python3
"""
Generate Supplementary Figures SF2, SF3, SF4 for Paper 2.

SF2 — Validation performance across all sets and families
SF3 — Statistical controls (ablation, single-component, propensity null)
SF4 — Sensitivity and holdout analyses

Output: paper2/v2_submission/figures/SF2_validation_performance.{png,pdf}
        paper2/v2_submission/figures/SF3_statistical_controls.{png,pdf}
        paper2/v2_submission/figures/SF4_sensitivity_holdout.{png,pdf}

Run from workspace root:
    python3 paper2/generate_supp_figures.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from collections import defaultdict

# ── Style matching Nature guidelines + paper2/generate_figures.py ──────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PALETTE = {
    "Set 1": "#1f77b4",
    "Set 2": "#ff7f0e",
    "Set 3": "#2ca02c",
    "Combined": "#9467bd",
    "null": "#aaaaaa",
    "full_model": "#d62728",
}
OUT_DIR = "paper2/v2_submission/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ───────────────────────────────────────────────────────────────
BASE = "paper2/v2_submission/raw_results"


def load(name):
    with open(os.path.join(BASE, name)) as f:
        return json.load(f)


s1_data = load("validation_77_results.json")
s2_data = load("validation_extended_results.json")
s3_data = load("validation_set3_results.json")
combo   = load("combined_all_sets_statistics.json")
null    = load("null_test_results.json")
prop    = load("propensity_null_tier2plus.json")
rev     = load("reviewer_response_tests.json")
holdout = load("strict_holdout_3166.json")
synth   = load("synthetic_rescue_experiment.json")
false_p = load("false_positive_results.json")


def extract_percentiles(dataset):
    """Return list of all factor percentile scores in a validation JSON."""
    pcts = []
    for ck in dataset.get("per_cocktail", []):
        for gene, info in ck.get("factor_ranks", {}).items():
            if isinstance(info, dict) and info.get("found"):
                pcts.append(info["percentile"])
    return pcts


s1_pcts = extract_percentiles(s1_data)
s2_pcts = extract_percentiles(s2_data)
s3_pcts = extract_percentiles(s3_data)

# ── Ordered components ───────────────────────────────────────────────────────
COMPONENT_KEYS = ["w_gtex", "w_pheno", "w_tau", "w_frac", "w_kern",
                  "w_temp", "w_act", "w_dir", "w_enr"]
COMPONENT_NAMES = {
    "w_gtex": "GTEx expression",
    "w_pheno": "Phenotype/disease",
    "w_tau": "Tissue specificity (τ)",
    "w_frac": "Enrichment fraction",
    "w_kern": "Kernel signature",
    "w_temp": "Temporal expression",
    "w_act": "Activation precision",
    "w_dir": "Directional connectivity",
    "w_enr": "Enrichment magnitude",
}


# ════════════════════════════════════════════════════════════════════════════
# SF2 — Validation Performance
# ════════════════════════════════════════════════════════════════════════════

def make_sf2():
    fig = plt.figure(figsize=(7.09, 10.0))  # 180mm wide; extra height for 17-family heatmap
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.50, wspace=0.38,
                           left=0.12, right=0.97, top=0.95, bottom=0.06)

    # ── a: Recall vs rank threshold ─────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, :])
    thresholds = [0.5, 1, 2, 3, 5, 7, 10, 15, 20]

    set_pcts = {
        "Set 1": s1_pcts,
        "Set 2": s2_pcts,
        "Set 3": s3_pcts,
        "Combined": s1_pcts + s2_pcts + s3_pcts,
    }

    for label, pcts in set_pcts.items():
        recalls = [sum(1 for p in pcts if p <= t) / len(pcts) * 100 for t in thresholds]
        ls = "--" if label == "Combined" else "-"
        lw = 2.5 if label == "Combined" else 1.8
        ax_a.plot(thresholds, recalls, ls=ls, lw=lw,
                  color=PALETTE[label], label=label, marker="o", ms=4)

    # Random expectation
    ax_a.plot(thresholds, thresholds, ":", color="#888", lw=1.3, label="Random (chance)")
    ax_a.axvline(5, color="#ccc", lw=1, ls="--", zorder=0)

    # Fold enrichment at 5%
    comb_at5 = sum(1 for p in s1_pcts + s2_pcts + s3_pcts if p <= 5) / len(s1_pcts + s2_pcts + s3_pcts) * 100
    ax_a.annotate(f"{comb_at5:.0f}% at top-5%\n({comb_at5/5:.1f}× enrichment)",
                  xy=(5, comb_at5), xytext=(7.5, comb_at5 - 12),
                  fontsize=8, color=PALETTE["Combined"],
                  arrowprops=dict(arrowstyle="-|>", color=PALETTE["Combined"],
                                  lw=1.2, mutation_scale=10))
    ax_a.set_xlabel("Rank threshold (percentile)")
    ax_a.set_ylabel("Recall of published factors (%)")
    ax_a.set_title("Recall vs. rank threshold across validation sets")
    ax_a.set_xlim(0, 21)
    ax_a.set_ylim(0, 100)
    ax_a.legend(loc="upper left", frameon=False, ncol=2)
    ax_a.text(-0.06, 1.04, "a", transform=ax_a.transAxes,
              fontsize=10, fontweight="bold")

    # ── b: Cumulative factor rank distributions ──────────────────────────────
    ax_b = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0, 100, 51)
    for label, pcts in [("Set 1", s1_pcts), ("Set 2", s2_pcts), ("Set 3", s3_pcts)]:
        counts, edges = np.histogram(pcts, bins=bins)
        cumulative = np.cumsum(counts) / len(pcts) * 100
        ax_b.plot(edges[1:], cumulative, color=PALETTE[label], lw=1.8, label=label)

    ax_b.plot([0, 100], [0, 100], ":", color="#888", lw=1.2, label="Random")
    ax_b.axvline(5, color="#ccc", lw=1, ls="--", zorder=0)
    ax_b.set_xlabel("Percentile rank threshold")
    ax_b.set_ylabel("Cumulative recall (%)")
    ax_b.set_title("Cumulative factor rank distribution")
    ax_b.legend(frameon=False, fontsize=8)
    ax_b.set_xlim(0, 50)
    ax_b.set_ylim(0, 100)
    ax_b.text(-0.14, 1.05, "b", transform=ax_b.transAxes,
              fontsize=10, fontweight="bold")

    # ── c: Per-family heatmap ────────────────────────────────────────────────
    # Read directly from the verified CSV — the JSON per_family_breakdown contains
    # placeholder None values and cannot be used here.
    ax_c = fig.add_subplot(gs[1, 1])
    import csv as _csv
    _s6_rows = list(_csv.DictReader(open(
        os.path.join(os.path.dirname(__file__),
                     "v2_submission/table_s6_family_breakdown.csv"))))
    sets_order = ["Set 1", "Set 2", "Set 3"]
    families = sorted(set(r["family"] for r in _s6_rows))
    _s6_lookup = {(r["family"], r["set"]): float(r["top5_pct"]) for r in _s6_rows}

    matrix = []
    for fam in families:
        row = []
        for s in sets_order:
            val = _s6_lookup.get((fam, s))
            row.append(val if val is not None else float("nan"))
        matrix.append(row)

    matrix = np.array(matrix, dtype=float)
    im = ax_c.imshow(matrix, aspect="auto", cmap="RdYlGn",
                     vmin=0, vmax=100, interpolation="nearest")
    ax_c.set_xticks(range(3))
    ax_c.set_xticklabels(["Set 1", "Set 2", "Set 3"], fontsize=8)
    ax_c.set_yticks(range(len(families)))
    ax_c.set_yticklabels(families, fontsize=7.5)
    ax_c.set_title("Top-5% recall by family (%)", fontsize=9)

    for i, fam in enumerate(families):
        for j, s in enumerate(sets_order):
            val = matrix[i, j]
            if not np.isnan(val):
                ax_c.text(j, i, f"{val:.0f}", ha="center", va="center",
                          fontsize=7, color="black" if 30 < val < 70 else "white")
            else:
                ax_c.text(j, i, "—", ha="center", va="center", fontsize=7, color="#ccc")

    plt.colorbar(im, ax=ax_c, fraction=0.04, pad=0.02, label="%")
    ax_c.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax_c.text(-0.18, 1.05, "c", transform=ax_c.transAxes,
              fontsize=10, fontweight="bold")

    # ── d: Factor rank scatter by set ────────────────────────────────────────
    ax_d = fig.add_subplot(gs[2, :])
    set_info = [("Set 1", s1_data, PALETTE["Set 1"]),
                ("Set 2", s2_data, PALETTE["Set 2"]),
                ("Set 3", s3_data, PALETTE["Set 3"])]
    families_seen = []
    family_color_map = {}
    fam_colors = plt.cm.tab20.colors
    x_offset = 0

    xtick_pos, xtick_labels = [], []
    for set_label, sdata, set_color in set_info:
        fam_groups = defaultdict(list)
        for ck in sdata.get("per_cocktail", []):
            fam = ck.get("family", "Other")
            for gene, info in ck.get("factor_ranks", {}).items():
                if isinstance(info, dict) and info.get("found"):
                    fam_groups[fam].append(info["percentile"])

        set_start = x_offset
        for fam in sorted(fam_groups):
            if fam not in family_color_map:
                family_color_map[fam] = fam_colors[len(family_color_map) % 20]
            pcts_fam = fam_groups[fam]
            jitter = np.random.default_rng(42).uniform(-0.3, 0.3, len(pcts_fam))
            ax_d.scatter([x_offset + jitter[i] for i in range(len(pcts_fam))],
                         pcts_fam, s=6, alpha=0.5,
                         color=family_color_map[fam], zorder=3)
            xtick_pos.append(x_offset)
            xtick_labels.append(fam[:7])
            x_offset += 1

        mid = (set_start + x_offset - 1) / 2
        ax_d.text(mid, -12, set_label, ha="center", fontsize=8.5, color=set_color,
                  fontweight="bold")
        if x_offset < 3 * len(set_info):
            ax_d.axvline(x_offset - 0.5, color="#ddd", lw=1, zorder=0)

    ax_d.axhline(5, color="#aaa", ls="--", lw=1.2, label="5% threshold")
    ax_d.axhline(10, color="#ccc", ls=":", lw=1, label="10% threshold")
    ax_d.set_xticks(xtick_pos)
    ax_d.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=7.5)
    ax_d.set_ylabel("Factor percentile rank")
    ax_d.set_title("Per-family factor percentile ranks across all three validation sets")
    ax_d.set_ylim(-5, 105)
    ax_d.legend(frameon=False, loc="upper right", fontsize=8)
    ax_d.text(-0.04, 1.04, "d", transform=ax_d.transAxes,
              fontsize=10, fontweight="bold")

    for path_suffix in [".png", ".pdf"]:
        out = os.path.join(OUT_DIR, f"SF2_validation_performance{path_suffix}")
        fig.savefig(out)
        print(f"  ✓ {out}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# SF3 — Statistical Controls
# ════════════════════════════════════════════════════════════════════════════

def make_sf3():
    fig = plt.figure(figsize=(7.09, 8.5))
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.52, wspace=0.42,
                           left=0.10, right=0.97, top=0.95, bottom=0.07)

    # ── a: Component ablation — mean delta across sets ───────────────────────
    ax_a = fig.add_subplot(gs[0, :])
    ablation = null["ablation"]

    items = []
    for key in COMPONENT_KEYS:
        comp = ablation.get(key, {})
        sets_d = comp.get("sets", {})
        deltas = [sets_d[s]["delta"] for s in sets_d if "delta" in sets_d[s]]
        mean_delta = np.mean(deltas) if deltas else 0
        items.append((COMPONENT_NAMES.get(key, key), mean_delta))

    items.sort(key=lambda x: x[1])
    names, deltas = zip(*items)

    colors = ["#d73027" if d < -10 else "#fc8d59" if d < -5 else "#fee08b" if d < -2 else "#91cf60"
              for d in deltas]
    bars = ax_a.barh(names, deltas, color=colors, edgecolor="white", height=0.6)
    ax_a.axvline(0, color="black", lw=0.8)

    for bar, delta in zip(bars, deltas):
        ax_a.text(delta - 0.2, bar.get_y() + bar.get_height() / 2,
                  f"{delta:.1f}pp", va="center", ha="right", fontsize=8)

    ax_a.set_xlabel("Mean top-5% recall drop (pp)")
    ax_a.set_title("Component ablation — impact when each component removed")
    ax_a.set_xlim(min(deltas) - 3, 2)
    ax_a.spines["left"].set_visible(False)
    ax_a.text(-0.06, 1.04, "a", transform=ax_a.transAxes,
              fontsize=10, fontweight="bold")

    # ── b: Single-component baselines ───────────────────────────────────────
    ax_b = fig.add_subplot(gs[1, 0])
    single = null["single_component"]
    full_mean = 69.3

    sc_items = []
    for key in COMPONENT_KEYS:
        comp = single.get(key, {})
        sets_d = comp.get("sets", {})
        means = [sets_d[s]["top5"] for s in sets_d if "top5" in sets_d[s]]
        sc_items.append((COMPONENT_NAMES.get(key, key), np.mean(means) if means else 0))

    sc_items.sort(key=lambda x: -x[1])
    sc_names, sc_vals = zip(*sc_items)

    bar_colors = ["#1f77b4" if v > 40 else "#aec7e8" if v > 20 else "#d9d9d9"
                  for v in sc_vals]
    ax_b.barh(sc_names, sc_vals, color=bar_colors, edgecolor="white", height=0.6)
    ax_b.axvline(full_mean, color=PALETTE["full_model"], lw=2, ls="--", label=f"Full model ({full_mean}%)")
    ax_b.axvline(5, color="#888", lw=1, ls=":", label="Random (5%)")
    ax_b.set_xlabel("Mean top-5% recall (%)")
    ax_b.set_title("Single-component baselines")
    ax_b.legend(frameon=False, fontsize=8, loc="lower right")
    ax_b.spines["left"].set_visible(False)
    ax_b.text(-0.14, 1.05, "b", transform=ax_b.transAxes,
              fontsize=10, fontweight="bold")

    # ── c: Permutation null tests ────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 1])
    perm = null["permutation"]
    sets_perm = ["Set 1", "Set 2", "Set 3"]
    real_vals  = [77.8, 64.7, 65.3]
    null_means = [5.0, 5.0, 5.0]
    null_stds  = [1.69, 1.90, 3.08]

    x = np.arange(len(sets_perm))
    w = 0.35
    bars_real = ax_c.bar(x + w/2, real_vals, w, color=[PALETTE[s] for s in sets_perm],
                         label="Observed", edgecolor="white")
    bars_null = ax_c.bar(x - w/2, null_means, w, color="#cccccc",
                         yerr=null_stds, capsize=4, label="Null (permuted)", edgecolor="white")

    for i, (rv, nm, ns) in enumerate(zip(real_vals, null_means, null_stds)):
        z = (rv - nm) / ns
        ax_c.text(x[i] + w/2, rv + 1.5, f"z={z:.0f}", ha="center", fontsize=8)

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(sets_perm)
    ax_c.set_ylabel("Top-5% recall (%)")
    ax_c.set_title("Gene-program permutation null")
    ax_c.legend(frameon=False, fontsize=8)
    ax_c.text(-0.18, 1.05, "c", transform=ax_c.transAxes,
              fontsize=10, fontweight="bold")

    # ── d: Propensity-matched null histogram ─────────────────────────────────
    ax_d = fig.add_subplot(gs[2, :])
    null_dist = prop.get("null_distribution", [])
    null_mean = prop["null_mean_pct"]
    null_std  = prop["null_std_pct"]
    real_top5 = prop["real_top5_pct"]

    ax_d.hist(null_dist, bins=25, color="#aaaaaa", edgecolor="white",
              alpha=0.85, label=f"Propensity-matched null\n(n={prop['n_permutations']} permutations)")
    ax_d.axvline(null_mean, color="#555", lw=1.5, ls="--",
                 label=f"Null mean = {null_mean:.1f}% ± {null_std:.1f}%")
    ax_d.axvline(real_top5, color=PALETTE["full_model"], lw=2.5,
                 label=f"Observed = {real_top5:.1f}%  (z = {prop['z_score']:.0f})")

    ax_d.fill_betweenx([0, ax_d.get_ylim()[1] if ax_d.get_ylim()[1] > 0 else 10],
                       null_mean - null_std, null_mean + null_std,
                       alpha=0.2, color="#555", zorder=0)

    z = prop["z_score"]
    ax_d.annotate(
        f"+{prop['effect_pp']:.0f}pp above null\nz = {z:.0f}, p < 0.005",
        xy=(real_top5, 3), xytext=(real_top5 - 35, 5),
        fontsize=8.5, color=PALETTE["full_model"],
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["full_model"],
                        lw=1.2, mutation_scale=10))

    ax_d.set_xlabel("Top-5% recall (%)")
    ax_d.set_ylabel("Count")
    ax_d.set_title(
        "Propensity-matched null: each real factor paired with a TF\n"
        "matched on GTEx expression level and phenotype annotation density"
    )
    ax_d.legend(frameon=False, fontsize=8)
    ax_d.text(-0.04, 1.04, "d", transform=ax_d.transAxes,
              fontsize=10, fontweight="bold")

    for path_suffix in [".png", ".pdf"]:
        out = os.path.join(OUT_DIR, f"SF3_statistical_controls{path_suffix}")
        fig.savefig(out)
        print(f"  ✓ {out}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# SF4 — Sensitivity and Holdout Analyses
# ════════════════════════════════════════════════════════════════════════════

def make_sf4():
    fig = plt.figure(figsize=(7.09, 9.0))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.52, wspace=0.42,
                           left=0.10, right=0.97, top=0.95, bottom=0.07)

    sets_order = ["Set 1", "Set 2", "Set 3"]

    # ── a: Two-tier vs single-tier ────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    two_tier  = [rev["baseline"][s]["top5"] for s in sets_order]
    one_tier  = [rev["test2_single_tier"][s]["top5"] for s in sets_order]

    x = np.arange(len(sets_order))
    w = 0.35
    ax_a.bar(x - w/2, two_tier, w, color=[PALETTE[s] for s in sets_order],
             label="Two-tier (adaptive)", edgecolor="white")
    ax_a.bar(x + w/2, one_tier, w, color=["#aec7e8", "#ffbb78", "#98df8a"],
             label="Single-tier (flat weights)", edgecolor="white")
    for i, (t, o) in enumerate(zip(two_tier, one_tier)):
        delta = t - o
        ax_a.text(x[i], max(t, o) + 1.5,
                  f"{'+'if delta>0 else ''}{delta:.1f}pp",
                  ha="center", fontsize=7.5,
                  color="#2ca02c" if delta > 0 else "#d62728")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(sets_order)
    ax_a.set_ylabel("Top-5% recall (%)")
    ax_a.set_title("Two-tier vs single-tier weights")
    ax_a.legend(frameon=False, fontsize=7.5)
    ax_a.set_ylim(0, 90)
    ax_a.text(-0.14, 1.05, "a", transform=ax_a.transAxes,
              fontsize=10, fontweight="bold")

    # ── b: Kernel-only ablation stack ────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    configs = [
        ("Full model\n(9 components)",  [rev["baseline"][s]["top5"] for s in sets_order], "#1f77b4"),
        ("Kernel stack\n(5 components)", [rev["test1_kernel_stack"][s]["top5"] for s in sets_order], "#ff7f0e"),
        ("Native only\n(4 components)", [rev["test1_native_only"][s]["top5"] for s in sets_order], "#2ca02c"),
        ("Kernel sig.\nalone",          [rev["test1_kernel_alone"][s]["top5"] for s in sets_order], "#9467bd"),
    ]

    x = np.arange(len(sets_order))
    bar_w = 0.2
    for i, (label, vals, color) in enumerate(configs):
        offset = (i - 1.5) * bar_w
        ax_b.bar(x + offset, vals, bar_w, color=color, label=label, edgecolor="white", alpha=0.9)

    ax_b.axhline(5, color="#888", lw=1, ls=":", label="Random (5%)")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(sets_order)
    ax_b.set_ylabel("Top-5% recall (%)")
    ax_b.set_title("Kernel-only architecture ablation")
    ax_b.legend(frameon=False, fontsize=7, ncol=2)
    ax_b.set_ylim(0, 90)
    ax_b.text(-0.18, 1.05, "b", transform=ax_b.transAxes,
              fontsize=10, fontweight="bold")

    # ── c: Strict holdout analysis ────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    # Set 2 full vs strict
    s2_full      = holdout["full_set2"]["top5_pct"]
    s2_strict    = holdout["strict_holdout_set2"]["top5_pct"]
    s2_n_full    = holdout["full_set2"]["total_instances"]
    s2_n_strict  = holdout["strict_holdout_set2"]["total_instances"]

    # Set 3 values from ALL_RESULTS
    s3_full_pct    = 56.0
    s3_strict_pct  = 30.8
    s3_n_full      = 50
    s3_n_strict    = 13

    cats   = ["Set 2\nfull", "Set 2\nstrict holdout", "Set 3\nfull", "Set 3\nstrict holdout"]
    vals   = [s2_full, s2_strict, s3_full_pct, s3_strict_pct]
    ns     = [s2_n_full, s2_n_strict, s3_n_full, s3_n_strict]
    colors = ["#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a"]

    bars = ax_c.bar(cats, vals, color=colors, edgecolor="white", width=0.55)
    # 95% CI (Wilson interval approximation)
    for bar, v, n in zip(bars, vals, ns):
        p = v / 100
        margin = 1.96 * np.sqrt(p * (1 - p) / n) * 100
        ax_c.errorbar(bar.get_x() + bar.get_width() / 2,
                      v, yerr=margin, fmt="none", color="black", capsize=4, lw=1.5)

    ax_c.axhline(5, color="#888", lw=1, ls=":", label="Random (5%)")
    ax_c.set_ylabel("Top-5% recall (%)")
    ax_c.set_title("Strict holdout analysis\n(factors exclusive to each set)")
    ax_c.set_ylim(0, 90)
    ax_c.legend(frameon=False, fontsize=8)
    ax_c.text(-0.14, 1.05, "c", transform=ax_c.transAxes,
              fontsize=10, fontweight="bold")

    # ── d: Vascular synthetic rescue ─────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    per_factor_list = synth.get("per_factor_results", [])
    vascular_summary = synth.get("vascular_family_summary", {})

    # Deduplicate: take first occurrence of each factor (cocktail ENDO-01)
    seen = {}
    for item in per_factor_list:
        g = item["factor"]
        if g not in seen:
            seen[g] = item

    factor_labels = list(seen.keys())
    before_ranks = [seen[g]["original_pctl"] for g in factor_labels]
    after_ranks  = [seen[g]["rescued_pctl"]  for g in factor_labels]

    x = np.arange(len(factor_labels))
    w = 0.35
    ax_d.bar(x - w/2, before_ranks, w, color="#fc8d59", label="Original rank", edgecolor="white")
    ax_d.bar(x + w/2, after_ranks,  w, color="#91bfdb", label="Synthetic rescue", edgecolor="white")

    for i, (b, a) in enumerate(zip(before_ranks, after_ranks)):
        ax_d.annotate("", xy=(x[i] + w/2, a + 0.5), xytext=(x[i] + w/2, b - 0.5),
                      arrowprops=dict(arrowstyle="-|>", color="#2c7bb6", lw=1.5, mutation_scale=10))

    family_before = vascular_summary.get("original_top5_pct", 25.0)
    family_after  = vascular_summary.get("rescued_top5_pct", 100.0)
    ax_d.axhline(5, color="#888", lw=1, ls=":", label="5% threshold")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(factor_labels)
    ax_d.set_ylabel("Percentile rank (%)")
    ax_d.set_title(
        "Vascular/endothelial synthetic rescue\n"
        f"Assign median phenotype score to ETV2, FLI1, ERG\n"
        f"Family top-5%: {family_before:.0f}% → {family_after:.0f}%"
    )
    ax_d.legend(frameon=False, fontsize=8)
    ax_d.text(-0.18, 1.05, "d", transform=ax_d.transAxes,
              fontsize=10, fontweight="bold")

    for path_suffix in [".png", ".pdf"]:
        out = os.path.join(OUT_DIR, f"SF4_sensitivity_holdout{path_suffix}")
        fig.savefig(out)
        print(f"  ✓ {out}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("Generating supplementary figures SF2–SF4...")
    print("\nSF2 — Validation performance:")
    make_sf2()
    print("\nSF3 — Statistical controls:")
    make_sf3()
    print("\nSF4 — Sensitivity and holdout analyses:")
    make_sf4()
    print("\nDone. All figures written to", OUT_DIR)
    print("\nNote: SF5–SF7 (methods diagrams) already exist:")
    for f in ["fig21_scoring_pipeline.png", "fig_supp_set3_detail.png",
              "fig_supp_ppi_shuffle.png"]:
        full = os.path.join(OUT_DIR, f)
        print(f"  {'✓' if os.path.exists(full) else '✗'} {full}")


if __name__ == "__main__":
    main()
