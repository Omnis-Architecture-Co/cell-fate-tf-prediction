#!/usr/bin/env python3
"""
Scoring Function Tuning for VM Cocktail Predictor
===================================================
Tests 8 scoring functions against 3 wetlab-validated iPSC reprogramming cocktails.
Identifies which scoring approach best enriches known reprogramming factors.

Usage: python3 paper2/scoring_function_tuning.py
"""

import csv, json, math, os, pickle, sys, time
from collections import defaultdict

import numpy as np
from scipy.stats import wasserstein_distance, binom

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
STATE_PATH = "/tmp/module8_full_state.pkl"
ROSETTA_PATH = os.path.join(PROJECT_ROOT, "beta_transfer", "convergence_rosetta_stone.csv")
DISRUPTION_PATH = os.path.join(PROJECT_ROOT, "validation", "knockout", "disruption_profiles_full.json")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "scoring_function_comparison.json")

KNOWN_ENHANCERS = [
    "MYC", "GLIS1", "LIN28A", "LIN28B", "NANOG", "ESRRB", "TBX3",
    "UTF1", "SALL4", "KDM2B", "JMJD1C", "DOT1L", "BRD4", "TERT",
    "TET1", "TET2", "KDM4B", "RCOR2", "MBD3", "PRDM14", "FOXH1", "NR5A2",
]

ICM_MARKERS = ["POU5F1", "NANOG", "SOX2", "KLF4", "ESRRB", "TBX3", "TFCP2L1", "GBX2"]

EXP1_ANCHOR = ["POU5F1", "SOX2", "KLF4"]
EXP2_ANCHOR = ["POU5F1", "SOX2"]
EXP3_ANCHOR = ["POU5F1", "SOX2", "KLF4"]


def load_state():
    if os.path.exists(STATE_PATH):
        print(f"  Loading cached state from {STATE_PATH}")
        with open(STATE_PATH, "rb") as f:
            state = pickle.load(f)
        print(f"  Loaded: {state['n_tokens']} tokens, {state['n_proteins']} proteins")
        return state
    else:
        print("  State not found, rebuilding from DB...")
        sys.path.insert(0, PROJECT_ROOT)
        from validation.sensitivity.module8_full_shuffle import load_state_from_db
        return load_state_from_db()


def load_rosetta():
    rosetta = {}
    all_functions = set()
    with open(ROSETTA_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            hex_raw = row["word_hex"].strip()
            token = hex_raw.lstrip("0x")
            func = row["primary_function"].strip()
            conv = int(row["convergences"])
            layer = row["layer"].strip()
            rosetta[token] = {"function": func, "convergences": conv, "layer": layer}
            all_functions.add(func)
    all_functions = sorted(all_functions)
    func_to_idx = {f: i for i, f in enumerate(all_functions)}
    print(f"  Rosetta: {len(rosetta)} tokens, {len(all_functions)} functions")
    return rosetta, all_functions, func_to_idx


def build_gene_data(state, rosetta, all_functions, func_to_idx):
    ptt = state["ptt"]
    gene_cache = state["gene_cache"]
    n_funcs = len(all_functions)

    gene_to_uids = defaultdict(list)
    for uid, name in gene_cache.items():
        gene_to_uids[name].append(uid)

    gene_names = sorted(gene_to_uids.keys())
    gene_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)

    raw_matrix = np.zeros((n_genes, n_funcs), dtype=np.float64)
    dispatch_matrix = np.zeros((n_genes, n_funcs), dtype=np.float64)
    total_conv = np.zeros(n_genes, dtype=np.float64)

    n_tokens_total = np.zeros(n_genes, dtype=np.float64)
    n_tokens_mapped = np.zeros(n_genes, dtype=np.float64)

    for gi, gene in enumerate(gene_names):
        seen_tokens = set()
        mapped_tokens = set()
        for uid in gene_to_uids[gene]:
            for tok in ptt.get(uid, []):
                seen_tokens.add(tok)
                if tok in rosetta:
                    mapped_tokens.add(tok)
                    entry = rosetta[tok]
                    fi = func_to_idx[entry["function"]]
                    c = entry["convergences"]
                    raw_matrix[gi, fi] += c
                    w = 2.0 if entry["layer"] == "DISPATCH" else 1.0
                    dispatch_matrix[gi, fi] += c * w
        n_tokens_total[gi] = len(seen_tokens)
        n_tokens_mapped[gi] = len(mapped_tokens)

    total_conv = raw_matrix.sum(axis=1)
    has_data = total_conv > 0
    valid_mask = has_data

    norm_matrix = np.zeros_like(raw_matrix)
    norm_matrix[has_data] = raw_matrix[has_data] / total_conv[has_data, None]

    dispatch_total = dispatch_matrix.sum(axis=1)
    dispatch_norm = np.zeros_like(dispatch_matrix)
    d_has = dispatch_total > 0
    dispatch_norm[d_has] = dispatch_matrix[d_has] / dispatch_total[d_has, None]

    valid_genes = [g for g, m in zip(gene_names, valid_mask) if m]
    print(f"  Gene vectors: {len(valid_genes)} genes with convergence data")

    return {
        "gene_names": gene_names,
        "gene_idx": gene_idx,
        "raw_matrix": raw_matrix,
        "norm_matrix": norm_matrix,
        "dispatch_norm": dispatch_norm,
        "total_conv": total_conv,
        "valid_mask": valid_mask,
        "n_funcs": n_funcs,
        "gene_to_uids": gene_to_uids,
        "n_tokens_total": n_tokens_total,
        "n_tokens_mapped": n_tokens_mapped,
    }


def build_target_signature(icm_markers, gd):
    vecs = []
    for g in icm_markers:
        if g in gd["gene_idx"] and gd["valid_mask"][gd["gene_idx"][g]]:
            vecs.append(gd["norm_matrix"][gd["gene_idx"][g]])
    sig = np.mean(vecs, axis=0)
    s = sig.sum()
    return sig / s if s > 0 else sig


def cosine_sim_batch(vecs, target):
    norms = np.linalg.norm(vecs, axis=1)
    t_norm = np.linalg.norm(target)
    dots = vecs @ target
    denom = norms * t_norm
    result = np.full(len(vecs), -999.0)
    good = denom > 0
    result[good] = dots[good] / denom[good]
    return result


def batch_score_all(anchor_genes, gd, target_sig, target_func_idxs,
                    all_genes_ranks, filtered_set, exclude_set):
    gene_idx = gd["gene_idx"]
    raw = gd["raw_matrix"]
    norm = gd["norm_matrix"]
    dnorm = gd["dispatch_norm"]
    tc = gd["total_conv"]
    valid = gd["valid_mask"]
    n = len(gd["gene_names"])
    n_funcs = gd["n_funcs"]

    anchor_idxs = [gene_idx[g] for g in anchor_genes if g in gene_idx and valid[gene_idx[g]]]
    if not anchor_idxs:
        empty = np.full(n, -999.0)
        names = [
            "S1_convergence_shift", "S2_cosine_target", "S3_kl_divergence",
            "S4_dispatch_weighted_cosine", "S5_earth_mover", "S6_coverage_x_alignment",
            "S7_rank_complementarity", "S8_residual_hybrid",
            "S9_marginal_kl_confidence", "S10_amplifier_specifier", "S11_geometric_rank_agg",
        ]
        return {name: empty.copy() for name in names}

    anchor_raw_sum = raw[anchor_idxs].sum(axis=0)
    anchor_total = anchor_raw_sum.sum()

    exclude_mask = np.zeros(n, dtype=bool)
    for g in exclude_set:
        if g in gene_idx:
            exclude_mask[gene_idx[g]] = True
    invalid = ~valid | exclude_mask

    scores = {}

    anchor_frac = sum(anchor_raw_sum[i] for i in target_func_idxs) / anchor_total if anchor_total > 0 else 0
    combined_raw = anchor_raw_sum[None, :] + raw
    combined_totals = combined_raw.sum(axis=1)
    target_sums = sum(combined_raw[:, i] for i in target_func_idxs)
    combined_frac = np.where(combined_totals > 0, target_sums / combined_totals, 0)
    shift = combined_frac - anchor_frac
    log_tc = np.where(tc > 1, np.log10(tc), 0)
    s1 = shift * log_tc
    s1[invalid] = -999.0
    scores["S1_convergence_shift"] = s1

    combined_norms = np.zeros_like(combined_raw)
    good_ct = combined_totals > 0
    combined_norms[good_ct] = combined_raw[good_ct] / combined_totals[good_ct, None]
    s2 = cosine_sim_batch(combined_norms, target_sig)
    s2[invalid] = -999.0
    scores["S2_cosine_target"] = s2

    eps = 1e-10
    p_clip = np.clip(combined_norms, eps, None)
    q_clip = np.clip(target_sig, eps, None)
    p_norm = p_clip / p_clip.sum(axis=1, keepdims=True)
    q_norm = q_clip / q_clip.sum()
    s3 = -np.sum(p_norm * np.log(p_norm / q_norm[None, :]), axis=1)
    s3[invalid | ~good_ct] = -999.0
    scores["S3_kl_divergence"] = s3

    anchor_dnorm = dnorm[anchor_idxs].mean(axis=0) if anchor_idxs else np.zeros(n_funcs)
    n_anchor = len(anchor_idxs)
    combined_dnorm = (anchor_dnorm * n_anchor + dnorm) / (n_anchor + 1)
    s4 = cosine_sim_batch(combined_dnorm, target_sig)
    s4[invalid] = -999.0
    scores["S4_dispatch_weighted_cosine"] = s4

    s5 = np.full(n, -999.0)
    for gi in range(n):
        if invalid[gi] or not good_ct[gi]:
            continue
        s5[gi] = -wasserstein_distance(combined_norms[gi], target_sig)
    scores["S5_earth_mover"] = s5

    target_active = target_sig > 0.01
    coverage = np.sum((combined_norms > 0.05) & target_active[None, :], axis=1).astype(float)
    alignment = cosine_sim_batch(combined_norms, target_sig)
    alignment[alignment < -998] = 0
    s6 = coverage * alignment
    s6[invalid] = -999.0
    scores["S6_coverage_x_alignment"] = s6

    target_func_active = [fi for fi in range(n_funcs) if target_sig[fi] > 0.01]
    if target_func_active and all_genes_ranks is not None:
        rr_sum = np.zeros(n, dtype=np.float64)
        for fi in target_func_active:
            ranks = all_genes_ranks[fi]
            rr_sum += 1.0 / ranks
        s7 = rr_sum / len(target_func_active)
    else:
        s7 = np.full(n, -999.0)
    s7[invalid] = -999.0
    scores["S7_rank_complementarity"] = s7

    s8 = s2.copy()
    for gi in range(n):
        if gd["gene_names"][gi] not in filtered_set:
            s8[gi] = -999.0
    s8[invalid] = -999.0
    scores["S8_residual_hybrid"] = s8

    anchor_norm_vec = anchor_raw_sum / anchor_total if anchor_total > 0 else np.zeros(n_funcs)
    anchor_norm_clip = np.clip(anchor_norm_vec, eps, None)
    anchor_norm_clip = anchor_norm_clip / anchor_norm_clip.sum()
    target_clip = np.clip(target_sig, eps, None)
    target_clip = target_clip / target_clip.sum()

    kl_anchor = float(np.sum(target_clip * np.log(target_clip / anchor_norm_clip)))

    n_mapped = gd["n_tokens_mapped"]
    coverage_confidence = n_mapped / (n_mapped + 5.0)

    s9 = np.full(n, -999.0)
    for gi in range(n):
        if invalid[gi] or not good_ct[gi]:
            continue
        cn = np.clip(combined_norms[gi], eps, None)
        cn = cn / cn.sum()
        kl_combined = float(np.sum(target_clip * np.log(target_clip / cn)))
        delta_kl = kl_anchor - kl_combined
        s9[gi] = delta_kl * coverage_confidence[gi]
    scores["S9_marginal_kl_confidence"] = s9

    interaction = anchor_norm_vec * target_sig
    interaction_sum = interaction.sum()
    if interaction_sum > 0:
        interaction_normalized = interaction / interaction_sum
    else:
        interaction_normalized = np.ones(n_funcs) / n_funcs

    specifier = cosine_sim_batch(norm, target_sig)
    specifier[specifier < -998] = 0.0

    synergy = cosine_sim_batch(norm, interaction_normalized)
    synergy[synergy < -998] = 0.0

    log_power = np.where(tc > 1, np.log10(tc), 0.0)

    s10 = 0.4 * specifier + 0.6 * synergy * log_power
    s10[invalid] = -999.0
    scores["S10_amplifier_specifier"] = s10

    valid_s1 = s1.copy()
    valid_s1[valid_s1 <= -999] = np.nan
    valid_s5 = s5.copy()
    valid_s5[valid_s5 <= -999] = np.nan

    n_valid_s1 = int(np.sum(~np.isnan(valid_s1)))
    n_valid_s5 = int(np.sum(~np.isnan(valid_s5)))

    rank_s1 = np.full(n, np.nan)
    rank_s5 = np.full(n, np.nan)

    if n_valid_s1 > 0:
        order_s1 = np.argsort(-np.nan_to_num(valid_s1, nan=-1e10))
        for r, gi in enumerate(order_s1[:n_valid_s1]):
            rank_s1[gi] = (r + 1) / n_valid_s1

    if n_valid_s5 > 0:
        order_s5 = np.argsort(-np.nan_to_num(valid_s5, nan=-1e10))
        for r, gi in enumerate(order_s5[:n_valid_s5]):
            rank_s5[gi] = (r + 1) / n_valid_s5

    s11 = np.full(n, -999.0)
    for gi in range(n):
        if not np.isnan(rank_s1[gi]) and not np.isnan(rank_s5[gi]) and not invalid[gi]:
            s11[gi] = -math.sqrt(rank_s1[gi] * rank_s5[gi])
    scores["S11_geometric_rank_agg"] = s11

    return scores


def rank_gene(gene, gene_idx, scores):
    if gene not in gene_idx:
        return None, None
    gi = gene_idx[gene]
    s = scores[gi]
    if s <= -999:
        return None, None
    valid_scores = scores[scores > -999]
    rank = int(np.sum(valid_scores > s)) + 1
    n_valid = len(valid_scores)
    return rank, round(rank / n_valid * 100, 1)


def compute_enrichment(known_genes, gene_idx, scores):
    valid_scores = scores[scores > -999]
    n_total = len(valid_scores)

    ranks = []
    percentiles = []
    gene_results = {}
    for g in known_genes:
        r, p = rank_gene(g, gene_idx, scores)
        if r is not None:
            ranks.append(r)
            percentiles.append(p)
            gene_results[g] = {"rank": r, "percentile": p}

    found = len(ranks)
    if found == 0:
        return {"found": 0}

    in_top5 = sum(1 for p in percentiles if p <= 5)
    in_top10 = sum(1 for p in percentiles if p <= 10)
    in_top25 = sum(1 for p in percentiles if p <= 25)
    p_binom = float(binom.sf(in_top25 - 1, found, 0.25))

    return {
        "found": found,
        "n_total": n_total,
        "median_rank": int(np.median(ranks)),
        "median_percentile": round(float(np.median(percentiles)), 1),
        "in_top5pct": in_top5,
        "in_top10pct": in_top10,
        "in_top25pct": in_top25,
        "binom_p_top25": round(p_binom, 6),
        "individual_ranks": gene_results,
    }


def get_top_n(scores, gene_names, n=10):
    valid = [(gene_names[i], float(scores[i])) for i in range(len(scores)) if scores[i] > -999]
    valid.sort(key=lambda x: -x[1])
    return [(g, round(s, 6)) for g, s in valid[:n]]


def main():
    t0 = time.time()
    print("=" * 72)
    print("  SCORING FUNCTION TUNING — iPSC Reprogramming Cocktails")
    print("=" * 72)

    print("\n[1] Loading state...")
    state = load_state()

    print("\n[2] Loading rosetta stone...")
    rosetta, all_functions, func_to_idx = load_rosetta()

    print("\n[3] Building gene data matrices...")
    gd = build_gene_data(state, rosetta, all_functions, func_to_idx)

    print("\n[4] Building target signature from ICM markers...")
    target_sig = build_target_signature(ICM_MARKERS, gd)
    top_idxs = np.argsort(-target_sig)[:5]
    top_funcs = [(all_functions[i], float(target_sig[i])) for i in top_idxs]
    print(f"  ICM signature top functions:")
    for fn, w in top_funcs:
        print(f"    {fn}: {w:.3f}")

    target_func_idxs = [int(top_idxs[0]), int(top_idxs[1])]

    print("\n[5] Pre-computing per-function ranks for S7...")
    n_genes = len(gd["gene_names"])
    n_funcs = gd["n_funcs"]
    all_genes_ranks = np.zeros((n_funcs, n_genes), dtype=np.float64)
    for fi in range(n_funcs):
        col = gd["norm_matrix"][:, fi].copy()
        col[~gd["valid_mask"]] = -1
        order = np.argsort(-col)
        rank_arr = np.zeros(n_genes, dtype=np.float64)
        for r, gi in enumerate(order):
            rank_arr[gi] = r + 1
        all_genes_ranks[fi] = rank_arr

    print("\n[6] Loading disruption profiles for S8...")
    with open(DISRUPTION_PATH) as f:
        disruption_data = json.load(f)
    profiles = disruption_data["profiles"]
    departments = disruption_data["departments"]

    profile_genes = sorted(profiles.keys())
    profile_matrix = np.array([[profiles[g].get(d, 0) for d in departments] for g in profile_genes])
    mean_profile = profile_matrix.mean(axis=0)
    centered = profile_matrix - mean_profile
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = Vt[0]
    projections = centered @ pc1
    residuals = centered - np.outer(projections, pc1)
    res_norms = np.linalg.norm(residuals, axis=1)
    median_res = float(np.median(res_norms))

    filtered_set = set()
    for i, g in enumerate(profile_genes):
        if res_norms[i] > median_res and g in gd["gene_idx"] and gd["valid_mask"][gd["gene_idx"][g]]:
            filtered_set.add(g)
    print(f"  S8 filter: {len(filtered_set)} genes with residual norm > median ({median_res:.4f})")

    scoring_names = [
        "S1_convergence_shift", "S2_cosine_target", "S3_kl_divergence",
        "S4_dispatch_weighted_cosine", "S5_earth_mover", "S6_coverage_x_alignment",
        "S7_rank_complementarity", "S8_residual_hybrid",
        "S9_marginal_kl_confidence", "S10_amplifier_specifier", "S11_geometric_rank_agg",
    ]

    def run_experiment(name, anchor, exclude_extra=None):
        exclude = set(anchor)
        if exclude_extra:
            exclude |= set(exclude_extra)
        all_scores = batch_score_all(
            anchor, gd, target_sig, target_func_idxs,
            all_genes_ranks, filtered_set, exclude
        )
        return all_scores

    print("\n" + "=" * 72)
    print("  EXPERIMENT 1: OSKM (Yamanaka) — Rank 4th factor")
    print("  Anchor: POU5F1, SOX2, KLF4 | Ground truth: MYC")
    print("=" * 72)

    exp1_scores = run_experiment("Exp1", EXP1_ANCHOR)
    exp1_results = {}
    for sn in scoring_names:
        sc = exp1_scores[sn]
        myc_r, myc_p = rank_gene("MYC", gd["gene_idx"], sc)
        glis1_r, glis1_p = rank_gene("GLIS1", gd["gene_idx"], sc)
        enr = compute_enrichment(KNOWN_ENHANCERS, gd["gene_idx"], sc)
        top10 = get_top_n(sc, gd["gene_names"], 10)
        n_valid = int(np.sum(sc > -999))

        exp1_results[sn] = {
            "n_candidates": n_valid,
            "myc_rank": myc_r, "myc_percentile": myc_p,
            "glis1_rank": glis1_r, "glis1_percentile": glis1_p,
            "enrichment": enr,
            "top10": top10,
        }
        print(f"\n  {sn}:")
        print(f"    MYC: rank {myc_r}/{n_valid} ({myc_p}%)")
        print(f"    GLIS1: rank {glis1_r}/{n_valid} ({glis1_p}%)")
        print(f"    Enhancers: median pct={enr.get('median_percentile','N/A')}, "
              f"top25={enr.get('in_top25pct',0)}/{enr.get('found',0)}, "
              f"p={enr.get('binom_p_top25','N/A')}")
        print(f"    Top 5: {[g for g, s in top10[:5]]}")

    print("\n" + "=" * 72)
    print("  EXPERIMENT 2: OSNL (Thomson) — Rank 3rd+4th factor (greedy)")
    print("  Anchor: POU5F1, SOX2 | Ground truth: NANOG + LIN28A")
    print("=" * 72)

    exp2_scores = run_experiment("Exp2", EXP2_ANCHOR)
    exp2_results = {}
    for sn in scoring_names:
        sc = exp2_scores[sn]
        n_valid = int(np.sum(sc > -999))
        nanog_r, nanog_p = rank_gene("NANOG", gd["gene_idx"], sc)
        lin28a_r, lin28a_p = rank_gene("LIN28A", gd["gene_idx"], sc)

        top_sorted = [(gd["gene_names"][i], float(sc[i])) for i in range(len(sc)) if sc[i] > -999]
        top_sorted.sort(key=lambda x: -x[1])
        best_3rd = top_sorted[0][0] if top_sorted else None

        exp2_4th_scores = run_experiment("Exp2_4th", EXP2_ANCHOR, exclude_extra=[best_3rd] if best_3rd else [])
        sc4 = exp2_4th_scores[sn]

        nanog_r4, nanog_p4 = rank_gene("NANOG", gd["gene_idx"], sc4)
        lin28a_r4, lin28a_p4 = rank_gene("LIN28A", gd["gene_idx"], sc4)

        exp2_results[sn] = {
            "best_3rd_factor": best_3rd,
            "nanog_rank_as_3rd": nanog_r, "nanog_percentile_as_3rd": nanog_p if nanog_p is not None else "N/A",
            "lin28a_rank_as_3rd": lin28a_r, "lin28a_percentile_as_3rd": lin28a_p if lin28a_p is not None else "N/A",
            "nanog_rank_as_4th": nanog_r4, "lin28a_rank_as_4th": lin28a_r4,
        }
        nanog_p_str = f"{nanog_p}%" if nanog_p is not None else "N/A"
        lin28a_p_str = f"{lin28a_p}%" if lin28a_p is not None else "N/A"
        print(f"\n  {sn}:")
        print(f"    Best 3rd: {best_3rd}")
        print(f"    NANOG as 3rd: rank {nanog_r} ({nanog_p_str})")
        print(f"    LIN28A as 3rd: rank {lin28a_r} ({lin28a_p_str})")

    print("\n" + "=" * 72)
    print("  EXPERIMENT 3: GLIS1 replacing MYC")
    print("  Anchor: POU5F1, SOX2, KLF4 | Ground truth: GLIS1 ranks highly")
    print("=" * 72)

    exp3_scores = run_experiment("Exp3", EXP3_ANCHOR)
    exp3_results = {}
    for sn in scoring_names:
        sc = exp3_scores[sn]
        n_valid = int(np.sum(sc > -999))
        glis1_r, glis1_p = rank_gene("GLIS1", gd["gene_idx"], sc)
        top10 = get_top_n(sc, gd["gene_names"], 10)

        exp3_results[sn] = {
            "n_candidates": n_valid,
            "glis1_rank": glis1_r, "glis1_percentile": glis1_p,
            "top10": top10,
        }
        print(f"\n  {sn}:")
        print(f"    GLIS1: rank {glis1_r}/{n_valid} ({glis1_p}%)")

    all_results = {
        "metadata": {
            "n_genes_total": len(gd["gene_names"]),
            "n_genes_with_convergence": int(gd["valid_mask"].sum()),
            "n_rosetta_tokens": len(rosetta),
            "n_functions": len(all_functions),
            "functions": all_functions,
            "icm_markers": ICM_MARKERS,
            "known_enhancers": KNOWN_ENHANCERS,
            "target_signature_top5": [(fn, round(w, 4)) for fn, w in top_funcs],
            "s8_median_residual": median_res,
            "s8_filtered_count": len(filtered_set),
            "runtime_seconds": round(time.time() - t0, 1),
        },
        "experiment_1_OSKM": exp1_results,
        "experiment_2_OSNL": exp2_results,
        "experiment_3_GLIS1": exp3_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_PATH}")

    print("\n" + "=" * 72)
    print("  SUMMARY COMPARISON TABLE — Experiment 1 (OSKM)")
    print("=" * 72)
    header = f"{'Scoring Function':<30} {'MYC Rank':<12} {'MYC %':<8} {'GLIS1 Rank':<12} {'GLIS1 %':<8} {'Med Enh %':<10} {'Top25':<6} {'p-val':<10}"
    print(header)
    print("-" * len(header))
    for sn in scoring_names:
        e1 = exp1_results[sn]
        enr = e1.get("enrichment", {})
        nc = e1["n_candidates"]
        mr = e1["myc_rank"]; mp = e1["myc_percentile"]
        gr = e1["glis1_rank"]; gp = e1["glis1_percentile"]
        med = enr.get("median_percentile", "?")
        t25 = enr.get("in_top25pct", "?"); fo = enr.get("found", "?")
        pv = enr.get("binom_p_top25", "?")
        mr_s = f"{mr}/{nc}" if mr else "N/A"
        gr_s = f"{gr}/{nc}" if gr else "N/A"
        pv_s = f"{pv:.6f}" if isinstance(pv, float) else str(pv)
        print(f"{sn:<30} {mr_s:<12} {mp:<8} {gr_s:<12} {gp:<8} {med:<10} {t25}/{fo:<4} {pv_s:<10}")

    print("\n" + "=" * 72)
    print("  EXPERIMENT 2: NANOG+LIN28A pair ranking (greedy)")
    print("=" * 72)
    header2 = f"{'Scoring Function':<30} {'Best 3rd':<12} {'NANOG 3rd%':<12} {'LIN28A 3rd%':<12}"
    print(header2)
    print("-" * len(header2))
    for sn in scoring_names:
        e2 = exp2_results[sn]
        np_str = str(e2['nanog_percentile_as_3rd'])
        lp_str = str(e2['lin28a_percentile_as_3rd'])
        print(f"{sn:<30} {e2['best_3rd_factor']:<12} {np_str:<12} {lp_str:<12}")

    print("\n" + "=" * 72)
    print("  EXPERIMENT 3: GLIS1 replacement rank")
    print("=" * 72)
    for sn in scoring_names:
        e3 = exp3_results[sn]
        print(f"  {sn:<30} GLIS1: rank {e3['glis1_rank']}/{e3['n_candidates']} ({e3['glis1_percentile']}%)")

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
