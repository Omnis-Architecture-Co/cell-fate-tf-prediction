#!/usr/bin/env python3
"""
Enhanced Null Model Analysis for Knockout Simulation
=====================================================

Three analyses:
1. Cohen's d for each gene (real vs 100 null knockouts) — all 19,375 genes
2. Distribution of real vs null disruption scores
3. Harder null: token-assignment shuffle on 500 stratified genes × 100 perms

Key optimization: precompute per-token disruption lookup table.
Total disruption = sum of dept_mask @ P column for each token.
This makes each permutation O(n_tokens_in_gene) instead of sparse matmul.

Usage:
    python3 -u validation/knockout/enhanced_null_analysis.py
"""

import csv
import json
import os
import pickle
import random
import sys
import time

import numpy as np
from collections import defaultdict
from scipy import sparse

STATE_PATH = "/tmp/module8_full_state.pkl"
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "gene_manifest.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "enhanced_null_results.json")

VALID_DEPARTMENTS = sorted([
    "Apoptosis", "Cell adhesion", "Cell cycle", "Chromatin", "Cytoskeleton",
    "DNA repair", "GTPase", "Immune", "Ion channel", "Kinase", "Methylation",
    "Nuc acid bind", "Phosphatase", "Protein folding", "Proteolysis",
    "RNA processing", "Signaling", "Structural", "Transcription",
    "Translation", "Transport", "Ubiquitin",
])

N_NULL_RANDOM = 100
N_NULL_ASSIGNMENT = 100
N_STRATIFIED = 500
PROGRESS_EVERY = 500


def find_dept_csv():
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "server", "data", "human", "gene_departments.csv"),
        "server/data/human/gene_departments.csv",
    ]
    for c in candidates:
        p = os.path.normpath(c)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("gene_departments.csv not found")


def load_all():
    print("[1/5] Loading state...")
    t0 = time.time()
    with open(STATE_PATH, "rb") as f:
        state = pickle.load(f)
    ptt = state["ptt"]
    ttp = state["ttp"]
    gene_cache = state["gene_cache"]
    gene_to_uid = state["gene_to_uid"]

    gene_depts = {}
    with open(find_dept_csv()) as f:
        for row in csv.DictReader(f):
            gene_depts[row["gene"]] = row["department"]

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    print(f"  Loaded in {time.time()-t0:.1f}s")
    return ptt, ttp, gene_cache, gene_to_uid, gene_depts, manifest


def build_matrix_and_lookup(ttp, ptt, gene_cache, gene_depts):
    print("[2/5] Building sparse matrix and per-token disruption lookup...")
    t0 = time.time()
    all_tokens = sorted(ttp.keys())
    all_proteins = sorted(ptt.keys())
    tok_to_idx = {t: i for i, t in enumerate(all_tokens)}
    uid_to_idx = {u: i for i, u in enumerate(all_proteins)}
    n_p, n_t = len(all_proteins), len(all_tokens)

    rows, cols = [], []
    for tok, uids in ttp.items():
        ti = tok_to_idx[tok]
        for uid in uids:
            rows.append(uid_to_idx[uid])
            cols.append(ti)

    P = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_p, n_t),
    )

    valid_set = set(VALID_DEPARTMENTS)
    combined_mask = np.zeros(n_p, dtype=np.float32)
    dept_masks = {}
    for dept in VALID_DEPARTMENTS:
        dept_masks[dept] = np.zeros(n_p, dtype=np.float32)
    for uid, idx in uid_to_idx.items():
        gene = gene_cache.get(uid)
        if gene:
            dept = gene_depts.get(gene)
            if dept and dept in valid_set:
                dept_masks[dept][idx] = 1.0
                combined_mask[idx] = 1.0

    total_indexed = int(combined_mask.sum())

    token_disruption = np.asarray(combined_mask @ P).flatten()

    print(f"  Matrix: {n_p}x{n_t}, {P.nnz} edges, {total_indexed} indexed proteins")
    print(f"  Token disruption lookup: {n_t} entries")
    print(f"    mean={token_disruption.mean():.4f}, max={token_disruption.max():.1f}, "
          f"nonzero={np.count_nonzero(token_disruption)}/{n_t}")
    print(f"  Built in {time.time()-t0:.1f}s")
    return P, tok_to_idx, uid_to_idx, dept_masks, token_disruption, n_p, n_t, total_indexed


def select_stratified_sample(manifest, n=500, seed=42):
    rng = random.Random(seed)
    genes = manifest["genes"]
    all_gene_names = sorted(genes.keys())

    by_tokens = defaultdict(list)
    for g in all_gene_names:
        nt = genes[g]["n_tokens"]
        if nt <= 5:
            by_tokens["1-5"].append(g)
        elif nt <= 20:
            by_tokens["6-20"].append(g)
        elif nt <= 50:
            by_tokens["21-50"].append(g)
        else:
            by_tokens["51+"].append(g)

    selected = []
    per_bin = n // len(by_tokens)
    for bin_name in sorted(by_tokens.keys()):
        pool = by_tokens[bin_name]
        take = min(per_bin, len(pool))
        selected.extend(rng.sample(pool, take))

    while len(selected) < n:
        remaining = [g for g in all_gene_names if g not in set(selected)]
        selected.append(rng.choice(remaining))

    return selected[:n]


def build_gene_token_pools(P, uid_to_idx, gene_to_uid, manifest):
    gene_tokens = {}
    for gene in manifest["genes"]:
        uid = manifest["genes"][gene]["uid"]
        if uid in uid_to_idx:
            pidx = uid_to_idx[uid]
            toks = list(P[pidx].indices)
            if toks:
                gene_tokens[gene] = np.array(toks, dtype=np.int32)
    return gene_tokens


def main():
    ptt, ttp, gene_cache, gene_to_uid, gene_depts, manifest = load_all()
    P, tok_to_idx, uid_to_idx, dept_masks, token_disruption, n_p, n_t, total_indexed = \
        build_matrix_and_lookup(ttp, ptt, gene_cache, gene_depts)

    all_genes = sorted(manifest["genes"].keys())
    gene_tokens = build_gene_token_pools(P, uid_to_idx, gene_to_uid, manifest)
    valid_genes = [g for g in all_genes if g in gene_tokens]
    print(f"  Valid genes (have tokens): {len(valid_genes)}")

    all_tok_indices = np.arange(n_t, dtype=np.int32)

    print(f"\n[3/5] Analysis 1 & 2: Real vs Random-Position Null ({N_NULL_RANDOM} perms x {len(valid_genes)} genes)...")
    print(f"  Using precomputed per-token disruption lookup (O(k) per eval)...")
    t0 = time.time()

    real_scores = np.zeros(len(valid_genes), dtype=np.float64)
    null_means = np.zeros(len(valid_genes), dtype=np.float64)
    null_stds = np.zeros(len(valid_genes), dtype=np.float64)
    cohens_d_values = np.zeros(len(valid_genes), dtype=np.float64)

    gene_sizes = [len(gene_tokens[g]) for g in valid_genes]
    unique_sizes = sorted(set(gene_sizes))
    print(f"  Unique token counts: {len(unique_sizes)} (range {min(unique_sizes)}-{max(unique_sizes)})")
    sys.stdout.flush()

    for gi, gene in enumerate(valid_genes):
        toks = gene_tokens[gene]
        k = len(toks)
        real_val = float(token_disruption[toks].sum())

        rand_matrix = np.zeros((N_NULL_RANDOM, k), dtype=np.int32)
        for pi in range(N_NULL_RANDOM):
            rand_matrix[pi] = np.random.choice(n_t, size=k, replace=False)
        null_vals = token_disruption[rand_matrix].sum(axis=1)

        nm = float(null_vals.mean())
        ns = float(null_vals.std())

        real_scores[gi] = real_val
        null_means[gi] = nm
        null_stds[gi] = ns
        cohens_d_values[gi] = (real_val - nm) / ns if ns > 0 else 0

        if (gi + 1) % PROGRESS_EVERY == 0 or gi == len(valid_genes) - 1:
            elapsed = time.time() - t0
            rate = (gi + 1) / elapsed
            eta = (len(valid_genes) - gi - 1) / rate
            d_so_far = cohens_d_values[:gi+1]
            print(f"  [{gi+1:5d}/{len(valid_genes)}] "
                  f"median_d={np.median(d_so_far):+.3f} "
                  f"mean_d={np.mean(d_so_far):+.3f} "
                  f"d>0.5={np.mean(d_so_far>0.5):.1%} "
                  f"d>0.8={np.mean(d_so_far>0.8):.1%} "
                  f"({elapsed:.0f}s, ETA={eta/60:.1f}m)")
            sys.stdout.flush()

    elapsed_1 = time.time() - t0

    d_arr = cohens_d_values
    ratios = np.where(null_means > 0, real_scores / null_means, 0)
    abs_diffs = real_scores - null_means
    valid_ratios_12 = ratios[null_means > 0]

    print(f"\n  Complete in {elapsed_1:.0f}s")
    print(f"\n  === Analysis 1: Per-Gene Cohen's d (real vs {N_NULL_RANDOM} random-position nulls) ===")
    print(f"  Mean d:   {d_arr.mean():+.4f}")
    print(f"  Median d: {np.median(d_arr):+.4f}")
    print(f"  IQR:      [{np.percentile(d_arr, 25):+.4f}, {np.percentile(d_arr, 75):+.4f}]")
    print(f"  d > 0.5 (medium): {np.mean(d_arr > 0.5):.1%} ({int(np.sum(d_arr > 0.5))}/{len(d_arr)})")
    print(f"  d > 0.8 (large):  {np.mean(d_arr > 0.8):.1%} ({int(np.sum(d_arr > 0.8))}/{len(d_arr)})")
    print(f"  d > 2.0:          {np.mean(d_arr > 2.0):.1%} ({int(np.sum(d_arr > 2.0))}/{len(d_arr)})")
    print(f"  d < 0 (null > real): {np.mean(d_arr < 0):.1%} ({int(np.sum(d_arr < 0))}/{len(d_arr)})")
    print(f"  5th percentile:   {np.percentile(d_arr, 5):+.4f}")
    print(f"  95th percentile:  {np.percentile(d_arr, 95):+.4f}")

    print(f"\n  === Analysis 2: Real vs Null Disruption Distributions ===")
    print(f"  Real mean total disruption:    {real_scores.mean():.2f}")
    print(f"  Null mean total disruption:    {null_means.mean():.2f}")
    print(f"  Absolute difference (mean):    {abs_diffs.mean():+.2f}")
    print(f"  Absolute difference (median):  {np.median(abs_diffs):+.2f}")
    print(f"  Ratio (real/null, mean):       {valid_ratios_12.mean():.4f}")
    print(f"  Ratio (real/null, median):     {np.median(valid_ratios_12):.4f}")
    print(f"  Real median:                   {np.median(real_scores):.2f}")
    print(f"  Null median:                   {np.median(null_means):.2f}")
    print(f"  Real std:                      {real_scores.std():.2f}")
    print(f"  Null mean std:                 {null_means.std():.2f}")

    gene_list_for_shuffle = list(gene_tokens.keys())

    print(f"\n[4/5] Analysis 3: Token-Assignment Shuffle ({N_STRATIFIED} genes x {N_NULL_ASSIGNMENT} perms)...")
    stratified = select_stratified_sample(manifest, N_STRATIFIED)
    stratified = [g for g in stratified if g in gene_tokens]
    print(f"  Selected {len(stratified)} stratified genes")
    print(f"  Using per-token disruption lookup for fast evaluation...")

    t0 = time.time()
    rng = random.Random(42)

    assign_real = np.zeros(len(stratified), dtype=np.float64)
    assign_null_means = np.zeros(len(stratified), dtype=np.float64)
    assign_null_stds = np.zeros(len(stratified), dtype=np.float64)
    assign_d_values = np.zeros(len(stratified), dtype=np.float64)

    for gi, gene in enumerate(stratified):
        toks = gene_tokens[gene]
        k = len(toks)
        real_val = float(token_disruption[toks].sum())

        null_vals = np.zeros(N_NULL_ASSIGNMENT, dtype=np.float64)
        for pi in range(N_NULL_ASSIGNMENT):
            donor = rng.choice(gene_list_for_shuffle)
            while donor == gene:
                donor = rng.choice(gene_list_for_shuffle)
            donor_toks = gene_tokens[donor]
            if len(donor_toks) >= k:
                sampled = np.array(rng.sample(list(donor_toks), k), dtype=np.int32)
            else:
                pool = list(donor_toks)
                while len(pool) < k:
                    d2 = rng.choice(gene_list_for_shuffle)
                    pool.extend(list(gene_tokens[d2]))
                sampled = np.array(rng.sample(pool, k), dtype=np.int32)
            null_vals[pi] = float(token_disruption[sampled].sum())

        nm = null_vals.mean()
        ns = null_vals.std()
        assign_real[gi] = real_val
        assign_null_means[gi] = nm
        assign_null_stds[gi] = ns
        assign_d_values[gi] = (real_val - nm) / ns if ns > 0 else 0

        if (gi + 1) % 100 == 0 or gi == len(stratified) - 1:
            elapsed = time.time() - t0
            rate = (gi + 1) / elapsed if elapsed > 0 else 1
            eta = (len(stratified) - gi - 1) / rate
            da = assign_d_values[:gi+1]
            print(f"  [{gi+1:3d}/{len(stratified)}] "
                  f"median_d={np.median(da):+.3f} "
                  f"d>0.5={np.mean(da>0.5):.1%} "
                  f"d>0.8={np.mean(da>0.8):.1%} "
                  f"({elapsed:.0f}s, ETA={eta/60:.1f}m)")
            sys.stdout.flush()

    elapsed_3 = time.time() - t0

    ad_arr = assign_d_values
    assign_ratios = np.where(assign_null_means > 0, assign_real / assign_null_means, 0)
    assign_abs_diffs = assign_real - assign_null_means
    valid_ratios_3 = assign_ratios[assign_null_means > 0]

    print(f"\n  Complete in {elapsed_3:.0f}s")
    print(f"\n  === Analysis 3: Token-Assignment Shuffle Results ===")
    print(f"  Per-gene Cohen's d (real vs gene-donor null):")
    print(f"    Mean d:   {ad_arr.mean():+.4f}")
    print(f"    Median d: {np.median(ad_arr):+.4f}")
    print(f"    IQR:      [{np.percentile(ad_arr, 25):+.4f}, {np.percentile(ad_arr, 75):+.4f}]")
    print(f"    d > 0.5:  {np.mean(ad_arr > 0.5):.1%} ({int(np.sum(ad_arr > 0.5))}/{len(ad_arr)})")
    print(f"    d > 0.8:  {np.mean(ad_arr > 0.8):.1%} ({int(np.sum(ad_arr > 0.8))}/{len(ad_arr)})")
    print(f"    d > 2.0:  {np.mean(ad_arr > 2.0):.1%} ({int(np.sum(ad_arr > 2.0))}/{len(ad_arr)})")
    print(f"    d < 0:    {np.mean(ad_arr < 0):.1%} ({int(np.sum(ad_arr < 0))}/{len(ad_arr)})")
    print(f"    5th pct:  {np.percentile(ad_arr, 5):+.4f}")
    print(f"    95th pct: {np.percentile(ad_arr, 95):+.4f}")
    print(f"  Disruption distributions:")
    print(f"    Real mean:    {assign_real.mean():.2f}")
    print(f"    Null mean:    {assign_null_means.mean():.2f}")
    print(f"    Abs diff:     {assign_abs_diffs.mean():+.2f}")
    print(f"    Ratio mean:   {valid_ratios_3.mean():.4f}")
    print(f"    Ratio median: {np.median(valid_ratios_3):.4f}")

    print(f"\n[5/5] Saving results...")
    results = {
        "analysis_1_random_position_null": {
            "description": "Per-gene Cohen's d: real knockout disruption vs 100 random-position token removals",
            "n_genes": int(len(valid_genes)),
            "n_null_perms": N_NULL_RANDOM,
            "cohens_d_mean": round(float(d_arr.mean()), 4),
            "cohens_d_median": round(float(np.median(d_arr)), 4),
            "cohens_d_iqr": [round(float(np.percentile(d_arr, 25)), 4),
                             round(float(np.percentile(d_arr, 75)), 4)],
            "cohens_d_5th_pct": round(float(np.percentile(d_arr, 5)), 4),
            "cohens_d_95th_pct": round(float(np.percentile(d_arr, 95)), 4),
            "cohens_d_gt_0.5_frac": round(float(np.mean(d_arr > 0.5)), 4),
            "cohens_d_gt_0.5_count": int(np.sum(d_arr > 0.5)),
            "cohens_d_gt_0.8_frac": round(float(np.mean(d_arr > 0.8)), 4),
            "cohens_d_gt_0.8_count": int(np.sum(d_arr > 0.8)),
            "cohens_d_gt_2.0_frac": round(float(np.mean(d_arr > 2.0)), 4),
            "cohens_d_gt_2.0_count": int(np.sum(d_arr > 2.0)),
            "cohens_d_lt_0_frac": round(float(np.mean(d_arr < 0)), 4),
            "cohens_d_lt_0_count": int(np.sum(d_arr < 0)),
        },
        "analysis_2_disruption_distributions": {
            "description": "Real vs null total disruption scores across all genes",
            "real_mean": round(float(real_scores.mean()), 2),
            "real_median": round(float(np.median(real_scores)), 2),
            "real_std": round(float(real_scores.std()), 2),
            "null_mean": round(float(null_means.mean()), 2),
            "null_median": round(float(np.median(null_means)), 2),
            "null_mean_std": round(float(null_means.std()), 2),
            "abs_diff_mean": round(float(abs_diffs.mean()), 2),
            "abs_diff_median": round(float(np.median(abs_diffs)), 2),
            "ratio_mean": round(float(valid_ratios_12.mean()), 4),
            "ratio_median": round(float(np.median(valid_ratios_12)), 4),
        },
        "analysis_3_token_assignment_shuffle": {
            "description": f"Harder null: replace gene tokens with tokens from other genes ({N_STRATIFIED} stratified genes x {N_NULL_ASSIGNMENT} perms)",
            "n_genes": int(len(stratified)),
            "n_null_perms": N_NULL_ASSIGNMENT,
            "cohens_d_mean": round(float(ad_arr.mean()), 4),
            "cohens_d_median": round(float(np.median(ad_arr)), 4),
            "cohens_d_iqr": [round(float(np.percentile(ad_arr, 25)), 4),
                             round(float(np.percentile(ad_arr, 75)), 4)],
            "cohens_d_5th_pct": round(float(np.percentile(ad_arr, 5)), 4),
            "cohens_d_95th_pct": round(float(np.percentile(ad_arr, 95)), 4),
            "cohens_d_gt_0.5_frac": round(float(np.mean(ad_arr > 0.5)), 4),
            "cohens_d_gt_0.5_count": int(np.sum(ad_arr > 0.5)),
            "cohens_d_gt_0.8_frac": round(float(np.mean(ad_arr > 0.8)), 4),
            "cohens_d_gt_0.8_count": int(np.sum(ad_arr > 0.8)),
            "cohens_d_gt_2.0_frac": round(float(np.mean(ad_arr > 2.0)), 4),
            "cohens_d_gt_2.0_count": int(np.sum(ad_arr > 2.0)),
            "cohens_d_lt_0_frac": round(float(np.mean(ad_arr < 0)), 4),
            "cohens_d_lt_0_count": int(np.sum(ad_arr < 0)),
            "real_mean": round(float(assign_real.mean()), 2),
            "null_mean": round(float(assign_null_means.mean()), 2),
            "abs_diff_mean": round(float(assign_abs_diffs.mean()), 2),
            "ratio_mean": round(float(valid_ratios_3.mean()), 4),
            "ratio_median": round(float(np.median(valid_ratios_3)), 4),
        },
        "runtime": {
            "analysis_1_2_seconds": round(elapsed_1, 1),
            "analysis_3_seconds": round(elapsed_3, 1),
        }
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")

    print(f"\n{'='*72}")
    print(f"  SUMMARY FOR MANUSCRIPT")
    print(f"{'='*72}")
    print(f"  Analysis 1 — Effect size: Random-position null ({len(valid_genes)} genes x {N_NULL_RANDOM} perms):")
    print(f"    Cohen's d: median={np.median(d_arr):+.3f}, mean={d_arr.mean():+.3f}")
    print(f"    IQR: [{np.percentile(d_arr, 25):+.3f}, {np.percentile(d_arr, 75):+.3f}]")
    print(f"    d > 0.8 (large effect): {np.mean(d_arr > 0.8):.1%}")
    print(f"    d < 0 (null beats real): {np.mean(d_arr < 0):.1%}")
    print(f"")
    print(f"  Analysis 2 — Disruption magnitudes:")
    print(f"    Real mean: {real_scores.mean():.1f}, Null mean: {null_means.mean():.1f}")
    print(f"    Ratio: {valid_ratios_12.mean():.3f}, Abs diff: {abs_diffs.mean():+.1f}")
    print(f"")
    print(f"  Analysis 3 — Token-assignment shuffle ({len(stratified)} genes x {N_NULL_ASSIGNMENT} perms):")
    print(f"    Cohen's d: median={np.median(ad_arr):+.3f}, mean={ad_arr.mean():+.3f}")
    print(f"    IQR: [{np.percentile(ad_arr, 25):+.3f}, {np.percentile(ad_arr, 75):+.3f}]")
    print(f"    d > 0.8 (large effect): {np.mean(ad_arr > 0.8):.1%}")
    print(f"    d < 0 (null beats real): {np.mean(ad_arr < 0):.1%}")
    print(f"    Real mean: {assign_real.mean():.1f}, Null mean: {assign_null_means.mean():.1f}")
    print(f"    Ratio: {valid_ratios_3.mean():.3f}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
