#!/usr/bin/env python3
"""
Token-Assignment Shuffle: Departmental Profile Concordance
============================================================

For each gene in the stratified 500-gene subset:
  1. Compute the real 22-element departmental disruption profile
     (how much connectivity lost in each department when gene's tokens removed).
  2. For 100 token-assignment shuffles (gene gets another gene's tokens),
     compute the same 22-element profile.
  3. Compute cosine similarity between each profile and the gene's ground truth
     vector (1.0 in the gene's known department, 0.0 elsewhere).
  4. Report Cohen's d and z-score: real cosine sim vs distribution of shuffled.

Prediction: real token assignments produce disruption profiles pointing toward
the correct department more often than shuffled assignments.

Usage:
    python3 -u validation/knockout/dept_profile_shuffle.py
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
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "dept_profile_shuffle_results.json")

VALID_DEPARTMENTS = sorted([
    "Apoptosis", "Cell adhesion", "Cell cycle", "Chromatin", "Cytoskeleton",
    "DNA repair", "GTPase", "Immune", "Ion channel", "Kinase", "Methylation",
    "Nuc acid bind", "Phosphatase", "Protein folding", "Proteolysis",
    "RNA processing", "Signaling", "Structural", "Transcription",
    "Translation", "Transport", "Ubiquitin",
])

N_PERMS = 100
N_STRATIFIED = 500


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
    print("[1/4] Loading state...")
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


def build_dept_token_lookups(ttp, ptt, gene_cache, gene_depts):
    print("[2/4] Building sparse matrix and per-department token disruption lookups...")
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
    dept_token_disruption = np.zeros((len(VALID_DEPARTMENTS), n_t), dtype=np.float32)
    dept_sizes = {}

    for di, dept in enumerate(VALID_DEPARTMENTS):
        mask = np.zeros(n_p, dtype=np.float32)
        count = 0
        for uid, idx in uid_to_idx.items():
            gene = gene_cache.get(uid)
            if gene:
                d = gene_depts.get(gene)
                if d == dept:
                    mask[idx] = 1.0
                    count += 1
        dept_sizes[dept] = count
        dept_token_disruption[di] = np.asarray(mask @ P).flatten()

    total_indexed = sum(dept_sizes.values())
    print(f"  Matrix: {n_p}x{n_t}, {P.nnz} edges")
    print(f"  Department token lookups: {len(VALID_DEPARTMENTS)} depts x {n_t} tokens")
    print(f"  Total indexed proteins: {total_indexed}")
    print(f"  Built in {time.time()-t0:.1f}s")
    return P, tok_to_idx, uid_to_idx, dept_token_disruption, dept_sizes, n_p, n_t


def cosine_sim(a, b):
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def dept_profile_from_tokens(token_indices, dept_token_disruption):
    return dept_token_disruption[:, token_indices].sum(axis=1)


def select_stratified_sample(manifest, gene_tokens, gene_depts, n=500, seed=42):
    rng = random.Random(seed)
    genes = manifest["genes"]

    candidates = [g for g in sorted(genes.keys())
                  if g in gene_tokens and g in gene_depts
                  and gene_depts[g] in set(VALID_DEPARTMENTS)]

    by_tokens = defaultdict(list)
    for g in candidates:
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

    remaining_pool = [g for g in candidates if g not in set(selected)]
    while len(selected) < n and remaining_pool:
        selected.append(rng.choice(remaining_pool))
        remaining_pool = [g for g in remaining_pool if g not in set(selected)]

    return selected[:n]


def build_gene_token_pools(P, uid_to_idx, manifest):
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
    P, tok_to_idx, uid_to_idx, dept_token_disruption, dept_sizes, n_p, n_t = \
        build_dept_token_lookups(ttp, ptt, gene_cache, gene_depts)

    gene_tokens = build_gene_token_pools(P, uid_to_idx, manifest)
    gene_list = list(gene_tokens.keys())

    stratified = select_stratified_sample(manifest, gene_tokens, gene_depts, N_STRATIFIED)
    print(f"\n[3/4] Running departmental profile shuffle ({len(stratified)} genes x {N_PERMS} perms)...")

    dept_to_idx = {d: i for i, d in enumerate(VALID_DEPARTMENTS)}

    rng = random.Random(42)
    t0 = time.time()

    results_per_gene = []
    all_d_values = []
    all_z_values = []
    real_wins = 0
    real_top1_match = 0
    null_top1_match_counts = []

    for gi, gene in enumerate(stratified):
        toks = gene_tokens[gene]
        k = len(toks)
        true_dept = gene_depts[gene]
        true_dept_idx = dept_to_idx[true_dept]

        gt_vec = np.zeros(len(VALID_DEPARTMENTS), dtype=np.float32)
        gt_vec[true_dept_idx] = 1.0

        real_profile = dept_profile_from_tokens(toks, dept_token_disruption)
        real_cos = cosine_sim(real_profile, gt_vec)

        real_top_dept = VALID_DEPARTMENTS[np.argmax(real_profile)]
        if real_top_dept == true_dept:
            real_top1_match += 1

        null_cos_vals = np.zeros(N_PERMS, dtype=np.float64)
        null_top1_matches_this_gene = 0

        for pi in range(N_PERMS):
            donor = rng.choice(gene_list)
            while donor == gene:
                donor = rng.choice(gene_list)
            donor_toks = gene_tokens[donor]
            if len(donor_toks) >= k:
                sampled = np.array(rng.sample(list(donor_toks), k), dtype=np.int32)
            else:
                pool = list(donor_toks)
                while len(pool) < k:
                    d2 = rng.choice(gene_list)
                    pool.extend(list(gene_tokens[d2]))
                sampled = np.array(rng.sample(pool, k), dtype=np.int32)

            null_profile = dept_profile_from_tokens(sampled, dept_token_disruption)
            null_cos_vals[pi] = cosine_sim(null_profile, gt_vec)

            null_top_dept = VALID_DEPARTMENTS[np.argmax(null_profile)]
            if null_top_dept == true_dept:
                null_top1_matches_this_gene += 1

        null_top1_match_counts.append(null_top1_matches_this_gene / N_PERMS)

        nm = float(null_cos_vals.mean())
        ns = float(null_cos_vals.std())

        z = (real_cos - nm) / ns if ns > 0 else 0
        d = (real_cos - nm) / ns if ns > 0 else 0

        if real_cos > nm:
            real_wins += 1

        all_d_values.append(d)
        all_z_values.append(z)

        results_per_gene.append({
            "gene": gene,
            "department": true_dept,
            "n_tokens": int(k),
            "real_cosine_sim": round(real_cos, 6),
            "null_cosine_mean": round(nm, 6),
            "null_cosine_std": round(ns, 6),
            "cohens_d": round(d, 4),
            "z_score": round(z, 4),
            "real_top_dept": real_top_dept,
            "real_top1_correct": real_top_dept == true_dept,
            "null_top1_rate": round(null_top1_matches_this_gene / N_PERMS, 4),
        })

        if (gi + 1) % 100 == 0 or gi == len(stratified) - 1:
            elapsed = time.time() - t0
            rate = (gi + 1) / elapsed if elapsed > 0 else 1
            eta = (len(stratified) - gi - 1) / rate
            da = np.array(all_d_values)
            win_rate = real_wins / (gi + 1)
            top1 = real_top1_match / (gi + 1)
            null_top1_avg = np.mean(null_top1_match_counts)
            print(f"  [{gi+1:3d}/{len(stratified)}] "
                  f"median_d={np.median(da):+.3f} "
                  f"d>0.5={np.mean(da>0.5):.1%} "
                  f"real_wins={win_rate:.1%} "
                  f"real_top1={top1:.1%} "
                  f"null_top1={null_top1_avg:.1%} "
                  f"({elapsed:.0f}s)")
            sys.stdout.flush()

    elapsed_total = time.time() - t0

    d_arr = np.array(all_d_values)
    z_arr = np.array(all_z_values)

    print(f"\n  Complete in {elapsed_total:.1f}s")

    print(f"\n{'='*72}")
    print(f"  DEPARTMENTAL PROFILE CONCORDANCE — TOKEN-ASSIGNMENT SHUFFLE")
    print(f"  {len(stratified)} genes x {N_PERMS} permutations")
    print(f"{'='*72}")

    print(f"\n  Cohen's d (real cosine sim vs shuffled cosine sim):")
    print(f"    Mean:   {d_arr.mean():+.4f}")
    print(f"    Median: {np.median(d_arr):+.4f}")
    print(f"    IQR:    [{np.percentile(d_arr, 25):+.4f}, {np.percentile(d_arr, 75):+.4f}]")
    print(f"    5th:    {np.percentile(d_arr, 5):+.4f}")
    print(f"    95th:   {np.percentile(d_arr, 95):+.4f}")
    print(f"    d > 0.5 (medium): {np.mean(d_arr > 0.5):.1%} ({int(np.sum(d_arr > 0.5))}/{len(d_arr)})")
    print(f"    d > 0.8 (large):  {np.mean(d_arr > 0.8):.1%} ({int(np.sum(d_arr > 0.8))}/{len(d_arr)})")
    print(f"    d > 2.0:          {np.mean(d_arr > 2.0):.1%} ({int(np.sum(d_arr > 2.0))}/{len(d_arr)})")
    print(f"    d < 0:            {np.mean(d_arr < 0):.1%} ({int(np.sum(d_arr < 0))}/{len(d_arr)})")

    real_cos_vals = [r["real_cosine_sim"] for r in results_per_gene]
    null_cos_vals_all = [r["null_cosine_mean"] for r in results_per_gene]
    print(f"\n  Cosine similarity distributions:")
    print(f"    Real mean:  {np.mean(real_cos_vals):.6f}")
    print(f"    Null mean:  {np.mean(null_cos_vals_all):.6f}")
    print(f"    Real > Null: {real_wins}/{len(stratified)} ({real_wins/len(stratified):.1%})")

    print(f"\n  Top-1 department accuracy:")
    print(f"    Real tokens:    {real_top1_match}/{len(stratified)} ({real_top1_match/len(stratified):.1%})")
    null_top1_avg = np.mean(null_top1_match_counts)
    print(f"    Shuffled tokens (avg): {null_top1_avg:.1%}")
    print(f"    Chance (1/22):  {1/22:.1%}")

    by_dept = defaultdict(list)
    for r in results_per_gene:
        by_dept[r["department"]].append(r)
    print(f"\n  Per-department breakdown (departments with 5+ genes):")
    print(f"  {'Department':<20s} {'N':>3s} {'Med d':>7s} {'Top1%':>6s} {'NullT1%':>8s}")
    for dept in VALID_DEPARTMENTS:
        genes_in = by_dept.get(dept, [])
        if len(genes_in) >= 5:
            dept_d = [r["cohens_d"] for r in genes_in]
            dept_top1 = sum(1 for r in genes_in if r["real_top1_correct"]) / len(genes_in)
            dept_null_top1 = np.mean([r["null_top1_rate"] for r in genes_in])
            print(f"  {dept:<20s} {len(genes_in):3d} {np.median(dept_d):+7.3f} {dept_top1:5.1%} {dept_null_top1:7.1%}")

    print(f"\n  Top 10 genes by Cohen's d:")
    sorted_by_d = sorted(results_per_gene, key=lambda r: r["cohens_d"], reverse=True)
    for r in sorted_by_d[:10]:
        print(f"    {r['gene']:<15s} dept={r['department']:<18s} d={r['cohens_d']:+7.3f} "
              f"real_cos={r['real_cosine_sim']:.4f} null_cos={r['null_cosine_mean']:.4f} "
              f"top1={'Y' if r['real_top1_correct'] else 'N'}")

    print(f"\n  Bottom 10 genes by Cohen's d:")
    for r in sorted_by_d[-10:]:
        print(f"    {r['gene']:<15s} dept={r['department']:<18s} d={r['cohens_d']:+7.3f} "
              f"real_cos={r['real_cosine_sim']:.4f} null_cos={r['null_cosine_mean']:.4f} "
              f"top1={'Y' if r['real_top1_correct'] else 'N'}")

    print(f"\n{'='*72}")

    output = {
        "description": "Token-assignment shuffle measuring departmental disruption profile concordance",
        "n_genes": len(stratified),
        "n_perms": N_PERMS,
        "n_departments": len(VALID_DEPARTMENTS),
        "cohens_d": {
            "mean": round(float(d_arr.mean()), 4),
            "median": round(float(np.median(d_arr)), 4),
            "iqr": [round(float(np.percentile(d_arr, 25)), 4),
                    round(float(np.percentile(d_arr, 75)), 4)],
            "pct_5": round(float(np.percentile(d_arr, 5)), 4),
            "pct_95": round(float(np.percentile(d_arr, 95)), 4),
            "gt_0.5_frac": round(float(np.mean(d_arr > 0.5)), 4),
            "gt_0.8_frac": round(float(np.mean(d_arr > 0.8)), 4),
            "gt_2.0_frac": round(float(np.mean(d_arr > 2.0)), 4),
            "lt_0_frac": round(float(np.mean(d_arr < 0)), 4),
        },
        "cosine_similarity": {
            "real_mean": round(float(np.mean(real_cos_vals)), 6),
            "null_mean": round(float(np.mean(null_cos_vals_all)), 6),
            "real_wins_frac": round(real_wins / len(stratified), 4),
        },
        "top1_accuracy": {
            "real": round(real_top1_match / len(stratified), 4),
            "null_avg": round(float(null_top1_avg), 4),
            "chance": round(1 / 22, 4),
        },
        "runtime_seconds": round(elapsed_total, 1),
        "per_gene_results": results_per_gene,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
