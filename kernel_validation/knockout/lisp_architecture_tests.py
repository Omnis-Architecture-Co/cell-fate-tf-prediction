#!/usr/bin/env python3
"""
Lisp Architecture Tests — probing the kernel as a computational system.
=======================================================================

Test 1: Primitive Consistency Across Calling Contexts
  For each of 116 primitives, find all proteins whose token-department sequence
  contains the primitive's departments. Knock out each protein, measure disruption
  to the primitive's own departments. If CV is low -> consistent function call.

Test 2: Subroutine Modularity
  Programs sharing a complete subroutine should have more similar disruption
  profiles than programs sharing the same departments but not as a contiguous block.

Test 3: Composition Order
  Programs with the same set of departments but different ordering should have
  different dispatch/disruption profiles if order matters.

Usage:
    python3 -u validation/knockout/lisp_architecture_tests.py
"""

import csv
import json
import os
import pickle
import random
import sys
import time
from collections import defaultdict, Counter

import numpy as np
from scipy import sparse

STATE_PATH = "/tmp/module8_full_state.pkl"
VOCAB_PATH = "server/data/human/vocabulary.csv"
PROTEIN_TOKENS_PATH = "server/data/human/protein_tokens_v2_with_genes.csv"
PRIMITIVES_PATH = "beta_transfer/genome_primitives.csv"
SUBROUTINES_PATH = "beta_transfer/genome_subroutines_ranked.csv"
PROGRAMS_PATH = "beta_transfer/genome_programs_all.csv"
NESTING_PATH = "beta_transfer/genome_nesting_hierarchy.csv"
GENE_DEPTS_PATH = "server/data/human/gene_departments.csv"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "lisp_architecture_results.json")

VALID_DEPARTMENTS = sorted([
    "Apoptosis", "Cell adhesion", "Cell cycle", "Chromatin", "Cytoskeleton",
    "DNA repair", "GTPase", "Immune", "Ion channel", "Kinase", "Methylation",
    "Nuc acid bind", "Phosphatase", "Protein folding", "Proteolysis",
    "RNA processing", "Signaling", "Structural", "Transcription",
    "Translation", "Transport", "Ubiquitin",
])
DEPT_TO_IDX = {d: i for i, d in enumerate(VALID_DEPARTMENTS)}
N_DEPTS = len(VALID_DEPARTMENTS)


def load_state():
    print("[1/6] Loading dispatch graph state...")
    t0 = time.time()
    with open(STATE_PATH, "rb") as f:
        state = pickle.load(f)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return state


def build_token_dept_map():
    vocab_dept = {}
    with open(VOCAB_PATH) as f:
        for row in csv.DictReader(f):
            stripped = row["word_hex"].replace("0x", "").upper()
            vocab_dept[stripped] = row["primary_function"]
    return vocab_dept


def build_protein_dept_sequences(state, vocab_dept):
    print("[2/6] Building protein department sequences...")
    t0 = time.time()
    ptt = state["ptt"]
    gene_cache = state["gene_cache"]

    protein_dept_seqs = {}
    protein_dept_lists = {}
    for uid, tokens in ptt.items():
        depts = []
        for tok in tokens:
            tok_upper = tok.upper()
            dept = vocab_dept.get(tok_upper)
            if dept and dept in DEPT_TO_IDX:
                depts.append(dept)
        if depts:
            compressed = []
            for d in depts:
                if not compressed or compressed[-1] != d:
                    compressed.append(d)
            protein_dept_seqs[uid] = "|".join(compressed)
            protein_dept_lists[uid] = compressed

    print(f"  {len(protein_dept_seqs)} proteins with department sequences ({time.time()-t0:.1f}s)")
    return protein_dept_seqs, protein_dept_lists


def build_dept_token_lookups(state):
    print("[3/6] Building per-department token disruption lookups...")
    t0 = time.time()
    ptt = state["ptt"]
    ttp = state["ttp"]

    gene_depts = {}
    with open(GENE_DEPTS_PATH) as f:
        for row in csv.DictReader(f):
            gene_depts[row["gene"]] = row["department"]

    gene_cache = state["gene_cache"]
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
    dept_token_disruption = np.zeros((N_DEPTS, n_t), dtype=np.float32)
    for di, dept in enumerate(VALID_DEPARTMENTS):
        mask = np.zeros(n_p, dtype=np.float32)
        for uid, idx in uid_to_idx.items():
            gene = gene_cache.get(uid)
            if gene:
                d = gene_depts.get(gene)
                if d == dept:
                    mask[idx] = 1.0
        dept_token_disruption[di] = np.asarray(mask @ P).flatten()

    print(f"  Matrix: {n_p}x{n_t}, lookup: {N_DEPTS}x{n_t} ({time.time()-t0:.1f}s)")
    return P, tok_to_idx, uid_to_idx, dept_token_disruption, n_t


def protein_dept_profile(uid, ptt, tok_to_idx, dept_token_disruption):
    tokens = ptt.get(uid, [])
    idxs = [tok_to_idx[t] for t in tokens if t in tok_to_idx]
    if not idxs:
        return np.zeros(N_DEPTS)
    idxs = np.array(idxs, dtype=np.int32)
    return dept_token_disruption[:, idxs].sum(axis=1)


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def test1_primitive_consistency(state, ptt, tok_to_idx, dept_token_disruption,
                                 protein_dept_lists):
    print("\n" + "=" * 72)
    print("  TEST 1: Primitive Consistency Across Calling Contexts")
    print("=" * 72)

    with open(PRIMITIVES_PATH) as f:
        primitives = list(csv.DictReader(f))
    print(f"  Loaded {len(primitives)} primitives")

    protein_seqs_compressed = {}
    for uid, dlist in protein_dept_lists.items():
        protein_seqs_compressed[uid] = "|".join(dlist)

    all_uids = list(ptt.keys())

    results = []
    t0 = time.time()

    for pi, prim in enumerate(primitives):
        prim_seq = prim["function_sequence"]
        prim_depts_raw = prim_seq.split("|")
        prim_depts = [d for d in prim_depts_raw if d in DEPT_TO_IDX]
        if not prim_depts:
            continue

        prim_dept_indices = [DEPT_TO_IDX[d] for d in prim_depts]
        unique_prim_depts = list(set(prim_depts))
        unique_dept_indices = [DEPT_TO_IDX[d] for d in unique_prim_depts]

        matching_uids = []
        prim_sub = "|".join(prim_depts)
        for uid, seq in protein_seqs_compressed.items():
            if prim_sub in seq:
                matching_uids.append(uid)

        if len(matching_uids) < 5:
            continue

        disruptions = np.zeros((len(matching_uids), len(unique_prim_depts)))
        for mi, uid in enumerate(matching_uids):
            profile = protein_dept_profile(uid, ptt, tok_to_idx, dept_token_disruption)
            for di, dept_idx in enumerate(unique_dept_indices):
                disruptions[mi, di] = profile[dept_idx]

        cvs = []
        for di in range(len(unique_prim_depts)):
            col = disruptions[:, di]
            mean_val = col.mean()
            std_val = col.std()
            cv = std_val / mean_val if mean_val > 0 else float("inf")
            cvs.append(cv)

        mean_cv = np.mean([c for c in cvs if np.isfinite(c)])

        null_cvs = []
        for _ in range(50):
            rand_uids = random.sample(all_uids, min(len(matching_uids), len(all_uids)))
            rand_disruptions = np.zeros((len(rand_uids), len(unique_prim_depts)))
            for mi, uid in enumerate(rand_uids):
                profile = protein_dept_profile(uid, ptt, tok_to_idx, dept_token_disruption)
                for di, dept_idx in enumerate(unique_dept_indices):
                    rand_disruptions[mi, di] = profile[dept_idx]
            rand_cv_list = []
            for di in range(len(unique_prim_depts)):
                col = rand_disruptions[:, di]
                m = col.mean()
                s = col.std()
                if m > 0:
                    rand_cv_list.append(s / m)
            if rand_cv_list:
                null_cvs.append(np.mean(rand_cv_list))

        null_mean_cv = np.mean(null_cvs) if null_cvs else float("inf")

        results.append({
            "primitive": prim_seq,
            "n_proteins": len(matching_uids),
            "departments": unique_prim_depts,
            "per_dept_cv": {d: round(cvs[i], 4) for i, d in enumerate(unique_prim_depts) if np.isfinite(cvs[i])},
            "mean_cv": round(float(mean_cv), 4) if np.isfinite(mean_cv) else None,
            "null_mean_cv": round(float(null_mean_cv), 4) if np.isfinite(null_mean_cv) else None,
            "cv_ratio": round(float(mean_cv / null_mean_cv), 4) if null_mean_cv > 0 and np.isfinite(mean_cv) else None,
        })

        if (pi + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{pi+1:3d}/{len(primitives)}] {prim_seq[:40]:<40s} "
                  f"N={len(matching_uids):4d} CV={mean_cv:.3f} null_CV={null_mean_cv:.3f} "
                  f"({elapsed:.0f}s)")
            sys.stdout.flush()

    valid_results = [r for r in results if r["mean_cv"] is not None]
    all_cvs = [r["mean_cv"] for r in valid_results]
    all_null_cvs = [r["null_mean_cv"] for r in valid_results if r["null_mean_cv"] is not None]
    all_ratios = [r["cv_ratio"] for r in valid_results if r["cv_ratio"] is not None]

    print(f"\n  === TEST 1 RESULTS ({len(valid_results)} primitives) ===")
    print(f"  Mean CV across primitives:       {np.mean(all_cvs):.4f}")
    print(f"  Median CV:                       {np.median(all_cvs):.4f}")
    print(f"  Null mean CV:                    {np.mean(all_null_cvs):.4f}")
    print(f"  CV ratio (real/null):            {np.mean(all_ratios):.4f}")
    print(f"  Fraction CV < 0.3 (consistent):  {np.mean(np.array(all_cvs) < 0.3):.1%}")
    print(f"  Fraction CV < 0.5:               {np.mean(np.array(all_cvs) < 0.5):.1%}")
    print(f"  Fraction CV < 1.0:               {np.mean(np.array(all_cvs) < 1.0):.1%}")
    print(f"  Real CV < Null CV:               {np.mean(np.array(all_ratios) < 1.0):.1%}")

    print(f"\n  Top 10 most consistent primitives:")
    sorted_results = sorted(valid_results, key=lambda r: r["mean_cv"])
    for r in sorted_results[:10]:
        print(f"    CV={r['mean_cv']:.3f} null={r['null_mean_cv']:.3f} "
              f"N={r['n_proteins']:4d} {r['primitive'][:50]}")
    print(f"  Bottom 10 (least consistent):")
    for r in sorted_results[-10:]:
        print(f"    CV={r['mean_cv']:.3f} null={r['null_mean_cv']:.3f} "
              f"N={r['n_proteins']:4d} {r['primitive'][:50]}")

    return {
        "n_primitives_tested": len(valid_results),
        "mean_cv": round(float(np.mean(all_cvs)), 4),
        "median_cv": round(float(np.median(all_cvs)), 4),
        "null_mean_cv": round(float(np.mean(all_null_cvs)), 4),
        "cv_ratio_mean": round(float(np.mean(all_ratios)), 4),
        "frac_cv_lt_0.3": round(float(np.mean(np.array(all_cvs) < 0.3)), 4),
        "frac_cv_lt_0.5": round(float(np.mean(np.array(all_cvs) < 0.5)), 4),
        "frac_cv_lt_1.0": round(float(np.mean(np.array(all_cvs) < 1.0)), 4),
        "frac_real_lt_null": round(float(np.mean(np.array(all_ratios) < 1.0)), 4),
        "per_primitive": results,
    }


def test2_subroutine_modularity(state, ptt, tok_to_idx, dept_token_disruption,
                                  protein_dept_seqs, protein_dept_lists):
    print("\n" + "=" * 72)
    print("  TEST 2: Subroutine Modularity")
    print("=" * 72)

    with open(SUBROUTINES_PATH) as f:
        subroutines = list(csv.DictReader(f))
    print(f"  Loaded {len(subroutines)} subroutines")

    subroutine_seqs = {}
    for sub in subroutines:
        seq = sub["function_sequence"]
        depts = [d for d in seq.split("|") if d in DEPT_TO_IDX]
        if len(depts) >= 2:
            subroutine_seqs["|".join(depts)] = {
                "raw": seq,
                "depts": depts,
                "dept_set": frozenset(depts),
            }

    print(f"  Valid subroutines (2+ known depts): {len(subroutine_seqs)}")

    uid_to_profile_cache = {}

    def get_profile(uid):
        if uid not in uid_to_profile_cache:
            uid_to_profile_cache[uid] = protein_dept_profile(
                uid, ptt, tok_to_idx, dept_token_disruption
            )
        return uid_to_profile_cache[uid]

    t0 = time.time()

    shared_sub_sims = []
    shared_dept_sims = []
    n_sub_pairs = 0
    n_dept_pairs = 0
    max_pairs_per_sub = 200

    tested_subs = 0
    for sub_seq, sub_info in subroutine_seqs.items():
        matching_uids = [uid for uid, seq in protein_dept_seqs.items()
                         if sub_seq in seq]
        if len(matching_uids) < 4:
            continue

        tested_subs += 1
        sample = matching_uids[:50]
        for i in range(len(sample)):
            for j in range(i + 1, min(i + 5, len(sample))):
                p1 = get_profile(sample[i])
                p2 = get_profile(sample[j])
                sim = cosine_sim(p1, p2)
                shared_sub_sims.append(sim)
                n_sub_pairs += 1

        dept_set = sub_info["dept_set"]
        dept_only_uids = []
        for uid, dlist in protein_dept_lists.items():
            if uid in set(matching_uids):
                continue
            if set(dlist) & dept_set == dept_set and sub_seq not in protein_dept_seqs.get(uid, ""):
                dept_only_uids.append(uid)
            if len(dept_only_uids) >= 50:
                break

        if len(dept_only_uids) >= 2:
            for i in range(min(len(dept_only_uids) - 1, 20)):
                j = i + 1
                p1 = get_profile(dept_only_uids[i])
                p2 = get_profile(dept_only_uids[j])
                sim = cosine_sim(p1, p2)
                shared_dept_sims.append(sim)
                n_dept_pairs += 1

        if tested_subs % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{tested_subs} subs] sub_pairs={n_sub_pairs} dept_pairs={n_dept_pairs} "
                  f"sub_sim={np.mean(shared_sub_sims):.3f} dept_sim={np.mean(shared_dept_sims) if shared_dept_sims else 0:.3f} "
                  f"({elapsed:.0f}s)")
            sys.stdout.flush()

    sub_arr = np.array(shared_sub_sims) if shared_sub_sims else np.array([0])
    dept_arr = np.array(shared_dept_sims) if shared_dept_sims else np.array([0])

    from scipy import stats
    if len(sub_arr) > 1 and len(dept_arr) > 1:
        t_stat, p_val = stats.mannwhitneyu(sub_arr, dept_arr, alternative="greater")
        pooled_std = np.sqrt((sub_arr.var() + dept_arr.var()) / 2)
        d = (sub_arr.mean() - dept_arr.mean()) / pooled_std if pooled_std > 0 else 0
    else:
        t_stat, p_val, d = 0, 1, 0

    print(f"\n  === TEST 2 RESULTS ===")
    print(f"  Subroutines tested:              {tested_subs}")
    print(f"  Shared-subroutine pairs:         {n_sub_pairs}")
    print(f"  Shared-department-only pairs:     {n_dept_pairs}")
    print(f"  Mean cosine sim (subroutine):    {sub_arr.mean():.4f}")
    print(f"  Mean cosine sim (dept-only):     {dept_arr.mean():.4f}")
    print(f"  Cohen's d:                       {d:+.4f}")
    print(f"  Mann-Whitney p:                  {p_val:.2e}")
    print(f"  Subroutine > dept-only:          {sub_arr.mean() > dept_arr.mean()}")

    return {
        "n_subroutines_tested": tested_subs,
        "n_sub_pairs": n_sub_pairs,
        "n_dept_pairs": n_dept_pairs,
        "sub_mean_cosine": round(float(sub_arr.mean()), 4),
        "dept_mean_cosine": round(float(dept_arr.mean()), 4),
        "cohens_d": round(float(d), 4),
        "mann_whitney_p": float(p_val),
        "subroutine_wins": bool(sub_arr.mean() > dept_arr.mean()),
    }


def test3_composition_order(state, ptt, tok_to_idx, dept_token_disruption,
                              protein_dept_lists):
    print("\n" + "=" * 72)
    print("  TEST 3: Composition Order")
    print("=" * 72)

    uid_to_dept_set = {}
    uid_to_dept_tuple = {}
    for uid, dlist in protein_dept_lists.items():
        uid_to_dept_set[uid] = frozenset(dlist)
        uid_to_dept_tuple[uid] = tuple(dlist)

    by_dept_set = defaultdict(list)
    for uid, ds in uid_to_dept_set.items():
        if len(ds) >= 2:
            by_dept_set[ds].append(uid)

    permuted_groups = {ds: uids for ds, uids in by_dept_set.items()
                       if len(uids) >= 4}

    print(f"  Groups with same dept set (>=4 proteins): {len(permuted_groups)}")

    t0 = time.time()

    same_order_sims = []
    diff_order_sims = []

    uid_to_profile_cache = {}
    def get_profile(uid):
        if uid not in uid_to_profile_cache:
            uid_to_profile_cache[uid] = protein_dept_profile(
                uid, ptt, tok_to_idx, dept_token_disruption
            )
        return uid_to_profile_cache[uid]

    groups_tested = 0
    for dept_set, uids in permuted_groups.items():
        by_order = defaultdict(list)
        for uid in uids:
            by_order[uid_to_dept_tuple[uid]].append(uid)

        orders_with_multiple = {order: uid_list for order, uid_list in by_order.items()
                                if len(uid_list) >= 2}
        distinct_orders = list(by_order.keys())

        if len(distinct_orders) < 2:
            continue

        groups_tested += 1
        sample_uids = uids[:40]

        for i in range(len(sample_uids)):
            for j in range(i + 1, min(i + 5, len(sample_uids))):
                u1, u2 = sample_uids[i], sample_uids[j]
                p1 = get_profile(u1)
                p2 = get_profile(u2)
                sim = cosine_sim(p1, p2)

                if uid_to_dept_tuple[u1] == uid_to_dept_tuple[u2]:
                    same_order_sims.append(sim)
                else:
                    diff_order_sims.append(sim)

        if groups_tested % 100 == 0:
            elapsed = time.time() - t0
            so = np.mean(same_order_sims) if same_order_sims else 0
            do = np.mean(diff_order_sims) if diff_order_sims else 0
            print(f"  [{groups_tested} groups] same_order={len(same_order_sims)} ({so:.3f}) "
                  f"diff_order={len(diff_order_sims)} ({do:.3f}) ({elapsed:.0f}s)")
            sys.stdout.flush()

    same_arr = np.array(same_order_sims) if same_order_sims else np.array([0])
    diff_arr = np.array(diff_order_sims) if diff_order_sims else np.array([0])

    from scipy import stats
    if len(same_arr) > 1 and len(diff_arr) > 1:
        t_stat, p_val = stats.mannwhitneyu(same_arr, diff_arr, alternative="greater")
        pooled_std = np.sqrt((same_arr.var() + diff_arr.var()) / 2)
        d = (same_arr.mean() - diff_arr.mean()) / pooled_std if pooled_std > 0 else 0
    else:
        t_stat, p_val, d = 0, 1, 0

    print(f"\n  === TEST 3 RESULTS ===")
    print(f"  Groups tested:                   {groups_tested}")
    print(f"  Same-order pairs:                {len(same_order_sims)}")
    print(f"  Different-order pairs:           {len(diff_order_sims)}")
    print(f"  Mean cosine sim (same order):    {same_arr.mean():.4f}")
    print(f"  Mean cosine sim (diff order):    {diff_arr.mean():.4f}")
    print(f"  Cohen's d:                       {d:+.4f}")
    print(f"  Mann-Whitney p:                  {p_val:.2e}")
    print(f"  Order matters (same > diff):     {same_arr.mean() > diff_arr.mean()}")

    return {
        "n_groups_tested": groups_tested,
        "n_same_order_pairs": len(same_order_sims),
        "n_diff_order_pairs": len(diff_order_sims),
        "same_order_mean_cosine": round(float(same_arr.mean()), 4),
        "diff_order_mean_cosine": round(float(diff_arr.mean()), 4),
        "cohens_d": round(float(d), 4),
        "mann_whitney_p": float(p_val),
        "order_matters": bool(same_arr.mean() > diff_arr.mean()),
    }


def main():
    state = load_state()
    vocab_dept = build_token_dept_map()
    protein_dept_seqs, protein_dept_lists = build_protein_dept_sequences(state, vocab_dept)
    P, tok_to_idx, uid_to_idx, dept_token_disruption, n_t = build_dept_token_lookups(state)
    ptt = state["ptt"]

    print(f"\n[4/6] Running Test 1: Primitive Consistency...")
    sys.stdout.flush()
    t1_results = test1_primitive_consistency(
        state, ptt, tok_to_idx, dept_token_disruption, protein_dept_lists
    )

    print(f"\n[5/6] Running Test 2: Subroutine Modularity...")
    sys.stdout.flush()
    t2_results = test2_subroutine_modularity(
        state, ptt, tok_to_idx, dept_token_disruption,
        protein_dept_seqs, protein_dept_lists
    )

    print(f"\n[6/6] Running Test 3: Composition Order...")
    sys.stdout.flush()
    t3_results = test3_composition_order(
        state, ptt, tok_to_idx, dept_token_disruption, protein_dept_lists
    )

    output = {
        "test1_primitive_consistency": t1_results,
        "test2_subroutine_modularity": t2_results,
        "test3_composition_order": t3_results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_PATH}")

    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")
    print(f"  Test 1 — Primitive Consistency:")
    print(f"    Mean CV: {t1_results['mean_cv']:.3f} (null: {t1_results['null_mean_cv']:.3f})")
    print(f"    CV < 0.3: {t1_results['frac_cv_lt_0.3']:.1%}")
    print(f"    Real < Null: {t1_results['frac_real_lt_null']:.1%}")
    print(f"  Test 2 — Subroutine Modularity:")
    print(f"    Shared sub cosine: {t2_results['sub_mean_cosine']:.3f}")
    print(f"    Shared dept cosine: {t2_results['dept_mean_cosine']:.3f}")
    print(f"    Cohen's d: {t2_results['cohens_d']:+.3f}, p={t2_results['mann_whitney_p']:.2e}")
    print(f"  Test 3 — Composition Order:")
    print(f"    Same order cosine: {t3_results['same_order_mean_cosine']:.3f}")
    print(f"    Diff order cosine: {t3_results['diff_order_mean_cosine']:.3f}")
    print(f"    Cohen's d: {t3_results['cohens_d']:+.3f}, p={t3_results['mann_whitney_p']:.2e}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
