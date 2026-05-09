#!/usr/bin/env python3
"""
Lisp Nesting Tests — testing the ACTUAL discovered nesting hierarchy.
=====================================================================

Uses the 2,795 explicit nesting relationships from genome_nesting_hierarchy.csv
to test whether the kernel's hierarchical program structure behaves like
nested function calls.

Test A: Nesting Inheritance
  If outer program nests inner program, do proteins carrying the outer's
  full function sequence show elevated disruption in the inner's departments?
  Compare to proteins that do NOT carry the outer sequence.

Test B: Shared-Child Similarity
  Two outer programs that share the same nested inner program — are their
  carrier proteins more similar (in the inner's departments) than outer
  programs that don't share a child?

Test C: Nesting Asymmetry
  Outer→inner disruption should be stronger than inner→outer.
  Calling the outer invokes the inner, but not vice versa.

Usage:
    python3 -u validation/knockout/lisp_nesting_tests.py
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
from scipy import sparse, stats

STATE_PATH = "/tmp/module8_full_state.pkl"
VOCAB_PATH = "server/data/human/vocabulary.csv"
PROTEIN_TOKENS_PATH = "server/data/human/protein_tokens_v2_with_genes.csv"
NESTING_PATH = "beta_transfer/genome_nesting_hierarchy.csv"
GENE_DEPTS_PATH = "server/data/human/gene_departments.csv"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "lisp_nesting_results.json")

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
    print("[1/5] Loading dispatch graph state...")
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


def build_protein_data(state, vocab_dept):
    print("[2/5] Building protein department sequences...")
    t0 = time.time()
    ptt = state["ptt"]

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

    print(f"  {len(protein_dept_seqs)} proteins with dept sequences ({time.time()-t0:.1f}s)")
    return protein_dept_seqs, protein_dept_lists


def build_disruption_infrastructure(state):
    print("[3/5] Building disruption lookup matrices...")
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

    print(f"  Built {n_p}×{n_t} matrix ({time.time()-t0:.1f}s)")
    return tok_to_idx, uid_to_idx, dept_token_disruption


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


def load_nesting():
    print("[4/5] Loading nesting hierarchy...")
    with open(NESTING_PATH) as f:
        nesting = list(csv.DictReader(f))
    print(f"  {len(nesting)} nesting relationships")

    by_outer = defaultdict(list)
    by_inner = defaultdict(list)
    for n in nesting:
        by_outer[n["outer_sequence"]].append(n)
        by_inner[n["inner_sequence"]].append(n)

    return nesting, by_outer, by_inner


def find_carriers(func_seq, protein_dept_seqs):
    depts = [d for d in func_seq.split("|") if d in DEPT_TO_IDX]
    if not depts:
        return []
    search = "|".join(depts)
    return [uid for uid, seq in protein_dept_seqs.items() if search in seq]


def extract_inner_dept_indices(inner_seq):
    depts = [d for d in inner_seq.split("|") if d in DEPT_TO_IDX]
    unique = list(dict.fromkeys(depts))
    return [DEPT_TO_IDX[d] for d in unique], unique


def test_a_nesting_inheritance(nesting, protein_dept_seqs, ptt, tok_to_idx,
                                 dept_token_disruption):
    print("\n" + "=" * 72)
    print("  TEST A: Nesting Inheritance")
    print("  Does the outer program invoke the inner program's departments?")
    print("=" * 72)

    all_uids = list(ptt.keys())
    t0 = time.time()

    tested = 0
    outer_disrupts_inner = []
    null_disrupts_inner = []
    per_relation = []

    seen_pairs = set()
    for n in nesting:
        outer_seq = n["outer_sequence"]
        inner_seq = n["inner_sequence"]
        pair_key = (outer_seq, inner_seq)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        inner_dept_idxs, inner_dept_names = extract_inner_dept_indices(inner_seq)
        if not inner_dept_idxs:
            continue

        outer_carriers = find_carriers(outer_seq, protein_dept_seqs)
        if len(outer_carriers) < 3:
            continue

        sample_outer = outer_carriers[:100]
        real_scores = []
        for uid in sample_outer:
            profile = protein_dept_profile(uid, ptt, tok_to_idx, dept_token_disruption)
            inner_disruption = sum(profile[di] for di in inner_dept_idxs)
            total_disruption = profile.sum()
            frac = inner_disruption / total_disruption if total_disruption > 0 else 0
            real_scores.append(frac)

        null_scores = []
        for _ in range(min(50, len(all_uids))):
            rand_uids = random.sample(all_uids, min(len(sample_outer), len(all_uids)))
            for uid in rand_uids[:len(sample_outer)]:
                profile = protein_dept_profile(uid, ptt, tok_to_idx, dept_token_disruption)
                inner_disruption = sum(profile[di] for di in inner_dept_idxs)
                total_disruption = profile.sum()
                frac = inner_disruption / total_disruption if total_disruption > 0 else 0
                null_scores.append(frac)
            break

        real_mean = np.mean(real_scores)
        null_mean = np.mean(null_scores) if null_scores else 0

        outer_disrupts_inner.append(real_mean)
        null_disrupts_inner.append(null_mean)

        per_relation.append({
            "outer": outer_seq[:60],
            "inner": inner_seq[:60],
            "outer_layer": n["outer_layer"],
            "inner_layer": n["inner_layer"],
            "n_carriers": len(outer_carriers),
            "inner_depts": inner_dept_names,
            "real_inner_frac": round(float(real_mean), 4),
            "null_inner_frac": round(float(null_mean), 4),
            "enrichment": round(float(real_mean / null_mean), 4) if null_mean > 0 else None,
        })

        tested += 1
        if tested % 50 == 0:
            elapsed = time.time() - t0
            real_arr = np.array(outer_disrupts_inner)
            null_arr = np.array(null_disrupts_inner)
            print(f"  [{tested} relations] real_frac={real_arr.mean():.3f} "
                  f"null_frac={null_arr.mean():.3f} "
                  f"real>null={np.mean(real_arr > null_arr):.1%} ({elapsed:.0f}s)")
            sys.stdout.flush()

    real_arr = np.array(outer_disrupts_inner)
    null_arr = np.array(null_disrupts_inner)

    if len(real_arr) > 1 and len(null_arr) > 1:
        t_stat, p_val = stats.wilcoxon(real_arr - null_arr, alternative="greater")
        pooled_std = np.sqrt((real_arr.var() + null_arr.var()) / 2)
        d = (real_arr.mean() - null_arr.mean()) / pooled_std if pooled_std > 0 else 0
    else:
        t_stat, p_val, d = 0, 1, 0

    enrichments = [r["enrichment"] for r in per_relation if r["enrichment"] is not None]

    print(f"\n  === TEST A RESULTS ({tested} nesting relations) ===")
    print(f"  Mean inner-dept fraction (outer carriers): {real_arr.mean():.4f}")
    print(f"  Mean inner-dept fraction (null):           {null_arr.mean():.4f}")
    print(f"  Outer > Null:                              {np.mean(real_arr > null_arr):.1%}")
    print(f"  Mean enrichment:                           {np.mean(enrichments):.2f}×")
    print(f"  Median enrichment:                         {np.median(enrichments):.2f}×")
    print(f"  Cohen's d:                                 {d:+.4f}")
    print(f"  Wilcoxon p:                                {p_val:.2e}")

    return {
        "n_relations_tested": tested,
        "real_inner_frac_mean": round(float(real_arr.mean()), 4),
        "null_inner_frac_mean": round(float(null_arr.mean()), 4),
        "frac_real_gt_null": round(float(np.mean(real_arr > null_arr)), 4),
        "enrichment_mean": round(float(np.mean(enrichments)), 4),
        "enrichment_median": round(float(np.median(enrichments)), 4),
        "cohens_d": round(float(d), 4),
        "wilcoxon_p": float(p_val),
        "per_relation_sample": per_relation[:50],
    }


def test_b_shared_child_similarity(nesting, by_inner, protein_dept_seqs, ptt,
                                     tok_to_idx, dept_token_disruption):
    print("\n" + "=" * 72)
    print("  TEST B: Shared-Child Similarity")
    print("  Outer programs sharing a child → more similar in child's depts?")
    print("=" * 72)

    t0 = time.time()

    shared_child_sims = []
    no_shared_child_sims = []

    uid_profile_cache = {}
    def get_profile(uid):
        if uid not in uid_profile_cache:
            uid_profile_cache[uid] = protein_dept_profile(
                uid, ptt, tok_to_idx, dept_token_disruption
            )
        return uid_profile_cache[uid]

    inner_seqs_with_multiple_parents = {}
    for inner_seq, relations in by_inner.items():
        parent_seqs = list(set(r["outer_sequence"] for r in relations))
        if len(parent_seqs) >= 2:
            inner_seqs_with_multiple_parents[inner_seq] = parent_seqs

    print(f"  Inner programs with 2+ parents: {len(inner_seqs_with_multiple_parents)}")

    tested_children = 0
    for inner_seq, parent_seqs in inner_seqs_with_multiple_parents.items():
        inner_dept_idxs, inner_dept_names = extract_inner_dept_indices(inner_seq)
        if not inner_dept_idxs:
            continue

        parent_carrier_groups = {}
        for ps in parent_seqs:
            carriers = find_carriers(ps, protein_dept_seqs)
            if len(carriers) >= 2:
                parent_carrier_groups[ps] = carriers[:30]

        if len(parent_carrier_groups) < 2:
            continue

        tested_children += 1
        parent_keys = list(parent_carrier_groups.keys())

        for i in range(len(parent_keys)):
            for j in range(i + 1, len(parent_keys)):
                g1 = parent_carrier_groups[parent_keys[i]]
                g2 = parent_carrier_groups[parent_keys[j]]
                for u1 in g1[:10]:
                    for u2 in g2[:10]:
                        p1 = get_profile(u1)
                        p2 = get_profile(u2)
                        p1_inner = p1[inner_dept_idxs]
                        p2_inner = p2[inner_dept_idxs]
                        sim = cosine_sim(p1_inner, p2_inner)
                        shared_child_sims.append(sim)

        all_parent_carriers = []
        for cs in parent_carrier_groups.values():
            all_parent_carriers.extend(cs)

        non_carriers = [uid for uid in list(ptt.keys())[:5000]
                        if uid not in set(all_parent_carriers)]

        if len(non_carriers) >= 20:
            sample_nc = random.sample(non_carriers, min(20, len(non_carriers)))
            for i in range(len(sample_nc)):
                for j in range(i + 1, min(i + 5, len(sample_nc))):
                    p1 = get_profile(sample_nc[i])
                    p2 = get_profile(sample_nc[j])
                    p1_inner = p1[inner_dept_idxs]
                    p2_inner = p2[inner_dept_idxs]
                    sim = cosine_sim(p1_inner, p2_inner)
                    no_shared_child_sims.append(sim)

        if tested_children % 20 == 0:
            elapsed = time.time() - t0
            sc = np.mean(shared_child_sims) if shared_child_sims else 0
            ns = np.mean(no_shared_child_sims) if no_shared_child_sims else 0
            print(f"  [{tested_children} children] shared={len(shared_child_sims)} ({sc:.3f}) "
                  f"null={len(no_shared_child_sims)} ({ns:.3f}) ({elapsed:.0f}s)")
            sys.stdout.flush()

    sc_arr = np.array(shared_child_sims) if shared_child_sims else np.array([0])
    ns_arr = np.array(no_shared_child_sims) if no_shared_child_sims else np.array([0])

    if len(sc_arr) > 1 and len(ns_arr) > 1:
        u_stat, p_val = stats.mannwhitneyu(sc_arr, ns_arr, alternative="greater")
        pooled_std = np.sqrt((sc_arr.var() + ns_arr.var()) / 2)
        d = (sc_arr.mean() - ns_arr.mean()) / pooled_std if pooled_std > 0 else 0
    else:
        u_stat, p_val, d = 0, 1, 0

    print(f"\n  === TEST B RESULTS ===")
    print(f"  Inner programs tested:           {tested_children}")
    print(f"  Shared-child pairs:              {len(shared_child_sims)}")
    print(f"  No-shared-child pairs:           {len(no_shared_child_sims)}")
    print(f"  Cosine sim (shared child):       {sc_arr.mean():.4f}")
    print(f"  Cosine sim (no shared child):    {ns_arr.mean():.4f}")
    print(f"  Cohen's d:                       {d:+.4f}")
    print(f"  Mann-Whitney p:                  {p_val:.2e}")

    return {
        "n_inner_tested": tested_children,
        "n_shared_pairs": len(shared_child_sims),
        "n_null_pairs": len(no_shared_child_sims),
        "shared_child_cosine": round(float(sc_arr.mean()), 4),
        "no_shared_child_cosine": round(float(ns_arr.mean()), 4),
        "cohens_d": round(float(d), 4),
        "mann_whitney_p": float(p_val),
    }


def test_c_nesting_asymmetry(nesting, protein_dept_seqs, ptt, tok_to_idx,
                                dept_token_disruption):
    print("\n" + "=" * 72)
    print("  TEST C: Nesting Asymmetry")
    print("  Outer→inner disruption stronger than inner→outer?")
    print("=" * 72)

    t0 = time.time()

    outer_to_inner_fracs = []
    inner_to_outer_fracs = []
    per_relation = []

    seen_pairs = set()
    tested = 0

    for n in nesting:
        outer_seq = n["outer_sequence"]
        inner_seq = n["inner_sequence"]
        pair_key = (outer_seq, inner_seq)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        outer_dept_raw = [d for d in outer_seq.split("|") if d in DEPT_TO_IDX]
        inner_dept_raw = [d for d in inner_seq.split("|") if d in DEPT_TO_IDX]
        if not outer_dept_raw or not inner_dept_raw:
            continue

        outer_dept_idxs = [DEPT_TO_IDX[d] for d in dict.fromkeys(outer_dept_raw)]
        inner_dept_idxs = [DEPT_TO_IDX[d] for d in dict.fromkeys(inner_dept_raw)]

        outer_carriers = find_carriers(outer_seq, protein_dept_seqs)
        inner_carriers = find_carriers(inner_seq, protein_dept_seqs)

        if len(outer_carriers) < 3 or len(inner_carriers) < 3:
            continue

        o2i_scores = []
        for uid in outer_carriers[:50]:
            profile = protein_dept_profile(uid, ptt, tok_to_idx, dept_token_disruption)
            inner_d = sum(profile[di] for di in inner_dept_idxs)
            total = profile.sum()
            o2i_scores.append(inner_d / total if total > 0 else 0)

        i2o_scores = []
        for uid in inner_carriers[:50]:
            profile = protein_dept_profile(uid, ptt, tok_to_idx, dept_token_disruption)
            outer_d = sum(profile[di] for di in outer_dept_idxs)
            total = profile.sum()
            i2o_scores.append(outer_d / total if total > 0 else 0)

        o2i_mean = np.mean(o2i_scores)
        i2o_mean = np.mean(i2o_scores)

        outer_to_inner_fracs.append(o2i_mean)
        inner_to_outer_fracs.append(i2o_mean)

        per_relation.append({
            "outer": outer_seq[:60],
            "inner": inner_seq[:60],
            "n_outer_carriers": len(outer_carriers),
            "n_inner_carriers": len(inner_carriers),
            "outer_to_inner_frac": round(float(o2i_mean), 4),
            "inner_to_outer_frac": round(float(i2o_mean), 4),
            "asymmetry": round(float(o2i_mean - i2o_mean), 4),
        })

        tested += 1
        if tested % 50 == 0:
            elapsed = time.time() - t0
            o2i = np.mean(outer_to_inner_fracs)
            i2o = np.mean(inner_to_outer_fracs)
            asym = np.mean(np.array(outer_to_inner_fracs) > np.array(inner_to_outer_fracs))
            print(f"  [{tested} pairs] O→I={o2i:.3f} I→O={i2o:.3f} O>I={asym:.1%} ({elapsed:.0f}s)")
            sys.stdout.flush()

    o2i_arr = np.array(outer_to_inner_fracs)
    i2o_arr = np.array(inner_to_outer_fracs)

    if len(o2i_arr) > 1:
        t_stat, p_val = stats.wilcoxon(o2i_arr - i2o_arr)
        pooled_std = np.sqrt((o2i_arr.var() + i2o_arr.var()) / 2)
        d = (o2i_arr.mean() - i2o_arr.mean()) / pooled_std if pooled_std > 0 else 0

        frac_asymmetric = float(np.mean(o2i_arr > i2o_arr))
    else:
        t_stat, p_val, d = 0, 1, 0
        frac_asymmetric = 0

    print(f"\n  === TEST C RESULTS ({tested} nesting pairs) ===")
    print(f"  Mean outer→inner fraction:       {o2i_arr.mean():.4f}")
    print(f"  Mean inner→outer fraction:       {i2o_arr.mean():.4f}")
    print(f"  Outer→inner > Inner→outer:       {frac_asymmetric:.1%}")
    print(f"  Cohen's d:                       {d:+.4f}")
    print(f"  Wilcoxon p:                      {p_val:.2e}")

    if frac_asymmetric > 0.5:
        print(f"  → ASYMMETRIC: outer programs disrupt inner depts more than reverse")
    elif frac_asymmetric < 0.5:
        print(f"  → REVERSE: inner programs disrupt outer depts more (unexpected)")
    else:
        print(f"  → SYMMETRIC: no directional preference")

    return {
        "n_pairs_tested": tested,
        "outer_to_inner_mean": round(float(o2i_arr.mean()), 4),
        "inner_to_outer_mean": round(float(i2o_arr.mean()), 4),
        "frac_outer_gt_inner": round(float(frac_asymmetric), 4),
        "cohens_d": round(float(d), 4),
        "wilcoxon_p": float(p_val),
        "per_relation_sample": per_relation[:50],
    }


def main():
    state = load_state()
    vocab_dept = build_token_dept_map()
    protein_dept_seqs, protein_dept_lists = build_protein_data(state, vocab_dept)
    tok_to_idx, uid_to_idx, dept_token_disruption = build_disruption_infrastructure(state)
    ptt = state["ptt"]

    nesting, by_outer, by_inner = load_nesting()

    print(f"\n[5/5] Running nesting tests...")
    sys.stdout.flush()

    ta = test_a_nesting_inheritance(
        nesting, protein_dept_seqs, ptt, tok_to_idx, dept_token_disruption
    )

    tb = test_b_shared_child_similarity(
        nesting, by_inner, protein_dept_seqs, ptt, tok_to_idx, dept_token_disruption
    )

    tc = test_c_nesting_asymmetry(
        nesting, protein_dept_seqs, ptt, tok_to_idx, dept_token_disruption
    )

    output = {
        "test_a_nesting_inheritance": ta,
        "test_b_shared_child_similarity": tb,
        "test_c_nesting_asymmetry": tc,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_PATH}")

    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")
    print(f"  Test A — Nesting Inheritance:")
    print(f"    Outer carriers disrupt inner depts: {ta['real_inner_frac_mean']:.3f}")
    print(f"    Null:                               {ta['null_inner_frac_mean']:.3f}")
    print(f"    Enrichment:                         {ta['enrichment_mean']:.2f}×")
    print(f"    d={ta['cohens_d']:+.3f}, p={ta['wilcoxon_p']:.2e}")
    print(f"  Test B — Shared-Child Similarity:")
    print(f"    Shared child cosine:                {tb['shared_child_cosine']:.3f}")
    print(f"    No shared child cosine:             {tb['no_shared_child_cosine']:.3f}")
    print(f"    d={tb['cohens_d']:+.3f}, p={tb['mann_whitney_p']:.2e}")
    print(f"  Test C — Nesting Asymmetry:")
    print(f"    Outer→Inner:                        {tc['outer_to_inner_mean']:.3f}")
    print(f"    Inner→Outer:                        {tc['inner_to_outer_mean']:.3f}")
    print(f"    Outer > Inner:                      {tc['frac_outer_gt_inner']:.1%}")
    print(f"    d={tc['cohens_d']:+.3f}, p={tc['wilcoxon_p']:.2e}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
