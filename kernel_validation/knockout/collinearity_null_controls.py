#!/usr/bin/env python3
"""
Collinearity Null Controls — three tests to rule out alternative explanations.
==============================================================================

Control 1 (CRITICAL): Graph rewiring
  - Degree-preserving edge swaps on the dispatch graph
  - Recompute disruption profiles on rewired graph
  - Test whether collinearity survives (algebra is generative vs topological)

Control 2: Department-matched groups
  - For each primitive, find genes sharing the same department but NOT the primitive
  - Test whether department-matched groups show equal collinearity
  - If yes: primitive isn't adding anything beyond functional annotation

Control 3: Token neighborhood overlap
  - Compute Jaccard overlap of token sets for within-primitive vs across-primitive
  - If within-primitive pairs have higher Jaccard, regress it out and retest
  - Tests whether collinearity is just "shared tokens → shared profiles"

Usage:
    python3 -u validation/knockout/collinearity_null_controls.py
"""

import csv
import json
import os
import pickle
import sys
import time
import numpy as np
from collections import defaultdict, Counter
from scipy import stats

STATE_PATH = "/tmp/module8_full_state.pkl"
VOCAB_PATH = "server/data/human/vocabulary.csv"
GENE_DEPTS_PATH = "server/data/human/gene_departments.csv"
PRIMITIVES_PATH = "beta_transfer/genome_primitives.csv"
PROFILES_PATH = "validation/knockout/disruption_profiles.json"
OUTPUT_PATH = "validation/knockout/collinearity_null_controls.json"

VALID_DEPARTMENTS = sorted([
    "Apoptosis", "Cell adhesion", "Cell cycle", "Chromatin", "Cytoskeleton",
    "DNA repair", "GTPase", "Immune", "Ion channel", "Kinase", "Methylation",
    "Nuc acid bind", "Phosphatase", "Protein folding", "Proteolysis",
    "RNA processing", "Signaling", "Structural", "Transcription",
    "Translation", "Transport", "Ubiquitin",
])
D2I = {d: i for i, d in enumerate(VALID_DEPARTMENTS)}
N_DEPTS = len(VALID_DEPARTMENTS)


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def load_data():
    print("[LOAD] Loading all data...")
    t0 = time.time()

    with open(STATE_PATH, "rb") as f:
        state = pickle.load(f)
    ptt = state["ptt"]
    ttp = state["ttp"]
    gene_cache = state["gene_cache"]

    vocab_dept = {}
    with open(VOCAB_PATH) as f:
        for row in csv.DictReader(f):
            vocab_dept[row["word_hex"].replace("0x", "").upper()] = row["primary_function"]

    gene_depts = {}
    with open(GENE_DEPTS_PATH) as f:
        for row in csv.DictReader(f):
            gene_depts[row["gene"]] = row["department"]

    with open(PROFILES_PATH) as f:
        profiles_data = json.load(f)
    profiles = {}
    for gene, prof in profiles_data["profiles"].items():
        profiles[gene] = np.array([prof.get(d, 0) for d in VALID_DEPARTMENTS])

    protein_dept_seqs = {}
    for uid, tokens in ptt.items():
        depts = []
        for tok in tokens:
            d = vocab_dept.get(tok.upper())
            if d and d in D2I:
                depts.append(d)
        if depts:
            compressed = []
            for d in depts:
                if not compressed or compressed[-1] != d:
                    compressed.append(d)
            protein_dept_seqs[uid] = "|".join(compressed)

    with open(PRIMITIVES_PATH) as f:
        raw_prims = list(csv.DictReader(f))
    primitives = []
    for p in raw_prims:
        ds = [d for d in p["function_sequence"].split("|") if d in D2I]
        if not ds:
            continue
        search = "|".join(ds)
        carriers = [uid for uid, seq in protein_dept_seqs.items() if search in seq]
        if len(carriers) >= 20:
            primitives.append({"search": search, "n_carriers": len(carriers)})

    gene_to_uids = defaultdict(list)
    for uid, g in gene_cache.items():
        if g:
            gene_to_uids[g].append(uid)

    gene_to_prim = defaultdict(set)
    prim_to_genes = defaultdict(list)
    for p in primitives:
        carriers = [uid for uid, seq in protein_dept_seqs.items() if p["search"] in seq]
        for uid in carriers:
            g = gene_cache.get(uid)
            if g and g in profiles:
                gene_to_prim[g].add(p["search"])
                prim_to_genes[p["search"]].append(g)

    for k in prim_to_genes:
        prim_to_genes[k] = list(set(prim_to_genes[k]))

    testable = {p: genes for p, genes in prim_to_genes.items() if len(genes) >= 5}

    gene_tokens = {}
    for gene in profiles:
        uids = gene_to_uids.get(gene, [])
        toks = set()
        for uid in uids:
            toks.update(ptt.get(uid, []))
        if toks:
            gene_tokens[gene] = toks

    dept_uids = defaultdict(list)
    for uid, g in gene_cache.items():
        if g and g in gene_depts:
            d = gene_depts[g]
            if d in D2I:
                dept_uids[d].append(uid)

    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Testable primitives: {len(testable)}")
    print(f"  Genes with profiles: {len(profiles)}")
    print(f"  Genes with tokens: {len(gene_tokens)}")

    return {
        "ptt": ptt, "ttp": ttp, "gene_cache": gene_cache,
        "gene_depts": gene_depts, "profiles": profiles,
        "gene_to_uids": gene_to_uids, "gene_tokens": gene_tokens,
        "primitives": primitives, "testable": testable,
        "prim_to_genes": prim_to_genes, "gene_to_prim": gene_to_prim,
        "dept_uids": dept_uids,
    }


def compute_collinearity_d(profiles_dict, testable, all_genes, rng=None):
    """Compute within vs across cosine and Cohen's d for testable primitives."""
    if rng is None:
        rng = np.random.RandomState(42)

    within_cos = []
    across_cos = []

    for prim, genes in testable.items():
        vecs = [profiles_dict[g] for g in genes
                if g in profiles_dict and np.linalg.norm(profiles_dict[g]) > 1e-10]
        if len(vecs) < 3:
            continue

        for i in range(len(vecs)):
            for j in range(i + 1, min(i + 10, len(vecs))):
                within_cos.append(cosine_sim(vecs[i], vecs[j]))

        rand = rng.choice(all_genes, size=min(len(genes), 50), replace=False)
        rvecs = [profiles_dict[g] for g in rand
                 if g in profiles_dict and np.linalg.norm(profiles_dict[g]) > 1e-10]
        for i in range(len(vecs)):
            for j in range(min(10, len(rvecs))):
                across_cos.append(cosine_sim(vecs[i], rvecs[j]))

    wc = np.array(within_cos)
    ac = np.array(across_cos)
    pooled = np.sqrt((wc.var() + ac.var()) / 2)
    d = (wc.mean() - ac.mean()) / pooled if pooled > 0 else 0

    return {
        "within_mean": float(wc.mean()),
        "across_mean": float(ac.mean()),
        "delta": float(wc.mean() - ac.mean()),
        "d": float(d),
        "n_within": len(within_cos),
        "n_across": len(across_cos),
    }


def control1_graph_rewiring(data, n_perms=20):
    """Degree-preserving edge swaps, recompute profiles, test collinearity."""
    print("\n" + "=" * 72)
    print("  CONTROL 1: GRAPH REWIRING (degree-preserving edge swaps)")
    print("  Question: Is collinearity a property of the algebra or the topology?")
    print("=" * 72)
    t0 = time.time()

    ptt = data["ptt"]
    ttp = data["ttp"]
    gene_cache = data["gene_cache"]
    gene_depts = data["gene_depts"]
    testable = data["testable"]
    profiles = data["profiles"]
    dept_uids = data["dept_uids"]
    gene_to_uids = data["gene_to_uids"]

    all_genes = list(profiles.keys())
    real_result = compute_collinearity_d(profiles, testable, all_genes)
    print(f"\n  Real graph collinearity: d={real_result['d']:+.4f} "
          f"(within={real_result['within_mean']:.4f}, across={real_result['across_mean']:.4f})")

    relevant_genes = set()
    for genes in testable.values():
        relevant_genes.update(genes)
    relevant_genes = [g for g in relevant_genes if g in profiles]
    print(f"  Genes to profile per permutation: {len(relevant_genes)}")

    tok_list = []
    uid_list = []
    for tok, uids in ttp.items():
        for uid in uids:
            tok_list.append(tok)
            uid_list.append(uid)
    n_edges = len(tok_list)

    all_uids_arr = np.array(uid_list, dtype=object)
    all_toks_arr = np.array(tok_list, dtype=object)

    uid_degree = Counter(uid_list)
    uid_stub_list = []
    for uid, deg in uid_degree.items():
        uid_stub_list.extend([uid] * deg)

    print(f"  Edges: {n_edges:,}")
    print(f"  Running {n_perms} permutations (stub-matching preserves degree)...")

    rewired_ds = []
    rng = np.random.RandomState(42)

    for perm_i in range(n_perms):
        t_perm = time.time()

        shuffled_uids = np.array(uid_stub_list, dtype=object)
        rng.shuffle(shuffled_uids)

        rewired_ptt = defaultdict(list)
        for i in range(n_edges):
            rewired_ptt[shuffled_uids[i]].append(tok_list[i])

        rewired_profiles = {}
        for gene in relevant_genes:
            uids = gene_to_uids.get(gene, [])
            if not uids:
                continue

            gene_toks = set()
            for uid in uids:
                gene_toks.update(rewired_ptt.get(uid, []))

            if not gene_toks:
                continue

            profile = np.zeros(N_DEPTS)
            for di, dept in enumerate(VALID_DEPARTMENTS):
                d_uids = dept_uids.get(dept, [])[:300]
                if not d_uids:
                    continue
                total = 0
                lost = 0
                for uid in d_uids:
                    toks = set(rewired_ptt.get(uid, []))
                    total += len(toks)
                    lost += len(toks & gene_toks)
                profile[di] = lost / max(total, 1)

            rewired_profiles[gene] = profile

        perm_result = compute_collinearity_d(
            rewired_profiles, testable,
            list(rewired_profiles.keys()), rng=np.random.RandomState(perm_i)
        )
        rewired_ds.append(perm_result["d"])

        elapsed = time.time() - t_perm
        print(f"    Perm {perm_i+1:2d}/{n_perms}: d={perm_result['d']:+.4f} ({elapsed:.1f}s)")
        sys.stdout.flush()

    rewired_arr = np.array(rewired_ds)

    survival_d = (real_result["d"] - rewired_arr.mean()) / max(rewired_arr.std(), 1e-6)

    print(f"\n  === CONTROL 1 RESULTS ===")
    print(f"  Real collinearity d:       {real_result['d']:+.4f}")
    print(f"  Rewired mean d:            {rewired_arr.mean():+.4f} ± {rewired_arr.std():.4f}")
    print(f"  Rewired range:             [{rewired_arr.min():+.4f}, {rewired_arr.max():+.4f}]")
    print(f"  Survival (real vs rewired): {survival_d:+.4f} SD above null")
    print(f"  Fraction retained:         {rewired_arr.mean()/real_result['d']:.1%}")

    if rewired_arr.mean() < 0.2:
        verdict = "TOPOLOGICAL"
        print(f"  VERDICT: Collinearity is primarily TOPOLOGICAL (dies on rewiring)")
    elif rewired_arr.mean() > real_result["d"] * 0.5:
        verdict = "ALGEBRAIC"
        print(f"  VERDICT: Collinearity is primarily ALGEBRAIC (survives rewiring)")
    else:
        verdict = "MIXED"
        print(f"  VERDICT: MIXED — partially topological, partially algebraic")

    print(f"  ({time.time()-t0:.0f}s total)")

    return {
        "real_d": round(real_result["d"], 4),
        "rewired_mean_d": round(float(rewired_arr.mean()), 4),
        "rewired_std_d": round(float(rewired_arr.std()), 4),
        "rewired_range": [round(float(rewired_arr.min()), 4), round(float(rewired_arr.max()), 4)],
        "survival_sd": round(float(survival_d), 4),
        "fraction_retained": round(float(rewired_arr.mean() / real_result["d"]), 4),
        "n_perms": n_perms,
        "verdict": verdict,
        "all_rewired_ds": [round(float(d), 4) for d in rewired_ds],
    }


def control2_dept_matched_groups(data):
    """Department-matched gene groups that don't share primitives."""
    print("\n" + "=" * 72)
    print("  CONTROL 2: DEPARTMENT-MATCHED GROUPS")
    print("  Question: Do primitives add information beyond functional annotation?")
    print("=" * 72)
    t0 = time.time()

    profiles = data["profiles"]
    gene_depts = data["gene_depts"]
    testable = data["testable"]
    gene_to_prim = data["gene_to_prim"]
    all_genes = list(profiles.keys())

    dept_genes = defaultdict(list)
    for gene in profiles:
        d = gene_depts.get(gene)
        if d:
            dept_genes[d].append(gene)

    primitive_results = []
    dept_matched_results = []

    rng = np.random.RandomState(42)

    for prim, prim_genes in testable.items():
        prim_vecs = [profiles[g] for g in prim_genes
                     if np.linalg.norm(profiles[g]) > 1e-10]
        if len(prim_vecs) < 3:
            continue

        prim_within = []
        for i in range(len(prim_vecs)):
            for j in range(i + 1, min(i + 10, len(prim_vecs))):
                prim_within.append(cosine_sim(prim_vecs[i], prim_vecs[j]))

        prim_depts = Counter(gene_depts.get(g, "Unknown") for g in prim_genes)
        dominant_dept = prim_depts.most_common(1)[0][0] if prim_depts else None

        if dominant_dept and dominant_dept in dept_genes:
            prim_gene_set = set(prim_genes)
            matched_pool = [g for g in dept_genes[dominant_dept]
                           if g not in prim_gene_set and g in profiles
                           and np.linalg.norm(profiles[g]) > 1e-10]

            if len(matched_pool) >= len(prim_genes):
                matched_sample = list(rng.choice(matched_pool,
                                                  size=min(len(prim_genes), len(matched_pool)),
                                                  replace=False))
                matched_vecs = [profiles[g] for g in matched_sample]

                dept_within = []
                for i in range(len(matched_vecs)):
                    for j in range(i + 1, min(i + 10, len(matched_vecs))):
                        dept_within.append(cosine_sim(matched_vecs[i], matched_vecs[j]))

                if dept_within:
                    primitive_results.append(np.mean(prim_within))
                    dept_matched_results.append(np.mean(dept_within))

    prim_arr = np.array(primitive_results)
    dept_arr = np.array(dept_matched_results)

    diff = prim_arr - dept_arr
    t_stat, t_p = stats.ttest_rel(prim_arr, dept_arr)

    print(f"\n  === CONTROL 2 RESULTS ===")
    print(f"  Primitives tested:              {len(primitive_results)}")
    print(f"  Primitive within-cos:            {prim_arr.mean():.4f} ± {prim_arr.std():.4f}")
    print(f"  Dept-matched within-cos:         {dept_arr.mean():.4f} ± {dept_arr.std():.4f}")
    print(f"  Primitive advantage (Δ):         {diff.mean():+.4f}")
    print(f"  Paired t-test: t={t_stat:.3f}, p={t_p:.2e}")
    print(f"  Primitives outperform dept:      {np.mean(diff > 0):.1%}")

    if diff.mean() > 0.01 and t_p < 0.01:
        verdict = "PRIMITIVES_ADD_INFORMATION"
        print(f"  VERDICT: Primitives capture structure BEYOND functional annotation")
    elif diff.mean() > 0 and t_p < 0.05:
        verdict = "WEAK_ADVANTAGE"
        print(f"  VERDICT: Primitives have a weak advantage over department matching")
    else:
        verdict = "NO_ADVANTAGE"
        print(f"  VERDICT: Primitives do NOT add significant info beyond dept annotation")

    print(f"  ({time.time()-t0:.1f}s)")

    return {
        "n_tested": len(primitive_results),
        "primitive_cos": round(float(prim_arr.mean()), 4),
        "dept_matched_cos": round(float(dept_arr.mean()), 4),
        "advantage": round(float(diff.mean()), 4),
        "t_stat": round(float(t_stat), 3),
        "p_value": float(t_p),
        "frac_outperform": round(float(np.mean(diff > 0)), 4),
        "verdict": verdict,
    }


def control3_token_overlap(data):
    """Token neighborhood overlap (Jaccard) and regression."""
    print("\n" + "=" * 72)
    print("  CONTROL 3: TOKEN NEIGHBORHOOD OVERLAP (Jaccard)")
    print("  Question: Is collinearity just 'shared tokens → shared profiles'?")
    print("=" * 72)
    t0 = time.time()

    profiles = data["profiles"]
    gene_tokens = data["gene_tokens"]
    testable = data["testable"]

    all_genes = [g for g in profiles if g in gene_tokens]
    rng = np.random.RandomState(42)

    within_jaccard = []
    within_cosine = []
    across_jaccard = []
    across_cosine = []

    for prim, genes in testable.items():
        genes_with_toks = [g for g in genes if g in gene_tokens
                           and np.linalg.norm(profiles[g]) > 1e-10]
        if len(genes_with_toks) < 3:
            continue

        for i in range(len(genes_with_toks)):
            for j in range(i + 1, min(i + 8, len(genes_with_toks))):
                g1, g2 = genes_with_toks[i], genes_with_toks[j]
                t1, t2 = gene_tokens[g1], gene_tokens[g2]
                intersection = len(t1 & t2)
                union = len(t1 | t2)
                jacc = intersection / max(union, 1)
                cos = cosine_sim(profiles[g1], profiles[g2])
                within_jaccard.append(jacc)
                within_cosine.append(cos)

        rand = rng.choice(all_genes, size=min(len(genes_with_toks), 30), replace=False)
        for i in range(min(len(genes_with_toks), 10)):
            for j in range(min(10, len(rand))):
                g1 = genes_with_toks[i]
                g2 = rand[j]
                if g2 not in gene_tokens:
                    continue
                t1, t2 = gene_tokens[g1], gene_tokens[g2]
                intersection = len(t1 & t2)
                union = len(t1 | t2)
                jacc = intersection / max(union, 1)
                cos = cosine_sim(profiles[g1], profiles[g2])
                across_jaccard.append(jacc)
                across_cosine.append(cos)

    wj = np.array(within_jaccard)
    wc = np.array(within_cosine)
    aj = np.array(across_jaccard)
    ac = np.array(across_cosine)

    print(f"\n  Within-primitive:")
    print(f"    Jaccard: {wj.mean():.4f} ± {wj.std():.4f}")
    print(f"    Cosine:  {wc.mean():.4f} ± {wc.std():.4f}")
    print(f"  Across-primitive:")
    print(f"    Jaccard: {aj.mean():.4f} ± {aj.std():.4f}")
    print(f"    Cosine:  {ac.mean():.4f} ± {ac.std():.4f}")

    jaccard_d_raw = (wj.mean() - aj.mean()) / np.sqrt((wj.var() + aj.var()) / 2)
    print(f"\n  Jaccard difference (within vs across): d={jaccard_d_raw:+.4f}")

    all_jacc = np.concatenate([wj, aj])
    all_cos = np.concatenate([wc, ac])
    all_label = np.concatenate([np.ones(len(wj)), np.zeros(len(aj))])

    if len(all_jacc) > 0 and np.std(all_jacc) > 0:
        slope, intercept, r, p_corr, se = stats.linregress(all_jacc, all_cos)
        print(f"  Jaccard-Cosine correlation: r={r:.4f}, p={p_corr:.2e}")

        predicted_cos = intercept + slope * all_jacc
        residuals = all_cos - predicted_cos

        within_resid = residuals[:len(wj)]
        across_resid = residuals[len(wj):]

        resid_pooled = np.sqrt((within_resid.var() + across_resid.var()) / 2)
        d_residual = ((within_resid.mean() - across_resid.mean()) / resid_pooled
                      if resid_pooled > 0 else 0)

        u_stat, u_p = stats.mannwhitneyu(within_resid, across_resid, alternative="greater")

        print(f"\n  === AFTER REGRESSING OUT TOKEN OVERLAP ===")
        print(f"  Within-primitive residual cos: {within_resid.mean():+.4f}")
        print(f"  Across-primitive residual cos: {across_resid.mean():+.4f}")
        print(f"  Residual Cohen's d:            {d_residual:+.4f}")
        print(f"  Mann-Whitney p:                {u_p:.2e}")
    else:
        d_residual = 0
        u_p = 1.0
        r = 0
        within_resid = np.array([0])
        across_resid = np.array([0])

    raw_d = (wc.mean() - ac.mean()) / np.sqrt((wc.var() + ac.var()) / 2)
    print(f"\n  Raw cosine d (before correction): {raw_d:+.4f}")
    print(f"  Residual d (after correction):    {d_residual:+.4f}")

    if raw_d > 0:
        retained = d_residual / raw_d
    else:
        retained = 0
    print(f"  Fraction of d retained:           {retained:.1%}")

    if d_residual > 0.5:
        verdict = "SURVIVES"
        print(f"\n  VERDICT: Collinearity SURVIVES after removing token overlap effect")
    elif d_residual > 0.2:
        verdict = "PARTIALLY_SURVIVES"
        print(f"\n  VERDICT: Collinearity partially survives (d={d_residual:+.3f})")
    else:
        verdict = "EXPLAINED_BY_OVERLAP"
        print(f"\n  VERDICT: Collinearity is largely explained by token overlap")

    print(f"  ({time.time()-t0:.1f}s)")

    return {
        "within_jaccard": round(float(wj.mean()), 4),
        "across_jaccard": round(float(aj.mean()), 4),
        "jaccard_d": round(float(jaccard_d_raw), 4),
        "jaccard_cos_r": round(float(r), 4),
        "raw_cosine_d": round(float(raw_d), 4),
        "residual_d": round(float(d_residual), 4),
        "residual_p": float(u_p),
        "fraction_retained": round(float(retained), 4),
        "verdict": verdict,
    }


def main():
    data = load_data()

    r1 = control1_graph_rewiring(data, n_perms=20)
    r2 = control2_dept_matched_groups(data)
    r3 = control3_token_overlap(data)

    output = {
        "control1_graph_rewiring": r1,
        "control2_dept_matched": r2,
        "control3_token_overlap": r3,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*72}")
    print(f"  COLLINEARITY NULL CONTROLS — SUMMARY")
    print(f"{'='*72}")
    print(f"  Control 1 (Graph rewiring):      {r1['verdict']}")
    print(f"    Real d={r1['real_d']:+.4f}, Rewired mean d={r1['rewired_mean_d']:+.4f} "
          f"({r1['fraction_retained']:.0%} retained)")
    print(f"  Control 2 (Dept-matched):        {r2['verdict']}")
    print(f"    Primitive cos={r2['primitive_cos']:.4f}, Dept cos={r2['dept_matched_cos']:.4f} "
          f"(p={r2['p_value']:.2e})")
    print(f"  Control 3 (Token overlap):       {r3['verdict']}")
    print(f"    Raw d={r3['raw_cosine_d']:+.4f}, After PPI correction d={r3['residual_d']:+.4f} "
          f"({r3['fraction_retained']:.0%} retained)")
    print(f"{'='*72}")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
