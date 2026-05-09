#!/usr/bin/env python3
"""
Lisp Computation Tests v2 — informed by PL theory critique.
============================================================

Tests the COMPUTATION, not just the components. Redesigned based on
analysis from a programming language theory perspective.

Test 1: EVALUATION TRACE
  When we knock out a gene, does disruption propagate following the
  nesting tree (primitives disrupted before composites, composites
  before top-level)? Or does it spread diffusely like a random network?
  If it follows the tree → this is an evaluator, not just a network.

Test 2: SEQUENTIAL TRANSFORMATION (Function Composition)
  Lisp composes via (f (g x)) = f transforms g's output.
  Learn transformation operators for each primitive, then test whether
  compositing T_f @ T_g predicts composite behavior better than
  additive (f + g) or random baselines.

Test 3: CONFLUENCE VERIFICATION (Church-Rosser)
  Verify that d≈0 for composition order is genuinely Church-Rosser
  and not a substring artifact. Check whether proteins with "different
  orders" are actually distinct proteins.

Usage:
    python3 -u validation/knockout/lisp_computation_tests.py
"""

import csv
import json
import os
import pickle
import random
import sys
import time
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
from scipy import stats, sparse

STATE_PATH = "/tmp/module8_full_state.pkl"
VOCAB_PATH = "server/data/human/vocabulary.csv"
PRIMITIVES_PATH = "beta_transfer/genome_primitives.csv"
NESTING_PATH = "beta_transfer/genome_nesting_hierarchy.csv"
GENE_DEPTS_PATH = "server/data/human/gene_departments.csv"
KO_RESULTS_PATH = "validation/knockout/knockout_full_results.json"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "lisp_computation_results.json")

VALID_DEPARTMENTS = sorted([
    "Apoptosis", "Cell adhesion", "Cell cycle", "Chromatin", "Cytoskeleton",
    "DNA repair", "GTPase", "Immune", "Ion channel", "Kinase", "Methylation",
    "Nuc acid bind", "Phosphatase", "Protein folding", "Proteolysis",
    "RNA processing", "Signaling", "Structural", "Transcription",
    "Translation", "Transport", "Ubiquitin",
])
DEPT_TO_IDX = {d: i for i, d in enumerate(VALID_DEPARTMENTS)}
N_DEPTS = len(VALID_DEPARTMENTS)


def load_all():
    print("[1/5] Loading dispatch graph state...")
    t0 = time.time()

    with open(STATE_PATH, "rb") as f:
        state = pickle.load(f)

    ptt = state["ptt"]
    ttp = state["ttp"]
    gene_cache = state["gene_cache"]

    print("[2/5] Building vocab and department lookups...")
    vocab_dept = {}
    with open(VOCAB_PATH) as f:
        for row in csv.DictReader(f):
            stripped = row["word_hex"].replace("0x", "").upper()
            vocab_dept[stripped] = row["primary_function"]

    protein_dept_seqs = {}
    for uid, tokens in ptt.items():
        depts = []
        for tok in tokens:
            dept = vocab_dept.get(tok.upper())
            if dept and dept in DEPT_TO_IDX:
                depts.append(dept)
        if depts:
            compressed = []
            for d in depts:
                if not compressed or compressed[-1] != d:
                    compressed.append(d)
            protein_dept_seqs[uid] = compressed

    gene_depts = {}
    with open(GENE_DEPTS_PATH) as f:
        for row in csv.DictReader(f):
            gene_depts[row["gene"]] = row["department"]

    all_tokens = sorted(ttp.keys())
    all_proteins = sorted(ptt.keys())
    tok_to_idx = {t: i for i, t in enumerate(all_tokens)}
    uid_to_idx = {u: i for i, u in enumerate(all_proteins)}
    n_p, n_t = len(all_proteins), len(all_tokens)

    rows_l, cols_l = [], []
    for tok, uids in ttp.items():
        ti = tok_to_idx[tok]
        for uid in uids:
            rows_l.append(uid_to_idx[uid])
            cols_l.append(ti)
    P = sparse.csr_matrix(
        (np.ones(len(rows_l), dtype=np.float32), (rows_l, cols_l)),
        shape=(n_p, n_t),
    )

    dept_mask = np.zeros((N_DEPTS, n_p), dtype=np.float32)
    for uid, idx in uid_to_idx.items():
        gene = gene_cache.get(uid)
        if gene:
            dept = gene_depts.get(gene)
            if dept and dept in DEPT_TO_IDX:
                dept_mask[DEPT_TO_IDX[dept], idx] = 1.0

    dept_token_disruption = dept_mask @ P

    print("[3/5] Loading primitives and nesting hierarchy...")
    with open(PRIMITIVES_PATH) as f:
        raw_prims = list(csv.DictReader(f))
    primitives = []
    for p in raw_prims:
        depts = [d for d in p["function_sequence"].split("|") if d in DEPT_TO_IDX]
        if depts:
            primitives.append({
                "raw": p["function_sequence"],
                "search": "|".join(depts),
                "depts": depts,
            })

    with open(NESTING_PATH) as f:
        nesting = list(csv.DictReader(f))

    print("[4/5] Loading knockout gene list...")
    with open(KO_RESULTS_PATH) as f:
        ko_data = json.load(f)
    ko_genes = {}
    for entry in ko_data["results"]:
        ko_genes[entry["gene"]] = entry

    print(f"[5/5] Ready: {len(protein_dept_seqs)} proteins, {len(primitives)} primitives, "
          f"{len(nesting)} nesting rows, {len(ko_genes)} knockout genes")
    print(f"  ({time.time()-t0:.1f}s)")

    return {
        "state": state, "ptt": ptt, "ttp": ttp, "gene_cache": gene_cache,
        "vocab_dept": vocab_dept, "protein_dept_seqs": protein_dept_seqs,
        "gene_depts": gene_depts, "tok_to_idx": tok_to_idx,
        "uid_to_idx": uid_to_idx, "P": P,
        "dept_token_disruption": dept_token_disruption,
        "primitives": primitives, "nesting": nesting,
        "ko_genes": ko_genes,
    }


def find_carriers(search_depts, protein_dept_seqs, as_list=True):
    search_str = "|".join(search_depts) if isinstance(search_depts, list) else search_depts
    results = set()
    for uid, depts in protein_dept_seqs.items():
        seq_str = "|".join(depts)
        if search_str in seq_str:
            results.add(uid)
    return list(results) if as_list else results


def protein_profile(uid, data):
    tokens = data["ptt"].get(uid, [])
    idxs = [data["tok_to_idx"][t] for t in tokens if t in data["tok_to_idx"]]
    if not idxs:
        return np.zeros(N_DEPTS)
    return data["dept_token_disruption"][:, np.array(idxs, dtype=np.int32)].sum(axis=1)


def mean_profile(uids, data, max_n=200):
    profiles = []
    for uid in uids[:max_n]:
        p = protein_profile(uid, data)
        profiles.append(p)
    if not profiles:
        return np.zeros(N_DEPTS)
    return np.mean(profiles, axis=0)


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_disruption_profile(gene, data):
    """
    Compute a gene's disruption profile directly from the dispatch graph.
    
    When gene is knocked out, its tokens are removed. We measure what 
    fraction of each department's token connectivity is lost.
    """
    gene_cache = data["gene_cache"]
    ptt = data["ptt"]
    gene_depts = data["gene_depts"]
    ttp = data["ttp"]

    gene_uids = [uid for uid, g in gene_cache.items() if g == gene]
    if not gene_uids:
        return None

    gene_tokens = set()
    for uid in gene_uids:
        gene_tokens.update(ptt.get(uid, []))

    if not gene_tokens:
        return None

    profile = {}
    for dept in VALID_DEPARTMENTS:
        dept_uids = [uid for uid, g in gene_cache.items()
                     if g and gene_depts.get(g) == dept and g != gene]

        if not dept_uids:
            profile[dept] = 0.0
            continue

        total_connections = 0
        lost_connections = 0
        for uid in dept_uids[:200]:
            tokens = set(ptt.get(uid, []))
            total_connections += len(tokens)
            lost_connections += len(tokens & gene_tokens)

        profile[dept] = lost_connections / max(total_connections, 1)

    return profile


def test1_evaluation_trace(data):
    """
    Test whether knockout disruption propagates following the nesting tree.
    
    For each knocked-out gene, compute its disruption profile, then check:
    do the departments in PRIMITIVE programs show more disruption than
    the departments in COMPOSITE programs that contain those primitives?
    
    If this is an evaluator with tree-structured evaluation, disruption 
    should hit leaves (primitives) harder than the composites above them.
    """
    print("\n" + "=" * 72)
    print("  TEST 1: EVALUATION TRACE")
    print("  Does disruption follow the nesting tree?")
    print("=" * 72)
    t0 = time.time()

    nesting = data["nesting"]
    primitives = data["primitives"]
    prim_search_set = {p["search"] for p in primitives}

    outer_to_inners = defaultdict(set)
    for n in nesting:
        outer_depts = [d for d in n["outer_sequence"].split("|") if d in DEPT_TO_IDX]
        inner_depts = [d for d in n["inner_sequence"].split("|") if d in DEPT_TO_IDX]
        if outer_depts and inner_depts:
            o = "|".join(outer_depts)
            i = "|".join(inner_depts)
            outer_to_inners[o].add(i)

    composites_containing_prims = {}
    for outer, inners in outer_to_inners.items():
        prim_inners = inners & prim_search_set
        if prim_inners and outer not in prim_search_set:
            composites_containing_prims[outer] = prim_inners

    print(f"  Composites with primitive children: {len(composites_containing_prims)}")

    gene_cache = data["gene_cache"]
    ko_genes = data["ko_genes"]

    test_genes = list(ko_genes.keys())[:100]
    print(f"  Computing disruption profiles for {len(test_genes)} genes...")

    inner_more_disrupted = 0
    outer_more_disrupted = 0
    ties = 0
    effect_sizes = []
    tested_genes = 0

    for gi, gene in enumerate(test_genes):
        dp = compute_disruption_profile(gene, data)
        if not dp:
            continue

        total_disruption = sum(dp.values())
        if total_disruption == 0:
            continue

        pair_count = 0
        for comp_seq, prim_set in composites_containing_prims.items():
            comp_depts = comp_seq.split("|")
            comp_disruption = np.mean([dp.get(d, 0) for d in comp_depts])

            for prim_seq in prim_set:
                prim_depts = prim_seq.split("|")
                prim_disruption = np.mean([dp.get(d, 0) for d in prim_depts])

                if prim_disruption > comp_disruption + 1e-10:
                    inner_more_disrupted += 1
                    effect_sizes.append(prim_disruption - comp_disruption)
                elif comp_disruption > prim_disruption + 1e-10:
                    outer_more_disrupted += 1
                    effect_sizes.append(-(comp_disruption - prim_disruption))
                else:
                    ties += 1

                pair_count += 1

        if pair_count > 0:
            tested_genes += 1

        if (gi + 1) % 20 == 0:
            print(f"  [{gi+1}/{len(test_genes)}] tested_genes={tested_genes} "
                  f"inner={inner_more_disrupted} outer={outer_more_disrupted}")
            sys.stdout.flush()

    total_pairs = inner_more_disrupted + outer_more_disrupted + ties

    frac_inner = inner_more_disrupted / max(inner_more_disrupted + outer_more_disrupted, 1)
    eff_arr = np.array(effect_sizes) if effect_sizes else np.array([0])

    if inner_more_disrupted + outer_more_disrupted > 0:
        bt = stats.binomtest(inner_more_disrupted,
                              inner_more_disrupted + outer_more_disrupted,
                              p=0.5, alternative="greater")
        binom_p = bt.pvalue
    else:
        binom_p = 1.0

    print(f"\n  === TEST 1 RESULTS ===")
    print(f"  Genes tested:             {tested_genes}")
    print(f"  Primitive-composite pairs: {total_pairs}")
    print(f"  Inner (primitive) more disrupted: {inner_more_disrupted} ({frac_inner:.1%})")
    print(f"  Outer (composite) more disrupted: {outer_more_disrupted}")
    print(f"  Ties:                     {ties}")
    print(f"  Binomial p:               {binom_p:.2e}")
    print(f"  Mean effect (inner-outer): {eff_arr.mean():+.6f}")
    print(f"")
    if frac_inner > 0.55:
        print(f"  INTERPRETATION: Disruption hits primitives harder than composites,")
        print(f"  consistent with tree-structured (leaf-first) evaluation.")
    elif frac_inner < 0.45:
        print(f"  INTERPRETATION: Disruption hits composites harder — outward propagation.")
    else:
        print(f"  INTERPRETATION: No directional preference — diffuse propagation.")
    print(f"  ({time.time()-t0:.1f}s)")

    return {
        "tested_genes": tested_genes,
        "total_pairs": total_pairs,
        "inner_more_disrupted": inner_more_disrupted,
        "outer_more_disrupted": outer_more_disrupted,
        "ties": ties,
        "frac_inner": round(frac_inner, 4),
        "binom_p": float(binom_p),
        "mean_effect": round(float(eff_arr.mean()), 6),
        "tree_structured": frac_inner > 0.55,
    }


def test2_sequential_transform(data):
    """
    Test the correct Lisp composition model: function composition.
    
    In Lisp, (f (g x)) means f transforms g's output — sequential
    state transformation, not f + g.
    
    For each primitive, learn a transformation matrix T_p (how it
    transforms a 22-dim department profile). Then for composites
    with primitives [p1, p2], predict composite profile as T_p2 @ T_p1 @ init.
    Compare to additive model (p1 + p2) and random baseline.
    """
    print("\n" + "=" * 72)
    print("  TEST 2: SEQUENTIAL TRANSFORMATION (Function Composition)")
    print("  Does (f (g x)) = T_f(T_g(x)) predict composites?")
    print("=" * 72)
    t0 = time.time()

    primitives = data["primitives"]
    protein_dept_seqs = data["protein_dept_seqs"]

    print("  [2a] Computing primitive carrier profiles...")
    prim_profiles = {}
    prim_carrier_profiles = {}
    for p in primitives:
        carriers = find_carriers(p["search"], protein_dept_seqs)
        if len(carriers) >= 10:
            profiles = [protein_profile(uid, data) for uid in carriers[:300]]
            mean_p = np.mean(profiles, axis=0)
            prim_profiles[p["search"]] = mean_p
            prim_carrier_profiles[p["search"]] = profiles

    print(f"  {len(prim_profiles)} primitives with sufficient carriers")

    print("  [2b] Learning transformation operators...")
    global_mean = np.zeros(N_DEPTS)
    count = 0
    for uid in list(protein_dept_seqs.keys())[:5000]:
        global_mean += protein_profile(uid, data)
        count += 1
    global_mean /= max(count, 1)
    global_norm = normalize(global_mean)

    prim_transforms = {}
    for prim_seq, prof in prim_profiles.items():
        diff = prof - global_mean
        norm_diff = normalize(diff) if np.linalg.norm(diff) > 0 else diff
        T = np.eye(N_DEPTS) + np.outer(norm_diff, norm_diff) * np.linalg.norm(diff)
        prim_transforms[prim_seq] = T

    nesting = data["nesting"]
    outer_to_inners = defaultdict(list)
    for n in nesting:
        if n.get("inner_layer") == "Primitive":
            outer_depts = [d for d in n["outer_sequence"].split("|") if d in DEPT_TO_IDX]
            inner_depts = [d for d in n["inner_sequence"].split("|") if d in DEPT_TO_IDX]
            if outer_depts and inner_depts:
                o = "|".join(outer_depts)
                i = "|".join(inner_depts)
                if i in prim_transforms:
                    outer_to_inners[o].append(i)

    prim_set = {p["search"] for p in primitives}
    composites = {o: list(set(inners))
                   for o, inners in outer_to_inners.items()
                   if len(set(inners)) >= 2 and o not in prim_set}

    print(f"  Composites with 2+ known primitive children: {len(composites)}")

    print("  [2c] Testing composition models...")

    composition_cos = []
    additive_cos = []
    mean_cos = []
    baseline_cos = []
    tested = 0

    for comp_seq, inner_seqs in composites.items():
        comp_carriers = find_carriers(comp_seq, protein_dept_seqs)
        if len(comp_carriers) < 5:
            continue

        actual = mean_profile(comp_carriers, data)
        if np.linalg.norm(actual) == 0:
            continue

        state_vec = global_mean.copy()
        for iseq in inner_seqs:
            T = prim_transforms[iseq]
            state_vec = T @ state_vec
        composition_cos.append(cosine_sim(state_vec, actual))

        additive_pred = sum(prim_profiles[iseq] for iseq in inner_seqs)
        additive_cos.append(cosine_sim(additive_pred, actual))

        mean_pred = np.mean([prim_profiles[iseq] for iseq in inner_seqs], axis=0)
        mean_cos.append(cosine_sim(mean_pred, actual))

        rand_prims = random.sample(list(prim_profiles.keys()),
                                     min(len(inner_seqs), len(prim_profiles)))
        rand_pred = sum(prim_profiles[rp] for rp in rand_prims)
        baseline_cos.append(cosine_sim(rand_pred, actual))

        tested += 1

    if tested == 0:
        print("  No testable composites found")
        return {"n_tested": 0, "status": "no_composites"}

    comp_arr = np.array(composition_cos)
    add_arr = np.array(additive_cos)
    mean_arr = np.array(mean_cos)
    base_arr = np.array(baseline_cos)

    print(f"\n  === TEST 2 RESULTS ({tested} composites) ===")
    print(f"  {'Model':<25s} {'Mean cosine':>12s} {'Median':>8s} {'vs Base':>10s}")
    print(f"  {'-'*58}")

    models = [
        ("Composition (T_f∘T_g)", comp_arr),
        ("Additive (f + g)", add_arr),
        ("Mean (avg)", mean_arr),
        ("Random baseline", base_arr),
    ]

    results = {}
    for name, arr in models:
        lift = arr.mean() - base_arr.mean()
        print(f"  {name:<25s} {arr.mean():12.4f} {np.median(arr):8.4f} {lift:+10.4f}")
        results[name.split('(')[0].strip()] = {
            "cosine_mean": round(float(arr.mean()), 4),
            "cosine_median": round(float(np.median(arr)), 4),
            "lift_over_baseline": round(float(lift), 4),
        }

    best_model = max(models[:3], key=lambda x: x[1].mean())
    print(f"\n  Best model: {best_model[0]} (cosine={best_model[1].mean():.4f})")

    if comp_arr.mean() > add_arr.mean():
        print(f"  → Function composition OUTPERFORMS additive!")
        print(f"    Consistent with Lisp-like sequential transformation.")
    elif add_arr.mean() > comp_arr.mean():
        print(f"  → Additive outperforms composition.")
        print(f"    System combines primitives by superposition, not transformation.")
    else:
        print(f"  → Models are equivalent.")

    if len(comp_arr) > 1 and len(base_arr) > 1:
        u_stat, u_p = stats.mannwhitneyu(comp_arr, base_arr, alternative="greater")
        pooled = np.sqrt((comp_arr.var() + base_arr.var()) / 2)
        d = (comp_arr.mean() - base_arr.mean()) / pooled if pooled > 0 else 0
    else:
        u_p, d = 1, 0

    print(f"  Composition vs baseline: d={d:+.3f}, p={u_p:.2e}")
    print(f"  ({time.time()-t0:.1f}s)")

    results["best_model"] = best_model[0]
    results["composition_vs_baseline_d"] = round(float(d), 4)
    results["composition_vs_baseline_p"] = float(u_p)
    results["n_tested"] = tested
    return results


def test3_confluence(data):
    """
    Verify Church-Rosser: d≈0 for composition order.
    
    Critical check: are proteins with "different orders" of the same
    primitives actually DISTINCT proteins, or is it a substring artifact
    (same proteins matched multiple ways)?
    """
    print("\n" + "=" * 72)
    print("  TEST 3: CONFLUENCE VERIFICATION (Church-Rosser)")
    print("  Is d≈0 for order genuine, or a substring artifact?")
    print("=" * 72)
    t0 = time.time()

    primitives = data["primitives"]
    protein_dept_seqs = data["protein_dept_seqs"]

    prim_list = [p["search"] for p in primitives]
    prim_depts_list = [p["depts"] for p in primitives]

    print("  [3a] Finding permutation groups (same departments, different order)...")
    by_dept_set = defaultdict(list)
    for i, p in enumerate(primitives):
        key = frozenset(p["depts"])
        by_dept_set[key].append(i)

    perm_groups = {k: v for k, v in by_dept_set.items() if len(v) >= 2}
    print(f"  Permutation groups (same dept set, different order): {len(perm_groups)}")

    print("  [3b] Checking for carrier overlap (artifact detection)...")
    artifact_count = 0
    genuine_count = 0
    within_cosines = []
    across_cosines = []
    overlap_fractions = []

    for dept_set, prim_indices in perm_groups.items():
        group_carriers = {}
        for pi in prim_indices:
            carriers = find_carriers(prim_list[pi], protein_dept_seqs, as_list=False)
            group_carriers[pi] = carriers

        for i in range(len(prim_indices)):
            for j in range(i + 1, len(prim_indices)):
                pi, pj = prim_indices[i], prim_indices[j]
                ci, cj = group_carriers[pi], group_carriers[pj]

                if not ci or not cj:
                    continue

                overlap = len(ci & cj)
                union = len(ci | cj)
                jaccard = overlap / union if union > 0 else 0
                overlap_fractions.append(jaccard)

                if jaccard > 0.9:
                    artifact_count += 1
                else:
                    genuine_count += 1

                    only_i = list(ci - cj)[:50]
                    only_j = list(cj - ci)[:50]

                    if len(only_i) >= 3 and len(only_j) >= 3:
                        prof_i = mean_profile(only_i, data)
                        prof_j = mean_profile(only_j, data)
                        within_cosines.append(cosine_sim(prof_i, prof_j))

    print(f"\n  === ARTIFACT DETECTION ===")
    print(f"  Permutation pairs checked:  {artifact_count + genuine_count}")
    print(f"  High overlap (Jaccard>0.9): {artifact_count} (ARTIFACTS)")
    print(f"  Genuine distinct carriers:  {genuine_count}")

    if overlap_fractions:
        of = np.array(overlap_fractions)
        print(f"  Mean Jaccard overlap:       {of.mean():.3f}")
        print(f"  Median Jaccard overlap:     {np.median(of):.3f}")

    print(f"\n  [3c] Testing genuine confluence (distinct-carrier pairs only)...")

    if within_cosines:
        all_uids = list(protein_dept_seqs.keys())
        for _ in range(len(within_cosines)):
            s1 = random.sample(all_uids, min(30, len(all_uids)))
            s2 = random.sample(all_uids, min(30, len(all_uids)))
            p1 = mean_profile(s1, data)
            p2 = mean_profile(s2, data)
            across_cosines.append(cosine_sim(p1, p2))

    if within_cosines and across_cosines:
        wc = np.array(within_cosines)
        ac = np.array(across_cosines)

        pooled = np.sqrt((wc.var() + ac.var()) / 2)
        d = (wc.mean() - ac.mean()) / pooled if pooled > 0 else 0

        print(f"\n  === GENUINE CONFLUENCE RESULTS ===")
        print(f"  Same-dept-set permutations (non-overlapping carriers only):")
        print(f"  Within-group cosine: {wc.mean():.4f} ({len(within_cosines)} pairs)")
        print(f"  Random baseline:     {ac.mean():.4f}")
        print(f"  Cohen's d:           {d:+.4f}")
        print(f"")

        if d > 0.5:
            print(f"  GENUINE CONFLUENCE: Same departments in different order → ")
            print(f"  similar profiles even with DISTINCT carrier proteins.")
            print(f"  Consistent with Church-Rosser (pure functional evaluation).")
        elif d > 0.2:
            print(f"  MODERATE confluence effect.")
        else:
            print(f"  WEAK/NO confluence from distinct carriers.")
    else:
        print(f"  Insufficient genuine distinct-carrier pairs for confluence test.")
        d = 0
        wc = np.array([0])
        ac = np.array([0])

    print(f"  ({time.time()-t0:.1f}s)")

    return {
        "permutation_groups": len(perm_groups),
        "artifact_pairs": artifact_count,
        "genuine_pairs": genuine_count,
        "mean_jaccard_overlap": round(float(np.mean(overlap_fractions)), 4) if overlap_fractions else 0,
        "within_cosine": round(float(wc.mean()), 4) if len(wc) > 0 else 0,
        "across_cosine": round(float(ac.mean()), 4) if len(ac) > 0 else 0,
        "cohens_d": round(float(d), 4),
        "genuine_confluence": bool(d > 0.5),
    }


def test4_fixed_point(data):
    """
    Test for fixed-point convergence in repetitive structures.
    
    Some chromosomal regions show patterns like A|B|A|B|A|B...
    In computation theory, this looks like iteration toward a fixed point.
    Test whether the disruption profile converges as repetition depth
    increases, and whether convergence is geometric.
    """
    print("\n" + "=" * 72)
    print("  TEST 4: FIXED-POINT CONVERGENCE")
    print("  Do repetitive patterns converge like iterative computation?")
    print("=" * 72)
    t0 = time.time()

    protein_dept_seqs = data["protein_dept_seqs"]
    primitives = data["primitives"]

    all_seqs = Counter()
    for uid, depts in protein_dept_seqs.items():
        seq = "|".join(depts)
        all_seqs[seq] += 1

    print("  [4a] Finding repetitive motifs...")
    motif_carriers = defaultdict(lambda: defaultdict(list))

    for p in primitives:
        base = p["search"]
        base_depts = base.split("|")

        for uid, depts in protein_dept_seqs.items():
            seq = "|".join(depts)
            rep2 = base + "|" + base
            rep3 = base + "|" + base + "|" + base

            if rep3 in seq:
                motif_carriers[base][3].append(uid)
            elif rep2 in seq:
                motif_carriers[base][2].append(uid)
            elif base in seq:
                motif_carriers[base][1].append(uid)

    testable = {motif: reps for motif, reps in motif_carriers.items()
                 if len(reps) >= 2 and all(len(uids) >= 3 for uids in reps.values())}

    print(f"  Motifs with multiple repetition depths: {len(testable)}")

    convergence_results = []
    for motif, reps in testable.items():
        depths = sorted(reps.keys())
        profiles = {}
        for depth in depths:
            profiles[depth] = mean_profile(reps[depth][:100], data)

        deltas = []
        for i in range(1, len(depths)):
            d_prev = depths[i-1]
            d_curr = depths[i]
            cos = cosine_sim(profiles[d_prev], profiles[d_curr])
            deltas.append(cos)

        if len(deltas) >= 1:
            convergence_results.append({
                "motif": motif,
                "depths": depths,
                "carriers_per_depth": {d: len(reps[d]) for d in depths},
                "cosines_between_depths": deltas,
                "converging": all(c > 0.9 for c in deltas),
            })

    print(f"\n  === TEST 4 RESULTS ===")
    print(f"  Testable motifs: {len(convergence_results)}")

    if convergence_results:
        converging = sum(1 for r in convergence_results if r["converging"])
        print(f"  Converging (all cos > 0.9): {converging}/{len(convergence_results)}")

        print(f"\n  Sample motifs:")
        for r in convergence_results[:10]:
            cos_str = ", ".join(f"{c:.3f}" for c in r["cosines_between_depths"])
            sizes = ", ".join(f"d{d}={r['carriers_per_depth'][d]}"
                              for d in r["depths"])
            print(f"    {r['motif'][:40]:40s} depths={r['depths']} "
                  f"cos=[{cos_str}] {sizes}")

        all_cosines = [c for r in convergence_results
                        for c in r["cosines_between_depths"]]
        if all_cosines:
            print(f"\n  Overall depth-transition cosines:")
            print(f"    Mean:   {np.mean(all_cosines):.4f}")
            print(f"    Median: {np.median(all_cosines):.4f}")
            print(f"    Min:    {np.min(all_cosines):.4f}")
    else:
        print("  No testable motifs found")

    print(f"  ({time.time()-t0:.1f}s)")

    return {
        "n_testable": len(convergence_results),
        "n_converging": sum(1 for r in convergence_results if r["converging"]),
        "motif_details": convergence_results[:20],
    }


def main():
    data = load_all()

    t1 = test1_evaluation_trace(data)
    t2 = test2_sequential_transform(data)
    t3 = test3_confluence(data)
    t4 = test4_fixed_point(data)

    output = {
        "test1_evaluation_trace": t1,
        "test2_sequential_transform": t2,
        "test3_confluence": t3,
        "test4_fixed_point": t4,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*72}")
    print(f"  COMPUTATION TEST SUMMARY")
    print(f"{'='*72}")

    if "frac_inner" in t1:
        trace_label = "tree-structured" if t1.get("tree_structured") else "diffuse"
        print(f"  Evaluation Trace:    {t1['frac_inner']:.1%} inner-first ({trace_label})")
    elif "depth_means" in t1:
        print(f"  Eval Trace (depth):  {t1['depth_means']}")

    if t2.get("n_tested", 0) > 0:
        print(f"  Seq. Transform:      best={t2.get('best_model','?')} "
              f"d={t2.get('composition_vs_baseline_d', 0):+.3f}")

    print(f"  Confluence:          artifacts={t3['artifact_pairs']}, "
          f"genuine_d={t3['cohens_d']:+.3f}, "
          f"Church-Rosser={'YES' if t3['genuine_confluence'] else 'NO'}")

    if t4.get("n_testable", 0) > 0:
        print(f"  Fixed-Point:         {t4['n_converging']}/{t4['n_testable']} "
              f"motifs converge")

    print(f"{'='*72}")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
