#!/usr/bin/env python3
"""
PL Theory + Physicist Experiments — Comprehensive Tests
========================================================

PART A: PL Theorist's Datalog Validation Tests
  Test 1: Monotonicity (does adding facts only add conclusions?)
  Test 2: Minimal Model (unique fixed point from any starting condition?)
  Test 3: Bottom-Up Evaluation (derivation follows depth layers?)
  Test 4: Rule Identification (can we extract discrete Datalog rules?)
  Test 5: Herbrand Base (discrete vs continuous domain?)

PART B: Physicist's Missing Structure Experiments
  Experiment 1: PC1-subtracted algebra (separate "how much" from "what kind")
  Experiment 2: Directed graph extension (add KEGG/Reactome directionality)
  Experiment 3: Conditional primitives (context-dependent composition)

Usage:
    python3 -u validation/knockout/pl_theory_and_physics_tests.py
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
PROFILES_PATH = "validation/knockout/disruption_profiles_full.json"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "pl_theory_physics_results.json")

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
    with open(STATE_PATH, "rb") as f:
        return pickle.load(f)


def load_profiles():
    with open(PROFILES_PATH) as f:
        data = json.load(f)
    profiles = {}
    depts = data.get("departments", VALID_DEPARTMENTS)
    for gene, p in data["profiles"].items():
        profiles[gene] = np.array([p[d] for d in depts])
    return profiles, depts


def load_vocab():
    vocab_dept = {}
    with open(VOCAB_PATH) as f:
        for row in csv.DictReader(f):
            tok = row["word_hex"].replace("0x", "").upper()
            vocab_dept[tok] = row["primary_function"]
    return vocab_dept


def load_primitives():
    prims = {}
    with open(PRIMITIVES_PATH) as f:
        for row in csv.DictReader(f):
            seq = row.get("function_sequence", "")
            recurrence = int(row.get("recurrence", 0))
            prims[seq] = {"recurrence": recurrence}
    return prims


def build_primitive_carriers(state):
    ptt = state["ptt"]
    gene_cache = state["gene_cache"]
    vocab_dept = load_vocab()

    protein_dept_seqs = {}
    for uid, tokens in ptt.items():
        depts = []
        for tok in tokens:
            d = vocab_dept.get(tok.upper())
            if d and d in DEPT_TO_IDX:
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
        ds = [d for d in p["function_sequence"].split("|") if d in DEPT_TO_IDX]
        if not ds:
            continue
        search = "|".join(ds)
        carriers = [uid for uid, seq in protein_dept_seqs.items() if search in seq]
        if len(carriers) >= 20:
            primitives.append({"search": search, "carriers": carriers})

    gene_to_uids = defaultdict(list)
    for uid, g in gene_cache.items():
        if g:
            gene_to_uids[g].append(uid)

    prim_to_genes = {}
    for p in primitives:
        genes = set()
        for uid in p["carriers"]:
            g = gene_cache.get(uid)
            if g:
                genes.add(g)
        prim_to_genes[p["search"]] = list(genes)

    return prim_to_genes, protein_dept_seqs


def load_gene_depts():
    gene_dept = {}
    with open(GENE_DEPTS_PATH) as f:
        for row in csv.DictReader(f):
            gene_dept[row["gene"]] = row["department"]
    return gene_dept


def load_nesting():
    nesting = []
    if not os.path.exists(NESTING_PATH):
        return nesting
    with open(NESTING_PATH) as f:
        for row in csv.DictReader(f):
            nesting.append(row)
    return nesting


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ============================================================
# PART A: PL THEORY TESTS
# ============================================================

def test_monotonicity(state, profiles):
    """
    Datalog Test 1: Monotonicity
    In positive Datalog, T_P is monotone: if I ⊆ J then T_P(I) ⊆ T_P(J).
    Adding facts should only add conclusions, never retract them.

    Test: Take gene sets S1 ⊂ S2. Does the aggregated profile of S2
    dominate S1 component-wise? Or does averaging cause dilution?
    """
    print("\n" + "=" * 72)
    print("PL TEST 1: MONOTONICITY")
    print("=" * 72)

    ptt = state["ptt"]
    gene_cache = state["gene_cache"]
    vocab_dept = load_vocab()

    gene_to_uids = defaultdict(list)
    for uid, g in gene_cache.items():
        if g:
            gene_to_uids[g].append(uid)

    genes_with_profiles = [g for g in profiles if g in gene_to_uids]
    random.seed(42)

    n_trials = 500
    monotone_additive = 0
    monotone_average = 0
    violations_additive = []
    violations_average = []

    for trial in range(n_trials):
        n_s1 = random.randint(5, 20)
        n_extra = random.randint(3, 15)
        s1_genes = random.sample(genes_with_profiles, min(n_s1, len(genes_with_profiles)))
        remaining = [g for g in genes_with_profiles if g not in s1_genes]
        extra_genes = random.sample(remaining, min(n_extra, len(remaining)))
        s2_genes = s1_genes + extra_genes

        p1_add = np.sum([profiles[g] for g in s1_genes], axis=0)
        p2_add = np.sum([profiles[g] for g in s2_genes], axis=0)

        p1_avg = np.mean([profiles[g] for g in s1_genes], axis=0)
        p2_avg = np.mean([profiles[g] for g in s2_genes], axis=0)

        if np.all(p2_add >= p1_add - 1e-12):
            monotone_additive += 1
        else:
            n_violated = np.sum(p2_add < p1_add - 1e-12)
            violations_additive.append(n_violated)

        if np.all(p2_avg >= p1_avg - 1e-12):
            monotone_average += 1
        else:
            n_violated = np.sum(p2_avg < p1_avg - 1e-12)
            violations_average.append(n_violated)

    print(f"  Trials: {n_trials}")
    print(f"  ADDITIVE composition:")
    print(f"    Monotone: {monotone_additive}/{n_trials} ({monotone_additive/n_trials:.1%})")
    if violations_additive:
        print(f"    Mean violated dims: {np.mean(violations_additive):.1f}")
    print(f"  AVERAGE composition:")
    print(f"    Monotone: {monotone_average}/{n_trials} ({monotone_average/n_trials:.1%})")
    if violations_average:
        print(f"    Mean violated dims: {np.mean(violations_average):.1f}")

    result = {
        "n_trials": n_trials,
        "additive_monotone_frac": round(monotone_additive / n_trials, 4),
        "average_monotone_frac": round(monotone_average / n_trials, 4),
        "additive_violations_mean_dims": round(float(np.mean(violations_additive)), 2) if violations_additive else 0,
        "average_violations_mean_dims": round(float(np.mean(violations_average)), 2) if violations_average else 0,
        "verdict": "MONOTONE" if monotone_additive / n_trials > 0.95 else "NOT MONOTONE",
    }

    if monotone_additive / n_trials > 0.95:
        print(f"\n  VERDICT: System IS monotone under additive composition")
        print(f"  → Consistent with positive Datalog")
    elif monotone_average / n_trials < 0.5:
        print(f"\n  VERDICT: System is NOT monotone under averaging")
        print(f"  → Rules out positive Datalog if averaging is the composition")
        print(f"  → But additive composition may still be monotone")
    else:
        print(f"\n  VERDICT: Mixed — depends on composition model")

    return result


def test_minimal_model(profiles, prim_to_genes):
    """
    Datalog Test 2: Minimal Model / Unique Fixed Point
    Datalog computes the unique minimal model. Starting from the same
    facts, you should always reach the same conclusion.

    Test: Bootstrap-resample carrier genes for each primitive,
    compute the signature from different random subsets. Do they converge?
    """
    print("\n" + "=" * 72)
    print("PL TEST 2: MINIMAL MODEL (Unique Fixed Point)")
    print("=" * 72)

    random.seed(42)

    n_bootstraps = 50
    tested = 0
    convergence_scores = []

    for seq, genes in prim_to_genes.items():
        carrier_profiles = [profiles[g] for g in genes if g in profiles]
        if len(carrier_profiles) < 20:
            continue

        tested += 1
        bootstrap_signatures = []
        for _ in range(n_bootstraps):
            sample = random.choices(carrier_profiles, k=len(carrier_profiles))
            sig = np.mean(sample, axis=0)
            bootstrap_signatures.append(sig)

        pairwise_cos = []
        for i in range(n_bootstraps):
            for j in range(i + 1, n_bootstraps):
                pairwise_cos.append(cosine_sim(bootstrap_signatures[i], bootstrap_signatures[j]))

        convergence_scores.append(np.mean(pairwise_cos))

    mean_convergence = float(np.mean(convergence_scores))
    min_convergence = float(np.min(convergence_scores))

    print(f"  Primitives tested: {tested}")
    print(f"  Bootstrap samples per primitive: {n_bootstraps}")
    print(f"  Mean pairwise cosine across bootstraps: {mean_convergence:.6f}")
    print(f"  Min pairwise cosine: {min_convergence:.6f}")
    print(f"  Fraction > 0.99: {np.mean(np.array(convergence_scores) > 0.99):.1%}")
    print(f"  Fraction > 0.999: {np.mean(np.array(convergence_scores) > 0.999):.1%}")

    result = {
        "n_primitives_tested": tested,
        "n_bootstraps": n_bootstraps,
        "mean_convergence_cosine": round(mean_convergence, 6),
        "min_convergence_cosine": round(min_convergence, 6),
        "frac_gt_0.99": round(float(np.mean(np.array(convergence_scores) > 0.99)), 4),
        "frac_gt_0.999": round(float(np.mean(np.array(convergence_scores) > 0.999)), 4),
        "verdict": "UNIQUE FIXED POINT" if mean_convergence > 0.995 else "MULTIPLE ATTRACTORS",
    }

    if mean_convergence > 0.995:
        print(f"\n  VERDICT: System converges to a UNIQUE fixed point")
        print(f"  → Consistent with Datalog's minimal model property")
    else:
        print(f"\n  VERDICT: Multiple attractors possible")
        print(f"  → Inconsistent with standard Datalog")

    return result


def test_bottom_up_evaluation(state, profiles, protein_dept_seqs):
    """
    Datalog Test 3: Bottom-Up Evaluation
    In Datalog, facts at depth k are derived ONLY from depth k-1.
    Information flows strictly upward.

    Test: Using the nesting hierarchy, check whether inner primitive
    signatures predict outer primitive behavior, or whether influence
    flows bidirectionally.
    """
    print("\n" + "=" * 72)
    print("PL TEST 3: BOTTOM-UP EVALUATION")
    print("=" * 72)

    nesting = load_nesting()
    if not nesting:
        print("  No nesting hierarchy file found — skipping")
        return {"skipped": True, "reason": "no nesting hierarchy file"}

    gene_cache = state["gene_cache"]

    parent_children = defaultdict(set)
    child_parents = defaultdict(set)
    all_prims = set()
    for row in nesting:
        parent = row.get("outer_sequence", "")
        child = row.get("inner_sequence", "")
        if parent and child:
            parent_children[parent].add(child)
            child_parents[child].add(parent)
            all_prims.add(parent)
            all_prims.add(child)

    roots = [p for p in all_prims if p not in child_parents]
    leaves = [p for p in all_prims if p not in parent_children]

    depths = {}
    def get_depth(prim):
        if prim in depths:
            return depths[prim]
        if prim not in child_parents:
            depths[prim] = 0
            return 0
        depths[prim] = -1
        max_d = 0
        for p in child_parents[prim]:
            d = get_depth(p)
            if d >= 0:
                max_d = max(max_d, d + 1)
        depths[prim] = max_d
        return max_d

    for p in all_prims:
        if p not in depths:
            get_depth(p)
    max_depth = max(depths.values()) if depths else 0

    print(f"  Nesting relations: {len(nesting)}")
    print(f"  Unique primitives: {len(all_prims)}")
    print(f"  Roots (no parents): {len(roots)}")
    print(f"  Leaves (no children): {len(leaves)}")
    print(f"  Max depth: {max_depth}")

    testable_prims = set()
    for row in nesting:
        parent = row.get("outer_sequence", "")
        child = row.get("inner_sequence", "")
        for seq in [parent, child]:
            ds = [d for d in seq.split("|") if d in DEPT_TO_IDX]
            if ds and len(ds) <= 3:
                testable_prims.add(seq)
    short_prims = testable_prims

    print(f"  Short primitives (≤4 depts) for signature computation: {len(short_prims)}")
    print(f"  Building index...")

    gene_to_pseqs = defaultdict(list)
    for uid, pseq in protein_dept_seqs.items():
        g = gene_cache.get(uid)
        if g and g in profiles:
            gene_to_pseqs[g].append(pseq)

    prim_sigs = {}
    for seq in short_prims:
        ds = [d for d in seq.split("|") if d in DEPT_TO_IDX]
        if not ds:
            continue
        search = "|".join(ds)
        carrier_genes = [g for g, pseqs in gene_to_pseqs.items() if any(search in p for p in pseqs)]
        if len(carrier_genes) >= 5:
            prim_sigs[seq] = np.mean([profiles[g] for g in carrier_genes], axis=0)

    inner_predicts_outer = []
    outer_predicts_inner = []

    for row in nesting:
        parent = row.get("parent_primitive") or row.get("outer", "")
        child = row.get("child_primitive") or row.get("inner", "")
        if parent in prim_sigs and child in prim_sigs:
            cos = cosine_sim(prim_sigs[parent], prim_sigs[child])
            inner_predicts_outer.append(cos)
            outer_predicts_inner.append(cos)

    unrelated_cos = []
    prim_list = list(prim_sigs.keys())
    random.seed(42)
    for _ in range(min(500, len(prim_list) * 5)):
        a, b = random.sample(prim_list, 2)
        if b not in parent_children.get(a, set()) and a not in parent_children.get(b, set()):
            unrelated_cos.append(cosine_sim(prim_sigs[a], prim_sigs[b]))

    related_mean = float(np.mean(inner_predicts_outer)) if inner_predicts_outer else 0
    unrelated_mean = float(np.mean(unrelated_cos)) if unrelated_cos else 0

    depth_sims = defaultdict(list)
    for row in nesting:
        parent = row.get("parent_primitive") or row.get("outer", "")
        child = row.get("child_primitive") or row.get("inner", "")
        if parent in prim_sigs and child in prim_sigs:
            d_parent = depths.get(parent, 0)
            d_child = depths.get(child, 0)
            depth_diff = d_parent - d_child
            cos = cosine_sim(prim_sigs[parent], prim_sigs[child])
            depth_sims[depth_diff].append(cos)

    print(f"\n  Parent-child cosine similarity: {related_mean:.4f}")
    print(f"  Unrelated primitive cosine: {unrelated_mean:.4f}")
    print(f"  Enrichment: {related_mean - unrelated_mean:+.4f}")

    if depth_sims:
        print(f"\n  Similarity by depth difference:")
        for dd in sorted(depth_sims.keys()):
            sims = depth_sims[dd]
            print(f"    Depth diff {dd}: mean={np.mean(sims):.4f} (n={len(sims)})")

    result = {
        "n_nesting_relations": len(nesting),
        "n_unique_primitives": len(all_prims),
        "n_roots": len(roots),
        "n_leaves": len(leaves),
        "max_depth": max_depth,
        "related_cosine_mean": round(related_mean, 4),
        "unrelated_cosine_mean": round(unrelated_mean, 4),
        "enrichment": round(related_mean - unrelated_mean, 4),
        "depth_similarities": {str(k): round(float(np.mean(v)), 4) for k, v in sorted(depth_sims.items())},
        "verdict": "BOTTOM-UP" if related_mean > unrelated_mean + 0.05 else "NO CLEAR DIRECTION",
    }

    return result


def test_rule_identification(profiles, prim_to_genes):
    """
    Datalog Test 4: Rule Identification
    Can we extract discrete rules of the form:
    "if protein carries primitives P1 and P2, it participates in department D"?
    Or is the mapping fundamentally continuous?
    """
    print("\n" + "=" * 72)
    print("PL TEST 4: RULE IDENTIFICATION (Discrete vs Continuous)")
    print("=" * 72)

    gene_depts = load_gene_depts()

    prim_to_dept_dist = {}
    for seq, genes in prim_to_genes.items():
        if len(genes) < 10:
            continue
        dept_counts = Counter()
        total = 0
        for g in genes:
            if g in gene_depts:
                dept_counts[gene_depts[g]] += 1
                total += 1
        if total > 0:
            dist = {d: dept_counts.get(d, 0) / total for d in VALID_DEPARTMENTS}
            prim_to_dept_dist[seq] = dist

    entropies = []
    max_probs = []
    for seq, dist in prim_to_dept_dist.items():
        probs = np.array([dist[d] for d in VALID_DEPARTMENTS])
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        entropies.append(entropy)
        max_probs.append(float(np.max(probs)))

    max_possible_entropy = np.log2(N_DEPTS)

    random.seed(42)
    null_entropies = []
    all_depts_list = [gene_depts[g] for g in gene_depts if gene_depts[g] in DEPT_TO_IDX]
    for _ in range(len(prim_to_dept_dist)):
        n = random.randint(20, 200)
        sample = random.choices(all_depts_list, k=n)
        counts = Counter(sample)
        total = sum(counts.values())
        probs = np.array([counts.get(d, 0) / total for d in VALID_DEPARTMENTS])
        probs = probs[probs > 0]
        null_entropies.append(-np.sum(probs * np.log2(probs)))

    rule_like = sum(1 for e in entropies if e < max_possible_entropy * 0.5)
    continuous_like = sum(1 for e in entropies if e >= max_possible_entropy * 0.5)

    print(f"  Primitives with enough carriers: {len(prim_to_dept_dist)}")
    print(f"  Max possible entropy: {max_possible_entropy:.2f} bits")
    print(f"  Mean primitive entropy: {np.mean(entropies):.2f} bits")
    print(f"  Mean null entropy: {np.mean(null_entropies):.2f} bits")
    print(f"  Max probability (most concentrated): {np.max(max_probs):.3f}")
    print(f"  Mean max probability: {np.mean(max_probs):.3f}")
    print(f"  'Rule-like' (entropy < 50% max): {rule_like}/{len(entropies)}")
    print(f"  'Continuous-like' (entropy >= 50% max): {continuous_like}/{len(entropies)}")

    d_val = (np.mean(null_entropies) - np.mean(entropies)) / np.std(null_entropies) if np.std(null_entropies) > 0 else 0

    result = {
        "n_primitives": len(prim_to_dept_dist),
        "max_possible_entropy": round(max_possible_entropy, 2),
        "mean_entropy": round(float(np.mean(entropies)), 4),
        "null_entropy": round(float(np.mean(null_entropies)), 4),
        "entropy_cohens_d": round(d_val, 4),
        "mean_max_probability": round(float(np.mean(max_probs)), 4),
        "frac_rule_like": round(rule_like / len(entropies), 4) if entropies else 0,
        "frac_continuous_like": round(continuous_like / len(entropies), 4) if entropies else 0,
        "verdict": "DISCRETE RULES" if rule_like > continuous_like else "CONTINUOUS MAPPING",
    }

    if rule_like > continuous_like:
        print(f"\n  VERDICT: Primitives map to concentrated department distributions")
        print(f"  → More rule-like than continuous; partial Datalog compatibility")
    else:
        print(f"\n  VERDICT: Primitives map to SPREAD department distributions")
        print(f"  → Continuous mapping, not discrete rules")
        print(f"  → Inconsistent with standard Datalog's ground atoms")

    return result


def test_herbrand_base(profiles):
    """
    Datalog Test 5: Herbrand Base Finiteness
    Datalog operates on finite discrete atoms. Is our system discrete or continuous?

    Test: Check whether disruption profiles cluster into discrete types
    or form a continuous manifold.
    """
    print("\n" + "=" * 72)
    print("PL TEST 5: HERBRAND BASE (Discrete vs Continuous Domain)")
    print("=" * 72)

    M = np.array(list(profiles.values()))
    M_centered = M - M.mean(axis=0)

    U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
    proj = M_centered @ Vt[:5].T

    from scipy.spatial.distance import pdist
    random.seed(42)
    sample_idx = random.sample(range(len(proj)), min(3000, len(proj)))
    sample = proj[sample_idx]
    dists = pdist(sample)

    n_unique_dists = len(set(np.round(dists, 6)))
    total_dists = len(dists)

    from scipy.stats import normaltest
    _, p_normal = normaltest(dists)

    hist, bin_edges = np.histogram(dists, bins=100)
    hist_entropy = -np.sum((hist / hist.sum()) * np.log2(hist / hist.sum() + 1e-12))
    max_hist_entropy = np.log2(100)

    n_effective_clusters = 0
    try:
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import KMeans
        sil_scores = []
        for k in [5, 10, 15, 22, 30, 50]:
            km = KMeans(n_clusters=k, random_state=42, n_init=3, max_iter=50)
            labels = km.fit_predict(sample)
            sil = silhouette_score(sample, labels, sample_size=min(1000, len(sample)))
            sil_scores.append((k, sil))
            n_effective_clusters = max(n_effective_clusters, k if sil > 0.3 else 0)
        best_k, best_sil = max(sil_scores, key=lambda x: x[1])
    except ImportError:
        best_k, best_sil = -1, -1
        sil_scores = []

    print(f"  Sampled profiles: {len(sample)}")
    print(f"  Pairwise distances: {total_dists}")
    print(f"  Unique distances (6dp): {n_unique_dists}/{total_dists}")
    print(f"  Distance distribution normality p: {p_normal:.2e}")
    print(f"  Distance histogram entropy: {hist_entropy:.2f} / {max_hist_entropy:.2f}")

    if sil_scores:
        print(f"\n  Silhouette scores by k:")
        for k, s in sil_scores:
            print(f"    k={k:3d}: silhouette={s:.4f}")
        print(f"  Best: k={best_k}, silhouette={best_sil:.4f}")

    is_discrete = best_sil > 0.5 and n_unique_dists < total_dists * 0.1

    result = {
        "n_sampled": len(sample),
        "n_unique_dists_6dp": n_unique_dists,
        "total_dists": total_dists,
        "uniqueness_ratio": round(n_unique_dists / total_dists, 4),
        "dist_normality_p": float(p_normal),
        "hist_entropy": round(hist_entropy, 4),
        "max_hist_entropy": round(max_hist_entropy, 4),
        "best_k": best_k,
        "best_silhouette": round(best_sil, 4),
        "silhouette_by_k": {str(k): round(s, 4) for k, s in sil_scores},
        "verdict": "DISCRETE" if is_discrete else "CONTINUOUS",
    }

    if is_discrete:
        print(f"\n  VERDICT: Domain is DISCRETE — profiles cluster into types")
        print(f"  → Compatible with Herbrand base")
    else:
        print(f"\n  VERDICT: Domain is CONTINUOUS — profiles form a manifold")
        print(f"  → Incompatible with standard Datalog's finite discrete atoms")
        print(f"  → Suggests lattice-based or semiring computation")

    return result


# ============================================================
# PART B: PHYSICIST'S EXPERIMENTS
# ============================================================

def experiment_pc1_subtracted(profiles, prim_to_genes):
    """
    Physicist Experiment 1: PC1-Subtracted Algebra
    Remove the dominant "how much" component. Recompute department
    prediction on the residual "what kind" dimensions.
    """
    print("\n" + "=" * 72)
    print("PHYSICS EXP 1: PC1-SUBTRACTED ALGEBRA")
    print("=" * 72)

    gene_depts = load_gene_depts()

    M = np.array(list(profiles.values()))
    gene_names = list(profiles.keys())
    M_centered = M - M.mean(axis=0)

    U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)

    pc1_component = np.outer(U[:, 0] * S[0], Vt[0])
    M_residual = M_centered - pc1_component

    cumvar_residual = np.cumsum(S[1:]**2) / np.sum(S[1:]**2)
    print(f"  Residual variance explained (after removing PC1):")
    for i in range(min(5, len(cumvar_residual))):
        print(f"    rPC{i+1} (was PC{i+2}): {S[i+1]**2/np.sum(S[1:]**2):.1%} (cumulative: {cumvar_residual[i]:.1%})")

    dept_centroids_full = {}
    dept_centroids_residual = {}
    for d in VALID_DEPARTMENTS:
        idxs = [i for i, g in enumerate(gene_names) if gene_depts.get(g) == d]
        if len(idxs) >= 5:
            dept_centroids_full[d] = np.mean(M_centered[idxs], axis=0)
            dept_centroids_residual[d] = np.mean(M_residual[idxs], axis=0)

    correct_full = 0
    correct_residual = 0
    correct_residual_top3 = 0
    correct_full_top3 = 0
    tested = 0

    for i, gene in enumerate(gene_names):
        true_dept = gene_depts.get(gene)
        if true_dept not in dept_centroids_full:
            continue
        tested += 1

        dists_full = {d: cosine_sim(M_centered[i], c) for d, c in dept_centroids_full.items()}
        pred_full = max(dists_full, key=dists_full.get)
        top3_full = sorted(dists_full, key=dists_full.get, reverse=True)[:3]
        if pred_full == true_dept:
            correct_full += 1
        if true_dept in top3_full:
            correct_full_top3 += 1

        dists_res = {d: cosine_sim(M_residual[i], c) for d, c in dept_centroids_residual.items()}
        pred_res = max(dists_res, key=dists_res.get)
        top3_res = sorted(dists_res, key=dists_res.get, reverse=True)[:3]
        if pred_res == true_dept:
            correct_residual += 1
        if true_dept in top3_res:
            correct_residual_top3 += 1

    acc_full = correct_full / tested if tested else 0
    acc_residual = correct_residual / tested if tested else 0
    acc_full_top3 = correct_full_top3 / tested if tested else 0
    acc_residual_top3 = correct_residual_top3 / tested if tested else 0
    chance = 1.0 / len(dept_centroids_full)

    print(f"\n  Genes tested: {tested}")
    print(f"  Chance: {chance:.1%}")
    print(f"\n  FULL profiles (with PC1):")
    print(f"    Top-1: {acc_full:.1%}  Top-3: {acc_full_top3:.1%}")
    print(f"  RESIDUAL profiles (PC1 removed):")
    print(f"    Top-1: {acc_residual:.1%}  Top-3: {acc_residual_top3:.1%}")
    print(f"\n  Change: Top-1 {acc_residual - acc_full:+.1%}  Top-3 {acc_residual_top3 - acc_full_top3:+.1%}")

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    prim_sigs_full = {}
    prim_sigs_residual = {}
    prim_depts_map = {}

    for seq, genes in prim_to_genes.items():
        idxs = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        if len(idxs) < 20:
            continue
        prim_sigs_full[seq] = np.mean(M_centered[idxs], axis=0)
        prim_sigs_residual[seq] = np.mean(M_residual[idxs], axis=0)
        seq_depts = seq.split("|")
        actual_depts = [d for d in seq_depts if d in DEPT_TO_IDX]
        prim_depts_map[seq] = actual_depts

    same_dept_full = []
    diff_dept_full = []
    same_dept_residual = []
    diff_dept_residual = []

    prim_list = list(prim_sigs_full.keys())
    for i in range(len(prim_list)):
        for j in range(i + 1, len(prim_list)):
            p1, p2 = prim_list[i], prim_list[j]
            d1 = set(prim_depts_map.get(p1, []))
            d2 = set(prim_depts_map.get(p2, []))
            cos_full = cosine_sim(prim_sigs_full[p1], prim_sigs_full[p2])
            cos_res = cosine_sim(prim_sigs_residual[p1], prim_sigs_residual[p2])

            if d1 & d2:
                same_dept_full.append(cos_full)
                same_dept_residual.append(cos_res)
            else:
                diff_dept_full.append(cos_full)
                diff_dept_residual.append(cos_res)

    if same_dept_full and diff_dept_full:
        d_full = (np.mean(same_dept_full) - np.mean(diff_dept_full)) / np.std(diff_dept_full)
        d_res = (np.mean(same_dept_residual) - np.mean(diff_dept_residual)) / np.std(diff_dept_residual) if np.std(diff_dept_residual) > 0 else 0

        print(f"\n  COLLINEARITY (same-dept vs diff-dept primitives):")
        print(f"    Full:     within={np.mean(same_dept_full):.4f} across={np.mean(diff_dept_full):.4f} d={d_full:.4f}")
        print(f"    Residual: within={np.mean(same_dept_residual):.4f} across={np.mean(diff_dept_residual):.4f} d={d_res:.4f}")
        print(f"    Change in d: {d_res - d_full:+.4f}")
    else:
        d_full, d_res = 0, 0

    result = {
        "genes_tested": tested,
        "chance": round(chance, 4),
        "full_top1": round(acc_full, 4),
        "full_top3": round(acc_full_top3, 4),
        "residual_top1": round(acc_residual, 4),
        "residual_top3": round(acc_residual_top3, 4),
        "top1_change": round(acc_residual - acc_full, 4),
        "top3_change": round(acc_residual_top3 - acc_full_top3, 4),
        "collinearity_d_full": round(d_full, 4),
        "collinearity_d_residual": round(d_res, 4),
        "verdict": "PC1_HIDING_SIGNAL" if acc_residual > acc_full * 1.1 else "DIRECTIONAL_INFO_WEAK" if acc_residual < acc_full * 0.5 else "MODEST_IMPROVEMENT",
    }

    return result


def experiment_conditional_primitives(state, profiles, prim_to_genes):
    """
    Physicist Experiment 3: Conditional Primitives
    Test whether primitives behave differently depending on
    the gene's connectivity context (high-degree vs low-degree nodes).

    Uses degree as the simplest available conditioning variable.
    """
    print("\n" + "=" * 72)
    print("PHYSICS EXP 3: CONDITIONAL PRIMITIVES (Context-Dependent)")
    print("=" * 72)

    ptt = state["ptt"]
    gene_cache = state["gene_cache"]

    gene_degrees = {}
    gene_to_uids = defaultdict(list)
    for uid, g in gene_cache.items():
        if g:
            gene_to_uids[g].append(uid)

    for gene, uids in gene_to_uids.items():
        total_tokens = sum(len(ptt.get(uid, [])) for uid in uids)
        gene_degrees[gene] = total_tokens

    degree_values = [gene_degrees[g] for g in profiles if g in gene_degrees]
    median_degree = np.median(degree_values)

    gene_depts = load_gene_depts()

    n_improved = 0
    n_tested = 0
    improvements = []
    unconditional_accs = []
    conditional_accs = []

    for seq, genes in prim_to_genes.items():
        carriers_with_profiles = [g for g in genes if g in profiles and g in gene_degrees and g in gene_depts]
        if len(carriers_with_profiles) < 30:
            continue

        high = [g for g in carriers_with_profiles if gene_degrees[g] >= median_degree]
        low = [g for g in carriers_with_profiles if gene_degrees[g] < median_degree]

        if len(high) < 10 or len(low) < 10:
            continue

        n_tested += 1

        sig_all = np.mean([profiles[g] for g in carriers_with_profiles], axis=0)
        sig_high = np.mean([profiles[g] for g in high], axis=0)
        sig_low = np.mean([profiles[g] for g in low], axis=0)

        correct_unconditional = 0
        correct_conditional = 0
        total = 0

        for g in carriers_with_profiles:
            true_dept = gene_depts.get(g)
            if true_dept not in DEPT_TO_IDX:
                continue
            total += 1

            pred_uncond = VALID_DEPARTMENTS[np.argmax(sig_all)]
            if gene_degrees[g] >= median_degree:
                pred_cond = VALID_DEPARTMENTS[np.argmax(sig_high)]
            else:
                pred_cond = VALID_DEPARTMENTS[np.argmax(sig_low)]

            if pred_uncond == true_dept:
                correct_unconditional += 1
            if pred_cond == true_dept:
                correct_conditional += 1

        if total > 0:
            acc_u = correct_unconditional / total
            acc_c = correct_conditional / total
            unconditional_accs.append(acc_u)
            conditional_accs.append(acc_c)
            improvement = acc_c - acc_u
            improvements.append(improvement)
            if acc_c > acc_u:
                n_improved += 1

    print(f"  Primitives tested: {n_tested}")
    print(f"  Median gene degree: {median_degree:.0f} tokens")
    print(f"  Unconditional accuracy: {np.mean(unconditional_accs):.1%}")
    print(f"  Conditional accuracy: {np.mean(conditional_accs):.1%}")
    print(f"  Improved with conditioning: {n_improved}/{n_tested} ({n_improved/n_tested:.1%})" if n_tested > 0 else "  None tested")
    print(f"  Mean improvement: {np.mean(improvements):+.1%}" if improvements else "")

    cos_between_contexts = []
    for seq, genes in prim_to_genes.items():
        carriers_with_profiles = [g for g in genes if g in profiles and g in gene_degrees]
        if len(carriers_with_profiles) < 30:
            continue
        high = [g for g in carriers_with_profiles if gene_degrees[g] >= median_degree]
        low = [g for g in carriers_with_profiles if gene_degrees[g] < median_degree]
        if len(high) < 10 or len(low) < 10:
            continue
        sig_high = np.mean([profiles[g] for g in high], axis=0)
        sig_low = np.mean([profiles[g] for g in low], axis=0)
        cos_between_contexts.append(cosine_sim(sig_high, sig_low))

    if cos_between_contexts:
        print(f"\n  Cosine between high-degree and low-degree signatures:")
        print(f"    Mean: {np.mean(cos_between_contexts):.4f}")
        print(f"    Min:  {np.min(cos_between_contexts):.4f}")
        print(f"    Fraction < 0.95: {np.mean(np.array(cos_between_contexts) < 0.95):.1%}")

    result = {
        "n_primitives_tested": n_tested,
        "median_gene_degree": float(median_degree),
        "unconditional_accuracy": round(float(np.mean(unconditional_accs)), 4) if unconditional_accs else 0,
        "conditional_accuracy": round(float(np.mean(conditional_accs)), 4) if conditional_accs else 0,
        "mean_improvement": round(float(np.mean(improvements)), 4) if improvements else 0,
        "frac_improved": round(n_improved / n_tested, 4) if n_tested > 0 else 0,
        "context_cosine_mean": round(float(np.mean(cos_between_contexts)), 4) if cos_between_contexts else 0,
        "context_cosine_min": round(float(np.min(cos_between_contexts)), 4) if cos_between_contexts else 0,
        "frac_cosine_lt_0.95": round(float(np.mean(np.array(cos_between_contexts) < 0.95)), 4) if cos_between_contexts else 0,
        "verdict": "CONTEXT_MATTERS" if (improvements and np.mean(improvements) > 0.01) else "CONTEXT_MINIMAL",
    }

    return result


def main():
    print("=" * 72)
    print("PL THEORY + PHYSICIST EXPERIMENTS")
    print("Comprehensive tests for Datalog hypothesis and missing structure")
    print("=" * 72)

    t0 = time.time()

    print("\nLoading state...")
    state = load_state()
    print(f"  Graph: {state['n_tokens']} tokens, {state['n_proteins']} proteins, {state['n_edges']} edges")

    print("Loading disruption profiles...")
    profiles, depts = load_profiles()
    print(f"  {len(profiles)} gene profiles loaded")

    print("Building primitive carrier mappings...")
    prim_to_genes, protein_dept_seqs = build_primitive_carriers(state)
    print(f"  {len(prim_to_genes)} primitives with ≥20 carriers")

    results = {}

    print("\n" + "#" * 72)
    print("# PART A: PL THEORY TESTS (Datalog Validation)")
    print("#" * 72)

    results["pl_test1_monotonicity"] = test_monotonicity(state, profiles)
    results["pl_test2_minimal_model"] = test_minimal_model(profiles, prim_to_genes)
    results["pl_test3_bottom_up"] = test_bottom_up_evaluation(state, profiles, protein_dept_seqs)
    results["pl_test4_rule_identification"] = test_rule_identification(profiles, prim_to_genes)
    results["pl_test5_herbrand_base"] = test_herbrand_base(profiles)

    print("\n" + "#" * 72)
    print("# PART B: PHYSICIST'S EXPERIMENTS")
    print("#" * 72)

    results["physics_exp1_pc1_subtracted"] = experiment_pc1_subtracted(profiles, prim_to_genes)
    results["physics_exp3_conditional"] = experiment_conditional_primitives(state, profiles, prim_to_genes)

    elapsed = time.time() - t0

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print("\nPL Theory Tests:")
    for k in ["pl_test1_monotonicity", "pl_test2_minimal_model", "pl_test3_bottom_up",
              "pl_test4_rule_identification", "pl_test5_herbrand_base"]:
        v = results[k]
        verdict = v.get("verdict", "?")
        print(f"  {k}: {verdict}")

    print("\nPhysicist Experiments:")
    for k in ["physics_exp1_pc1_subtracted", "physics_exp3_conditional"]:
        v = results[k]
        verdict = v.get("verdict", "?")
        print(f"  {k}: {verdict}")

    print(f"\nTotal runtime: {elapsed:.0f}s")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
