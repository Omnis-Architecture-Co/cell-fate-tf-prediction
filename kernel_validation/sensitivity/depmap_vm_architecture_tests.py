#!/usr/bin/env python3
"""
DepMap Essentiality — VM Architecture Layer Tests
==================================================
Test 1: Convergence profile as continuous feature vector
Test 2: Convergence-derived (GO-free) department assignments vs DepMap
Test 3: Dispatch layer position (hub vs endpoint) vs essentiality

These use features entirely derived from the kernel's own architecture —
no GO, no external annotation — just the genome's computational organization.

Output: validation/sensitivity/depmap_vm_architecture_results.json
"""

import csv
import json
import math
import os
import random
import statistics
import time
from collections import defaultdict, Counter

import numpy as np

random.seed(42)
np.random.seed(42)

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "depmap_vm_architecture_results.json")
DEPMAP_CACHE = "/tmp/depmap_gene_essentiality.csv"


def load_depmap():
    depmap = {}
    with open(DEPMAP_CACHE) as f:
        for row in csv.DictReader(f):
            depmap[row["gene"]] = float(row["mean_chronos"])
    return depmap


def compute_eta_squared(groups):
    all_vals = []
    for v in groups.values():
        all_vals.extend(v)
    if len(all_vals) < 2:
        return 0.0
    grand_mean = sum(all_vals) / len(all_vals)
    ss_between = sum(len(v) * (sum(v) / len(v) - grand_mean) ** 2
                     for v in groups.values() if len(v) >= 1)
    ss_total = sum((x - grand_mean) ** 2 for x in all_vals)
    return ss_between / ss_total if ss_total > 0 else 0.0


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    return (m1 - m2) / pooled_std if pooled_std > 0 else 0.0


def auroc(scores, labels):
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp = fp = 0
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    auc = 0.0
    for score, label in paired:
        if label:
            tp += 1
        else:
            fp += 1
            auc += tp
    return auc / (n_pos * n_neg)


def get_db_connection():
    import psycopg2
    return psycopg2.connect(os.environ['BETA_DATABASE_URL'])


def main():
    t0 = time.time()
    print("=" * 70)
    print("  DEPMAP — VM ARCHITECTURE LAYER TESTS")
    print("=" * 70)

    depmap = load_depmap()
    print(f"\n  DepMap genes: {len(depmap)}")

    results = {"test_suite": "VM Architecture Layer Tests"}

    conn = get_db_connection()
    cur = conn.cursor()

    # ==================================================================
    # TEST 1: CONVERGENCE PROFILE AS FEATURE VECTOR
    # ==================================================================
    print("\n" + "=" * 70)
    print("  TEST 1: CONVERGENCE PROFILE AS CONTINUOUS FEATURE VECTOR")
    print("=" * 70)

    print("  Loading convergence profiles from protein_convergence_map...")
    cur.execute("""
        SELECT gene_name, primary_function, count(*), sum(avg_hits)
        FROM protein_convergence_map
        WHERE primary_function != 'Unclassified'
        GROUP BY gene_name, primary_function
    """)

    gene_profiles_raw = defaultdict(dict)
    for gene, func, cnt, total_hits in cur.fetchall():
        gene_profiles_raw[gene][func] = float(total_hits)

    all_functions = sorted(set(f for p in gene_profiles_raw.values() for f in p))
    func_idx = {f: i for i, f in enumerate(all_functions)}
    n_funcs = len(all_functions)
    print(f"  Functions in convergence profiles: {n_funcs}")
    print(f"  Genes with convergence profiles: {len(gene_profiles_raw)}")

    genes_with_profile_and_depmap = sorted(set(gene_profiles_raw.keys()) & set(depmap.keys()))
    n = len(genes_with_profile_and_depmap)
    print(f"  Genes with profile + DepMap: {n}")

    profile_matrix = np.zeros((n, n_funcs))
    chronos_arr = np.zeros(n)
    for i, g in enumerate(genes_with_profile_and_depmap):
        for func, hits in gene_profiles_raw[g].items():
            profile_matrix[i, func_idx[func]] = hits
        chronos_arr[i] = depmap[g]

    row_sums = profile_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    profile_norm = profile_matrix / row_sums

    essential_mask = chronos_arr < -0.5
    n_essential = essential_mask.sum()
    n_non_essential = n - n_essential
    print(f"  Essential genes: {n_essential} ({n_essential/n*100:.1f}%)")

    essential_centroid = profile_norm[essential_mask].mean(axis=0)
    non_essential_centroid = profile_norm[~essential_mask].mean(axis=0)

    cosine_to_essential = np.zeros(n)
    norm_ess = np.linalg.norm(essential_centroid)
    for i in range(n):
        norm_i = np.linalg.norm(profile_norm[i])
        if norm_i > 0 and norm_ess > 0:
            cosine_to_essential[i] = np.dot(profile_norm[i], essential_centroid) / (norm_i * norm_ess)

    profile_auroc = auroc(cosine_to_essential.tolist(), essential_mask.astype(int).tolist())

    pearson_r = np.corrcoef(cosine_to_essential, chronos_arr)[0, 1]

    print(f"  AUROC (cosine to essential centroid): {profile_auroc:.4f}")
    print(f"  Pearson r (cosine similarity vs Chronos): {pearson_r:.4f}")

    func_correlations = {}
    for f in all_functions:
        fi = func_idx[f]
        if profile_norm[:, fi].std() > 0:
            r = np.corrcoef(profile_norm[:, fi], chronos_arr)[0, 1]
            func_correlations[f] = round(float(r), 4)

    print(f"\n  Function-Chronos correlations (negative = more essential):")
    for f in sorted(func_correlations, key=lambda x: func_correlations[x]):
        print(f"    {f:<22} r={func_correlations[f]:+.4f}")

    n_perm = 1000
    null_aurocs = []
    for p in range(n_perm):
        perm_labels = np.random.permutation(essential_mask.astype(int))
        null_aurocs.append(auroc(cosine_to_essential.tolist(), perm_labels.tolist()))

    z_auroc = (profile_auroc - np.mean(null_aurocs)) / np.std(null_aurocs) if np.std(null_aurocs) > 0 else 0
    p_auroc = sum(1 for na in null_aurocs if na >= profile_auroc) / n_perm

    print(f"\n  Permutation test (1000 iterations):")
    print(f"    Null AUROC: {np.mean(null_aurocs):.4f} +/- {np.std(null_aurocs):.4f}")
    print(f"    Z-score: {z_auroc:.2f}")
    print(f"    p-value: {p_auroc:.4f}")

    results["test1_convergence_profile"] = {
        "n_genes": n,
        "n_functions": n_funcs,
        "n_essential": int(n_essential),
        "auroc_cosine_to_essential": round(float(profile_auroc), 4),
        "pearson_r_cosine_vs_chronos": round(float(pearson_r), 4),
        "function_correlations": func_correlations,
        "permutation_z": round(float(z_auroc), 2),
        "permutation_p": round(float(p_auroc), 4),
        "null_auroc_mean": round(float(np.mean(null_aurocs)), 4),
    }

    # ==================================================================
    # TEST 2: CONVERGENCE-DERIVED DEPARTMENTS (GO-FREE)
    # ==================================================================
    print("\n" + "=" * 70)
    print("  TEST 2: CONVERGENCE-DERIVED DEPARTMENTS (GO-FREE)")
    print("=" * 70)

    cur.execute("""
        SELECT gene_name, primary_department, all_departments, confidence
        FROM gene_department_map
        WHERE source = 'omnis_convergence'
    """)

    conv_depts = {}
    conv_all_depts = {}
    conv_confidence = {}
    for gene, primary, all_d, conf in cur.fetchall():
        conv_depts[gene] = primary
        conv_all_depts[gene] = all_d
        conv_confidence[gene] = float(conf) if conf else 0

    print(f"  Convergence-classified genes: {len(conv_depts)}")
    conv_overlap = set(conv_depts.keys()) & set(depmap.keys())
    print(f"  With DepMap scores: {len(conv_overlap)}")

    conv_groups = defaultdict(list)
    for g in conv_overlap:
        conv_groups[conv_depts[g]].append(depmap[g])
    conv_groups_filt = {d: v for d, v in conv_groups.items() if len(v) >= 10}

    eta2_conv = compute_eta_squared(conv_groups_filt)
    n_conv_used = sum(len(v) for v in conv_groups_filt.values())
    print(f"  Departments with >= 10 genes: {len(conv_groups_filt)}")
    print(f"  Genes used: {n_conv_used}")
    print(f"  Eta-squared (convergence-only): {eta2_conv:.4f} ({eta2_conv*100:.1f}%)")

    print(f"\n  Department breakdown:")
    for d in sorted(conv_groups_filt, key=lambda x: np.mean(conv_groups_filt[x])):
        v = conv_groups_filt[d]
        pct_ess = sum(1 for s in v if s < -0.5) / len(v) * 100
        print(f"    {d:<22} n={len(v):>5}  mean={np.mean(v):+.4f}  %ess={pct_ess:.1f}%")

    scores_conv = np.array([depmap[g] for g in conv_overlap if conv_depts[g] in conv_groups_filt])
    labels_conv_list = [conv_depts[g] for g in conv_overlap if conv_depts[g] in conv_groups_filt]
    unique_labels = sorted(set(labels_conv_list))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    labels_conv = np.array([label_map[l] for l in labels_conv_list])

    n_perm_conv = 10000
    null_eta2s = []
    print(f"\n  Running {n_perm_conv} permutations...")
    for p in range(n_perm_conv):
        perm_labels = np.random.permutation(labels_conv)
        perm_groups = defaultdict(list)
        for i, l in enumerate(perm_labels):
            perm_groups[l].append(scores_conv[i])
        perm_groups_f = {l: v for l, v in perm_groups.items() if len(v) >= 10}
        null_eta2s.append(compute_eta_squared(perm_groups_f))
        if (p + 1) % 2000 == 0:
            print(f"    {p+1}/{n_perm_conv}...")

    null_mean = np.mean(null_eta2s)
    null_std = np.std(null_eta2s)
    z_conv = (eta2_conv - null_mean) / null_std if null_std > 0 else float('inf')
    p_conv = sum(1 for ne in null_eta2s if ne >= eta2_conv) / n_perm_conv

    print(f"\n  Permutation results:")
    print(f"    Observed eta2: {eta2_conv:.4f}")
    print(f"    Null: {null_mean:.4f} +/- {null_std:.4f}")
    print(f"    Z-score: {z_conv:.1f}")
    print(f"    p-value: {p_conv}" if p_conv > 0 else f"    p-value: < {1/n_perm_conv}")

    cur.execute("""
        SELECT gene_name, primary_department
        FROM gene_department_map
        WHERE source IN ('api', 'heuristic')
    """)
    go_depts = {}
    for gene, dept in cur.fetchall():
        go_depts[gene] = dept

    go_overlap = set(go_depts.keys()) & set(depmap.keys())
    go_groups = defaultdict(list)
    for g in go_overlap:
        go_groups[go_depts[g]].append(depmap[g])
    go_groups_filt = {d: v for d, v in go_groups.items() if len(v) >= 10}
    eta2_go = compute_eta_squared(go_groups_filt)

    print(f"\n  Comparison with GO-derived departments:")
    print(f"    GO-derived: eta2={eta2_go:.4f} ({len(go_groups_filt)} depts, {sum(len(v) for v in go_groups_filt.values())} genes)")
    print(f"    Convergence-only: eta2={eta2_conv:.4f} ({len(conv_groups_filt)} depts, {n_conv_used} genes)")
    if eta2_go > 0:
        print(f"    Convergence captures {eta2_conv/eta2_go*100:.1f}% of GO-derived signal")

    results["test2_convergence_departments"] = {
        "n_convergence_genes": len(conv_depts),
        "n_with_depmap": len(conv_overlap),
        "n_depts": len(conv_groups_filt),
        "n_genes_used": n_conv_used,
        "eta2_convergence": round(eta2_conv, 4),
        "permutation_null_mean": round(float(null_mean), 4),
        "permutation_null_std": round(float(null_std), 4),
        "permutation_z": round(float(z_conv), 1),
        "permutation_p": float(p_conv),
        "n_permutations": n_perm_conv,
        "eta2_go_derived": round(eta2_go, 4),
        "ratio_conv_over_go": round(eta2_conv / eta2_go, 3) if eta2_go > 0 else None,
        "dept_breakdown": {
            d: {"n": len(v), "mean_chronos": round(float(np.mean(v)), 4),
                "pct_essential": round(sum(1 for s in v if s < -0.5) / len(v) * 100, 1)}
            for d, v in conv_groups_filt.items()
        },
    }

    # ==================================================================
    # TEST 3: DISPATCH LAYER POSITION
    # ==================================================================
    print("\n" + "=" * 70)
    print("  TEST 3: DISPATCH LAYER POSITION VS ESSENTIALITY")
    print("=" * 70)

    cur.execute("""
        SELECT hop_depth, nearest_gene, count(DISTINCT entry_point) as n_entries
        FROM execution_trace_gene_map
        GROUP BY hop_depth, nearest_gene
    """)

    gene_layers = defaultdict(lambda: {"hops": set(), "n_entries": 0})
    for hop, gene, n_entries in cur.fetchall():
        gene_layers[gene]["hops"].add(hop)
        gene_layers[gene]["n_entries"] += n_entries

    hop1_genes = {g for g, info in gene_layers.items() if 1 in info["hops"]}
    hop2_genes = {g for g, info in gene_layers.items() if 2 in info["hops"]} - hop1_genes
    not_dispatch = set(depmap.keys()) - hop1_genes - hop2_genes

    hop1_depmap = [depmap[g] for g in hop1_genes if g in depmap]
    hop2_depmap = [depmap[g] for g in hop2_genes if g in depmap]
    other_depmap = [depmap[g] for g in not_dispatch if g in depmap]

    print(f"  Hop-1 relay hub genes: {len(hop1_genes)} ({len(hop1_depmap)} in DepMap)")
    print(f"  Hop-2 target genes: {len(hop2_genes)} ({len(hop2_depmap)} in DepMap)")
    print(f"  Non-dispatch genes: {len(not_dispatch)} ({len(other_depmap)} in DepMap)")

    if hop1_depmap and hop2_depmap and other_depmap:
        print(f"\n  Mean Chronos by dispatch layer:")
        print(f"    Hop-1 (relay hubs):     {np.mean(hop1_depmap):+.4f}  "
              f"%ess={sum(1 for s in hop1_depmap if s < -0.5)/len(hop1_depmap)*100:.1f}%  "
              f"median={np.median(hop1_depmap):+.4f}")
        print(f"    Hop-2 (targets):        {np.mean(hop2_depmap):+.4f}  "
              f"%ess={sum(1 for s in hop2_depmap if s < -0.5)/len(hop2_depmap)*100:.1f}%  "
              f"median={np.median(hop2_depmap):+.4f}")
        print(f"    Non-dispatch:           {np.mean(other_depmap):+.4f}  "
              f"%ess={sum(1 for s in other_depmap if s < -0.5)/len(other_depmap)*100:.1f}%  "
              f"median={np.median(other_depmap):+.4f}")

        d_hub_vs_other = cohens_d(hop1_depmap, other_depmap)
        d_hub_vs_target = cohens_d(hop1_depmap, hop2_depmap)
        d_target_vs_other = cohens_d(hop2_depmap, other_depmap)

        print(f"\n  Cohen's d effect sizes:")
        print(f"    Hub vs non-dispatch: d={d_hub_vs_other:+.3f}")
        print(f"    Hub vs target: d={d_hub_vs_target:+.3f}")
        print(f"    Target vs non-dispatch: d={d_target_vs_other:+.3f}")

        layer_groups = {"hop1_relay": hop1_depmap, "hop2_target": hop2_depmap}
        eta2_dispatch = compute_eta_squared(layer_groups)

        layer_groups_3 = {"hop1_relay": hop1_depmap, "hop2_target": hop2_depmap, "non_dispatch": other_depmap}
        eta2_dispatch_3 = compute_eta_squared(layer_groups_3)

        print(f"\n  Eta-squared (hub vs target only): {eta2_dispatch:.4f}")
        print(f"  Eta-squared (3-way: hub/target/other): {eta2_dispatch_3:.4f}")

        hub_connectivity = {}
        cur.execute("""
            SELECT nearest_gene, count(DISTINCT entry_point) as n_entries,
                   count(*) as n_traces
            FROM execution_trace_gene_map
            WHERE hop_depth = 1
            GROUP BY nearest_gene
            ORDER BY n_entries DESC
            LIMIT 20
        """)
        top_hubs = []
        for gene, n_entries, n_traces in cur.fetchall():
            if gene in depmap:
                top_hubs.append({
                    "gene": gene,
                    "n_dispatch_entries": n_entries,
                    "chronos": round(depmap[gene], 4),
                    "essential": depmap[gene] < -0.5,
                })

        if top_hubs:
            print(f"\n  Top-20 most connected relay hubs:")
            for h in top_hubs:
                ess_str = "ESSENTIAL" if h["essential"] else ""
                print(f"    {h['gene']:<15} entries={h['n_dispatch_entries']:>3}  "
                      f"chronos={h['chronos']:+.4f}  {ess_str}")

        n_perm_dispatch = 10000
        all_dispatch_scores = hop1_depmap + hop2_depmap
        all_dispatch_labels = [0] * len(hop1_depmap) + [1] * len(hop2_depmap)
        null_eta2_dispatch = []
        for p in range(n_perm_dispatch):
            perm_labels = list(all_dispatch_labels)
            random.shuffle(perm_labels)
            g0 = [all_dispatch_scores[i] for i in range(len(all_dispatch_scores)) if perm_labels[i] == 0]
            g1 = [all_dispatch_scores[i] for i in range(len(all_dispatch_scores)) if perm_labels[i] == 1]
            null_eta2_dispatch.append(compute_eta_squared({"a": g0, "b": g1}))

        null_d_mean = np.mean(null_eta2_dispatch)
        null_d_std = np.std(null_eta2_dispatch)
        z_d = (eta2_dispatch - null_d_mean) / null_d_std if null_d_std > 0 else 0
        p_d = sum(1 for ne in null_eta2_dispatch if ne >= eta2_dispatch) / n_perm_dispatch

        print(f"\n  Permutation test (hub vs target, {n_perm_dispatch} iterations):")
        print(f"    Null: {null_d_mean:.4f} +/- {null_d_std:.4f}")
        print(f"    Z-score: {z_d:.2f}")
        print(f"    p-value: {p_d:.4f}" if p_d > 0 else f"    p-value: < {1/n_perm_dispatch}")

        results["test3_dispatch_layers"] = {
            "hop1_relay": {
                "n_genes": len(hop1_depmap),
                "mean_chronos": round(float(np.mean(hop1_depmap)), 4),
                "median_chronos": round(float(np.median(hop1_depmap)), 4),
                "pct_essential": round(sum(1 for s in hop1_depmap if s < -0.5) / len(hop1_depmap) * 100, 1),
            },
            "hop2_target": {
                "n_genes": len(hop2_depmap),
                "mean_chronos": round(float(np.mean(hop2_depmap)), 4),
                "median_chronos": round(float(np.median(hop2_depmap)), 4),
                "pct_essential": round(sum(1 for s in hop2_depmap if s < -0.5) / len(hop2_depmap) * 100, 1),
            },
            "non_dispatch": {
                "n_genes": len(other_depmap),
                "mean_chronos": round(float(np.mean(other_depmap)), 4),
                "pct_essential": round(sum(1 for s in other_depmap if s < -0.5) / len(other_depmap) * 100, 1),
            },
            "cohens_d_hub_vs_other": round(float(d_hub_vs_other), 3),
            "cohens_d_hub_vs_target": round(float(d_hub_vs_target), 3),
            "eta2_hub_vs_target": round(float(eta2_dispatch), 4),
            "eta2_3way": round(float(eta2_dispatch_3), 4),
            "permutation_z": round(float(z_d), 2),
            "permutation_p": round(float(p_d), 4),
            "top_hubs": top_hubs[:10],
        }
    else:
        print("  SKIPPED: insufficient dispatch gene data")
        results["test3_dispatch_layers"] = {"status": "skipped"}

    conn.close()

    # ==================================================================
    # SUMMARY
    # ==================================================================
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("  SUMMARY — VM ARCHITECTURE LAYER TESTS")
    print("=" * 70)

    t1 = results.get("test1_convergence_profile", {})
    t2 = results.get("test2_convergence_departments", {})
    t3 = results.get("test3_dispatch_layers", {})

    print(f"\n  Test 1 — Convergence profile AUROC:")
    print(f"    AUROC: {t1.get('auroc_cosine_to_essential', 'N/A')}")
    print(f"    Pearson r: {t1.get('pearson_r_cosine_vs_chronos', 'N/A')}")
    print(f"    Z-score: {t1.get('permutation_z', 'N/A')}, p={t1.get('permutation_p', 'N/A')}")

    print(f"\n  Test 2 — Convergence departments (GO-free):")
    print(f"    Eta2: {t2.get('eta2_convergence', 'N/A')} ({t2.get('n_depts', 'N/A')} depts, {t2.get('n_genes_used', 'N/A')} genes)")
    print(f"    Z-score: {t2.get('permutation_z', 'N/A')}, p={t2.get('permutation_p', 'N/A')}")
    if t2.get('ratio_conv_over_go'):
        print(f"    Captures {t2['ratio_conv_over_go']*100:.1f}% of GO-derived signal")

    print(f"\n  Test 3 — Dispatch layer position:")
    if t3.get("hop1_relay"):
        print(f"    Hub genes: n={t3['hop1_relay']['n_genes']}, mean={t3['hop1_relay']['mean_chronos']}, %ess={t3['hop1_relay']['pct_essential']}%")
        print(f"    Target genes: n={t3['hop2_target']['n_genes']}, mean={t3['hop2_target']['mean_chronos']}, %ess={t3['hop2_target']['pct_essential']}%")
        print(f"    Cohen's d (hub vs other): {t3.get('cohens_d_hub_vs_other', 'N/A')}")
        print(f"    Z-score: {t3.get('permutation_z', 'N/A')}, p={t3.get('permutation_p', 'N/A')}")

    results["elapsed_seconds"] = round(elapsed, 1)

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {OUTPUT}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
