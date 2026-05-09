#!/usr/bin/env python3
"""
Comprehensive null tests for cocktail prediction validation.
1. Component ablation: drop each of 9 components, measure Top-5% degradation
2. Single-component baselines: each component alone vs full model
3. Random-draw permutation: 10,000 permutations vs pure chance
4. Score-shuffle permutation: shuffles gene→score mapping (1,000 perms)

Two-stage execution: stage1 precomputes raw components to disk, stage2 runs all tests.
Usage: python3 run_null_tests.py stage1   # precompute (~2min)
       python3 run_null_tests.py stage2   # run tests + figures (~30s)
       python3 run_null_tests.py          # run both
"""

import json
import os
import sys
import time
import csv
import math
import pickle
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

BASE = os.path.dirname(__file__)
ROOT = os.path.dirname(BASE)
CACHE_PATH = os.path.join(BASE, '.null_test_cache.pkl')

CSV_77 = os.path.join(ROOT, 'attached_assets', 'published_reprogramming_cocktails_1774053006993.csv')
CSV_EXT = os.path.join(ROOT, 'attached_assets', 'additional_reprogramming_cocktails_1774066435911.csv')
CSV_S3 = os.path.join(ROOT, 'attached_assets', 'validation_set3_cocktails_1774073324946.csv')

COMPONENTS = ['w_dir', 'w_act', 'w_frac', 'w_enr', 'w_gtex', 'w_tau', 'w_pheno', 'w_kern', 'w_temp']
COMPONENT_NAMES = {
    'w_dir': 'Directional connectivity',
    'w_act': 'Activation precision',
    'w_frac': 'Enrichment fraction',
    'w_enr': 'Enrichment magnitude',
    'w_gtex': 'GTEx expression',
    'w_tau': 'Tissue specificity (tau)',
    'w_pheno': 'Phenotype/disease',
    'w_kern': 'Kernel signature',
    'w_temp': 'Temporal expression',
}


def load_maps():
    from validate_77_cocktails import (
        TARGET_CELL_MAP as T1, SOURCE_CELL_MAP as S1,
        ALIAS_MAP as A1, parse_factors as pf1,
    )
    from validate_extended_cocktails import (
        TARGET_CELL_MAP as T2, SOURCE_CELL_MAP as S2,
        ALIAS_MAP as A2, parse_factors as pf2,
    )
    from validate_set3_cocktails import (
        TARGET_CELL_MAP as T3, SOURCE_CELL_MAP as S3,
        ALIAS_MAP as A3, parse_factors as pf3,
    )
    return (T1, S1, A1, pf1), (T2, S2, A2, pf2), (T3, S3, A3, pf3)


def build_transitions(csv_path, target_map, source_map, alias_map, parse_fn, source_col='source_cell'):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    transitions = []
    for row in rows:
        target_cell = row['target_cell']
        source_cell = row.get(source_col, row.get('source', ''))
        gene_factors = parse_fn(row['factors'])
        if not gene_factors:
            continue
        target_types = target_map.get(target_cell)
        source_type = source_map.get(source_cell)
        if target_types is None or source_type is None:
            continue
        transitions.append({
            'target_types': target_types,
            'source_type': source_type,
            'factors': gene_factors,
        })
    return transitions


def compute_raw_for_transition(predictor, source_type, target_types,
                                gene_pheno, gene_temporal, M_res, kernel_gene_idx):
    from vm_cocktail_predictor import CELL_GTEX_MAP
    from calibrate_weights import compute_phenotype_score, compute_temporal_score

    p = predictor
    source_markers = {}
    target_markers = {}
    for ct in [source_type]:
        if ct in p.ct_markers:
            source_markers.update(p.ct_markers[ct])
    for ct in target_types:
        if ct in p.ct_markers:
            target_markers.update(p.ct_markers[ct])

    source_programs = set()
    for m in source_markers:
        if m in p.gene_progs:
            source_programs |= p.gene_progs[m]
    target_programs = set()
    for m in target_markers:
        if m in p.gene_progs:
            target_programs |= p.gene_progs[m]
    activate = target_programs - source_programs

    target_markers_in_progs = set(m for m in target_markers if m in p.gene_progs)
    source_markers_in_progs = set(m for m in source_markers if m in p.gene_progs)
    n_markers = len(target_markers_in_progs)
    n_source_markers = len(source_markers_in_progs)

    gene_target_conn = defaultdict(int)
    for m in target_markers_in_progs:
        for pid in p.gene_progs[m]:
            seen = set()
            for gene in p.prog_genes.get(pid, ()):
                if gene not in seen:
                    seen.add(gene)
                    gene_target_conn[gene] += 1

    gene_source_conn = defaultdict(int)
    for m in source_markers_in_progs:
        for pid in p.gene_progs[m]:
            seen = set()
            for gene in p.prog_genes.get(pid, ()):
                if gene not in seen:
                    seen.add(gene)
                    gene_source_conn[gene] += 1

    prog_enr = {}
    for pid, genes in p.prog_genes.items():
        n_in = len(genes)
        n_mark = sum(1 for g in genes if g in target_markers_in_progs)
        if n_mark > 0:
            expected = n_in * n_markers / p.n_genes if p.n_genes > 0 else 0
            prog_enr[pid] = n_mark / max(expected, 1e-10)

    gtex_tissues = list(set(t for ct in target_types for t in CELL_GTEX_MAP.get(ct, [])))
    source_gtex = CELL_GTEX_MAP.get(source_type, [])
    t_indices = [p.tissue_idx[t] for t in gtex_tissues if t in p.tissue_idx]
    s_indices = [p.tissue_idx[t] for t in source_gtex if t in p.tissue_idx]

    target_marker_genes = set(target_markers.keys())
    kernel_sig = None
    if M_res is not None and kernel_gene_idx is not None:
        marker_indices_k = [kernel_gene_idx[m] for m in target_marker_genes if m in kernel_gene_idx]
        if marker_indices_k:
            marker_profiles_k = M_res[marker_indices_k]
            kernel_sig = np.mean(marker_profiles_k, axis=0)
            ksn = np.linalg.norm(kernel_sig)
            if ksn > 1e-10:
                kernel_sig = kernel_sig / ksn
            else:
                kernel_sig = None

    reg_genes_set = set(g for g in predictor.regulatory_genes if predictor.tf_tier.get(g, 0) >= 2)

    genes = []
    vectors = []
    for gene, progs in p.gene_progs.items():
        if gene not in reg_genes_set:
            continue

        n_progs = len(progs)
        act_precision = len(progs & activate) / max(n_progs, 1)
        tc = gene_target_conn.get(gene, 0)
        target_conn_frac = tc / max(n_markers, 1)
        sc = gene_source_conn.get(gene, 0)
        source_conn_frac = sc / max(n_source_markers, 1)

        n_enriched = 0
        sum_log_enr = 0.0
        for pid in progs:
            e = prog_enr.get(pid, 0)
            if e > 1.0:
                n_enriched += 1
                sum_log_enr += math.log2(e)
        frac_enriched = n_enriched / max(n_progs, 1)
        mean_enr = sum_log_enr / n_enriched if n_enriched > 0 else 0

        gtex_score = 0.0
        if gene in p.gene_expr and t_indices:
            expr = p.gene_expr[gene]
            target_expr = sum(expr[i] for i in t_indices) / len(t_indices)
            all_expr = float(expr.mean())
            source_expr = sum(expr[i] for i in s_indices) / len(s_indices) if s_indices else all_expr
            if all_expr > 0 and target_expr > 1:
                ratio = target_expr / max(all_expr, 0.1)
                gtex_score = min(math.log2(max(ratio, 1)), 5) / 5
                if source_expr > 0 and target_expr > source_expr:
                    gtex_score = min(gtex_score * 1.3, 1.0)

        tau_score = min(p.gene_tau.get(gene, 0.5) / 0.9, 1.0)
        pheno_s = compute_phenotype_score(gene, target_types, gene_pheno) if gene_pheno else 0.0

        kern_s = 0.0
        if kernel_sig is not None and gene in kernel_gene_idx:
            gene_res = M_res[kernel_gene_idx[gene]]
            gn = np.linalg.norm(gene_res)
            if gn > 1e-10:
                kern_s = max(np.dot(gene_res, kernel_sig) / gn, 0.0)

        temp_s = compute_temporal_score(gene, target_types, gene_temporal) if gene_temporal else 0.0

        genes.append(gene)
        vectors.append([target_conn_frac, source_conn_frac,
                        act_precision, frac_enriched, min(mean_enr / 5, 1),
                        max(gtex_score, 0), tau_score, pheno_s, kern_s, temp_s])

    return genes, np.array(vectors, dtype=np.float32)


def run_stage1():
    from vm_cocktail_predictor import CocktailPredictor
    from calibrate_weights import load_phenotype_data, load_temporal_data, load_kernel_data

    t0 = time.time()
    print("=== STAGE 1: Precompute raw component vectors ===\n")

    print("Loading predictor and data...")
    predictor = CocktailPredictor()
    predictor.load()
    gene_pheno = load_phenotype_data()
    gene_temporal = load_temporal_data()
    M_res, kernel_gene_idx, _ = load_kernel_data()
    print(f"  Loaded in {time.time()-t0:.1f}s\n")

    m1, m2, m3 = load_maps()
    transitions = {
        'Set 1': build_transitions(CSV_77, *m1),
        'Set 2': build_transitions(CSV_EXT, *m2),
        'Set 3': build_transitions(CSV_S3, *m3, source_col='source'),
    }
    for s, t in transitions.items():
        print(f"  {s}: {len(t)} cocktails")

    unique_keys = set()
    for s, trans in transitions.items():
        for t in trans:
            unique_keys.add((t['source_type'], tuple(sorted(t['target_types']))))
    print(f"\n  {len(unique_keys)} unique transitions to compute\n")

    raw_cache = {}
    for i, key in enumerate(sorted(unique_keys)):
        src, tgts = key
        print(f"  [{i+1}/{len(unique_keys)}] {src} → {tgts}...", end="", flush=True)
        t1 = time.time()
        genes, vectors = compute_raw_for_transition(
            predictor, src, list(tgts), gene_pheno, gene_temporal, M_res, kernel_gene_idx
        )
        raw_cache[key] = (genes, vectors)
        print(f" {len(genes)} genes, {time.time()-t1:.1f}s")

    transition_factors = {}
    for s, trans in transitions.items():
        tf_list = []
        for t in trans:
            key = (t['source_type'], tuple(sorted(t['target_types'])))
            tf_list.append({'key': key, 'factors': t['factors']})
        transition_factors[s] = tf_list

    cache_data = {
        'raw_cache': raw_cache,
        'transition_factors': transition_factors,
    }
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(cache_data, f)

    print(f"\n  Cache saved to {CACHE_PATH} ({os.path.getsize(CACHE_PATH)/1e6:.1f} MB)")
    print(f"  Stage 1 completed in {time.time()-t0:.1f}s")


def rank_with_weights(genes, vectors, weights):
    src_pen = weights['src_pen']
    target_conn = vectors[:, 0]
    source_conn = vectors[:, 1]
    directional = np.maximum(target_conn - src_pen * source_conn, 0)

    w_vec = np.array([
        weights['w_dir'], weights['w_act'], weights['w_frac'],
        weights['w_enr'], weights['w_gtex'], weights['w_tau'],
        weights['w_pheno'], weights['w_kern'], weights['w_temp'],
    ], dtype=np.float32)

    component_vals = np.column_stack([
        directional, vectors[:, 2], vectors[:, 3], vectors[:, 4],
        vectors[:, 5], vectors[:, 6], vectors[:, 7], vectors[:, 8], vectors[:, 9],
    ])

    composites = component_vals @ w_vec
    order = np.argsort(-composites)
    gene_rank = {genes[idx]: r + 1 for r, idx in enumerate(order)}
    return gene_rank, len(genes)


def get_calibrated_weights_local(target_types):
    from validate_77_cocktails import get_calibrated_weights
    return get_calibrated_weights(target_types)


def evaluate(raw_cache, transition_factors, weight_fn):
    results = {}
    for set_name, tf_list in transition_factors.items():
        rank_cache = {}
        entries = []
        for tf in tf_list:
            key = tf['key']
            if key not in rank_cache:
                genes, vectors = raw_cache[key]
                w = weight_fn(list(key[1]))
                gene_rank, n_reg = rank_with_weights(genes, vectors, w)
                rank_cache[key] = (gene_rank, n_reg)
            gene_rank, n_reg = rank_cache[key]
            for f in tf['factors']:
                if f in gene_rank:
                    entries.append(gene_rank[f] / n_reg * 100)
        top5 = sum(1 for e in entries if e <= 5) / len(entries) * 100 if entries else 0
        results[set_name] = {'top5': top5, 'n': len(entries), 'entries': entries}
    return results


def make_ablated_fn(drop_comp):
    def fn(target_types):
        w = dict(get_calibrated_weights_local(target_types))
        w[drop_comp] = 0.0
        total = sum(w[c] for c in COMPONENTS)
        if total > 0:
            for c in COMPONENTS:
                w[c] /= total
        w['src_pen'] = get_calibrated_weights_local(target_types)['src_pen']
        return w
    return fn


def make_single_fn(comp):
    def fn(target_types):
        w = {c: 0.0 for c in COMPONENTS}
        w[comp] = 1.0
        w['src_pen'] = get_calibrated_weights_local(target_types)['src_pen']
        return w
    return fn


def run_stage2():
    t0 = time.time()
    print("=== STAGE 2: Run all null tests from cached data ===\n")

    with open(CACHE_PATH, 'rb') as f:
        cache_data = pickle.load(f)
    raw_cache = cache_data['raw_cache']
    transition_factors = cache_data['transition_factors']
    print(f"  Cache loaded ({len(raw_cache)} transitions)")

    baseline = evaluate(raw_cache, transition_factors, get_calibrated_weights_local)
    sets = list(baseline.keys())
    for s in sets:
        print(f"  Baseline {s}: {baseline[s]['top5']:.1f}% Top-5% ({baseline[s]['n']} pairs)")

    print("\n" + "="*100)
    print("  TEST 1: COMPONENT ABLATION")
    print("="*100)
    ablation = {}
    for comp in COMPONENTS:
        res = evaluate(raw_cache, transition_factors, make_ablated_fn(comp))
        comp_res = {}
        for s in sets:
            delta = res[s]['top5'] - baseline[s]['top5']
            comp_res[s] = {'top5': res[s]['top5'], 'delta': delta, 'n': res[s]['n']}
        ablation[comp] = comp_res

    print(f"\n  {'Component':<30s} {'Set 1 Δ':>10s} {'Set 2 Δ':>10s} {'Set 3 Δ':>10s} {'Mean Δ':>10s}")
    print(f"  {'─'*70}")
    sorted_abl = sorted(COMPONENTS, key=lambda c: np.mean([ablation[c][s]['delta'] for s in sets]))
    for comp in sorted_abl:
        deltas = [ablation[comp][s]['delta'] for s in sets]
        vals = [f"{d:+.1f}pp" for d in deltas]
        print(f"  {COMPONENT_NAMES[comp]:<30s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s} {np.mean(deltas):+.1f}pp")

    print("\n" + "="*100)
    print("  TEST 2: SINGLE-COMPONENT BASELINES")
    print("="*100)
    single = {}
    for comp in COMPONENTS:
        res = evaluate(raw_cache, transition_factors, make_single_fn(comp))
        single[comp] = {s: {'top5': res[s]['top5'], 'n': res[s]['n']} for s in sets}

    print(f"\n  {'Component':<30s} {'Set 1':>10s} {'Set 2':>10s} {'Set 3':>10s} {'Mean':>10s}")
    print(f"  {'─'*70}")
    sorted_single = sorted(COMPONENTS, key=lambda c: -np.mean([single[c][s]['top5'] for s in sets]))
    for comp in sorted_single:
        vals = [f"{single[comp][s]['top5']:.1f}%" for s in sets]
        mean_v = np.mean([single[comp][s]['top5'] for s in sets])
        print(f"  {COMPONENT_NAMES[comp]:<30s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s} {mean_v:.1f}%")
    vals = [f"{baseline[s]['top5']:.1f}%" for s in sets]
    mean_bl = np.mean([baseline[s]['top5'] for s in sets])
    print(f"  {'─'*70}")
    print(f"  {'Full 9-component model':<30s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s} {mean_bl:.1f}%")

    print("\n" + "="*100)
    print("  TEST 3: RANDOM-DRAW PERMUTATION (10,000 perms)")
    print("="*100)
    np.random.seed(42)
    perm_results = {}
    for s in sets:
        entries = baseline[s]['entries']
        n = len(entries)
        observed = sum(1 for e in entries if e <= 5) / n * 100
        null_top5s = np.sum(np.random.uniform(0, 100, (10000, n)) <= 5, axis=1) / n * 100
        p_val = float(np.sum(null_top5s >= observed) / 10000)
        nm = float(np.mean(null_top5s))
        ns = float(np.std(null_top5s))
        z = (observed - nm) / ns if ns > 0 else float('inf')
        perm_results[s] = {'observed': observed, 'null_mean': nm, 'null_std': ns,
                           'p_value': p_val, 'z_score': z, 'n': n}
        print(f"  {s}: {observed:.1f}% observed, null {nm:.1f}±{ns:.1f}%, "
              f"z={z:.1f}, p={'<0.0001' if p_val == 0 else f'{p_val:.4f}'}, "
              f"fold={observed/max(nm, 0.01):.1f}×")

    print("\n" + "="*100)
    print("  TEST 4: SCORE-SHUFFLE PERMUTATION (1,000 perms)")
    print("="*100)
    np.random.seed(42)
    shuffle_results = {}
    for s in sets:
        rank_cache = {}
        factor_n_regs = []
        factor_pctiles = []
        for tf in transition_factors[s]:
            key = tf['key']
            if key not in rank_cache:
                genes, vectors = raw_cache[key]
                w = get_calibrated_weights_local(list(key[1]))
                gene_rank, n_reg = rank_with_weights(genes, vectors, w)
                rank_cache[key] = (gene_rank, n_reg)
            gene_rank, n_reg = rank_cache[key]
            for f in tf['factors']:
                if f in gene_rank:
                    factor_n_regs.append(n_reg)
                    factor_pctiles.append(gene_rank[f] / n_reg * 100)

        n = len(factor_pctiles)
        observed = sum(1 for p in factor_pctiles if p <= 5) / n * 100
        n_regs_arr = np.array(factor_n_regs)

        null_top5s = []
        for _ in range(1000):
            random_ranks = np.array([np.random.randint(1, nr + 1) for nr in n_regs_arr])
            random_pctiles = random_ranks / n_regs_arr * 100
            null_top5s.append(np.sum(random_pctiles <= 5) / n * 100)

        null_top5s = np.array(null_top5s)
        p_val = float(np.sum(null_top5s >= observed) / 1000)
        nm = float(np.mean(null_top5s))
        ns = float(np.std(null_top5s))
        z = (observed - nm) / ns if ns > 0 else float('inf')
        shuffle_results[s] = {'observed': observed, 'null_mean': nm, 'null_std': ns,
                              'p_value': p_val, 'z_score': z, 'n': n}
        print(f"  {s}: {observed:.1f}% observed, null {nm:.1f}±{ns:.1f}%, "
              f"z={z:.1f}, p={'<0.001' if p_val == 0 else f'{p_val:.4f}'}, "
              f"fold={observed/max(nm, 0.01):.1f}×")

    print("\n  Saving results and generating figures...")
    save_all(ablation, single, perm_results, shuffle_results, baseline, sets)
    generate_figures(ablation, single, perm_results, shuffle_results, baseline, sets)

    print(f"\n  Stage 2 completed in {time.time()-t0:.1f}s")


def save_all(ablation, single, perm_results, shuffle_results, baseline, sets):
    output = {
        'ablation': {c: {'name': COMPONENT_NAMES[c],
                         'sets': {s: {'top5': ablation[c][s]['top5'], 'delta': ablation[c][s]['delta']} for s in sets}}
                     for c in COMPONENTS},
        'single_component': {c: {'name': COMPONENT_NAMES[c],
                                  'sets': {s: {'top5': single[c][s]['top5']} for s in sets}}
                             for c in COMPONENTS},
        'permutation': perm_results,
        'score_shuffle': shuffle_results,
        'baseline': {s: {'top5': baseline[s]['top5'], 'n': baseline[s]['n'],
                         'median_pctile': float(np.median(baseline[s]['entries']))}
                     for s in sets},
    }
    with open(os.path.join(BASE, 'null_test_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    out_dir = os.path.join(BASE, 'v2_submission')
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'table_s7_ablation.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['component', 'component_name', 'set1_top5', 'set1_delta', 'set2_top5', 'set2_delta',
                     'set3_top5', 'set3_delta', 'mean_delta'])
        for comp in sorted(COMPONENTS, key=lambda c: np.mean([ablation[c][s]['delta'] for s in sets])):
            row = [comp, COMPONENT_NAMES[comp]]
            deltas = []
            for s in sets:
                row.extend([f"{ablation[comp][s]['top5']:.1f}", f"{ablation[comp][s]['delta']:+.1f}"])
                deltas.append(ablation[comp][s]['delta'])
            row.append(f"{np.mean(deltas):+.1f}")
            w.writerow(row)

    with open(os.path.join(out_dir, 'table_s8_single_component.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['component', 'component_name', 'set1_top5', 'set2_top5', 'set3_top5', 'mean_top5'])
        for comp in sorted(COMPONENTS, key=lambda c: -np.mean([single[c][s]['top5'] for s in sets])):
            row = [comp, COMPONENT_NAMES[comp]]
            vals = []
            for s in sets:
                row.append(f"{single[comp][s]['top5']:.1f}")
                vals.append(single[comp][s]['top5'])
            row.append(f"{np.mean(vals):.1f}")
            w.writerow(row)
        row = ['full_model', 'Full 9-component model']
        for s in sets:
            row.append(f"{baseline[s]['top5']:.1f}")
        row.append(f"{np.mean([baseline[s]['top5'] for s in sets]):.1f}")
        w.writerow(row)

    print(f"  Results + tables saved")


def generate_figures(ablation, single, perm_results, shuffle_results, baseline, sets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    C_BLUE, C_RED, C_TEAL, C_ORANGE, C_GRAY = "#2166AC", "#B2182B", "#01665E", "#E08214", "#969696"
    fig_dir = os.path.join(BASE, 'v2_submission', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    set_labels = ["Set 1", "Set 2", "Set 3"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    ax = axes[0, 0]
    ax.text(-0.08, 1.05, "a", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    sorted_comps = sorted(COMPONENTS, key=lambda c: np.mean([ablation[c][s]['delta'] for s in sets]))
    x = np.arange(len(sorted_comps))
    width = 0.25
    colors = [C_BLUE, C_TEAL, C_ORANGE]
    for i, s in enumerate(sets):
        deltas = [ablation[c][s]['delta'] for c in sorted_comps]
        ax.bar(x + i*width, deltas, width, label=set_labels[i], color=colors[i], edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels([COMPONENT_NAMES[c] for c in sorted_comps], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Top-5% change (pp)")
    ax.set_title("Component ablation: impact of removing each component")
    ax.legend(loc="lower left", fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[0, 1]
    ax.text(-0.08, 1.05, "b", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    sorted_single = sorted(COMPONENTS, key=lambda c: -np.mean([single[c][s]['top5'] for s in sets]))
    for i, s in enumerate(sets):
        vals = [single[c][s]['top5'] for c in sorted_single]
        ax.bar(x + i*width, vals, width, label=set_labels[i], color=colors[i], edgecolor="white")
    bl_means = [baseline[s]['top5'] for s in sets]
    ax.axhline(np.mean(bl_means), color=C_RED, linestyle="--", linewidth=2, alpha=0.7, label="Full model (mean)")
    ax.axhline(5, color=C_GRAY, linestyle=":", linewidth=1, alpha=0.5)
    ax.text(len(x)-0.5, 6.5, "Random (5%)", fontsize=7, color=C_GRAY)
    ax.set_xticks(x + width)
    ax.set_xticklabels([COMPONENT_NAMES[c] for c in sorted_single], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Top-5% recovery (%)")
    ax.set_title("Single-component baselines vs full model")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, max(bl_means) * 1.15)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1, 0]
    ax.text(-0.08, 1.05, "c", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    x_pos = np.arange(3)
    obs = [perm_results[s]['observed'] for s in sets]
    nms = [perm_results[s]['null_mean'] for s in sets]
    nss = [perm_results[s]['null_std'] for s in sets]
    ax.bar(x_pos - 0.15, obs, 0.3, color=C_BLUE, label="Observed", edgecolor="white")
    ax.bar(x_pos + 0.15, nms, 0.3, color=C_GRAY, label="Null (random)", edgecolor="white", yerr=nss, capsize=4)
    for i, (o, n) in enumerate(zip(obs, nms)):
        ax.text(i, o + 2, f"{o/n:.0f}×", ha="center", fontsize=10, fontweight="bold", color=C_RED)
    ax.set_xticks(x_pos); ax.set_xticklabels(set_labels)
    ax.set_ylabel("Top-5% recovery (%)"); ax.set_title("Permutation test: observed vs random expectation")
    ax.legend(loc="upper right"); ax.set_ylim(0, max(obs) * 1.25)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1, 1]
    ax.text(-0.08, 1.05, "d", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    obs_s = [shuffle_results[s]['observed'] for s in sets]
    nms_s = [shuffle_results[s]['null_mean'] for s in sets]
    nss_s = [shuffle_results[s]['null_std'] for s in sets]
    zs = [shuffle_results[s]['z_score'] for s in sets]
    ax.bar(x_pos - 0.15, obs_s, 0.3, color=C_BLUE, label="Observed", edgecolor="white")
    ax.bar(x_pos + 0.15, nms_s, 0.3, color=C_GRAY, label="Null (shuffled)", edgecolor="white", yerr=nss_s, capsize=4)
    for i, (o, z) in enumerate(zip(obs_s, zs)):
        ax.text(i, o + 2, f"z={z:.0f}", ha="center", fontsize=10, fontweight="bold", color=C_RED)
    ax.set_xticks(x_pos); ax.set_xticklabels(set_labels)
    ax.set_ylabel("Top-5% recovery (%)"); ax.set_title("Score-shuffle test: observed vs shuffled gene→score mapping")
    ax.legend(loc="upper right"); ax.set_ylim(0, max(obs_s) * 1.25)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig9_null_tests.png")
    fig.savefig(f"{fig_dir}/fig9_null_tests.pdf")
    plt.close(fig)
    print(f"  Fig 9 saved to {fig_dir}/")


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if stage in ('stage1', 'both'):
        run_stage1()
    if stage in ('stage2', 'both'):
        run_stage2()
