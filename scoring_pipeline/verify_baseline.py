#!/usr/bin/env python3
"""
Baseline Verification — Canonical Two-Tier Metrics
=====================================================
Verifies canonical validation numbers for all three sets under both
Tier 2+ filter (manuscript) and all-regulatory filter (reviewer response).

Precomputes ALL regulatory genes once per transition, then applies
tier filter at evaluation time for speed.
"""

import csv
import os
import sys
import time
import math
import json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from vm_cocktail_predictor import CocktailPredictor, CELL_GTEX_MAP
from calibrate_weights import (
    load_phenotype_data, load_temporal_data, load_kernel_data,
    compute_phenotype_score, compute_temporal_score,
    CELL_PHENOTYPE_MAP, CELL_DEV_WINDOWS,
)
from validate_77_cocktails import (
    TIER_A_WEIGHTS, TIER_B_WEIGHTS, TIER_A_CELL_TYPES,
    get_calibrated_weights,
    TARGET_CELL_MAP as SET1_TARGET_MAP,
    SOURCE_CELL_MAP as SET1_SOURCE_MAP,
    parse_factors as parse_factors_set1,
)
from validate_extended_cocktails import (
    TARGET_CELL_MAP as SET2_TARGET_MAP,
    SOURCE_CELL_MAP as SET2_SOURCE_MAP,
    parse_factors as parse_factors_set2,
)


CSV_PATH_1 = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           'attached_assets',
                           'published_reprogramming_cocktails_1774053006993.csv')
CSV_PATH_2 = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           'attached_assets',
                           'additional_reprogramming_cocktails_1774066435911.csv')
CSV_PATH_3 = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           'attached_assets',
                           'validation_set3_cocktails_1774073324946.csv')

SET3_TARGET_MAP = {
    'Cardiomyocyte human': ['CARDIOMYOCYTE'],
    'Cortical neuron': ['NEURON_EXCITATORY'],
    'GABAergic interneuron': ['NEURON_INHIBITORY'],
    'Parvalbumin interneuron': ['NEURON_INHIBITORY'],
    'Induced neuron': ['NEURON_EXCITATORY', 'NEURON_INHIBITORY'],
    'Neuron from astrocyte': ['NEURON_EXCITATORY', 'NEURON_INHIBITORY'],
    'Neural stem cell': ['NEURON_EXCITATORY'],
    'Oligodendrocyte': ['OLIGODENDROCYTE'],
    'OPC from astrocyte': ['OLIGODENDROCYTE'],
    'NK cell': ['NK_CELL'],
    'Regulatory T cell': ['T_CELL'],
    'iTreg': ['T_CELL'],
    'Macrophage': ['MACROPHAGE'],
    'Monocyte/Macrophage': ['MACROPHAGE'],
    'cDC1': ['DENDRITIC_CELL'],
    'cDC2': ['DENDRITIC_CELL'],
    'mature regulatory DC': ['DENDRITIC_CELL'],
    'AT2 maintenance': ['PNEUMOCYTE_II'],
    'AT2 regeneration': ['PNEUMOCYTE_II'],
    'Podocyte progenitor': ['PODOCYTE'],
    'Renal tubular epithelial': ['PODOCYTE'],
    'Corticotroph': ['BETA_CELL'],
    'Somatotroph': ['BETA_CELL'],
    'Parathyroid': ['BETA_CELL'],
}

SET3_SOURCE_MAP = {
    'Fibroblast': 'FIBROBLAST', 'Astrocyte': 'ASTROCYTE',
    'Astroglia': 'ASTROCYTE', 'Cardiac fibroblast': 'FIBROBLAST',
    'CD4+ T conv': 'T_CELL', 'CD4+CD25- T cells': 'T_CELL',
    'NPC': 'NEURON_EXCITATORY', 'AT2': 'PNEUMOCYTE_II',
    'PSC-derived AFE': 'FIBROBLAST', 'Pituitary progenitor': 'FIBROBLAST',
}

SET3_ALIAS_MAP = {
    'OCT4': 'POU5F1', 'BRN2': 'POU3F2', 'PU.1': 'SPI1',
    'cMYC': 'MYC', 'NURR1': 'NR4A2', 'NGN2': 'NEUROG2',
    'CTIP2': 'BCL11B', 'T-BET': 'TBX21', 'BLIMP1': 'PRDM1',
}


def parse_factors_set3(raw_factors):
    parts = [p.strip() for p in raw_factors.replace(';', ',').split(',')]
    out = []
    for p in parts:
        if not p:
            continue
        if p.startswith('miR-') or 'chemical' in p.lower() or p == 'various':
            continue
        if 'pathway' in p.lower() or 'activation' in p.lower() or p == 'WNT':
            continue
        p = p.replace(' + chemical', '').replace(' + mechanical', '').strip()
        p = p.replace(' (knockdown)', '').strip()
        if p:
            out.append(SET3_ALIAS_MAP.get(p, p))
    return out


def precompute_transition(predictor, source_type, target_types,
                          gene_pheno, gene_temporal, M_res, kernel_gene_idx):
    """Precompute scoring components for ALL regulatory genes in one pass.
    Returns genes list, tier array, and component arrays."""
    p = predictor
    source_markers = p.ct_markers.get(source_type, {})
    target_markers = {}
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

    target_markers_in = set(m for m in target_markers if m in p.gene_progs)
    source_markers_in = set(m for m in source_markers if m in p.gene_progs)
    n_markers = max(len(target_markers_in), 1)
    n_src = max(len(source_markers_in), 1)

    gene_tc = defaultdict(int)
    for m in target_markers_in:
        seen = set()
        for pid in p.gene_progs[m]:
            for g in p.prog_genes.get(pid, ()):
                if g not in seen:
                    seen.add(g)
                    gene_tc[g] += 1

    gene_sc = defaultdict(int)
    for m in source_markers_in:
        seen = set()
        for pid in p.gene_progs[m]:
            for g in p.prog_genes.get(pid, ()):
                if g not in seen:
                    seen.add(g)
                    gene_sc[g] += 1

    prog_enr = {}
    for pid, genes in p.prog_genes.items():
        n_in = len(genes)
        n_mark = sum(1 for g in genes if g in target_markers_in)
        if n_mark > 0:
            expected = n_in * len(target_markers_in) / p.n_genes if p.n_genes > 0 else 0
            prog_enr[pid] = n_mark / max(expected, 1e-10)

    gtex_tissues = list(set(t for ct in target_types for t in CELL_GTEX_MAP.get(ct, [])))
    source_gtex = CELL_GTEX_MAP.get(source_type, [])
    t_idx = [p.tissue_idx[t] for t in gtex_tissues if t in p.tissue_idx]
    s_idx = [p.tissue_idx[t] for t in source_gtex if t in p.tissue_idx]

    target_marker_genes = set(target_markers.keys())
    mk_indices = [kernel_gene_idx[m] for m in target_marker_genes if m in kernel_gene_idx]
    kernel_sig = None
    if len(mk_indices) >= 3:
        sig = np.mean(M_res[mk_indices], axis=0)
        sn = np.linalg.norm(sig)
        if sn > 1e-10:
            kernel_sig = sig / sn

    target_cats = set()
    for ct in target_types:
        target_cats.update(CELL_PHENOTYPE_MAP.get(ct, []))

    dev_windows = []
    for ct in target_types:
        dev_windows.extend(CELL_DEV_WINDOWS.get(ct, []))

    genes_list = []
    gene_to_idx = {}
    tiers = []
    components = []

    for gene in p.gene_progs:
        if gene not in p.regulatory_genes:
            continue

        progs = p.gene_progs[gene]
        n_p = len(progs)
        act_prec = len(progs & activate) / max(n_p, 1)
        tc_frac = gene_tc.get(gene, 0) / n_markers
        sc_frac = gene_sc.get(gene, 0) / n_src

        n_en = 0; s_log = 0.0
        for pid in progs:
            e = prog_enr.get(pid, 0)
            if e > 1.0:
                n_en += 1
                s_log += math.log2(e)
        frac_en = n_en / max(n_p, 1)
        mean_en = s_log / n_en if n_en > 0 else 0

        gs = 0.0
        if gene in p.gene_expr and t_idx:
            expr = p.gene_expr[gene]
            tgt_e = sum(expr[i] for i in t_idx) / len(t_idx)
            all_e = float(expr.mean())
            src_e = sum(expr[i] for i in s_idx) / len(s_idx) if s_idx else all_e
            if all_e > 0 and tgt_e > 1:
                ratio = tgt_e / max(all_e, 0.1)
                gs = min(math.log2(max(ratio, 1)), 5) / 5
                if src_e > 0 and tgt_e > src_e:
                    gs = min(gs * 1.3, 1.0)

        tau_s = min(p.gene_tau.get(gene, 0.5) / 0.9, 1.0)

        pheno_s = 0.0
        gene_cats = gene_pheno.get(gene)
        if gene_cats and target_cats:
            overlap = len(gene_cats & target_cats)
            if overlap > 0:
                pheno_s = overlap / len(gene_cats)

        kern_s = 0.0
        if kernel_sig is not None and gene in kernel_gene_idx:
            gene_res = M_res[kernel_gene_idx[gene]]
            gn = np.linalg.norm(gene_res)
            if gn > 1e-10:
                kern_s = max(np.dot(gene_res, kernel_sig) / gn, 0.0)

        temp_s = 0.0
        t = gene_temporal.get(gene)
        if t and dev_windows:
            gene_start, gene_peak, gene_end = t
            best = 0.0
            for ws, we in dev_windows:
                os_ = max(gene_start, ws)
                oe = min(gene_end, we)
                if os_ <= oe:
                    score = min((oe - os_) / max(we - ws, 1), 1.0)
                    if ws <= gene_peak <= we:
                        score = min(score * 1.5, 1.0)
                    best = max(best, score)
            temp_s = best

        gene_to_idx[gene] = len(genes_list)
        genes_list.append(gene)
        tiers.append(p.tf_tier.get(gene, 0))
        components.append((tc_frac, sc_frac, act_prec, frac_en, min(mean_en/5, 1), gs, tau_s, pheno_s, kern_s, temp_s))

    arr = np.array(components, dtype=np.float64)
    tier_arr = np.array(tiers, dtype=np.int32)

    return {
        'gene_to_idx': gene_to_idx,
        'tier': tier_arr,
        'tc': arr[:, 0], 'sc': arr[:, 1], 'act': arr[:, 2],
        'frac': arr[:, 3], 'enr': arr[:, 4], 'gtex': arr[:, 5],
        'tau': arr[:, 6], 'pheno': arr[:, 7], 'kernel': arr[:, 8],
        'temporal': arr[:, 9], 'n_all': len(genes_list),
        'target_types': target_types,
    }


def evaluate_from_precomputed(trans_map, test_cases, tier_filter):
    rank_cache = {}
    entries = []

    for tc in test_cases:
        ck = tc['cache_key']
        tt = tc['target_types']
        filter_key = (ck, tier_filter)
        if ck not in trans_map:
            continue

        if filter_key not in rank_cache:
            td = trans_map[ck]
            if tier_filter == 'tier2plus':
                mask = td['tier'] >= 2
            else:
                mask = np.ones(td['n_all'], dtype=bool)

            w = get_calibrated_weights(tt)
            directional = np.maximum(td['tc'] - w['src_pen'] * td['sc'], 0)
            composites = (w['w_dir'] * directional +
                          w['w_act'] * td['act'] +
                          w['w_frac'] * td['frac'] +
                          w['w_enr'] * td['enr'] +
                          w['w_gtex'] * np.maximum(td['gtex'], 0) +
                          w['w_tau'] * td['tau'] +
                          w['w_pheno'] * td['pheno'] +
                          w['w_kern'] * td['kernel'] +
                          w['w_temp'] * td['temporal'])

            filtered_composites = composites[mask]
            filtered_indices = np.where(mask)[0]
            order = np.argsort(-filtered_composites)
            n_filtered = len(filtered_composites)

            rank_of_filtered = {}
            for rank_pos, idx_in_filtered in enumerate(order):
                orig_idx = filtered_indices[idx_in_filtered]
                rank_of_filtered[orig_idx] = rank_pos + 1

            rank_cache[filter_key] = (rank_of_filtered, td['gene_to_idx'], n_filtered)

        rank_of, g2i, n = rank_cache[filter_key]
        f = tc['factor']
        if f not in g2i:
            continue
        gene_idx = g2i[f]
        if gene_idx not in rank_of:
            continue
        rank = rank_of[gene_idx]
        pctile = rank / n * 100
        entries.append({'factor': f, 'rank': rank, 'percentile': pctile})

    if not entries:
        return {'top1': 0, 'top5': 0, 'top10': 0, 'top20': 0, 'n': 0}
    n = len(entries)
    return {
        'top1': sum(1 for e in entries if e['percentile'] <= 1) / n * 100,
        'top5': sum(1 for e in entries if e['percentile'] <= 5) / n * 100,
        'top10': sum(1 for e in entries if e['percentile'] <= 10) / n * 100,
        'top20': sum(1 for e in entries if e['percentile'] <= 20) / n * 100,
        'n': n,
    }


def main():
    predictor = CocktailPredictor()
    predictor.load()

    n_tier2 = sum(1 for g in predictor.gene_progs
                  if g in predictor.regulatory_genes and predictor.tf_tier.get(g, 0) >= 2)
    n_all_reg = sum(1 for g in predictor.gene_progs if g in predictor.regulatory_genes)

    gene_pheno = load_phenotype_data()
    gene_temporal = load_temporal_data()
    M_res, kernel_gene_idx, _ = load_kernel_data()

    set_configs = [
        (CSV_PATH_1, SET1_TARGET_MAP, SET1_SOURCE_MAP, parse_factors_set1, 'Set 1 (77 cocktails)'),
        (CSV_PATH_2, SET2_TARGET_MAP, SET2_SOURCE_MAP, parse_factors_set2, 'Set 2 (64 cocktails)'),
        (CSV_PATH_3, SET3_TARGET_MAP, SET3_SOURCE_MAP, parse_factors_set3, 'Set 3 blind (33 cocktails)'),
    ]

    all_transitions = set()
    set_test_cases = []

    for path, tmap, smap, pfn, label in set_configs:
        if not os.path.exists(path):
            print(f"  WARNING: {label} CSV not found at {path}")
            set_test_cases.append((label, []))
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        test_cases = []
        for row in rows:
            factors = pfn(row['factors'])
            if not factors:
                continue
            tt = tmap.get(row['target_cell'])
            st = smap.get(row.get('source_cell', row.get('source', '')))
            if not tt or not st:
                continue
            ck = (st, tuple(sorted(tt)))
            all_transitions.add(ck)
            for f in factors:
                if f in predictor.regulatory_genes:
                    test_cases.append({'cache_key': ck, 'factor': f, 'target_types': tt})
        set_test_cases.append((label, test_cases))

    print(f"\n  Precomputing {len(all_transitions)} unique transitions (all regulatory genes)...")
    t0 = time.time()
    trans_map = {}
    for i, ck in enumerate(sorted(all_transitions)):
        st, tt = ck[0], list(ck[1])
        td = precompute_transition(predictor, st, tt,
                                   gene_pheno, gene_temporal,
                                   M_res, kernel_gene_idx)
        trans_map[ck] = td
        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{len(all_transitions)} transitions done ({time.time()-t0:.0f}s)")
    print(f"  All transitions precomputed in {time.time()-t0:.1f}s")

    results = {}

    for tier_filter, tier_label in [
        ('tier2plus', f'TIER 2+ FILTER ({n_tier2:,} TFs) — Manuscript numbers'),
        ('all_reg', f'ALL REGULATORY ({n_all_reg:,}) — Reviewer numbers'),
    ]:
        print(f"\n  {'─'*80}")
        print(f"  {tier_label}")
        print(f"  {'─'*80}")

        for label, test_cases in set_test_cases:
            if not test_cases:
                print(f"  {label:30s}: NO DATA")
                continue
            res = evaluate_from_precomputed(trans_map, test_cases, tier_filter)
            key = f'{tier_filter}_{label[:5].strip()}'
            results[key] = res
            print(f"  {label:30s}:  Top-1%={res['top1']:.1f}%  Top-5%={res['top5']:.1f}%  "
                  f"Top-10%={res['top10']:.1f}%  Top-20%={res['top20']:.1f}%  (n={res['n']})")

    print(f"\n  {'─'*80}")
    print(f"  EXPECTED CANONICAL VALUES")
    print(f"  {'─'*80}")
    print(f"  Tier 2+: Set1=77.8/83.2  Set2=64.7/76.7  Set3=65.3/73.5")
    print(f"  Reviewer: Set1=71.4  Set2=57.0  Set3=56.0")

    output = {
        'n_tier2plus': n_tier2,
        'n_all_regulatory': n_all_reg,
        'weights': {
            'tier_a': TIER_A_WEIGHTS,
            'tier_b': TIER_B_WEIGHTS,
            'tier_a_cell_types': sorted(TIER_A_CELL_TYPES),
        },
        'results': results,
        'canonical_expected': {
            'tier2plus': {
                'set1_top5': 77.8, 'set1_top10': 83.2,
                'set2_top5': 64.7, 'set2_top10': 76.7,
                'set3_top5': 65.3, 'set3_top10': 73.5,
            },
            'reviewer': {'set1_top5': 71.4, 'set2_top5': 57.0, 'set3_top5': 56.0},
        },
    }

    out_path = os.path.join(os.path.dirname(__file__), 'baseline_verification_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    t0 = time.time()
    print("=" * 100)
    print("  TWO-TIER BASELINE VERIFICATION")
    print("=" * 100)
    main()
    print(f"\n  Total runtime: {time.time()-t0:.1f}s")
