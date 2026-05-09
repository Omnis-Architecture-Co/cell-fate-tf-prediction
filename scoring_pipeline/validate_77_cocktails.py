#!/usr/bin/env python3
"""
Comprehensive 77-Cocktail Validation
=====================================
Validates the VM Cocktail Predictor against 77 published reprogramming cocktails.
Optimized: uses a lightweight scoring path that only computes composites + ranks.
"""

import csv
import json
import os
import sys
import time
import math
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from vm_cocktail_predictor import CocktailPredictor, CELL_GTEX_MAP
from calibrate_weights import (
    load_phenotype_data, load_temporal_data, load_kernel_data,
    compute_phenotype_score, compute_temporal_score,
    CELL_DEV_WINDOWS, CELL_PHENOTYPE_MAP,
)

TIER_A_CELL_TYPES = frozenset({
    'NEURON_EXCITATORY', 'NEURON_INHIBITORY', 'ASTROCYTE', 'OLIGODENDROCYTE',
    'SCHWANN_CELL', 'CARDIOMYOCYTE', 'HEPATOCYTE', 'SKELETAL_MUSCLE',
    'SMOOTH_MUSCLE', 'BETA_CELL', 'ADIPOCYTE', 'MACROPHAGE', 'MICROGLIA',
})

TIER_A_WEIGHTS = {
    'w_dir': 0.0734, 'w_act': 0.0874, 'w_frac': 0.0856,
    'w_enr': 0.0321, 'w_gtex': 0.2912, 'w_tau': 0.2084,
    'w_pheno': 0.0853, 'w_kern': 0.0933, 'w_temp': 0.0434,
    'src_pen': 0.2605,
}

TIER_B_WEIGHTS = {
    'w_dir': 0.0805, 'w_act': 0.0330, 'w_frac': 0.0407,
    'w_enr': 0.0297, 'w_gtex': 0.1944, 'w_tau': 0.2192,
    'w_pheno': 0.3410, 'w_kern': 0.0491, 'w_temp': 0.0123,
    'src_pen': 0.2759,
}

def get_calibrated_weights(target_types):
    if any(t in TIER_A_CELL_TYPES for t in target_types):
        return TIER_A_WEIGHTS
    return TIER_B_WEIGHTS

CALIBRATED_WEIGHTS = TIER_A_WEIGHTS

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         'attached_assets',
                         'published_reprogramming_cocktails_1774053006993.csv')

ALIAS_MAP = {
    'OCT4': 'POU5F1', 'BRN2': 'POU3F2', 'PU.1': 'SPI1',
    'cFOS': 'FOS', 'cMYC': 'MYC', 'NURR1': 'NR4A2',
    'NGN2': 'NEUROG2', 'NGN1': 'NEUROG1', 'NGN3': 'NEUROG3',
    'HB9': 'MNX1', 'KROX20': 'EGR2', 'OSTERIX': 'SP7',
    'NKX2.2': 'NKX2-2', 'NKX6.2': 'NKX6-2', 'MYOD1-VP64': 'MYOD1',
    'HNF6': 'ONECUT1', 'OCT4 (brief)': 'POU5F1',
}

TARGET_CELL_MAP = {
    'iPSC': None,
    'Neuron (generic)': ['NEURON_EXCITATORY', 'NEURON_INHIBITORY'],
    'Dopaminergic neuron': ['NEURON_EXCITATORY'],
    'Motor neuron': ['NEURON_EXCITATORY'],
    'Neural progenitor': ['NEURON_EXCITATORY', 'NEURON_INHIBITORY'],
    'Neuron (in vivo)': ['NEURON_EXCITATORY', 'NEURON_INHIBITORY'],
    'GABAergic neuron': ['NEURON_INHIBITORY'],
    'Serotonergic neuron': ['NEURON_EXCITATORY'],
    'Cardiomyocyte': ['CARDIOMYOCYTE'],
    'Cardiomyocyte (in vivo)': ['CARDIOMYOCYTE'],
    'Hepatocyte': ['HEPATOCYTE'],
    'Hepatocyte (in vivo)': ['HEPATOCYTE'],
    'Pancreatic beta cell': ['BETA_CELL'],
    'Blood progenitor': ['HSC'],
    'Macrophage': ['MACROPHAGE'],
    'Hematopoietic progenitor': ['HSC'],
    'Erythroid progenitor': ['ERYTHROCYTE'],
    'Dendritic cell (cDC1)': ['DENDRITIC_CELL'],
    'Dendritic cell (cDC2)': ['DENDRITIC_CELL'],
    'Plasmacytoid DC': ['DENDRITIC_CELL'],
    'Myoblast': ['SKELETAL_MUSCLE'],
    'Smooth muscle': ['SMOOTH_MUSCLE'],
    'OPC': ['OLIGODENDROCYTE'],
    'Schwann cell': ['SCHWANN_CELL'],
    'Astrocyte': ['ASTROCYTE'],
    'Endothelial': ['ENDOTHELIAL'],
    'Microvascular endothelial': ['ENDOTHELIAL'],
    'Keratinocyte': ['KERATINOCYTE'],
    'Trophoblast stem cell': None,
    'Brown adipocyte': ['ADIPOCYTE'],
    'White adipocyte': ['ADIPOCYTE'],
    'Chondrocyte': ['CHONDROCYTE'],
    'Osteoblast': ['OSTEOBLAST'],
    'Nephron progenitor': None,
    'Sertoli cell': None,
    'Granulosa cell': None,
}

SOURCE_CELL_MAP = {
    'Fibroblast': 'FIBROBLAST', 'Keratinocyte': 'KERATINOCYTE',
    'CD133+ cord blood': 'HSC', 'Astrocyte': 'ASTROCYTE',
    'Hepatocyte': 'HEPATOCYTE', 'hESC/iPSC': None, 'hESC': None,
    'B cell': 'B_CELL', 'Amniotic cell': 'FIBROBLAST',
    'Exocrine cell': None, 'Alpha cell': 'BETA_CELL',
    'Myofibroblast': 'FIBROBLAST', 'Granulosa cell': None, 'Sertoli cell': None,
}

FAMILY_MAP = {
    'iPSC': 'iPSC', 'Neuron (generic)': 'Neuron', 'Dopaminergic neuron': 'Neuron',
    'Motor neuron': 'Neuron', 'Neural progenitor': 'Neuron',
    'Neuron (in vivo)': 'Neuron', 'GABAergic neuron': 'Neuron',
    'Serotonergic neuron': 'Neuron', 'Cardiomyocyte': 'Cardiac',
    'Cardiomyocyte (in vivo)': 'Cardiac', 'Hepatocyte': 'Hepatic',
    'Hepatocyte (in vivo)': 'Hepatic', 'Pancreatic beta cell': 'Endocrine',
    'Blood progenitor': 'Blood', 'Macrophage': 'Blood',
    'Hematopoietic progenitor': 'Blood', 'Erythroid progenitor': 'Blood',
    'Dendritic cell (cDC1)': 'Immune', 'Dendritic cell (cDC2)': 'Immune',
    'Plasmacytoid DC': 'Immune', 'Myoblast': 'Muscle', 'Smooth muscle': 'Muscle',
    'OPC': 'Glial', 'Schwann cell': 'Glial', 'Astrocyte': 'Glial',
    'Endothelial': 'Vascular', 'Microvascular endothelial': 'Vascular',
    'Keratinocyte': 'Epithelial', 'Trophoblast stem cell': 'Other',
    'Brown adipocyte': 'Mesenchymal', 'White adipocyte': 'Mesenchymal',
    'Chondrocyte': 'Mesenchymal', 'Osteoblast': 'Mesenchymal',
    'Nephron progenitor': 'Other', 'Sertoli cell': 'Other', 'Granulosa cell': 'Other',
}


def fast_score_candidates(predictor, source_types, target_types,
                          gene_pheno=None, gene_temporal=None,
                          M_res=None, kernel_gene_idx=None,
                          weights=None):
    """Lightweight scoring — returns only composite floats per gene, no dict overhead.
    Uses calibrated 9-component weights (two-tier: selects by target cell type)."""
    w = weights if weights is not None else get_calibrated_weights(target_types)
    p = predictor
    source_markers = {}
    target_markers = {}
    for ct in source_types:
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
        pid_set = p.gene_progs[m]
        seen = set()
        for pid in pid_set:
            for gene in p.prog_genes.get(pid, ()):
                if gene not in seen:
                    seen.add(gene)
                    gene_target_conn[gene] += 1

    gene_source_conn = defaultdict(int)
    for m in source_markers_in_progs:
        pid_set = p.gene_progs[m]
        seen = set()
        for pid in pid_set:
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

    gtex_tissues = []
    for ct in target_types:
        gtex_tissues.extend(CELL_GTEX_MAP.get(ct, []))
    gtex_tissues = list(set(gtex_tissues))

    source_gtex = CELL_GTEX_MAP.get(source_types[0] if source_types else '', [])

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

    composites = {}
    for gene, progs in p.gene_progs.items():
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

        directional = max(target_conn_frac - w['src_pen'] * source_conn_frac, 0)

        composite = (w['w_dir'] * directional +
                     w['w_act'] * act_precision +
                     w['w_frac'] * frac_enriched +
                     w['w_enr'] * min(mean_enr / 5, 1) +
                     w['w_gtex'] * max(gtex_score, 0) +
                     w['w_tau'] * tau_score +
                     w['w_pheno'] * pheno_s +
                     w['w_kern'] * kern_s +
                     w['w_temp'] * temp_s)

        composites[gene] = composite

    return composites


def parse_factors(raw_factors):
    parts = [p.strip() for p in raw_factors.split(';')]
    gene_factors = []
    for p in parts:
        if p.startswith('miR-') or p.startswith('miR_'):
            continue
        if 'chemical' in p.lower() or 'Chemical' in p:
            continue
        if p == 'various':
            continue
        p = p.replace(' + 2C inhibitors', '').replace(' + chemical', '').strip()
        if p:
            gene_factors.append(ALIAS_MAP.get(p, p))
    return gene_factors


def run_validation():
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    predictor = CocktailPredictor()
    predictor.load()

    print("Loading phenotype, temporal, and kernel data...")
    gene_pheno = load_phenotype_data()
    gene_temporal = load_temporal_data()
    M_res, kernel_gene_idx, kernel_depts = load_kernel_data()
    print(f"  Phenotype: {len(gene_pheno):,} genes, Temporal: {len(gene_temporal):,} genes, Kernel: {len(kernel_gene_idx):,} genes")

    n_regulatory = sum(1 for g in predictor.gene_progs
                       if g in predictor.regulatory_genes
                       and predictor.tf_tier.get(g, 0) >= 2)
    print(f"\nTier 2+ regulatory genes: {n_regulatory:,}")

    transition_cache = {}
    all_results = []
    skipped = []
    all_factor_entries = []
    family_results = defaultdict(list)

    for row in rows:
        cid = row['cocktail_id']
        target_cell = row['target_cell']
        source_cell = row['source_cell']

        gene_factors = parse_factors(row['factors'])
        if not gene_factors:
            skipped.append((cid, 'no gene factors'))
            continue

        target_types = TARGET_CELL_MAP.get(target_cell)
        source_type = SOURCE_CELL_MAP.get(source_cell)
        if target_types is None:
            skipped.append((cid, f'unmappable target: {target_cell}'))
            continue
        if source_type is None:
            skipped.append((cid, f'unmappable source: {source_cell}'))
            continue

        cache_key = (source_type, tuple(sorted(target_types)))
        if cache_key not in transition_cache:
            t0 = time.time()
            tier_w = get_calibrated_weights(target_types)
            composites = fast_score_candidates(predictor, [source_type], target_types,
                                                gene_pheno=gene_pheno, gene_temporal=gene_temporal,
                                                M_res=M_res, kernel_gene_idx=kernel_gene_idx,
                                                weights=tier_w)
            ranked_reg = sorted(
                [(g, s) for g, s in composites.items()
                 if g in predictor.regulatory_genes and predictor.tf_tier.get(g, 0) >= 2],
                key=lambda x: -x[1]
            )
            reg_genes = [g for g, _ in ranked_reg]
            ranked_all = sorted(composites.items(), key=lambda x: -x[1])
            all_genes = [g for g, _ in ranked_all]
            transition_cache[cache_key] = {
                'reg_genes': reg_genes,
                'all_genes': all_genes,
                'composites': composites,
                'n_reg': len(reg_genes),
            }
            print(f"  Scored {cache_key[0]}→{','.join(cache_key[1])} "
                  f"in {time.time()-t0:.1f}s ({len(reg_genes)} TFs)")

        tc = transition_cache[cache_key]

        factor_ranks = {}
        for f in gene_factors:
            if f in tc['reg_genes']:
                rank = tc['reg_genes'].index(f) + 1
                pctile = rank / tc['n_reg'] * 100
                factor_ranks[f] = {'rank': rank, 'percentile': pctile, 'found': True}
                all_factor_entries.append({
                    'factor': f, 'cocktail': cid, 'target': target_cell,
                    'source': source_cell, 'rank': rank, 'percentile': pctile,
                    'family': FAMILY_MAP.get(target_cell, 'Other'),
                })
            elif f in tc['composites']:
                rank_all = tc['all_genes'].index(f) + 1 if f in tc['all_genes'] else -1
                factor_ranks[f] = {
                    'found': False,
                    'reason': f'not Tier2+ (tier={predictor.tf_tier.get(f, 0)})',
                    'rank_all_genes': rank_all,
                    'total_genes': len(tc['all_genes']),
                }
            else:
                factor_ranks[f] = {
                    'found': False,
                    'reason': 'not in program map',
                    'in_gene_progs': f in predictor.gene_progs,
                }

        result = {
            'cocktail_id': cid, 'target_cell': target_cell,
            'source_cell': source_cell, 'family': FAMILY_MAP.get(target_cell, 'Other'),
            'known_factors': gene_factors, 'n_factors': len(gene_factors),
            'factor_ranks': factor_ranks,
            'year': int(row.get('year', 0)) if row.get('year', '').isdigit() else 0,
            'first_author': row.get('first_author', ''),
            'journal': row.get('journal', ''),
        }
        all_results.append(result)
        family_results[FAMILY_MAP.get(target_cell, 'Other')].append(result)

    print(f"\n{'='*100}")
    print(f"  77-COCKTAIL VALIDATION — COMPREHENSIVE RESULTS")
    print(f"{'='*100}")
    print(f"\n  Cocktails tested: {len(all_results)}")
    print(f"  Cocktails skipped: {len(skipped)}")
    for cid, reason in skipped:
        print(f"    {cid}: {reason}")

    n_total = 0
    n_in_pool = len(all_factor_entries)
    n_not_in_pool = 0
    not_in_pool = set()
    for result in all_results:
        for f, fr in result['factor_ranks'].items():
            n_total += 1
            if not fr.get('found'):
                n_not_in_pool += 1
                not_in_pool.add((f, fr.get('reason', 'unknown')))

    print(f"\n  Total factor-cocktail tests: {n_total}")
    print(f"  In Tier 2+ regulatory pool: {n_in_pool}")
    print(f"  Not in pool: {n_not_in_pool}")

    thresholds = [
        ('Top 1%', 1), ('Top 2%', 2), ('Top 5%', 5),
        ('Top 10%', 10), ('Top 20%', 20),
    ]

    print(f"\n  {'─'*90}")
    print(f"  AGGREGATE FACTOR RECOVERY (factor-cocktail pairs in Tier 2+ pool)")
    print(f"  {'─'*90}")
    for label, thresh in thresholds:
        count = sum(1 for e in all_factor_entries if e['percentile'] <= thresh)
        pct = count / n_in_pool * 100 if n_in_pool else 0
        bar = '█' * int(pct / 2)
        print(f"  {label:>8s}: {count:>4d}/{n_in_pool} ({pct:5.1f}%) {bar}")

    unique_factors = {}
    for e in all_factor_entries:
        f = e['factor']
        if f not in unique_factors or e['percentile'] < unique_factors[f]['percentile']:
            unique_factors[f] = e
    sorted_unique = sorted(unique_factors.values(), key=lambda x: x['percentile'])

    print(f"\n  {'─'*90}")
    print(f"  UNIQUE FACTOR RECOVERY (best rank across all cocktails, {len(sorted_unique)} unique factors)")
    print(f"  {'─'*90}")
    for label, thresh in thresholds:
        count = sum(1 for f in sorted_unique if f['percentile'] <= thresh)
        pct = count / len(sorted_unique) * 100 if sorted_unique else 0
        print(f"  {label:>8s}: {count:>3d}/{len(sorted_unique)} ({pct:5.1f}%)")

    print(f"\n  Top 30 factors (by best rank):")
    for e in sorted_unique[:30]:
        print(f"    {e['factor']:12s}: rank {e['rank']:>4d}/{n_regulatory:,} ({e['percentile']:5.2f}%) "
              f"[{e['source']}→{e['target']}]")

    if len(sorted_unique) > 30:
        print(f"\n  Bottom 10 factors:")
        for e in sorted_unique[-10:]:
            print(f"    {e['factor']:12s}: rank {e['rank']:>4d}/{n_regulatory:,} ({e['percentile']:5.2f}%) "
                  f"[{e['source']}→{e['target']}]")

    if not_in_pool:
        print(f"\n  Factors NOT in Tier 2+ pool ({len(not_in_pool)} unique):")
        for f, reason in sorted(not_in_pool):
            print(f"    {f:15s}: {reason}")

    print(f"\n  {'─'*90}")
    print(f"  PER-COCKTAIL DETAIL")
    print(f"  {'─'*90}")
    for result in all_results:
        factors = result['known_factors']
        ranks = result['factor_ranks']
        found_ranks = [(f, ranks[f]) for f in factors if ranks[f].get('found')]
        in_top10 = sum(1 for f, r in found_ranks if r['percentile'] <= 10)
        total_rankable = len(found_ranks)

        parts = []
        for f in factors:
            r = ranks[f]
            if r.get('found'):
                parts.append(f"{f}(#{r['rank']})")
            else:
                parts.append(f"{f}[×]")

        print(f"  {result['cocktail_id']:12s} {result['source_cell']:15s}→"
              f"{result['target_cell']:25s} {in_top10}/{total_rankable} top10%  "
              f"{', '.join(parts)}")

    print(f"\n  {'─'*90}")
    print(f"  FAMILY BREAKDOWN")
    print(f"  {'─'*90}")

    for family in sorted(family_results.keys()):
        fam_entries = [e for e in all_factor_entries if e['family'] == family]
        n_cocktails = len(family_results[family])
        if not fam_entries:
            print(f"\n  {family} ({n_cocktails} cocktails, 0 rankable factors)")
            continue
        top5 = sum(1 for e in fam_entries if e['percentile'] <= 5)
        top10 = sum(1 for e in fam_entries if e['percentile'] <= 10)
        median_p = np.median([e['percentile'] for e in fam_entries])
        print(f"\n  {family} ({n_cocktails} cocktails, {len(fam_entries)} factor tests):")
        print(f"    Top 5%:  {top5}/{len(fam_entries)} ({top5/len(fam_entries)*100:.0f}%)")
        print(f"    Top 10%: {top10}/{len(fam_entries)} ({top10/len(fam_entries)*100:.0f}%)")
        print(f"    Median percentile: {median_p:.1f}%")

    print(f"\n  {'─'*90}")
    print(f"  SOURCE CELL COMPARISON")
    print(f"  {'─'*90}")
    source_groups = defaultdict(list)
    for e in all_factor_entries:
        source_groups[e['source']].append(e)
    for src in sorted(source_groups.keys()):
        entries = source_groups[src]
        top10 = sum(1 for e in entries if e['percentile'] <= 10)
        median = np.median([e['percentile'] for e in entries])
        print(f"  {src:20s}: {top10}/{len(entries)} top 10% (median {median:.1f}%)")

    output = {
        'summary': {
            'cocktails_tested': len(all_results),
            'cocktails_skipped': len(skipped),
            'total_factor_tests': n_total,
            'factors_in_pool': n_in_pool,
            'factors_not_in_pool': n_not_in_pool,
            'n_regulatory_genes': n_regulatory,
        },
        'aggregate_thresholds': {l: {
            'count': sum(1 for e in all_factor_entries if e['percentile'] <= t),
            'total': n_in_pool,
            'pct': sum(1 for e in all_factor_entries if e['percentile'] <= t) / n_in_pool * 100 if n_in_pool else 0,
        } for l, t in thresholds},
        'unique_factor_thresholds': {l: {
            'count': sum(1 for f in sorted_unique if f['percentile'] <= t),
            'total': len(sorted_unique),
            'pct': sum(1 for f in sorted_unique if f['percentile'] <= t) / len(sorted_unique) * 100 if sorted_unique else 0,
        } for l, t in thresholds},
        'unique_factors': sorted_unique,
        'not_in_pool': [{'factor': f, 'reason': r} for f, r in sorted(not_in_pool)],
        'family_summary': {
            fam: {
                'n_cocktails': len(family_results[fam]),
                'n_factor_tests': len([e for e in all_factor_entries if e['family'] == fam]),
                'top5_pct': sum(1 for e in all_factor_entries if e['family'] == fam and e['percentile'] <= 5) / max(len([e for e in all_factor_entries if e['family'] == fam]), 1) * 100,
                'top10_pct': sum(1 for e in all_factor_entries if e['family'] == fam and e['percentile'] <= 10) / max(len([e for e in all_factor_entries if e['family'] == fam]), 1) * 100,
                'median_percentile': float(np.median([e['percentile'] for e in all_factor_entries if e['family'] == fam])) if [e for e in all_factor_entries if e['family'] == fam] else None,
            } for fam in sorted(family_results.keys())
        },
        'per_cocktail': all_results,
        'skipped': [{'cocktail_id': c, 'reason': r} for c, r in skipped],
    }

    out_path = os.path.join(os.path.dirname(__file__), 'validation_77_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return output


if __name__ == '__main__':
    t0 = time.time()
    run_validation()
    print(f"\n  Total runtime: {time.time()-t0:.1f}s")
