#!/usr/bin/env python3
"""
Extended Cocktail Validation (64 additional published cocktails)
================================================================
Validates the VM Cocktail Predictor against an expanded set of reprogramming
cocktails spanning hair cells, melanocytes, photoreceptors, immune subtypes,
lung, gut, kidney, thyroid, and more.
Uses calibrated 9-component weights.
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
from validate_77_cocktails import (
    fast_score_candidates, CALIBRATED_WEIGHTS,
    get_calibrated_weights, TIER_A_CELL_TYPES, TIER_A_WEIGHTS, TIER_B_WEIGHTS,
)

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         'attached_assets',
                         'additional_reprogramming_cocktails_1774066435911.csv')

ALIAS_MAP = {
    'OCT4': 'POU5F1', 'BRN2': 'POU3F2', 'PU.1': 'SPI1',
    'cFOS': 'FOS', 'cMYC': 'MYC', 'NURR1': 'NR4A2',
    'NGN2': 'NEUROG2', 'NGN1': 'NEUROG1', 'NGN3': 'NEUROG3',
    'HB9': 'MNX1', 'KROX20': 'EGR2', 'OSTERIX': 'SP7',
    'NKX2.2': 'NKX2-2', 'NKX6.2': 'NKX6-2',
    'BRN3B': 'POU4F2', 'AP2A': 'TFAP2A',
    'CTIP2': 'BCL11B', 'T-BET': 'TBX21',
    'BLIMP1': 'PRDM1', 'OSX': 'SP7',
    'SF1': 'NR5A1', 'SLUG': 'SNAI2',
    'GM-CSF': 'CSF2',
}

TARGET_CELL_MAP = {
    'Cochlear hair cell': ['HAIR_CELL'],
    'Outer hair cell': ['HAIR_CELL'],
    'Melanocyte': ['MELANOCYTE'],
    'Photoreceptor': ['PHOTORECEPTOR'],
    'Rod photoreceptor': ['PHOTORECEPTOR'],
    'Striatal MSN': ['NEURON_INHIBITORY'],
    'Cortical neuron': ['NEURON_EXCITATORY'],
    'Noradrenergic neuron': ['NEURON_EXCITATORY'],
    'Striatal neuron': ['NEURON_INHIBITORY'],
    'Nociceptor': ['NEURON_EXCITATORY'],
    'Astrocyte': ['ASTROCYTE'],
    'Astrocyte (reactive)': ['ASTROCYTE'],
    'Microglia-like': ['MICROGLIA'],
    'Pacemaker cell': ['CARDIOMYOCYTE'],
    'Cardiomyocyte': ['CARDIOMYOCYTE'],
    'Cardiomyocyte (matured)': ['CARDIOMYOCYTE'],
    'Endothelial': ['ENDOTHELIAL'],
    'Beta cell': ['BETA_CELL'],
    'Alpha cell': ['BETA_CELL'],
    'Enterocyte': ['ENTEROCYTE'],
    'Intestinal': ['ENTEROCYTE'],
    'Goblet cell': ['GOBLET_CELL'],
    'Tuft cell': None,
    'AT2 cell': ['PNEUMOCYTE_II'],
    'Club cell': None,
    'Thyrocyte': None,
    'Retinal pigment epi': ['MELANOCYTE'],
    'Retinal ganglion': ['NEURON_EXCITATORY'],
    'Lens fiber': None,
    'Corneal epithelial': ['KERATINOCYTE'],
    'Sebocyte': ['ADIPOCYTE'],
    'Eccrine sweat': None,
    'Chemosensory': None,
    'Enteroendocrine': ['BETA_CELL'],
    'Paneth cell': None,
    'NK cell': ['NK_CELL'],
    'Regulatory T cell': ['T_CELL'],
    'Th17 cell': ['T_CELL'],
    'Mast cell': ['HSC'],
    'Megakaryocyte': ['PLATELET'],
    'Plasma cell': ['B_CELL'],
    'T cell progenitor': ['T_CELL'],
    'Prostate epithelial': None,
    'Mammary epithelial': ['KERATINOCYTE'],
    'Urothelial': ['KERATINOCYTE'],
    'Pituitary': None,
    'Langerhans cell': ['DENDRITIC_CELL'],
    'MSC-like': ['FIBROBLAST'],
    'Tenocyte': ['FIBROBLAST'],
    'Ligament cell': ['CHONDROCYTE'],
    'Nucleus pulposus': ['CHONDROCYTE'],
    'Periosteal cell': ['OSTEOBLAST'],
    'Renal tubular': ['PODOCYTE'],
    'Podocyte': ['PODOCYTE'],
    'Leydig cell': None,
}

SOURCE_CELL_MAP = {
    'Fibroblast': 'FIBROBLAST',
    'Keratinocyte': 'KERATINOCYTE',
    'Supporting cell': 'ASTROCYTE',
    'Muller glia': 'ASTROCYTE',
    'Monocyte': 'MACROPHAGE',
    'NG2 glia': 'OLIGODENDROCYTE',
    'Enterocyte': 'ENTEROCYTE',
    'Duct cell': 'FIBROBLAST',
    'Gastric': 'ENTEROCYTE',
    'Intestinal stem': 'ENTEROCYTE',
    'AT2': 'PNEUMOCYTE_II',
    'ESC': None,
    'Amniotic': 'FIBROBLAST',
    'Oral ectoderm': 'KERATINOCYTE',
    'T cell': 'T_CELL',
    'CD4 T cell': 'T_CELL',
    'B cell': 'B_CELL',
    'HSC': 'HSC',
    'MSC': 'FIBROBLAST',
    'Iris epithelium': 'KERATINOCYTE',
    'Beta cell': 'BETA_CELL',
    'Respiratory': 'PNEUMOCYTE_II',
}

FAMILY_MAP = {
    'Cochlear hair cell': 'Sensory',
    'Outer hair cell': 'Sensory',
    'Melanocyte': 'Pigment',
    'Photoreceptor': 'Sensory',
    'Rod photoreceptor': 'Sensory',
    'Striatal MSN': 'Neuron',
    'Cortical neuron': 'Neuron',
    'Noradrenergic neuron': 'Neuron',
    'Striatal neuron': 'Neuron',
    'Nociceptor': 'Neuron',
    'Astrocyte': 'Glial',
    'Astrocyte (reactive)': 'Glial',
    'Microglia-like': 'Immune',
    'Pacemaker cell': 'Cardiac',
    'Cardiomyocyte': 'Cardiac',
    'Cardiomyocyte (matured)': 'Cardiac',
    'Endothelial': 'Vascular',
    'Beta cell': 'Endocrine',
    'Alpha cell': 'Endocrine',
    'Enterocyte': 'Gut',
    'Intestinal': 'Gut',
    'Goblet cell': 'Gut',
    'Tuft cell': 'Gut',
    'AT2 cell': 'Lung',
    'Club cell': 'Lung',
    'Thyrocyte': 'Endocrine',
    'Retinal pigment epi': 'Ocular',
    'Retinal ganglion': 'Ocular',
    'Lens fiber': 'Ocular',
    'Corneal epithelial': 'Ocular',
    'Sebocyte': 'Epithelial',
    'Eccrine sweat': 'Epithelial',
    'Chemosensory': 'Gut',
    'Enteroendocrine': 'Gut',
    'Paneth cell': 'Gut',
    'NK cell': 'Immune',
    'Regulatory T cell': 'Immune',
    'Th17 cell': 'Immune',
    'Mast cell': 'Immune',
    'Megakaryocyte': 'Blood',
    'Plasma cell': 'Immune',
    'T cell progenitor': 'Immune',
    'Prostate epithelial': 'Epithelial',
    'Mammary epithelial': 'Epithelial',
    'Urothelial': 'Epithelial',
    'Pituitary': 'Endocrine',
    'Langerhans cell': 'Immune',
    'MSC-like': 'Mesenchymal',
    'Tenocyte': 'Mesenchymal',
    'Ligament cell': 'Mesenchymal',
    'Nucleus pulposus': 'Mesenchymal',
    'Periosteal cell': 'Mesenchymal',
    'Renal tubular': 'Renal',
    'Podocyte': 'Renal',
    'Leydig cell': 'Reproductive',
}


def parse_factors(raw_factors):
    parts = [p.strip() for p in raw_factors.split(';')]
    gene_factors = []
    for p in parts:
        if p.startswith('miR-') or p.startswith('miR_'):
            continue
        if 'chemical' in p.lower() or 'pathway' in p.lower():
            continue
        if 'activation' in p.lower():
            continue
        if p in ('various', 'WNT'):
            continue
        p = p.replace(' + 2C inhibitors', '').replace(' + chemical', '').strip()
        p = p.replace(' + mechanical', '').replace(' (knockdown)', '').strip()
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
                }

        result = {
            'cocktail_id': cid,
            'target_cell': target_cell,
            'source_cell': source_cell,
            'vm_target': target_types,
            'vm_source': source_type,
            'known_factors': gene_factors,
            'factor_ranks': factor_ranks,
            'n_reg': tc['n_reg'],
            'species': row.get('species', ''),
            'year': row.get('year', ''),
            'journal': row.get('journal', ''),
        }
        all_results.append(result)
        family = FAMILY_MAP.get(target_cell, 'Other')
        family_results[family].append(result)

    print(f"\n{'='*100}")
    print(f"  EXTENDED VALIDATION — {len(rows)} COCKTAILS")
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

        print(f"  {result['cocktail_id']:18s} {result['source_cell']:18s}→"
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

    out_path = os.path.join(os.path.dirname(__file__), 'validation_extended_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return output


if __name__ == '__main__':
    t0 = time.time()
    run_validation()
    print(f"\n  Total runtime: {time.time()-t0:.1f}s")
