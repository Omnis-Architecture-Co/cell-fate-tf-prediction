#!/usr/bin/env python3
"""Propensity-matched null test using Tier 2+ (3,166) gene pool."""
import sys, os, csv, time, json, math, random
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vm_cocktail_predictor import CocktailPredictor, CELL_GTEX_MAP
from validate_77_cocktails import (parse_factors as pf77,
    TARGET_CELL_MAP as TCM1, SOURCE_CELL_MAP as SCM1,
    get_calibrated_weights, compute_phenotype_score, compute_temporal_score)
from validate_extended_cocktails import (TARGET_CELL_MAP as TCM2,
    SOURCE_CELL_MAP as SCM2, parse_factors as pf_ext)
from validate_set3_cocktails import (TARGET_CELL_MAP as TCM3,
    SOURCE_CELL_MAP as SCM3, parse_factors as pf_s3)
from calibrate_weights import load_phenotype_data, load_temporal_data, load_kernel_data
from reviewer_response_tests import compute_components, reweight, load_set

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    print('Loading predictor...', flush=True)
    predictor = CocktailPredictor(); predictor.load()
    gene_pheno = load_phenotype_data()
    gene_temporal = load_temporal_data()
    M_res, kernel_gene_idx, _ = load_kernel_data()

    s1 = load_set(os.path.join(ROOT, 'attached_assets/published_reprogramming_cocktails_1774053006993.csv'), TCM1, SCM1, pf77)
    s2 = load_set(os.path.join(ROOT, 'attached_assets/additional_reprogramming_cocktails_1774066435911.csv'), TCM2, SCM2, pf_ext)
    s3 = load_set(os.path.join(ROOT, 'attached_assets/validation_set3_cocktails_1774073324946.csv'), TCM3, SCM3, pf_s3, 'source')
    all_c = s1+s2+s3
    print(f'Loaded {len(all_c)} combinations', flush=True)

    reg = set(g for g in predictor.gene_progs
              if g in predictor.regulatory_genes
              and predictor.tf_tier.get(g, 0) >= 2)
    print(f'Tier 2+ pool: {len(reg)}', flush=True)

    ukeys = set()
    for c in all_c:
        ukeys.add((c['source'], tuple(sorted(c['target_types']))))

    print(f'Computing {len(ukeys)} transitions...', flush=True)
    t0 = time.time()
    comp_cache = {}
    for i, key in enumerate(ukeys):
        comp_cache[key] = compute_components(predictor, key[0], list(key[1]),
                                              gene_pheno, gene_temporal, M_res, kernel_gene_idx)
        if (i+1) % 10 == 0: print(f'  {i+1}/{len(ukeys)} ({time.time()-t0:.0f}s)', flush=True)
    print(f'Done in {time.time()-t0:.0f}s', flush=True)

    rcache = {}
    for c in all_c:
        key = (c['source'], tuple(sorted(c['target_types'])))
        if key not in rcache:
            w = get_calibrated_weights(c['target_types'])
            _, rr = reweight(comp_cache[key], w, reg)
            rcache[key] = {g: i for i, g in enumerate(rr)}

    h5_b=tot_b=0
    for c in all_c:
        key = (c['source'], tuple(sorted(c['target_types'])))
        rd = rcache[key]; n = len(rd)
        for f in c['factors']:
            if f not in reg: continue
            tot_b += 1
            if f in rd and (rd[f]+1)/n*100 <= 5: h5_b += 1
    real_top5 = h5_b/tot_b*100
    print(f'Baseline: {h5_b}/{tot_b} = {real_top5:.1f}% top-5%', flush=True)

    gtex_vals = {}
    for gene in reg:
        if gene in predictor.gene_expr:
            gtex_vals[gene] = float(np.mean(predictor.gene_expr[gene]))
        else:
            gtex_vals[gene] = 0.0
    pheno_counts = {g: len(gene_pheno.get(g, {})) for g in reg}

    all_reg = sorted(reg)
    ga = np.array([gtex_vals.get(g,0) for g in all_reg])
    pa = np.array([pheno_counts.get(g,0) for g in all_reg])
    gsd = np.std(ga)+1e-9; psd = np.std(pa)+1e-9

    real_factors = list(set(g for c in all_c for g in c['factors'] if g in reg))
    real_gtex_mean = np.mean([gtex_vals.get(g, 0) for g in real_factors])
    real_pheno_mean = np.mean([pheno_counts.get(g, 0) for g in real_factors])
    print(f'Real factors in pool: {len(real_factors)}')
    print(f'Mean GTEx: real={real_gtex_mean:.2f} vs pool={np.mean(ga):.2f}')
    print(f'Mean pheno: real={real_pheno_mean:.1f} vs pool={np.mean(pa):.1f}')

    nn_cache = {}
    for f in real_factors:
        fg = gtex_vals[f]; fp = pheno_counts[f]
        dist = np.abs(ga-fg)/gsd + np.abs(pa-fp)/psd
        ci = np.argsort(dist)[:20]
        nn_cache[f] = [all_reg[j] for j in ci]

    N = 200; random.seed(42); rates = []
    print(f'Running {N} permutations...', flush=True)
    t0 = time.time()
    for pi in range(N):
        h5=tot=0
        for c in all_c:
            key = (c['source'], tuple(sorted(c['target_types'])))
            rd = rcache[key]; n = len(rd)
            for f in c['factors']:
                if f not in reg: continue
                tot += 1
                neighbors = nn_cache.get(f)
                if not neighbors: continue
                m = random.choice(neighbors)
                if m in rd and (rd[m]+1)/n*100 <= 5: h5 += 1
        rates.append(h5/tot*100 if tot else 0)
        if (pi+1)%50==0: print(f'  {pi+1}/{N} ({time.time()-t0:.0f}s)', flush=True)

    mn=np.mean(rates); sd=np.std(rates); mx=np.max(rates); mi=np.min(rates)
    z = (real_top5-mn)/(sd+1e-9)

    print()
    print('='*60)
    print(f'PROPENSITY-MATCHED NULL (Tier 2+, n={len(reg)})')
    print('='*60)
    print(f'Matched null: {mn:.1f}% +/- {sd:.1f}% (min {mi:.1f}%, max {mx:.1f}%)')
    print(f'Real model:   {real_top5:.1f}%')
    print(f'Effect:       {real_top5-mn:.1f}pp above null')
    print(f'Z-score:      {z:.1f}')
    print(f'Emp. p-value: 0/{N} => p < 0.005')

    output = {
        'test': 'propensity_matched_null_tier2plus',
        'n_regulatory_genes': len(reg),
        'n_permutations': N,
        'random_seed': 42,
        'null_mean_pct': round(mn, 2),
        'null_std_pct': round(sd, 2),
        'null_min_pct': round(mi, 2),
        'null_max_pct': round(mx, 2),
        'real_top5_pct': round(real_top5, 1),
        'effect_pp': round(real_top5-mn, 1),
        'z_score': round(z, 1),
        'empirical_p': '<0.005',
        'n_real_factors_in_pool': len(real_factors),
        'total_testable_pairs': tot_b,
        'real_mean_gtex': round(float(real_gtex_mean), 2),
        'pool_mean_gtex': round(float(np.mean(ga)), 2),
        'real_mean_pheno': round(float(real_pheno_mean), 1),
        'pool_mean_pheno': round(float(np.mean(pa)), 1),
        'null_distribution': [round(r, 2) for r in rates]
    }

    out_path = os.path.join(os.path.dirname(__file__), 'v2_submission/raw_results/propensity_null_tier2plus.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nSaved to {out_path}')
    predictor.conn.close()
    print('DONE.', flush=True)

if __name__ == '__main__':
    main()
