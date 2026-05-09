#!/usr/bin/env python3
"""Re-run strict factor holdout with 3,166 Tier 2+ denominator."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))

from vm_cocktail_predictor import CocktailPredictor
from validate_77_cocktails import (parse_factors as pf77,
    TARGET_CELL_MAP as TCM1, SOURCE_CELL_MAP as SCM1,
    get_calibrated_weights, compute_phenotype_score, compute_temporal_score)
from validate_extended_cocktails import (TARGET_CELL_MAP as TCM2,
    SOURCE_CELL_MAP as SCM2, parse_factors as pf_ext)
from calibrate_weights import load_phenotype_data, load_temporal_data, load_kernel_data
from reviewer_response_tests import load_set, compute_components, reweight, eval_w

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_77 = os.path.join(ROOT, 'attached_assets', 'published_reprogramming_cocktails_1774053006993.csv')
CSV_EXT = os.path.join(ROOT, 'attached_assets', 'additional_reprogramming_cocktails_1774066435911.csv')

print("Loading predictor...", flush=True)
predictor = CocktailPredictor()
predictor.load()
gene_pheno = load_phenotype_data()
gene_temporal = load_temporal_data()
M_res, kernel_gene_idx, _ = load_kernel_data()

s1 = load_set(CSV_77, TCM1, SCM1, pf77, 'source_cell')
s2 = load_set(CSV_EXT, TCM2, SCM2, pf_ext, 'source_cell')
print(f"Set1={len(s1)}, Set2={len(s2)}", flush=True)

ukeys = set()
for c in s1+s2:
    ukeys.add((c['source'], tuple(sorted(c['target_types']))))
print(f"{len(ukeys)} transitions to compute", flush=True)

t0 = time.time()
comp_cache = {}
for i, key in enumerate(ukeys):
    comp_cache[key] = compute_components(predictor, key[0], list(key[1]),
                                          gene_pheno, gene_temporal, M_res, kernel_gene_idx)
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(ukeys)} ({time.time()-t0:.0f}s)", flush=True)
print(f"Components done in {time.time()-t0:.0f}s", flush=True)

reg_t2 = set(g for g in predictor.regulatory_genes if predictor.tf_tier.get(g, 0) >= 2)
print(f"Tier 2+ denominator: {len(reg_t2)} genes", flush=True)

full_s1 = eval_w([('S1', s1)], comp_cache, get_calibrated_weights, reg_t2)
full_s2 = eval_w([('S2', s2)], comp_cache, get_calibrated_weights, reg_t2)
print(f"\nFull Set 1: {full_s1['S1']['top5']:.1f}% ({full_s1['S1']['h5']}/{full_s1['S1']['n']})", flush=True)
print(f"Full Set 2: {full_s2['S2']['top5']:.1f}% ({full_s2['S2']['h5']}/{full_s2['S2']['n']})", flush=True)

f1=set(); f2=set()
for c in s1: f1.update(c['factors'])
for c in s2: f2.update(c['factors'])
o12 = f1 & f2
print(f"\nShared S1∩S2: {len(o12)}", flush=True)
print(f"Novel S2 genes: {len(f2 - o12)}", flush=True)

strict = eval_w([('S2', s2)], comp_cache, get_calibrated_weights, reg_t2, exclude=o12)
delta = full_s2['S2']['top5'] - strict['S2']['top5']
print(f"\nStrict Set 2: {strict['S2']['top5']:.1f}% ({strict['S2']['h5']}/{strict['S2']['n']})", flush=True)
print(f"Delta: {delta:+.1f}pp", flush=True)

results = {
    "denominator": f"{len(reg_t2)} Tier 2+ transcription factors",
    "full_set1": {"top5_pct": round(full_s1['S1']['top5'],1), "hits": full_s1['S1']['h5'], "total": full_s1['S1']['n']},
    "full_set2": {"top5_pct": round(full_s2['S2']['top5'],1), "hits": full_s2['S2']['h5'], "total": full_s2['S2']['n']},
    "strict_holdout_set2": {"top5_pct": round(strict['S2']['top5'],1), "hits": strict['S2']['h5'],
        "total_instances": strict['S2']['n'], "unique_novel_genes": len(f2-o12), "shared_excluded": len(o12)},
    "delta_pp": round(delta, 1),
    "shared_factors": sorted(o12)
}
outpath = os.path.join(os.path.dirname(__file__), 'v2_submission', 'raw_results', 'strict_holdout_3166.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {outpath}", flush=True)
print("DONE", flush=True)
