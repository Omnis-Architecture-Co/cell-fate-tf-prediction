#!/usr/bin/env python3
"""Reviewer-response ablation tests for VM Cocktail Predictor."""
import sys, os, csv, time, json, math, random
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from vm_cocktail_predictor import (CocktailPredictor, TIER_A_WEIGHTS,
    TIER_B_WEIGHTS, get_tier_weights, CELL_GTEX_MAP)
from validate_77_cocktails import (parse_factors as pf77,
    TARGET_CELL_MAP as TCM1, SOURCE_CELL_MAP as SCM1,
    get_calibrated_weights, compute_phenotype_score, compute_temporal_score)
from validate_extended_cocktails import (TARGET_CELL_MAP as TCM2,
    SOURCE_CELL_MAP as SCM2, parse_factors as pf_ext)
from validate_set3_cocktails import (TARGET_CELL_MAP as TCM3,
    SOURCE_CELL_MAP as SCM3, parse_factors as pf_s3)
from calibrate_weights import load_phenotype_data, load_temporal_data, load_kernel_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_77 = os.path.join(ROOT, 'attached_assets', 'published_reprogramming_cocktails_1774053006993.csv')
CSV_EXT = os.path.join(ROOT, 'attached_assets', 'additional_reprogramming_cocktails_1774066435911.csv')
CSV_S3 = os.path.join(ROOT, 'attached_assets', 'validation_set3_cocktails_1774073324946.csv')

def load_set(csv_path, tcm, scm, parse_fn, source_col='source_cell'):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    cocktails = []
    for row in rows:
        target = row['target_cell'].strip()
        source = row.get(source_col, '').strip()
        tt = tcm.get(target); st = scm.get(source)
        if tt is None or st is None: continue
        factors = parse_fn(row['factors'])
        if not factors: continue
        cocktails.append({'source': st, 'target_types': tt, 'factors': factors, 'target': target})
    return cocktails

def compute_components(predictor, src, tgt, gene_pheno, gene_temporal, M_res, kernel_gene_idx):
    p = predictor
    source_markers = {}; target_markers = {}
    for ct in [src]:
        if ct in p.ct_markers: source_markers.update(p.ct_markers[ct])
    for ct in tgt:
        if ct in p.ct_markers: target_markers.update(p.ct_markers[ct])
    source_programs = set()
    for m in source_markers:
        if m in p.gene_progs: source_programs |= p.gene_progs[m]
    target_programs = set()
    for m in target_markers:
        if m in p.gene_progs: target_programs |= p.gene_progs[m]
    activate = target_programs - source_programs
    target_mp = set(m for m in target_markers if m in p.gene_progs)
    source_mp = set(m for m in source_markers if m in p.gene_progs)
    n_markers = len(target_mp); n_source = len(source_mp)
    gene_tc = defaultdict(int)
    for m in target_mp:
        seen = set()
        for pid in p.gene_progs[m]:
            for gene in p.prog_genes.get(pid, ()):
                if gene not in seen: seen.add(gene); gene_tc[gene] += 1
    gene_sc = defaultdict(int)
    for m in source_mp:
        seen = set()
        for pid in p.gene_progs[m]:
            for gene in p.prog_genes.get(pid, ()):
                if gene not in seen: seen.add(gene); gene_sc[gene] += 1
    prog_enr = {}
    for pid, genes in p.prog_genes.items():
        n_in = len(genes); n_mark = sum(1 for g in genes if g in target_mp)
        if n_mark > 0:
            expected = n_in * n_markers / p.n_genes if p.n_genes > 0 else 0
            prog_enr[pid] = n_mark / max(expected, 1e-10)
    gtex_t = list(set(t for ct in tgt for t in CELL_GTEX_MAP.get(ct, [])))
    gtex_s = CELL_GTEX_MAP.get(src, [])
    ti = [p.tissue_idx[t] for t in gtex_t if t in p.tissue_idx]
    si = [p.tissue_idx[t] for t in gtex_s if t in p.tissue_idx]
    tmg = set(target_markers.keys())
    kernel_sig = None
    if M_res is not None and kernel_gene_idx is not None:
        mk = [kernel_gene_idx[m] for m in tmg if m in kernel_gene_idx]
        if mk:
            kernel_sig = np.mean(M_res[mk], axis=0)
            ksn = np.linalg.norm(kernel_sig)
            kernel_sig = kernel_sig / ksn if ksn > 1e-10 else None
    comps = {}
    for gene, progs in p.gene_progs.items():
        np_ = len(progs)
        act = len(progs & activate) / max(np_, 1)
        tf = gene_tc.get(gene, 0) / max(n_markers, 1)
        sf = gene_sc.get(gene, 0) / max(n_source, 1)
        ne = 0; sle = 0.0
        for pid in progs:
            e = prog_enr.get(pid, 0)
            if e > 1.0: ne += 1; sle += math.log2(e)
        fe = ne / max(np_, 1)
        me = sle / ne if ne > 0 else 0
        gs = 0.0
        if gene in p.gene_expr and ti:
            expr = p.gene_expr[gene]
            te = sum(expr[i] for i in ti) / len(ti)
            ae = float(expr.mean())
            se = sum(expr[i] for i in si) / len(si) if si else ae
            if ae > 0 and te > 1:
                ratio = te / max(ae, 0.1)
                gs = min(math.log2(max(ratio, 1)), 5) / 5
                if se > 0 and te > se: gs = min(gs * 1.3, 1.0)
        tau = min(p.gene_tau.get(gene, 0.5) / 0.9, 1.0)
        ph = compute_phenotype_score(gene, tgt, gene_pheno) if gene_pheno else 0.0
        ks = 0.0
        if kernel_sig is not None and gene in kernel_gene_idx:
            gr = M_res[kernel_gene_idx[gene]]
            gn = np.linalg.norm(gr)
            if gn > 1e-10: ks = max(np.dot(gr, kernel_sig) / gn, 0.0)
        ts = compute_temporal_score(gene, tgt, gene_temporal) if gene_temporal else 0.0
        comps[gene] = (tf, sf, act, fe, me, gs, tau, ph, ks, ts)
    return comps

def reweight(comps, w, reg):
    sp = w['src_pen']
    scored = {}
    for g, (tf,sf,act,fe,me,gs,tau,ph,ks,ts) in comps.items():
        d = max(tf - sp*sf, 0)
        scored[g] = (w['w_dir']*d + w['w_act']*act + w['w_frac']*fe +
                     w['w_enr']*min(me/5,1) + w['w_gtex']*max(gs,0) +
                     w['w_tau']*tau + w['w_pheno']*ph + w['w_kern']*ks + w['w_temp']*ts)
    rr = sorted(((g,s) for g,s in scored.items() if g in reg), key=lambda x:-x[1])
    return scored, [g for g,_ in rr]

def eval_w(sets, comp_cache, wfn, reg, exclude=None):
    results = {}
    for name, cktls in sets:
        h5=h10=tot=0; cache={}
        for c in cktls:
            key = (c['source'], tuple(sorted(c['target_types'])))
            if key not in cache:
                w = wfn(c['target_types'])
                _, rr = reweight(comp_cache[key], w, reg)
                cache[key] = rr
            rr = cache[key]; n = len(rr)
            for f in c['factors']:
                if exclude and f in exclude: continue
                if f not in comp_cache[key]: continue
                tot += 1
                idx = rr.index(f) if f in rr else n
                pct = (idx+1)/n*100
                if pct<=5: h5+=1
                if pct<=10: h10+=1
        results[name] = {'top5':h5/tot*100 if tot else 0, 'top10':h10/tot*100 if tot else 0, 'n':tot, 'h5':h5, 'h10':h10}
    return results

def main():
    print("Loading predictor...", flush=True)
    predictor = CocktailPredictor(); predictor.load()
    gene_pheno = load_phenotype_data()
    gene_temporal = load_temporal_data()
    M_res, kernel_gene_idx, _ = load_kernel_data()

    s1 = load_set(CSV_77, TCM1, SCM1, pf77, 'source_cell')
    s2 = load_set(CSV_EXT, TCM2, SCM2, pf_ext, 'source_cell')
    s3 = load_set(CSV_S3, TCM3, SCM3, pf_s3, 'source')
    print(f"Loaded: Set1={len(s1)}, Set2={len(s2)}, Set3={len(s3)}", flush=True)

    all_c = s1 + s2 + s3
    all_sets = [('Set 1', s1), ('Set 2', s2), ('Set 3', s3)]
    ukeys = set()
    for c in all_c: ukeys.add((c['source'], tuple(sorted(c['target_types']))))
    print(f"{len(ukeys)} unique transitions", flush=True)

    t0 = time.time()
    comp_cache = {}
    for i, key in enumerate(ukeys):
        comp_cache[key] = compute_components(predictor, key[0], list(key[1]),
                                              gene_pheno, gene_temporal, M_res, kernel_gene_idx)
        if (i+1) % 10 == 0: print(f"  {i+1}/{len(ukeys)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"Component scoring: {len(ukeys)} transitions in {time.time()-t0:.0f}s", flush=True)

    reg = predictor.regulatory_genes

    rb = eval_w(all_sets, comp_cache, get_calibrated_weights, reg)
    print("\n" + "="*80)
    print("  BASELINE (2-tier)")
    print("="*80)
    for s, d in rb.items():
        print(f"  {s}: Top-5%={d['top5']:.1f}% ({d['h5']}/{d['n']})  Top-10%={d['top10']:.1f}%")

    # TEST 1
    def ksw(tgt):
        b = get_calibrated_weights(tgt).copy()
        b['w_gtex']=0; b['w_pheno']=0; b['w_tau']=0; b['w_temp']=0
        tw = sum(b[k] for k in ['w_dir','w_act','w_frac','w_enr','w_kern'])
        for k in ['w_dir','w_act','w_frac','w_enr','w_kern']: b[k]/=tw
        return b
    rk = eval_w(all_sets, comp_cache, ksw, reg)
    rka = eval_w(all_sets, comp_cache,
                 lambda t: {'w_dir':0,'w_act':0,'w_frac':0,'w_enr':0,'w_gtex':0,
                            'w_tau':0,'w_pheno':0,'w_kern':1.0,'w_temp':0,'src_pen':0.27}, reg)
    print("\n" + "="*80)
    print("  TEST 1: KERNEL-ONLY STACK (no GTEx/pheno/tau/temporal)")
    print("="*80)
    for s in ['Set 1','Set 2','Set 3']:
        print(f"  {s}: Stack={rk[s]['top5']:.1f}%/{rk[s]['top10']:.1f}%  "
              f"Alone={rka[s]['top5']:.1f}%/{rka[s]['top10']:.1f}%  "
              f"[full: {rb[s]['top5']:.1f}%/{rb[s]['top10']:.1f}%]")

    # TEST 2
    avg_w = {k:(TIER_A_WEIGHTS[k]+TIER_B_WEIGHTS[k])/2 for k in TIER_A_WEIGHTS}
    ra = eval_w(all_sets, comp_cache, lambda t: avg_w, reg)
    print("\n" + "="*80)
    print("  TEST 2: SINGLE-TIER vs TWO-TIER")
    print("="*80)
    for s in ['Set 1','Set 2','Set 3']:
        d = ra[s]['top5']-rb[s]['top5']
        print(f"  {s}: 2-tier={rb[s]['top5']:.1f}%  1-tier={ra[s]['top5']:.1f}%  delta={d:+.1f}pp")

    # TEST 3
    f1=set(); f2=set(); f3=set()
    for c in s1: f1.update(c['factors'])
    for c in s2: f2.update(c['factors'])
    for c in s3: f3.update(c['factors'])
    o12=f1&f2; o13=f1&f3; o23=f2&f3; o123=f1&f2&f3
    r2x = eval_w([('S2',s2)], comp_cache, get_calibrated_weights, reg, exclude=o12)
    r3x = eval_w([('S3',s3)], comp_cache, get_calibrated_weights, reg, exclude=o13)
    r3xx = eval_w([('S3',s3)], comp_cache, get_calibrated_weights, reg, exclude=(o13|o23))
    print("\n" + "="*80)
    print("  TEST 3: FACTOR OVERLAP")
    print("="*80)
    print(f"  S1: {len(f1)} factors ({len(f1-f2-f3)} exclusive)")
    print(f"  S2: {len(f2)} factors ({len(f2-f1-f3)} exclusive)")
    print(f"  S3: {len(f3)} factors ({len(f3-f1-f2)} exclusive)")
    print(f"  S1∩S2: {len(o12)} {sorted(o12)}")
    print(f"  S1∩S3: {len(o13)} {sorted(o13)}")
    print(f"  S2∩S3: {len(o23)} {sorted(o23)}")
    print(f"  All 3: {len(o123)} {sorted(o123)}")
    print(f"  S2 strict (excl {len(o12)}): {r2x['S2']['top5']:.1f}% ({r2x['S2']['h5']}/{r2x['S2']['n']})  [full: {rb['Set 2']['top5']:.1f}%]")
    print(f"  S3 strict (excl {len(o13)}): {r3x['S3']['top5']:.1f}% ({r3x['S3']['h5']}/{r3x['S3']['n']})  [full: {rb['Set 3']['top5']:.1f}%]")
    print(f"  S3 ultra-strict (excl {len(o13|o23)}): {r3xx['S3']['top5']:.1f}% ({r3xx['S3']['h5']}/{r3xx['S3']['n']})")

    # TEST 4
    print("\n" + "="*80)
    print("  TEST 4: PROPENSITY-MATCHED NULL")
    print("="*80, flush=True)
    gtex_vals = {}
    for gene in reg:
        vals = [predictor.ct_gtex[ct][gene] for ct in predictor.ct_gtex if gene in predictor.ct_gtex[ct]]
        gtex_vals[gene] = np.mean(vals) if vals else 0.0
    pheno_counts = {g: len(gene_pheno.get(g, {})) for g in reg}
    all_reg = sorted(reg)
    ga = np.array([gtex_vals.get(g,0) for g in all_reg])
    pa = np.array([pheno_counts.get(g,0) for g in all_reg])
    gs = np.std(ga)+1e-9; ps = np.std(pa)+1e-9
    real_in = [f for f in set(g for c in all_c for g in c['factors']) if f in reg]
    print(f"  Real factors in reg: {len(real_in)}")
    print(f"  Mean GTEx real={np.mean([gtex_vals.get(g,0) for g in real_in]):.2f} vs all={np.mean(ga):.2f}")
    print(f"  Mean pheno real={np.mean([pheno_counts.get(g,0) for g in real_in]):.1f} vs all={np.mean(pa):.1f}")

    rcache = {}
    for key in comp_cache:
        w = get_calibrated_weights(list(key[1]))
        _, rr = reweight(comp_cache[key], w, reg)
        rcache[key] = rr

    N = 200; random.seed(42); rates = []
    t0 = time.time()
    for pi in range(N):
        h5=tot=0
        for c in all_c:
            key = (c['source'], tuple(sorted(c['target_types'])))
            rr = rcache[key]; n = len(rr)
            for f in c['factors']:
                if f not in reg or f not in comp_cache[key]: continue
                tot += 1
                fg = gtex_vals.get(f,0); fp = pheno_counts.get(f,0)
                dist = np.abs(ga-fg)/gs + np.abs(pa-fp)/ps
                ci = np.argsort(dist)[:20]
                m = all_reg[random.choice(ci)]
                if m in rr:
                    idx = rr.index(m)
                    if (idx+1)/n*100 <= 5: h5 += 1
        rates.append(h5/tot*100 if tot else 0)
        if (pi+1)%50==0: print(f"    {pi+1}/{N} ({time.time()-t0:.0f}s)", flush=True)

    mn=np.mean(rates); sd=np.std(rates); mx=np.max(rates)
    ov = sum(rb[s]['h5'] for s in rb) / sum(rb[s]['n'] for s in rb) * 100
    z = (ov-mn)/(sd+1e-9)
    print(f"\n  Matched null: {mn:.1f}% +/- {sd:.1f}% (max {mx:.1f}%)")
    print(f"  Real model:   {ov:.1f}%")
    print(f"  Effect size:  {ov-mn:.1f}pp above matched null")
    print(f"  Z-score:      {z:.1f}")

    output = {
        'baseline': {s: {'top5':d['top5'],'top10':d['top10'],'n':d['n']} for s,d in rb.items()},
        'test1_kernel_stack': {s: {'top5':d['top5'],'top10':d['top10'],'n':d['n']} for s,d in rk.items()},
        'test1_kernel_alone': {s: {'top5':d['top5'],'top10':d['top10'],'n':d['n']} for s,d in rka.items()},
        'test2_single_tier': {s: {'top5':d['top5'],'top10':d['top10'],'n':d['n']} for s,d in ra.items()},
        'test3_overlap': {
            's1':len(f1),'s2':len(f2),'s3':len(f3),
            'o12':len(o12),'o13':len(o13),'o23':len(o23),'all3':len(o123),
            'genes_12':sorted(o12),'genes_13':sorted(o13),'genes_all3':sorted(o123),
            's2_strict':{'top5':r2x['S2']['top5'],'n':r2x['S2']['n']},
            's3_strict':{'top5':r3x['S3']['top5'],'n':r3x['S3']['n']},
            's3_ultra':{'top5':r3xx['S3']['top5'],'n':r3xx['S3']['n']},
        },
        'test4_propensity_null': {
            'n_perm':N,'mean':mn,'std':sd,'max':mx,
            'real_top5':ov,'effect_pp':ov-mn,'z_score':z,
        }
    }
    out_path = os.path.join(ROOT, 'paper2', 'v2_submission', 'raw_results', 'reviewer_response_tests.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    predictor.conn.close()
    print(f"\nResults saved to {out_path}")
    print("ALL DONE.", flush=True)

if __name__ == '__main__':
    main()
