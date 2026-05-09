#!/usr/bin/env python3
"""
Chromosome 22 Holdout Test — ultimate leave-one-out validation.
================================================================

Protocol:
1. Remove all chr22 proteins from the dispatch graph
2. Rebuild dept_uids and token mappings without chr22
3. Test chr22 genes against the chr22-excluded kernel:
   a. Knockout disruption profiles — top-1/top-3 concordance
   b. Collinearity — within-primitive cosine for chr22 genes
   c. Primitive consistency — chr22 carriers vs non-chr22 carriers

Usage:
    python3 -u validation/knockout/chromosome_holdout_test.py
"""

import csv
import json
import os
import pickle
import sys
import time
import numpy as np
from collections import defaultdict, Counter
from scipy import stats

STATE_PATH = "/tmp/module8_full_state.pkl"
VOCAB_PATH = "server/data/human/vocabulary.csv"
GENE_DEPTS_PATH = "server/data/human/gene_departments.csv"
PRIMITIVES_PATH = "beta_transfer/genome_primitives.csv"
CHR22_GENES_PATH = "/tmp/chr22_genes.json"
FULL_PROFILES_PATH = "validation/knockout/disruption_profiles_full.json"
OUTPUT_PATH = "validation/knockout/chromosome_holdout_results.json"

VALID_DEPARTMENTS = sorted([
    "Apoptosis", "Cell adhesion", "Cell cycle", "Chromatin", "Cytoskeleton",
    "DNA repair", "GTPase", "Immune", "Ion channel", "Kinase", "Methylation",
    "Nuc acid bind", "Phosphatase", "Protein folding", "Proteolysis",
    "RNA processing", "Signaling", "Structural", "Transcription",
    "Translation", "Transport", "Ubiquitin",
])
D2I = {d: i for i, d in enumerate(VALID_DEPARTMENTS)}
N_DEPTS = len(VALID_DEPARTMENTS)


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    print("=" * 72)
    print("  CHROMOSOME 22 HOLDOUT TEST")
    print("  Remove chr22 from kernel, test chr22 genes against holdout kernel")
    print("=" * 72)
    t0 = time.time()

    with open(STATE_PATH, "rb") as f:
        state = pickle.load(f)
    ptt = state["ptt"]
    ttp = state["ttp"]
    gene_cache = state["gene_cache"]
    gene_to_uid = state["gene_to_uid"]

    gene_depts = {}
    with open(GENE_DEPTS_PATH) as f:
        for row in csv.DictReader(f):
            gene_depts[row["gene"]] = row["department"]

    with open(CHR22_GENES_PATH) as f:
        chr22_genes = set(json.load(f))

    vocab_dept = {}
    with open(VOCAB_PATH) as f:
        for row in csv.DictReader(f):
            vocab_dept[row["word_hex"].replace("0x", "").upper()] = row["primary_function"]

    gene_to_uids = defaultdict(list)
    for uid, g in gene_cache.items():
        if g:
            gene_to_uids[g].append(uid)

    chr22_uids = set()
    for gene in chr22_genes:
        for uid in gene_to_uids.get(gene, []):
            chr22_uids.add(uid)

    chr22_genes_in_kernel = chr22_genes & set(gene_to_uid.keys())
    chr22_genes_with_dept = {g for g in chr22_genes_in_kernel if g in gene_depts
                             and gene_depts[g] in D2I}

    print(f"\n  Chr22 genes total:         {len(chr22_genes)}")
    print(f"  Chr22 genes in kernel:     {len(chr22_genes_in_kernel)}")
    print(f"  Chr22 genes with dept:     {len(chr22_genes_with_dept)}")
    print(f"  Chr22 UIDs:                {len(chr22_uids)}")
    print(f"  Total UIDs:                {len(ptt)}")
    print(f"  Non-chr22 UIDs:            {len(ptt) - len(chr22_uids)}")

    holdout_ptt = {}
    for uid, toks in ptt.items():
        if uid not in chr22_uids:
            holdout_ptt[uid] = toks

    holdout_ttp = defaultdict(list)
    for uid, toks in holdout_ptt.items():
        for tok in toks:
            holdout_ttp[tok].append(uid)

    holdout_dept_uids = defaultdict(list)
    for uid, g in gene_cache.items():
        if uid in chr22_uids:
            continue
        if g and g in gene_depts:
            d = gene_depts[g]
            if d in D2I:
                holdout_dept_uids[d].append(uid)

    holdout_dept_tok_counts = {}
    for dept in VALID_DEPARTMENTS:
        d_uids = holdout_dept_uids.get(dept, [])[:200]
        tok_counts = {}
        total = 0
        for uid in d_uids:
            for tok in holdout_ptt.get(uid, []):
                tok_counts[tok] = tok_counts.get(tok, 0) + 1
                total += 1
        holdout_dept_tok_counts[dept] = (tok_counts, total)

    print(f"\n  Holdout graph: {len(holdout_ptt)} proteins, "
          f"{sum(len(t) for t in holdout_ptt.values())} edges")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # =====================================================================
    # TEST A: Knockout disruption profiles for chr22 genes
    # =====================================================================
    print(f"\n{'='*72}")
    print(f"  TEST A: Chr22 knockout disruption profiles on holdout kernel")
    print(f"{'='*72}")

    chr22_profiles_holdout = {}
    chr22_profiles_full = {}

    for gene in sorted(chr22_genes_with_dept):
        gene_uids_all = gene_to_uids.get(gene, [])
        if not gene_uids_all:
            continue

        gene_tokens = set()
        for uid in gene_uids_all:
            gene_tokens.update(ptt.get(uid, []))
        if not gene_tokens:
            continue

        profile_holdout = np.zeros(N_DEPTS)
        for di, dept in enumerate(VALID_DEPARTMENTS):
            tok_counts, total = holdout_dept_tok_counts[dept]
            lost = sum(tok_counts.get(t, 0) for t in gene_tokens if t in tok_counts)
            profile_holdout[di] = lost / max(total, 1)
        chr22_profiles_holdout[gene] = profile_holdout

        profile_full = np.zeros(N_DEPTS)
        full_dept_uids = defaultdict(list)
        for uid, g in gene_cache.items():
            if g and g in gene_depts:
                d = gene_depts[g]
                if d in D2I:
                    full_dept_uids[d].append(uid)

        for di, dept in enumerate(VALID_DEPARTMENTS):
            d_uids = full_dept_uids.get(dept, [])[:200]
            tok_counts_full = {}
            total_full = 0
            for uid in d_uids:
                for tok in ptt.get(uid, []):
                    tok_counts_full[tok] = tok_counts_full.get(tok, 0) + 1
                    total_full += 1
            lost = sum(tok_counts_full.get(t, 0) for t in gene_tokens if t in tok_counts_full)
            profile_full[di] = lost / max(total_full, 1)
        chr22_profiles_full[gene] = profile_full

    print(f"  Computed profiles for {len(chr22_profiles_holdout)} chr22 genes")

    top1_holdout = 0
    top3_holdout = 0
    top1_full = 0
    top3_full = 0
    n_tested = 0
    per_gene_results = []

    for gene in sorted(chr22_profiles_holdout.keys()):
        true_dept = gene_depts.get(gene)
        if not true_dept or true_dept not in D2I:
            continue

        prof_h = chr22_profiles_holdout[gene]
        prof_f = chr22_profiles_full[gene]
        if np.linalg.norm(prof_h) < 1e-12:
            continue

        n_tested += 1
        sorted_h = sorted(range(N_DEPTS), key=lambda i: prof_h[i], reverse=True)
        sorted_f = sorted(range(N_DEPTS), key=lambda i: prof_f[i], reverse=True)

        top1_h = VALID_DEPARTMENTS[sorted_h[0]]
        top3_h = [VALID_DEPARTMENTS[i] for i in sorted_h[:3]]
        top1_f = VALID_DEPARTMENTS[sorted_f[0]]
        top3_f = [VALID_DEPARTMENTS[i] for i in sorted_f[:3]]

        if top1_h == true_dept:
            top1_holdout += 1
        if true_dept in top3_h:
            top3_holdout += 1
        if top1_f == true_dept:
            top1_full += 1
        if true_dept in top3_f:
            top3_full += 1

        cos_h_f = cosine_sim(prof_h, prof_f)

        per_gene_results.append({
            "gene": gene,
            "department": true_dept,
            "holdout_top1": top1_h,
            "holdout_top3": top3_h,
            "full_top1": top1_f,
            "full_top3": top3_f,
            "holdout_full_cosine": round(cos_h_f, 4),
            "holdout_top1_correct": top1_h == true_dept,
            "full_top1_correct": top1_f == true_dept,
        })

    print(f"\n  Results ({n_tested} chr22 genes tested):")
    print(f"  {'Metric':<30s} {'Holdout':>10s} {'Full kernel':>12s} {'Chance':>8s}")
    print(f"  {'Top-1 concordance':<30s} "
          f"{top1_holdout/n_tested:>9.1%} {top1_full/n_tested:>11.1%} {1/22:>7.1%}")
    print(f"  {'Top-3 concordance':<30s} "
          f"{top3_holdout/n_tested:>9.1%} {top3_full/n_tested:>11.1%} {3/22:>7.1%}")

    profile_cosines = [r["holdout_full_cosine"] for r in per_gene_results]
    print(f"\n  Holdout vs full profile cosine: {np.mean(profile_cosines):.4f} ± {np.std(profile_cosines):.4f}")

    test_a = {
        "n_tested": n_tested,
        "holdout_top1": round(top1_holdout / n_tested, 4),
        "holdout_top3": round(top3_holdout / n_tested, 4),
        "full_top1": round(top1_full / n_tested, 4),
        "full_top3": round(top3_full / n_tested, 4),
        "chance_top1": round(1 / 22, 4),
        "chance_top3": round(3 / 22, 4),
        "profile_cosine_mean": round(float(np.mean(profile_cosines)), 4),
        "profile_cosine_std": round(float(np.std(profile_cosines)), 4),
    }

    # =====================================================================
    # TEST B: Collinearity on holdout kernel for chr22 genes
    # =====================================================================
    print(f"\n{'='*72}")
    print(f"  TEST B: Collinearity of chr22 genes on holdout kernel")
    print(f"{'='*72}")

    protein_dept_seqs = {}
    for uid, tokens in ptt.items():
        depts = []
        for tok in tokens:
            d = vocab_dept.get(tok.upper())
            if d and d in D2I:
                depts.append(d)
        if depts:
            compressed = []
            for d in depts:
                if not compressed or compressed[-1] != d:
                    compressed.append(d)
            protein_dept_seqs[uid] = "|".join(compressed)

    with open(PRIMITIVES_PATH) as f:
        raw_prims = list(csv.DictReader(f))

    prim_to_chr22_genes = defaultdict(list)
    for p in raw_prims:
        ds = [d for d in p["function_sequence"].split("|") if d in D2I]
        if not ds:
            continue
        search = "|".join(ds)
        carriers = [uid for uid, seq in protein_dept_seqs.items() if search in seq]
        if len(carriers) < 20:
            continue
        for uid in carriers:
            g = gene_cache.get(uid)
            if g and g in chr22_genes_with_dept and g in chr22_profiles_holdout:
                prim_to_chr22_genes[search].append(g)

    for k in prim_to_chr22_genes:
        prim_to_chr22_genes[k] = list(set(prim_to_chr22_genes[k]))

    testable_chr22 = {p: genes for p, genes in prim_to_chr22_genes.items()
                      if len(genes) >= 3}

    print(f"  Primitives with ≥3 chr22 carrier genes: {len(testable_chr22)}")

    rng = np.random.RandomState(42)
    all_chr22_genes_list = list(chr22_profiles_holdout.keys())

    within_cos = []
    across_cos = []
    positive_count = 0
    total_tested = 0

    for prim, genes in testable_chr22.items():
        vecs = [chr22_profiles_holdout[g] for g in genes
                if np.linalg.norm(chr22_profiles_holdout[g]) > 1e-10]
        if len(vecs) < 3:
            continue

        prim_within = []
        for i in range(len(vecs)):
            for j in range(i + 1, min(i + 10, len(vecs))):
                c = cosine_sim(vecs[i], vecs[j])
                within_cos.append(c)
                prim_within.append(c)

        rand = rng.choice(all_chr22_genes_list,
                          size=min(len(genes), 30), replace=False)
        prim_across = []
        for i in range(min(len(vecs), 10)):
            for j in range(min(10, len(rand))):
                g2 = rand[j]
                if g2 in chr22_profiles_holdout and np.linalg.norm(chr22_profiles_holdout[g2]) > 1e-10:
                    c = cosine_sim(vecs[i], chr22_profiles_holdout[g2])
                    across_cos.append(c)
                    prim_across.append(c)

        if prim_within and prim_across:
            lift = np.mean(prim_within) - np.mean(prim_across)
            if lift > 0:
                positive_count += 1
            total_tested += 1

    if within_cos and across_cos:
        wc = np.array(within_cos)
        ac = np.array(across_cos)
        pooled = np.sqrt((wc.var() + ac.var()) / 2)
        d_holdout = (wc.mean() - ac.mean()) / pooled if pooled > 0 else 0

        print(f"\n  Chr22 collinearity on holdout kernel:")
        print(f"  Within-primitive cosine:  {wc.mean():.4f} ± {wc.std():.4f}")
        print(f"  Across-primitive cosine:  {ac.mean():.4f} ± {ac.std():.4f}")
        print(f"  Cohen's d:               {d_holdout:+.4f}")
        print(f"  Positive lift:           {positive_count}/{total_tested}")
    else:
        d_holdout = 0.0
        print(f"  Insufficient data for collinearity test")

    test_b = {
        "n_primitives_tested": total_tested,
        "within_cos": round(float(np.mean(within_cos)), 4) if within_cos else None,
        "across_cos": round(float(np.mean(across_cos)), 4) if across_cos else None,
        "d": round(float(d_holdout), 4),
        "positive_count": positive_count,
        "positive_frac": round(positive_count / max(total_tested, 1), 4),
    }

    # =====================================================================
    # TEST C: Primitive consistency (chr22 vs non-chr22 carriers)
    # =====================================================================
    print(f"\n{'='*72}")
    print(f"  TEST C: Primitive consistency (chr22 vs non-chr22 carriers)")
    print(f"{'='*72}")

    with open(FULL_PROFILES_PATH) as f:
        full_profiles_data = json.load(f)
    all_profiles = {}
    for gene, prof in full_profiles_data["profiles"].items():
        all_profiles[gene] = np.array([prof.get(d, 0) for d in VALID_DEPARTMENTS])

    prim_to_all_genes = defaultdict(lambda: {"chr22": [], "other": []})
    for p in raw_prims:
        ds = [d for d in p["function_sequence"].split("|") if d in D2I]
        if not ds:
            continue
        search = "|".join(ds)
        carriers = [uid for uid, seq in protein_dept_seqs.items() if search in seq]
        if len(carriers) < 20:
            continue
        for uid in carriers:
            g = gene_cache.get(uid)
            if g and g in all_profiles:
                if g in chr22_genes:
                    prim_to_all_genes[search]["chr22"].append(g)
                else:
                    prim_to_all_genes[search]["other"].append(g)

    for k in prim_to_all_genes:
        prim_to_all_genes[k]["chr22"] = list(set(prim_to_all_genes[k]["chr22"]))
        prim_to_all_genes[k]["other"] = list(set(prim_to_all_genes[k]["other"]))

    consistency_results = []
    for prim, groups in prim_to_all_genes.items():
        chr22_g = groups["chr22"]
        other_g = groups["other"]
        if len(chr22_g) < 2 or len(other_g) < 5:
            continue

        chr22_vecs = [all_profiles[g] for g in chr22_g
                      if np.linalg.norm(all_profiles[g]) > 1e-10]
        other_vecs = [all_profiles[g] for g in other_g[:50]
                      if np.linalg.norm(all_profiles[g]) > 1e-10]

        if len(chr22_vecs) < 2 or len(other_vecs) < 3:
            continue

        chr22_mean = np.mean(chr22_vecs, axis=0)
        other_mean = np.mean(other_vecs, axis=0)

        cos = cosine_sim(chr22_mean, other_mean)
        consistency_results.append({
            "primitive": prim[:40],
            "n_chr22": len(chr22_g),
            "n_other": len(other_g),
            "cosine": round(cos, 4),
        })

    if consistency_results:
        cos_vals = [r["cosine"] for r in consistency_results]
        print(f"\n  Primitives with chr22 + non-chr22 carriers: {len(consistency_results)}")
        print(f"  Chr22 vs non-chr22 mean profile cosine:")
        print(f"    Mean:   {np.mean(cos_vals):.4f}")
        print(f"    Median: {np.median(cos_vals):.4f}")
        print(f"    Range:  [{min(cos_vals):.4f}, {max(cos_vals):.4f}]")
        print(f"    >0.9:   {sum(1 for c in cos_vals if c > 0.9)}/{len(cos_vals)}")

        rand_cosines = []
        rng2 = np.random.RandomState(42)
        for _ in range(len(consistency_results)):
            r_chr22 = rng2.choice(list(chr22_profiles_holdout.keys()),
                                   size=3, replace=False)
            r_other = rng2.choice([g for g in all_profiles if g not in chr22_genes],
                                  size=10, replace=False)
            rv1 = np.mean([all_profiles[g] for g in r_chr22], axis=0)
            rv2 = np.mean([all_profiles[g] for g in r_other], axis=0)
            rand_cosines.append(cosine_sim(rv1, rv2))

        print(f"\n  Random baseline cosine: {np.mean(rand_cosines):.4f} ± {np.std(rand_cosines):.4f}")
        print(f"  Primitive advantage: {np.mean(cos_vals) - np.mean(rand_cosines):+.4f}")
    else:
        cos_vals = []
        print(f"  No primitives with sufficient chr22 + non-chr22 carriers")

    test_c = {
        "n_primitives": len(consistency_results),
        "cosine_mean": round(float(np.mean(cos_vals)), 4) if cos_vals else None,
        "cosine_median": round(float(np.median(cos_vals)), 4) if cos_vals else None,
        "cosine_gt_0.9": sum(1 for c in cos_vals if c > 0.9) if cos_vals else 0,
    }

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n{'='*72}")
    print(f"  CHROMOSOME 22 HOLDOUT — SUMMARY")
    print(f"{'='*72}")
    print(f"\n  Test A (Disruption profiles):")
    print(f"    Top-1: holdout={test_a['holdout_top1']:.1%} vs full={test_a['full_top1']:.1%} "
          f"(chance={test_a['chance_top1']:.1%})")
    print(f"    Top-3: holdout={test_a['holdout_top3']:.1%} vs full={test_a['full_top3']:.1%} "
          f"(chance={test_a['chance_top3']:.1%})")
    print(f"    Profile cosine (holdout vs full): {test_a['profile_cosine_mean']:.4f}")

    print(f"\n  Test B (Collinearity):")
    print(f"    d={test_b['d']:+.4f} ({test_b['positive_count']}/{test_b['n_primitives_tested']} positive)")

    print(f"\n  Test C (Primitive consistency):")
    if test_c["cosine_mean"]:
        print(f"    Chr22 vs non-chr22 cosine: {test_c['cosine_mean']:.4f}")
    else:
        print(f"    Insufficient data")

    gen = "GENERALIZES" if (test_a['holdout_top1'] > 2 * test_a['chance_top1']
                            and test_a['profile_cosine_mean'] > 0.8) else "PARTIAL"
    print(f"\n  VERDICT: {gen}")
    print(f"  ({time.time()-t0:.0f}s total)")

    output = {
        "chr22_genes_tested": len(chr22_genes_with_dept),
        "chr22_uids_excluded": len(chr22_uids),
        "test_a_disruption_profiles": test_a,
        "test_b_collinearity": test_b,
        "test_c_consistency": test_c,
        "verdict": gen,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
