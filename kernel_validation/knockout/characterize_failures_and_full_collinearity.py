#!/usr/bin/env python3
"""
Two pre-submission analyses:
  Item 1: Characterize the 14.4% failure genes from the token-assignment shuffle
  Item 3: Compute full-proteome disruption profiles (19,375 genes) and retest collinearity

Usage:
    python3 -u validation/knockout/characterize_failures_and_full_collinearity.py
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
KO_RESULTS_PATH = "validation/knockout/knockout_full_results.json"
SHUFFLE_RESULTS_PATH = "validation/knockout/dept_profile_shuffle_results.json"
FULL_PROFILES_PATH = "validation/knockout/disruption_profiles_full.json"
OUTPUT_PATH = "validation/knockout/presubmission_analyses.json"

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


def item1_characterize_failures():
    """Characterize the 72 genes where shuffled tokens beat real tokens."""
    print("=" * 72)
    print("  ITEM 1: CHARACTERIZE THE 14.4% FAILURE GENES")
    print("=" * 72)

    with open(SHUFFLE_RESULTS_PATH) as f:
        shuffle_data = json.load(f)

    per_gene = shuffle_data["per_gene_results"]
    failures = [g for g in per_gene if g["cohens_d"] < 0]
    successes = [g for g in per_gene if g["cohens_d"] >= 0]

    print(f"\n  Total genes: {len(per_gene)}")
    print(f"  Successes (d >= 0): {len(successes)} ({len(successes)/len(per_gene):.1%})")
    print(f"  Failures (d < 0):   {len(failures)} ({len(failures)/len(per_gene):.1%})")

    fail_tokens = np.array([g["n_tokens"] for g in failures])
    succ_tokens = np.array([g["n_tokens"] for g in successes])

    print(f"\n  === TOKEN COUNT ===")
    print(f"  Failures:  median={np.median(fail_tokens):.0f}, "
          f"mean={fail_tokens.mean():.1f}, range=[{fail_tokens.min()}, {fail_tokens.max()}]")
    print(f"  Successes: median={np.median(succ_tokens):.0f}, "
          f"mean={succ_tokens.mean():.1f}, range=[{succ_tokens.min()}, {succ_tokens.max()}]")

    t_stat, t_p = stats.mannwhitneyu(fail_tokens, succ_tokens, alternative="less")
    print(f"  Mann-Whitney (failures have fewer tokens?): U={t_stat:.0f}, p={t_p:.4f}")

    token_bins = {"1-5": 0, "6-20": 0, "21-50": 0, "51+": 0}
    token_bins_total = {"1-5": 0, "6-20": 0, "21-50": 0, "51+": 0}
    for g in per_gene:
        nt = g["n_tokens"]
        if nt <= 5:
            b = "1-5"
        elif nt <= 20:
            b = "6-20"
        elif nt <= 50:
            b = "21-50"
        else:
            b = "51+"
        token_bins_total[b] += 1
        if g["cohens_d"] < 0:
            token_bins[b] += 1

    print(f"\n  Failure rate by token count:")
    for b in ["1-5", "6-20", "21-50", "51+"]:
        rate = token_bins[b] / max(token_bins_total[b], 1)
        print(f"    {b:>5s} tokens: {token_bins[b]:3d}/{token_bins_total[b]:3d} = {rate:.1%}")

    fail_depts = Counter(g["department"] for g in failures)
    all_depts = Counter(g["department"] for g in per_gene)

    print(f"\n  === DEPARTMENT DISTRIBUTION ===")
    print(f"  {'Department':<20s} {'Fail':>4s} {'Total':>5s} {'Rate':>6s}")
    dept_rates = {}
    for dept in sorted(all_depts.keys()):
        n_fail = fail_depts.get(dept, 0)
        n_total = all_depts[dept]
        rate = n_fail / n_total
        dept_rates[dept] = rate
        if n_total >= 5:
            flag = " <<<" if rate > 0.25 else ""
            print(f"  {dept:<20s} {n_fail:4d} {n_total:5d} {rate:5.1%}{flag}")

    top1_fail = [g for g in failures if g["real_top1_correct"]]
    top1_succ = [g for g in successes if g["real_top1_correct"]]
    print(f"\n  === TOP-1 DEPARTMENT ACCURACY ===")
    print(f"  Failures with correct top-1:  {len(top1_fail)}/{len(failures)} "
          f"({len(top1_fail)/len(failures):.1%})")
    print(f"  Successes with correct top-1: {len(top1_succ)}/{len(successes)} "
          f"({len(top1_succ)/len(successes):.1%})")

    fail_real_cos = np.array([g["real_cosine_sim"] for g in failures])
    succ_real_cos = np.array([g["real_cosine_sim"] for g in successes])
    print(f"\n  === COSINE SIMILARITY (real profile vs ground truth) ===")
    print(f"  Failures:  {fail_real_cos.mean():.4f} ± {fail_real_cos.std():.4f}")
    print(f"  Successes: {succ_real_cos.mean():.4f} ± {succ_real_cos.std():.4f}")

    fail_null_cos = np.array([g["null_cosine_mean"] for g in failures])
    succ_null_cos = np.array([g["null_cosine_mean"] for g in successes])
    print(f"\n  === NULL COSINE (shuffled profile vs ground truth) ===")
    print(f"  Failures:  {fail_null_cos.mean():.4f} ± {fail_null_cos.std():.4f}")
    print(f"  Successes: {succ_null_cos.mean():.4f} ± {succ_null_cos.std():.4f}")

    fail_null_std = np.array([g["null_cosine_std"] for g in failures])
    succ_null_std = np.array([g["null_cosine_std"] for g in successes])
    print(f"\n  === NULL STD (variance of shuffled cosines) ===")
    print(f"  Failures:  {fail_null_std.mean():.4f} ± {fail_null_std.std():.4f}")
    print(f"  Successes: {succ_null_std.mean():.4f} ± {succ_null_std.std():.4f}")

    fail_actual_top = Counter(g["real_top_dept"] for g in failures)
    print(f"\n  === WHAT DEPARTMENT DO FAILURES ACTUALLY DISRUPT MOST? ===")
    print(f"  {'Actual top dept':<20s} {'Count':>5s}")
    for dept, count in fail_actual_top.most_common(10):
        print(f"  {dept:<20s} {count:5d}")

    mismatch_pairs = Counter()
    for g in failures:
        pair = f"{g['department']} → {g['real_top_dept']}"
        mismatch_pairs[pair] += 1
    print(f"\n  === MOST COMMON MISMATCHES (assigned dept → actual top disruption) ===")
    for pair, count in mismatch_pairs.most_common(10):
        print(f"    {pair}: {count}")

    patterns = []
    if fail_tokens.mean() < succ_tokens.mean() * 0.8 and t_p < 0.05:
        patterns.append("SHORT_PROTEINS")
    high_fail_depts = [d for d, r in dept_rates.items() if r > 0.3 and all_depts[d] >= 5]
    if high_fail_depts:
        patterns.append(f"DEPT_CONCENTRATED:{','.join(high_fail_depts)}")
    if fail_null_cos.mean() > succ_null_cos.mean() + 0.02:
        patterns.append("HIGH_NULL_BASELINE")

    if not patterns:
        verdict = "NO_SYSTEMATIC_PATTERN"
        print(f"\n  VERDICT: No strong systematic pattern — failures appear random")
    else:
        verdict = "|".join(patterns)
        print(f"\n  VERDICT: Systematic patterns found: {verdict}")

    return {
        "n_total": len(per_gene),
        "n_failures": len(failures),
        "failure_rate": round(len(failures) / len(per_gene), 4),
        "fail_token_median": float(np.median(fail_tokens)),
        "succ_token_median": float(np.median(succ_tokens)),
        "token_count_p": float(t_p),
        "failure_rate_by_token_bin": {
            b: round(token_bins[b] / max(token_bins_total[b], 1), 4)
            for b in ["1-5", "6-20", "21-50", "51+"]
        },
        "dept_failure_rates": {d: round(r, 4) for d, r in dept_rates.items()},
        "fail_real_cos_mean": round(float(fail_real_cos.mean()), 4),
        "succ_real_cos_mean": round(float(succ_real_cos.mean()), 4),
        "fail_null_cos_mean": round(float(fail_null_cos.mean()), 4),
        "succ_null_cos_mean": round(float(succ_null_cos.mean()), 4),
        "fail_top1_correct_rate": round(len(top1_fail) / len(failures), 4),
        "succ_top1_correct_rate": round(len(top1_succ) / len(successes), 4),
        "top_mismatch_pairs": dict(mismatch_pairs.most_common(10)),
        "verdict": verdict,
        "patterns": patterns,
    }


def item3_full_proteome_collinearity():
    """Compute profiles for ALL 19,375 genes and retest collinearity."""
    print("\n" + "=" * 72)
    print("  ITEM 3: FULL-PROTEOME COLLINEARITY TEST")
    print("  Computing 22D profiles for all 19,375 knockout genes...")
    print("=" * 72)
    t0 = time.time()

    with open(STATE_PATH, "rb") as f:
        state = pickle.load(f)
    ptt = state["ptt"]
    gene_cache = state["gene_cache"]

    gene_depts = {}
    with open(GENE_DEPTS_PATH) as f:
        for row in csv.DictReader(f):
            gene_depts[row["gene"]] = row["department"]

    with open(KO_RESULTS_PATH) as f:
        ko_data = json.load(f)
    ko_genes = [e["gene"] for e in ko_data["results"]]

    gene_to_uids = defaultdict(list)
    for uid, g in gene_cache.items():
        if g:
            gene_to_uids[g].append(uid)

    dept_uids = defaultdict(list)
    for uid, g in gene_cache.items():
        if g and g in gene_depts:
            d = gene_depts[g]
            if d in D2I:
                dept_uids[d].append(uid)

    dept_token_sets = {}
    for dept, uids in dept_uids.items():
        all_tokens = set()
        for uid in uids[:500]:
            all_tokens.update(ptt.get(uid, []))
        dept_token_sets[dept] = all_tokens

    print(f"  Loaded data in {time.time()-t0:.1f}s")
    print(f"  Computing profiles for {len(ko_genes)} genes...")

    profiles = {}
    t1 = time.time()

    for gi, gene in enumerate(ko_genes):
        gene_uids = gene_to_uids.get(gene, [])
        if not gene_uids:
            continue

        gene_tokens = set()
        for uid in gene_uids:
            gene_tokens.update(ptt.get(uid, []))

        if not gene_tokens:
            continue

        profile = np.zeros(N_DEPTS)
        for di, dept in enumerate(VALID_DEPARTMENTS):
            d_uids = dept_uids.get(dept, [])[:300]
            if not d_uids:
                continue
            total = 0
            lost = 0
            for uid in d_uids:
                toks = set(ptt.get(uid, []))
                total += len(toks)
                lost += len(toks & gene_tokens)
            profile[di] = lost / max(total, 1)

        profiles[gene] = profile

        if (gi + 1) % 2000 == 0:
            elapsed = time.time() - t1
            rate = (gi + 1) / elapsed
            remaining = (len(ko_genes) - gi - 1) / rate
            print(f"    [{gi+1:6d}/{len(ko_genes)}] "
                  f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")
            sys.stdout.flush()

    print(f"  Computed {len(profiles)} profiles in {time.time()-t1:.1f}s")

    profiles_dict = {}
    for gene, prof in profiles.items():
        profiles_dict[gene] = {d: float(prof[i]) for i, d in enumerate(VALID_DEPARTMENTS)}

    full_profiles_output = {
        "n_genes": len(profiles),
        "departments": VALID_DEPARTMENTS,
        "profiles": profiles_dict,
    }
    with open(FULL_PROFILES_PATH, "w") as f:
        json.dump(full_profiles_output, f)
    print(f"  Saved full profiles: {FULL_PROFILES_PATH}")

    vocab_dept = {}
    with open(VOCAB_PATH) as f:
        for row in csv.DictReader(f):
            vocab_dept[row["word_hex"].replace("0x", "").upper()] = row["primary_function"]

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

    prim_to_genes = defaultdict(list)
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
            if g and g in profiles:
                prim_to_genes[search].append(g)

    for k in prim_to_genes:
        prim_to_genes[k] = list(set(prim_to_genes[k]))

    testable = {p: genes for p, genes in prim_to_genes.items() if len(genes) >= 5}
    print(f"\n  Testable primitives (full proteome): {len(testable)}")

    all_genes_list = list(profiles.keys())
    rng = np.random.RandomState(42)

    within_cos = []
    across_cos = []
    per_primitive = []

    for prim, genes in testable.items():
        vecs = [profiles[g] for g in genes if np.linalg.norm(profiles[g]) > 1e-10]
        if len(vecs) < 3:
            continue

        prim_within = []
        for i in range(len(vecs)):
            for j in range(i + 1, min(i + 10, len(vecs))):
                c = cosine_sim(vecs[i], vecs[j])
                within_cos.append(c)
                prim_within.append(c)

        rand = rng.choice(all_genes_list, size=min(len(genes), 50), replace=False)
        prim_across = []
        for i in range(min(len(vecs), 10)):
            for j in range(min(10, len(rand))):
                g2 = rand[j]
                if g2 in profiles and np.linalg.norm(profiles[g2]) > 1e-10:
                    c = cosine_sim(vecs[i], profiles[g2])
                    across_cos.append(c)
                    prim_across.append(c)

        if prim_within and prim_across:
            lift = np.mean(prim_within) - np.mean(prim_across)
            per_primitive.append({
                "primitive": prim[:40],
                "n_genes": len(genes),
                "within_cos": round(float(np.mean(prim_within)), 4),
                "across_cos": round(float(np.mean(prim_across)), 4),
                "lift": round(float(lift), 4),
                "positive": lift > 0,
            })

    wc = np.array(within_cos)
    ac = np.array(across_cos)
    pooled = np.sqrt((wc.var() + ac.var()) / 2)
    d = (wc.mean() - ac.mean()) / pooled if pooled > 0 else 0

    positive_count = sum(1 for p in per_primitive if p["positive"])
    total_tested = len(per_primitive)

    print(f"\n  === FULL-PROTEOME COLLINEARITY RESULTS ===")
    print(f"  Genes with profiles:         {len(profiles):,}")
    print(f"  Testable primitives:         {total_tested}")
    print(f"  Within-primitive cosine:     {wc.mean():.4f} ± {wc.std():.4f}")
    print(f"  Across-primitive cosine:     {ac.mean():.4f} ± {ac.std():.4f}")
    print(f"  Cohen's d:                   {d:+.4f}")
    print(f"  Positive lift:               {positive_count}/{total_tested} "
          f"({positive_count/total_tested:.1%})")

    with open("validation/knockout/disruption_profiles.json") as f:
        old_profiles_data = json.load(f)
    old_profiles = {}
    for gene, prof in old_profiles_data["profiles"].items():
        old_profiles[gene] = np.array([prof.get(d, 0) for d in VALID_DEPARTMENTS])

    old_testable = {p: [g for g in genes if g in old_profiles]
                    for p, genes in prim_to_genes.items()}
    old_testable = {p: genes for p, genes in old_testable.items() if len(genes) >= 5}

    old_within = []
    old_across = []
    for prim, genes in old_testable.items():
        vecs = [old_profiles[g] for g in genes if np.linalg.norm(old_profiles[g]) > 1e-10]
        if len(vecs) < 3:
            continue
        for i in range(len(vecs)):
            for j in range(i + 1, min(i + 10, len(vecs))):
                old_within.append(cosine_sim(vecs[i], vecs[j]))
        rand = rng.choice(list(old_profiles.keys()), size=min(len(genes), 50), replace=False)
        for i in range(min(len(vecs), 10)):
            for j in range(min(10, len(rand))):
                g2 = rand[j]
                if g2 in old_profiles and np.linalg.norm(old_profiles[g2]) > 1e-10:
                    old_across.append(cosine_sim(vecs[i], old_profiles[g2]))

    owc = np.array(old_within)
    oac = np.array(old_across)
    old_pooled = np.sqrt((owc.var() + oac.var()) / 2)
    old_d = (owc.mean() - oac.mean()) / old_pooled if old_pooled > 0 else 0

    print(f"\n  === COMPARISON: 2,060 vs {len(profiles):,} GENES ===")
    print(f"  Old (2,060 genes):   d={old_d:+.4f} (within={owc.mean():.4f}, across={oac.mean():.4f})")
    print(f"  Full ({len(profiles):,} genes): d={d:+.4f} (within={wc.mean():.4f}, across={ac.mean():.4f})")

    if abs(d - old_d) < 0.3 and d > 0.5:
        verdict = "CONFIRMED_NO_SELECTION_BIAS"
        print(f"  VERDICT: Collinearity CONFIRMED on full proteome — no selection bias")
    elif d > 0.5:
        verdict = "CONFIRMED_WITH_CHANGE"
        print(f"  VERDICT: Collinearity confirmed but d changed (Δ={d-old_d:+.3f})")
    else:
        verdict = "WEAKENED"
        print(f"  VERDICT: Collinearity weakened on full proteome (d={d:+.3f})")

    print(f"\n  ({time.time()-t0:.0f}s total)")

    return {
        "n_profiles_computed": len(profiles),
        "n_testable_primitives": total_tested,
        "full_within_cos": round(float(wc.mean()), 4),
        "full_across_cos": round(float(ac.mean()), 4),
        "full_d": round(float(d), 4),
        "full_positive_lift_frac": round(positive_count / total_tested, 4),
        "full_positive_count": positive_count,
        "old_d": round(float(old_d), 4),
        "old_within_cos": round(float(owc.mean()), 4),
        "old_across_cos": round(float(oac.mean()), 4),
        "delta_d": round(float(d - old_d), 4),
        "verdict": verdict,
    }


def main():
    r1 = item1_characterize_failures()
    r3 = item3_full_proteome_collinearity()

    output = {
        "item1_failure_characterization": r1,
        "item3_full_proteome_collinearity": r3,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*72}")
    print(f"  PRE-SUBMISSION ANALYSES — SUMMARY")
    print(f"{'='*72}")
    print(f"\n  Item 1: Failure characterization")
    print(f"    Verdict: {r1['verdict']}")
    print(f"    Patterns: {r1['patterns']}")
    print(f"\n  Item 3: Full-proteome collinearity")
    print(f"    Verdict: {r3['verdict']}")
    print(f"    d (2,060 genes): {r3['old_d']:+.4f}")
    print(f"    d (full):        {r3['full_d']:+.4f}")
    print(f"{'='*72}")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
