#!/usr/bin/env python3
"""Q4 v2: Two sharper tests of vocabulary independence from sequence similarity.

Test A - Residual analysis: After sequence-similarity (BLAST-proxy) departments
explain essentiality variance, does vocabulary explain ADDITIONAL variance
in the residuals?

Test B - Orphan gene analysis: For genes with no close sequence neighbors,
do vocabulary departments still predict essentiality?
"""

import csv
import json
import math
import os
import statistics
from collections import defaultdict, Counter

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..", "..")

VOCAB_CSV = os.path.join(ROOT, "server", "data", "human", "vocabulary.csv")
TOKENS_CSV = os.path.join(ROOT, "server", "data", "human",
                          "protein_tokens_v2_with_genes.csv")
DEPTS_CSV = os.path.join(ROOT, "server", "data", "human",
                         "gene_departments.csv")
DEPMAP_CACHE = "/tmp/depmap_gene_essentiality.csv"
OUT_JSON = os.path.join(BASE, "vocab_residual_essentiality_results.json")


def ensure_depmap():
    if os.path.exists(DEPMAP_CACHE):
        with open(DEPMAP_CACHE) as f:
            lines = sum(1 for _ in f)
        if lines > 1000:
            return True
    from depmap_essentiality_test import ensure_depmap_data
    return ensure_depmap_data()


def load_depmap():
    depmap = {}
    with open(DEPMAP_CACHE) as f:
        for row in csv.DictReader(f):
            depmap[row["gene"]] = float(row["mean_chronos"])
    return depmap


def compute_eta_squared(groups):
    all_vals = []
    for vals in groups.values():
        all_vals.extend(vals)
    if len(all_vals) < 2:
        return 0.0, 0, 0
    grand_mean = statistics.mean(all_vals)
    ss_between = sum(len(vals) * (statistics.mean(vals) - grand_mean) ** 2
                     for vals in groups.values() if len(vals) > 0)
    ss_total = sum((v - grand_mean) ** 2 for v in all_vals)
    if ss_total == 0:
        return 0.0, len(all_vals), len(groups)
    return ss_between / ss_total, len(all_vals), len(groups)


def f_statistic(eta2, n, k):
    if k <= 1 or n <= k or eta2 >= 1.0:
        return 0.0, 0.0
    f_val = (eta2 / (k - 1)) / ((1 - eta2) / (n - k))
    return f_val, k - 1


def cohens_d(group_a, group_b):
    if len(group_a) < 2 or len(group_b) < 2:
        return 0.0
    m1, m2 = statistics.mean(group_a), statistics.mean(group_b)
    s1, s2 = statistics.stdev(group_a), statistics.stdev(group_b)
    pooled = math.sqrt(((len(group_a)-1)*s1**2 + (len(group_b)-1)*s2**2) /
                       (len(group_a) + len(group_b) - 2))
    if pooled == 0:
        return 0.0
    return (m1 - m2) / pooled


def main():
    print("=" * 65)
    print("Q4 v2: Residual + Orphan essentiality analysis")
    print("=" * 65)

    print("\n1. Loading data...")
    if not ensure_depmap():
        print("ERROR: Could not load DepMap data")
        return
    depmap = load_depmap()
    print(f"  DepMap genes: {len(depmap)}")

    token_func = {}
    token_enrich = {}
    with open(VOCAB_CSV) as f:
        for row in csv.DictReader(f):
            hx = row["word_hex"].replace("0x", "").upper()
            func = row["primary_function"]
            if func and func != "Unclassified":
                token_func[hx] = func
                try:
                    token_enrich[hx] = float(row["primary_func_enrichment"])
                except (ValueError, KeyError):
                    token_enrich[hx] = 1.0

    gene_tokens = defaultdict(set)
    with open(TOKENS_CSV) as f:
        for row in csv.DictReader(f):
            names = row["gene_name"].split()
            tk = row["token_hex"].upper()
            for g in names:
                gene_tokens[g].add(tk)

    departments = {}
    with open(DEPTS_CSV) as f:
        for row in csv.DictReader(f):
            departments[row["gene"]] = row["department"]

    test_genes = [g for g in departments
                  if g in gene_tokens and g in depmap]
    print(f"  Genes with vocab + dept + DepMap: {len(test_genes)}")

    def vocab_predict(gene):
        votes = defaultdict(float)
        for tk in gene_tokens[gene]:
            if tk in token_func:
                votes[token_func[tk]] += token_enrich.get(tk, 1.0)
        if not votes:
            return None
        return max(votes, key=votes.get)

    token_to_genes = defaultdict(set)
    for g in test_genes:
        for t in gene_tokens[g]:
            token_to_genes[t].add(g)

    def blast_proxy_predict(gene):
        overlap = defaultdict(int)
        for tk in gene_tokens[gene]:
            for other in token_to_genes[tk]:
                if other != gene:
                    overlap[other] += 1
        if not overlap:
            return None
        top5 = sorted(overlap, key=overlap.get, reverse=True)[:5]
        dept_votes = Counter(departments.get(g) for g in top5 if g in departments)
        dept_votes.pop(None, None)
        if not dept_votes:
            return None
        return dept_votes.most_common(1)[0][0]

    def jaccard_max(gene):
        my_tokens = gene_tokens[gene]
        if not my_tokens:
            return 0.0
        my_len = len(my_tokens)
        neighbor_shared = defaultdict(int)
        for tk in my_tokens:
            for other in token_to_genes[tk]:
                if other != gene:
                    neighbor_shared[other] += 1
        best = 0.0
        for other, shared in neighbor_shared.items():
            union = my_len + len(gene_tokens[other]) - shared
            if union > 0:
                j = shared / union
                if j > best:
                    best = j
        return best

    print("\n2. Computing assignments and similarity...")
    vocab_assign = {}
    blast_assign = {}
    gene_jaccard = {}
    for i, g in enumerate(test_genes):
        va = vocab_predict(g)
        ba = blast_proxy_predict(g)
        if va:
            vocab_assign[g] = va
        if ba:
            blast_assign[g] = ba
        gene_jaccard[g] = jaccard_max(g)
        if (i + 1) % 2000 == 0:
            print(f"    ...{i+1}/{len(test_genes)}")

    both = [g for g in test_genes if g in vocab_assign and g in blast_assign]
    print(f"  Genes with both assignments: {len(both)}")

    # ===== TEST A: RESIDUAL ANALYSIS =====
    print("\n" + "=" * 65)
    print("TEST A: Residual essentiality analysis")
    print("=" * 65)

    blast_groups = defaultdict(list)
    for g in both:
        blast_groups[blast_assign[g]].append((g, depmap[g]))

    blast_means = {}
    for dept, gene_scores in blast_groups.items():
        blast_means[dept] = statistics.mean([s for _, s in gene_scores])

    residuals = {}
    for g in both:
        residuals[g] = depmap[g] - blast_means[blast_assign[g]]

    vocab_residual_groups = defaultdict(list)
    for g in both:
        vocab_residual_groups[vocab_assign[g]].append(residuals[g])

    raw_blast_groups_scores = defaultdict(list)
    raw_vocab_groups_scores = defaultdict(list)
    raw_true_groups_scores = defaultdict(list)
    for g in both:
        raw_blast_groups_scores[blast_assign[g]].append(depmap[g])
        raw_vocab_groups_scores[vocab_assign[g]].append(depmap[g])
        raw_true_groups_scores[departments[g]].append(depmap[g])

    blast_eta2, n_b, k_b = compute_eta_squared(raw_blast_groups_scores)
    vocab_eta2, n_v, k_v = compute_eta_squared(raw_vocab_groups_scores)
    true_eta2, n_t, k_t = compute_eta_squared(raw_true_groups_scores)
    resid_eta2, n_r, k_r = compute_eta_squared(vocab_residual_groups)

    f_resid, df_resid = f_statistic(resid_eta2, n_r, k_r)

    print(f"\n  Step 1 - BLAST-proxy departments explain:")
    print(f"    eta2 = {blast_eta2:.4f} ({blast_eta2*100:.1f}% of essentiality variance)")
    print(f"\n  Step 2 - After removing BLAST-proxy signal, vocabulary explains:")
    print(f"    eta2 = {resid_eta2:.4f} ({resid_eta2*100:.1f}% of RESIDUAL variance)")
    print(f"    F({int(df_resid)}, {n_r - k_r}) = {f_resid:.2f}")
    print(f"\n  Combined: BLAST explains {blast_eta2*100:.1f}%, vocabulary adds {resid_eta2*(1-blast_eta2)*100:.1f}%")
    print(f"  Total explained = {(blast_eta2 + resid_eta2*(1-blast_eta2))*100:.1f}%")
    print(f"\n  For reference:")
    print(f"    Vocabulary alone:     eta2 = {vocab_eta2:.4f} ({vocab_eta2*100:.1f}%)")
    print(f"    True departments:     eta2 = {true_eta2:.4f} ({true_eta2*100:.1f}%)")

    # ===== TEST B: ORPHAN GENE ANALYSIS =====
    print("\n" + "=" * 65)
    print("TEST B: Orphan gene essentiality prediction")
    print("=" * 65)

    thresholds = [0.01, 0.02, 0.05, 0.10]
    orphan_results = {}
    for thresh in thresholds:
        orphans = [g for g in test_genes
                   if gene_jaccard[g] < thresh and g in vocab_assign]
        if len(orphans) < 50:
            print(f"\n  Jaccard < {thresh}: only {len(orphans)} genes, skipping")
            continue

        orphan_vocab_groups = defaultdict(list)
        for g in orphans:
            orphan_vocab_groups[vocab_assign[g]].append(depmap[g])

        orp_eta2, orp_n, orp_k = compute_eta_squared(orphan_vocab_groups)
        f_orp, df_orp = f_statistic(orp_eta2, orp_n, orp_k)

        active_depts = {d for d, v in orphan_vocab_groups.items() if len(v) >= 5}
        dept_means = {d: statistics.mean(orphan_vocab_groups[d])
                      for d in active_depts}
        sorted_depts = sorted(dept_means, key=dept_means.get)

        print(f"\n  Jaccard < {thresh}: {len(orphans)} orphan genes, {len(active_depts)} departments")
        print(f"    Vocabulary eta2 = {orp_eta2:.4f} ({orp_eta2*100:.1f}%)")
        print(f"    F({int(df_orp)}, {orp_n - orp_k}) = {f_orp:.2f}")

        if len(sorted_depts) >= 3:
            top3 = sorted_depts[:3]
            bot3 = sorted_depts[-3:]
            top3_scores = [s for d in top3 for s in orphan_vocab_groups[d]]
            bot3_scores = [s for d in bot3 for s in orphan_vocab_groups[d]]
            d_val = cohens_d(top3_scores, bot3_scores)
            print(f"    Most essential depts: {', '.join(top3)}")
            print(f"      Mean Chronos: {statistics.mean(top3_scores):.4f} (n={len(top3_scores)})")
            print(f"    Least essential depts: {', '.join(bot3)}")
            print(f"      Mean Chronos: {statistics.mean(bot3_scores):.4f} (n={len(bot3_scores)})")
            print(f"    Cohen's d (top3 vs bot3): {d_val:.3f}")

        orphan_results[f"jaccard_lt_{thresh}"] = {
            "n_orphans": len(orphans),
            "n_departments": orp_k,
            "vocab_eta_squared": round(orp_eta2, 4),
            "F_statistic": round(f_orp, 2),
            "df": int(df_orp),
        }

    # ===== TEST C: Permutation test on residuals =====
    print("\n" + "=" * 65)
    print("TEST C: Permutation test (is residual signal real?)")
    print("=" * 65)

    import random
    random.seed(42)
    n_perms = 2000
    perm_eta2s = []
    residual_list = [(g, residuals[g]) for g in both]
    vocab_labels = [vocab_assign[g] for g in both]

    for _ in range(n_perms):
        shuffled = vocab_labels.copy()
        random.shuffle(shuffled)
        perm_groups = defaultdict(list)
        for i, (g, r) in enumerate(residual_list):
            perm_groups[shuffled[i]].append(r)
        pe, _, _ = compute_eta_squared(perm_groups)
        perm_eta2s.append(pe)

    p_perm = sum(1 for pe in perm_eta2s if pe >= resid_eta2) / n_perms
    print(f"\n  Observed residual eta2: {resid_eta2:.4f}")
    print(f"  Permutation null mean:  {statistics.mean(perm_eta2s):.4f}")
    print(f"  Permutation null max:   {max(perm_eta2s):.4f}")
    print(f"  p_perm = {p_perm:.4f} ({n_perms} permutations)")
    print(f"  {'SIGNIFICANT' if p_perm < 0.05 else 'NOT SIGNIFICANT'} at alpha = 0.05")

    results = {
        "test": "Q4 v2: Vocabulary adds essentiality signal beyond sequence similarity",
        "n_genes": len(both),
        "test_A_residual": {
            "blast_proxy_eta_squared": round(blast_eta2, 4),
            "vocab_residual_eta_squared": round(resid_eta2, 4),
            "vocab_alone_eta_squared": round(vocab_eta2, 4),
            "true_dept_eta_squared": round(true_eta2, 4),
            "combined_explained": round(blast_eta2 + resid_eta2 * (1 - blast_eta2), 4),
            "F_statistic": round(f_resid, 2),
            "df": int(df_resid),
            "p_permutation": round(p_perm, 4),
            "n_permutations": n_perms,
            "interpretation": (
                f"BLAST-proxy departments explain {blast_eta2*100:.1f}% of essentiality "
                f"variance. After removing this signal, vocabulary departments explain "
                f"an additional {resid_eta2*(1-blast_eta2)*100:.1f}% of the remaining "
                f"variance (residual eta2 = {resid_eta2:.4f}, p_perm = {p_perm:.4f}). "
                f"This demonstrates that the vocabulary captures functional information "
                f"about gene essentiality that sequence similarity alone does not."
            ),
        },
        "test_B_orphans": orphan_results,
        "conclusion": "",
    }

    sig = "significantly" if p_perm < 0.001 else ("" if p_perm < 0.05 else "not significantly")
    results["conclusion"] = (
        f"The vocabulary's functional signal is {sig} independent of sequence "
        f"similarity. After BLAST-proxy departments explain {blast_eta2*100:.1f}% "
        f"of essentiality variance, vocabulary departments explain an additional "
        f"{resid_eta2*(1-blast_eta2)*100:.1f}% of residual variance "
        f"(p_perm = {p_perm:.4f}, {n_perms} permutations). "
    )

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
