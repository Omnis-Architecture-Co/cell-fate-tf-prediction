#!/usr/bin/env python3
"""Q4: Head-to-head vocabulary vs BLAST-proxy essentiality prediction.

For genes where vocabulary and BLAST-proxy department assignments DISAGREE,
which assignment better predicts DepMap gene essentiality (Chronos scores)?

Uses existing infrastructure from depmap_essentiality_test.py and
vocab_vs_blast_prediction.py.
"""

import csv
import json
import os
import statistics
import math
from collections import defaultdict, Counter

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..", "..")

VOCAB_CSV = os.path.join(ROOT, "server", "data", "human", "vocabulary.csv")
TOKENS_CSV = os.path.join(ROOT, "server", "data", "human",
                          "protein_tokens_v2_with_genes.csv")
DEPTS_CSV = os.path.join(ROOT, "server", "data", "human",
                         "gene_departments.csv")
DEPMAP_CACHE = "/tmp/depmap_gene_essentiality.csv"
OUT_JSON = os.path.join(BASE, "vocab_vs_blast_essentiality_results.json")


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
        return 0.0, 0
    grand_mean = statistics.mean(all_vals)
    ss_between = sum(len(vals) * (statistics.mean(vals) - grand_mean) ** 2
                     for vals in groups.values() if len(vals) > 0)
    ss_total = sum((v - grand_mean) ** 2 for v in all_vals)
    if ss_total == 0:
        return 0.0, len(all_vals)
    return ss_between / ss_total, len(all_vals)


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
    print("Q4: Vocabulary vs BLAST-proxy essentiality prediction")
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

    print("\n2. Computing assignments...")

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

    vocab_assign = {}
    blast_assign = {}
    for g in test_genes:
        va = vocab_predict(g)
        ba = blast_proxy_predict(g)
        if va and ba:
            vocab_assign[g] = va
            blast_assign[g] = ba

    testable = [g for g in test_genes if g in vocab_assign and g in blast_assign]
    print(f"  Genes with both assignments: {len(testable)}")

    agree = [g for g in testable if vocab_assign[g] == blast_assign[g]]
    disagree = [g for g in testable if vocab_assign[g] != blast_assign[g]]
    print(f"  Agree: {len(agree)} ({100*len(agree)/len(testable):.1f}%)")
    print(f"  Disagree: {len(disagree)} ({100*len(disagree)/len(testable):.1f}%)")

    print("\n3. Essentiality prediction on DISAGREEMENT genes...")

    vocab_groups_disagree = defaultdict(list)
    blast_groups_disagree = defaultdict(list)
    true_groups_disagree = defaultdict(list)
    for g in disagree:
        score = depmap[g]
        vocab_groups_disagree[vocab_assign[g]].append(score)
        blast_groups_disagree[blast_assign[g]].append(score)
        true_groups_disagree[departments[g]].append(score)

    vocab_eta2, vocab_n = compute_eta_squared(vocab_groups_disagree)
    blast_eta2, blast_n = compute_eta_squared(blast_groups_disagree)
    true_eta2, true_n = compute_eta_squared(true_groups_disagree)

    print(f"\n  Essentiality variance explained (eta-squared) on disagreement genes:")
    print(f"    Vocabulary departments:   eta2 = {vocab_eta2:.4f} ({vocab_eta2*100:.1f}%)")
    print(f"    BLAST-proxy departments:  eta2 = {blast_eta2:.4f} ({blast_eta2*100:.1f}%)")
    print(f"    True (ground-truth) dept: eta2 = {true_eta2:.4f} ({true_eta2*100:.1f}%)")
    print(f"    Vocabulary advantage:     {vocab_eta2 - blast_eta2:+.4f}")

    print("\n4. Same analysis on AGREEMENT genes (control)...")

    vocab_groups_agree = defaultdict(list)
    for g in agree:
        vocab_groups_agree[vocab_assign[g]].append(depmap[g])
    agree_eta2, agree_n = compute_eta_squared(vocab_groups_agree)
    print(f"    Agreement subset eta2 = {agree_eta2:.4f} ({agree_eta2*100:.1f}%)")

    print("\n5. Full sample comparison...")

    vocab_groups_all = defaultdict(list)
    blast_groups_all = defaultdict(list)
    true_groups_all = defaultdict(list)
    for g in testable:
        score = depmap[g]
        vocab_groups_all[vocab_assign[g]].append(score)
        blast_groups_all[blast_assign[g]].append(score)
        true_groups_all[departments[g]].append(score)

    vocab_eta2_all, _ = compute_eta_squared(vocab_groups_all)
    blast_eta2_all, _ = compute_eta_squared(blast_groups_all)
    true_eta2_all, _ = compute_eta_squared(true_groups_all)

    print(f"    Vocabulary:   eta2 = {vocab_eta2_all:.4f}")
    print(f"    BLAST-proxy:  eta2 = {blast_eta2_all:.4f}")
    print(f"    True dept:    eta2 = {true_eta2_all:.4f}")

    print("\n6. Top-5 essential departments comparison (disagreement genes)...")

    def top5_analysis(groups, label):
        dept_means = {d: statistics.mean(v) for d, v in groups.items() if len(v) >= 5}
        sorted_depts = sorted(dept_means, key=dept_means.get)
        top5 = sorted_depts[:5]
        top5_scores = [s for d in top5 for s in groups[d]]
        rest_scores = [s for d in sorted_depts[5:] for s in groups[d]]
        if top5_scores and rest_scores:
            d = cohens_d(top5_scores, rest_scores)
            print(f"    {label}:")
            print(f"      Top-5 depts: {', '.join(top5)}")
            print(f"      Top-5 mean Chronos: {statistics.mean(top5_scores):.4f} (n={len(top5_scores)})")
            print(f"      Rest mean Chronos:  {statistics.mean(rest_scores):.4f} (n={len(rest_scores)})")
            print(f"      Cohen's d: {d:.3f}")
            return d, top5
        return 0.0, top5

    vocab_d, vocab_top5 = top5_analysis(vocab_groups_disagree, "Vocabulary")
    blast_d, blast_top5 = top5_analysis(blast_groups_disagree, "BLAST-proxy")

    winner = "vocabulary" if vocab_eta2 > blast_eta2 else "BLAST-proxy"
    margin = abs(vocab_eta2 - blast_eta2)

    results = {
        "test": "Q4: Vocabulary vs BLAST-proxy essentiality prediction (head-to-head)",
        "total_testable_genes": len(testable),
        "agreement": {
            "n": len(agree),
            "pct": round(100 * len(agree) / len(testable), 1),
            "eta_squared": round(agree_eta2, 4),
        },
        "disagreement": {
            "n": len(disagree),
            "pct": round(100 * len(disagree) / len(testable), 1),
            "vocabulary_eta_squared": round(vocab_eta2, 4),
            "blast_proxy_eta_squared": round(blast_eta2, 4),
            "true_dept_eta_squared": round(true_eta2, 4),
            "vocabulary_advantage": round(vocab_eta2 - blast_eta2, 4),
            "winner": winner,
        },
        "full_sample": {
            "vocabulary_eta_squared": round(vocab_eta2_all, 4),
            "blast_proxy_eta_squared": round(blast_eta2_all, 4),
            "true_dept_eta_squared": round(true_eta2_all, 4),
        },
        "top5_essential_cohens_d": {
            "vocabulary": round(vocab_d, 3),
            "blast_proxy": round(blast_d, 3),
        },
        "conclusion": (
            f"On the {len(disagree)} genes where vocabulary and BLAST-proxy "
            f"disagree on department assignment ({100*len(disagree)/len(testable):.1f}% of testable genes), "
            f"{winner} departments explain more essentiality variance "
            f"(eta2 = {max(vocab_eta2, blast_eta2):.4f} vs {min(vocab_eta2, blast_eta2):.4f}, "
            f"advantage = {margin:.4f}). "
        ),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")
    print(f"\nCONCLUSION: {winner} wins on disagreement genes (eta2 advantage = {margin:.4f})")


if __name__ == "__main__":
    main()
