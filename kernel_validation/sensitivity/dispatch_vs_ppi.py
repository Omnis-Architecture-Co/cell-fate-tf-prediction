#!/usr/bin/env python3
"""Validate dispatch network against STRING protein-protein interactions.

Tests whether gene pairs connected by vocabulary dispatch edges have
higher PPI evidence than random gene pairs.
"""

import csv
import json
import os
import random
import time
import urllib.request
from collections import Counter, defaultdict

random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..", "..")
VOCAB_CSV = os.path.join(ROOT, "server", "data", "human", "vocabulary.csv")
DEPTS_CSV = os.path.join(ROOT, "server", "data", "human",
                         "gene_departments.csv")
OUT_JSON = os.path.join(BASE, "dispatch_vs_ppi_results.json")

STRING_API = "https://string-db.org/api/json/network"
SPECIES = 9606


def query_pair(g1, g2, retries=2):
    ids = f"{g1}%0d{g2}"
    url = (f"{STRING_API}?identifiers={ids}&species={SPECIES}"
           "&caller_identity=omnis_validation&required_score=0")
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                for r in data:
                    a = r.get("preferredName_A", "")
                    b = r.get("preferredName_B", "")
                    if (a.upper() == g1.upper() and b.upper() == g2.upper()) or \
                       (a.upper() == g2.upper() and b.upper() == g1.upper()):
                        return r.get("score", 0)
                    return r.get("score", 0)
                return 0
        except Exception:
            if attempt < retries:
                time.sleep(0.5)
    return None


def query_batch_pairs(pairs, label, delay=0.15):
    print(f"  Querying STRING for {label} ({len(pairs)} pairs)...")
    found = 0
    high_conf = 0
    scores = []
    errors = 0

    for i, (g1, g2) in enumerate(pairs):
        score = query_pair(g1, g2)
        if score is None:
            errors += 1
            continue
        if score > 0:
            found += 1
            scores.append(score)
            if score >= 0.4:
                high_conf += 1
        if i % 25 == 0 and i > 0:
            print(f"    {i}/{len(pairs)}: {found} found so far...")
        time.sleep(delay)

    n = len(pairs) - errors
    return {
        "n_pairs_tested": n,
        "n_with_ppi": found,
        "ppi_detection_rate": round(found / n, 4) if n > 0 else 0,
        "n_high_confidence": high_conf,
        "high_conf_rate": round(high_conf / n, 4) if n > 0 else 0,
        "mean_score": round(sum(scores) / len(scores), 4) if scores else 0,
        "errors": errors,
    }


def main():
    print("Loading data...")

    word_carriers = {}
    word_funcs = {}
    with open(VOCAB_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            hx = row["word_hex"]
            carriers = row["all_carrier_genes"]
            if carriers:
                genes = [g.strip() for g in carriers.split("; ")
                         if g.strip() and g.strip()[0].isupper()]
                if len(genes) >= 2:
                    word_carriers[hx] = genes
            func = row["primary_function"]
            if func and func != "Unclassified":
                word_funcs[hx] = func

    departments = {}
    with open(DEPTS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            departments[row["gene"]] = row["department"]

    print(f"  Words with human carriers: {len(word_carriers)}")
    print(f"  Genes with departments: {len(departments)}")

    print("\nBuilding gene pair sets...")
    dispatch_pairs = set()
    for hx, genes in word_carriers.items():
        if hx not in word_funcs:
            continue
        dept_genes = [g for g in genes if g in departments]
        for i, g1 in enumerate(dept_genes[:10]):
            for g2 in dept_genes[i+1:10]:
                if departments[g1] != departments[g2]:
                    dispatch_pairs.add(tuple(sorted([g1, g2])))

    same_dept_pairs = set()
    dept_to_genes = defaultdict(list)
    for g, d in departments.items():
        dept_to_genes[d].append(g)
    for dept, genes in dept_to_genes.items():
        if len(genes) >= 2:
            s = random.sample(genes, min(20, len(genes)))
            for i in range(len(s)):
                for j in range(i+1, len(s)):
                    same_dept_pairs.add(tuple(sorted([s[i], s[j]])))

    all_genes = list(departments.keys())
    random_pairs = set()
    while len(random_pairs) < 300:
        g1, g2 = random.sample(all_genes, 2)
        p = tuple(sorted([g1, g2]))
        if p not in dispatch_pairs and p not in same_dept_pairs:
            random_pairs.add(p)

    n_test = 100
    sample_dispatch = random.sample(list(dispatch_pairs), min(n_test, len(dispatch_pairs)))
    sample_same = random.sample(list(same_dept_pairs), min(n_test, len(same_dept_pairs)))
    sample_random = random.sample(list(random_pairs), min(n_test, len(random_pairs)))

    print(f"  Dispatch pairs: {len(dispatch_pairs)} total, testing {len(sample_dispatch)}")
    print(f"  Same-dept pairs: {len(same_dept_pairs)} total, testing {len(sample_same)}")
    print(f"  Random pairs: {len(random_pairs)} total, testing {len(sample_random)}")

    dispatch_ppi = query_batch_pairs(sample_dispatch, "dispatch-connected")
    same_ppi = query_batch_pairs(sample_same, "same-department")
    random_ppi = query_batch_pairs(sample_random, "random control")

    dr = dispatch_ppi["ppi_detection_rate"]
    sr = same_ppi["ppi_detection_rate"]
    rr = random_ppi["ppi_detection_rate"]

    enrichment = dr / rr if rr > 0 else float("inf")
    same_enrichment = sr / rr if rr > 0 else float("inf")

    results = {
        "test": "Dispatch Network vs STRING Protein-Protein Interactions",
        "method": (
            "Gene pairs co-carrying vocabulary tokens across different "
            "departments (dispatch-connected) were compared against "
            "same-department gene pairs and random gene pairs. Each pair "
            "was individually queried against STRING v12.0 (H. sapiens, "
            "9606) for PPI evidence. N=100 pairs per group."
        ),
        "dispatch_connected": dispatch_ppi,
        "same_department": same_ppi,
        "random_control": random_ppi,
        "enrichment_dispatch_vs_random": round(enrichment, 2),
        "enrichment_same_dept_vs_random": round(same_enrichment, 2),
        "conclusion": (
            f"Dispatch-connected gene pairs show {dr:.1%} PPI detection "
            f"rate vs {rr:.1%} for random pairs ({enrichment:.1f}x enrichment). "
            f"Same-department pairs show {sr:.1%} ({same_enrichment:.1f}x). "
        ),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Dispatch PPI rate:    {dr:.1%} ({dispatch_ppi['n_with_ppi']}/{dispatch_ppi['n_pairs_tested']})")
    print(f"Same-dept PPI rate:   {sr:.1%} ({same_ppi['n_with_ppi']}/{same_ppi['n_pairs_tested']})")
    print(f"Random PPI rate:      {rr:.1%} ({random_ppi['n_with_ppi']}/{random_ppi['n_pairs_tested']})")
    print(f"Enrichment (dispatch/random): {enrichment:.1f}x")
    print(f"Enrichment (same-dept/random): {same_enrichment:.1f}x")
    print(f"\nSaved to: {OUT_JSON}")


if __name__ == "__main__":
    main()
