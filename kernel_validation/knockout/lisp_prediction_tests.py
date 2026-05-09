#!/usr/bin/env python3
"""
Prediction Tests — can primitive composition predict protein function & disease?
================================================================================

Test 1: Department Prediction
  Given only a protein's primitive composition (which primitives appear in its
  function sequence), predict which functional department the protein belongs to.
  Train on 50%, test on held-out 50%.

Test 2: Disease Prediction from Primitives
  Given only a gene's primitive composition, predict which diseases it's associated
  with — WITHOUT running the knockout simulation. Compare to the full knockout
  simulation results (which achieved 27% top-1, MRR=0.425).

Usage:
    python3 -u validation/knockout/lisp_prediction_tests.py
"""

import csv
import json
import os
import pickle
import random
import sys
import time
from collections import defaultdict, Counter

import numpy as np
from scipy import stats

STATE_PATH = "/tmp/module8_full_state.pkl"
VOCAB_PATH = "server/data/human/vocabulary.csv"
PRIMITIVES_PATH = "beta_transfer/genome_primitives.csv"
GENE_DEPTS_PATH = "server/data/human/gene_departments.csv"
KNOCKOUT_RESULTS_PATH = "validation/knockout/knockout_full_results.json"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "lisp_prediction_results.json")

VALID_DEPARTMENTS = sorted([
    "Apoptosis", "Cell adhesion", "Cell cycle", "Chromatin", "Cytoskeleton",
    "DNA repair", "GTPase", "Immune", "Ion channel", "Kinase", "Methylation",
    "Nuc acid bind", "Phosphatase", "Protein folding", "Proteolysis",
    "RNA processing", "Signaling", "Structural", "Transcription",
    "Translation", "Transport", "Ubiquitin",
])
DEPT_TO_IDX = {d: i for i, d in enumerate(VALID_DEPARTMENTS)}
N_DEPTS = len(VALID_DEPARTMENTS)
SPLIT_SEED = 42


def load_state():
    print("[1] Loading dispatch graph state...")
    with open(STATE_PATH, "rb") as f:
        state = pickle.load(f)
    return state


def build_token_dept_map():
    vocab_dept = {}
    with open(VOCAB_PATH) as f:
        for row in csv.DictReader(f):
            stripped = row["word_hex"].replace("0x", "").upper()
            vocab_dept[stripped] = row["primary_function"]
    return vocab_dept


def build_protein_dept_sequences(state, vocab_dept):
    print("[2] Building protein department sequences...")
    ptt = state["ptt"]
    protein_dept_seqs = {}
    for uid, tokens in ptt.items():
        depts = []
        for tok in tokens:
            dept = vocab_dept.get(tok.upper())
            if dept and dept in DEPT_TO_IDX:
                depts.append(dept)
        if depts:
            compressed = []
            for d in depts:
                if not compressed or compressed[-1] != d:
                    compressed.append(d)
            protein_dept_seqs[uid] = "|".join(compressed)
    print(f"  {len(protein_dept_seqs)} proteins with dept sequences")
    return protein_dept_seqs


def load_primitives():
    with open(PRIMITIVES_PATH) as f:
        primitives = list(csv.DictReader(f))
    valid_prims = []
    for p in primitives:
        depts = [d for d in p["function_sequence"].split("|") if d in DEPT_TO_IDX]
        if depts:
            valid_prims.append({
                "raw": p["function_sequence"],
                "search": "|".join(depts),
                "depts": depts,
                "unique_depts": list(dict.fromkeys(depts)),
            })
    print(f"[3] Loaded {len(valid_prims)} valid primitives")
    return valid_prims


def encode_protein_primitives(uid, seq, primitives):
    features = np.zeros(len(primitives))
    for i, prim in enumerate(primitives):
        if prim["search"] in seq:
            features[i] = 1.0
    return features


def load_gene_departments():
    gene_to_dept = {}
    with open(GENE_DEPTS_PATH) as f:
        for row in csv.DictReader(f):
            gene_to_dept[row["gene"]] = row["department"]
    return gene_to_dept


def test1_department_prediction(state, protein_dept_seqs, primitives):
    print("\n" + "=" * 72)
    print("  TEST 1: Department Prediction from Primitive Composition")
    print("=" * 72)
    t0 = time.time()

    gene_cache = state["gene_cache"]
    gene_to_dept = load_gene_departments()

    uid_dept = {}
    for uid in protein_dept_seqs:
        gene = gene_cache.get(uid)
        if gene and gene in gene_to_dept:
            dept = gene_to_dept[gene]
            if dept in DEPT_TO_IDX:
                uid_dept[uid] = dept

    print(f"  Proteins with known department: {len(uid_dept)}")

    all_uids = sorted(uid_dept.keys())
    random.seed(SPLIT_SEED)
    random.shuffle(all_uids)
    mid = len(all_uids) // 2
    train_uids = all_uids[:mid]
    test_uids = all_uids[mid:]

    print(f"  Train: {len(train_uids)}, Test: {len(test_uids)}")

    dept_prim_profiles = defaultdict(list)
    for uid in train_uids:
        seq = protein_dept_seqs[uid]
        dept = uid_dept[uid]
        features = encode_protein_primitives(uid, seq, primitives)
        dept_prim_profiles[dept].append(features)

    dept_centroids = {}
    for dept, profiles in dept_prim_profiles.items():
        dept_centroids[dept] = np.mean(profiles, axis=0)

    print(f"  Departments with training data: {len(dept_centroids)}")

    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    total = 0
    per_dept_correct = defaultdict(int)
    per_dept_total = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))

    most_common_dept = Counter(uid_dept[u] for u in train_uids).most_common(1)[0][0]
    majority_correct = 0

    for uid in test_uids:
        seq = protein_dept_seqs[uid]
        actual_dept = uid_dept[uid]
        features = encode_protein_primitives(uid, seq, primitives)

        if features.sum() == 0:
            continue

        sims = {}
        for dept, centroid in dept_centroids.items():
            dot = np.dot(features, centroid)
            n1, n2 = np.linalg.norm(features), np.linalg.norm(centroid)
            if n1 > 0 and n2 > 0:
                sims[dept] = dot / (n1 * n2)
            else:
                sims[dept] = 0

        ranked = sorted(sims.keys(), key=lambda d: -sims[d])

        total += 1
        per_dept_total[actual_dept] += 1
        confusion[actual_dept][ranked[0]] += 1

        if ranked[0] == actual_dept:
            correct_top1 += 1
            per_dept_correct[actual_dept] += 1
        if actual_dept in ranked[:3]:
            correct_top3 += 1
        if actual_dept in ranked[:5]:
            correct_top5 += 1
        if most_common_dept == actual_dept:
            majority_correct += 1

    top1_acc = correct_top1 / total if total > 0 else 0
    top3_acc = correct_top3 / total if total > 0 else 0
    top5_acc = correct_top5 / total if total > 0 else 0
    majority_acc = majority_correct / total if total > 0 else 0
    chance = 1.0 / len(dept_centroids)

    print(f"\n  === TEST 1 RESULTS ({total} proteins tested) ===")
    print(f"  Top-1 accuracy:    {top1_acc:.1%}  ({correct_top1}/{total})")
    print(f"  Top-3 accuracy:    {top3_acc:.1%}")
    print(f"  Top-5 accuracy:    {top5_acc:.1%}")
    print(f"  Majority baseline: {majority_acc:.1%}  (always predict '{most_common_dept}')")
    print(f"  Chance (1/{len(dept_centroids)}):    {chance:.1%}")
    print(f"  Lift over majority: {top1_acc - majority_acc:+.1%}")
    print(f"  Lift over chance:   {top1_acc - chance:+.1%}")

    per_dept_accs = {}
    print(f"\n  Per-department accuracy:")
    for dept in sorted(per_dept_total.keys()):
        n = per_dept_total[dept]
        c = per_dept_correct.get(dept, 0)
        acc = c / n if n > 0 else 0
        per_dept_accs[dept] = {"correct": c, "total": n, "accuracy": round(acc, 4)}
        print(f"    {dept:20s}: {acc:5.1%} ({c}/{n})")

    elapsed = time.time() - t0
    print(f"  ({elapsed:.1f}s)")

    return {
        "n_tested": total,
        "top1_accuracy": round(top1_acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "majority_baseline": round(majority_acc, 4),
        "majority_dept": most_common_dept,
        "chance": round(chance, 4),
        "lift_over_majority": round(top1_acc - majority_acc, 4),
        "lift_over_chance": round(top1_acc - chance, 4),
        "n_departments": len(dept_centroids),
        "per_department": per_dept_accs,
    }


def test2_disease_prediction(state, protein_dept_seqs, primitives):
    print("\n" + "=" * 72)
    print("  TEST 2: Disease Prediction from Primitive Composition")
    print("=" * 72)
    t0 = time.time()

    if not os.path.exists(KNOCKOUT_RESULTS_PATH):
        print("  WARNING: knockout results not found, skipping")
        return {"skipped": True, "reason": "knockout results not found"}

    with open(KNOCKOUT_RESULTS_PATH) as f:
        ko_data = json.load(f)

    ko_results = ko_data["results"]
    disease_entries = [g for g in ko_results
                       if "disease" in g.get("categories", [])
                       and g.get("disease_ground_truth")]

    print(f"  Disease genes with ground truth: {len(disease_entries)}")

    gene_cache = state["gene_cache"]
    gene_to_uids = defaultdict(list)
    for uid, gene in gene_cache.items():
        if gene:
            gene_to_uids[gene].append(uid)

    gene_prim_features = {}
    for gene, uids in gene_to_uids.items():
        all_features = []
        for uid in uids:
            if uid in protein_dept_seqs:
                features = encode_protein_primitives(uid, protein_dept_seqs[uid], primitives)
                all_features.append(features)
        if all_features:
            gene_prim_features[gene] = np.mean(all_features, axis=0)

    print(f"  Genes with primitive features: {len(gene_prim_features)}")

    gene_to_dept = load_gene_departments()

    all_dept_genes = defaultdict(list)
    for gene, dept in gene_to_dept.items():
        if gene in gene_prim_features and dept in DEPT_TO_IDX:
            all_dept_genes[dept].append(gene)

    dept_centroids = {}
    for dept, genes in all_dept_genes.items():
        if len(genes) >= 5:
            profiles = [gene_prim_features[g] for g in genes]
            dept_centroids[dept] = np.mean(profiles, axis=0)

    print(f"  Department centroids: {len(dept_centroids)}")

    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    total = 0
    results_per_gene = []

    ko_correct_top1 = 0
    ko_total = 0

    for entry in disease_entries:
        gene = entry["gene"]
        ground_truth = entry["disease_ground_truth"]

        if isinstance(ground_truth, dict):
            dept_val = ground_truth.get("department", "")
            known_depts = [dept_val] if dept_val else []
        elif isinstance(ground_truth, str):
            known_depts = [ground_truth]
        elif isinstance(ground_truth, list):
            known_depts = []
            for gt in ground_truth:
                if isinstance(gt, dict):
                    d = gt.get("department", "")
                    if d:
                        known_depts.append(d)
                elif isinstance(gt, str):
                    known_depts.append(gt)
        else:
            continue

        known_depts = [d for d in known_depts if d in DEPT_TO_IDX]
        if not known_depts:
            continue

        if gene not in gene_prim_features:
            continue

        features = gene_prim_features[gene]
        if features.sum() == 0:
            continue

        sims = {}
        for dept, centroid in dept_centroids.items():
            n1, n2 = np.linalg.norm(features), np.linalg.norm(centroid)
            if n1 > 0 and n2 > 0:
                sims[dept] = float(np.dot(features, centroid) / (n1 * n2))
            else:
                sims[dept] = 0

        ranked = sorted(sims.keys(), key=lambda d: -sims[d])

        top1_match = ranked[0] in known_depts if ranked else False
        top3_match = any(d in known_depts for d in ranked[:3])
        top5_match = any(d in known_depts for d in ranked[:5])

        if top1_match:
            correct_top1 += 1
        if top3_match:
            correct_top3 += 1
        if top5_match:
            correct_top5 += 1
        total += 1

        ko_disrupt_concordant = entry.get("disease_disrupt_concordant", False)
        ko_enrich_concordant = entry.get("disease_enrich_concordant", False)
        if ko_disrupt_concordant or ko_enrich_concordant:
            ko_correct_top1 += 1
        ko_total += 1

        results_per_gene.append({
            "gene": gene,
            "department": entry.get("department", "?"),
            "known_disease_depts": known_depts,
            "predicted_top3": ranked[:3],
            "top1_match": top1_match,
            "top3_match": top3_match,
            "top5_match": top5_match,
            "ko_concordant": ko_disrupt_concordant or ko_enrich_concordant,
            "ko_top_disrupted": entry.get("top_disrupted_dept"),
            "ko_top_enriched": entry.get("top_enriched_dept"),
        })

    if total == 0:
        return {"skipped": True, "reason": "no testable disease genes"}

    top1_acc = correct_top1 / total
    top3_acc = correct_top3 / total
    top5_acc = correct_top5 / total
    chance = 1.0 / len(dept_centroids) if dept_centroids else 0
    ko_top1_acc = ko_correct_top1 / ko_total if ko_total > 0 else 0

    print(f"\n  === TEST 2 RESULTS ({total} disease genes) ===")
    print(f"  PRIMITIVES ONLY (no knockout simulation):")
    print(f"    Top-1 accuracy:   {top1_acc:.1%} ({correct_top1}/{total})")
    print(f"    Top-3 accuracy:   {top3_acc:.1%}")
    print(f"    Top-5 accuracy:   {top5_acc:.1%}")
    print(f"    Chance (1/{len(dept_centroids)}):  {chance:.1%}")
    print(f"    Lift over chance: {top1_acc - chance:+.1%}")
    print(f"")
    print(f"  KNOCKOUT SIMULATION (for comparison):")
    print(f"    Top-1 concordance: {ko_top1_acc:.1%} ({ko_correct_top1}/{ko_total})")
    print(f"")
    print(f"  PRIMITIVE vs KNOCKOUT:")
    print(f"    Primitive/Knockout ratio: {top1_acc/ko_top1_acc:.2f}x" if ko_top1_acc > 0 else "    N/A")

    print(f"\n  Per-gene predictions:")
    for r in sorted(results_per_gene, key=lambda x: -x["top1_match"])[:20]:
        prim_mark = "P+" if r["top1_match"] else "P-"
        ko_mark = "K+" if r["ko_concordant"] else "K-"
        print(f"    {prim_mark} {ko_mark} {r['gene']:10s} dept={r['department']:15s} "
              f"known={r['known_disease_depts'][:2]} pred={r['predicted_top3'][:2]} "
              f"ko_top={r['ko_top_disrupted']}")

    both_correct = sum(1 for r in results_per_gene if r["top1_match"] and r["ko_concordant"])
    prim_only = sum(1 for r in results_per_gene if r["top1_match"] and not r["ko_concordant"])
    ko_only = sum(1 for r in results_per_gene if not r["top1_match"] and r["ko_concordant"])
    neither = sum(1 for r in results_per_gene if not r["top1_match"] and not r["ko_concordant"])

    print(f"\n  Agreement matrix (Primitive × Knockout):")
    print(f"    Both correct:    {both_correct}")
    print(f"    Primitive only:  {prim_only}")
    print(f"    Knockout only:   {ko_only}")
    print(f"    Neither:         {neither}")

    elapsed = time.time() - t0
    print(f"  ({elapsed:.1f}s)")

    return {
        "n_disease_genes": total,
        "primitive_top1": round(top1_acc, 4),
        "primitive_top3": round(top3_acc, 4),
        "primitive_top5": round(top5_acc, 4),
        "chance": round(chance, 4),
        "lift_over_chance": round(top1_acc - chance, 4),
        "knockout_top1": round(ko_top1_acc, 4),
        "primitive_vs_knockout_ratio": round(top1_acc / ko_top1_acc, 4) if ko_top1_acc > 0 else None,
        "agreement": {
            "both_correct": both_correct,
            "primitive_only": prim_only,
            "knockout_only": ko_only,
            "neither": neither,
        },
        "n_dept_centroids": len(dept_centroids),
        "per_gene": results_per_gene,
    }


def main():
    state = load_state()
    vocab_dept = build_token_dept_map()
    protein_dept_seqs = build_protein_dept_sequences(state, vocab_dept)
    primitives = load_primitives()

    t1 = test1_department_prediction(state, protein_dept_seqs, primitives)
    t2 = test2_disease_prediction(state, protein_dept_seqs, primitives)

    output = {
        "test1_department_prediction": t1,
        "test2_disease_prediction": t2,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*72}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*72}")
    print(f"  Test 1 — Department Prediction:")
    print(f"    Top-1: {t1['top1_accuracy']:.1%}  (majority: {t1['majority_baseline']:.1%}, "
          f"chance: {t1['chance']:.1%})")
    print(f"    Top-3: {t1['top3_accuracy']:.1%}")
    print(f"    Lift over majority: {t1['lift_over_majority']:+.1%}")
    if not t2.get("skipped"):
        t2_top1 = t2.get("primitive_top1", t2.get("top1_accuracy", 0))
        t2_top3 = t2.get("primitive_top3", t2.get("top3_accuracy", 0))
        t2_chance = t2.get("chance", 0)
        t2_ko = t2.get("knockout_top1", 0)
        print(f"  Test 2 — Disease Prediction:")
        print(f"    Primitives top-1: {t2_top1:.1%}  (chance: {t2_chance:.1%})")
        print(f"    Primitives top-3: {t2_top3:.1%}")
        print(f"    Knockout top-1:   {t2_ko:.1%}")
    else:
        print(f"  Test 2 — Skipped: {t2.get('reason', 'unknown')}")
    print(f"{'='*72}")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
