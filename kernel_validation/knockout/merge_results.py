#!/usr/bin/env python3
"""
Merge knockout simulation shard results into final analysis.
Run after all shards complete:
    python3 validation/knockout/merge_results.py
"""

import csv
import json
import os
import sys
import glob
import numpy as np
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "gene_manifest.json")
DEPMAP_CACHE = "/tmp/depmap_gene_essentiality.csv"
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "knockout_full_results.json")
OUTPUT_SUPP = os.path.join(os.path.dirname(__file__), "knockout_supplementary_table.csv")

sys.path.insert(0, os.path.dirname(__file__))
from disease_gene_ground_truth import DISEASE_GENE_GROUND_TRUTH


def load_all_results():
    shard_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "shard_*.jsonl")))
    results = {}
    for sf in shard_files:
        with open(sf) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    results[rec["gene"]] = rec
                except (json.JSONDecodeError, KeyError):
                    pass
    return results


def compute_auroc(labels, scores):
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = 0
    auc = 0.0
    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            auc += tp
    return auc / (n_pos * n_neg)


def compute_ci_bootstrap(labels, scores, metric_fn, n_boot=1000, ci=0.95):
    rng = np.random.RandomState(42)
    n = len(labels)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        bl = [labels[i] for i in idx]
        bs = [scores[i] for i in idx]
        if sum(bl) > 0 and sum(bl) < len(bl):
            vals.append(metric_fn(bl, bs))
    if not vals:
        return (0, 0)
    lo = np.percentile(vals, (1 - ci) / 2 * 100)
    hi = np.percentile(vals, (1 + ci) / 2 * 100)
    return (round(lo, 4), round(hi, 4))


def main():
    print("Loading results from all shards...")
    results = load_all_results()
    print(f"  Total genes with results: {len(results)}")

    if len(results) == 0:
        print("ERROR: No results found. Run shards first.")
        return

    essential_genes = [g for g, r in results.items() if "essential" in r.get("categories", [])]
    nonessential_genes = [g for g, r in results.items() if "nonessential" in r.get("categories", [])]

    print(f"\n{'='*72}")
    print(f"  1. AUROC: Essential vs Non-Essential Discrimination")
    print(f"{'='*72}")
    print(f"  Essential genes: {len(essential_genes)}")
    print(f"  Non-essential genes: {len(nonessential_genes)}")

    auroc_total = auroc_z = auroc_total_z = cohens_d = None
    if essential_genes and nonessential_genes:
        labels = [1]*len(essential_genes) + [0]*len(nonessential_genes)
        scores_total = [results[g]["total_disruption"] for g in essential_genes] + \
                       [results[g]["total_disruption"] for g in nonessential_genes]
        scores_z = [results[g]["total_disruption_z"] for g in essential_genes] + \
                   [results[g]["total_disruption_z"] for g in nonessential_genes]

        auroc_total = compute_auroc(labels, scores_total)
        auroc_total_z = compute_auroc(labels, scores_z)
        ci_total = compute_ci_bootstrap(labels, scores_total, compute_auroc)
        ci_z = compute_ci_bootstrap(labels, scores_z, compute_auroc)

        ess_vals = [results[g]["total_disruption"] for g in essential_genes]
        noness_vals = [results[g]["total_disruption"] for g in nonessential_genes]
        ess_mean, noness_mean = np.mean(ess_vals), np.mean(noness_vals)
        pooled_std = np.sqrt((np.var(ess_vals) + np.var(noness_vals)) / 2)
        cohens_d = (ess_mean - noness_mean) / pooled_std if pooled_std > 0 else 0

        print(f"  AUROC (total disruption): {auroc_total:.4f}  95% CI: {ci_total}")
        print(f"  AUROC (total disruption z): {auroc_total_z:.4f}  95% CI: {ci_z}")
        print(f"  Essential mean disruption: {ess_mean:.2f}")
        print(f"  Non-essential mean disruption: {noness_mean:.2f}")
        print(f"  Cohen's d: {cohens_d:.4f}")

    disease_genes_results = {g: r for g, r in results.items() if g in DISEASE_GENE_GROUND_TRUTH}
    print(f"\n{'='*72}")
    print(f"  2. Disease Mechanism Concordance")
    print(f"{'='*72}")
    print(f"  Disease genes with results: {len(disease_genes_results)}/50")

    concordant_disrupt = 0
    concordant_enrich = 0
    disease_details = []
    enrich_ranks = []
    disrupt_ranks = []

    for gene in sorted(disease_genes_results.keys()):
        result = disease_genes_results[gene]
        gt = DISEASE_GENE_GROUND_TRUTH[gene]
        gt_dept = gt["department"]
        pred_disrupt = result["top_disrupted_dept"]
        pred_enrich = result["top_enriched_dept"]
        match_d = result.get("disease_disrupt_concordant", pred_disrupt == gt_dept)
        match_e = result.get("disease_enrich_concordant", pred_enrich == gt_dept)

        gt_d_rank = result.get("disease_gt_disrupt_rank", 99)
        gt_e_rank = result.get("disease_gt_enrich_rank", 99)

        if match_d: concordant_disrupt += 1
        if match_e: concordant_enrich += 1
        if gt_d_rank: disrupt_ranks.append(gt_d_rank)
        if gt_e_rank: enrich_ranks.append(gt_e_rank)

        disease_details.append({
            "gene": gene,
            "disease": gt["disease"],
            "gt_department": gt_dept,
            "pred_disruption": pred_disrupt,
            "pred_enrichment": pred_enrich,
            "concordant_disruption": match_d,
            "concordant_enrichment": match_e,
            "gt_disrupt_rank": gt_d_rank,
            "gt_enrich_rank": gt_e_rank,
            "gt_disrupt_z": result.get("disease_gt_disrupt_z", 0),
            "gt_enrich_z": result.get("disease_gt_enrich_z", 0),
        })

        d_tag = "MATCH" if match_d else "     "
        e_tag = "MATCH" if match_e else "     "
        print(
            f"  {gene:10s} gt={gt_dept:15s} | "
            f"disrupt: {pred_disrupt:15s} rk={gt_d_rank:2d} [{d_tag}] | "
            f"enrich: {pred_enrich:15s} rk={gt_e_rank:2d} [{e_tag}]"
        )

    total_disease = len(disease_genes_results)
    conc_d_rate = concordant_disrupt / total_disease if total_disease > 0 else 0
    conc_e_rate = concordant_enrich / total_disease if total_disease > 0 else 0
    mrr_d = np.mean([1/r for r in disrupt_ranks]) if disrupt_ranks else 0
    mrr_e = np.mean([1/r for r in enrich_ranks]) if enrich_ranks else 0
    top3_d = sum(1 for r in disrupt_ranks if r <= 3) / total_disease if total_disease > 0 else 0
    top3_e = sum(1 for r in enrich_ranks if r <= 3) / total_disease if total_disease > 0 else 0
    top5_d = sum(1 for r in disrupt_ranks if r <= 5) / total_disease if total_disease > 0 else 0
    top5_e = sum(1 for r in enrich_ranks if r <= 5) / total_disease if total_disease > 0 else 0

    print(f"\n  Concordance (top-1):  Disruption={concordant_disrupt}/{total_disease} ({conc_d_rate:.1%})  Enrichment={concordant_enrich}/{total_disease} ({conc_e_rate:.1%})")
    print(f"  Top-3 rate:           Disruption={top3_d:.1%}  Enrichment={top3_e:.1%}")
    print(f"  Top-5 rate:           Disruption={top5_d:.1%}  Enrichment={top5_e:.1%}")
    print(f"  MRR:                  Disruption={mrr_d:.4f}  Enrichment={mrr_e:.4f}")
    print(f"  Median rank (/22):    Disruption={np.median(disrupt_ranks):.0f}  Enrichment={np.median(enrich_ranks):.0f}")

    all_total_z = [r["total_disruption_z"] for r in results.values()]
    all_own_disrupt_z = [r.get("own_dept_disrupt_z", 0) for r in results.values()]
    all_own_enrich_z = [r.get("own_dept_enrich_z", 0) for r in results.values()]

    print(f"\n{'='*72}")
    print(f"  3. Aggregate Null Model Statistics")
    print(f"{'='*72}")
    print(f"  Total disruption z-score (all genes):")
    print(f"    Mean: {np.mean(all_total_z):.4f}  Median: {np.median(all_total_z):.4f}")
    print(f"    z > 2: {sum(1 for z in all_total_z if z > 2)}  z > 3: {sum(1 for z in all_total_z if z > 3)}")
    print(f"  Own-department disruption z (all genes):")
    print(f"    Mean: {np.mean(all_own_disrupt_z):.4f}  Median: {np.median(all_own_disrupt_z):.4f}")
    print(f"    z > 2: {sum(1 for z in all_own_disrupt_z if z > 2)}")
    print(f"  Own-department enrichment z (all genes):")
    print(f"    Mean: {np.mean(all_own_enrich_z):.4f}  Median: {np.median(all_own_enrich_z):.4f}")
    print(f"    z > 2: {sum(1 for z in all_own_enrich_z if z > 2)}")

    summary = {
        "total_genes": len(results),
        "essential_count": len(essential_genes),
        "nonessential_count": len(nonessential_genes),
        "disease_count": total_disease,
        "auroc_total_disruption": round(auroc_total, 4) if auroc_total is not None else None,
        "auroc_total_z": round(auroc_total_z, 4) if auroc_total_z is not None else None,
        "cohens_d": round(cohens_d, 4) if cohens_d is not None else None,
        "disease_concordance_disruption": round(conc_d_rate, 4),
        "disease_concordance_enrichment": round(conc_e_rate, 4),
        "disease_top3_disruption": round(top3_d, 4),
        "disease_top3_enrichment": round(top3_e, 4),
        "disease_top5_disruption": round(top5_d, 4),
        "disease_top5_enrichment": round(top5_e, 4),
        "disease_mrr_disruption": round(mrr_d, 4),
        "disease_mrr_enrichment": round(mrr_e, 4),
        "aggregate_total_z_mean": round(float(np.mean(all_total_z)), 4),
        "aggregate_total_z_median": round(float(np.median(all_total_z)), 4),
        "genes_total_z_gt2": sum(1 for z in all_total_z if z > 2),
        "genes_total_z_gt3": sum(1 for z in all_total_z if z > 3),
        "own_dept_disrupt_z_mean": round(float(np.mean(all_own_disrupt_z)), 4),
        "own_dept_enrich_z_mean": round(float(np.mean(all_own_enrich_z)), 4),
        "disease_details": disease_details,
    }

    all_results_list = []
    for gene in sorted(results.keys()):
        r = results[gene]
        entry = {
            "gene": gene,
            "uid": r["uid"],
            "n_tokens": r["n_tokens"],
            "department": r["department"],
            "categories": r["categories"],
            "chronos": r.get("chronos"),
            "total_disruption": r["total_disruption"],
            "total_disruption_z": r["total_disruption_z"],
            "top_disrupted_dept": r["top_disrupted_dept"],
            "top_disrupted_z": r["top_disrupted_z"],
            "top_enriched_dept": r["top_enriched_dept"],
            "top_enriched_z": r["top_enriched_z"],
            "own_dept_disrupt_z": r.get("own_dept_disrupt_z", 0),
            "own_dept_disrupt_rank": r.get("own_dept_disrupt_rank"),
            "own_dept_enrich_z": r.get("own_dept_enrich_z", 0),
            "own_dept_enrich_rank": r.get("own_dept_enrich_rank"),
            "n_null_perms": r["n_null_perms"],
        }
        if gene in DISEASE_GENE_GROUND_TRUTH:
            entry["disease_ground_truth"] = DISEASE_GENE_GROUND_TRUTH[gene]
            entry["disease_disrupt_concordant"] = r.get("disease_disrupt_concordant")
            entry["disease_enrich_concordant"] = r.get("disease_enrich_concordant")
        all_results_list.append(entry)

    final = {"summary": summary, "results": all_results_list}
    with open(OUTPUT_JSON, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved: {OUTPUT_JSON}")

    with open(OUTPUT_SUPP, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "gene", "uniprot_id", "n_tokens", "assigned_department",
            "category", "chronos_mean",
            "total_disruption", "total_disruption_z",
            "top_disrupted_dept", "top_disrupted_z",
            "top_enriched_dept", "top_enriched_z",
            "own_dept_disrupt_z", "own_dept_disrupt_rank",
            "own_dept_enrich_z", "own_dept_enrich_rank",
            "n_null_perms",
            "disease_gt_dept", "disease_disrupt_concordant", "disease_enrich_concordant",
        ])
        for r in all_results_list:
            writer.writerow([
                r["gene"], r["uid"], r["n_tokens"], r["department"],
                "|".join(r["categories"]), r.get("chronos", ""),
                r["total_disruption"], r["total_disruption_z"],
                r["top_disrupted_dept"], r["top_disrupted_z"],
                r["top_enriched_dept"], r["top_enriched_z"],
                r["own_dept_disrupt_z"], r["own_dept_disrupt_rank"],
                r["own_dept_enrich_z"], r["own_dept_enrich_rank"],
                r["n_null_perms"],
                r.get("disease_ground_truth", {}).get("department", ""),
                r.get("disease_disrupt_concordant", ""),
                r.get("disease_enrich_concordant", ""),
            ])
    print(f"Saved: {OUTPUT_SUPP}")

    print(f"\n{'='*72}")
    print(f"  HEADLINE NUMBERS")
    print(f"{'='*72}")
    if auroc_total is not None:
        print(f"  AUROC (essential vs non-essential): {auroc_total:.4f}")
    print(f"  Disease concordance (enrichment):   {conc_e_rate:.1%} top-1, {top5_e:.1%} top-5 ({total_disease} genes)")
    print(f"  Disease MRR (enrichment):           {mrr_e:.4f}")
    print(f"  Aggregate null model z (total):     {np.mean(all_total_z):.4f}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
