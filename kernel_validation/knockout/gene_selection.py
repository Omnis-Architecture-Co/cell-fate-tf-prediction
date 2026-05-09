"""
Programmatic gene selection for knockout simulation.
Selects all genes in the dispatch graph, tags them into categories.
Categories are post-hoc filters on the full run, not pre-hoc selections.
"""

import csv
import json
import os
import pickle
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from disease_gene_ground_truth import DISEASE_GENE_GROUND_TRUTH

DEPMAP_CACHE = "/tmp/depmap_gene_essentiality.csv"
STATE_PATH = "/tmp/module8_full_state.pkl"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "gene_manifest.json")


def load_depmap():
    depmap = {}
    if not os.path.exists(DEPMAP_CACHE):
        print(f"ERROR: DepMap cache not found at {DEPMAP_CACHE}")
        print("Run depmap_essentiality_test.py first or re-download")
        return depmap
    with open(DEPMAP_CACHE) as f:
        for row in csv.DictReader(f):
            depmap[row["gene"]] = {
                "mean_chronos": float(row["mean_chronos"]),
                "n_lines": int(row["n_lines"]),
                "pct_dependent": float(row["pct_dependent"]),
            }
    return depmap


def build_manifest():
    print("Loading dispatch graph state...")
    with open(STATE_PATH, "rb") as f:
        state = pickle.load(f)

    ptt = state["ptt"]
    gene_to_uid = state["gene_to_uid"]
    dept_cache = state["dept_cache"]

    print("Loading DepMap data...")
    depmap = load_depmap()

    print("Loading gene departments...")
    gene_depts = {}
    with open("server/data/human/gene_departments.csv") as f:
        for row in csv.DictReader(f):
            gene_depts[row["gene"]] = row["department"]

    all_genes = sorted(gene_to_uid.keys())
    print(f"Total genes in dispatch graph: {len(all_genes)}")

    manifest = {}
    for gene in all_genes:
        uid = gene_to_uid[gene]
        n_tokens = len(ptt.get(uid, []))
        if n_tokens == 0:
            continue

        dept = gene_depts.get(gene, dept_cache.get(gene, ["Unknown"])[0] if gene in dept_cache else "Unknown")
        chronos = depmap.get(gene, {}).get("mean_chronos")

        categories = []

        if gene in DISEASE_GENE_GROUND_TRUTH:
            categories.append("disease")

        if chronos is not None:
            if chronos < -0.5:
                categories.append("essential")
            elif chronos > 0.0:
                categories.append("nonessential")

        if not categories:
            categories.append("other")

        manifest[gene] = {
            "uid": uid,
            "n_tokens": n_tokens,
            "department": dept,
            "categories": categories,
            "chronos": round(chronos, 4) if chronos is not None else None,
        }

    cat_counts = {}
    for g, info in manifest.items():
        for c in info["categories"]:
            cat_counts[c] = cat_counts.get(c, 0) + 1

    print(f"\nManifest: {len(manifest)} genes")
    for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:15s}: {n}")

    disease_in_graph = [g for g in DISEASE_GENE_GROUND_TRUTH if g in manifest]
    disease_missing = [g for g in DISEASE_GENE_GROUND_TRUTH if g not in manifest]
    print(f"\nDisease genes in graph: {len(disease_in_graph)}/50")
    if disease_missing:
        print(f"  Missing: {disease_missing}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "total_genes": len(manifest),
            "category_counts": cat_counts,
            "disease_ground_truth": {
                g: DISEASE_GENE_GROUND_TRUTH[g] for g in disease_in_graph
            },
            "genes": manifest,
        }, f, indent=2)

    print(f"\nSaved to {OUTPUT_PATH}")
    return manifest


if __name__ == "__main__":
    build_manifest()
