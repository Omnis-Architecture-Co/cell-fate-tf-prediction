#!/usr/bin/env python3
"""
Compute full 22-dimensional disruption profiles for knockout genes.
Saves to disruption_profiles.json for use by the 5D algebra applications.
"""

import csv
import json
import os
import pickle
import sys
import time
import numpy as np
from collections import defaultdict

STATE_PATH = "/tmp/module8_full_state.pkl"
VOCAB_PATH = "server/data/human/vocabulary.csv"
GENE_DEPTS_PATH = "server/data/human/gene_departments.csv"
KO_RESULTS_PATH = "validation/knockout/knockout_full_results.json"
OUTPUT_PATH = "validation/knockout/disruption_profiles.json"

VALID_DEPARTMENTS = sorted([
    "Apoptosis", "Cell adhesion", "Cell cycle", "Chromatin", "Cytoskeleton",
    "DNA repair", "GTPase", "Immune", "Ion channel", "Kinase", "Methylation",
    "Nuc acid bind", "Phosphatase", "Protein folding", "Proteolysis",
    "RNA processing", "Signaling", "Structural", "Transcription",
    "Translation", "Transport", "Ubiquitin",
])
D2I = {d: i for i, d in enumerate(VALID_DEPARTMENTS)}
N = len(VALID_DEPARTMENTS)


def main():
    print("[1] Loading state...")
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

    ko_entries = sorted(ko_data["results"],
                         key=lambda e: abs(e.get("total_disruption_z", 0)),
                         reverse=True)

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
        for uid in uids:
            all_tokens.update(ptt.get(uid, []))
        dept_token_sets[dept] = all_tokens

    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  {len(ko_entries)} knockout genes, computing top 2000...")

    profiles = {}
    t1 = time.time()

    for gi, entry in enumerate(ko_entries[:2000]):
        gene = entry["gene"]

        gene_uids = gene_to_uids.get(gene, [])
        if not gene_uids:
            continue

        gene_tokens = set()
        for uid in gene_uids:
            gene_tokens.update(ptt.get(uid, []))

        if not gene_tokens:
            continue

        profile = {}
        for dept in VALID_DEPARTMENTS:
            d_uids = dept_uids.get(dept, [])
            if not d_uids:
                profile[dept] = 0.0
                continue

            sample = d_uids[:500]
            total_connections = 0
            lost_connections = 0
            for uid in sample:
                toks = set(ptt.get(uid, []))
                total_connections += len(toks)
                lost_connections += len(toks & gene_tokens)

            profile[dept] = lost_connections / max(total_connections, 1)

        profiles[gene] = profile

        if (gi + 1) % 200 == 0:
            elapsed = time.time() - t1
            rate = (gi + 1) / elapsed
            remaining = (2000 - gi - 1) / rate
            print(f"  [{gi+1}/2000] {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")
            sys.stdout.flush()

    print(f"\n[2] Computed {len(profiles)} disruption profiles in {time.time()-t1:.1f}s")

    output = {
        "n_genes": len(profiles),
        "departments": VALID_DEPARTMENTS,
        "profiles": profiles,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
