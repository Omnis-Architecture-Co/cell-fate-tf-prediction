#!/usr/bin/env python3
"""
Follow-up Experiments: Gauged Tropical Module Hypothesis
========================================================

Experiment 1: Identify the 5 clusters on PC1-removed profiles
Experiment 2: Compiler test on residuals (PC1-removed)
Experiment 3: Parent-child cosine after PC1 removal
Experiment 4: Tropical idempotency test

Usage:
    python3 -u validation/knockout/followup_experiments.py
"""

import csv
import json
import os
import pickle
import random
import time
from collections import defaultdict, Counter

import numpy as np
from scipy import stats

STATE_PATH = "/tmp/module8_full_state.pkl"
VOCAB_PATH = "server/data/human/vocabulary.csv"
PRIMITIVES_PATH = "beta_transfer/genome_primitives.csv"
NESTING_PATH = "beta_transfer/genome_nesting_hierarchy.csv"
GENE_DEPTS_PATH = "server/data/human/gene_departments.csv"
PROFILES_PATH = "validation/knockout/disruption_profiles_full.json"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "followup_results.json")

VALID_DEPARTMENTS = sorted([
    "Apoptosis", "Cell adhesion", "Cell cycle", "Chromatin", "Cytoskeleton",
    "DNA repair", "GTPase", "Immune", "Ion channel", "Kinase", "Methylation",
    "Nuc acid bind", "Phosphatase", "Protein folding", "Proteolysis",
    "RNA processing", "Signaling", "Structural", "Transcription",
    "Translation", "Transport", "Ubiquitin",
])
DEPT_TO_IDX = {d: i for i, d in enumerate(VALID_DEPARTMENTS)}


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def load_state():
    if not os.path.exists(STATE_PATH):
        from validation.sensitivity.module8_full_shuffle import load_state_from_db
        load_state_from_db()
    with open(STATE_PATH, "rb") as f:
        return pickle.load(f)


def load_profiles():
    with open(PROFILES_PATH) as f:
        data = json.load(f)
    profiles = {}
    for gene, profile in data["profiles"].items():
        if isinstance(profile, dict):
            profiles[gene] = np.array([profile.get(d, 0.0) for d in VALID_DEPARTMENTS])
        else:
            profiles[gene] = np.array(profile)
    return profiles, data.get("departments", VALID_DEPARTMENTS)


def load_gene_depts():
    gene_depts = {}
    with open(GENE_DEPTS_PATH) as f:
        for row in csv.DictReader(f):
            gene_depts[row["gene"]] = row["department"]
    return gene_depts


def build_primitive_carriers(state, min_carriers=20):
    ptt = state["ptt"]
    gene_cache = state["gene_cache"]

    vocab_dept = {}
    with open(VOCAB_PATH) as f:
        for row in csv.DictReader(f):
            tok = row["word_hex"].replace("0x", "").upper()
            vocab_dept[tok] = row["primary_function"]

    protein_dept_seqs = {}
    for uid, tokens in ptt.items():
        depts = []
        for tok in tokens:
            d = vocab_dept.get(tok.upper())
            if d and d in DEPT_TO_IDX:
                depts.append(d)
        if depts:
            compressed = []
            for d in depts:
                if not compressed or compressed[-1] != d:
                    compressed.append(d)
            protein_dept_seqs[uid] = "|".join(compressed)

    primitives = []
    with open(PRIMITIVES_PATH) as f:
        for row in csv.DictReader(f):
            primitives.append(row["function_sequence"])

    prim_to_genes = {}
    for seq in primitives:
        ds = [d for d in seq.split("|") if d in DEPT_TO_IDX]
        if not ds:
            continue
        search = "|".join(ds)
        genes = set()
        for uid, pseq in protein_dept_seqs.items():
            if search in pseq:
                g = gene_cache.get(uid)
                if g:
                    genes.add(g)
        if len(genes) >= min_carriers:
            prim_to_genes[seq] = genes

    return prim_to_genes, protein_dept_seqs


def compute_pc1_removal(profiles):
    gene_names = list(profiles.keys())
    M = np.array([profiles[g] for g in gene_names])
    M_centered = M - M.mean(axis=0)
    U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
    pc1_component = np.outer(U[:, 0] * S[0], Vt[0])
    M_residual = M_centered - pc1_component

    residual_profiles = {}
    for i, g in enumerate(gene_names):
        residual_profiles[g] = M_residual[i]

    return residual_profiles, M_centered, M_residual, gene_names, U, S, Vt


def experiment_1_identify_clusters(profiles, residual_profiles, gene_names, gene_depts):
    print("\n" + "=" * 72)
    print("FOLLOWUP 1: IDENTIFY THE 5 CLUSTERS")
    print("=" * 72)

    from sklearn.cluster import KMeans

    M_res = np.array([residual_profiles[g] for g in gene_names if g in residual_profiles])
    names = [g for g in gene_names if g in residual_profiles]

    km = KMeans(n_clusters=5, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(M_res)

    cluster_dept_profiles = {}
    cluster_sizes = {}
    cluster_top_depts = {}

    for c in range(5):
        mask = labels == c
        cluster_genes = [names[i] for i in range(len(names)) if mask[i]]
        cluster_sizes[c] = int(mask.sum())

        dept_counts = Counter()
        for g in cluster_genes:
            d = gene_depts.get(g)
            if d and d in DEPT_TO_IDX:
                dept_counts[d] += 1

        total = sum(dept_counts.values())
        if total > 0:
            dept_pcts = {d: round(count / total, 4) for d, count in dept_counts.most_common()}
        else:
            dept_pcts = {}

        cluster_dept_profiles[c] = dept_pcts
        cluster_top_depts[c] = [d for d, _ in dept_counts.most_common(5)]

        print(f"\n  Cluster {c} ({cluster_sizes[c]} genes):")
        print(f"    Top departments:")
        for d, count in dept_counts.most_common(5):
            print(f"      {d}: {count} ({count/total:.1%})")

    centroid_cosines = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            centroid_cosines[i][j] = cosine_sim(km.cluster_centers_[i], km.cluster_centers_[j])

    print(f"\n  Inter-cluster centroid cosines:")
    print(f"  {'':>20s}", end="")
    for c in range(5):
        print(f"  C{c:d}", end="")
    print()
    for i in range(5):
        top = cluster_top_depts[i][0] if cluster_top_depts[i] else "?"
        print(f"  C{i} ({top[:12]:>12s})", end="")
        for j in range(5):
            print(f"  {centroid_cosines[i][j]:+.2f}", end="")
        print()

    from sklearn.metrics import silhouette_score
    sil = silhouette_score(M_res, labels, sample_size=min(3000, len(M_res)))
    print(f"\n  Silhouette score (k=5): {sil:.4f}")

    cluster_names = []
    for c in range(5):
        top3 = cluster_top_depts[c][:3]
        cluster_names.append(" / ".join(top3))
    print(f"\n  Proposed super-category names:")
    for c in range(5):
        print(f"    Cluster {c}: {cluster_names[c]}")

    return {
        "cluster_sizes": cluster_sizes,
        "cluster_dept_profiles": {str(k): v for k, v in cluster_dept_profiles.items()},
        "cluster_top_depts": {str(k): v for k, v in cluster_top_depts.items()},
        "cluster_names": cluster_names,
        "silhouette": round(sil, 4),
        "centroid_cosines": centroid_cosines.tolist(),
    }


def experiment_2_compiler_on_residuals(profiles, residual_profiles, prim_to_genes, gene_depts):
    print("\n" + "=" * 72)
    print("FOLLOWUP 2: COMPILER TEST ON RESIDUALS")
    print("=" * 72)

    gene_names = list(profiles.keys())
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    M_full = np.array([profiles[g] for g in gene_names])
    M_full_centered = M_full - M_full.mean(axis=0)
    M_res = np.array([residual_profiles[g] for g in gene_names])

    prim_sigs_full = {}
    prim_sigs_residual = {}
    for seq, genes in prim_to_genes.items():
        idxs = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        if len(idxs) < 20:
            continue
        prim_sigs_full[seq] = np.mean(M_full_centered[idxs], axis=0)
        prim_sigs_residual[seq] = np.mean(M_res[idxs], axis=0)

    print(f"  Primitives with signatures: {len(prim_sigs_full)}")

    dept_centroids_full = {}
    dept_centroids_res = {}
    for d in VALID_DEPARTMENTS:
        idxs = [gene_to_idx[g] for g in gene_names if gene_depts.get(g) == d and g in gene_to_idx]
        if len(idxs) >= 5:
            dept_centroids_full[d] = np.mean(M_full_centered[idxs], axis=0)
            dept_centroids_res[d] = np.mean(M_res[idxs], axis=0)

    cos_full = []
    cos_res = []
    for seq, genes in prim_to_genes.items():
        carrier_idxs = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        if len(carrier_idxs) < 20:
            continue

        np.random.seed(42)
        for trial in range(10):
            half = np.random.choice(carrier_idxs, size=len(carrier_idxs)//2, replace=False)
            other = [i for i in carrier_idxs if i not in set(half)]
            if len(other) < 5:
                continue

            sig1_full = np.mean(M_full_centered[half], axis=0)
            sig2_full = np.mean(M_full_centered[other], axis=0)
            cos_full.append(cosine_sim(sig1_full, sig2_full))

            sig1_res = np.mean(M_res[half], axis=0)
            sig2_res = np.mean(M_res[other], axis=0)
            cos_res.append(cosine_sim(sig1_res, sig2_res))

    print(f"\n  Split-half consistency:")
    print(f"    Full:     mean cosine {np.mean(cos_full):.4f} (n={len(cos_full)})")
    print(f"    Residual: mean cosine {np.mean(cos_res):.4f}")

    correct_full_1 = 0
    correct_res_1 = 0
    correct_full_3 = 0
    correct_res_3 = 0
    total_tested = 0

    for seq, genes in prim_to_genes.items():
        carrier_genes = [g for g in genes if g in gene_to_idx and g in gene_depts]
        if len(carrier_genes) < 20:
            continue

        sig_full = prim_sigs_full.get(seq)
        sig_res = prim_sigs_residual.get(seq)
        if sig_full is None or sig_res is None:
            continue

        actual_depts = [d for d in seq.split("|") if d in DEPT_TO_IDX]
        if not actual_depts:
            continue

        dists_full = {d: cosine_sim(sig_full, c) for d, c in dept_centroids_full.items()}
        dists_res = {d: cosine_sim(sig_res, c) for d, c in dept_centroids_res.items()}

        pred_full = max(dists_full, key=dists_full.get)
        pred_res = max(dists_res, key=dists_res.get)
        top3_full = sorted(dists_full, key=dists_full.get, reverse=True)[:3]
        top3_res = sorted(dists_res, key=dists_res.get, reverse=True)[:3]

        total_tested += 1
        if pred_full in actual_depts:
            correct_full_1 += 1
        if pred_res in actual_depts:
            correct_res_1 += 1
        if any(d in actual_depts for d in top3_full):
            correct_full_3 += 1
        if any(d in actual_depts for d in top3_res):
            correct_res_3 += 1

    if total_tested > 0:
        print(f"\n  Primitive → Department prediction ({total_tested} primitives):")
        print(f"    Full:     Top-1 {correct_full_1/total_tested:.1%}  Top-3 {correct_full_3/total_tested:.1%}")
        print(f"    Residual: Top-1 {correct_res_1/total_tested:.1%}  Top-3 {correct_res_3/total_tested:.1%}")
        print(f"    Δ Top-1: {(correct_res_1-correct_full_1)/total_tested:+.1%}")
        print(f"    Δ Top-3: {(correct_res_3-correct_full_3)/total_tested:+.1%}")

    gene_correct_full_1 = 0
    gene_correct_res_1 = 0
    gene_correct_full_3 = 0
    gene_correct_res_3 = 0
    gene_tested = 0

    for i, gene in enumerate(gene_names):
        true_dept = gene_depts.get(gene)
        if true_dept not in dept_centroids_full:
            continue
        gene_tested += 1

        dists_full = {d: cosine_sim(M_full_centered[i], c) for d, c in dept_centroids_full.items()}
        pred_full = max(dists_full, key=dists_full.get)
        top3_full = sorted(dists_full, key=dists_full.get, reverse=True)[:3]
        if pred_full == true_dept:
            gene_correct_full_1 += 1
        if true_dept in top3_full:
            gene_correct_full_3 += 1

        dists_res = {d: cosine_sim(M_res[i], c) for d, c in dept_centroids_res.items()}
        pred_res = max(dists_res, key=dists_res.get)
        top3_res = sorted(dists_res, key=dists_res.get, reverse=True)[:3]
        if pred_res == true_dept:
            gene_correct_res_1 += 1
        if true_dept in top3_res:
            gene_correct_res_3 += 1

    if gene_tested > 0:
        print(f"\n  Gene → Department prediction ({gene_tested} genes):")
        print(f"    Full:     Top-1 {gene_correct_full_1/gene_tested:.1%}  Top-3 {gene_correct_full_3/gene_tested:.1%}")
        print(f"    Residual: Top-1 {gene_correct_res_1/gene_tested:.1%}  Top-3 {gene_correct_res_3/gene_tested:.1%}")
        print(f"    Δ Top-1: {(gene_correct_res_1-gene_correct_full_1)/gene_tested:+.1%}")

    return {
        "n_primitives": total_tested,
        "prim_full_top1": round(correct_full_1/total_tested, 4) if total_tested else 0,
        "prim_res_top1": round(correct_res_1/total_tested, 4) if total_tested else 0,
        "prim_full_top3": round(correct_full_3/total_tested, 4) if total_tested else 0,
        "prim_res_top3": round(correct_res_3/total_tested, 4) if total_tested else 0,
        "gene_tested": gene_tested,
        "gene_full_top1": round(gene_correct_full_1/gene_tested, 4) if gene_tested else 0,
        "gene_res_top1": round(gene_correct_res_1/gene_tested, 4) if gene_tested else 0,
        "split_half_full": round(float(np.mean(cos_full)), 4) if cos_full else 0,
        "split_half_res": round(float(np.mean(cos_res)), 4) if cos_res else 0,
    }


def experiment_3_parent_child_residual(state, profiles, residual_profiles, protein_dept_seqs):
    print("\n" + "=" * 72)
    print("FOLLOWUP 3: PARENT-CHILD COSINE ON RESIDUALS")
    print("=" * 72)

    gene_cache = state["gene_cache"]
    nesting = list(csv.DictReader(open(NESTING_PATH)))

    parent_children = defaultdict(set)
    all_prims = set()
    for row in nesting:
        parent = row.get("outer_sequence", "")
        child = row.get("inner_sequence", "")
        if parent and child:
            parent_children[parent].add(child)
            all_prims.add(parent)
            all_prims.add(child)

    gene_to_pseqs = defaultdict(list)
    for uid, pseq in protein_dept_seqs.items():
        g = gene_cache.get(uid)
        if g and g in profiles:
            gene_to_pseqs[g].append(pseq)

    testable = set()
    for seq in all_prims:
        ds = [d for d in seq.split("|") if d in DEPT_TO_IDX]
        if ds and len(ds) <= 3:
            testable.add(seq)

    def get_signature(seq, prof_dict):
        ds = [d for d in seq.split("|") if d in DEPT_TO_IDX]
        if not ds:
            return None, 0
        search = "|".join(ds)
        carrier_genes = [g for g, pseqs in gene_to_pseqs.items()
                         if any(search in p for p in pseqs) and g in prof_dict]
        if len(carrier_genes) < 5:
            return None, 0
        return np.mean([prof_dict[g] for g in carrier_genes], axis=0), len(carrier_genes)

    sigs_full = {}
    sigs_res = {}
    for seq in testable:
        sf, n = get_signature(seq, profiles)
        sr, _ = get_signature(seq, residual_profiles)
        if sf is not None and sr is not None:
            sigs_full[seq] = sf
            sigs_res[seq] = sr

    print(f"  Primitives with signatures: {len(sigs_full)}")

    pc_cos_full = []
    pc_cos_res = []
    unrelated_cos_full = []
    unrelated_cos_res = []

    tested_pairs = set()
    for parent, children in parent_children.items():
        if parent not in sigs_full:
            continue
        for child in children:
            if child not in sigs_full:
                continue
            pair = (parent, child)
            if pair in tested_pairs:
                continue
            tested_pairs.add(pair)
            pc_cos_full.append(cosine_sim(sigs_full[parent], sigs_full[child]))
            pc_cos_res.append(cosine_sim(sigs_res[parent], sigs_res[child]))

    sig_keys = list(sigs_full.keys())
    np.random.seed(42)
    for _ in range(min(len(pc_cos_full) * 2, 500)):
        i, j = np.random.choice(len(sig_keys), size=2, replace=False)
        p1, p2 = sig_keys[i], sig_keys[j]
        if p2 in parent_children.get(p1, set()) or p1 in parent_children.get(p2, set()):
            continue
        unrelated_cos_full.append(cosine_sim(sigs_full[p1], sigs_full[p2]))
        unrelated_cos_res.append(cosine_sim(sigs_res[p1], sigs_res[p2]))

    print(f"  Parent-child pairs tested: {len(pc_cos_full)}")
    print(f"  Unrelated pairs tested: {len(unrelated_cos_full)}")

    if pc_cos_full and unrelated_cos_full:
        print(f"\n  FULL profiles:")
        print(f"    Parent-child cosine: {np.mean(pc_cos_full):.4f} ± {np.std(pc_cos_full):.4f}")
        print(f"    Unrelated cosine:    {np.mean(unrelated_cos_full):.4f} ± {np.std(unrelated_cos_full):.4f}")
        d_full = (np.mean(pc_cos_full) - np.mean(unrelated_cos_full)) / np.std(unrelated_cos_full) if np.std(unrelated_cos_full) > 0 else 0
        print(f"    Effect size (d):     {d_full:.4f}")

        print(f"\n  RESIDUAL profiles (PC1 removed):")
        print(f"    Parent-child cosine: {np.mean(pc_cos_res):.4f} ± {np.std(pc_cos_res):.4f}")
        print(f"    Unrelated cosine:    {np.mean(unrelated_cos_res):.4f} ± {np.std(unrelated_cos_res):.4f}")
        d_res = (np.mean(pc_cos_res) - np.mean(unrelated_cos_res)) / np.std(unrelated_cos_res) if np.std(unrelated_cos_res) > 0 else 0
        print(f"    Effect size (d):     {d_res:.4f}")

        print(f"\n  Change: d={d_full:.4f} → d={d_res:.4f} (Δ={d_res-d_full:+.4f})")

        if d_res > d_full + 0.1:
            verdict = "NESTING_REVEALS_STRUCTURE"
            print(f"\n  VERDICT: PC1 removal REVEALS nesting carries functional information!")
        elif abs(d_res - d_full) < 0.1:
            verdict = "NESTING_TRULY_DECORRELATED"
            print(f"\n  VERDICT: Nesting is truly decorrelated from function — even on residuals")
        else:
            verdict = "NESTING_MIXED"
            print(f"\n  VERDICT: Mixed — nesting shows weak functional relationship")
    else:
        d_full, d_res = 0, 0
        verdict = "INSUFFICIENT_DATA"

    return {
        "n_parent_child_pairs": len(pc_cos_full),
        "n_unrelated_pairs": len(unrelated_cos_full),
        "full_pc_cosine": round(float(np.mean(pc_cos_full)), 4) if pc_cos_full else 0,
        "full_unrelated_cosine": round(float(np.mean(unrelated_cos_full)), 4) if unrelated_cos_full else 0,
        "res_pc_cosine": round(float(np.mean(pc_cos_res)), 4) if pc_cos_res else 0,
        "res_unrelated_cosine": round(float(np.mean(unrelated_cos_res)), 4) if unrelated_cos_res else 0,
        "d_full": round(d_full, 4),
        "d_residual": round(d_res, 4),
        "verdict": verdict,
    }


def experiment_4_tropical_idempotency(profiles, prim_to_genes):
    print("\n" + "=" * 72)
    print("FOLLOWUP 4: TROPICAL IDEMPOTENCY")
    print("=" * 72)
    print("  Test: Does adding more carriers of an already-represented")
    print("  primitive saturate the aggregated profile?")

    saturation_results = []
    convergence_curves = {}

    for seq, genes in prim_to_genes.items():
        carrier_genes = [g for g in genes if g in profiles]
        if len(carrier_genes) < 40:
            continue

        np.random.seed(hash(seq) % 2**32)
        np.random.shuffle(carrier_genes)

        full_sig = np.mean([profiles[g] for g in carrier_genes], axis=0)
        full_norm = np.linalg.norm(full_sig)
        if full_norm == 0:
            continue

        cosines_at_n = []
        for n in range(5, len(carrier_genes) + 1, max(1, len(carrier_genes) // 20)):
            partial_sig = np.mean([profiles[g] for g in carrier_genes[:n]], axis=0)
            c = cosine_sim(partial_sig, full_sig)
            cosines_at_n.append((n, c))

        if cosines_at_n:
            n_at_99 = None
            n_at_999 = None
            for n, c in cosines_at_n:
                if c >= 0.99 and n_at_99 is None:
                    n_at_99 = n
                if c >= 0.999 and n_at_999 is None:
                    n_at_999 = n

            saturation_results.append({
                "primitive": seq,
                "total_carriers": len(carrier_genes),
                "n_at_99": n_at_99,
                "n_at_999": n_at_999,
                "final_cosine": cosines_at_n[-1][1],
                "cosine_at_half": next((c for n, c in cosines_at_n if n >= len(carrier_genes)//2), None),
            })

            if len(convergence_curves) < 5:
                convergence_curves[seq[:30]] = [(n, round(c, 4)) for n, c in cosines_at_n]

    print(f"\n  Primitives tested: {len(saturation_results)}")

    if saturation_results:
        n_at_99_vals = [r["n_at_99"] for r in saturation_results if r["n_at_99"]]
        n_at_999_vals = [r["n_at_999"] for r in saturation_results if r["n_at_999"]]
        cos_at_half = [r["cosine_at_half"] for r in saturation_results if r["cosine_at_half"]]

        print(f"  Reach 0.99 cosine at median {np.median(n_at_99_vals):.0f} carriers" if n_at_99_vals else "")
        print(f"  Reach 0.999 cosine at median {np.median(n_at_999_vals):.0f} carriers" if n_at_999_vals else "")
        print(f"  Cosine at half-carriers: {np.mean(cos_at_half):.4f} ± {np.std(cos_at_half):.4f}" if cos_at_half else "")
        print(f"  All reach 0.99: {len(n_at_99_vals)}/{len(saturation_results)} ({len(n_at_99_vals)/len(saturation_results):.1%})")

        pcts_needed = [r["n_at_99"] / r["total_carriers"] for r in saturation_results if r["n_at_99"]]
        if pcts_needed:
            print(f"  Median % of carriers needed for 0.99: {np.median(pcts_needed):.1%}")

        incremental_deltas = []
        for seq, genes in prim_to_genes.items():
            carrier_genes = [g for g in genes if g in profiles]
            if len(carrier_genes) < 40:
                continue

            np.random.seed(hash(seq) % 2**32)
            np.random.shuffle(carrier_genes)

            sig_first_half = np.mean([profiles[g] for g in carrier_genes[:len(carrier_genes)//2]], axis=0)
            sig_full = np.mean([profiles[g] for g in carrier_genes], axis=0)
            delta_norm = np.linalg.norm(sig_full - sig_first_half) / np.linalg.norm(sig_full)
            incremental_deltas.append(delta_norm)

        print(f"\n  Incremental change (second half vs full):")
        print(f"    Mean relative norm change: {np.mean(incremental_deltas):.4f}")
        print(f"    Max relative norm change:  {np.max(incremental_deltas):.4f}")

        is_idempotent = np.mean(cos_at_half) > 0.99 if cos_at_half else False
        verdict = "SATURATES" if is_idempotent else "DOES_NOT_SATURATE"
        print(f"\n  VERDICT: Signatures {'DO' if is_idempotent else 'DO NOT'} saturate rapidly")
        print(f"  → {'Consistent' if is_idempotent else 'Inconsistent'} with tropical idempotency")
    else:
        verdict = "INSUFFICIENT_DATA"

    return {
        "n_tested": len(saturation_results),
        "median_n_at_99": round(float(np.median(n_at_99_vals)), 1) if n_at_99_vals else None,
        "median_n_at_999": round(float(np.median(n_at_999_vals)), 1) if n_at_999_vals else None,
        "mean_cosine_at_half": round(float(np.mean(cos_at_half)), 4) if cos_at_half else None,
        "pct_reach_99": round(len(n_at_99_vals)/len(saturation_results), 4) if saturation_results else 0,
        "median_pct_needed_99": round(float(np.median(pcts_needed)), 4) if pcts_needed else None,
        "mean_incremental_change": round(float(np.mean(incremental_deltas)), 4) if incremental_deltas else None,
        "convergence_examples": convergence_curves,
        "verdict": verdict,
    }


def main():
    print("=" * 72)
    print("FOLLOW-UP EXPERIMENTS: GAUGED TROPICAL MODULE HYPOTHESIS")
    print("=" * 72)

    t0 = time.time()

    print("\nLoading state...")
    state = load_state()
    print(f"  Graph: {state['n_tokens']} tokens, {state['n_proteins']} proteins")

    print("Loading disruption profiles...")
    profiles, _ = load_profiles()
    print(f"  {len(profiles)} gene profiles loaded")

    print("Building primitive carrier mappings...")
    prim_to_genes, protein_dept_seqs = build_primitive_carriers(state)
    print(f"  {len(prim_to_genes)} primitives with ≥20 carriers")

    print("Computing PC1 removal...")
    residual_profiles, M_centered, M_residual, gene_names, U, S, Vt = compute_pc1_removal(profiles)
    pc1_var = S[0]**2 / np.sum(S**2)
    print(f"  PC1 explains {pc1_var:.1%} of variance")

    gene_depts = load_gene_depts()

    results = {}

    results["exp1_clusters"] = experiment_1_identify_clusters(
        profiles, residual_profiles, gene_names, gene_depts
    )

    results["exp2_compiler_residuals"] = experiment_2_compiler_on_residuals(
        profiles, residual_profiles, prim_to_genes, gene_depts
    )

    results["exp3_parent_child_residual"] = experiment_3_parent_child_residual(
        state, profiles, residual_profiles, protein_dept_seqs
    )

    results["exp4_tropical_idempotency"] = experiment_4_tropical_idempotency(
        profiles, prim_to_genes
    )

    elapsed = time.time() - t0

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for key, res in results.items():
        verdict = res.get("verdict", "—")
        print(f"  {key}: {verdict}")

    print(f"\nTotal runtime: {elapsed:.0f}s")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
