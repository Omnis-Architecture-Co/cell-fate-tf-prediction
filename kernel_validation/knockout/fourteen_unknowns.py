#!/usr/bin/env python3
"""
14 Unknowns — Comprehensive Exploration Suite
==============================================

Explores all open unknowns about the protein interaction kernel
before deciding on VM optimization strategy.

U01: Does the algebraic signal predict ADRs at all?
U02: Is cascade size just PC1 in disguise?
U03: Residual accuracy gap (62% primitive → 17.5% gene) — why?
U04: What does cascade capture vs disruption profile?
U05: Does drug class matter in algebraic space?
U06: Is 22 departments the right granularity?
U07: Do FDT ratios beat manual priors?
U08: Does entropy predict individual drug prediction accuracy?
U09: What determines boot order if not topology?
U10: Why are Signaling/Kinase out of equilibrium?
U11: 75% bulk — truly featureless or hidden structure?
U12: Cross-species conservation of tropical structure
U13: Power law exponent (-0.55) — theoretical interpretation
U14: CNC Machine — do drug interactions happen on specific
     developmental planes (L1/L2/L3)?

Usage:
    python3 -u validation/knockout/fourteen_unknowns.py
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
from scipy.linalg import svd
from scipy.spatial.distance import cosine as cos_dist

STATE_PATH = "/tmp/module8_full_state.pkl"
VOCAB_PATH = "server/data/human/vocabulary.csv"
PRIMITIVES_PATH = "beta_transfer/genome_primitives.csv"
NESTING_PATH = "beta_transfer/genome_nesting_hierarchy.csv"
GENE_DEPTS_PATH = "server/data/human/gene_departments.csv"
PROFILES_PATH = "validation/knockout/disruption_profiles_full.json"
GROUND_TRUTH_PATH = "validation_145_drugs_GROUND_TRUTH.json"
OUTPUT_PATH = "validation/knockout/fourteen_unknowns_results.json"

VALID_DEPARTMENTS = sorted([
    "Apoptosis", "Cell adhesion", "Cell cycle", "Chromatin", "Cytoskeleton",
    "DNA repair", "GTPase", "Immune", "Ion channel", "Kinase", "Methylation",
    "Nuc acid bind", "Phosphatase", "Protein folding", "Proteolysis",
    "RNA processing", "Signaling", "Structural", "Transcription",
    "Translation", "Transport", "Ubiquitin",
])
D2I = {d: i for i, d in enumerate(VALID_DEPARTMENTS)}
N_DEPTS = len(VALID_DEPARTMENTS)

L1_DEPTS = {"Chromatin", "Cytoskeleton", "DNA repair", "Structural", "Cell cycle"}
L2_DEPTS = {"Transcription", "Nuc acid bind", "Methylation", "RNA processing", "Translation"}
L3_DEPTS = {"Kinase", "Signaling", "Phosphatase", "GTPase", "Immune", "Ion channel",
            "Apoptosis", "Cell adhesion", "Protein folding", "Proteolysis", "Transport", "Ubiquitin"}

DEPT_PHENOTYPE_PRIORS = [
    ('Ion channel', 'peripheral_neuropathy', 0.8),
    ('Ion channel', 'epilepsy_seizure', 0.7),
    ('Ion channel', 'cardiac_arrhythmia', 0.7),
    ('Ion channel', 'ion_channel_dysfunction', 0.9),
    ('Ion channel', 'hearing_loss', 0.5),
    ('Ion channel', 'renal_disorders', 0.4),
    ('Ion channel', 'vision_disorders', 0.4),
    ('Ion channel', 'respiratory_disorders', 0.3),
    ('Ubiquitin', 'proteasome_dysfunction', 0.8),
    ('Ubiquitin', 'cancer_risk', 0.5),
    ('Ubiquitin', 'immune_dysregulation', 0.4),
    ('Proteolysis', 'proteasome_dysfunction', 0.8),
    ('Proteolysis', 'cancer_risk', 0.5),
    ('Proteolysis', 'immune_dysregulation', 0.4),
    ('Proteolysis', 'bleeding_coagulation', 0.6),
    ('Cytoskeleton', 'myopathy', 0.6),
    ('Cytoskeleton', 'gi_dysmotility', 0.5),
    ('Cytoskeleton', 'peripheral_neuropathy', 0.4),
    ('Cytoskeleton', 'developmental_structural', 0.4),
    ('Cytoskeleton', 'skin_disorders', 0.4),
    ('Cytoskeleton', 'hearing_loss', 0.3),
    ('Chromatin', 'cancer_risk', 0.5),
    ('Chromatin', 'developmental_structural', 0.4),
    ('Chromatin', 'neurodevelopmental', 0.3),
    ('Chromatin', 'sexual_reproductive', 0.3),
    ('Transcription', 'cancer_risk', 0.5),
    ('Transcription', 'developmental_structural', 0.4),
    ('Transcription', 'neurodevelopmental', 0.3),
    ('Immune', 'immune_dysregulation', 0.7),
    ('Immune', 'immune_deficiency', 0.6),
    ('Immune', 'autoinflammatory', 0.5),
    ('Immune', 'immune_cytopenia', 0.5),
    ('Immune', 'skin_disorders', 0.3),
    ('Immune', 'respiratory_disorders', 0.4),
    ('DNA repair', 'genomic_instability', 0.9),
    ('DNA repair', 'cancer_risk', 0.7),
    ('DNA repair', 'hematological_anemia', 0.4),
    ('DNA repair', 'immune_cytopenia', 0.3),
    ('Apoptosis', 'cancer_risk', 0.6),
    ('Apoptosis', 'apoptosis_dysregulation', 0.8),
    ('Apoptosis', 'immune_dysregulation', 0.4),
    ('Apoptosis', 'immune_cytopenia', 0.3),
    ('Kinase', 'cancer_risk', 0.4),
    ('Kinase', 'immune_dysregulation', 0.3),
    ('Kinase', 'vascular_disorders', 0.3),
    ('Kinase', 'metabolic_diabetes', 0.3),
    ('Signaling', 'cancer_risk', 0.4),
    ('Signaling', 'metabolic_diabetes', 0.3),
    ('Signaling', 'developmental_structural', 0.3),
    ('Signaling', 'vascular_disorders', 0.3),
    ('Signaling', 'skin_disorders', 0.3),
    ('Signaling', 'respiratory_disorders', 0.3),
    ('Phosphatase', 'immune_dysregulation', 0.3),
    ('Phosphatase', 'metabolic_diabetes', 0.3),
    ('Phosphatase', 'cancer_risk', 0.3),
    ('Cell adhesion', 'connective_tissue', 0.5),
    ('Cell adhesion', 'developmental_structural', 0.4),
    ('Cell adhesion', 'skin_disorders', 0.5),
    ('Cell adhesion', 'hearing_loss', 0.4),
    ('Cell adhesion', 'vascular_disorders', 0.3),
    ('Cell cycle', 'cancer_risk', 0.6),
    ('Cell cycle', 'developmental_structural', 0.4),
    ('Cell cycle', 'immune_cytopenia', 0.3),
    ('Cell cycle', 'sexual_reproductive', 0.3),
    ('Translation', 'developmental_structural', 0.3),
    ('Translation', 'neurological_general', 0.3),
    ('Translation', 'immune_cytopenia', 0.3),
    ('Protein folding', 'proteasome_dysfunction', 0.5),
    ('Protein folding', 'neurodegeneration', 0.5),
    ('Protein folding', 'connective_tissue', 0.3),
    ('RNA processing', 'neurodevelopmental', 0.3),
    ('RNA processing', 'neurodegeneration', 0.3),
    ('RNA processing', 'immune_cytopenia', 0.3),
    ('Transport', 'renal_disorders', 0.5),
    ('Transport', 'metabolic_storage', 0.4),
    ('Transport', 'hepatotoxicity', 0.3),
    ('Transport', 'metabolic_lipid', 0.3),
    ('Structural', 'connective_tissue', 0.5),
    ('Structural', 'skin_disorders', 0.4),
    ('Structural', 'skeletal_disorders', 0.4),
    ('GTPase', 'cancer_risk', 0.4),
    ('GTPase', 'immune_dysregulation', 0.3),
    ('GTPase', 'neurological_general', 0.3),
    ('Methylation', 'cancer_risk', 0.5),
    ('Methylation', 'neurodevelopmental', 0.4),
    ('Methylation', 'developmental_structural', 0.3),
]


def load_profiles():
    with open(PROFILES_PATH) as f:
        data = json.load(f)
    profiles = {}
    for gene, prof in data["profiles"].items():
        vec = np.array([prof.get(d, 0.0) for d in VALID_DEPARTMENTS])
        profiles[gene] = vec
    return profiles


def load_ground_truth():
    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)
    return gt["drugs"]


def load_gene_depts():
    gene_depts = {}
    with open(GENE_DEPTS_PATH) as f:
        for row in csv.DictReader(f):
            gene_depts[row["gene"]] = row["department"]
    return gene_depts


def profiles_to_matrix(profiles):
    genes = sorted(profiles.keys())
    M = np.array([profiles[g] for g in genes])
    return genes, M


def subtract_pc1(M):
    U, S, Vt = svd(M, full_matrices=False)
    pc1 = Vt[0]
    projections = M @ pc1
    M_residual = M - np.outer(projections, pc1)
    return M_residual, pc1, S


def compute_entropy(vec):
    v = vec / (vec.sum() + 1e-30)
    v = v[v > 0]
    return -np.sum(v * np.log2(v))


def build_dept_to_phenotype_map():
    d2p = defaultdict(dict)
    for dept, pheno, weight in DEPT_PHENOTYPE_PRIORS:
        if dept in D2I:
            d2p[dept][pheno] = weight
    return d2p


def build_phenotype_to_dept_vec():
    pheno_to_dept = defaultdict(lambda: np.zeros(N_DEPTS))
    for dept, pheno, weight in DEPT_PHENOTYPE_PRIORS:
        if dept in D2I:
            pheno_to_dept[pheno][D2I[dept]] = weight
    for pheno in pheno_to_dept:
        n = np.linalg.norm(pheno_to_dept[pheno])
        if n > 0:
            pheno_to_dept[pheno] = pheno_to_dept[pheno] / n
    return dict(pheno_to_dept)


def run_u01(profiles, drugs, pheno_dept_vecs):
    """U01: Does the algebraic signal predict ADRs at all?"""
    print("\n" + "="*70)
    print("U01: Does the algebraic signal predict ADRs at all?")
    print("="*70)

    genes_list, M = profiles_to_matrix(profiles)
    gene_idx = {g: i for i, g in enumerate(genes_list)}
    M_resid, pc1, S = subtract_pc1(M)

    hits_raw = []
    hits_resid = []
    drugs_tested = 0
    recall_at_5_raw = []
    recall_at_5_resid = []
    recall_at_10_raw = []
    recall_at_10_resid = []

    for drug in drugs:
        target = drug["primary_target_gene"]
        if target not in gene_idx:
            continue
        gt_cats = drug["ground_truth"]["known_adverse_effect_categories"]
        if not gt_cats:
            continue

        idx = gene_idx[target]
        raw_profile = M[idx]
        resid_profile = M_resid[idx]
        drugs_tested += 1

        scores_raw = {}
        scores_resid = {}
        for pheno, pvec in pheno_dept_vecs.items():
            pvec_full = pvec * np.linalg.norm(pvec) if np.linalg.norm(pvec) > 0 else pvec
            scores_raw[pheno] = float(np.dot(raw_profile, pvec))
            scores_resid[pheno] = float(np.dot(resid_profile, pvec))

        ranked_raw = sorted(scores_raw.items(), key=lambda x: -x[1])
        ranked_resid = sorted(scores_resid.items(), key=lambda x: -x[1])

        top5_raw = {p for p, _ in ranked_raw[:5]}
        top10_raw = {p for p, _ in ranked_raw[:10]}
        top5_resid = {p for p, _ in ranked_resid[:5]}
        top10_resid = {p for p, _ in ranked_resid[:10]}

        gt_set = set(gt_cats)
        r5_raw = len(gt_set & top5_raw) / len(gt_set)
        r10_raw = len(gt_set & top10_raw) / len(gt_set)
        r5_resid = len(gt_set & top5_resid) / len(gt_set)
        r10_resid = len(gt_set & top10_resid) / len(gt_set)

        recall_at_5_raw.append(r5_raw)
        recall_at_10_raw.append(r10_raw)
        recall_at_5_resid.append(r5_resid)
        recall_at_10_resid.append(r10_resid)

    result = {
        "drugs_tested": drugs_tested,
        "raw_profile_recall_at_5": round(np.mean(recall_at_5_raw), 4) if recall_at_5_raw else 0,
        "raw_profile_recall_at_10": round(np.mean(recall_at_10_raw), 4) if recall_at_10_raw else 0,
        "residual_profile_recall_at_5": round(np.mean(recall_at_5_resid), 4) if recall_at_5_resid else 0,
        "residual_profile_recall_at_10": round(np.mean(recall_at_10_resid), 4) if recall_at_10_resid else 0,
        "improvement_r5": round(np.mean(recall_at_5_resid) - np.mean(recall_at_5_raw), 4) if recall_at_5_raw else 0,
        "improvement_r10": round(np.mean(recall_at_10_resid) - np.mean(recall_at_10_raw), 4) if recall_at_10_raw else 0,
    }

    print(f"  Drugs tested: {drugs_tested}")
    print(f"  Raw profile R@5:  {result['raw_profile_recall_at_5']}")
    print(f"  Raw profile R@10: {result['raw_profile_recall_at_10']}")
    print(f"  Residual R@5:     {result['residual_profile_recall_at_5']}")
    print(f"  Residual R@10:    {result['residual_profile_recall_at_10']}")
    print(f"  Improvement R@10: {result['improvement_r10']}")

    return result


def run_u02(profiles):
    """U02: Is cascade size just PC1 in disguise?"""
    print("\n" + "="*70)
    print("U02: Is cascade size just PC1 in disguise?")
    print("="*70)

    genes_list, M = profiles_to_matrix(profiles)
    _, pc1, S = subtract_pc1(M)

    profile_sums = np.sum(M, axis=1)
    pc1_scores = M @ pc1

    r_sum_pc1, p_sum = stats.spearmanr(profile_sums, pc1_scores)

    try:
        state = load_state_safe()
        ptt = state["ptt"]
        token_counts = []
        sums_matched = []
        for i, g in enumerate(genes_list):
            if g in ptt:
                token_counts.append(len(ptt[g]))
                sums_matched.append(profile_sums[i])
        r_tc_sum, p_tc = stats.spearmanr(token_counts, sums_matched)
    except Exception:
        r_tc_sum = None
        p_tc = None

    n_ppi_neighbors = []
    ppi_sums = []
    try:
        state = load_state_safe()
        gc = state["gene_cache"]
        for i, g in enumerate(genes_list):
            if g in gc and "edges" in gc[g]:
                n_ppi = len(gc[g]["edges"])
                n_ppi_neighbors.append(n_ppi)
                ppi_sums.append(profile_sums[i])
        r_ppi_sum, p_ppi = stats.spearmanr(n_ppi_neighbors, ppi_sums)
    except Exception:
        r_ppi_sum = None
        p_ppi = None

    result = {
        "profile_sum_vs_pc1_score": {"r": round(r_sum_pc1, 4), "p": float(p_sum)},
        "token_count_vs_profile_sum": {"r": round(r_tc_sum, 4) if r_tc_sum else None},
        "ppi_neighbors_vs_profile_sum": {"r": round(r_ppi_sum, 4) if r_ppi_sum else None},
        "interpretation": "If PPI neighbors correlate strongly with profile sum (= PC1), then cascade expansion IS PC1 propagation"
    }

    print(f"  Profile sum vs PC1 score: r={r_sum_pc1:.4f}")
    if r_tc_sum is not None:
        print(f"  Token count vs profile sum: r={r_tc_sum:.4f}")
    if r_ppi_sum is not None:
        print(f"  PPI neighbors vs profile sum: r={r_ppi_sum:.4f}")

    return result


def run_u03(profiles, gene_depts):
    """U03: Residual accuracy gap — 62% primitive → 17.5% gene. Why?"""
    print("\n" + "="*70)
    print("U03: Why does primitive prediction (62%) far exceed gene prediction (17.5%)?")
    print("="*70)

    genes_list, M = profiles_to_matrix(profiles)
    gene_idx = {g: i for i, g in enumerate(genes_list)}
    M_resid, pc1, S = subtract_pc1(M)

    n_depts_per_gene = []
    dept_counts = defaultdict(int)
    for g, dept in gene_depts.items():
        dept_counts[g] = 1

    multi_dept_genes = defaultdict(set)
    with open(GENE_DEPTS_PATH) as f:
        for row in csv.DictReader(f):
            multi_dept_genes[row["gene"]].add(row["department"])

    dept_count_dist = [len(ds) for ds in multi_dept_genes.values()]

    entropies = []
    for i, g in enumerate(genes_list):
        entropies.append(compute_entropy(M[i]))

    max_entropy = np.log2(N_DEPTS)
    entropy_ratios = [e / max_entropy for e in entropies]

    top1_correct = 0
    top3_correct = 0
    total = 0
    entropy_when_correct = []
    entropy_when_wrong = []

    for g, depts in multi_dept_genes.items():
        if g not in gene_idx:
            continue
        idx = gene_idx[g]
        resid = M_resid[idx]
        ranked_depts = [VALID_DEPARTMENTS[j] for j in np.argsort(-resid)]
        total += 1

        if ranked_depts[0] in depts:
            top1_correct += 1
            entropy_when_correct.append(entropies[idx])
        else:
            entropy_when_wrong.append(entropies[idx])
        if any(d in depts for d in ranked_depts[:3]):
            top3_correct += 1

    profile_norms = np.linalg.norm(M_resid, axis=1)
    norm_quartiles = np.percentile(profile_norms, [25, 50, 75])

    low_norm = profile_norms < norm_quartiles[0]
    high_norm = profile_norms > norm_quartiles[2]

    low_norm_correct = 0
    high_norm_correct = 0
    low_norm_total = 0
    high_norm_total = 0

    for g, depts in multi_dept_genes.items():
        if g not in gene_idx:
            continue
        idx = gene_idx[g]
        resid = M_resid[idx]
        ranked_depts = [VALID_DEPARTMENTS[j] for j in np.argsort(-resid)]
        if low_norm[idx]:
            low_norm_total += 1
            if ranked_depts[0] in depts:
                low_norm_correct += 1
        if high_norm[idx]:
            high_norm_total += 1
            if ranked_depts[0] in depts:
                high_norm_correct += 1

    result = {
        "gene_level_top1": round(top1_correct / max(total, 1), 3),
        "gene_level_top3": round(top3_correct / max(total, 1), 3),
        "total_genes_tested": total,
        "mean_entropy_when_correct": round(np.mean(entropy_when_correct), 3) if entropy_when_correct else None,
        "mean_entropy_when_wrong": round(np.mean(entropy_when_wrong), 3) if entropy_when_wrong else None,
        "low_residual_norm_accuracy": round(low_norm_correct / max(low_norm_total, 1), 3),
        "high_residual_norm_accuracy": round(high_norm_correct / max(high_norm_total, 1), 3),
        "median_depts_per_gene": float(np.median(dept_count_dist)),
        "mean_entropy_ratio": round(np.mean(entropy_ratios), 3),
        "interpretation": "If genes are mostly high-entropy (near uniform), they have no strong directional signal. The gap exists because primitives AGGREGATE multiple genes (tropical max reveals direction), while individual genes are near-thermodynamic."
    }

    print(f"  Gene top-1 accuracy: {result['gene_level_top1']}")
    print(f"  Gene top-3 accuracy: {result['gene_level_top3']}")
    print(f"  Entropy when correct: {result['mean_entropy_when_correct']}")
    print(f"  Entropy when wrong: {result['mean_entropy_when_wrong']}")
    print(f"  Low-norm accuracy: {result['low_residual_norm_accuracy']}")
    print(f"  High-norm accuracy: {result['high_residual_norm_accuracy']}")

    return result


def run_u04(profiles):
    """U04: What does cascade capture vs disruption profile?"""
    print("\n" + "="*70)
    print("U04: Cascade (PPI hops) vs Disruption Profile — same or different?")
    print("="*70)

    try:
        state = load_state_safe()
    except Exception as e:
        print(f"  Cannot load state: {e}")
        return {"error": "state not available"}

    gc = state["gene_cache"]
    ptt = state["ptt"]
    genes_list, M = profiles_to_matrix(profiles)
    gene_idx = {g: i for i, g in enumerate(genes_list)}
    M_resid, pc1, S = subtract_pc1(M)

    sample_genes = [g for g in ["HMGCR", "EGFR", "DRD2", "BRCA1", "TP53",
                                 "BRAF", "JAK2", "ABL1", "ESR1", "PTGS2",
                                 "CACNA1C", "SCN5A", "KCNH2", "CFTR", "HTT"]
                    if g in gene_idx and g in gc]

    cosines_raw = []
    cosines_resid = []
    cascade_sizes = []

    for g in sample_genes:
        if "edges" not in gc[g]:
            continue
        neighbors = set()
        for e in gc[g]["edges"]:
            partner = e.get("partner") or e.get("target")
            if partner:
                neighbors.add(partner)

        if not neighbors:
            continue

        cascade_profile = np.zeros(N_DEPTS)
        n_found = 0
        for nb in neighbors:
            if nb in gene_idx:
                cascade_profile += M[gene_idx[nb]]
                n_found += 1
        if n_found > 0:
            cascade_profile /= n_found

        target_profile_raw = M[gene_idx[g]]
        target_profile_resid = M_resid[gene_idx[g]]

        if np.linalg.norm(cascade_profile) > 0 and np.linalg.norm(target_profile_raw) > 0:
            cos_raw = 1 - cos_dist(target_profile_raw, cascade_profile)
            cosines_raw.append(cos_raw)
        if np.linalg.norm(cascade_profile) > 0 and np.linalg.norm(target_profile_resid) > 0:
            cascade_resid = cascade_profile - pc1 * np.dot(cascade_profile, pc1)
            if np.linalg.norm(cascade_resid) > 0:
                cos_r = 1 - cos_dist(target_profile_resid, cascade_resid)
                cosines_resid.append(cos_r)

        cascade_sizes.append(len(neighbors))

    profile_sums = np.sum(M, axis=1)
    cascade_size_all = []
    profile_sum_all = []
    for g in genes_list:
        if g in gc and "edges" in gc[g]:
            cascade_size_all.append(len(gc[g]["edges"]))
            profile_sum_all.append(profile_sums[gene_idx[g]])

    r_cascade_profile, p_cp = stats.spearmanr(cascade_size_all, profile_sum_all) if len(cascade_size_all) > 10 else (0, 1)

    result = {
        "sample_genes_tested": len(sample_genes),
        "mean_cosine_raw_target_vs_cascade": round(np.mean(cosines_raw), 4) if cosines_raw else None,
        "mean_cosine_resid_target_vs_cascade": round(np.mean(cosines_resid), 4) if cosines_resid else None,
        "cascade_size_vs_profile_sum_r": round(r_cascade_profile, 4),
        "cascade_size_vs_profile_sum_p": float(p_cp),
        "mean_cascade_size": round(np.mean(cascade_sizes), 1) if cascade_sizes else None,
        "interpretation": "High raw cosine = cascade and profile see the same thing (PC1). Low residual cosine = they capture DIFFERENT directional information after magnitude removal."
    }

    print(f"  Raw cosine (target vs cascade avg): {result['mean_cosine_raw_target_vs_cascade']}")
    print(f"  Residual cosine (target vs cascade avg): {result['mean_cosine_resid_target_vs_cascade']}")
    print(f"  Cascade size vs profile sum: r={result['cascade_size_vs_profile_sum_r']}")

    return result


def run_u05(profiles, drugs, pheno_dept_vecs):
    """U05: Does drug class matter in algebraic space?"""
    print("\n" + "="*70)
    print("U05: Does drug class / therapeutic area matter?")
    print("="*70)

    genes_list, M = profiles_to_matrix(profiles)
    gene_idx = {g: i for i, g in enumerate(genes_list)}
    M_resid, pc1, S = subtract_pc1(M)

    by_area = defaultdict(list)
    by_mechanism = defaultdict(list)

    for drug in drugs:
        target = drug["primary_target_gene"]
        if target not in gene_idx:
            continue
        gt_cats = drug["ground_truth"]["known_adverse_effect_categories"]
        if not gt_cats:
            continue

        idx = gene_idx[target]
        resid = M_resid[idx]

        scores = {}
        for pheno, pvec in pheno_dept_vecs.items():
            scores[pheno] = float(np.dot(resid, pvec))
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        top10 = {p for p, _ in ranked[:10]}
        gt_set = set(gt_cats)
        r10 = len(gt_set & top10) / len(gt_set)

        area = drug.get("therapeutic_area", "Unknown")
        mech = drug.get("mechanism_class", "Unknown")
        by_area[area].append(r10)
        by_mechanism[mech].append(r10)

    area_results = {}
    for area, recalls in sorted(by_area.items()):
        area_results[area] = {
            "n_drugs": len(recalls),
            "mean_recall_at_10": round(np.mean(recalls), 3),
            "std": round(np.std(recalls), 3),
        }
        print(f"  {area}: n={len(recalls)}, R@10={np.mean(recalls):.3f} ± {np.std(recalls):.3f}")

    mech_results = {}
    for mech, recalls in sorted(by_mechanism.items()):
        mech_results[mech] = {
            "n_drugs": len(recalls),
            "mean_recall_at_10": round(np.mean(recalls), 3),
        }

    result = {
        "by_therapeutic_area": area_results,
        "by_mechanism_class": mech_results,
        "interpretation": "Large variance across areas = algebraic signal is class-dependent. Uniform = universal."
    }

    return result


def run_u06(profiles, gene_depts):
    """U06: Is 22 departments the right granularity?"""
    print("\n" + "="*70)
    print("U06: Is 22 the right number of departments?")
    print("="*70)

    genes_list, M = profiles_to_matrix(profiles)
    M_resid, pc1, S = subtract_pc1(M)

    svs = S[1:]
    n = len(svs)
    effective_dims = []
    for thresh in [0.50, 0.80, 0.90, 0.95, 0.99]:
        cumvar = np.cumsum(svs**2) / np.sum(svs**2)
        n_dims = np.searchsorted(cumvar, thresh) + 1
        effective_dims.append((thresh, int(n_dims)))

    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(M_resid[:5000].T, method='ward')

    dept_merge_order = []
    for i in range(len(Z)):
        c1, c2 = int(Z[i, 0]), int(Z[i, 1])
        dist = Z[i, 2]
        dept_merge_order.append({
            "step": i,
            "merge_distance": round(float(dist), 4),
            "clusters_remaining": N_DEPTS - i - 1
        })

    inter_dept_cors = []
    for i in range(N_DEPTS):
        for j in range(i+1, N_DEPTS):
            r, _ = stats.pearsonr(M_resid[:, i], M_resid[:, j])
            inter_dept_cors.append((VALID_DEPARTMENTS[i], VALID_DEPARTMENTS[j], round(r, 3)))
    inter_dept_cors.sort(key=lambda x: -abs(x[2]))

    result = {
        "effective_dimensions_residual": effective_dims,
        "top_correlated_dept_pairs": inter_dept_cors[:10],
        "most_anticorrelated_dept_pairs": inter_dept_cors[-5:],
        "dept_merge_order_first_5": dept_merge_order[:5],
        "interpretation": "If effective dims << 22, departments are redundant. Highly correlated pairs could be merged. Anti-correlated pairs are the most informative axes."
    }

    print(f"  Effective dimensions: {effective_dims}")
    print(f"  Top correlated pairs: {inter_dept_cors[:3]}")
    print(f"  Most anti-correlated: {inter_dept_cors[-3:]}")

    return result


def run_u07(profiles, drugs, pheno_dept_vecs):
    """U07: Do FDT ratios beat manual priors?"""
    print("\n" + "="*70)
    print("U07: FDT-derived weights vs manual department priors")
    print("="*70)

    genes_list, M = profiles_to_matrix(profiles)
    gene_idx = {g: i for i, g in enumerate(genes_list)}
    M_resid, pc1, S = subtract_pc1(M)

    dept_var = np.var(M_resid, axis=0)
    dept_mean_abs = np.mean(np.abs(M_resid), axis=0)
    fdt_ratio = dept_var / (dept_mean_abs + 1e-30)

    fdt_weights = fdt_ratio / fdt_ratio.max()

    manual_weight_by_dept = defaultdict(float)
    for dept, pheno, weight in DEPT_PHENOTYPE_PRIORS:
        if dept in D2I:
            manual_weight_by_dept[dept] = max(manual_weight_by_dept[dept], weight)
    manual_vec = np.array([manual_weight_by_dept.get(d, 0.1) for d in VALID_DEPARTMENTS])
    manual_vec = manual_vec / manual_vec.max()

    r_fdt_manual, p_fm = stats.spearmanr(fdt_weights, manual_vec)

    recall_manual = []
    recall_fdt = []

    for drug in drugs:
        target = drug["primary_target_gene"]
        if target not in gene_idx:
            continue
        gt_cats = drug["ground_truth"]["known_adverse_effect_categories"]
        if not gt_cats:
            continue

        idx = gene_idx[target]
        resid = M_resid[idx]

        scores_manual = {}
        scores_fdt = {}
        for pheno, pvec in pheno_dept_vecs.items():
            scores_manual[pheno] = float(np.dot(resid * manual_vec, pvec))
            scores_fdt[pheno] = float(np.dot(resid * fdt_weights, pvec))

        ranked_m = sorted(scores_manual.items(), key=lambda x: -x[1])
        ranked_f = sorted(scores_fdt.items(), key=lambda x: -x[1])

        gt_set = set(gt_cats)
        top10_m = {p for p, _ in ranked_m[:10]}
        top10_f = {p for p, _ in ranked_f[:10]}

        recall_manual.append(len(gt_set & top10_m) / len(gt_set))
        recall_fdt.append(len(gt_set & top10_f) / len(gt_set))

    fdt_dept_ranking = sorted(zip(VALID_DEPARTMENTS, fdt_weights), key=lambda x: -x[1])

    result = {
        "fdt_vs_manual_correlation": round(r_fdt_manual, 3),
        "manual_weighted_R10": round(np.mean(recall_manual), 4) if recall_manual else None,
        "fdt_weighted_R10": round(np.mean(recall_fdt), 4) if recall_fdt else None,
        "fdt_dept_ranking": [(d, round(float(w), 3)) for d, w in fdt_dept_ranking[:5]],
        "interpretation": "If FDT R@10 > manual R@10, physics-derived weights are better. High correlation with manual = manual tuning already found the physics."
    }

    print(f"  FDT vs manual weight correlation: r={r_fdt_manual:.3f}")
    print(f"  Manual-weighted R@10: {result['manual_weighted_R10']}")
    print(f"  FDT-weighted R@10: {result['fdt_weighted_R10']}")
    print(f"  Top FDT depts: {fdt_dept_ranking[:5]}")

    return result


def run_u08(profiles, drugs, pheno_dept_vecs):
    """U08: Does entropy predict individual drug prediction accuracy?"""
    print("\n" + "="*70)
    print("U08: Does target gene entropy predict prediction accuracy?")
    print("="*70)

    genes_list, M = profiles_to_matrix(profiles)
    gene_idx = {g: i for i, g in enumerate(genes_list)}
    M_resid, pc1, S = subtract_pc1(M)

    entropies = []
    recalls = []
    gene_names = []

    for drug in drugs:
        target = drug["primary_target_gene"]
        if target not in gene_idx:
            continue
        gt_cats = drug["ground_truth"]["known_adverse_effect_categories"]
        if not gt_cats:
            continue

        idx = gene_idx[target]
        ent = compute_entropy(M[idx])

        scores = {}
        for pheno, pvec in pheno_dept_vecs.items():
            scores[pheno] = float(np.dot(M_resid[idx], pvec))
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        top10 = {p for p, _ in ranked[:10]}
        gt_set = set(gt_cats)
        r10 = len(gt_set & top10) / len(gt_set)

        entropies.append(ent)
        recalls.append(r10)
        gene_names.append(target)

    r_ent_recall, p_er = stats.spearmanr(entropies, recalls) if len(entropies) > 5 else (0, 1)

    max_ent = np.log2(N_DEPTS)
    low_ent = [r for e, r in zip(entropies, recalls) if e < 0.9 * max_ent]
    high_ent = [r for e, r in zip(entropies, recalls) if e >= 0.9 * max_ent]

    result = {
        "entropy_vs_recall_r": round(r_ent_recall, 3),
        "entropy_vs_recall_p": float(p_er),
        "low_entropy_mean_recall": round(np.mean(low_ent), 3) if low_ent else None,
        "high_entropy_mean_recall": round(np.mean(high_ent), 3) if high_ent else None,
        "n_low_entropy": len(low_ent),
        "n_high_entropy": len(high_ent),
        "interpretation": "Negative r = low-entropy (specific) targets predict better. Positive r = high-entropy targets predict better (unlikely)."
    }

    print(f"  Entropy vs recall: r={r_ent_recall:.3f}, p={p_er:.4f}")
    print(f"  Low-entropy drugs: n={len(low_ent)}, mean R@10={np.mean(low_ent):.3f}" if low_ent else "  No low-entropy drugs")
    print(f"  High-entropy drugs: n={len(high_ent)}, mean R@10={np.mean(high_ent):.3f}" if high_ent else "  No high-entropy drugs")

    return result


def run_u09():
    """U09: What determines boot order if not topology?"""
    print("\n" + "="*70)
    print("U09: What determines the ZGA boot order?")
    print("="*70)

    boot_phases = [
        {"phase": "Minor ZGA", "genes": ["DUX4", "ZSCAN4"], "order": 1},
        {"phase": "Major ZGA", "genes": ["SP1", "NELFA", "NELFB", "DPPA2", "DPPA4"], "order": 2},
        {"phase": "First fate", "genes": ["TEAD4", "CDX2", "YAP1", "GATA3"], "order": 3},
        {"phase": "ICM core", "genes": ["POU5F1", "NANOG", "SOX2"], "order": 4},
        {"phase": "ICM split", "genes": ["GATA6", "GATA4"], "order": 5},
        {"phase": "Germ layers", "genes": ["TBXT", "FOXA2", "SOX17", "PAX6", "SOX1", "FOXF1", "FOXC2", "HAND1", "TBX5", "NKX2-5"], "order": 6},
    ]

    try:
        state = load_state_safe()
        ptt = state["ptt"]
        gc = state["gene_cache"]
    except Exception:
        return {"error": "state not available"}

    vocab_dept = {}
    with open(VOCAB_PATH) as f:
        for row in csv.DictReader(f):
            vocab_dept[row["word_hex"].replace("0x", "").upper()] = row["primary_function"]

    gene_depts_from_tokens = {}
    gene_token_counts = {}
    for phase in boot_phases:
        for g in phase["genes"]:
            if g in ptt:
                tokens = ptt[g]
                gene_token_counts[g] = len(tokens)
                dept_counts = Counter()
                for t_hex in tokens:
                    t_upper = t_hex.replace("0x", "").upper()
                    dept = vocab_dept.get(t_upper, "Unknown")
                    dept_counts[dept] += 1
                gene_depts_from_tokens[g] = dict(dept_counts)
            else:
                gene_token_counts[g] = 0
                gene_depts_from_tokens[g] = {}

    boot_orders = []
    token_counts = []
    for phase in boot_phases:
        for g in phase["genes"]:
            boot_orders.append(phase["order"])
            token_counts.append(gene_token_counts.get(g, 0))

    r_boot_tc, p_bt = stats.spearmanr(boot_orders, token_counts) if len(boot_orders) > 5 else (0, 1)

    phase_dept_profiles = {}
    for phase in boot_phases:
        dept_sum = Counter()
        n = 0
        for g in phase["genes"]:
            if g in gene_depts_from_tokens:
                for d, c in gene_depts_from_tokens[g].items():
                    if d in D2I:
                        dept_sum[d] += c
                n += 1
        total = sum(dept_sum.values()) or 1
        phase_dept_profiles[phase["phase"]] = {d: round(c/total, 3) for d, c in dept_sum.most_common(5)}

    result = {
        "boot_order_vs_token_count_r": round(r_boot_tc, 3),
        "interpretation_topology": "r=0.09 with centrality (from rabbit hole 3). Biology, not topology.",
        "boot_order_vs_token_count": round(r_boot_tc, 3),
        "phase_dept_profiles": phase_dept_profiles,
        "pattern": "Each phase activates DIFFERENT departments. The boot order follows a functional initialization sequence, not a connectivity-based one.",
    }

    print(f"  Boot order vs token count: r={r_boot_tc:.3f}")
    for phase, profile in phase_dept_profiles.items():
        print(f"  {phase}: {profile}")

    return result


def run_u10(profiles):
    """U10: Why are Signaling/Kinase out of equilibrium?"""
    print("\n" + "="*70)
    print("U10: Why are Signaling and Kinase out of FDT equilibrium?")
    print("="*70)

    genes_list, M = profiles_to_matrix(profiles)
    M_resid, pc1, S = subtract_pc1(M)

    dept_var = np.var(M_resid, axis=0)
    dept_mean_abs = np.mean(np.abs(M_resid), axis=0)
    fdt_ratio = dept_var / (dept_mean_abs + 1e-30)

    dept_kurtosis = [float(stats.kurtosis(M_resid[:, i])) for i in range(N_DEPTS)]

    dept_skewness = [float(stats.skew(M_resid[:, i])) for i in range(N_DEPTS)]

    dept_tail_fraction = []
    for i in range(N_DEPTS):
        col = M_resid[:, i]
        mu, sigma = np.mean(col), np.std(col)
        tail = np.sum(np.abs(col - mu) > 2 * sigma) / len(col)
        dept_tail_fraction.append(round(float(tail), 4))

    dept_stats = []
    for i, d in enumerate(VALID_DEPARTMENTS):
        layer = "L1" if d in L1_DEPTS else ("L2" if d in L2_DEPTS else "L3")
        dept_stats.append({
            "department": d,
            "layer": layer,
            "fdt_ratio": round(float(fdt_ratio[i]), 3),
            "kurtosis": round(dept_kurtosis[i], 3),
            "skewness": round(dept_skewness[i], 3),
            "tail_fraction": dept_tail_fraction[i],
        })
    dept_stats.sort(key=lambda x: -x["fdt_ratio"])

    l1_fdt = [ds["fdt_ratio"] for ds in dept_stats if ds["layer"] == "L1"]
    l2_fdt = [ds["fdt_ratio"] for ds in dept_stats if ds["layer"] == "L2"]
    l3_fdt = [ds["fdt_ratio"] for ds in dept_stats if ds["layer"] == "L3"]

    result = {
        "dept_stats_ranked_by_fdt": dept_stats[:22],
        "mean_fdt_L1": round(np.mean(l1_fdt), 3),
        "mean_fdt_L2": round(np.mean(l2_fdt), 3),
        "mean_fdt_L3": round(np.mean(l3_fdt), 3),
        "interpretation": "High kurtosis + high FDT = heavy tails = signal amplification. Departments that act as switches/amplifiers SHOULD be out of equilibrium."
    }

    print(f"  Mean FDT by layer: L1={np.mean(l1_fdt):.3f}, L2={np.mean(l2_fdt):.3f}, L3={np.mean(l3_fdt):.3f}")
    for ds in dept_stats[:5]:
        print(f"  {ds['department']} ({ds['layer']}): FDT={ds['fdt_ratio']:.3f}, kurt={ds['kurtosis']:.3f}, tail={ds['tail_fraction']}")

    return result


def run_u11(profiles, gene_depts):
    """U11: 75% bulk — truly featureless or hidden structure?"""
    print("\n" + "="*70)
    print("U11: Is the 75% bulk truly featureless?")
    print("="*70)

    genes_list, M = profiles_to_matrix(profiles)
    gene_idx = {g: i for i, g in enumerate(genes_list)}
    M_resid, pc1, S = subtract_pc1(M)

    from sklearn.cluster import KMeans

    km5 = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels5 = km5.fit_predict(M_resid)

    cluster_sizes = Counter(labels5)
    bulk_label = cluster_sizes.most_common(1)[0][0]
    bulk_mask = labels5 == bulk_label
    bulk_genes = [g for g, m in zip(genes_list, bulk_mask) if m]
    M_bulk = M_resid[bulk_mask]

    print(f"  Bulk cluster: {len(bulk_genes)} genes ({len(bulk_genes)/len(genes_list)*100:.1f}%)")

    U_bulk, S_bulk, Vt_bulk = svd(M_bulk, full_matrices=False)
    bulk_svs = S_bulk / S_bulk[0]

    n_eff_bulk = float(np.sum(S_bulk**2)**2 / np.sum(S_bulk**4))

    try:
        from sklearn.metrics import silhouette_score
        sub_sils = {}
        for k in [2, 3, 5, 8]:
            km_sub = KMeans(n_clusters=k, random_state=42, n_init=10)
            sub_labels = km_sub.fit_predict(M_bulk)
            sil = silhouette_score(M_bulk, sub_labels, sample_size=min(5000, len(M_bulk)))
            sub_sils[k] = round(float(sil), 3)
        print(f"  Sub-clustering silhouettes: {sub_sils}")
    except Exception:
        sub_sils = {}

    bulk_dept_entropies = []
    for i in range(len(M_bulk)):
        bulk_dept_entropies.append(compute_entropy(np.abs(M_bulk[i])))
    max_ent = np.log2(N_DEPTS)

    prim_data = []
    try:
        with open(PRIMITIVES_PATH) as f:
            reader = csv.DictReader(f)
            for row in reader:
                func_seq = row.get("function_sequence", "")
                genes_in_prim = [g.strip() for g in func_seq.split(",") if g.strip()]
                prim_data.append(genes_in_prim)
    except Exception:
        pass

    bulk_set = set(bulk_genes)
    prims_mostly_bulk = 0
    prims_total = 0
    for pg in prim_data:
        if len(pg) < 3:
            continue
        prims_total += 1
        frac_bulk = sum(1 for g in pg if g in bulk_set) / len(pg)
        if frac_bulk > 0.5:
            prims_mostly_bulk += 1

    result = {
        "bulk_size": len(bulk_genes),
        "bulk_pct": round(len(bulk_genes) / len(genes_list) * 100, 1),
        "bulk_effective_dimensions": round(n_eff_bulk, 1),
        "bulk_top5_sv_ratios": [round(float(s), 3) for s in bulk_svs[:5]],
        "sub_clustering_silhouettes": sub_sils,
        "bulk_mean_entropy_ratio": round(np.mean(bulk_dept_entropies) / max_ent, 3),
        "primitives_mostly_bulk": prims_mostly_bulk,
        "primitives_total": prims_total,
        "interpretation": "Low silhouettes + high entropy ratio = genuinely diffuse. High effective dims = uses many axes but without clustering."
    }

    print(f"  Effective dimensions in bulk: {n_eff_bulk:.1f}")
    print(f"  Bulk entropy ratio: {result['bulk_mean_entropy_ratio']}")
    print(f"  Primitives mostly-bulk: {prims_mostly_bulk}/{prims_total}")

    return result


def run_u12():
    """U12: Cross-species conservation of tropical structure"""
    print("\n" + "="*70)
    print("U12: Cross-species tropical structure conservation")
    print("="*70)

    xsp_path = "validation/VAL-XSP-001_cross_species_tau.json"
    try:
        with open(xsp_path) as f:
            xsp = json.load(f)
    except Exception:
        return {"error": "cross-species data not found", "note": "Would need to recompute disruption profiles for non-human species to test tropical structure."}

    result = {
        "existing_cross_species_tau": xsp.get("overall", {}),
        "note": "Existing tau tests validate FUNCTIONAL VOCABULARY conservation across species. Tropical structure (saturation, PC1 dominance, thermodynamics) has NOT been tested in non-human genomes.",
        "what_we_know": "Mean Kendall tau=0.36 across 15 species pairs. Strongest: mouse-zebrafish 0.79, fly-yeast 0.74.",
        "what_we_dont_know": [
            "Does PC1 dominate in mouse/fly/yeast/zebrafish?",
            "Is the power law exponent the same across species?",
            "Does tropical saturation require the same ~4 genes in other genomes?",
            "Is the 92% entropy ratio universal or human-specific?",
        ],
        "required_for_test": "Need to compute disruption profiles for at least mouse and zebrafish (both have primitives data).",
    }

    print(f"  Cross-species tau: {xsp.get('overall', {})}")
    print(f"  NOTE: Tropical structure NOT tested cross-species")

    return result


def run_u13(profiles):
    """U13: Power law exponent (-0.55) — theoretical interpretation"""
    print("\n" + "="*70)
    print("U13: What does the power law exponent -0.55 mean?")
    print("="*70)

    genes_list, M = profiles_to_matrix(profiles)
    _, pc1, S = subtract_pc1(M)

    svs = S[1:]
    log_k = np.log(np.arange(1, len(svs)+1))
    log_sv = np.log(svs)
    slope, intercept, r, p, se = stats.linregress(log_k, log_sv)

    known_exponents = {
        "-1.0": "1/f noise (pink noise) — scale-free temporal correlations",
        "-0.5": "Random walk / Brownian motion — uncorrelated steps accumulate",
        "-0.55": "OBSERVED — between Brownian and 1/f",
        "-0.33": "Kolmogorov turbulence (energy cascade)",
        "-0.25": "Mean-field percolation at criticality",
        "-1.5": "Zipf's law / scale-free networks",
    }

    residuals = log_sv - (slope * log_k + intercept)
    residual_std = np.std(residuals)

    sub1 = slice(0, len(svs)//2)
    sub2 = slice(len(svs)//2, len(svs))
    slope1, _, r1, _, _ = stats.linregress(log_k[sub1], log_sv[sub1])
    slope2, _, r2, _, _ = stats.linregress(log_k[sub2], log_sv[sub2])

    result = {
        "observed_exponent": round(float(-slope), 3),
        "fit_r": round(float(r), 4),
        "fit_r_squared": round(float(r**2), 4),
        "residual_std": round(float(residual_std), 4),
        "first_half_exponent": round(float(-slope1), 3),
        "second_half_exponent": round(float(-slope2), 3),
        "known_exponents": known_exponents,
        "interpretation": "Exponent -0.55 sits between Brownian (-0.5) and 1/f (-1.0). This suggests weak long-range correlations — each functional dimension is not independent but has mild coupling to neighbors. Consistent with a system near a critical point where correlations are power-law distributed.",
        "is_single_power_law": abs(slope1 - slope2) < 0.2,
    }

    print(f"  Observed exponent: {-slope:.3f}")
    print(f"  Fit quality: r={r:.4f}, r²={r**2:.4f}")
    print(f"  First half: {-slope1:.3f}, Second half: {-slope2:.3f}")
    print(f"  Single power law: {result['is_single_power_law']}")

    return result


def run_u14(profiles, drugs, pheno_dept_vecs, gene_depts):
    """U14: CNC Machine — do drugs act on specific developmental planes?"""
    print("\n" + "="*70)
    print("U14: CNC MACHINE — Do drugs act on specific L1/L2/L3 planes?")
    print("="*70)

    genes_list, M = profiles_to_matrix(profiles)
    gene_idx = {g: i for i, g in enumerate(genes_list)}
    M_resid, pc1, S = subtract_pc1(M)

    l1_idx = sorted([D2I[d] for d in L1_DEPTS if d in D2I])
    l2_idx = sorted([D2I[d] for d in L2_DEPTS if d in D2I])
    l3_idx = sorted([D2I[d] for d in L3_DEPTS if d in D2I])

    drug_layer_data = []

    for drug in drugs:
        target = drug["primary_target_gene"]
        if target not in gene_idx:
            continue
        gt_cats = drug["ground_truth"]["known_adverse_effect_categories"]
        if not gt_cats:
            continue

        idx = gene_idx[target]
        full_resid = M_resid[idx]

        l1_energy = float(np.sum(full_resid[l1_idx]**2))
        l2_energy = float(np.sum(full_resid[l2_idx]**2))
        l3_energy = float(np.sum(full_resid[l3_idx]**2))
        total_energy = l1_energy + l2_energy + l3_energy + 1e-30

        l1_frac = l1_energy / total_energy
        l2_frac = l2_energy / total_energy
        l3_frac = l3_energy / total_energy

        dominant_layer = "L1" if l1_frac > l2_frac and l1_frac > l3_frac else \
                         "L2" if l2_frac > l3_frac else "L3"

        l1_profile = np.zeros(N_DEPTS)
        l1_profile[l1_idx] = full_resid[l1_idx]
        l2_profile = np.zeros(N_DEPTS)
        l2_profile[l2_idx] = full_resid[l2_idx]
        l3_profile = np.zeros(N_DEPTS)
        l3_profile[l3_idx] = full_resid[l3_idx]

        scores_full = {}
        scores_l1 = {}
        scores_l2 = {}
        scores_l3 = {}
        for pheno, pvec in pheno_dept_vecs.items():
            scores_full[pheno] = float(np.dot(full_resid, pvec))
            scores_l1[pheno] = float(np.dot(l1_profile, pvec))
            scores_l2[pheno] = float(np.dot(l2_profile, pvec))
            scores_l3[pheno] = float(np.dot(l3_profile, pvec))

        gt_set = set(gt_cats)

        def recall_at_k(scores, k=10):
            ranked = sorted(scores.items(), key=lambda x: -x[1])
            topk = {p for p, _ in ranked[:k]}
            return len(gt_set & topk) / len(gt_set)

        drug_layer_data.append({
            "drug": drug["drug_name"],
            "target": target,
            "area": drug.get("therapeutic_area", "Unknown"),
            "l1_frac": round(l1_frac, 3),
            "l2_frac": round(l2_frac, 3),
            "l3_frac": round(l3_frac, 3),
            "dominant_layer": dominant_layer,
            "r10_full": recall_at_k(scores_full),
            "r10_l1_only": recall_at_k(scores_l1),
            "r10_l2_only": recall_at_k(scores_l2),
            "r10_l3_only": recall_at_k(scores_l3),
        })

    by_dom = defaultdict(list)
    for d in drug_layer_data:
        by_dom[d["dominant_layer"]].append(d)

    layer_summary = {}
    for layer, drugs_in_layer in sorted(by_dom.items()):
        r10_full = np.mean([d["r10_full"] for d in drugs_in_layer])
        r10_own = np.mean([d[f"r10_{layer.lower()}_only"] for d in drugs_in_layer])
        r10_others = []
        for other_l in ["l1", "l2", "l3"]:
            if other_l != layer.lower():
                r10_others.extend([d[f"r10_{other_l}_only"] for d in drugs_in_layer])
        r10_other_mean = np.mean(r10_others) if r10_others else 0

        layer_summary[layer] = {
            "n_drugs": len(drugs_in_layer),
            "mean_r10_full": round(r10_full, 3),
            "mean_r10_own_layer_only": round(r10_own, 3),
            "mean_r10_other_layers": round(r10_other_mean, 3),
            "layer_specificity": round(r10_own - r10_other_mean, 3),
        }
        print(f"  {layer}: n={len(drugs_in_layer)}, R@10_full={r10_full:.3f}, R@10_own={r10_own:.3f}, R@10_other={r10_other_mean:.3f}")

    l1_fracs = [d["l1_frac"] for d in drug_layer_data]
    l2_fracs = [d["l2_frac"] for d in drug_layer_data]
    l3_fracs = [d["l3_frac"] for d in drug_layer_data]
    r10s = [d["r10_full"] for d in drug_layer_data]

    r_l1_r10, _ = stats.spearmanr(l1_fracs, r10s)
    r_l2_r10, _ = stats.spearmanr(l2_fracs, r10s)
    r_l3_r10, _ = stats.spearmanr(l3_fracs, r10s)

    coupling_l1_l2 = stats.spearmanr(l1_fracs, l2_fracs)[0]
    coupling_l1_l3 = stats.spearmanr(l1_fracs, l3_fracs)[0]
    coupling_l2_l3 = stats.spearmanr(l2_fracs, l3_fracs)[0]

    by_area_layer = defaultdict(lambda: defaultdict(list))
    for d in drug_layer_data:
        by_area_layer[d["area"]]["l1"].append(d["l1_frac"])
        by_area_layer[d["area"]]["l2"].append(d["l2_frac"])
        by_area_layer[d["area"]]["l3"].append(d["l3_frac"])

    area_layer_profiles = {}
    for area, layers in sorted(by_area_layer.items()):
        if len(layers["l1"]) >= 3:
            area_layer_profiles[area] = {
                "n": len(layers["l1"]),
                "mean_l1": round(np.mean(layers["l1"]), 3),
                "mean_l2": round(np.mean(layers["l2"]), 3),
                "mean_l3": round(np.mean(layers["l3"]), 3),
            }

    result = {
        "layer_summary": layer_summary,
        "layer_fraction_vs_recall": {
            "L1_vs_R10": round(r_l1_r10, 3),
            "L2_vs_R10": round(r_l2_r10, 3),
            "L3_vs_R10": round(r_l3_r10, 3),
        },
        "inter_layer_coupling": {
            "L1_L2": round(coupling_l1_l2, 3),
            "L1_L3": round(coupling_l1_l3, 3),
            "L2_L3": round(coupling_l2_l3, 3),
        },
        "area_layer_profiles": area_layer_profiles,
        "mean_l1_frac": round(np.mean(l1_fracs), 3),
        "mean_l2_frac": round(np.mean(l2_fracs), 3),
        "mean_l3_frac": round(np.mean(l3_fracs), 3),
        "interpretation": "If layers are UNCOUPLED (coupling near 0) and layer-specific R@10 is high, the CNC model works — each layer cuts independently. If coupled, the system is holistic.",
        "cnc_verdict": None,
    }

    if abs(coupling_l1_l2) < 0.3 and abs(coupling_l1_l3) < 0.3:
        result["cnc_verdict"] = "CNC-LIKE: Layers are weakly coupled. Different axes of disruption."
    else:
        result["cnc_verdict"] = "NOT CNC: Layers are coupled. Disruption propagates across planes."

    print(f"\n  Inter-layer coupling: L1-L2={coupling_l1_l2:.3f}, L1-L3={coupling_l1_l3:.3f}, L2-L3={coupling_l2_l3:.3f}")
    print(f"  Mean fractions: L1={np.mean(l1_fracs):.3f}, L2={np.mean(l2_fracs):.3f}, L3={np.mean(l3_fracs):.3f}")
    print(f"  CNC verdict: {result['cnc_verdict']}")

    if area_layer_profiles:
        print(f"\n  Layer profiles by therapeutic area:")
        for area, prof in area_layer_profiles.items():
            print(f"    {area} (n={prof['n']}): L1={prof['mean_l1']:.3f} L2={prof['mean_l2']:.3f} L3={prof['mean_l3']:.3f}")

    return result


def load_state_safe():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "rb") as f:
            return pickle.load(f)
    print("  State not found, rebuilding from DB...")
    sys.path.insert(0, ".")
    from validation.sensitivity.module8_full_shuffle import load_state_from_db
    load_state_from_db()
    with open(STATE_PATH, "rb") as f:
        return pickle.load(f)


def main():
    t0 = time.time()
    print("=" * 70)
    print("14 UNKNOWNS — Comprehensive Exploration Suite")
    print("=" * 70)

    print("\n[Loading data...]")
    profiles = load_profiles()
    print(f"  Loaded {len(profiles)} gene disruption profiles")
    drugs = load_ground_truth()
    print(f"  Loaded {len(drugs)} drug ground truths")
    gene_depts = load_gene_depts()
    print(f"  Loaded {len(gene_depts)} gene-department mappings")
    pheno_dept_vecs = build_phenotype_to_dept_vec()
    print(f"  Built {len(pheno_dept_vecs)} phenotype→department vectors")

    results = {}

    results["U01"] = run_u01(profiles, drugs, pheno_dept_vecs)
    results["U02"] = run_u02(profiles)
    results["U03"] = run_u03(profiles, gene_depts)
    results["U04"] = run_u04(profiles)
    results["U05"] = run_u05(profiles, drugs, pheno_dept_vecs)
    results["U06"] = run_u06(profiles, gene_depts)
    results["U07"] = run_u07(profiles, drugs, pheno_dept_vecs)
    results["U08"] = run_u08(profiles, drugs, pheno_dept_vecs)
    results["U09"] = run_u09()
    results["U10"] = run_u10(profiles)
    results["U11"] = run_u11(profiles, gene_depts)
    results["U12"] = run_u12()
    results["U13"] = run_u13(profiles)
    results["U14"] = run_u14(profiles, drugs, pheno_dept_vecs, gene_depts)

    elapsed = time.time() - t0
    results["metadata"] = {
        "total_runtime_seconds": round(elapsed, 1),
        "n_profiles": len(profiles),
        "n_drugs": len(drugs),
        "n_departments": N_DEPTS,
        "n_phenotype_categories": len(pheno_dept_vecs),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"ALL 14 UNKNOWNS COMPLETE — {elapsed:.1f}s")
    print(f"Results: {OUTPUT_PATH}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
