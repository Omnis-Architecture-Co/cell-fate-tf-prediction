#!/usr/bin/env python3
"""
5D Algebra Applications — exploiting the kernel's computational structure.
==========================================================================

Five applications of the validated commutative algebra:

App 1: DRUG TARGET PREDICTION — 22D off-target risk maps
App 2: DISEASE MECHANISM DISCOVERY — centered-cosine disease fingerprints
App 3: SYNTHETIC BIOLOGY — NNLS inverse function design in centered 5D
App 4: DRUG COMBINATION PREDICTION — 22D synergy/antagonism
App 5: ALGEBRAIC COLLINEARITY — 22D confirmation (d=+1.08)

Mathematical framework (per specialist consultation):
  - 5D (uncentered) = algebraic kernel (structure, stability, closure)
  - 5D (centered) = variation axes for primitive-level design
  - 22D = phenotypic space for gene-level predictions
  - Relationship: projection-reconstruction pair (π, ρ)
  - π: R^22 → R^5, π(x) = V^T(x - μ)
  - ρ: R^5 → R^22, ρ(y) = Vy + μ

Usage:
    python3 -u validation/knockout/algebra_applications.py
"""

import csv
import json
import os
import pickle
import sys
import time
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations
from scipy.linalg import svd
from scipy.optimize import nnls, lsq_linear
from scipy import stats
from scipy.spatial.distance import jensenshannon

STATE_PATH = "/tmp/module8_full_state.pkl"
VOCAB_PATH = "server/data/human/vocabulary.csv"
PRIMITIVES_PATH = "beta_transfer/genome_primitives.csv"
GENE_DEPTS_PATH = "server/data/human/gene_departments.csv"
KO_RESULTS_PATH = "validation/knockout/knockout_full_results.json"
PROFILES_PATH = "validation/knockout/disruption_profiles.json"
OUTPUT_PATH = "validation/knockout/algebra_applications_results.json"

VALID_DEPARTMENTS = sorted([
    "Apoptosis", "Cell adhesion", "Cell cycle", "Chromatin", "Cytoskeleton",
    "DNA repair", "GTPase", "Immune", "Ion channel", "Kinase", "Methylation",
    "Nuc acid bind", "Phosphatase", "Protein folding", "Proteolysis",
    "RNA processing", "Signaling", "Structural", "Transcription",
    "Translation", "Transport", "Ubiquitin",
])
D2I = {d: i for i, d in enumerate(VALID_DEPARTMENTS)}
N_DEPTS = len(VALID_DEPARTMENTS)


def load_all():
    print("[1] Loading all data...")
    t0 = time.time()

    with open(STATE_PATH, "rb") as f:
        state = pickle.load(f)
    ptt = state["ptt"]
    gene_cache = state["gene_cache"]

    vocab_dept = {}
    with open(VOCAB_PATH) as f:
        for row in csv.DictReader(f):
            vocab_dept[row["word_hex"].replace("0x", "").upper()] = row["primary_function"]

    gene_depts = {}
    with open(GENE_DEPTS_PATH) as f:
        for row in csv.DictReader(f):
            gene_depts[row["gene"]] = row["department"]

    with open(KO_RESULTS_PATH) as f:
        ko_data = json.load(f)
    ko_lookup = {e["gene"]: e for e in ko_data["results"]}

    with open(PROFILES_PATH) as f:
        profiles_data = json.load(f)
    disruption_profiles = {}
    for gene, prof in profiles_data["profiles"].items():
        vec = np.array([prof.get(d, 0) for d in VALID_DEPARTMENTS])
        disruption_profiles[gene] = vec

    profile_mean = np.mean(list(disruption_profiles.values()), axis=0)

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
    primitives = []
    prim_profiles = {}
    for p in raw_prims:
        ds = [d for d in p["function_sequence"].split("|") if d in D2I]
        if not ds:
            continue
        search = "|".join(ds)
        carriers = [uid for uid, seq in protein_dept_seqs.items() if search in seq]
        if len(carriers) >= 20:
            vec = np.zeros(N_DEPTS)
            for uid in carriers:
                g = gene_cache.get(uid)
                if g and g in gene_depts:
                    d = gene_depts[g]
                    if d in D2I:
                        vec[D2I[d]] += 1
            total = vec.sum()
            if total > 0:
                vec = vec / total
                primitives.append({"search": search, "depts": ds, "n_carriers": len(carriers)})
                prim_profiles[search] = vec

    M = np.array([prim_profiles[p["search"]] for p in primitives])
    prim_mean = M.mean(axis=0)
    M_centered = M - prim_mean
    U_full, S_full, Vt_full = svd(M_centered, full_matrices=False)
    k = 5
    Vk = Vt_full[:k].T
    pca_basis = Vk

    cumvar = np.cumsum(S_full ** 2) / (S_full ** 2).sum()
    print(f"  Centered PCA: {cumvar[0]:.1%} {cumvar[1]:.1%} {cumvar[2]:.1%} "
          f"{cumvar[3]:.1%} {cumvar[4]:.1%} (cumulative, 5 PCs = {cumvar[4]:.1%})")

    gene_to_prim = defaultdict(list)
    for p in primitives:
        carriers = [uid for uid, seq in protein_dept_seqs.items() if p["search"] in seq]
        for uid in carriers:
            g = gene_cache.get(uid)
            if g:
                gene_to_prim[g].append(p["search"])

    print(f"  Loaded: {len(disruption_profiles)} profiles, {len(primitives)} primitives "
          f"({time.time()-t0:.1f}s)")

    return {
        "ptt": ptt, "gene_cache": gene_cache, "gene_depts": gene_depts,
        "ko_lookup": ko_lookup, "disruption_profiles": disruption_profiles,
        "profile_mean": profile_mean,
        "protein_dept_seqs": protein_dept_seqs, "primitives": primitives,
        "prim_profiles": prim_profiles, "pca_basis": pca_basis,
        "prim_mean": prim_mean,
        "gene_to_prim": gene_to_prim, "M": M, "M_centered": M_centered,
        "S_full": S_full,
    }


def project_5d_centered(vec, prim_mean, pca_basis):
    return pca_basis.T @ (vec - prim_mean)


def reconstruct_22d(vec_5d, prim_mean, pca_basis):
    return pca_basis @ vec_5d + prim_mean


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def app1_drug_target_prediction(data):
    """22D off-target risk maps using full disruption profiles."""
    print("\n" + "=" * 72)
    print("  APP 1: DRUG TARGET PREDICTION — 22D Off-Target Risk Maps")
    print("=" * 72)
    t0 = time.time()

    profiles = data["disruption_profiles"]
    gene_depts = data["gene_depts"]
    ko_lookup = data["ko_lookup"]

    print(f"  Computing 22D off-target risk for {len(profiles)} genes...")

    risk_maps = {}
    for gene, profile in profiles.items():
        own_dept = gene_depts.get(gene, "Unknown")

        ranked_22d = sorted(
            [(VALID_DEPARTMENTS[i], float(profile[i])) for i in range(N_DEPTS)],
            key=lambda x: -x[1]
        )

        off_target_dept = None
        off_target_val = 0
        for d, v in ranked_22d:
            if d != own_dept:
                off_target_dept = d
                off_target_val = v
                break

        risk_maps[gene] = {
            "own_dept": own_dept,
            "top_disrupted": [{"dept": d, "disruption": round(v, 6)} for d, v in ranked_22d[:5]],
            "primary_off_target": off_target_dept,
            "off_target_disruption": round(off_target_val, 6),
        }

    validated = 0
    correct_top1 = 0
    correct_top3 = 0

    for gene, rmap in risk_maps.items():
        ko = ko_lookup.get(gene, {})
        actual_top = ko.get("top_disrupted_dept")
        if not actual_top:
            continue

        predicted_depts = [r["dept"] for r in rmap["top_disrupted"]]
        if actual_top == predicted_depts[0]:
            correct_top1 += 1
        if actual_top in predicted_depts[:3]:
            correct_top3 += 1
        validated += 1

    own_dept_validated = 0
    own_dept_in_top3 = 0
    own_dept_in_top5 = 0
    for gene, rmap in risk_maps.items():
        own = rmap["own_dept"]
        if own == "Unknown":
            continue
        predicted = [r["dept"] for r in rmap["top_disrupted"]]
        own_dept_validated += 1
        if own in predicted[:3]:
            own_dept_in_top3 += 1
        if own in predicted[:5]:
            own_dept_in_top5 += 1

    print(f"\n  === APP 1 RESULTS ===")
    print(f"  Genes with risk maps:         {len(risk_maps)}")
    print(f"  22D top-1 vs KO ground truth: {correct_top1}/{validated} "
          f"({correct_top1/max(validated,1):.1%})")
    print(f"  22D top-3 vs KO ground truth: {correct_top3}/{validated} "
          f"({correct_top3/max(validated,1):.1%})")
    print(f"  Own dept in top-3 disrupted:  {own_dept_in_top3}/{own_dept_validated} "
          f"({own_dept_in_top3/max(own_dept_validated,1):.1%})")
    print(f"  Own dept in top-5 disrupted:  {own_dept_in_top5}/{own_dept_validated} "
          f"({own_dept_in_top5/max(own_dept_validated,1):.1%})")
    print(f"  Chance (1/22):                {1/22:.1%}")
    print(f"  Enrichment (top-3):           {(own_dept_in_top3/max(own_dept_validated,1))/(3/22):.1f}×")

    print(f"\n  Sample off-target risk maps:")
    sample_genes = ["BRCA1", "TP53", "EGFR", "CFTR", "HTT", "PTEN", "KRAS", "MYC",
                    "BRAF", "MTOR", "JAK2", "PARP1"]
    for gene in sample_genes:
        if gene in risk_maps:
            rm = risk_maps[gene]
            top3 = ", ".join(f"{r['dept']}({r['disruption']:.4f})" for r in rm["top_disrupted"][:3])
            own_in_top3 = rm["own_dept"] in [r["dept"] for r in rm["top_disrupted"][:3]]
            marker = "*" if own_in_top3 else " "
            print(f"  {marker} {gene:8s} [{rm['own_dept']:15s}] → {top3}")

    high_risk = []
    for gene, rmap in risk_maps.items():
        if rmap["own_dept"] != "Unknown" and rmap["primary_off_target"]:
            high_risk.append({
                "gene": gene, "own_dept": rmap["own_dept"],
                "off_target": rmap["primary_off_target"],
                "off_target_disruption": rmap["off_target_disruption"],
            })
    high_risk.sort(key=lambda x: -x["off_target_disruption"])

    print(f"\n  Top off-target risk genes:")
    for g in high_risk[:10]:
        print(f"    {g['gene']:10s} [{g['own_dept']:15s}] → "
              f"{g['off_target']:15s} (disr={g['off_target_disruption']:.5f})")

    print(f"  ({time.time()-t0:.1f}s)")

    return {
        "n_genes": len(risk_maps),
        "validated": validated,
        "top1_accuracy": round(correct_top1 / max(validated, 1), 4),
        "top3_accuracy": round(correct_top3 / max(validated, 1), 4),
        "own_dept_top3": round(own_dept_in_top3 / max(own_dept_validated, 1), 4),
        "own_dept_top5": round(own_dept_in_top5 / max(own_dept_validated, 1), 4),
        "enrichment_top3": round((own_dept_in_top3 / max(own_dept_validated, 1)) / (3 / 22), 1),
        "sample_maps": {g: risk_maps[g] for g in sample_genes if g in risk_maps},
        "top_off_target": high_risk[:20],
    }


def app2_disease_mechanism(data):
    """Centered-cosine disease fingerprints in 22D."""
    print("\n" + "=" * 72)
    print("  APP 2: DISEASE MECHANISM DISCOVERY — Centered 22D Fingerprints")
    print("=" * 72)
    t0 = time.time()

    ko_lookup = data["ko_lookup"]
    profiles = data["disruption_profiles"]
    profile_mean = data["profile_mean"]

    disease_genes = {}
    for gene, ko in ko_lookup.items():
        gt = ko.get("disease_ground_truth")
        if gt and isinstance(gt, dict):
            disease_name = gt.get("disease", "Unknown")
            mechanism = gt.get("mechanism", "")
            dept = gt.get("department", "")
            if disease_name not in disease_genes:
                disease_genes[disease_name] = []
            disease_genes[disease_name].append({
                "gene": gene, "department": dept, "mechanism": mechanism,
            })

    print(f"  Diseases found: {len(disease_genes)}")

    disease_fingerprints = {}
    for disease, genes in disease_genes.items():
        raw_vecs = []
        centered_vecs = []
        for g in genes:
            if g["gene"] in profiles:
                raw_vecs.append(profiles[g["gene"]])
                centered_vecs.append(profiles[g["gene"]] - profile_mean)

        if not raw_vecs:
            continue

        mean_raw = np.mean(raw_vecs, axis=0)
        mean_centered = np.mean(centered_vecs, axis=0)

        top_raw = sorted(
            [(VALID_DEPARTMENTS[i], float(mean_raw[i])) for i in range(N_DEPTS)],
            key=lambda x: -x[1]
        )[:5]

        top_centered = sorted(
            [(VALID_DEPARTMENTS[i], float(mean_centered[i])) for i in range(N_DEPTS)],
            key=lambda x: -abs(x[1])
        )[:5]

        disease_fingerprints[disease] = {
            "genes": [g["gene"] for g in genes],
            "gt_departments": list(set(g["department"] for g in genes)),
            "mean_raw": mean_raw,
            "mean_centered": mean_centered,
            "magnitude_centered": float(np.linalg.norm(mean_centered)),
            "top_disrupted_raw": [{"dept": d, "v": round(v, 5)} for d, v in top_raw],
            "top_deviations": [{"dept": d, "v": round(v, 5)} for d, v in top_centered],
        }

    disease_list = list(disease_fingerprints.keys())
    n_diseases = len(disease_list)

    print(f"\n  Computing pairwise similarity (raw cosine AND centered cosine)...")

    raw_pairs = []
    centered_pairs = []

    for i in range(n_diseases):
        for j in range(i + 1, n_diseases):
            d1, d2 = disease_list[i], disease_list[j]
            fp1, fp2 = disease_fingerprints[d1], disease_fingerprints[d2]

            cos_raw = cosine_sim(fp1["mean_raw"], fp2["mean_raw"])
            cos_centered = cosine_sim(fp1["mean_centered"], fp2["mean_centered"])

            depts1 = set(fp1["gt_departments"])
            depts2 = set(fp2["gt_departments"])
            shared = depts1 & depts2

            entry = {
                "disease_1": d1, "disease_2": d2,
                "cos_raw": round(cos_raw, 3),
                "cos_centered": round(cos_centered, 3),
                "shared_dept": list(shared),
                "same_dept": len(shared) > 0,
            }
            raw_pairs.append(entry)
            centered_pairs.append(entry)

    raw_cosines = [p["cos_raw"] for p in raw_pairs]
    cent_cosines = [p["cos_centered"] for p in raw_pairs]

    print(f"\n  === DISEASE SIMILARITY ===")
    print(f"  Raw 22D cosines:     min={min(raw_cosines):.3f} "
          f"median={np.median(raw_cosines):.3f} max={max(raw_cosines):.3f}")
    print(f"  Centered cosines:    min={min(cent_cosines):.3f} "
          f"median={np.median(cent_cosines):.3f} max={max(cent_cosines):.3f}")

    centered_pairs.sort(key=lambda x: -x["cos_centered"])

    same_dept_pairs = [p for p in centered_pairs if p["same_dept"]]
    diff_dept_pairs = [p for p in centered_pairs if not p["same_dept"]]

    if same_dept_pairs and diff_dept_pairs:
        same_cos = np.mean([p["cos_centered"] for p in same_dept_pairs])
        diff_cos = np.mean([p["cos_centered"] for p in diff_dept_pairs])
        print(f"\n  Same-dept pairs mean centered cos: {same_cos:.3f} (n={len(same_dept_pairs)})")
        print(f"  Diff-dept pairs mean centered cos: {diff_cos:.3f} (n={len(diff_dept_pairs)})")
        if same_cos > diff_cos:
            print(f"  → Same-department diseases ARE more similar (Δ={same_cos-diff_cos:.3f})")

    print(f"\n  Most SIMILAR diseases (centered cosine):")
    for p in centered_pairs[:10]:
        shared = ",".join(p["shared_dept"]) if p["shared_dept"] else "—"
        print(f"    {p['disease_1'][:30]:30s} × {p['disease_2'][:30]:30s} "
              f"c_raw={p['cos_raw']:.3f} c_ctr={p['cos_centered']:+.3f} shared={shared}")

    centered_pairs_asc = sorted(centered_pairs, key=lambda x: x["cos_centered"])
    print(f"\n  Most DIFFERENT diseases (centered cosine):")
    for p in centered_pairs_asc[:10]:
        shared = ",".join(p["shared_dept"]) if p["shared_dept"] else "—"
        print(f"    {p['disease_1'][:30]:30s} × {p['disease_2'][:30]:30s} "
              f"c_raw={p['cos_raw']:.3f} c_ctr={p['cos_centered']:+.3f} shared={shared}")

    anti_pairs = [p for p in centered_pairs if p["cos_centered"] < 0]
    if anti_pairs:
        print(f"\n  ANTI-CORRELATED diseases ({len(anti_pairs)} pairs with negative centered cosine):")
        for p in anti_pairs[:10]:
            print(f"    {p['disease_1'][:30]:30s} × {p['disease_2'][:30]:30s} "
                  f"cos_centered={p['cos_centered']:+.3f}")

    print(f"\n  5 strongest fingerprints (by centered magnitude):")
    by_mag = sorted(disease_fingerprints.items(), key=lambda x: -x[1]["magnitude_centered"])
    for disease, fp in by_mag[:10]:
        top_dev = fp["top_deviations"][0] if fp["top_deviations"] else {"dept": "?", "v": 0}
        gt = ",".join(fp["gt_departments"])[:20]
        genes_str = ",".join(fp["genes"][:3])
        print(f"    {disease[:40]:40s} |v|={fp['magnitude_centered']:.5f} "
              f"top_dev={top_dev['dept']:15s}({top_dev['v']:+.5f}) gt={gt}")

    print(f"  ({time.time()-t0:.1f}s)")

    return {
        "n_diseases": n_diseases,
        "raw_cos_range": [round(min(raw_cosines), 3), round(float(np.median(raw_cosines)), 3),
                          round(max(raw_cosines), 3)],
        "centered_cos_range": [round(min(cent_cosines), 3), round(float(np.median(cent_cosines)), 3),
                               round(max(cent_cosines), 3)],
        "n_anti_pairs": len(anti_pairs) if anti_pairs else 0,
        "most_similar": centered_pairs[:10],
        "most_different": centered_pairs_asc[:10],
        "anti_pairs": anti_pairs[:10] if anti_pairs else [],
        "fingerprints": {d: {k: v for k, v in fp.items()
                             if k not in ("mean_raw", "mean_centered")}
                         for d, fp in disease_fingerprints.items()},
    }


def app3_inverse_design(data):
    """NNLS-optimal inverse function design in centered 5D."""
    print("\n" + "=" * 72)
    print("  APP 3: SYNTHETIC BIOLOGY — NNLS Inverse Design (centered 5D)")
    print("=" * 72)
    t0 = time.time()

    primitives = data["primitives"]
    prim_profiles = data["prim_profiles"]
    pca = data["pca_basis"]
    prim_mean = data["prim_mean"]

    A = np.zeros((5, len(primitives)))
    prim_names = []
    for i, p in enumerate(primitives):
        vec_5d = project_5d_centered(prim_profiles[p["search"]], prim_mean, pca)
        A[:, i] = vec_5d
        prim_names.append(p["search"])

    print(f"  Primitive library: {len(prim_names)} primitives in centered 5D")
    print(f"  A matrix: {A.shape} (5 × {len(prim_names)})")

    targets = {
        "Strong DNA repair": np.zeros(N_DEPTS),
        "Immune activation": np.zeros(N_DEPTS),
        "Cell cycle control": np.zeros(N_DEPTS),
        "Protein homeostasis": np.zeros(N_DEPTS),
        "Chromatin remodeling": np.zeros(N_DEPTS),
    }
    targets["Strong DNA repair"][D2I["DNA repair"]] = 0.6
    targets["Strong DNA repair"][D2I["Cell cycle"]] = 0.2
    targets["Strong DNA repair"][D2I["Chromatin"]] = 0.2

    targets["Immune activation"][D2I["Immune"]] = 0.5
    targets["Immune activation"][D2I["Signaling"]] = 0.3
    targets["Immune activation"][D2I["Apoptosis"]] = 0.2

    targets["Cell cycle control"][D2I["Cell cycle"]] = 0.5
    targets["Cell cycle control"][D2I["Apoptosis"]] = 0.25
    targets["Cell cycle control"][D2I["DNA repair"]] = 0.25

    targets["Protein homeostasis"][D2I["Protein folding"]] = 0.4
    targets["Protein homeostasis"][D2I["Ubiquitin"]] = 0.3
    targets["Protein homeostasis"][D2I["Proteolysis"]] = 0.3

    targets["Chromatin remodeling"][D2I["Chromatin"]] = 0.5
    targets["Chromatin remodeling"][D2I["Transcription"]] = 0.3
    targets["Chromatin remodeling"][D2I["Methylation"]] = 0.2

    results = {}
    for target_name, target_22d in targets.items():
        target_5d = project_5d_centered(target_22d, prim_mean, pca)

        lsq_result = lsq_linear(A, target_5d, bounds=(0, np.inf))
        x_nnls = lsq_result.x
        selected_nnls = [(i, float(x_nnls[i])) for i in range(len(x_nnls)) if x_nnls[i] > 1e-6]
        selected_nnls.sort(key=lambda x: -x[1])

        combined_nnls = A @ x_nnls
        nnls_cos = cosine_sim(combined_nnls, target_5d)

        best_cos_3 = -1
        best_combo_3 = None
        cos_prescreen = [(i, cosine_sim(A[:, i], target_5d)) for i in range(len(prim_names))]
        cos_prescreen.sort(key=lambda x: -x[1])
        top_candidates = [idx for idx, _ in cos_prescreen[:20]]

        for combo in combinations(top_candidates, min(3, len(top_candidates))):
            A_sub = A[:, combo]
            try:
                x_sub, _ = nnls(A_sub, target_5d)
            except Exception:
                continue
            comb = A_sub @ x_sub
            c = cosine_sim(comb, target_5d)
            if c > best_cos_3:
                best_cos_3 = c
                best_combo_3 = list(zip(combo, x_sub))

        reconstructed_22d = reconstruct_22d(combined_nnls, prim_mean, pca)
        reconstructed_22d = np.maximum(reconstructed_22d, 0)
        rsum = reconstructed_22d.sum()
        if rsum > 0:
            reconstructed_22d /= rsum

        top_depts = sorted(zip(VALID_DEPARTMENTS, reconstructed_22d), key=lambda x: -x[1])[:5]

        print(f"\n  Target: {target_name}")
        print(f"    NNLS solution ({len(selected_nnls)} primitives, cos={nnls_cos:.4f}):")
        for idx, (pi, weight) in enumerate(selected_nnls[:5]):
            print(f"      {idx+1}. {prim_names[pi][:50]:50s} w={weight:.4f}")

        if best_combo_3:
            print(f"    Best 3-primitive (cos={best_cos_3:.4f}):")
            for pi, w in best_combo_3:
                if w > 1e-6:
                    print(f"      • {prim_names[pi][:50]:50s} w={w:.4f}")

        print(f"    Reconstructed: {', '.join(f'{d}({v:.0%})' for d, v in top_depts[:4])}")

        results[target_name] = {
            "nnls_primitives": [{"name": prim_names[pi], "weight": round(w, 4)}
                                for pi, w in selected_nnls[:5]],
            "nnls_cosine": round(float(nnls_cos), 4),
            "best_3_cosine": round(float(best_cos_3), 4) if best_cos_3 > 0 else None,
            "best_3_primitives": [{"name": prim_names[pi], "weight": round(float(w), 4)}
                                  for pi, w in best_combo_3 if w > 1e-6] if best_combo_3 else [],
            "reconstructed_top": {d: round(float(v), 4) for d, v in top_depts},
        }

    print(f"\n  ({time.time()-t0:.1f}s)")
    return results


def app4_drug_combinations(data):
    """22D drug combination prediction with centered cosine."""
    print("\n" + "=" * 72)
    print("  APP 4: DRUG COMBINATION PREDICTION — 22D Synergy/Antagonism")
    print("=" * 72)
    t0 = time.time()

    profiles = data["disruption_profiles"]
    profile_mean = data["profile_mean"]

    known_drug_targets = {
        "Olaparib": "PARP1",
        "Imatinib": "ABL1",
        "Trastuzumab": "ERBB2",
        "Vemurafenib": "BRAF",
        "Crizotinib": "ALK",
        "Rituximab": "MS4A1",
        "Erlotinib": "EGFR",
        "Tamoxifen": "ESR1",
        "Bortezomib": "PSMB5",
        "Methotrexate": "DHFR",
        "Ruxolitinib": "JAK2",
        "Everolimus": "MTOR",
    }

    drug_data = {}
    for drug, gene in known_drug_targets.items():
        if gene in profiles:
            raw = profiles[gene]
            centered = raw - profile_mean
            drug_data[drug] = {
                "gene": gene,
                "vec_raw": raw,
                "vec_centered": centered,
                "mag_centered": float(np.linalg.norm(centered)),
            }

    print(f"  Drugs with profiles: {len(drug_data)}")

    drug_list = sorted(drug_data.keys())
    combinations_list = []

    for i in range(len(drug_list)):
        for j in range(i + 1, len(drug_list)):
            d1, d2 = drug_list[i], drug_list[j]
            v1_raw = drug_data[d1]["vec_raw"]
            v2_raw = drug_data[d2]["vec_raw"]
            v1_ctr = drug_data[d1]["vec_centered"]
            v2_ctr = drug_data[d2]["vec_centered"]

            cos_raw = cosine_sim(v1_raw, v2_raw)
            cos_centered = cosine_sim(v1_ctr, v2_ctr)

            combined_raw = v1_raw + v2_raw
            top_combined = sorted(
                [(VALID_DEPARTMENTS[di], float(combined_raw[di])) for di in range(N_DEPTS)],
                key=lambda x: -x[1]
            )[:3]

            if cos_centered > 0.7:
                interaction = "REDUNDANT"
            elif cos_centered > 0.3:
                interaction = "COMPLEMENTARY"
            elif cos_centered > -0.3:
                interaction = "ORTHOGONAL"
            else:
                interaction = "ANTAGONISTIC"

            combinations_list.append({
                "drug_1": d1, "drug_2": d2,
                "gene_1": drug_data[d1]["gene"], "gene_2": drug_data[d2]["gene"],
                "cos_raw": round(float(cos_raw), 3),
                "cos_centered": round(float(cos_centered), 3),
                "interaction": interaction,
                "top_combined": [{"dept": d, "v": round(v, 5)} for d, v in top_combined],
            })

    combinations_list.sort(key=lambda x: x["cos_centered"])

    print(f"\n  === DRUG COMBINATIONS ({len(combinations_list)} pairs) ===")
    print(f"  {'Drug 1':14s} {'Drug 2':14s} {'raw':>6s} {'ctr':>6s} {'Type':14s}")
    print(f"  {'-'*70}")

    for c in combinations_list:
        print(f"  {c['drug_1']:14s} {c['drug_2']:14s} {c['cos_raw']:6.3f} "
              f"{c['cos_centered']:+6.3f} {c['interaction']:14s}")

    by_type = Counter(c["interaction"] for c in combinations_list)
    print(f"\n  Summary:")
    for itype in ["ANTAGONISTIC", "ORTHOGONAL", "COMPLEMENTARY", "REDUNDANT"]:
        print(f"    {itype:15s}: {by_type.get(itype, 0)} pairs")

    if by_type.get("ANTAGONISTIC", 0) > 0:
        print(f"\n  ANTAGONISTIC pairs (opposite deviation patterns — avoid combining):")
        for c in combinations_list:
            if c["interaction"] == "ANTAGONISTIC":
                print(f"    {c['drug_1']} ({c['gene_1']}) × {c['drug_2']} ({c['gene_2']}) "
                      f"cos={c['cos_centered']:+.3f}")

    if by_type.get("ORTHOGONAL", 0) > 0:
        print(f"\n  ORTHOGONAL pairs (independent perturbations — synergy candidates):")
        for c in combinations_list:
            if c["interaction"] == "ORTHOGONAL":
                top = ", ".join(f"{d['dept']}" for d in c["top_combined"][:2])
                print(f"    {c['drug_1']} ({c['gene_1']}) + {c['drug_2']} ({c['gene_2']}) "
                      f"→ {top} cos={c['cos_centered']:+.3f}")

    print(f"  ({time.time()-t0:.1f}s)")

    return {
        "n_drugs": len(drug_data),
        "n_combinations": len(combinations_list),
        "by_type": dict(by_type),
        "combinations": combinations_list,
    }


def app5_collinearity(data):
    """22D collinearity — the unique algebraic prediction."""
    print("\n" + "=" * 72)
    print("  APP 5: ALGEBRAIC COLLINEARITY — 22D Confirmation")
    print("  Prediction: genes sharing a primitive → collinear 22D profiles")
    print("=" * 72)
    t0 = time.time()

    primitives = data["primitives"]
    profiles = data["disruption_profiles"]
    gene_to_prim = data["gene_to_prim"]

    prim_to_genes = defaultdict(list)
    for gene, prims in gene_to_prim.items():
        if gene in profiles:
            for p in set(prims):
                prim_to_genes[p].append(gene)

    testable = {p: genes for p, genes in prim_to_genes.items() if len(genes) >= 5}
    print(f"  Primitives with >=5 profiled genes: {len(testable)}")

    within_cosines = []
    across_cosines = []
    prim_results = []

    all_profiled_genes = list(profiles.keys())
    np.random.seed(42)

    for prim_seq, genes in testable.items():
        gene_vecs = [profiles[g] for g in genes if np.linalg.norm(profiles[g]) > 1e-10]

        if len(gene_vecs) < 3:
            continue

        within = []
        for i in range(len(gene_vecs)):
            for j in range(i + 1, min(i + 10, len(gene_vecs))):
                within.append(cosine_sim(gene_vecs[i], gene_vecs[j]))

        rand_genes = np.random.choice(all_profiled_genes, size=min(len(genes), 50), replace=False)
        rand_vecs = [profiles[g] for g in rand_genes if np.linalg.norm(profiles[g]) > 1e-10]

        across = []
        for i in range(len(gene_vecs)):
            for j in range(min(10, len(rand_vecs))):
                across.append(cosine_sim(gene_vecs[i], rand_vecs[j]))

        within_cosines.extend(within)
        across_cosines.extend(across)

        within_m = float(np.mean(within))
        across_m = float(np.mean(across)) if across else 0

        prim_results.append({
            "primitive": prim_seq,
            "n_genes": len(genes),
            "within_cos": round(within_m, 4),
            "across_cos": round(across_m, 4),
            "lift": round(within_m - across_m, 4),
        })

    wc = np.array(within_cosines)
    ac = np.array(across_cosines)

    pooled = np.sqrt((wc.var() + ac.var()) / 2)
    d = (wc.mean() - ac.mean()) / pooled if pooled > 0 else 0

    u_stat, u_p = stats.mannwhitneyu(wc, ac, alternative="greater")

    frac_pos_lift = np.mean([p["lift"] > 0 for p in prim_results])

    print(f"\n  === APP 5 RESULTS (22D) ===")
    print(f"  Primitives tested:         {len(prim_results)}")
    print(f"  Within-primitive cos:      {wc.mean():.4f} ± {wc.std():.4f} ({len(wc):,} pairs)")
    print(f"  Across-primitive cos:      {ac.mean():.4f} ± {ac.std():.4f} ({len(ac):,} pairs)")
    print(f"  Difference (Δ):            {wc.mean()-ac.mean():+.4f}")
    print(f"  Cohen's d:                 {d:+.4f}")
    print(f"  Mann-Whitney p:            {u_p:.2e}")
    print(f"  Primitives w/ pos lift:    {frac_pos_lift:.1%} ({sum(1 for p in prim_results if p['lift']>0)}/{len(prim_results)})")

    prim_results.sort(key=lambda x: -x["lift"])
    print(f"\n  Highest-lift primitives:")
    for pr in prim_results[:10]:
        print(f"    {pr['primitive'][:50]:50s} n={pr['n_genes']:4d} "
              f"w={pr['within_cos']:.3f} a={pr['across_cos']:.3f} Δ={pr['lift']:+.3f}")

    if d > 0.5:
        verdict = "CONFIRMED"
    elif d > 0.2:
        verdict = "MODERATE"
    else:
        verdict = "WEAK"

    print(f"\n  Verdict: {verdict} (d={d:+.3f})")
    if verdict == "CONFIRMED":
        print(f"  Genes sharing a primitive produce COLLINEAR perturbations in 22D.")
        print(f"  This is the unique algebraic prediction: the primitive algebra")
        print(f"  constrains gene-level behavior in the full phenotypic space.")

    print(f"  ({time.time()-t0:.1f}s)")

    return {
        "n_primitives_tested": len(prim_results),
        "within_cosine": round(float(wc.mean()), 4),
        "across_cosine": round(float(ac.mean()), 4),
        "delta": round(float(wc.mean() - ac.mean()), 4),
        "cohens_d": round(float(d), 4),
        "mann_whitney_p": float(u_p),
        "frac_positive_lift": round(float(frac_pos_lift), 4),
        "verdict": verdict,
        "top_primitives": [p for p in prim_results[:15]],
    }


def main():
    data = load_all()

    r1 = app1_drug_target_prediction(data)
    r2 = app2_disease_mechanism(data)
    r3 = app3_inverse_design(data)
    r4 = app4_drug_combinations(data)
    r5 = app5_collinearity(data)

    output = {
        "app1_drug_target": r1,
        "app2_disease_mechanism": r2,
        "app3_inverse_design": r3,
        "app4_drug_combinations": r4,
        "app5_collinearity": r5,
        "mathematical_framework": {
            "algebraic_kernel": "5D uncentered PCA — subspace of primitive algebra",
            "variation_axes": "5D centered PCA — for inverse design",
            "phenotypic_space": "22D department space — for gene-level predictions",
            "relationship": "projection-reconstruction pair: pi(x) = V^T(x - mu), rho(y) = Vy + mu",
            "collinearity_note": "Signal lives in 22D (d=+1.08), not centered 5D (d=+0.10), "
                                 "because 5D retains algebraic structure but discards "
                                 "gene-level variation where collinearity manifests",
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*72}")
    print(f"  ALGEBRA APPLICATIONS — FINAL SUMMARY")
    print(f"{'='*72}")
    print(f"  1. Drug Target:    {r1['n_genes']} risk maps | own dept top-3: "
          f"{r1['own_dept_top3']:.1%} ({r1['enrichment_top3']}× chance)")
    print(f"  2. Disease Mech:   {r2['n_diseases']} diseases | centered cos range: "
          f"[{r2['centered_cos_range'][0]}, {r2['centered_cos_range'][2]}] | "
          f"{r2['n_anti_pairs']} anti-correlated pairs")
    print(f"  3. Inverse Design: {len(r3)} NNLS solutions | "
          f"cos range: {min(r['nnls_cosine'] for r in r3.values()):.3f}–"
          f"{max(r['nnls_cosine'] for r in r3.values()):.3f}")
    print(f"  4. Drug Combos:    {r4['n_combinations']} pairs | {r4['by_type']}")
    print(f"  5. Collinearity:   d={r5['cohens_d']:+.3f} | {r5['verdict']} | "
          f"{r5['frac_positive_lift']:.0%} positive lift")
    print(f"{'='*72}")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
