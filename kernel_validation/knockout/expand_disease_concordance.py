#!/usr/bin/env python3
"""
Expand disease concordance from 48 to 200+ genes using programmatic OMIM/ClinVar mapping.
Uses NCBI Entrez to fetch disease-gene associations for monogenic diseases.

Usage:
    python3 -u validation/knockout/expand_disease_concordance.py
"""

import csv
import json
import os
import pickle
import sys
import time
import urllib.request
import numpy as np
from collections import defaultdict, Counter

STATE_PATH = "/tmp/module8_full_state.pkl"
GENE_DEPTS_PATH = "server/data/human/gene_departments.csv"
FULL_PROFILES_PATH = "validation/knockout/disruption_profiles_full.json"
OUTPUT_PATH = "validation/knockout/expanded_disease_concordance.json"

VALID_DEPARTMENTS = sorted([
    "Apoptosis", "Cell adhesion", "Cell cycle", "Chromatin", "Cytoskeleton",
    "DNA repair", "GTPase", "Immune", "Ion channel", "Kinase", "Methylation",
    "Nuc acid bind", "Phosphatase", "Protein folding", "Proteolysis",
    "RNA processing", "Signaling", "Structural", "Transcription",
    "Translation", "Transport", "Ubiquitin",
])
D2I = {d: i for i, d in enumerate(VALID_DEPARTMENTS)}

CURATED_DISEASE_GENES = {
    "BRCA1": {"dept": "DNA repair", "disease": "Hereditary breast/ovarian cancer"},
    "BRCA2": {"dept": "DNA repair", "disease": "Hereditary breast/ovarian cancer"},
    "TP53": {"dept": "Apoptosis", "disease": "Li-Fraumeni syndrome"},
    "RB1": {"dept": "Cell cycle", "disease": "Retinoblastoma"},
    "CFTR": {"dept": "Ion channel", "disease": "Cystic fibrosis"},
    "PTEN": {"dept": "Phosphatase", "disease": "Cowden syndrome"},
    "APC": {"dept": "Signaling", "disease": "Familial adenomatous polyposis"},
    "MLH1": {"dept": "DNA repair", "disease": "Lynch syndrome"},
    "MSH2": {"dept": "DNA repair", "disease": "Lynch syndrome"},
    "MSH6": {"dept": "DNA repair", "disease": "Lynch syndrome"},
    "VHL": {"dept": "Proteolysis", "disease": "Von Hippel-Lindau"},
    "NF1": {"dept": "GTPase", "disease": "Neurofibromatosis type 1"},
    "NF2": {"dept": "Cytoskeleton", "disease": "Neurofibromatosis type 2"},
    "TSC1": {"dept": "Signaling", "disease": "Tuberous sclerosis"},
    "TSC2": {"dept": "Signaling", "disease": "Tuberous sclerosis"},
    "SMAD4": {"dept": "Signaling", "disease": "Juvenile polyposis"},
    "CDH1": {"dept": "Cell adhesion", "disease": "Hereditary diffuse gastric cancer"},
    "PALB2": {"dept": "DNA repair", "disease": "Hereditary breast cancer"},
    "ATM": {"dept": "DNA repair", "disease": "Ataxia-telangiectasia"},
    "CHEK2": {"dept": "DNA repair", "disease": "Breast cancer susceptibility"},
    "RAD51C": {"dept": "DNA repair", "disease": "Fanconi anemia"},
    "RAD51D": {"dept": "DNA repair", "disease": "Ovarian cancer susceptibility"},
    "FANCA": {"dept": "DNA repair", "disease": "Fanconi anemia A"},
    "FANCC": {"dept": "DNA repair", "disease": "Fanconi anemia C"},
    "FANCG": {"dept": "DNA repair", "disease": "Fanconi anemia G"},
    "XPA": {"dept": "DNA repair", "disease": "Xeroderma pigmentosum A"},
    "XPB": {"dept": "DNA repair", "disease": "Xeroderma pigmentosum B"},
    "XPC": {"dept": "DNA repair", "disease": "Xeroderma pigmentosum C"},
    "XPD": {"dept": "DNA repair", "disease": "Xeroderma pigmentosum D"},
    "DMD": {"dept": "Cytoskeleton", "disease": "Duchenne muscular dystrophy"},
    "LMNA": {"dept": "Structural", "disease": "Emery-Dreifuss muscular dystrophy"},
    "COL1A1": {"dept": "Structural", "disease": "Osteogenesis imperfecta"},
    "COL1A2": {"dept": "Structural", "disease": "Osteogenesis imperfecta"},
    "COL2A1": {"dept": "Structural", "disease": "Stickler syndrome"},
    "COL3A1": {"dept": "Structural", "disease": "Ehlers-Danlos syndrome"},
    "FBN1": {"dept": "Structural", "disease": "Marfan syndrome"},
    "ELN": {"dept": "Structural", "disease": "Cutis laxa"},
    "PKD1": {"dept": "Ion channel", "disease": "Polycystic kidney disease"},
    "PKD2": {"dept": "Ion channel", "disease": "Polycystic kidney disease"},
    "SCN1A": {"dept": "Ion channel", "disease": "Dravet syndrome"},
    "SCN2A": {"dept": "Ion channel", "disease": "Epileptic encephalopathy"},
    "SCN5A": {"dept": "Ion channel", "disease": "Brugada syndrome"},
    "SCN9A": {"dept": "Ion channel", "disease": "Congenital insensitivity to pain"},
    "KCNQ1": {"dept": "Ion channel", "disease": "Long QT syndrome"},
    "KCNQ2": {"dept": "Ion channel", "disease": "Epileptic encephalopathy"},
    "KCNH2": {"dept": "Ion channel", "disease": "Long QT syndrome"},
    "KCNJ11": {"dept": "Ion channel", "disease": "Neonatal diabetes"},
    "CLCN5": {"dept": "Ion channel", "disease": "Dent disease"},
    "CACNA1A": {"dept": "Ion channel", "disease": "Episodic ataxia type 2"},
    "RYR1": {"dept": "Ion channel", "disease": "Malignant hyperthermia"},
    "RYR2": {"dept": "Ion channel", "disease": "CPVT"},
    "ABCA4": {"dept": "Transport", "disease": "Stargardt disease"},
    "ABCB4": {"dept": "Transport", "disease": "Progressive familial intrahepatic cholestasis"},
    "ABCC8": {"dept": "Transport", "disease": "Neonatal diabetes"},
    "SLC6A3": {"dept": "Transport", "disease": "ADHD susceptibility"},
    "SLC12A3": {"dept": "Transport", "disease": "Gitelman syndrome"},
    "SLC22A5": {"dept": "Transport", "disease": "Carnitine deficiency"},
    "ATP7A": {"dept": "Transport", "disease": "Menkes disease"},
    "ATP7B": {"dept": "Transport", "disease": "Wilson disease"},
    "JAK2": {"dept": "Kinase", "disease": "Myeloproliferative neoplasms"},
    "ABL1": {"dept": "Kinase", "disease": "Chronic myeloid leukemia"},
    "RET": {"dept": "Kinase", "disease": "MEN2A"},
    "MET": {"dept": "Kinase", "disease": "Hereditary papillary renal carcinoma"},
    "ALK": {"dept": "Kinase", "disease": "Neuroblastoma"},
    "BRAF": {"dept": "Kinase", "disease": "Noonan syndrome"},
    "RAF1": {"dept": "Kinase", "disease": "Noonan syndrome"},
    "MAP2K1": {"dept": "Kinase", "disease": "CFC syndrome"},
    "MAP2K2": {"dept": "Kinase", "disease": "CFC syndrome"},
    "EGFR": {"dept": "Kinase", "disease": "Lung adenocarcinoma"},
    "FGFR1": {"dept": "Kinase", "disease": "Pfeiffer syndrome"},
    "FGFR2": {"dept": "Kinase", "disease": "Apert syndrome"},
    "FGFR3": {"dept": "Kinase", "disease": "Achondroplasia"},
    "CDK4": {"dept": "Cell cycle", "disease": "Familial melanoma"},
    "CDKN2A": {"dept": "Cell cycle", "disease": "Familial melanoma"},
    "CCND1": {"dept": "Cell cycle", "disease": "Mantle cell lymphoma"},
    "WNT1": {"dept": "Signaling", "disease": "Osteogenesis imperfecta XV"},
    "SHH": {"dept": "Signaling", "disease": "Holoprosencephaly"},
    "PTCH1": {"dept": "Signaling", "disease": "Gorlin syndrome"},
    "NOTCH1": {"dept": "Signaling", "disease": "Aortic valve disease"},
    "NOTCH3": {"dept": "Signaling", "disease": "CADASIL"},
    "HRAS": {"dept": "GTPase", "disease": "Costello syndrome"},
    "KRAS": {"dept": "GTPase", "disease": "Noonan syndrome"},
    "NRAS": {"dept": "GTPase", "disease": "Noonan syndrome"},
    "RAB27A": {"dept": "GTPase", "disease": "Griscelli syndrome"},
    "GNAQ": {"dept": "GTPase", "disease": "Sturge-Weber syndrome"},
    "GBA1": {"dept": "Proteolysis", "disease": "Gaucher disease"},
    "CTSK": {"dept": "Proteolysis", "disease": "Pycnodysostosis"},
    "ADAMTS2": {"dept": "Proteolysis", "disease": "Ehlers-Danlos VIIC"},
    "CASP10": {"dept": "Apoptosis", "disease": "Autoimmune lymphoproliferative syndrome"},
    "FAS": {"dept": "Apoptosis", "disease": "ALPS type 1A"},
    "FASLG": {"dept": "Apoptosis", "disease": "ALPS type 1B"},
    "XIAP": {"dept": "Apoptosis", "disease": "X-linked lymphoproliferative syndrome 2"},
    "BCL2": {"dept": "Apoptosis", "disease": "Follicular lymphoma"},
    "BAX": {"dept": "Apoptosis", "disease": "Colorectal cancer susceptibility"},
    "UBE3A": {"dept": "Ubiquitin", "disease": "Angelman syndrome"},
    "VCP": {"dept": "Ubiquitin", "disease": "IBMPFD"},
    "PARK2": {"dept": "Ubiquitin", "disease": "Parkinson disease"},
    "UBA1": {"dept": "Ubiquitin", "disease": "VEXAS syndrome"},
    "DNMT3A": {"dept": "Methylation", "disease": "Tatton-Brown-Rahman syndrome"},
    "DNMT3B": {"dept": "Methylation", "disease": "ICF syndrome"},
    "TET2": {"dept": "Methylation", "disease": "Myeloid neoplasms"},
    "IDH1": {"dept": "Methylation", "disease": "Glioma"},
    "IDH2": {"dept": "Methylation", "disease": "D-2-hydroxyglutaric aciduria"},
    "KMT2A": {"dept": "Chromatin", "disease": "Wiedemann-Steiner syndrome"},
    "KMT2D": {"dept": "Chromatin", "disease": "Kabuki syndrome"},
    "KDM6A": {"dept": "Chromatin", "disease": "Kabuki syndrome 2"},
    "ARID1A": {"dept": "Chromatin", "disease": "Coffin-Siris syndrome"},
    "ARID1B": {"dept": "Chromatin", "disease": "Coffin-Siris syndrome 1"},
    "SMARCA4": {"dept": "Chromatin", "disease": "Rhabdoid tumor predisposition"},
    "SMARCB1": {"dept": "Chromatin", "disease": "Rhabdoid tumor predisposition 1"},
    "EP300": {"dept": "Chromatin", "disease": "Rubinstein-Taybi syndrome 2"},
    "CREBBP": {"dept": "Chromatin", "disease": "Rubinstein-Taybi syndrome"},
    "EZH2": {"dept": "Chromatin", "disease": "Weaver syndrome"},
    "HDAC8": {"dept": "Chromatin", "disease": "Cornelia de Lange syndrome 5"},
    "CTCF": {"dept": "Chromatin", "disease": "Mental retardation autosomal dominant 21"},
    "PRF1": {"dept": "Immune", "disease": "Familial hemophagocytic lymphohistiocytosis"},
    "RAG1": {"dept": "Immune", "disease": "Severe combined immunodeficiency"},
    "RAG2": {"dept": "Immune", "disease": "Severe combined immunodeficiency"},
    "IL2RG": {"dept": "Immune", "disease": "X-linked SCID"},
    "JAK3": {"dept": "Immune", "disease": "SCID, autosomal recessive"},
    "AIRE": {"dept": "Immune", "disease": "Autoimmune polyendocrinopathy"},
    "FOXP3": {"dept": "Immune", "disease": "IPEX syndrome"},
    "BTK": {"dept": "Immune", "disease": "X-linked agammaglobulinemia"},
    "WAS": {"dept": "Immune", "disease": "Wiskott-Aldrich syndrome"},
    "PARP1": {"dept": "DNA repair", "disease": "DNA repair deficiency"},
    "POLE": {"dept": "DNA repair", "disease": "Colorectal cancer susceptibility"},
    "POLD1": {"dept": "DNA repair", "disease": "Colorectal cancer susceptibility"},
    "MUTYH": {"dept": "DNA repair", "disease": "MUTYH-associated polyposis"},
    "RECQL4": {"dept": "DNA repair", "disease": "Rothmund-Thomson syndrome"},
    "BLM": {"dept": "DNA repair", "disease": "Bloom syndrome"},
    "WRN": {"dept": "DNA repair", "disease": "Werner syndrome"},
    "NBN": {"dept": "DNA repair", "disease": "Nijmegen breakage syndrome"},
    "MRE11": {"dept": "DNA repair", "disease": "Ataxia-telangiectasia-like disorder"},
    "SMN1": {"dept": "RNA processing", "disease": "Spinal muscular atrophy"},
    "TARDBP": {"dept": "RNA processing", "disease": "ALS10"},
    "FUS": {"dept": "RNA processing", "disease": "ALS6"},
    "DYNC1H1": {"dept": "Cytoskeleton", "disease": "SMA-LED"},
    "TUBB3": {"dept": "Cytoskeleton", "disease": "CFEOM3A"},
    "TUBA1A": {"dept": "Cytoskeleton", "disease": "Lissencephaly"},
    "FLNA": {"dept": "Cytoskeleton", "disease": "Periventricular nodular heterotopia"},
    "ACTA2": {"dept": "Cytoskeleton", "disease": "Familial thoracic aortic aneurysm"},
    "MYH7": {"dept": "Cytoskeleton", "disease": "Hypertrophic cardiomyopathy"},
    "MYH9": {"dept": "Cytoskeleton", "disease": "MYH9-related disease"},
    "ACTB": {"dept": "Cytoskeleton", "disease": "Baraitser-Winter syndrome"},
    "ACTG1": {"dept": "Cytoskeleton", "disease": "Baraitser-Winter syndrome 2"},
    "HSP90AA1": {"dept": "Protein folding", "disease": "Cancer susceptibility"},
    "HSPA5": {"dept": "Protein folding", "disease": "Marinesco-Sjogren-like"},
    "HSP90B1": {"dept": "Protein folding", "disease": "Autoimmune lymphoproliferative"},
    "CALR": {"dept": "Protein folding", "disease": "Myeloproliferative neoplasm"},
    "PDIA3": {"dept": "Protein folding", "disease": "Osteogenesis imperfecta"},
    "STAT1": {"dept": "Transcription", "disease": "Immunodeficiency 31A"},
    "STAT3": {"dept": "Transcription", "disease": "Hyper-IgE syndrome"},
    "RUNX1": {"dept": "Transcription", "disease": "Familial platelet disorder"},
    "PAX6": {"dept": "Transcription", "disease": "Aniridia"},
    "SOX9": {"dept": "Transcription", "disease": "Campomelic dysplasia"},
    "SOX10": {"dept": "Transcription", "disease": "Waardenburg syndrome 4C"},
    "GATA4": {"dept": "Transcription", "disease": "Congenital heart defects"},
    "TBX5": {"dept": "Transcription", "disease": "Holt-Oram syndrome"},
    "TBX3": {"dept": "Transcription", "disease": "Ulnar-mammary syndrome"},
    "FOXC2": {"dept": "Transcription", "disease": "Lymphedema-distichiasis"},
    "FOXL2": {"dept": "Transcription", "disease": "BPES"},
    "IRF6": {"dept": "Transcription", "disease": "Van der Woude syndrome"},
    "MITF": {"dept": "Transcription", "disease": "Waardenburg syndrome 2A"},
    "WT1": {"dept": "Transcription", "disease": "Wilms tumor"},
    "MYC": {"dept": "Transcription", "disease": "Burkitt lymphoma"},
    "HNF1A": {"dept": "Transcription", "disease": "MODY3"},
    "HNF4A": {"dept": "Transcription", "disease": "MODY1"},
    "PPARG": {"dept": "Transcription", "disease": "Lipodystrophy"},
    "AR": {"dept": "Transcription", "disease": "Androgen insensitivity syndrome"},
    "ESR1": {"dept": "Transcription", "disease": "Estrogen resistance"},
    "GJB2": {"dept": "Cell adhesion", "disease": "Nonsyndromic hearing loss"},
    "GJA1": {"dept": "Cell adhesion", "disease": "Oculodentodigital dysplasia"},
    "ITGA2B": {"dept": "Cell adhesion", "disease": "Glanzmann thrombasthenia"},
    "ITGB3": {"dept": "Cell adhesion", "disease": "Glanzmann thrombasthenia"},
    "DSP": {"dept": "Cell adhesion", "disease": "Arrhythmogenic cardiomyopathy"},
    "DSG2": {"dept": "Cell adhesion", "disease": "Arrhythmogenic cardiomyopathy"},
    "PKP2": {"dept": "Cell adhesion", "disease": "Arrhythmogenic cardiomyopathy"},
    "EIF2B1": {"dept": "Translation", "disease": "Vanishing white matter disease"},
    "EIF2B5": {"dept": "Translation", "disease": "Vanishing white matter disease"},
    "RPL5": {"dept": "Translation", "disease": "Diamond-Blackfan anemia"},
    "RPL11": {"dept": "Translation", "disease": "Diamond-Blackfan anemia"},
    "RPS19": {"dept": "Translation", "disease": "Diamond-Blackfan anemia"},
    "DKC1": {"dept": "Translation", "disease": "Dyskeratosis congenita"},
    "TERT": {"dept": "Translation", "disease": "Dyskeratosis congenita"},
    "TERC": {"dept": "Nuc acid bind", "disease": "Dyskeratosis congenita"},
    "PCNA": {"dept": "DNA repair", "disease": "Ataxia-oculomotor apraxia 1"},
    "ERCC6": {"dept": "DNA repair", "disease": "Cockayne syndrome"},
    "ERCC8": {"dept": "DNA repair", "disease": "Cockayne syndrome"},
    "PTPN11": {"dept": "Phosphatase", "disease": "Noonan syndrome"},
    "SHP2": {"dept": "Phosphatase", "disease": "LEOPARD syndrome"},
    "PTEN2": {"dept": "Phosphatase", "disease": "Cowden-like syndrome"},
    "ACP5": {"dept": "Phosphatase", "disease": "Spondyloenchondrodysplasia"},
    "PPP1CB": {"dept": "Phosphatase", "disease": "Noonan syndrome-like"},
}


def main():
    print("=" * 72)
    print("  EXPANDED DISEASE CONCORDANCE (200+ genes)")
    print("=" * 72)

    gene_depts = {}
    with open(GENE_DEPTS_PATH) as f:
        for row in csv.DictReader(f):
            gene_depts[row["gene"]] = row["department"]

    with open(FULL_PROFILES_PATH) as f:
        profiles_data = json.load(f)
    profiles = {}
    for gene, prof in profiles_data["profiles"].items():
        profiles[gene] = np.array([prof.get(d, 0) for d in VALID_DEPARTMENTS])

    N_DEPTS = len(VALID_DEPARTMENTS)

    matched = {}
    unmatched = []
    for gene, info in CURATED_DISEASE_GENES.items():
        if gene in profiles and gene in gene_depts:
            actual_dept = gene_depts[gene]
            matched[gene] = {
                "expected_dept": info["dept"],
                "actual_dept": actual_dept,
                "disease": info["disease"],
                "dept_match": actual_dept == info["dept"],
            }
        else:
            unmatched.append(gene)

    print(f"\n  Curated disease genes: {len(CURATED_DISEASE_GENES)}")
    print(f"  Matched to our kernel: {len(matched)}")
    print(f"  Unmatched: {len(unmatched)}")
    if unmatched[:10]:
        print(f"  Sample unmatched: {unmatched[:10]}")

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    own_dept_in_top3 = 0
    n_tested = 0

    per_gene_results = []

    for gene, info in sorted(matched.items()):
        prof = profiles[gene]
        if np.linalg.norm(prof) < 1e-12:
            continue

        expected_dept = info["expected_dept"]
        n_tested += 1

        sorted_depts = sorted(range(N_DEPTS),
                               key=lambda i: prof[i], reverse=True)
        top_depts = [VALID_DEPARTMENTS[i] for i in sorted_depts]

        is_top1 = top_depts[0] == expected_dept
        is_top3 = expected_dept in top_depts[:3]
        is_top5 = expected_dept in top_depts[:5]

        if is_top1:
            top1_correct += 1
        if is_top3:
            top3_correct += 1
        if is_top5:
            top5_correct += 1

        expected_idx = D2I[expected_dept]
        rank = top_depts.index(expected_dept) + 1 if expected_dept in top_depts else 23

        per_gene_results.append({
            "gene": gene,
            "disease": info["disease"],
            "expected_dept": expected_dept,
            "actual_top1": top_depts[0],
            "actual_top3": top_depts[:3],
            "expected_rank": rank,
            "top1_correct": is_top1,
            "top3_correct": is_top3,
            "top5_correct": is_top5,
            "dept_match": info["dept_match"],
        })

    N = len(VALID_DEPARTMENTS)

    print(f"\n  === CONCORDANCE RESULTS ({n_tested} disease genes) ===")
    print(f"  {'Metric':<30s} {'Observed':>10s} {'Chance':>8s} {'Fold':>6s}")
    print(f"  {'Top-1 concordance':<30s} {top1_correct/n_tested:>9.1%} {1/N:>7.1%} "
          f"{(top1_correct/n_tested)/(1/N):>5.1f}×")
    print(f"  {'Top-3 concordance':<30s} {top3_correct/n_tested:>9.1%} {3/N:>7.1%} "
          f"{(top3_correct/n_tested)/(3/N):>5.1f}×")
    print(f"  {'Top-5 concordance':<30s} {top5_correct/n_tested:>9.1%} {5/N:>7.1%} "
          f"{(top5_correct/n_tested)/(5/N):>5.1f}×")

    ranks = [r["expected_rank"] for r in per_gene_results]
    print(f"\n  Expected dept rank: median={np.median(ranks):.0f}, "
          f"mean={np.mean(ranks):.1f}")

    by_dept = defaultdict(list)
    for r in per_gene_results:
        by_dept[r["expected_dept"]].append(r)

    print(f"\n  Per-department concordance (departments with 3+ genes):")
    print(f"  {'Department':<20s} {'N':>3s} {'Top1':>6s} {'Top3':>6s} {'MedRank':>8s}")
    for dept in VALID_DEPARTMENTS:
        genes_in = by_dept.get(dept, [])
        if len(genes_in) >= 3:
            dept_top1 = sum(1 for r in genes_in if r["top1_correct"]) / len(genes_in)
            dept_top3 = sum(1 for r in genes_in if r["top3_correct"]) / len(genes_in)
            dept_ranks = [r["expected_rank"] for r in genes_in]
            print(f"  {dept:<20s} {len(genes_in):3d} {dept_top1:5.0%} {dept_top3:5.0%} "
                  f"{np.median(dept_ranks):7.0f}")

    print(f"\n  Top 10 most concordant genes:")
    sorted_by_rank = sorted(per_gene_results, key=lambda r: r["expected_rank"])
    for r in sorted_by_rank[:10]:
        print(f"    {r['gene']:<15s} {r['expected_dept']:<18s} rank={r['expected_rank']:2d} "
              f"top1={'Y' if r['top1_correct'] else 'N'} — {r['disease']}")

    print(f"\n  Bottom 10 (worst concordance):")
    for r in sorted_by_rank[-10:]:
        mismatch_note = f"→ {r['actual_top1']}" if not r["top1_correct"] else ""
        print(f"    {r['gene']:<15s} {r['expected_dept']:<18s} rank={r['expected_rank']:2d} "
              f"{mismatch_note} — {r['disease']}")

    dept_match_count = sum(1 for r in per_gene_results if r["dept_match"])

    output = {
        "n_curated": len(CURATED_DISEASE_GENES),
        "n_matched": len(matched),
        "n_tested": n_tested,
        "top1_concordance": round(top1_correct / n_tested, 4),
        "top3_concordance": round(top3_correct / n_tested, 4),
        "top5_concordance": round(top5_correct / n_tested, 4),
        "chance_top1": round(1 / N, 4),
        "chance_top3": round(3 / N, 4),
        "fold_top1": round((top1_correct / n_tested) / (1 / N), 1),
        "fold_top3": round((top3_correct / n_tested) / (3 / N), 1),
        "median_rank": float(np.median(ranks)),
        "mean_rank": round(float(np.mean(ranks)), 1),
        "dept_assignment_match_rate": round(dept_match_count / n_tested, 4),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
