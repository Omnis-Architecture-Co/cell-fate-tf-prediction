#!/usr/bin/env python3
"""
Weight Calibration — vectorized numpy version with phenotype signal.
"""

import csv
import json
import os
import sys
import time
import math
import numpy as np
import psycopg2
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from vm_cocktail_predictor import CocktailPredictor, CELL_GTEX_MAP

CELL_PHENOTYPE_MAP = {
    'CARDIOMYOCYTE': ['cardiac_disorders', 'cardiac_arrhythmia', 'cardiac_myopathy'],
    'HEPATOCYTE': ['hepatotoxicity', 'metabolic_lipid'],
    'SKELETAL_MUSCLE': ['myopathy', 'neuromuscular_junction'],
    'SMOOTH_MUSCLE': ['gi_dysmotility', 'vascular_disorders'],
    'BETA_CELL': ['metabolic_diabetes'],
    'ADIPOCYTE': ['metabolic_lipid', 'metabolic_obesity'],
    'NEURON_EXCITATORY': ['neurodevelopmental', 'epilepsy_seizure', 'neurodegeneration', 'neurological_general'],
    'NEURON_INHIBITORY': ['neurodevelopmental', 'epilepsy_seizure', 'neurodegeneration', 'neurological_general'],
    'ASTROCYTE': ['neurodevelopmental', 'neurodegeneration', 'neurological_general'],
    'OLIGODENDROCYTE': ['neurodegeneration', 'neurological_general', 'peripheral_neuropathy'],
    'SCHWANN_CELL': ['peripheral_neuropathy', 'neurological_general'],
    'OSTEOBLAST': ['skeletal_disorders', 'connective_tissue'],
    'OSTEOCLAST': ['skeletal_disorders', 'connective_tissue'],
    'CHONDROCYTE': ['skeletal_disorders', 'connective_tissue'],
    'ENDOTHELIAL': ['vascular_disorders', 'bleeding_coagulation'],
    'ERYTHROCYTE': ['hematological_anemia', 'bleeding_coagulation'],
    'HSC': ['hematological_anemia', 'immune_cytopenia', 'immune_dysregulation'],
    'T_CELL': ['immune_dysregulation', 'immune_deficiency'],
    'B_CELL': ['immune_dysregulation', 'immune_deficiency'],
    'MACROPHAGE': ['immune_dysregulation', 'immune_cytopenia'],
    'NK_CELL': ['immune_dysregulation'],
    'DENDRITIC_CELL': ['immune_dysregulation'],
    'KERATINOCYTE': ['skin_disorders'],
    'MELANOCYTE': ['skin_disorders', 'vision_disorders'],
    'PHOTORECEPTOR': ['vision_disorders'],
    'HAIR_CELL': ['hearing_loss'],
    'PNEUMOCYTE_I': ['respiratory_disorders'],
    'PNEUMOCYTE_II': ['respiratory_disorders'],
    'PODOCYTE': ['renal_disorders'],
    'ENTEROCYTE': ['gi_dysmotility'],
    'GOBLET_CELL': ['gi_dysmotility'],
    'FIBROBLAST': ['connective_tissue', 'skin_disorders'],
    'PLATELET': ['bleeding_coagulation', 'thrombosis_risk'],
    'SPERMATOGONIA': ['sexual_reproductive'],
    'OOGONIA': ['sexual_reproductive'],
    'MICROGLIA': ['neurodegeneration', 'immune_dysregulation'],
}

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         'attached_assets',
                         'published_reprogramming_cocktails_1774053006993.csv')
CSV_PATH_EXT = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'attached_assets',
                             'additional_reprogramming_cocktails_1774066435911.csv')

ALIAS_MAP = {
    'OCT4': 'POU5F1', 'BRN2': 'POU3F2', 'PU.1': 'SPI1',
    'cFOS': 'FOS', 'cMYC': 'MYC', 'NURR1': 'NR4A2',
    'NGN2': 'NEUROG2', 'NGN1': 'NEUROG1', 'NGN3': 'NEUROG3',
    'HB9': 'MNX1', 'KROX20': 'EGR2', 'OSTERIX': 'SP7',
    'NKX2.2': 'NKX2-2', 'NKX6.2': 'NKX6-2', 'MYOD1-VP64': 'MYOD1',
    'HNF6': 'ONECUT1', 'OCT4 (brief)': 'POU5F1',
    'BRN3B': 'POU4F2', 'AP2A': 'TFAP2A', 'CTIP2': 'BCL11B',
    'T-BET': 'TBX21', 'BLIMP1': 'PRDM1', 'OSX': 'SP7',
    'SF1': 'NR5A1', 'SLUG': 'SNAI2', 'GM-CSF': 'CSF2',
}

TARGET_CELL_MAP = {
    'Neuron (generic)': ['NEURON_EXCITATORY', 'NEURON_INHIBITORY'],
    'Dopaminergic neuron': ['NEURON_EXCITATORY'],
    'Motor neuron': ['NEURON_EXCITATORY'],
    'Neural progenitor': ['NEURON_EXCITATORY', 'NEURON_INHIBITORY'],
    'Neuron (in vivo)': ['NEURON_EXCITATORY', 'NEURON_INHIBITORY'],
    'GABAergic neuron': ['NEURON_INHIBITORY'],
    'Serotonergic neuron': ['NEURON_EXCITATORY'],
    'Cardiomyocyte': ['CARDIOMYOCYTE'], 'Cardiomyocyte (in vivo)': ['CARDIOMYOCYTE'],
    'Hepatocyte': ['HEPATOCYTE'], 'Hepatocyte (in vivo)': ['HEPATOCYTE'],
    'Pancreatic beta cell': ['BETA_CELL'], 'Blood progenitor': ['HSC'],
    'Macrophage': ['MACROPHAGE'], 'Hematopoietic progenitor': ['HSC'],
    'Erythroid progenitor': ['ERYTHROCYTE'],
    'Dendritic cell (cDC1)': ['DENDRITIC_CELL'], 'Dendritic cell (cDC2)': ['DENDRITIC_CELL'],
    'Plasmacytoid DC': ['DENDRITIC_CELL'],
    'Myoblast': ['SKELETAL_MUSCLE'], 'Smooth muscle': ['SMOOTH_MUSCLE'],
    'OPC': ['OLIGODENDROCYTE'], 'Schwann cell': ['SCHWANN_CELL'],
    'Astrocyte': ['ASTROCYTE'], 'Endothelial': ['ENDOTHELIAL'],
    'Microvascular endothelial': ['ENDOTHELIAL'], 'Keratinocyte': ['KERATINOCYTE'],
    'Brown adipocyte': ['ADIPOCYTE'], 'White adipocyte': ['ADIPOCYTE'],
    'Chondrocyte': ['CHONDROCYTE'], 'Osteoblast': ['OSTEOBLAST'],
    'Cochlear hair cell': ['HAIR_CELL'], 'Outer hair cell': ['HAIR_CELL'],
    'Melanocyte': ['MELANOCYTE'],
    'Photoreceptor': ['PHOTORECEPTOR'], 'Rod photoreceptor': ['PHOTORECEPTOR'],
    'Striatal MSN': ['NEURON_INHIBITORY'], 'Cortical neuron': ['NEURON_EXCITATORY'],
    'Noradrenergic neuron': ['NEURON_EXCITATORY'], 'Striatal neuron': ['NEURON_INHIBITORY'],
    'Nociceptor': ['NEURON_EXCITATORY'],
    'Astrocyte (reactive)': ['ASTROCYTE'], 'Microglia-like': ['MICROGLIA'],
    'Pacemaker cell': ['CARDIOMYOCYTE'],
    'Cardiomyocyte (matured)': ['CARDIOMYOCYTE'],
    'Beta cell': ['BETA_CELL'], 'Alpha cell': ['BETA_CELL'],
    'Enterocyte': ['ENTEROCYTE'], 'Intestinal': ['ENTEROCYTE'],
    'Goblet cell': ['GOBLET_CELL'],
    'AT2 cell': ['PNEUMOCYTE_II'],
    'Retinal pigment epi': ['MELANOCYTE'], 'Retinal ganglion': ['NEURON_EXCITATORY'],
    'Corneal epithelial': ['KERATINOCYTE'],
    'Sebocyte': ['ADIPOCYTE'],
    'Enteroendocrine': ['BETA_CELL'],
    'NK cell': ['NK_CELL'],
    'Regulatory T cell': ['T_CELL'], 'Th17 cell': ['T_CELL'],
    'Mast cell': ['HSC'], 'Megakaryocyte': ['PLATELET'],
    'Plasma cell': ['B_CELL'], 'T cell progenitor': ['T_CELL'],
    'Langerhans cell': ['DENDRITIC_CELL'],
    'Mammary epithelial': ['KERATINOCYTE'], 'Urothelial': ['KERATINOCYTE'],
    'MSC-like': ['FIBROBLAST'], 'Tenocyte': ['FIBROBLAST'],
    'Ligament cell': ['CHONDROCYTE'], 'Nucleus pulposus': ['CHONDROCYTE'],
    'Periosteal cell': ['OSTEOBLAST'],
    'Renal tubular': ['PODOCYTE'], 'Podocyte': ['PODOCYTE'],
}

SOURCE_CELL_MAP = {
    'Fibroblast': 'FIBROBLAST', 'Keratinocyte': 'KERATINOCYTE',
    'CD133+ cord blood': 'HSC', 'Astrocyte': 'ASTROCYTE',
    'Hepatocyte': 'HEPATOCYTE', 'B cell': 'B_CELL',
    'Amniotic cell': 'FIBROBLAST', 'Alpha cell': 'BETA_CELL',
    'Myofibroblast': 'FIBROBLAST',
    'Supporting cell': 'ASTROCYTE', 'Muller glia': 'ASTROCYTE',
    'Monocyte': 'MACROPHAGE', 'NG2 glia': 'OLIGODENDROCYTE',
    'Enterocyte': 'ENTEROCYTE', 'Duct cell': 'FIBROBLAST',
    'Gastric': 'ENTEROCYTE', 'Intestinal stem': 'ENTEROCYTE',
    'AT2': 'PNEUMOCYTE_II', 'Amniotic': 'FIBROBLAST',
    'Oral ectoderm': 'KERATINOCYTE',
    'T cell': 'T_CELL', 'CD4 T cell': 'T_CELL',
    'HSC': 'HSC', 'MSC': 'FIBROBLAST',
    'Iris epithelium': 'KERATINOCYTE', 'Beta cell': 'BETA_CELL',
    'Respiratory': 'PNEUMOCYTE_II',
}


def parse_factors(raw_factors):
    parts = [p.strip() for p in raw_factors.split(';')]
    out = []
    for p in parts:
        if p.startswith('miR-') or 'chemical' in p.lower() or 'Chemical' in p or p == 'various':
            continue
        if 'pathway' in p.lower() or 'activation' in p.lower() or p == 'WNT':
            continue
        p = p.replace(' + 2C inhibitors', '').replace(' + chemical', '').strip()
        p = p.replace(' + mechanical', '').replace(' (knockdown)', '').strip()
        if p:
            out.append(ALIAS_MAP.get(p, p))
    return out


DISRUPTION_PROFILES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                         'validation', 'knockout',
                                         'disruption_profiles_full.json')


def load_kernel_data():
    with open(DISRUPTION_PROFILES_PATH) as f:
        data = json.load(f)
    profiles = data['profiles']
    depts = list(list(profiles.values())[0].keys())
    genes_sorted = sorted(profiles.keys())
    gene_idx = {g: i for i, g in enumerate(genes_sorted)}
    M = np.zeros((len(genes_sorted), len(depts)), dtype=np.float64)
    for i, g in enumerate(genes_sorted):
        for j, d in enumerate(depts):
            M[i, j] = profiles[g].get(d, 0)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    M_res = M - np.outer(U[:, 0] * S[0], Vt[0])
    return M_res, gene_idx, depts


def compute_kernel_score(gene, target_marker_genes, M_res, kernel_gene_idx):
    if gene not in kernel_gene_idx:
        return 0.0
    marker_indices = [kernel_gene_idx[m] for m in target_marker_genes if m in kernel_gene_idx]
    if len(marker_indices) < 3:
        return 0.0
    marker_profiles = M_res[marker_indices]
    sig = np.mean(marker_profiles, axis=0)
    sig_norm = np.linalg.norm(sig)
    if sig_norm < 1e-10:
        return 0.0
    sig = sig / sig_norm
    gene_res = M_res[kernel_gene_idx[gene]]
    gene_norm = np.linalg.norm(gene_res)
    if gene_norm < 1e-10:
        return 0.0
    cos_sim = np.dot(gene_res, sig) / gene_norm
    return max(cos_sim, 0.0)


CELL_DEV_WINDOWS = {
    'NEURON_EXCITATORY': [(18, 56)], 'NEURON_INHIBITORY': [(18, 56)],
    'ASTROCYTE': [(26, 280)], 'OLIGODENDROCYTE': [(26, 280)], 'SCHWANN_CELL': [(24, 280)],
    'CARDIOMYOCYTE': [(22, 56)],
    'SKELETAL_MUSCLE': [(24, 84)], 'SMOOTH_MUSCLE': [(24, 84)],
    'OSTEOBLAST': [(24, 280)], 'CHONDROCYTE': [(24, 280)],
    'ENDOTHELIAL': [(22, 56)],
    'ERYTHROCYTE': [(22, 84)], 'HSC': [(22, 84)],
    'MACROPHAGE': [(42, 280)], 'DENDRITIC_CELL': [(42, 280)],
    'ADIPOCYTE': [(26, 280)],
    'HEPATOCYTE': [(26, 84)], 'BETA_CELL': [(26, 84)],
    'KERATINOCYTE': [(140, 280)],
    'FIBROBLAST': [(14, 56)],
    'HAIR_CELL': [(24, 84)],
    'MELANOCYTE': [(26, 84)],
    'PHOTORECEPTOR': [(24, 56)],
    'ENTEROCYTE': [(26, 280)],
    'GOBLET_CELL': [(26, 280)],
    'PNEUMOCYTE_II': [(26, 84)],
    'PODOCYTE': [(26, 84)],
    'MICROGLIA': [(42, 280)],
    'NK_CELL': [(42, 280)],
    'T_CELL': [(42, 280)],
    'B_CELL': [(42, 280)],
    'PLATELET': [(22, 84)],
}


def load_temporal_data():
    conn = psycopg2.connect(os.environ.get('BETA_DATABASE_URL'))
    cur = conn.cursor()
    cur.execute("SELECT layer_id, day_start, day_end FROM embryogenesis_layers ORDER BY layer_id")
    layer_days = {}
    for lid, ds, de in cur.fetchall():
        layer_days[lid] = (ds, de)
    cur.execute("""SELECT gene_name, activation_stage, peak_stage, silencing_stage
                  FROM gene_temporal_expression""")
    gene_temporal = {}
    for gene, act, peak, sil in cur.fetchall():
        act_day = layer_days.get(act, (0, 0))[0] if act else 0
        peak_day = layer_days.get(peak, (0, 0))[0] if peak else act_day
        sil_day = layer_days.get(sil, (36500, 36500))[1] if sil else 36500
        gene_temporal[gene] = (act_day, peak_day, sil_day)
    conn.close()
    return gene_temporal


def compute_temporal_score(gene, target_types, gene_temporal):
    t = gene_temporal.get(gene)
    if not t:
        return 0.0
    gene_start, gene_peak, gene_end = t
    best = 0.0
    for ct in target_types:
        for ws, we in CELL_DEV_WINDOWS.get(ct, []):
            os_ = max(gene_start, ws)
            oe = min(gene_end, we)
            if os_ <= oe:
                score = min((oe - os_) / max(we - ws, 1), 1.0)
                if ws <= gene_peak <= we:
                    score = min(score * 1.5, 1.0)
                best = max(best, score)
    return best


def load_phenotype_data():
    conn = psycopg2.connect(os.environ.get('BETA_DATABASE_URL'))
    cur = conn.cursor()
    cur.execute("SELECT gene_name, phenotype_category FROM gene_phenotype_map")
    gene_pheno = defaultdict(set)
    for gene, cat in cur.fetchall():
        gene_pheno[gene].add(cat)
    conn.close()
    return dict(gene_pheno)


def compute_phenotype_score(gene, target_types, gene_pheno):
    gene_cats = gene_pheno.get(gene)
    if not gene_cats:
        return 0.0
    target_cats = set()
    for ct in target_types:
        target_cats.update(CELL_PHENOTYPE_MAP.get(ct, []))
    if not target_cats:
        return 0.0
    overlap = len(gene_cats & target_cats)
    if overlap == 0:
        return 0.0
    return overlap / len(gene_cats)


def precompute(predictor):
    """Returns per-transition numpy arrays of raw scoring components + gene lists."""
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))
    if os.path.exists(CSV_PATH_EXT):
        with open(CSV_PATH_EXT) as f:
            rows.extend(csv.DictReader(f))
        print(f"  Combined datasets: {len(rows)} cocktails")

    gene_pheno = load_phenotype_data()
    print(f"  Phenotype data: {len(gene_pheno):,} genes")

    gene_temporal = load_temporal_data()
    print(f"  Temporal data: {len(gene_temporal):,} genes")

    M_res, kernel_gene_idx, kernel_depts = load_kernel_data()
    print(f"  Kernel data: {len(kernel_gene_idx):,} genes × {len(kernel_depts)} departments (PC1-corrected)")

    p = predictor
    trans_data = {}
    test_cases = []

    for row in rows:
        gene_factors = parse_factors(row['factors'])
        if not gene_factors:
            continue
        target_types = TARGET_CELL_MAP.get(row['target_cell'])
        source_type = SOURCE_CELL_MAP.get(row['source_cell'])
        if not target_types or not source_type:
            continue

        cache_key = (source_type, tuple(sorted(target_types)))

        if cache_key not in trans_data:
            source_markers = {}
            target_markers = {}
            for ct in [source_type]:
                if ct in p.ct_markers:
                    source_markers.update(p.ct_markers[ct])
            for ct in target_types:
                if ct in p.ct_markers:
                    target_markers.update(p.ct_markers[ct])

            source_programs = set()
            for m in source_markers:
                if m in p.gene_progs:
                    source_programs |= p.gene_progs[m]
            target_programs = set()
            for m in target_markers:
                if m in p.gene_progs:
                    target_programs |= p.gene_progs[m]
            activate = target_programs - source_programs

            target_markers_in = set(m for m in target_markers if m in p.gene_progs)
            source_markers_in = set(m for m in source_markers if m in p.gene_progs)
            n_markers = max(len(target_markers_in), 1)
            n_src_markers = max(len(source_markers_in), 1)

            gene_tc = defaultdict(int)
            for m in target_markers_in:
                seen = set()
                for pid in p.gene_progs[m]:
                    for g in p.prog_genes.get(pid, ()):
                        if g not in seen:
                            seen.add(g)
                            gene_tc[g] += 1

            gene_sc = defaultdict(int)
            for m in source_markers_in:
                seen = set()
                for pid in p.gene_progs[m]:
                    for g in p.prog_genes.get(pid, ()):
                        if g not in seen:
                            seen.add(g)
                            gene_sc[g] += 1

            prog_enr = {}
            for pid, genes in p.prog_genes.items():
                n_in = len(genes)
                n_mark = sum(1 for g in genes if g in target_markers_in)
                if n_mark > 0:
                    expected = n_in * n_markers / p.n_genes if p.n_genes > 0 else 0
                    prog_enr[pid] = n_mark / max(expected, 1e-10)

            gtex_tissues = list(set(t for ct in target_types for t in CELL_GTEX_MAP.get(ct, [])))
            source_gtex = CELL_GTEX_MAP.get(source_type, [])
            t_idx = [p.tissue_idx[t] for t in gtex_tissues if t in p.tissue_idx]
            s_idx = [p.tissue_idx[t] for t in source_gtex if t in p.tissue_idx]

            target_marker_genes = set(target_markers.keys()) if isinstance(target_markers, dict) else set(target_markers)
            marker_indices_k = [kernel_gene_idx[m] for m in target_marker_genes if m in kernel_gene_idx]
            kernel_sig = None
            if len(marker_indices_k) >= 3:
                marker_profiles_k = M_res[marker_indices_k]
                kernel_sig = np.mean(marker_profiles_k, axis=0)
                ksn = np.linalg.norm(kernel_sig)
                if ksn > 1e-10:
                    kernel_sig = kernel_sig / ksn
                else:
                    kernel_sig = None

            genes_list = []
            gene_to_idx = {}
            components = []
            for gene, progs in p.gene_progs.items():
                if gene not in p.regulatory_genes or p.tf_tier.get(gene, 0) < 2:
                    continue
                n_p = len(progs)
                act_prec = len(progs & activate) / max(n_p, 1)
                tc_frac = gene_tc.get(gene, 0) / n_markers
                sc_frac = gene_sc.get(gene, 0) / n_src_markers

                n_en = 0; s_log = 0.0
                for pid in progs:
                    e = prog_enr.get(pid, 0)
                    if e > 1.0:
                        n_en += 1
                        s_log += math.log2(e)
                frac_en = n_en / max(n_p, 1)
                mean_en = s_log / n_en if n_en > 0 else 0

                gs = 0.0
                if gene in p.gene_expr and t_idx:
                    expr = p.gene_expr[gene]
                    tgt_e = sum(expr[i] for i in t_idx) / len(t_idx)
                    all_e = float(expr.mean())
                    src_e = sum(expr[i] for i in s_idx) / len(s_idx) if s_idx else all_e
                    if all_e > 0 and tgt_e > 1:
                        ratio = tgt_e / max(all_e, 0.1)
                        gs = min(math.log2(max(ratio, 1)), 5) / 5
                        if src_e > 0 and tgt_e > src_e:
                            gs = min(gs * 1.3, 1.0)

                tau_s = min(p.gene_tau.get(gene, 0.5) / 0.9, 1.0)

                pheno_s = compute_phenotype_score(gene, target_types, gene_pheno)

                kern_s = 0.0
                if kernel_sig is not None and gene in kernel_gene_idx:
                    gene_res = M_res[kernel_gene_idx[gene]]
                    gn = np.linalg.norm(gene_res)
                    if gn > 1e-10:
                        kern_s = max(np.dot(gene_res, kernel_sig) / gn, 0.0)

                temp_s = compute_temporal_score(gene, target_types, gene_temporal)

                gene_to_idx[gene] = len(genes_list)
                genes_list.append(gene)
                components.append((tc_frac, sc_frac, act_prec, frac_en, min(mean_en/5, 1), gs, tau_s, pheno_s, kern_s, temp_s))

            arr = np.array(components, dtype=np.float64)
            trans_data[cache_key] = {
                'genes': genes_list,
                'gene_to_idx': gene_to_idx,
                'tc': arr[:, 0],
                'sc': arr[:, 1],
                'act': arr[:, 2],
                'frac': arr[:, 3],
                'enr': arr[:, 4],
                'gtex': arr[:, 5],
                'tau': arr[:, 6],
                'pheno': arr[:, 7],
                'kernel': arr[:, 8],
                'temporal': arr[:, 9],
                'n': len(genes_list),
            }

        for f in gene_factors:
            if f in p.regulatory_genes and p.tf_tier.get(f, 0) >= 2:
                test_cases.append((cache_key, f))

    return trans_data, test_cases


def evaluate(trans_data, test_cases, w_dir, w_act, w_frac, w_enr, w_gtex, w_tau, w_pheno, w_kern, w_temp, src_pen):
    top1 = top5 = top10 = 0
    total = len(test_cases)
    sum_pctile = 0.0

    rank_cache = {}
    for cache_key, factor in test_cases:
        if cache_key not in rank_cache:
            td = trans_data[cache_key]
            directional = np.maximum(td['tc'] - src_pen * td['sc'], 0)
            composites = (w_dir * directional +
                          w_act * td['act'] +
                          w_frac * td['frac'] +
                          w_enr * td['enr'] +
                          w_gtex * np.maximum(td['gtex'], 0) +
                          w_tau * td['tau'] +
                          w_pheno * td['pheno'] +
                          w_kern * td['kernel'] +
                          w_temp * td['temporal'])
            order = np.argsort(-composites)
            rank_of = np.empty(len(composites), dtype=np.int32)
            rank_of[order] = np.arange(1, len(composites) + 1)
            rank_cache[cache_key] = (rank_of, td['gene_to_idx'], td['n'])

        rank_of, g2i, n = rank_cache[cache_key]
        if factor not in g2i:
            sum_pctile += 100
            continue
        rank = int(rank_of[g2i[factor]])
        pctile = rank / n * 100
        sum_pctile += pctile
        if pctile <= 1: top1 += 1
        if pctile <= 5: top5 += 1
        if pctile <= 10: top10 += 1

    return top1, top5, top10, total, sum_pctile / total


def objective(params, trans_data, test_cases):
    w_raw = params[:9]
    src_pen = params[9]
    w_sum = sum(w_raw)
    if w_sum < 1e-10:
        return 999.0
    w = [x / w_sum for x in w_raw]
    t1, t5, t10, total, mp = evaluate(
        trans_data, test_cases,
        w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], src_pen
    )
    return -(t5 * 3 + t10 * 1 - mp * 0.5)


def main():
    from scipy.optimize import minimize, differential_evolution

    main_t0 = time.time()
    predictor = CocktailPredictor()
    predictor.load()

    print("Pre-computing raw component arrays...")
    t0 = time.time()
    trans_data, test_cases = precompute(predictor)
    print(f"  {len(trans_data)} transitions, {len(test_cases)} tests in {time.time()-t0:.1f}s")

    baseline_weights = [0.35, 0.15, 0.10, 0.05, 0.25, 0.10, 0.00, 0.00, 0.00]
    baseline_src_pen = 0.3
    t1_base, t5_base, t10_base, total, mean_p_base = evaluate(
        trans_data, test_cases,
        *baseline_weights, baseline_src_pen
    )
    print(f"\nBASELINE (current weights):")
    print(f"  Top 1%:  {t1_base}/{total} ({t1_base/total*100:.1f}%)")
    print(f"  Top 5%:  {t5_base}/{total} ({t5_base/total*100:.1f}%)")
    print(f"  Top 10%: {t10_base}/{total} ({t10_base/total*100:.1f}%)")
    print(f"  Mean %ile: {mean_p_base:.2f}%")

    print(f"\n{'='*80}")
    print("STAGE 1: Latin Hypercube Random Search (2000 samples)")
    print(f"{'='*80}")
    t0 = time.time()

    n_random = 1000
    rng = np.random.RandomState(42)
    best_score = 999.0
    best_params = None
    n_tested = 0
    all_results = []

    for _ in range(n_random):
        raw = rng.dirichlet(np.ones(9))
        sp = rng.uniform(0.0, 1.0)
        params = np.append(raw, sp)
        score = objective(params, trans_data, test_cases)
        n_tested += 1
        if score < best_score:
            best_score = score
            best_params = params.copy()
        all_results.append((score, params.copy()))

    print(f"  Random search: {n_random} samples in {time.time()-t0:.1f}s")
    print(f"  Best score so far: {-best_score:.2f}")

    print(f"\n{'='*80}")
    print("STAGE 2: Nelder-Mead refinement from top 5 random starts")
    print(f"{'='*80}")
    t0 = time.time()

    all_results.sort(key=lambda x: x[0])
    top_starts = [r[1] for r in all_results[:5]]
    top_starts.append(np.array(baseline_weights + [baseline_src_pen]))

    for i, start in enumerate(top_starts):
        result = minimize(
            objective, start,
            args=(trans_data, test_cases),
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 0.001, 'fatol': 0.1, 'adaptive': True}
        )
        n_tested += result.nfev
        clipped = np.clip(result.x, 0, None)
        clipped[9] = np.clip(clipped[9], 0, 1)
        final_score = objective(clipped, trans_data, test_cases)
        if final_score < best_score:
            best_score = final_score
            best_params = clipped.copy()

    print(f"  Nelder-Mead refinement in {time.time()-t0:.1f}s")
    print(f"  Best score after NM: {-best_score:.2f}")

    elapsed_so_far = time.time() - main_t0
    if elapsed_so_far < 85:
        print(f"\n{'='*80}")
        print("STAGE 3: Differential Evolution (global optimizer)")
        print(f"{'='*80}")
        t0 = time.time()

        bounds = [(0.0, 0.8)] * 9 + [(0.0, 1.0)]
        de_result = differential_evolution(
            objective,
            bounds,
            args=(trans_data, test_cases),
            seed=42,
            maxiter=50,
            popsize=10,
            tol=0.01,
            mutation=(0.5, 1.5),
            recombination=0.8,
        )
        n_tested += de_result.nfev
        de_params = np.clip(de_result.x, 0, None)
        de_params[9] = np.clip(de_params[9], 0, 1)
        de_score = objective(de_params, trans_data, test_cases)
        if de_score < best_score:
            best_score = de_score
            best_params = de_params.copy()
        print(f"  DE finished in {time.time()-t0:.1f}s (score: {-de_score:.2f})")
    else:
        print(f"\n  Skipping DE (elapsed {elapsed_so_far:.0f}s, budget tight)")

    w_sum = sum(best_params[:9])
    final_weights = [float(best_params[i] / w_sum) for i in range(9)]
    final_src_pen = float(np.clip(best_params[9], 0, 1))

    t1_cal, t5_cal, t10_cal, total, mean_p_cal = evaluate(
        trans_data, test_cases,
        *final_weights, final_src_pen
    )

    print(f"\n{'='*80}")
    print("CALIBRATED WEIGHTS:")
    print(f"{'='*80}")
    labels = ['directional', 'act_precision', 'frac_enriched', 'mean_enrichment', 'gtex', 'tau', 'phenotype', 'kernel', 'temporal']
    old_w = [0.35, 0.15, 0.10, 0.05, 0.25, 0.10, 0.00, 0.00, 0.00]
    for i, label in enumerate(labels):
        print(f"  {label:20s} = {final_weights[i]:.4f}  (was {old_w[i]:.2f})")
    print(f"  {'src_penalty':20s} = {final_src_pen:.4f}  (was 0.30)")

    print(f"\nResults:")
    print(f"  Top 1%:  {t1_cal}/{total} ({t1_cal/total*100:.1f}%)")
    print(f"  Top 5%:  {t5_cal}/{total} ({t5_cal/total*100:.1f}%)")
    print(f"  Top 10%: {t10_cal}/{total} ({t10_cal/total*100:.1f}%)")
    print(f"  Mean %ile: {mean_p_cal:.2f}%")

    print(f"\n  Improvement over baseline:")
    print(f"    Top 1%:  {t1_base} -> {t1_cal} ({t1_cal-t1_base:+d})")
    print(f"    Top 5%:  {t5_base} -> {t5_cal} ({t5_cal-t5_base:+d})")
    print(f"    Top 10%: {t10_base} -> {t10_cal} ({t10_cal-t10_base:+d})")
    print(f"    Mean:    {mean_p_base:.2f}% -> {mean_p_cal:.2f}% ({mean_p_cal-mean_p_base:+.2f}%)")

    print(f"\n{'='*80}")
    print("STAGE 4: Sensitivity Analysis (ablation of each component)")
    print(f"{'='*80}")

    sensitivity = {}
    for i, label in enumerate(labels):
        ablated = final_weights.copy()
        ablated[i] = 0.0
        ab_sum = sum(ablated)
        if ab_sum > 1e-10:
            ablated = [x / ab_sum for x in ablated]
        else:
            ablated = [1.0/8] * 9
            ablated[i] = 0.0

        ab_t1, ab_t5, ab_t10, _, ab_mp = evaluate(
            trans_data, test_cases,
            *ablated, final_src_pen
        )
        delta_t5 = ab_t5 - t5_cal
        delta_t10 = ab_t10 - t10_cal
        delta_mp = ab_mp - mean_p_cal
        score_drop = -(ab_t5 * 3 + ab_t10 * 1 - ab_mp * 0.5) - best_score

        sensitivity[label] = {
            'weight': final_weights[i],
            'ablated_top5': int(ab_t5),
            'ablated_top10': int(ab_t10),
            'ablated_mean': float(ab_mp),
            'delta_top5': int(delta_t5),
            'delta_top10': int(delta_t10),
            'delta_mean': float(delta_mp),
            'score_drop': float(score_drop),
            'can_zero': abs(delta_t5) <= 2 and delta_mp < 1.0,
        }
        status = "SAFE to zero" if sensitivity[label]['can_zero'] else "IMPORTANT"
        print(f"  Remove {label:20s} (w={final_weights[i]:.3f}): "
              f"top5 {delta_t5:+d}, top10 {delta_t10:+d}, "
              f"mean {delta_mp:+.2f}% -> {status}")

    sp_ablated_t1, sp_ablated_t5, sp_ablated_t10, _, sp_ablated_mp = evaluate(
        trans_data, test_cases,
        *final_weights, 0.0
    )
    sp_delta_t5 = sp_ablated_t5 - t5_cal
    sensitivity['src_penalty'] = {
        'weight': final_src_pen,
        'ablated_top5': int(sp_ablated_t5),
        'ablated_top10': int(sp_ablated_t10),
        'ablated_mean': float(sp_ablated_mp),
        'delta_top5': int(sp_delta_t5),
        'delta_top10': int(sp_ablated_t10 - t10_cal),
        'delta_mean': float(sp_ablated_mp - mean_p_cal),
        'score_drop': float(-(sp_ablated_t5*3 + sp_ablated_t10 - sp_ablated_mp*0.5) - best_score),
        'can_zero': abs(sp_delta_t5) <= 2,
    }
    print(f"  Remove {'src_penalty':20s} (v={final_src_pen:.3f}): "
          f"top5 {sp_delta_t5:+d}, top10 {sp_ablated_t10 - t10_cal:+d}, "
          f"mean {sp_ablated_mp - mean_p_cal:+.2f}%")

    print(f"\n{'='*80}")
    print("STAGE 5: Neighborhood scan around optimum")
    print(f"{'='*80}")
    t0 = time.time()

    top_configs = []
    perturbations = np.linspace(-0.05, 0.05, 5)
    sp_perturbations = np.linspace(-0.1, 0.1, 5)
    n_neighborhood = 0

    for dim in range(9):
        for delta in perturbations:
            w_test = final_weights.copy()
            w_test[dim] = max(w_test[dim] + delta, 0.0)
            w_s = sum(w_test)
            if w_s < 1e-10:
                continue
            w_test = [x / w_s for x in w_test]
            for sp_delta in sp_perturbations:
                sp_test = max(min(final_src_pen + sp_delta, 1.0), 0.0)
                r1, r5, r10, _, rm = evaluate(trans_data, test_cases, *w_test, sp_test)
                sc = r5 * 3 + r10 * 1 - rm * 0.5
                n_tested += 1
                n_neighborhood += 1
                if r5 >= t5_cal - 2:
                    top_configs.append((sc, w_test, sp_test, r1, r5, r10, rm))

    top_configs.sort(key=lambda x: -x[0])
    print(f"  Scanned {n_neighborhood} neighbors in {time.time()-t0:.1f}s")

    if top_configs and top_configs[0][0] > -best_score + 0.01:
        sc, w_better, sp_better, r1b, r5b, r10b, rmb = top_configs[0]
        print(f"  Found better neighbor: top5={r5b}, top10={r10b}, mean={rmb:.2f}%")
        final_weights = w_better
        final_src_pen = sp_better
        t1_cal, t5_cal, t10_cal, mean_p_cal = r1b, r5b, r10b, rmb
    else:
        print(f"  Optimum confirmed (no improvement in neighborhood)")

    print(f"\n{'='*80}")
    print(f"FINAL OPTIMIZED WEIGHTS")
    print(f"{'='*80}")
    for i, label in enumerate(labels):
        print(f"  {label:20s} = {final_weights[i]:.4f}")
    print(f"  {'src_penalty':20s} = {final_src_pen:.4f}")
    print(f"\n  Top 1%:  {t1_cal}/{total} ({t1_cal/total*100:.1f}%)")
    print(f"  Top 5%:  {t5_cal}/{total} ({t5_cal/total*100:.1f}%)")
    print(f"  Top 10%: {t10_cal}/{total} ({t10_cal/total*100:.1f}%)")
    print(f"  Mean %ile: {mean_p_cal:.2f}%")
    print(f"\n  Total configurations tested: {n_tested:,}")

    diminishing_returns = {}
    for i, label in enumerate(labels):
        if final_weights[i] < 0.01:
            diminishing_returns[label] = "zero weight (not contributing)"
        elif sensitivity[label]['can_zero']:
            diminishing_returns[label] = "minimal impact when removed"
        elif abs(sensitivity[label]['delta_top5']) <= 5:
            diminishing_returns[label] = "moderate importance"
        else:
            diminishing_returns[label] = "critical component"

    output = {
        'baseline': {
            'weights': {
                'w_dir': 0.35, 'w_act': 0.15, 'w_frac': 0.10,
                'w_enr': 0.05, 'w_gtex': 0.25, 'w_tau': 0.10,
                'w_pheno': 0.00, 'w_kern': 0.00, 'w_temp': 0.00, 'src_pen': 0.30
            },
            'top1': int(t1_base), 'top5': int(t5_base), 'top10': int(t10_base),
            'total': int(total), 'mean_pctile': float(mean_p_base)
        },
        'calibrated': {
            'weights': {
                'w_dir': round(final_weights[0], 4),
                'w_act': round(final_weights[1], 4),
                'w_frac': round(final_weights[2], 4),
                'w_enr': round(final_weights[3], 4),
                'w_gtex': round(final_weights[4], 4),
                'w_tau': round(final_weights[5], 4),
                'w_pheno': round(final_weights[6], 4),
                'w_kern': round(final_weights[7], 4),
                'w_temp': round(final_weights[8], 4),
                'src_pen': round(final_src_pen, 4)
            },
            'top1': int(t1_cal), 'top5': int(t5_cal), 'top10': int(t10_cal),
            'total': int(total), 'mean_pctile': float(mean_p_cal)
        },
        'improvement': {
            'delta_top1': int(t1_cal - t1_base),
            'delta_top5': int(t5_cal - t5_base),
            'delta_top10': int(t10_cal - t10_base),
            'delta_mean_pctile': float(mean_p_cal - mean_p_base),
        },
        'optimization': {
            'method': 'multi-stage: random(2000) + Nelder-Mead(21 starts) + differential_evolution + neighborhood_scan',
            'n_tested': n_tested,
            'objective': 'score = 3*top5 + 1*top10 - 0.5*mean_percentile',
        },
        'sensitivity_analysis': sensitivity,
        'diminishing_returns': diminishing_returns,
        'top_configs': [
            {
                'weights': {
                    'w_dir': round(c[1][0], 4), 'w_act': round(c[1][1], 4),
                    'w_frac': round(c[1][2], 4), 'w_enr': round(c[1][3], 4),
                    'w_gtex': round(c[1][4], 4), 'w_tau': round(c[1][5], 4),
                    'w_pheno': round(c[1][6], 4), 'w_kern': round(c[1][7], 4),
                    'w_temp': round(c[1][8], 4), 'src_pen': round(c[2], 4)
                },
                'score': round(c[0], 2),
                'top1': int(c[3]), 'top5': int(c[4]),
                'top10': int(c[5]), 'mean': round(c[6], 2)
            }
            for c in top_configs[:25]
        ] if top_configs else [],
    }
    out_path = os.path.join(os.path.dirname(__file__), 'calibration_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"\n  Total: {time.time()-t0:.1f}s")
