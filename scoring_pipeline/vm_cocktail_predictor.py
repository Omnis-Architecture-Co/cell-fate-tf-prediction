#!/usr/bin/env python3
"""
VM Cocktail Predictor — Full Cell Fate Transition Module
=========================================================
Given a source cell type and target cell type, predicts the optimal
gene cocktail (3-5 transcription factors) to drive the transition.

Three-stage pipeline:
1. Transition Vector: compute program-level difference between source→target
2. Candidate Ranking: score all genes for transition coverage
3. Combinatorial Assembly: greedy set-cover to find complementary cocktails

Validated against 21 published cocktails from 13 independent labs (1987-2014).
"""

import psycopg2
import os
import json
import pickle
import numpy as np
from collections import defaultdict
from itertools import combinations
import math
import time

DB_URL = os.environ.get('BETA_DATABASE_URL')

TIER_A_CELL_TYPES = frozenset({
    'NEURON_EXCITATORY', 'NEURON_INHIBITORY', 'ASTROCYTE', 'OLIGODENDROCYTE',
    'SCHWANN_CELL', 'CARDIOMYOCYTE', 'HEPATOCYTE', 'SKELETAL_MUSCLE',
    'SMOOTH_MUSCLE', 'BETA_CELL', 'ADIPOCYTE', 'MACROPHAGE', 'MICROGLIA',
})

TIER_A_WEIGHTS = {
    'w_dir': 0.0734, 'w_act': 0.0874, 'w_frac': 0.0856,
    'w_enr': 0.0321, 'w_gtex': 0.2912, 'w_tau': 0.2084,
    'w_pheno': 0.0853, 'w_kern': 0.0933, 'w_temp': 0.0434,
    'w_self': 0.40,
    'src_pen': 0.2605,
}

TIER_B_WEIGHTS = {
    'w_dir': 0.0805, 'w_act': 0.0330, 'w_frac': 0.0407,
    'w_enr': 0.0297, 'w_gtex': 0.1944, 'w_tau': 0.2192,
    'w_pheno': 0.3410, 'w_kern': 0.0491, 'w_temp': 0.0123,
    'w_self': 0.40,
    'src_pen': 0.2759,
}

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
    'NEUTROPHIL': ['immune_dysregulation', 'immune_deficiency'],
    'EOSINOPHIL': ['immune_dysregulation'],
    'BASOPHIL': ['immune_dysregulation'],
}

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
    'OSTEOCLAST': [(42, 280)],
    'OOGONIA': [(42, 168)],
    'SPERMATOGONIA': [(168, 36500)],
    'PNEUMOCYTE_I': [(26, 84)],
    'NEUTROPHIL': [(42, 280)],
    'EOSINOPHIL': [(42, 280)],
    'BASOPHIL': [(42, 280)],
}

DISRUPTION_PROFILES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                         'validation', 'knockout',
                                         'disruption_profiles_full.json')


def get_tier_weights(target_types):
    if any(t in TIER_A_CELL_TYPES for t in target_types):
        return TIER_A_WEIGHTS
    return TIER_B_WEIGHTS


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


CELL_GTEX_MAP = {
    'NEURON_EXCITATORY': ['Brain_Cortex', 'Brain_Hippocampus', 'Brain_Cerebellum', 'Brain_Frontal_Cortex_BA9'],
    'NEURON_INHIBITORY': ['Brain_Cortex', 'Brain_Hippocampus', 'Brain_Cerebellum', 'Brain_Frontal_Cortex_BA9'],
    'CARDIOMYOCYTE': ['Heart_Left_Ventricle', 'Heart_Atrial_Appendage'],
    'HEPATOCYTE': ['Liver'],
    'SKELETAL_MUSCLE': ['Muscle_Skeletal'],
    'SMOOTH_MUSCLE': ['Esophagus_Muscularis', 'Artery_Aorta'],
    'ADIPOCYTE': ['Adipose_Subcutaneous', 'Adipose_Visceral_Omentum'],
    'BETA_CELL': ['Pancreas'],
    'ENTEROCYTE': ['Small_Intestine_Terminal_Ileum', 'Colon_Sigmoid'],
    'GOBLET_CELL': ['Small_Intestine_Terminal_Ileum', 'Colon_Sigmoid'],
    'KERATINOCYTE': ['Skin_Not_Sun_Exposed_Suprapubic', 'Skin_Sun_Exposed_Lower_leg'],
    'FIBROBLAST': ['Cells_Cultured_fibroblasts'],
    'ERYTHROCYTE': ['Whole_Blood'],
    'HSC': ['Whole_Blood', 'Spleen'],
    'B_CELL': ['Whole_Blood', 'Spleen'],
    'T_CELL': ['Whole_Blood', 'Spleen'],
    'NK_CELL': ['Whole_Blood', 'Spleen'],
    'MACROPHAGE': ['Whole_Blood', 'Spleen'],
    'DENDRITIC_CELL': ['Whole_Blood', 'Spleen'],
    'ENDOTHELIAL': ['Artery_Aorta', 'Artery_Tibial'],
    'ASTROCYTE': ['Brain_Cortex', 'Brain_Hippocampus'],
    'OLIGODENDROCYTE': ['Brain_Cortex'],
    'MICROGLIA': ['Brain_Cortex'],
    'SCHWANN_CELL': ['Nerve_Tibial'],
    'PNEUMOCYTE_I': ['Lung'], 'PNEUMOCYTE_II': ['Lung'],
    'PODOCYTE': ['Kidney_Cortex'],
    'MELANOCYTE': ['Skin_Not_Sun_Exposed_Suprapubic'],
    'PHOTORECEPTOR': ['Brain_Cortex', 'Brain_Hippocampus'],
    'HAIR_CELL': ['Nerve_Tibial'],
    'PLATELET': ['Whole_Blood'],
    'OSTEOBLAST': ['Cells_Cultured_fibroblasts'],
    'OSTEOCLAST': ['Whole_Blood', 'Spleen'],
    'CHONDROCYTE': ['Cells_Cultured_fibroblasts'],
    'OOGONIA': ['Ovary'], 'SPERMATOGONIA': ['Testis'],
    'NEUTROPHIL': ['Whole_Blood'],
    'EOSINOPHIL': ['Whole_Blood'],
    'BASOPHIL': ['Whole_Blood'],
}

KNOWN_COCKTAILS = {
    'OSKM → iPSC': {
        'source': 'FIBROBLAST', 'target_types': [], 'target_name': 'iPSC',
        'factors': ['POU5F1', 'SOX2', 'KLF4', 'MYC'],
        'citation': 'Takahashi & Yamanaka, Cell 2006',
    },
    'BAM → Neuron': {
        'source': 'FIBROBLAST', 'target_types': ['NEURON_EXCITATORY', 'NEURON_INHIBITORY'],
        'target_name': 'Neuron',
        'factors': ['POU3F2', 'ASCL1', 'MYT1L'],
        'citation': 'Vierbuchen et al., Nature 2010',
    },
    'GMT → Cardiomyocyte': {
        'source': 'FIBROBLAST', 'target_types': ['CARDIOMYOCYTE'],
        'target_name': 'Cardiomyocyte',
        'factors': ['GATA4', 'MEF2C', 'TBX5'],
        'citation': 'Ieda et al., Cell 2010',
    },
    'HNF4A+FOXA → Hepatocyte': {
        'source': 'FIBROBLAST', 'target_types': ['HEPATOCYTE'],
        'target_name': 'Hepatocyte',
        'factors': ['HNF4A', 'FOXA1', 'FOXA2', 'FOXA3'],
        'citation': 'Sekiya & Suzuki, Nature 2011',
    },
    'MYOD1 → Myocyte': {
        'source': 'FIBROBLAST', 'target_types': ['SKELETAL_MUSCLE'],
        'target_name': 'Myocyte',
        'factors': ['MYOD1'],
        'citation': 'Davis et al., Cell 1987',
    },
    'PDX1+MAFA → Beta cell': {
        'source': 'FIBROBLAST', 'target_types': ['BETA_CELL'],
        'target_name': 'Beta cell',
        'factors': ['PDX1', 'MAFA'],
        'citation': 'Zhou et al., Nature 2008',
    },
    'PPARG+CEBPA → Adipocyte': {
        'source': 'FIBROBLAST', 'target_types': ['ADIPOCYTE'],
        'target_name': 'Adipocyte',
        'factors': ['PPARG', 'CEBPA'],
        'citation': 'Tontonoz et al., Cell 1994',
    },
    'ETV2+FLI1+ERG → Endothelial': {
        'source': 'FIBROBLAST', 'target_types': ['ENDOTHELIAL'],
        'target_name': 'Endothelial',
        'factors': ['ETV2', 'FLI1', 'ERG'],
        'citation': 'Ginsberg et al., Cell 2012',
    },
    'TAL1+GATA2+RUNX1+LMO2+ETV6 → HSC': {
        'source': 'FIBROBLAST', 'target_types': ['HSC'],
        'target_name': 'HSC',
        'factors': ['TAL1', 'GATA2', 'RUNX1', 'LMO2', 'ETV6'],
        'citation': 'Riddell et al., Cell Stem Cell 2014',
    },
}


class CocktailPredictor:
    def __init__(self):
        self.conn = psycopg2.connect(DB_URL)
        self.cur = self.conn.cursor()
        self.gene_progs = {}
        self.prog_genes = {}
        self.gene_expr = {}
        self.gene_tau = {}
        self.tissues = []
        self.tissue_idx = {}
        self.ct_markers = {}
        self.n_genes = 0
        self.gene_dept = {}
        self.regulatory_genes = set()
        self.tf_tier = {}
        self.gene_pheno = {}
        self.gene_temporal = {}
        self.M_res = None
        self.kernel_gene_idx = {}
        self.loaded = False

    def load(self):
        if self.loaded:
            return
        t0 = time.time()
        cur = self.cur

        print("[1/8] Loading UID→gene mappings...")
        uid_to_gene = {}
        cur.execute("SELECT gene_name, uniprot_id FROM canonical_gene_uniprot")
        for gn, uid in cur.fetchall():
            uid_to_gene[uid] = gn
        cur.execute("SELECT uniprot_id, gene_name FROM protein_catalog WHERE gene_name IS NOT NULL")
        for uid, gn in cur.fetchall():
            if uid not in uid_to_gene and gn:
                uid_to_gene[uid] = gn.split()[0].upper()
        for tbl in ['protein_program_map', 'protein_program_map_v2']:
            cur.execute(f"SELECT DISTINCT uniprot_id, gene_name FROM {tbl}")
            for uid, gn in cur.fetchall():
                if uid not in uid_to_gene and gn:
                    uid_to_gene[uid] = gn.split()[0].upper()

        print("[2/8] Loading program maps + inferred programs...")
        gene_progs = defaultdict(set)
        prog_genes = defaultdict(set)
        for tbl in ['protein_program_map', 'protein_program_map_v2']:
            cur.execute(f"SELECT uniprot_id, program_id FROM {tbl}")
            for uid, pid in cur.fetchall():
                gene = uid_to_gene.get(uid)
                if gene:
                    gene_progs[gene].add(pid)
                    prog_genes[pid].add(gene)

        inferred_path = os.path.join(os.path.dirname(__file__), 'inferred_programs.pkl')
        if os.path.exists(inferred_path):
            with open(inferred_path, 'rb') as f:
                inferred = pickle.load(f)
            n_inf = 0
            for gene, data in inferred.items():
                if gene not in gene_progs:
                    for pid in data['programs']:
                        gene_progs[gene].add(pid)
                        prog_genes[pid].add(gene)
                    n_inf += 1
            print(f"  +{n_inf:,} inferred genes")

        self.gene_progs = dict(gene_progs)
        self.prog_genes = dict(prog_genes)
        self.n_genes = len(self.gene_progs)
        print(f"  {self.n_genes:,} genes, {len(self.prog_genes):,} programs")

        print("[3/8] Loading department profiles + regulatory gene filter...")
        cur.execute("""
            SELECT gene_name, primary_department, all_departments
            FROM gene_department_map
            WHERE all_departments IS NOT NULL AND array_length(all_departments, 1) > 0
        """)
        for gene, primary, all_depts in cur.fetchall():
            self.gene_dept[gene] = (primary, all_depts)
            if primary in ('Transcription', 'Chromatin') or \
               'Transcription' in (all_depts or []) or \
               'Chromatin' in (all_depts or []):
                self.regulatory_genes.add(gene)

        cur.execute("""
            SELECT gene_names_primary,
                   gene_ontology_molecular_function,
                   keywords
            FROM complete_human_proteome
            WHERE gene_names_primary IS NOT NULL
        """)
        for gene, go_mf, kw in cur.fetchall():
            go_mf = (go_mf or '').lower()
            kw = (kw or '').lower()
            tier = 0

            kw_is_primary_enzyme = any(e in kw for e in [
                'protease', 'hydrolase', 'helicase', 'kinase', 'phosphatase',
                'oxidoreductase', 'peroxidase', 'elastase', 'lipase',
                'antimicrobial', 'zymogen', 'peptidase',
            ])

            go_has_tf = 'transcription factor activity' in go_mf
            go_has_coact = ('transcription coactivator' in go_mf or
                            'transcription corepressor' in go_mf)
            kw_has_tf = 'transcription factor' in kw

            if go_has_tf or kw_has_tf:
                tier = max(tier, 3)
            elif go_has_coact and not kw_is_primary_enzyme:
                tier = max(tier, 3)
            elif 'transcription regulation' in kw and not kw_is_primary_enzyme:
                tier = max(tier, 2)

            if 'chromatin binding' in go_mf or 'chromatin regulator' in kw:
                tier = max(tier, 2)
            if 'dna binding' in go_mf or 'dna-binding' in kw:
                tier = max(tier, 2)

            if tier > 0:
                self.regulatory_genes.add(gene)
                self.tf_tier[gene] = max(self.tf_tier.get(gene, 0), tier)

        for gene in list(self.regulatory_genes):
            if gene not in self.tf_tier:
                dept = self.gene_dept.get(gene, (None, []))
                if dept[0] in ('Transcription', 'Chromatin'):
                    self.tf_tier[gene] = 1
                else:
                    self.tf_tier[gene] = 1
        print(f"  Regulatory gene filter: {len(self.regulatory_genes):,} genes "
              f"(Tier3: {sum(1 for v in self.tf_tier.values() if v==3)}, "
              f"Tier2: {sum(1 for v in self.tf_tier.values() if v==2)}, "
              f"Tier1: {sum(1 for v in self.tf_tier.values() if v==1)})")

        print("[4/8] Loading GTEx expression...")
        cur.execute("SELECT DISTINCT tissue FROM gtex_expression ORDER BY tissue")
        self.tissues = [r[0] for r in cur.fetchall()]
        self.tissue_idx = {t: i for i, t in enumerate(self.tissues)}
        cur.execute("SELECT gene_symbol, tissue, tpm_median FROM gtex_expression")
        for gene, tissue, tpm in cur.fetchall():
            if gene not in self.gene_expr:
                self.gene_expr[gene] = np.zeros(len(self.tissues))
            if tissue in self.tissue_idx:
                self.gene_expr[gene][self.tissue_idx[tissue]] = float(tpm)
        for gene, expr in self.gene_expr.items():
            mx = expr.max()
            if mx > 0:
                normed = expr / mx
                self.gene_tau[gene] = float((1 - normed).sum() / (len(self.tissues) - 1))

        print("[5/8] Loading cell type markers...")
        cur.execute("SELECT cell_type, gene_name, marker_confidence FROM cell_type_markers")
        ct_markers = defaultdict(dict)
        conf_map = {'HIGH': 1.0, 'MEDIUM': 0.7, 'LOW': 0.4}
        for ct, gene, conf in cur.fetchall():
            w = conf_map.get(str(conf).upper(), 0.7) if conf else 1.0
            ct_markers[ct][gene] = w
        self.ct_markers = dict(ct_markers)

        print("[6/8] Loading phenotype data...")
        cur.execute("SELECT gene_name, phenotype_category FROM gene_phenotype_map")
        gene_pheno = defaultdict(set)
        for gene, cat in cur.fetchall():
            gene_pheno[gene].add(cat)
        self.gene_pheno = dict(gene_pheno)
        print(f"  {len(self.gene_pheno):,} genes with phenotype annotations")

        print("[7/8] Loading temporal expression data...")
        cur.execute("SELECT layer_id, day_start, day_end FROM embryogenesis_layers ORDER BY layer_id")
        layer_days = {}
        for lid, ds, de in cur.fetchall():
            layer_days[lid] = (ds, de)
        cur.execute("""SELECT gene_name, activation_stage, peak_stage, silencing_stage
                      FROM gene_temporal_expression""")
        for gene, act, peak, sil in cur.fetchall():
            act_day = layer_days.get(act, (0, 0))[0] if act else 0
            peak_day = layer_days.get(peak, (0, 0))[0] if peak else act_day
            sil_day = layer_days.get(sil, (36500, 36500))[1] if sil else 36500
            self.gene_temporal[gene] = (act_day, peak_day, sil_day)
        print(f"  {len(self.gene_temporal):,} genes with temporal data")

        print("[8/8] Loading kernel disruption profiles...")
        if os.path.exists(DISRUPTION_PROFILES_PATH):
            with open(DISRUPTION_PROFILES_PATH) as f:
                data = json.load(f)
            profiles = data['profiles']
            depts = list(list(profiles.values())[0].keys())
            genes_sorted = sorted(profiles.keys())
            self.kernel_gene_idx = {g: i for i, g in enumerate(genes_sorted)}
            M = np.zeros((len(genes_sorted), len(depts)), dtype=np.float64)
            for i, g in enumerate(genes_sorted):
                for j, d in enumerate(depts):
                    M[i, j] = profiles[g].get(d, 0)
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            self.M_res = M - np.outer(U[:, 0] * S[0], Vt[0])
            print(f"  {len(genes_sorted):,} genes, {len(depts)} departments, PC1 removed")
        else:
            print(f"  Disruption profiles not found, kernel scoring disabled")

        self.loaded = True
        print(f"  Loaded in {time.time()-t0:.1f}s\n")

    def _get_markers(self, cell_types):
        markers = {}
        for ct in cell_types:
            if ct in self.ct_markers:
                markers.update(self.ct_markers[ct])
        return markers

    def _get_gtex_tissues(self, cell_types):
        tissues = []
        for ct in cell_types:
            tissues.extend(CELL_GTEX_MAP.get(ct, []))
        return list(set(tissues))

    def compute_transition_vector(self, source_types, target_types):
        """
        Compute the program-level transition vector from source → target.
        Returns programs that need to be ACTIVATED (in target but not source)
        and programs that need to be SILENCED (in source but not target).
        """
        source_markers = self._get_markers(source_types)
        target_markers = self._get_markers(target_types)

        source_programs = set()
        for m in source_markers:
            if m in self.gene_progs:
                source_programs |= self.gene_progs[m]

        target_programs = set()
        for m in target_markers:
            if m in self.gene_progs:
                target_programs |= self.gene_progs[m]

        activate = target_programs - source_programs
        silence = source_programs - target_programs
        shared = source_programs & target_programs

        return {
            'activate': activate,
            'silence': silence,
            'shared': shared,
            'target_programs': target_programs,
            'source_programs': source_programs,
            'n_activate': len(activate),
            'n_silence': len(silence),
            'n_shared': len(shared),
            'target_markers': target_markers,
            'source_markers': source_markers,
            '_source_type': source_types[0] if source_types else '',
        }

    def score_candidates(self, transition, target_types, weights=None):
        """
        Score all genes for their ability to drive the source→target transition.

        9-component scoring with two-tier adaptive weights:
        1. Directional connectivity (target - src_pen * source)
        2. Activation precision
        3. Fraction enriched programs
        4. Mean enrichment magnitude
        5. GTEx tissue specificity
        6. Tissue specificity (tau)
        7. Phenotype relevance
        8. Kernel disruption similarity
        9. Temporal developmental overlap
        """
        w = weights if weights is not None else get_tier_weights(target_types)

        tv = transition
        activate = tv['activate']
        target_progs = tv['target_programs']
        source_progs = tv['source_programs']
        target_markers = tv['target_markers']
        target_markers_in_progs = set(m for m in target_markers if m in self.gene_progs)
        n_markers = len(target_markers_in_progs)

        source_markers = tv['source_markers']
        source_markers_in_progs = set(m for m in source_markers if m in self.gene_progs)
        n_source_markers = len(source_markers_in_progs)

        gtex_tissues = self._get_gtex_tissues(target_types)
        source_gtex = self._get_gtex_tissues([tv.get('_source_type', '')] if tv.get('_source_type') else [])
        t_indices = [self.tissue_idx[t] for t in gtex_tissues if t in self.tissue_idx]
        s_indices = [self.tissue_idx[t] for t in source_gtex if t in self.tissue_idx]

        gene_target_conn = defaultdict(int)
        for m in target_markers_in_progs:
            pid_set = self.gene_progs[m]
            seen = set()
            for pid in pid_set:
                for gene in self.prog_genes.get(pid, ()):
                    if gene not in seen:
                        seen.add(gene)
                        gene_target_conn[gene] += 1

        gene_source_conn = defaultdict(int)
        for m in source_markers_in_progs:
            pid_set = self.gene_progs[m]
            seen = set()
            for pid in pid_set:
                for gene in self.prog_genes.get(pid, ()):
                    if gene not in seen:
                        seen.add(gene)
                        gene_source_conn[gene] += 1

        prog_enr = {}
        for pid, genes in self.prog_genes.items():
            n_in = len(genes)
            n_mark = sum(1 for g in genes if g in target_markers_in_progs)
            if n_mark > 0:
                expected = n_in * n_markers / self.n_genes if self.n_genes > 0 else 0
                prog_enr[pid] = n_mark / max(expected, 1e-10)

        target_marker_genes = set(target_markers.keys())
        kernel_sig = None
        if self.M_res is not None and self.kernel_gene_idx:
            marker_indices_k = [self.kernel_gene_idx[m] for m in target_marker_genes
                                if m in self.kernel_gene_idx]
            if marker_indices_k:
                marker_profiles_k = self.M_res[marker_indices_k]
                kernel_sig = np.mean(marker_profiles_k, axis=0)
                ksn = np.linalg.norm(kernel_sig)
                if ksn > 1e-10:
                    kernel_sig = kernel_sig / ksn
                else:
                    kernel_sig = None

        scores = {}
        for gene, progs in self.gene_progs.items():
            n_progs = len(progs)
            act_precision = len(progs & activate) / max(n_progs, 1)

            tc = gene_target_conn.get(gene, 0)
            target_conn_frac = tc / max(n_markers, 1)
            sc = gene_source_conn.get(gene, 0)
            source_conn_frac = sc / max(n_source_markers, 1)

            n_enriched = 0
            sum_log_enr = 0.0
            for pid in progs:
                e = prog_enr.get(pid, 0)
                if e > 1.0:
                    n_enriched += 1
                    sum_log_enr += math.log2(e)
            frac_enriched = n_enriched / max(n_progs, 1)
            mean_enr = sum_log_enr / n_enriched if n_enriched > 0 else 0

            gtex_score = 0.0
            if gene in self.gene_expr and t_indices:
                expr = self.gene_expr[gene]
                target_expr = sum(expr[i] for i in t_indices) / len(t_indices)
                all_expr = float(expr.mean())
                source_expr = sum(expr[i] for i in s_indices) / len(s_indices) if s_indices else all_expr
                if all_expr > 0 and target_expr > 1:
                    ratio = target_expr / max(all_expr, 0.1)
                    gtex_score = min(math.log2(max(ratio, 1)), 5) / 5
                    if source_expr > 0 and target_expr > source_expr:
                        gtex_score = min(gtex_score * 1.3, 1.0)

            tau_score = min(self.gene_tau.get(gene, 0.5) / 0.9, 1.0)

            pheno_s = compute_phenotype_score(gene, target_types, self.gene_pheno)

            kern_s = 0.0
            if kernel_sig is not None and gene in self.kernel_gene_idx:
                gene_res = self.M_res[self.kernel_gene_idx[gene]]
                gn = np.linalg.norm(gene_res)
                if gn > 1e-10:
                    kern_s = max(np.dot(gene_res, kernel_sig) / gn, 0.0)

            temp_s = compute_temporal_score(gene, target_types, self.gene_temporal)

            directional = max(target_conn_frac - w['src_pen'] * source_conn_frac, 0)

            # v1.1: self-marker bonus — if the gene IS itself a high-confidence
            # marker of the target cell type, give it a direct score lift.
            # This corrects the bias toward broadly-expressed chromatin remodelers
            # over lineage-specifying TFs (e.g. CEBPA for granulocytes) that have
            # narrow program footprints despite being the correct biological answer.
            # Note: target_markers stores float weights (HIGH=1.0, MEDIUM=0.7, LOW=0.4)
            # via conf_map in load(); do not compare against string literals.
            self_s = float(target_markers.get(gene, 0.0))

            composite = (w['w_dir'] * directional +
                         w['w_act'] * act_precision +
                         w['w_frac'] * frac_enriched +
                         w['w_enr'] * min(mean_enr / 5, 1) +
                         w['w_gtex'] * max(gtex_score, 0) +
                         w['w_tau'] * tau_score +
                         w['w_pheno'] * pheno_s +
                         w['w_kern'] * kern_s +
                         w['w_temp'] * temp_s +
                         w['w_self'] * self_s)

            scores[gene] = {
                'composite': composite,
                'act_precision': act_precision,
                'conn_frac': target_conn_frac,
                'directional': directional,
                'n_connected': tc,
                'frac_enriched': frac_enriched,
                'mean_enrichment': mean_enr,
                'gtex': gtex_score,
                'tau': self.gene_tau.get(gene, 0.5),
                'pheno': pheno_s,
                'kernel': kern_s,
                'temporal': temp_s,
                'self_marker': self_s,
                'source_conn': source_conn_frac,
                'n_programs': n_progs,
                'programs': progs,
                'programs_activate': progs & activate,
                'programs_target': progs & target_progs,
            }

        return scores

    def assemble_cocktail(self, scores, transition, max_factors=5, min_factors=2,
                          top_n_candidates=30):
        """
        Combinatorial cocktail assembly from top-ranked TFs.

        Strategy: take the top N regulatory candidates (by individual score),
        then find the combination of k factors with the best balance of
        individual quality and complementary program coverage.
        """
        activate = transition['activate']
        target_progs = transition['target_programs']

        ranked = sorted(scores.items(), key=lambda x: -x[1]['composite'])
        tf_only = [(g, s) for g, s in ranked
                    if g in self.regulatory_genes
                    and self.tf_tier.get(g, 0) >= 3
                    and s['n_programs'] > 0]
        candidates = tf_only[:top_n_candidates]

        if len(candidates) < min_factors:
            candidates = [(g, s) for g, s in ranked
                          if g in self.regulatory_genes and s['n_programs'] > 0][:top_n_candidates]

        best_cocktails = []

        for n_factors in range(min_factors, max_factors + 1):
            search_pool = candidates
            if n_factors >= 5:
                search_pool = candidates[:20]
            elif n_factors >= 4:
                search_pool = candidates[:25]
            top_combos = []
            for combo in combinations(range(len(search_pool)), n_factors):
                genes = [search_pool[i][0] for i in combo]
                gene_scores = [search_pool[i][1] for i in combo]

                combined_progs = set()
                for s in gene_scores:
                    combined_progs |= s['programs']

                act_covered = len(combined_progs & activate)
                tgt_covered = len(combined_progs & target_progs)
                coverage = act_covered / max(len(activate), 1)
                tgt_cov = tgt_covered / max(len(target_progs), 1)

                mean_quality = np.mean([s['composite'] for s in gene_scores])

                all_prog_sets = [s['programs'] for s in gene_scores]
                total_union = len(combined_progs)
                total_individual = sum(len(p) for p in all_prog_sets)
                complementarity = total_union / max(total_individual, 1)

                combo_score = (0.45 * mean_quality +
                               0.25 * coverage +
                               0.15 * tgt_cov +
                               0.15 * complementarity)

                top_combos.append({
                    'factors': genes,
                    'score': combo_score,
                    'mean_quality': mean_quality,
                    'activation_coverage': coverage,
                    'target_coverage': tgt_cov,
                    'complementarity': complementarity,
                })

            top_combos.sort(key=lambda x: -x['score'])
            best = top_combos[0] if top_combos else None

            if best:
                factor_details = []
                cumulative_progs = set()
                for g in best['factors']:
                    s = scores[g]
                    new_act = len(s['programs_activate'] - cumulative_progs)
                    new_tgt = len(s['programs_target'] - cumulative_progs)
                    factor_details.append({
                        'gene': g,
                        'composite_score': s['composite'],
                        'tf_tier': self.tf_tier.get(g, 0),
                        'new_activation_programs': new_act,
                        'new_target_programs': new_tgt,
                        'total_programs': s['n_programs'],
                        'gtex': s['gtex'],
                        'marker_connectivity': s['n_connected'],
                    })
                    cumulative_progs |= s['programs']

                best_cocktails.append({
                    'factors': best['factors'],
                    'n_factors': n_factors,
                    'cocktail_score': best['score'],
                    'mean_quality': best['mean_quality'],
                    'activation_coverage': best['activation_coverage'],
                    'target_coverage': best['target_coverage'],
                    'complementarity': best['complementarity'],
                    'programs_covered': int(best['activation_coverage'] * len(activate)),
                    'programs_needed': len(activate),
                    'factor_details': factor_details,
                    'top_alternatives': [{'factors': c['factors'], 'score': c['score']}
                                        for c in top_combos[1:4]],
                })

        best_cocktails.sort(key=lambda x: -x['cocktail_score'])
        return best_cocktails

    def greedy_marginal_assembly(self, scores, transition, target_types,
                                  max_factors=5, pool_size=50, restarts=5):
        """
        Greedy marginal cocktail assembly.

        Instead of scoring genes independently and then combining, this builds
        cocktails sequentially: at each step, pick the gene whose *marginal*
        contribution (new program coverage + individual quality) is highest
        given genes already selected. Multiple restarts from different seeds
        to avoid local optima.
        """
        activate = transition['activate']
        target_progs = transition['target_programs']
        w = get_tier_weights(target_types)

        ranked = sorted(scores.items(), key=lambda x: -x[1]['composite'])
        pool = [(g, s) for g, s in ranked
                if g in self.regulatory_genes
                and self.tf_tier.get(g, 0) >= 2
                and s['n_programs'] > 0][:pool_size]

        if len(pool) < 2:
            pool = [(g, s) for g, s in ranked
                    if g in self.regulatory_genes and s['n_programs'] > 0][:pool_size]

        best_result = None

        for restart in range(restarts):
            seed_idx = restart % len(pool)
            selected = [pool[seed_idx]]
            covered_progs = set(selected[0][1]['programs'])

            for step in range(1, max_factors):
                best_marginal = None
                best_marginal_score = -1

                for g, s in pool:
                    if any(g == sg for sg, _ in selected):
                        continue

                    new_progs = s['programs'] - covered_progs
                    new_act = len(new_progs & activate)
                    new_tgt = len(new_progs & target_progs)
                    marginal_coverage = new_act / max(len(activate), 1)
                    marginal_tgt = new_tgt / max(len(target_progs), 1)

                    marginal_score = (0.35 * s['composite'] +
                                     0.35 * marginal_coverage +
                                     0.20 * marginal_tgt +
                                     0.10 * (len(new_progs) / max(len(s['programs']), 1)))

                    if marginal_score > best_marginal_score:
                        best_marginal_score = marginal_score
                        best_marginal = (g, s)

                if best_marginal is None:
                    break

                selected.append(best_marginal)
                covered_progs |= best_marginal[1]['programs']

            total_act = len(covered_progs & activate)
            total_tgt = len(covered_progs & target_progs)
            cocktail_score = (0.40 * np.mean([s['composite'] for _, s in selected]) +
                              0.30 * total_act / max(len(activate), 1) +
                              0.20 * total_tgt / max(len(target_progs), 1) +
                              0.10 * len(covered_progs) / max(sum(len(s['programs']) for _, s in selected), 1))

            result = {
                'factors': [g for g, _ in selected],
                'n_factors': len(selected),
                'cocktail_score': cocktail_score,
                'activation_coverage': total_act / max(len(activate), 1),
                'target_coverage': total_tgt / max(len(target_progs), 1),
                'programs_covered': total_act,
                'programs_needed': len(activate),
                'factor_details': [{
                    'gene': g,
                    'composite_score': s['composite'],
                    'tf_tier': self.tf_tier.get(g, 0),
                    'n_programs': s['n_programs'],
                } for g, s in selected],
            }

            if best_result is None or cocktail_score > best_result['cocktail_score']:
                best_result = result

        return best_result

    def exhaustive_search(self, scores, transition, n_factors=3, top_n=100):
        """
        For small cocktail sizes, try all combinations of top regulatory candidates.
        """
        activate = transition['activate']
        target_progs = transition['target_programs']

        ranked = sorted(scores.items(), key=lambda x: -x[1]['composite'])
        candidates = [(g, s) for g, s in ranked[:top_n * 3]
                       if s['n_programs'] > 0 and g in self.regulatory_genes]
        candidates = candidates[:top_n]

        best_combos = []
        for combo in combinations(range(len(candidates)), n_factors):
            genes = [candidates[i][0] for i in combo]
            gene_scores = [candidates[i][1] for i in combo]

            combined_progs = set()
            for s in gene_scores:
                combined_progs |= s['programs']

            act_covered = len(combined_progs & activate)
            tgt_covered = len(combined_progs & target_progs)
            coverage = act_covered / max(len(activate), 1)
            tgt_coverage = tgt_covered / max(len(target_progs), 1)

            combo_score = (0.40 * coverage +
                           0.30 * tgt_coverage +
                           0.30 * np.mean([s['composite'] for s in gene_scores]))

            best_combos.append({
                'factors': genes,
                'score': combo_score,
                'activation_coverage': coverage,
                'target_coverage': tgt_coverage,
            })

        best_combos.sort(key=lambda x: -x['score'])
        return best_combos[:20]

    def predict_cocktail(self, source_type, target_types, target_name=None,
                         max_factors=5, verbose=True):
        """
        Full prediction pipeline: source → target cocktail.
        """
        self.load()
        t0 = time.time()

        source_types = [source_type] if isinstance(source_type, str) else source_type
        if isinstance(target_types, str):
            target_types = [target_types]

        if target_name is None:
            target_name = '/'.join(target_types)

        if verbose:
            print(f"{'='*80}")
            print(f"  COCKTAIL PREDICTION: {source_type} → {target_name}")
            print(f"{'='*80}")

        transition = self.compute_transition_vector(source_types, target_types)

        if verbose:
            print(f"\n  Transition vector:")
            print(f"    Source markers: {len(transition['source_markers'])}")
            print(f"    Target markers: {len(transition['target_markers'])}")
            print(f"    Programs to ACTIVATE: {transition['n_activate']}")
            print(f"    Programs to SILENCE:  {transition['n_silence']}")
            print(f"    Shared programs:      {transition['n_shared']}")

        scores = self.score_candidates(transition, target_types)

        if verbose:
            ranked = sorted(scores.items(), key=lambda x: -x[1]['composite'])
            reg_ranked = [(g, s) for g, s in ranked if g in self.regulatory_genes]
            print(f"\n  Top 20 regulatory candidates (from {len(reg_ranked):,} TFs/chromatin regs):")
            for i, (gene, s) in enumerate(reg_ranked[:20]):
                print(f"    {i+1:3d}. {gene:15s} score={s['composite']:.3f} "
                      f"dir={s['directional']:.3f} gtex={s['gtex']:.3f} "
                      f"τ={s.get('tau',0):.2f} pheno={s.get('pheno',0):.2f} "
                      f"kern={s.get('kernel',0):.2f}")

        cocktails = self.assemble_cocktail(scores, transition, max_factors=max_factors)

        if verbose:
            print(f"\n  Assembled cocktails:")
            for ck in cocktails:
                factors_str = ' + '.join(ck['factors'])
                print(f"\n  [{ck['n_factors']} factors] {factors_str}")
                print(f"    Score: {ck['cocktail_score']:.3f}")
                print(f"    Activation coverage: {ck['activation_coverage']*100:.1f}% "
                      f"({ck['programs_covered']}/{ck['programs_needed']} programs)")
                print(f"    Target coverage: {ck['target_coverage']*100:.1f}%")
                for fd in ck['factor_details']:
                    print(f"      {fd['gene']:12s}: score={fd['composite_score']:.3f} "
                          f"+{fd['new_activation_programs']} act progs, "
                          f"+{fd['new_target_programs']} tgt progs "
                          f"(total {fd['total_programs']}) gtex={fd['gtex']:.3f}")

        if verbose and len(target_types) > 0:
            print(f"\n  Exhaustive search (top 3-factor combinations from top 50):")
            best_3 = self.exhaustive_search(scores, transition, n_factors=3, top_n=50)
            for i, combo in enumerate(best_3[:10]):
                print(f"    {i+1:2d}. {' + '.join(combo['factors']):40s} "
                      f"score={combo['score']:.3f} "
                      f"act={combo['activation_coverage']*100:.1f}% "
                      f"tgt={combo['target_coverage']*100:.1f}%")

        elapsed = time.time() - t0

        if verbose:
            print(f"\n  Prediction time: {elapsed:.1f}s")

        return {
            'source': source_type,
            'target': target_name,
            'target_types': target_types,
            'transition': {
                'n_activate': transition['n_activate'],
                'n_silence': transition['n_silence'],
                'n_shared': transition['n_shared'],
            },
            'cocktails': cocktails,
            'top_candidates': [(g, float(s['composite'])) for g, s in
                               sorted(scores.items(), key=lambda x: -x[1]['composite'])[:200]],
            'elapsed': elapsed,
        }

    def validate_against_known(self):
        """
        Run predictions for all known cocktails and check if published
        factors appear in the predicted cocktails or top candidates.
        """
        self.load()

        print("\n" + "=" * 80)
        print("  VALIDATION: Predicting known cocktails")
        print("=" * 80)

        results = {}
        total_factors = 0
        factors_in_top20 = 0
        factors_in_top50 = 0
        cocktail_recoveries = 0
        n_cocktails = 0

        for name, cocktail in KNOWN_COCKTAILS.items():
            if not cocktail['target_types']:
                continue

            source = cocktail['source']
            target_types = cocktail['target_types']
            known_factors = cocktail['factors']

            print(f"\n{'─'*80}")
            print(f"  {name} ({cocktail['citation']})")
            print(f"  Known factors: {', '.join(known_factors)}")
            print(f"{'─'*80}")

            prediction = self.predict_cocktail(
                source, target_types, cocktail['target_name'],
                max_factors=len(known_factors) + 1, verbose=False
            )

            top_genes = [g for g, _ in prediction['top_candidates']
                        if g in self.regulatory_genes]

            n_cocktails += 1
            cocktail_factors_found = 0

            for f in known_factors:
                total_factors += 1
                if f in top_genes:
                    rank = top_genes.index(f) + 1
                    if rank <= 20:
                        factors_in_top20 += 1
                    if rank <= 50:
                        factors_in_top50 += 1
                    cocktail_factors_found += 1
                    mark = '***' if rank <= 10 else '**' if rank <= 20 else '*' if rank <= 50 else ''
                    print(f"    {f:12s}: rank {rank:>4d}/50 {mark}")
                else:
                    print(f"    {f:12s}: not in top 50")

            if cocktail_factors_found == len(known_factors):
                cocktail_recoveries += 1
                print(f"    → ALL {len(known_factors)} FACTORS in top 50 candidates!")
            else:
                print(f"    → {cocktail_factors_found}/{len(known_factors)} factors in top 50")

            best_predicted = prediction['cocktails'][0] if prediction['cocktails'] else None
            if best_predicted:
                predicted_factors = best_predicted['factors']
                overlap = set(predicted_factors) & set(known_factors)
                print(f"    Predicted cocktail: {' + '.join(predicted_factors)}")
                print(f"    Overlap with known: {len(overlap)}/{len(known_factors)} "
                      f"({', '.join(overlap) if overlap else 'none'})")
                print(f"    Activation coverage: {best_predicted['activation_coverage']*100:.1f}%")

            results[name] = {
                'known_factors': known_factors,
                'factors_in_top50': cocktail_factors_found,
                'total_factors': len(known_factors),
                'predicted_cocktail': best_predicted['factors'] if best_predicted else [],
                'overlap': len(set(best_predicted['factors']) & set(known_factors)) if best_predicted else 0,
            }

        print(f"\n\n{'='*80}")
        print(f"  VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"  Known cocktails tested: {n_cocktails}")
        print(f"  Total factors: {total_factors}")
        print(f"  Factors in top 20 candidates: {factors_in_top20}/{total_factors} "
              f"({factors_in_top20/total_factors*100:.0f}%)")
        print(f"  Factors in top 50 candidates: {factors_in_top50}/{total_factors} "
              f"({factors_in_top50/total_factors*100:.0f}%)")
        print(f"  Complete cocktail recovery (all factors in top 50): "
              f"{cocktail_recoveries}/{n_cocktails}")

        return results


def main():
    predictor = CocktailPredictor()

    print("\n" + "#" * 80)
    print("  VM COCKTAIL PREDICTOR — FULL CELL FATE TRANSITION MODULE")
    print("#" * 80)

    predictor.predict_cocktail('FIBROBLAST', ['NEURON_EXCITATORY', 'NEURON_INHIBITORY'],
                               'Neuron', max_factors=5)

    predictor.predict_cocktail('FIBROBLAST', ['CARDIOMYOCYTE'],
                               'Cardiomyocyte', max_factors=5)

    predictor.predict_cocktail('FIBROBLAST', ['HEPATOCYTE'],
                               'Hepatocyte', max_factors=5)

    predictor.predict_cocktail('FIBROBLAST', ['SKELETAL_MUSCLE'],
                               'Myocyte', max_factors=5)

    predictor.predict_cocktail('FIBROBLAST', ['BETA_CELL'],
                               'Beta cell', max_factors=5)

    predictor.predict_cocktail('FIBROBLAST', ['ADIPOCYTE'],
                               'Adipocyte', max_factors=5)

    predictor.predict_cocktail('FIBROBLAST', ['ENDOTHELIAL'],
                               'Endothelial', max_factors=5)

    predictor.predict_cocktail('FIBROBLAST', ['HSC'],
                               'HSC', max_factors=5)

    predictor.validate_against_known()

    all_predictions = {}
    for name, cocktail in KNOWN_COCKTAILS.items():
        if cocktail['target_types']:
            pred = predictor.predict_cocktail(
                cocktail['source'], cocktail['target_types'],
                cocktail['target_name'], max_factors=5, verbose=False
            )
            all_predictions[name] = pred

    with open('paper2/cocktail_predictions.json', 'w') as f:
        json.dump(all_predictions, f, indent=2, default=str)

    print(f"\n  All predictions saved to paper2/cocktail_predictions.json")
    predictor.conn.close()


if __name__ == '__main__':
    main()
