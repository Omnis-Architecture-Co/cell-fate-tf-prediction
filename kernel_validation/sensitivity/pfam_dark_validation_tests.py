#!/usr/bin/env python3
"""
Annotation-Dark Protein Validation — Three Independent Tests
=============================================================
Test 1: Interaction enrichment (same-dept pairs more likely to physically interact?)
Test 2: Co-expression consistency (same-dept pairs more correlated in GTEx?)
Test 3: GO cross-validation (convergence-derived depts match GO-derived depts?)

Uses convergence-classified genes (source='omnis_convergence') as the
annotation-independent set, with protein_interactions, GTEx expression,
and GO term→department mapping as independent validation sources.

Output: validation/sensitivity/pfam_dark_validation_results.json
"""

import json
import os
import random
import re
import time
from collections import defaultdict, Counter

import numpy as np
import psycopg2

np.random.seed(42)
random.seed(42)

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "pfam_dark_validation_results.json")

GO_DEPARTMENT_MAP = {
    "GO:0005840": "Translation", "GO:0003735": "Translation",
    "GO:0006412": "Translation", "GO:0005762": "Translation",
    "GO:0005763": "Translation", "GO:0022626": "Translation",
    "GO:0022627": "Translation", "GO:0030529": "Translation",
    "GO:0003743": "Translation", "GO:0003746": "Translation",
    "GO:0006119": "Mitochondrial", "GO:0042773": "Mitochondrial",
    "GO:0005747": "Mitochondrial", "GO:0005746": "Mitochondrial",
    "GO:0000421": "Mitochondrial", "GO:0022904": "Mitochondrial",
    "GO:0005739": "Mitochondrial", "GO:0006120": "Mitochondrial",
    "GO:0045256": "Mitochondrial", "GO:0005750": "Mitochondrial",
    "GO:0005216": "Ion channel", "GO:0022832": "Ion channel",
    "GO:0005251": "Ion channel", "GO:0005267": "Ion channel",
    "GO:0005245": "Ion channel", "GO:0005249": "Ion channel",
    "GO:0005261": "Ion channel", "GO:0022836": "Ion channel",
    "GO:0008158": "Ion channel",
    "GO:0006915": "Apoptosis", "GO:0008630": "Apoptosis",
    "GO:0097194": "Apoptosis", "GO:0043066": "Apoptosis",
    "GO:0006917": "Apoptosis", "GO:0097553": "Apoptosis",
    "GO:0006260": "DNA replication", "GO:0006261": "DNA replication",
    "GO:0003887": "DNA replication", "GO:0003896": "DNA replication",
    "GO:0006270": "DNA replication",
    "GO:0016192": "Vesicle trafficking", "GO:0006886": "Vesicle trafficking",
    "GO:0006888": "Vesicle trafficking", "GO:0007264": "Vesicle trafficking",
    "GO:0045054": "Vesicle trafficking", "GO:0006890": "Vesicle trafficking",
    "GO:0006952": "Immune response", "GO:0006955": "Immune response",
    "GO:0002376": "Immune response", "GO:0006954": "Immune response",
    "GO:0045087": "Immune response", "GO:0043312": "Immune response",
    "GO:0006487": "Glycosylation", "GO:0006493": "Glycosylation",
    "GO:0016757": "Glycosylation", "GO:0008194": "Glycosylation",
    "GO:0006914": "Autophagy", "GO:0061048": "Autophagy",
    "GO:0000045": "Autophagy", "GO:0016236": "Autophagy",
    "GO:0006629": "Lipid metabolism", "GO:0006631": "Lipid metabolism",
    "GO:0006665": "Lipid metabolism", "GO:0044255": "Lipid metabolism",
    "GO:0051170": "Nuclear transport", "GO:0051169": "Nuclear transport",
    "GO:0006606": "Nuclear transport", "GO:0006607": "Nuclear transport",
    "GO:0007186": "Receptor signaling", "GO:0007187": "Receptor signaling",
    "GO:0004930": "Receptor signaling", "GO:0004871": "Receptor signaling",
    "GO:0006457": "Protein folding", "GO:0051082": "Protein folding",
    "GO:0042026": "Protein folding", "GO:0044183": "Protein folding",
    "GO:0004984": "Olfactory", "GO:0007608": "Olfactory",
    "GO:0016570": "Chromatin", "GO:0006325": "Chromatin",
    "GO:0000786": "Chromatin", "GO:0051276": "Chromatin",
    "GO:0004672": "Kinase", "GO:0006468": "Kinase", "GO:0004674": "Kinase",
    "GO:0006351": "Transcription", "GO:0006355": "Transcription",
    "GO:0003700": "Transcription",
    "GO:0007010": "Cytoskeleton", "GO:0015629": "Cytoskeleton",
    "GO:0005200": "Cytoskeleton",
    "GO:0007049": "Cell cycle", "GO:0022402": "Cell cycle",
    "GO:0006281": "DNA repair", "GO:0000723": "DNA repair",
    "GO:0006396": "RNA processing", "GO:0008380": "RNA processing",
    "GO:0007155": "Cell adhesion", "GO:0005488": "Cell adhesion",
    "GO:0004721": "Phosphatase", "GO:0004722": "Phosphatase",
    "GO:0006511": "Ubiquitin", "GO:0016567": "Ubiquitin",
    "GO:0006508": "Proteolysis", "GO:0004252": "Proteolysis",
    "GO:0003676": "Nuc acid bind", "GO:0004386": "Nuc acid bind",
    "GO:0006790": "Methylation",
    "GO:0006810": "Transport",
    "GO:0007165": "Signaling",
}


def go_terms_to_department(go_terms):
    """Map a list of GO terms to department(s) using the same mapping as the pipeline."""
    depts = []
    for gt in go_terms:
        if gt in GO_DEPARTMENT_MAP:
            depts.append(GO_DEPARTMENT_MAP[gt])
    if not depts:
        return None, []
    dept_counts = Counter(depts)
    primary = dept_counts.most_common(1)[0][0]
    return primary, list(set(depts))


def get_conn():
    return psycopg2.connect(os.environ['BETA_DATABASE_URL'])


def main():
    t0 = time.time()
    print("=" * 70)
    print("  ANNOTATION-DARK PROTEIN VALIDATION — THREE TESTS")
    print("=" * 70)

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT gene_name, primary_department, all_departments, confidence
        FROM gene_department_map
        WHERE source = 'omnis_convergence'
    """)
    conv_genes = {}
    conv_all_depts = {}
    for gene, dept, all_d, conf in cur.fetchall():
        conv_genes[gene] = dept
        conv_all_depts[gene] = all_d if all_d else [dept]

    print(f"  Convergence-classified genes: {len(conv_genes)}")

    results = {"n_convergence_genes": len(conv_genes)}

    # ==================================================================
    # TEST 1: INTERACTION ENRICHMENT (CURATED ONLY — NO CIRCULARITY)
    # ==================================================================
    print("\n" + "=" * 70)
    print("  TEST 1: INTERACTION ENRICHMENT (curated protein_interactions)")
    print("  Validation source: pathway-curated physical interactions")
    print("  NO circularity — interactions are from pathway databases,")
    print("  departments are from sequence token convergence.")
    print("=" * 70)

    cur.execute("""
        SELECT gene_a, gene_b, interaction_type, confidence
        FROM protein_interactions
        WHERE confidence >= 0.7
    """)
    interactions = []
    for ga, gb, itype, conf in cur.fetchall():
        interactions.append((ga, gb, itype, float(conf)))
    print(f"  Total curated interactions (conf >= 0.7): {len(interactions)}")

    conv_set = set(conv_genes.keys())
    same_dept = 0
    cross_dept = 0
    pairs_conv = []
    for ga, gb, itype, conf in interactions:
        if ga in conv_genes and gb in conv_genes:
            pairs_conv.append((ga, gb))
            if conv_genes[ga] == conv_genes[gb]:
                same_dept += 1
            else:
                cross_dept += 1
    total_conv = same_dept + cross_dept

    print(f"\n  Curated interactions where BOTH genes are convergence-classified: {total_conv}")
    print(f"    Same department: {same_dept}")
    print(f"    Different department: {cross_dept}")

    dept_counts = Counter(conv_genes.values())
    n_total = len(conv_genes)
    expected_same = sum(c * (c - 1) for c in dept_counts.values()) / (n_total * (n_total - 1))
    observed_same = same_dept / total_conv if total_conv > 0 else 0

    print(f"\n  Expected same-dept rate (random pairing): {expected_same:.4f} ({expected_same*100:.1f}%)")
    print(f"  Observed same-dept rate: {observed_same:.4f} ({observed_same*100:.1f}%)")

    if total_conv > 0:
        odds_num = same_dept / max(cross_dept, 1)
        odds_denom = expected_same / (1 - expected_same)
        odds_ratio = odds_num / odds_denom
        print(f"  Odds ratio: {odds_ratio:.2f}")

        n_perm = 10000
        gene_list = sorted(conv_genes.keys())
        dept_list = [conv_genes[g] for g in gene_list]
        gene_idx = {g: i for i, g in enumerate(gene_list)}

        print(f"\n  Running {n_perm} permutations (shuffling dept labels)...")
        null_same = []
        for p in range(n_perm):
            perm = list(dept_list)
            random.shuffle(perm)
            perm_map = {gene_list[i]: perm[i] for i in range(len(gene_list))}
            s = sum(1 for ga, gb in pairs_conv if perm_map.get(ga) == perm_map.get(gb))
            null_same.append(s)
            if (p + 1) % 2000 == 0:
                print(f"    {p+1}/{n_perm}...")

        null_mean = np.mean(null_same)
        null_std = np.std(null_same)
        z1 = (same_dept - null_mean) / null_std if null_std > 0 else 0
        p1 = sum(1 for ns in null_same if ns >= same_dept) / n_perm

        print(f"\n  Permutation results:")
        print(f"    Observed same-dept: {same_dept}")
        print(f"    Null: {null_mean:.1f} +/- {null_std:.1f}")
        print(f"    Z = {z1:.2f}")
        pstr = f"p = {p1:.6f}" if p1 > 0 else f"p < {1/n_perm}"
        print(f"    {pstr}")

        # Per interaction type
        print(f"\n  Per interaction type:")
        type_stats = defaultdict(lambda: {"same": 0, "cross": 0})
        for ga, gb, itype, conf in interactions:
            if ga in conv_genes and gb in conv_genes:
                if conv_genes[ga] == conv_genes[gb]:
                    type_stats[itype]["same"] += 1
                else:
                    type_stats[itype]["cross"] += 1
        for it in sorted(type_stats.keys(), key=lambda x: -(type_stats[x]["same"] + type_stats[x]["cross"])):
            s, c = type_stats[it]["same"], type_stats[it]["cross"]
            rate = s / (s + c) if (s + c) > 0 else 0
            print(f"    {it:<20} same={s:>4} cross={c:>4} rate={rate:.3f}")

        results["test1_interaction_enrichment"] = {
            "validation_source": "curated protein_interactions (pathway databases)",
            "total_interactions": len(interactions),
            "both_convergence_classified": total_conv,
            "same_dept": same_dept,
            "cross_dept": cross_dept,
            "observed_same_rate": round(observed_same, 4),
            "expected_same_rate": round(expected_same, 4),
            "odds_ratio": round(odds_ratio, 2),
            "permutation_z": round(float(z1), 2),
            "permutation_p": float(p1),
            "n_permutations": n_perm,
            "per_type": {k: dict(v) for k, v in type_stats.items()},
        }
    else:
        results["test1_interaction_enrichment"] = {"status": "no_data"}

    # ==================================================================
    # TEST 2: CO-EXPRESSION CONSISTENCY (GTEx)
    # ==================================================================
    print("\n" + "=" * 70)
    print("  TEST 2: CO-EXPRESSION CONSISTENCY (GTEx)")
    print("  Validation source: GTEx tissue expression profiles")
    print("=" * 70)

    cur.execute("""
        SELECT gene_symbol, tissue, tpm_median
        FROM gtex_expression
        WHERE gene_symbol = ANY(%s)
        ORDER BY gene_symbol, tissue
    """, (list(conv_genes.keys()),))

    gene_expr = defaultdict(dict)
    for gene, tissue, tpm in cur.fetchall():
        gene_expr[gene][tissue] = float(tpm)

    all_tissues = sorted(set(t for p in gene_expr.values() for t in p))
    n_tissues = len(all_tissues)
    tissue_idx = {t: i for i, t in enumerate(all_tissues)}

    expr_genes = sorted([g for g in gene_expr if len(gene_expr[g]) >= n_tissues * 0.8
                         and g in conv_genes])
    n_expr = len(expr_genes)
    print(f"  Convergence genes with GTEx data: {n_expr}")
    print(f"  Tissues: {n_tissues}")

    if n_expr >= 50:
        expr_matrix = np.zeros((n_expr, n_tissues))
        for i, g in enumerate(expr_genes):
            for t, tpm in gene_expr[g].items():
                expr_matrix[i, tissue_idx[t]] = tpm

        expr_matrix = np.log2(expr_matrix + 1)
        means = expr_matrix.mean(axis=0)
        stds = expr_matrix.std(axis=0)
        stds[stds == 0] = 1
        expr_z = (expr_matrix - means) / stds

        MAX_PAIRS = 50000
        same_pairs = []
        cross_pairs = []
        for i in range(n_expr):
            for j in range(i + 1, n_expr):
                if conv_genes[expr_genes[i]] == conv_genes[expr_genes[j]]:
                    same_pairs.append((i, j))
                else:
                    cross_pairs.append((i, j))

        print(f"  Total same-dept pairs: {len(same_pairs)}")
        print(f"  Total cross-dept pairs: {len(cross_pairs)}")

        same_sample = random.sample(same_pairs, min(MAX_PAIRS, len(same_pairs)))
        cross_sample = random.sample(cross_pairs, min(MAX_PAIRS, len(cross_pairs)))

        print(f"  Sampling {len(same_sample)} same, {len(cross_sample)} cross...")
        same_corrs = []
        for i, j in same_sample:
            r = np.corrcoef(expr_z[i], expr_z[j])[0, 1]
            if not np.isnan(r):
                same_corrs.append(r)

        cross_corrs = []
        for i, j in cross_sample:
            r = np.corrcoef(expr_z[i], expr_z[j])[0, 1]
            if not np.isnan(r):
                cross_corrs.append(r)

        mean_same = np.mean(same_corrs)
        mean_cross = np.mean(cross_corrs)
        diff = mean_same - mean_cross
        pooled_std = np.sqrt((np.var(same_corrs) + np.var(cross_corrs)) / 2)
        d = diff / pooled_std if pooled_std > 0 else 0

        print(f"\n  Mean correlation (same dept): {mean_same:.4f} (n={len(same_corrs)})")
        print(f"  Mean correlation (cross dept): {mean_cross:.4f} (n={len(cross_corrs)})")
        print(f"  Difference: {diff:+.4f}")
        print(f"  Cohen's d: {d:.3f}")

        # Mann-Whitney U test instead of permutation (faster, more interpretable)
        from scipy.stats import mannwhitneyu
        u_stat, u_p = mannwhitneyu(same_corrs, cross_corrs, alternative='greater')
        print(f"  Mann-Whitney U (same > cross): U={u_stat:.0f}, p={u_p:.6f}")

        # Also do permutation
        n_perm2 = 5000
        all_c = same_corrs + cross_corrs
        n_same = len(same_corrs)
        null_diffs = []
        print(f"  Running {n_perm2} permutations...")
        for p in range(n_perm2):
            perm = list(range(len(all_c)))
            random.shuffle(perm)
            ps = [all_c[perm[i]] for i in range(n_same)]
            pc = [all_c[perm[i]] for i in range(n_same, len(all_c))]
            null_diffs.append(np.mean(ps) - np.mean(pc))

        z2 = (diff - np.mean(null_diffs)) / np.std(null_diffs) if np.std(null_diffs) > 0 else 0
        p2 = sum(1 for nd in null_diffs if nd >= diff) / n_perm2

        print(f"  Permutation: z={z2:.2f}, p={p2:.4f}")

        # Per-department within-correlation
        dept_expr = defaultdict(list)
        for i, g in enumerate(expr_genes):
            dept_expr[conv_genes[g]].append(i)

        print(f"\n  Per-department within-correlation (depts with >= 10 genes):")
        dept_within = {}
        for dept in sorted(dept_expr.keys(), key=lambda x: -len(dept_expr[x])):
            idxs = dept_expr[dept]
            if len(idxs) >= 10:
                within = []
                sample_idxs = idxs if len(idxs) <= 200 else random.sample(idxs, 200)
                for a in range(len(sample_idxs)):
                    for b in range(a + 1, len(sample_idxs)):
                        r = np.corrcoef(expr_z[sample_idxs[a]], expr_z[sample_idxs[b]])[0, 1]
                        if not np.isnan(r):
                            within.append(r)
                if within:
                    dept_within[dept] = round(float(np.mean(within)), 4)
                    print(f"    {dept:<22} n={len(idxs):>4} mean_r={np.mean(within):.4f} (from {len(within)} pairs)")

        results["test2_coexpression"] = {
            "validation_source": "GTEx tissue expression profiles (54 tissues)",
            "n_genes": n_expr,
            "n_tissues": n_tissues,
            "mean_corr_same_dept": round(float(mean_same), 4),
            "mean_corr_cross_dept": round(float(mean_cross), 4),
            "difference": round(float(diff), 4),
            "cohens_d": round(float(d), 3),
            "n_same_pairs": len(same_corrs),
            "n_cross_pairs": len(cross_corrs),
            "mann_whitney_U": round(float(u_stat), 0),
            "mann_whitney_p": float(u_p),
            "permutation_z": round(float(z2), 2),
            "permutation_p": float(p2),
            "dept_within_correlations": dept_within,
        }
    else:
        results["test2_coexpression"] = {"status": "insufficient_data"}

    # ==================================================================
    # TEST 3: GO CROSS-VALIDATION (Data-Learned Mapping)
    # ==================================================================
    print("\n" + "=" * 70)
    print("  TEST 3: GO CROSS-VALIDATION")
    print("  Step 1: Learn GO term → department mapping from API-classified genes")
    print("  Step 2: Apply to convergence genes that have GO terms")
    print("  Step 3: Compare predicted department to convergence department")
    print("=" * 70)

    # Step 1: Learn mapping from API-classified (GO-derived) genes
    cur.execute("""
        SELECT ga.gene_name, ga.go_terms, gdm.primary_department
        FROM go_annotations ga
        JOIN gene_department_map gdm ON ga.gene_name = gdm.gene_name
        WHERE gdm.source IN ('api', 'heuristic')
        AND ga.go_terms IS NOT NULL
        AND array_length(ga.go_terms, 1) > 0
        AND gdm.primary_department IS NOT NULL
    """)

    dept_term_counts = defaultdict(Counter)
    dept_gene_counts = Counter()
    for gene, go_terms, dept in cur.fetchall():
        dept_gene_counts[dept] += 1
        for t in go_terms:
            dept_term_counts[dept][t] += 1

    total_api = sum(dept_gene_counts.values())
    print(f"  Training set (API/heuristic-classified): {total_api} genes, {len(dept_term_counts)} departments")

    all_go_terms = set()
    for c in dept_term_counts.values():
        all_go_terms.update(c.keys())

    learned_go_map = {}
    for term in all_go_terms:
        total_with = sum(dept_term_counts[d].get(term, 0) for d in dept_term_counts)
        if total_with < 3:
            continue
        best_dept = None
        best_specificity = 0
        for dept in dept_term_counts:
            cnt = dept_term_counts[dept].get(term, 0)
            specificity = cnt / total_with
            if specificity > 0.4 and specificity > best_specificity:
                best_specificity = specificity
                best_dept = dept
        if best_dept:
            learned_go_map[term] = best_dept

    print(f"  Learned GO→dept mappings (specificity > 0.4): {len(learned_go_map)}")

    # Step 2: Apply to convergence genes
    cur.execute("""
        SELECT ga.gene_name, ga.go_terms, gdm.primary_department, gdm.all_departments
        FROM go_annotations ga
        JOIN gene_department_map gdm ON ga.gene_name = gdm.gene_name
        WHERE gdm.source = 'omnis_convergence'
        AND ga.go_terms IS NOT NULL
        AND array_length(ga.go_terms, 1) > 0
    """)

    test3_genes = []
    for gene, go_terms, conv_dept, conv_all in cur.fetchall():
        dept_votes = Counter()
        for t in go_terms:
            if t in learned_go_map:
                dept_votes[learned_go_map[t]] += 1
        if dept_votes:
            go_predicted = dept_votes.most_common(1)[0][0]
            go_all_predicted = list(dept_votes.keys())
            test3_genes.append({
                "gene": gene,
                "conv_dept": conv_dept,
                "conv_all": conv_all if conv_all else [conv_dept],
                "go_dept": go_predicted,
                "go_all": go_all_predicted,
            })

    n_with_go = len(test3_genes)
    print(f"  Convergence genes mappable via learned GO→dept: {n_with_go}")

    if n_with_go >= 20:
        n3 = n_with_go
        exact_match = sum(1 for g in test3_genes if g["conv_dept"] == g["go_dept"])
        any_overlap = sum(1 for g in test3_genes if set(g["conv_all"]) & set(g["go_all"]))

        print(f"\n  Concordance (n={n3}):")
        print(f"    Exact match (primary): {exact_match} ({exact_match/n3*100:.1f}%)")
        print(f"    Any department overlap: {any_overlap} ({any_overlap/n3*100:.1f}%)")

        conv_dept_dist = Counter(g["conv_dept"] for g in test3_genes)
        go_dept_dist = Counter(g["go_dept"] for g in test3_genes)
        all_depts = set(list(conv_dept_dist.keys()) + list(go_dept_dist.keys()))
        expected_exact = sum(
            (conv_dept_dist.get(d, 0) / n3) * (go_dept_dist.get(d, 0) / n3)
            for d in all_depts
        )
        enrichment = (exact_match / n3) / expected_exact if expected_exact > 0 else float('inf')

        print(f"\n  Expected exact match (chance): {expected_exact:.4f} ({expected_exact*100:.1f}%)")
        print(f"  Observed: {exact_match/n3:.4f} ({exact_match/n3*100:.1f}%)")
        print(f"  Enrichment: {enrichment:.1f}x over chance")

        # Permutation test
        n_perm3 = 10000
        conv_depts = [g["conv_dept"] for g in test3_genes]
        go_depts = [g["go_dept"] for g in test3_genes]
        null_exact = []
        print(f"\n  Running {n_perm3} permutations...")
        for p in range(n_perm3):
            perm = list(go_depts)
            random.shuffle(perm)
            null_exact.append(sum(1 for i in range(n3) if conv_depts[i] == perm[i]))
            if (p + 1) % 2000 == 0:
                print(f"    {p+1}/{n_perm3}...")

        null_mean = np.mean(null_exact)
        null_std = np.std(null_exact)
        z3 = (exact_match - null_mean) / null_std if null_std > 0 else 0
        p3 = sum(1 for ne in null_exact if ne >= exact_match) / n_perm3

        print(f"\n  Permutation results:")
        print(f"    Observed exact matches: {exact_match}")
        print(f"    Null: {null_mean:.1f} +/- {null_std:.1f}")
        print(f"    Z = {z3:.2f}")
        pstr3 = f"p = {p3:.6f}" if p3 > 0 else f"p < {1/n_perm3}"
        print(f"    {pstr3}")

        # Per-department confusion
        print(f"\n  Per-department concordance (top 15):")
        dept_conf = defaultdict(lambda: Counter())
        for g in test3_genes:
            dept_conf[g["conv_dept"]][g["go_dept"]] += 1

        for cd in sorted(dept_conf.keys(), key=lambda x: -sum(dept_conf[x].values()))[:15]:
            total = sum(dept_conf[cd].values())
            self_match = dept_conf[cd].get(cd, 0)
            purity = self_match / total if total > 0 else 0
            top3 = dept_conf[cd].most_common(3)
            top3_str = ", ".join(f"{d}:{c}" for d, c in top3)
            print(f"    {cd:<22} n={total:>5} self={purity:.3f} [{top3_str}]")

        print(f"\n  GO-derived department distribution:")
        for d, c in go_dept_dist.most_common(15):
            print(f"    {d:<22} n={c}")

        print(f"\n  Convergence department distribution:")
        for d, c in conv_dept_dist.most_common(15):
            print(f"    {d:<22} n={c}")

        results["test3_go_crossvalidation"] = {
            "validation_source": "Data-learned GO→dept mapping (trained on 25K API-classified genes, applied to convergence genes)",
            "n_training_genes": total_api,
            "n_learned_go_mappings": len(learned_go_map),
            "n_convergence_with_go": n_with_go,
            "exact_match": exact_match,
            "exact_match_rate": round(exact_match / n3, 4),
            "any_overlap": any_overlap,
            "any_overlap_rate": round(any_overlap / n3, 4),
            "expected_by_chance": round(expected_exact, 4),
            "enrichment_over_chance": round(enrichment, 1),
            "permutation_z": round(float(z3), 2),
            "permutation_p": float(p3),
            "n_permutations": n_perm3,
        }
    else:
        results["test3_go_crossvalidation"] = {"status": "insufficient_overlap"}

    # ==================================================================
    # SUMMARY
    # ==================================================================
    conn.close()
    elapsed = time.time() - t0
    results["elapsed_seconds"] = round(elapsed, 1)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    t1r = results.get("test1_interaction_enrichment", {})
    t2r = results.get("test2_coexpression", {})
    t3r = results.get("test3_go_crossvalidation", {})

    print(f"\n  Test 1 — Interaction Enrichment (curated, non-circular):")
    if "permutation_z" in t1r:
        print(f"    Same-dept rate: {t1r['observed_same_rate']*100:.1f}% vs {t1r['expected_same_rate']*100:.1f}% expected")
        print(f"    Odds ratio: {t1r['odds_ratio']:.2f}")
        print(f"    Z = {t1r['permutation_z']:.1f}, p = {t1r['permutation_p']}")
        verdict1 = "POSITIVE" if t1r['permutation_p'] < 0.05 else "NOT SIGNIFICANT"
        print(f"    Verdict: {verdict1}")

    print(f"\n  Test 2 — GTEx Co-expression:")
    if "permutation_z" in t2r:
        print(f"    Same-dept r = {t2r['mean_corr_same_dept']:.4f} vs cross r = {t2r['mean_corr_cross_dept']:.4f}")
        print(f"    Diff = {t2r['difference']:+.4f}, d = {t2r['cohens_d']:.3f}")
        print(f"    Z = {t2r['permutation_z']:.1f}, p = {t2r['permutation_p']}")
        verdict2 = "POSITIVE" if t2r['permutation_p'] < 0.05 else "NOT SIGNIFICANT"
        print(f"    Verdict: {verdict2}")

    print(f"\n  Test 3 — GO Cross-Validation:")
    if "permutation_z" in t3r:
        print(f"    Exact match: {t3r['exact_match_rate']*100:.1f}% vs {t3r['expected_by_chance']*100:.1f}% chance")
        print(f"    Any overlap: {t3r['any_overlap_rate']*100:.1f}%")
        print(f"    Enrichment: {t3r['enrichment_over_chance']:.1f}x")
        print(f"    Z = {t3r['permutation_z']:.1f}, p = {t3r['permutation_p']}")
        verdict3 = "POSITIVE" if t3r['permutation_p'] < 0.05 else "NOT SIGNIFICANT"
        print(f"    Verdict: {verdict3}")

    print(f"\n  Elapsed: {elapsed:.1f}s")

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
