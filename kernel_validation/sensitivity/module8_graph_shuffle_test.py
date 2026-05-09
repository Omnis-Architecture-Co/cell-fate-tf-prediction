#!/usr/bin/env python3
"""
Module 8 Dispatch Graph Shuffle Test (10-Drug Pilot)
=====================================================
Tests whether Module 8's side-effect predictions depend on the specific
token-sharing graph topology or just on the static scoring tables
(department priors, phenotype maps, GTEx weights) applied to whatever
genes the cascade finds.

Architecture under test (claude_ai_service.py _predict_side_effects):
  - BFS cascade expansion through protein_tokens_v2 token-sharing graph
  - Hop weights: {1: 1.0, 2: 0.5, 3: 0.25, 4: 0.125, 5: 0.0625, 6: 0.03125}
  - MAX_PER_HOP = 150, MIN_SHARED_TOKENS = 2
  - Firewall: target gene excluded from phenotype lookups
  - Scoring: gene_phenotype_map + dept_phenotype_priors + token uniqueness

Shuffle method: Degree-preserving bipartite edge swaps on the
protein-token assignment graph. This rewires which proteins share
tokens while preserving each protein's token count and each token's
protein count.

Prediction: If cascade topology matters, shuffling should degrade R@10
for high-performing drugs. If signal comes only from static tables,
R@10 should be unchanged.

5 expected-high: Atorvastatin (HMGCR), Omeprazole (ATP4A),
                  Haloperidol (DRD2), Dapagliflozin (SLC5A2), Fluoxetine (SLC6A4)
5 expected-low:  Imatinib (BCR-ABL1→ABL1), Isoniazid (INHA),
                  Adalimumab (TNF), Methotrexate (DHFR), Cisplatin (DNA→ATM)

Output: validation/sensitivity/module8_graph_shuffle_results.json
"""

import json
import math
import os
import random
import time
from collections import defaultdict, Counter

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

np.random.seed(42)
random.seed(42)

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "module8_graph_shuffle_results.json")

PILOT_DRUGS = [
    {"drug": "Atorvastatin",  "gene": "HMGCR",  "expected": "high",
     "ground_truth": ["myopathy", "hepatotoxicity", "metabolic_diabetes", "gi_dysmotility", "neurological_general", "peripheral_neuropathy"]},
    {"drug": "Omeprazole",    "gene": "ATP4A",  "expected": "high",
     "ground_truth": ["gi_dysmotility", "skeletal_disorders", "renal_disorders", "neurological_general", "hematological_anemia", "immune_dysregulation"]},
    {"drug": "Haloperidol",   "gene": "DRD2",   "expected": "high",
     "ground_truth": ["neurological_general", "cardiac_arrhythmia", "endocrine_disorders", "gi_dysmotility", "psychiatric_disorders"]},
    {"drug": "Dapagliflozin", "gene": "SLC5A2", "expected": "high",
     "ground_truth": ["renal_disorders", "metabolic_diabetes", "immune_dysregulation", "vascular_disorders", "gi_dysmotility"]},
    {"drug": "Fluoxetine",    "gene": "SLC6A4", "expected": "high",
     "ground_truth": ["gi_dysmotility", "neurological_general", "psychiatric_disorders", "sexual_reproductive", "bleeding_coagulation", "weight_metabolic"]},
    {"drug": "Imatinib",      "gene": "ABL1",   "expected": "low",
     "ground_truth": ["gi_dysmotility", "myopathy", "vascular_disorders", "hepatotoxicity", "skin_disorders", "immune_cytopenia", "renal_disorders", "cardiac_disorders"]},
    {"drug": "Isoniazid",     "gene": "INHA",   "expected": "low",
     "ground_truth": ["hepatotoxicity", "peripheral_neuropathy", "neurological_general", "hematological_anemia", "immune_dysregulation", "skin_disorders"]},
    {"drug": "Adalimumab",    "gene": "TNF",    "expected": "low",
     "ground_truth": ["immune_dysregulation", "hepatotoxicity", "cancer_risk", "hematological_anemia", "neurological_general", "skin_disorders", "cardiac_disorders"]},
    {"drug": "Methotrexate",  "gene": "DHFR",   "expected": "low",
     "ground_truth": ["hepatotoxicity", "immune_cytopenia", "renal_disorders", "pulmonary_toxicity", "gi_dysmotility", "teratogenicity", "skin_disorders", "immune_dysregulation"]},
    {"drug": "Cisplatin",     "gene": "ATM",    "expected": "low",
     "ground_truth": ["renal_disorders", "hearing_loss", "peripheral_neuropathy", "gi_dysmotility", "immune_cytopenia", "hematological_anemia", "vascular_disorders"]},
]

CATEGORY_SYNONYMS = {
    "cardiac_arrhythmia": ["qt_prolongation", "ion_channel_dysfunction"],
    "cardiac_disorders": ["cardiotoxicity"],
    "renal_disorders": ["nephrotoxicity"],
    "hearing_loss": ["ototoxicity"],
    "immune_cytopenia": ["myelosuppression", "hematological_anemia"],
    "gi_dysmotility": ["pancreatitis"],
    "hepatotoxicity": [],
    "myopathy": ["skeletal_disorders"],
    "peripheral_neuropathy": [],
    "bleeding_coagulation": [],
    "vascular_disorders": ["thrombosis_risk"],
    "skin_disorders": ["hypersensitivity", "infusion_reactions"],
    "neurological_general": ["psychiatric_disorders"],
    "vision_disorders": [],
    "developmental_structural": ["teratogenicity"],
    "metabolic_diabetes": ["weight_metabolic", "endocrine_disorders"],
    "cancer_risk": ["tumor_lysis"],
    "immune_dysregulation": [],
    "metabolic_lipid": [],
    "respiratory_disorders": ["pulmonary_toxicity"],
    "sexual_reproductive": [],
}

HOP_WEIGHTS = {1: 1.0, 2: 0.5, 3: 0.25, 4: 0.125, 5: 0.0625, 6: 0.03125}
MIN_SHARED_TOKENS = 2
MAX_PER_HOP = 150
MAX_HOPS = 6
N_SHUFFLES = 3


def normalize_category(cat):
    cat = cat.strip().lower()
    for canon, synonyms in CATEGORY_SYNONYMS.items():
        if cat == canon or cat in synonyms:
            return canon
    return cat


def compute_recall_at_k(predicted_categories, ground_truth, k=10):
    gt_set = set(normalize_category(c) for c in ground_truth)
    pred_list = [normalize_category(c) for c in predicted_categories[:k]]
    if not gt_set:
        return 0.0
    hits = len(gt_set & set(pred_list))
    return hits / len(gt_set)


def run_cascade_bfs(cur, target_gene, target_uniprot, target_tokens,
                    token_to_proteins=None, protein_to_tokens_map=None):
    """
    Replicate Module 8's BFS cascade expansion.
    If token_to_proteins is provided, use that in-memory graph instead of DB queries.
    protein_to_tokens_map: optional precomputed {uid: set(tokens)} for fast token lookups.
    Returns dict of {hop_num: {uniprot_id: gene_name}}
    """
    in_cascade = {target_uniprot}
    frontier_tokens = {target_uniprot: target_tokens}
    hop_proteins = {}

    for hop_num in range(1, MAX_HOPS + 1):
        all_frontier_tokens = set()
        for ts in frontier_tokens.values():
            all_frontier_tokens.update(ts)
        if not all_frontier_tokens:
            break

        hop_threshold = MIN_SHARED_TOKENS
        if hop_num == 1 and len(target_tokens) < MIN_SHARED_TOKENS:
            hop_threshold = max(1, len(target_tokens))

        if token_to_proteins is not None:
            candidates = _find_candidates_memory(
                all_frontier_tokens, in_cascade, hop_threshold, token_to_proteins, cur
            )
        else:
            candidates = _find_candidates_db(
                all_frontier_tokens, in_cascade, hop_threshold, cur
            )

        scored = []
        for cand_uid, cand_gene, cand_matched_tokens in candidates:
            best_shared = 0
            for f_uid, f_tokens in frontier_tokens.items():
                shared = len(cand_matched_tokens & f_tokens)
                if shared > best_shared:
                    best_shared = shared
            if best_shared >= hop_threshold:
                scored.append((cand_uid, best_shared, cand_gene))

        scored.sort(key=lambda x: -x[1])
        seen_genes = set()
        deduped = []
        for uid, shared_count, gname in scored:
            if gname and gname in seen_genes:
                continue
            if gname:
                seen_genes.add(gname)
            deduped.append((uid, shared_count, gname))
            if len(deduped) >= MAX_PER_HOP:
                break
        scored = deduped

        if not scored:
            break

        hop_prots = {}
        for uid, shared_count, gname in scored:
            hop_prots[uid] = gname
            in_cascade.add(uid)
        hop_proteins[hop_num] = hop_prots

        new_uids = list(hop_prots.keys())
        if protein_to_tokens_map is not None:
            frontier_tokens = {uid: protein_to_tokens_map.get(uid, set()) for uid in new_uids}
        else:
            frontier_tokens = _batch_get_token_sets(new_uids, cur, token_to_proteins)

    return hop_proteins


def _find_candidates_db(frontier_tokens, exclude_set, threshold, cur):
    exclude_list = list(exclude_set)
    cur.execute("""
        SELECT pt.uniprot_id, pc.gene_name,
               ARRAY_AGG(DISTINCT pt.token_hex) as matched_tokens
        FROM protein_tokens_v2 pt
        JOIN protein_catalog pc ON pt.uniprot_id = pc.uniprot_id
        WHERE pt.token_hex = ANY(%s)
          AND pt.uniprot_id != ALL(%s)
        GROUP BY pt.uniprot_id, pc.gene_name
        HAVING COUNT(DISTINCT pt.token_hex) >= %s
    """, (list(frontier_tokens), exclude_list, threshold))
    results = []
    for r in cur.fetchall():
        results.append((r['uniprot_id'], r['gene_name'], set(r['matched_tokens'])))
    return results


def _find_candidates_memory(frontier_tokens, exclude_set, threshold, token_to_proteins, cur):
    protein_matched = defaultdict(set)
    for tok in frontier_tokens:
        for uid in token_to_proteins.get(tok, set()):
            if uid not in exclude_set:
                protein_matched[uid].add(tok)

    candidates = []
    for uid, matched in protein_matched.items():
        if len(matched) >= threshold:
            candidates.append((uid, GENE_CACHE.get(uid), matched))
    return candidates


def _batch_resolve_genes(uids, cur):
    result = {}
    bsz = 500
    for i in range(0, len(uids), bsz):
        batch = uids[i:i+bsz]
        cur.execute("""
            SELECT uniprot_id, gene_name FROM protein_catalog
            WHERE uniprot_id = ANY(%s)
        """, (batch,))
        for r in cur.fetchall():
            result[r['uniprot_id']] = r['gene_name']
    return result


def _batch_get_token_sets(uids, cur, token_to_proteins=None):
    if not uids:
        return {}
    if token_to_proteins is not None:
        protein_to_tokens = defaultdict(set)
        for tok, prots in token_to_proteins.items():
            for uid in prots:
                if uid in set(uids):
                    protein_to_tokens[uid].add(tok)
        return dict(protein_to_tokens)

    sets = {}
    bsz = 500
    for i in range(0, len(uids), bsz):
        batch = uids[i:i+bsz]
        cur.execute("""
            SELECT uniprot_id, ARRAY_AGG(DISTINCT token_hex) as tokens
            FROM protein_tokens_v2 WHERE uniprot_id = ANY(%s)
            GROUP BY uniprot_id
        """, (batch,))
        for r in cur.fetchall():
            sets[r['uniprot_id']] = set(r['tokens'])
    return sets


DEPT_PRIORS_CACHE = None

def _ensure_dept_priors(cur):
    global DEPT_PRIORS_CACHE
    if DEPT_PRIORS_CACHE is not None:
        return DEPT_PRIORS_CACHE
    cur.execute("SELECT department, phenotype_category, prior_weight FROM dept_phenotype_priors")
    DEPT_PRIORS_CACHE = defaultdict(list)
    for r in cur.fetchall():
        DEPT_PRIORS_CACHE[r['department']].append({
            'category': r['phenotype_category'],
            'prior_weight': float(r['prior_weight']),
        })
    return DEPT_PRIORS_CACHE


def score_cascade(hop_proteins, target_gene, cur, use_cache=False):
    """
    Replicate Module 8's scoring pipeline (Steps 5-8).
    Returns list of (category, score) sorted by score descending.
    """
    all_cascade_genes = set()
    for hp, proteins in hop_proteins.items():
        all_cascade_genes.update(v for v in proteins.values() if v)
    all_cascade_genes.discard(target_gene)
    all_cascade_genes.discard(None)

    if not all_cascade_genes:
        return []

    if use_cache:
        dept_map = {g: DEPT_CACHE.get(g, ['Unknown']) for g in all_cascade_genes}
    else:
        dept_map = _get_dept_assignments(list(all_cascade_genes), cur)

    hop_dept_genes = defaultdict(lambda: defaultdict(set))
    for hop_num, proteins in hop_proteins.items():
        for uid, gn in proteins.items():
            if gn and gn != target_gene:
                for dept in dept_map.get(gn, ['Unknown']):
                    hop_dept_genes[hop_num][dept].add(gn)

    if use_cache:
        gene_phenotypes = {g: PHENO_CACHE.get(g, []) for g in all_cascade_genes if g != target_gene}
    else:
        gene_phenotypes = _get_phenotypes(list(all_cascade_genes), target_gene, cur)

    dept_priors = _ensure_dept_priors(cur)

    phenotype_scores = defaultdict(lambda: {'score': 0.0, 'genes': set(), 'hops': set(), 'depts': set()})

    for hop_level, depts in hop_dept_genes.items():
        hw = HOP_WEIGHTS.get(hop_level, 0.5 ** (hop_level - 1))
        for dept, genes_in_dept in depts.items():
            for gn in genes_in_dept:
                if gn in gene_phenotypes:
                    for ph in gene_phenotypes[gn]:
                        cat = ph['category']
                        phenotype_scores[cat]['score'] += hw
                        phenotype_scores[cat]['genes'].add(gn)
                        phenotype_scores[cat]['hops'].add(hop_level)
                        phenotype_scores[cat]['depts'].add(dept)

            if dept in dept_priors:
                for prior in dept_priors[dept]:
                    cat = prior['category']
                    prior_contribution = prior['prior_weight'] * hw * len(genes_in_dept) * 0.1
                    phenotype_scores[cat]['score'] += prior_contribution
                    phenotype_scores[cat]['hops'].add(hop_level)
                    phenotype_scores[cat]['depts'].add(dept)

    for cat, data in phenotype_scores.items():
        gc = len(data['genes'])
        if gc > 0:
            data['score'] = data['score'] / math.sqrt(gc)

    ranked = sorted(
        [(cat, data['score']) for cat, data in phenotype_scores.items() if data['score'] >= 0.1],
        key=lambda x: -x[1]
    )
    return ranked


def _get_dept_assignments(gene_names, cur):
    result = defaultdict(list)
    bsz = 500
    for i in range(0, len(gene_names), bsz):
        batch = gene_names[i:i+bsz]
        cur.execute("""
            SELECT gene_name, all_departments
            FROM gene_department_map WHERE gene_name = ANY(%s)
        """, (batch,))
        for r in cur.fetchall():
            depts = r['all_departments'] or []
            result[r['gene_name']] = depts if depts else ['Unknown']
    return result


def _get_phenotypes(gene_names, exclude_gene, cur):
    result = defaultdict(list)
    bsz = 500
    for i in range(0, len(gene_names), bsz):
        batch = gene_names[i:i+bsz]
        cur.execute("""
            SELECT gene_name, phenotype_category, source
            FROM gene_phenotype_map
            WHERE gene_name = ANY(%s) AND gene_name != %s
        """, (batch, exclude_gene))
        for r in cur.fetchall():
            result[r['gene_name']].append({'category': r['phenotype_category'], 'source': r['source']})
    return result


def build_bipartite_graph(cur, relevant_tokens):
    """
    Load token→protein assignments for all tokens that appear
    in the relevant cascades. Returns {token_hex: set(uniprot_ids)}.
    Also pre-caches protein_catalog gene names for fast lookup.
    """
    token_to_proteins = defaultdict(set)
    protein_to_tokens = defaultdict(set)

    token_list = list(relevant_tokens)
    bsz = 1000
    for i in range(0, len(token_list), bsz):
        batch = token_list[i:i+bsz]
        cur.execute("""
            SELECT token_hex, uniprot_id
            FROM protein_tokens_v2
            WHERE token_hex = ANY(%s)
        """, (batch,))
        for r in cur.fetchall():
            token_to_proteins[r['token_hex']].add(r['uniprot_id'])
            protein_to_tokens[r['uniprot_id']].add(r['token_hex'])

    return token_to_proteins, protein_to_tokens


GENE_CACHE = {}
DEPT_CACHE = {}
PHENO_CACHE = {}


def preload_caches(cur, protein_to_tokens):
    global GENE_CACHE, DEPT_CACHE, PHENO_CACHE
    all_uids = list(protein_to_tokens.keys())
    print(f"  Pre-caching gene names for {len(all_uids)} proteins...")
    bsz = 2000
    for i in range(0, len(all_uids), bsz):
        batch = all_uids[i:i+bsz]
        cur.execute("SELECT uniprot_id, gene_name FROM protein_catalog WHERE uniprot_id = ANY(%s)", (batch,))
        for r in cur.fetchall():
            GENE_CACHE[r['uniprot_id']] = r['gene_name']

    all_genes = list(set(GENE_CACHE.values()))
    print(f"  Pre-caching departments for {len(all_genes)} genes...")
    for i in range(0, len(all_genes), bsz):
        batch = all_genes[i:i+bsz]
        cur.execute("SELECT gene_name, all_departments FROM gene_department_map WHERE gene_name = ANY(%s)", (batch,))
        for r in cur.fetchall():
            depts = r['all_departments'] or []
            DEPT_CACHE[r['gene_name']] = depts if depts else ['Unknown']

    print(f"  Pre-caching phenotypes for {len(all_genes)} genes...")
    for i in range(0, len(all_genes), bsz):
        batch = all_genes[i:i+bsz]
        cur.execute("SELECT gene_name, phenotype_category, source FROM gene_phenotype_map WHERE gene_name = ANY(%s)", (batch,))
        for r in cur.fetchall():
            if r['gene_name'] not in PHENO_CACHE:
                PHENO_CACHE[r['gene_name']] = []
            PHENO_CACHE[r['gene_name']].append({'category': r['phenotype_category'], 'source': r['source']})

    print(f"  Cached: {len(GENE_CACHE)} genes, {len(DEPT_CACHE)} depts, {len(PHENO_CACHE)} pheno")


def degree_preserving_shuffle(token_to_proteins, protein_to_tokens, n_swaps=None):
    """
    Degree-preserving bipartite edge swaps.
    Swap (protA, tokX) + (protB, tokY) → (protA, tokY) + (protB, tokX)
    if the new edges don't already exist.
    """
    ttp = {t: set(ps) for t, ps in token_to_proteins.items()}
    ptt = {p: set(ts) for p, ts in protein_to_tokens.items()}

    edges = [(p, t) for t, prots in ttp.items() for p in prots]
    n_edges = len(edges)
    if n_swaps is None:
        n_swaps = n_edges * 5

    successful = 0
    attempts = 0
    max_attempts = n_swaps * 10

    while successful < n_swaps and attempts < max_attempts:
        attempts += 1
        i1 = random.randint(0, n_edges - 1)
        i2 = random.randint(0, n_edges - 1)
        if i1 == i2:
            continue

        p1, t1 = edges[i1]
        p2, t2 = edges[i2]

        if t1 == t2 or p1 == p2:
            continue

        if t2 in ptt[p1] or t1 in ptt[p2]:
            continue

        ttp[t1].discard(p1)
        ttp[t1].add(p2)
        ttp[t2].discard(p2)
        ttp[t2].add(p1)

        ptt[p1].discard(t1)
        ptt[p1].add(t2)
        ptt[p2].discard(t2)
        ptt[p2].add(t1)

        edges[i1] = (p1, t2)
        edges[i2] = (p2, t1)
        successful += 1

    return ttp, ptt, successful


def main():
    t0 = time.time()
    print("=" * 70)
    print("  MODULE 8 DISPATCH GRAPH SHUFFLE TEST (10-Drug Pilot)")
    print("=" * 70)

    db_url = os.environ.get('BETA_DATABASE_URL', os.environ.get('DATABASE_URL'))
    conn = psycopg2.connect(db_url)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    results = {"drugs": {}, "summary": {}}

    # ======================================================================
    # PHASE 1: Run real cascades for all 10 drugs
    # ======================================================================
    print("\n" + "=" * 70)
    print("  PHASE 1: Real cascade runs")
    print("=" * 70)

    all_relevant_tokens = set()
    drug_data = {}

    for drug_info in PILOT_DRUGS:
        drug = drug_info["drug"]
        gene = drug_info["gene"]
        gt = drug_info["ground_truth"]

        cur.execute("""
            SELECT uniprot_id, gene_name FROM protein_catalog
            WHERE gene_name = %s
            ORDER BY CASE WHEN uniprot_id ~ '^[OPQ][0-9][A-Z0-9]{3}[0-9]$' THEN 0 ELSE 1 END,
                     token_count DESC
            LIMIT 1
        """, (gene,))
        target_row = cur.fetchone()
        if not target_row:
            print(f"  {drug} ({gene}): gene not found in protein_catalog — SKIPPED")
            results["drugs"][drug] = {"status": "gene_not_found", "gene": gene}
            continue

        target_uniprot = target_row['uniprot_id']

        cur.execute("""
            SELECT DISTINCT token_hex FROM protein_tokens_v2
            WHERE uniprot_id = %s
        """, (target_uniprot,))
        target_tokens = set(r['token_hex'] for r in cur.fetchall())

        if not target_tokens:
            print(f"  {drug} ({gene}): no tokens — SKIPPED")
            results["drugs"][drug] = {"status": "no_tokens", "gene": gene}
            continue

        print(f"\n  {drug} ({gene}, {target_uniprot}, {len(target_tokens)} tokens):")

        hop_proteins = run_cascade_bfs(cur, gene, target_uniprot, target_tokens)
        ranked = score_cascade(hop_proteins, gene, cur)

        total_cascade = sum(len(hp) for hp in hop_proteins.values())
        predicted_cats = [cat for cat, score in ranked]
        r_at_10 = compute_recall_at_k(predicted_cats, gt, k=10)

        cascade_tokens = set()
        cascade_tokens.update(target_tokens)
        for hp, prots in hop_proteins.items():
            uids = list(prots.keys())
            tsets = _batch_get_token_sets(uids, cur)
            for ts in tsets.values():
                cascade_tokens.update(ts)
        all_relevant_tokens.update(cascade_tokens)

        hop_sizes = {h: len(p) for h, p in sorted(hop_proteins.items())}
        print(f"    Cascade: {total_cascade} proteins, hops: {hop_sizes}")
        print(f"    Top 10 predictions: {predicted_cats[:10]}")
        print(f"    Ground truth: {gt}")
        print(f"    R@10 = {r_at_10:.3f} ({int(r_at_10 * len(gt))}/{len(gt)})")

        drug_data[drug] = {
            "gene": gene,
            "target_uniprot": target_uniprot,
            "target_tokens": target_tokens,
            "real_cascade": hop_proteins,
            "real_ranked": ranked,
            "real_r_at_10": r_at_10,
            "ground_truth": gt,
            "cascade_size": total_cascade,
            "hop_sizes": hop_sizes,
        }

        results["drugs"][drug] = {
            "gene": gene,
            "expected": drug_info["expected"],
            "cascade_size": total_cascade,
            "hop_sizes": hop_sizes,
            "real_r_at_10": round(r_at_10, 3),
            "real_top10": predicted_cats[:10],
            "ground_truth": gt,
            "n_ground_truth": len(gt),
        }

    # ======================================================================
    # PHASE 2: Load bipartite graph and shuffle
    # ======================================================================
    print("\n" + "=" * 70)
    print(f"  PHASE 2: Building bipartite graph ({len(all_relevant_tokens)} tokens)")
    print("=" * 70)

    token_to_proteins, protein_to_tokens = build_bipartite_graph(cur, all_relevant_tokens)
    n_edges = sum(len(ps) for ps in token_to_proteins.values())
    print(f"  Loaded: {len(token_to_proteins)} tokens, {len(protein_to_tokens)} proteins, {n_edges} edges")

    preload_caches(cur, protein_to_tokens)

    # ======================================================================
    # PHASE 3: Shuffled cascade runs
    # ======================================================================
    print(f"\n" + "=" * 70)
    print(f"  PHASE 3: Running {N_SHUFFLES} shuffled cascades per drug")
    print("=" * 70)

    shuffled_r10 = {drug: [] for drug in drug_data}

    for shuf_idx in range(N_SHUFFLES):
        t_shuf = time.time()
        ttp_shuf, ptt_shuf, n_swaps = degree_preserving_shuffle(
            token_to_proteins, protein_to_tokens, n_swaps=n_edges // 2
        )
        print(f"\n  Shuffle {shuf_idx+1}/{N_SHUFFLES} ({n_swaps} swaps, {time.time()-t_shuf:.1f}s)")

        for drug, dd in drug_data.items():
            hop_proteins_shuf = run_cascade_bfs(
                cur, dd["gene"], dd["target_uniprot"], dd["target_tokens"],
                token_to_proteins=ttp_shuf, protein_to_tokens_map=ptt_shuf
            )
            ranked_shuf = score_cascade(hop_proteins_shuf, dd["gene"], cur, use_cache=True)
            predicted_shuf = [cat for cat, score in ranked_shuf]
            r10_shuf = compute_recall_at_k(predicted_shuf, dd["ground_truth"], k=10)
            shuffled_r10[drug].append(r10_shuf)

            cascade_shuf = sum(len(hp) for hp in hop_proteins_shuf.values())
            print(f"    {drug:20s}: R@10={r10_shuf:.3f} (cascade={cascade_shuf})")

    # ======================================================================
    # PHASE 4: Analysis
    # ======================================================================
    print("\n" + "=" * 70)
    print("  PHASE 4: Results")
    print("=" * 70)

    for drug, dd in drug_data.items():
        real_r10 = dd["real_r_at_10"]
        shuf_vals = shuffled_r10[drug]
        mean_shuf = np.mean(shuf_vals)
        std_shuf = np.std(shuf_vals)
        delta = real_r10 - mean_shuf
        z = delta / std_shuf if std_shuf > 0 else 0
        p = sum(1 for s in shuf_vals if s >= real_r10) / len(shuf_vals) if shuf_vals else 1.0

        exp = results["drugs"][drug].get("expected", "?")
        print(f"\n  {drug} ({exp}):")
        print(f"    Real R@10:     {real_r10:.3f}")
        print(f"    Shuffled mean: {mean_shuf:.3f} ± {std_shuf:.3f}")
        print(f"    Δ (real-shuf): {delta:+.3f}")
        print(f"    Z-score:       {z:.2f}")
        print(f"    p-value:       {p:.4f}")

        results["drugs"][drug].update({
            "shuffled_mean_r10": round(float(mean_shuf), 3),
            "shuffled_std_r10": round(float(std_shuf), 3),
            "shuffled_r10_values": [round(v, 3) for v in shuf_vals],
            "delta_r10": round(float(delta), 3),
            "z_score": round(float(z), 2),
            "p_value": round(float(p), 4),
        })

    # Summary
    high_drugs = [d for d in drug_data if results["drugs"][d].get("expected") == "high"]
    low_drugs = [d for d in drug_data if results["drugs"][d].get("expected") == "low"]

    high_real = np.mean([dd["real_r_at_10"] for d, dd in drug_data.items() if d in high_drugs]) if high_drugs else 0
    high_shuf = np.mean([np.mean(shuffled_r10[d]) for d in high_drugs]) if high_drugs else 0
    low_real = np.mean([dd["real_r_at_10"] for d, dd in drug_data.items() if d in low_drugs]) if low_drugs else 0
    low_shuf = np.mean([np.mean(shuffled_r10[d]) for d in low_drugs]) if low_drugs else 0

    all_real = np.mean([dd["real_r_at_10"] for dd in drug_data.values()])
    all_shuf = np.mean([np.mean(shuffled_r10[d]) for d in drug_data])

    print(f"\n  {'='*50}")
    print(f"  AGGREGATE SUMMARY")
    print(f"  {'='*50}")
    print(f"  High-expected drugs (n={len(high_drugs)}):")
    print(f"    Real R@10:     {high_real:.3f}")
    print(f"    Shuffled R@10: {high_shuf:.3f}")
    print(f"    Δ:             {high_real - high_shuf:+.3f}")
    print(f"\n  Low-expected drugs (n={len(low_drugs)}):")
    print(f"    Real R@10:     {low_real:.3f}")
    print(f"    Shuffled R@10: {low_shuf:.3f}")
    print(f"    Δ:             {low_real - low_shuf:+.3f}")
    print(f"\n  All drugs (n={len(drug_data)}):")
    print(f"    Real R@10:     {all_real:.3f}")
    print(f"    Shuffled R@10: {all_shuf:.3f}")
    print(f"    Δ:             {all_real - all_shuf:+.3f}")

    topology_dependent = sum(1 for d in drug_data if results["drugs"][d].get("delta_r10", 0) > 0.05)
    topology_independent = sum(1 for d in drug_data if abs(results["drugs"][d].get("delta_r10", 0)) <= 0.05)

    print(f"\n  Topology-dependent (Δ > 0.05):   {topology_dependent}/{len(drug_data)}")
    print(f"  Topology-independent (|Δ| ≤ 0.05): {topology_independent}/{len(drug_data)}")

    if all_real - all_shuf < 0.05:
        verdict = "TOPOLOGY INDEPENDENT — signal primarily from static scoring tables"
    elif all_real - all_shuf > 0.15:
        verdict = "TOPOLOGY DEPENDENT — cascade wiring drives predictions"
    else:
        verdict = "MIXED — partial topology dependence"
    print(f"\n  VERDICT: {verdict}")

    results["summary"] = {
        "n_drugs": len(drug_data),
        "n_shuffles": N_SHUFFLES,
        "high_real_r10": round(float(high_real), 3),
        "high_shuf_r10": round(float(high_shuf), 3),
        "low_real_r10": round(float(low_real), 3),
        "low_shuf_r10": round(float(low_shuf), 3),
        "all_real_r10": round(float(all_real), 3),
        "all_shuf_r10": round(float(all_shuf), 3),
        "delta_all": round(float(all_real - all_shuf), 3),
        "topology_dependent_count": topology_dependent,
        "topology_independent_count": topology_independent,
        "verdict": verdict,
    }

    elapsed = time.time() - t0
    results["elapsed_seconds"] = round(elapsed, 1)

    conn.close()

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {OUTPUT}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
