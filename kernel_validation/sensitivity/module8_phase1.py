#!/usr/bin/env python3
"""Phase 1: Run real cascades and save intermediate data for shuffle test."""
import json, math, os, time, pickle
from collections import defaultdict
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

np.random.seed(42)

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

HOP_WEIGHTS = {1: 1.0, 2: 0.5, 3: 0.25, 4: 0.125, 5: 0.0625, 6: 0.03125}
MIN_SHARED_TOKENS = 2
MAX_PER_HOP = 150
MAX_HOPS = 6

CATEGORY_SYNONYMS = {
    "cardiac_arrhythmia": ["qt_prolongation", "ion_channel_dysfunction"],
    "cardiac_disorders": ["cardiotoxicity"],
    "renal_disorders": ["nephrotoxicity"],
    "hearing_loss": ["ototoxicity"],
    "immune_cytopenia": ["myelosuppression", "hematological_anemia"],
    "gi_dysmotility": ["pancreatitis"],
    "myopathy": ["skeletal_disorders"],
    "neurological_general": ["psychiatric_disorders"],
    "developmental_structural": ["teratogenicity"],
    "metabolic_diabetes": ["weight_metabolic", "endocrine_disorders"],
    "cancer_risk": ["tumor_lysis"],
    "skin_disorders": ["hypersensitivity", "infusion_reactions"],
    "respiratory_disorders": ["pulmonary_toxicity"],
}

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
    return len(gt_set & set(pred_list)) / len(gt_set)


def main():
    t0 = time.time()
    db_url = os.environ.get('BETA_DATABASE_URL', os.environ.get('DATABASE_URL'))
    conn = psycopg2.connect(db_url)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    drug_data = {}
    all_relevant_tokens = set()

    for drug_info in PILOT_DRUGS:
        drug = drug_info["drug"]
        gene = drug_info["gene"]
        gt = drug_info["ground_truth"]

        cur.execute("""
            SELECT uniprot_id FROM protein_catalog
            WHERE gene_name = %s
            ORDER BY CASE WHEN uniprot_id ~ '^[OPQ][0-9][A-Z0-9]{3}[0-9]$' THEN 0 ELSE 1 END,
                     token_count DESC LIMIT 1
        """, (gene,))
        row = cur.fetchone()
        if not row:
            print(f"  {drug} ({gene}): NOT FOUND")
            continue
        target_uid = row['uniprot_id']

        cur.execute("SELECT DISTINCT token_hex FROM protein_tokens_v2 WHERE uniprot_id = %s", (target_uid,))
        target_tokens = set(r['token_hex'] for r in cur.fetchall())
        if not target_tokens:
            print(f"  {drug} ({gene}): no tokens")
            continue

        in_cascade = {target_uid}
        frontier_tokens_map = {target_uid: target_tokens}
        hop_proteins = {}

        for hop_num in range(1, MAX_HOPS + 1):
            ftoks = set()
            for ts in frontier_tokens_map.values():
                ftoks.update(ts)
            if not ftoks:
                break

            thresh = MIN_SHARED_TOKENS
            if hop_num == 1 and len(target_tokens) < MIN_SHARED_TOKENS:
                thresh = max(1, len(target_tokens))

            cur.execute("""
                SELECT pt.uniprot_id, pc.gene_name,
                       ARRAY_AGG(DISTINCT pt.token_hex) as matched_tokens
                FROM protein_tokens_v2 pt
                JOIN protein_catalog pc ON pt.uniprot_id = pc.uniprot_id
                WHERE pt.token_hex = ANY(%s)
                  AND pt.uniprot_id != ALL(%s)
                GROUP BY pt.uniprot_id, pc.gene_name
                HAVING COUNT(DISTINCT pt.token_hex) >= %s
            """, (list(ftoks), list(in_cascade), thresh))

            scored = []
            for r in cur.fetchall():
                mt = set(r['matched_tokens'])
                best = 0
                for fts in frontier_tokens_map.values():
                    s = len(mt & fts)
                    if s > best:
                        best = s
                if best >= thresh:
                    scored.append((r['uniprot_id'], best, r['gene_name']))

            scored.sort(key=lambda x: -x[1])
            seen_genes = set()
            deduped = []
            for uid, sc, gn in scored:
                if gn and gn in seen_genes:
                    continue
                if gn:
                    seen_genes.add(gn)
                deduped.append((uid, sc, gn))
                if len(deduped) >= MAX_PER_HOP:
                    break

            if not deduped:
                break

            hp = {}
            for uid, sc, gn in deduped:
                hp[uid] = gn
                in_cascade.add(uid)
            hop_proteins[hop_num] = hp

            new_uids = list(hp.keys())
            frontier_tokens_map = {}
            for i in range(0, len(new_uids), 500):
                batch = new_uids[i:i+500]
                cur.execute("""
                    SELECT uniprot_id, ARRAY_AGG(DISTINCT token_hex) as tokens
                    FROM protein_tokens_v2 WHERE uniprot_id = ANY(%s)
                    GROUP BY uniprot_id
                """, (batch,))
                for r in cur.fetchall():
                    frontier_tokens_map[r['uniprot_id']] = set(r['tokens'])
                    all_relevant_tokens.update(r['tokens'])

        all_relevant_tokens.update(target_tokens)

        all_genes = set()
        for hp_num, prots in hop_proteins.items():
            all_genes.update(v for v in prots.values() if v)
        all_genes.discard(gene)

        dept_map = {}
        gene_list = list(all_genes)
        for i in range(0, len(gene_list), 500):
            batch = gene_list[i:i+500]
            cur.execute("SELECT gene_name, all_departments FROM gene_department_map WHERE gene_name = ANY(%s)", (batch,))
            for r in cur.fetchall():
                dept_map[r['gene_name']] = r['all_departments'] or ['Unknown']

        pheno_map = {}
        for i in range(0, len(gene_list), 500):
            batch = gene_list[i:i+500]
            cur.execute("SELECT gene_name, phenotype_category, source FROM gene_phenotype_map WHERE gene_name = ANY(%s) AND gene_name != %s", (batch, gene))
            for r in cur.fetchall():
                if r['gene_name'] not in pheno_map:
                    pheno_map[r['gene_name']] = []
                pheno_map[r['gene_name']].append({'category': r['phenotype_category']})

        cur.execute("SELECT department, phenotype_category, prior_weight FROM dept_phenotype_priors")
        dept_priors = defaultdict(list)
        for r in cur.fetchall():
            dept_priors[r['department']].append({'category': r['phenotype_category'], 'prior_weight': float(r['prior_weight'])})

        phenotype_scores = defaultdict(float)
        for hop_level, prots in hop_proteins.items():
            hw = HOP_WEIGHTS.get(hop_level, 0.5 ** (hop_level - 1))
            dept_genes = defaultdict(set)
            for uid, gn in prots.items():
                if gn and gn != gene:
                    for dept in dept_map.get(gn, ['Unknown']):
                        dept_genes[dept].add(gn)

            for dept, genes_in_dept in dept_genes.items():
                for gn in genes_in_dept:
                    for ph in pheno_map.get(gn, []):
                        phenotype_scores[ph['category']] += hw
                if dept in dept_priors:
                    for prior in dept_priors[dept]:
                        phenotype_scores[prior['category']] += prior['prior_weight'] * hw * len(genes_in_dept) * 0.1

        ranked = sorted([(cat, sc) for cat, sc in phenotype_scores.items() if sc >= 0.1], key=lambda x: -x[1])
        predicted = [cat for cat, sc in ranked]
        r10 = compute_recall_at_k(predicted, gt, k=10)

        cascade_size = sum(len(hp) for hp in hop_proteins.values())
        hop_sizes = {h: len(p) for h, p in sorted(hop_proteins.items())}
        print(f"  {drug} ({gene}, {target_uid}): cascade={cascade_size}, R@10={r10:.3f}, hops={hop_sizes}")
        print(f"    Top 10: {predicted[:10]}")

        drug_data[drug] = {
            "gene": gene,
            "expected": drug_info["expected"],
            "target_uniprot": target_uid,
            "target_tokens": list(target_tokens),
            "real_r_at_10": r10,
            "real_top10": predicted[:10],
            "ground_truth": gt,
            "cascade_size": cascade_size,
            "hop_sizes": hop_sizes,
        }

    print(f"\n  Loading bipartite graph ({len(all_relevant_tokens)} tokens)...")
    token_to_proteins = defaultdict(list)
    protein_to_tokens = defaultdict(list)
    token_list = list(all_relevant_tokens)
    for i in range(0, len(token_list), 1000):
        batch = token_list[i:i+1000]
        cur.execute("SELECT token_hex, uniprot_id FROM protein_tokens_v2 WHERE token_hex = ANY(%s)", (batch,))
        for r in cur.fetchall():
            token_to_proteins[r['token_hex']].append(r['uniprot_id'])
            protein_to_tokens[r['uniprot_id']].append(r['token_hex'])

    n_edges = sum(len(ps) for ps in token_to_proteins.values())
    print(f"  Graph: {len(token_to_proteins)} tokens, {len(protein_to_tokens)} proteins, {n_edges} edges")

    print("  Pre-caching gene names...")
    gene_cache = {}
    uid_list = list(protein_to_tokens.keys())
    for i in range(0, len(uid_list), 2000):
        batch = uid_list[i:i+2000]
        cur.execute("SELECT uniprot_id, gene_name FROM protein_catalog WHERE uniprot_id = ANY(%s)", (batch,))
        for r in cur.fetchall():
            gene_cache[r['uniprot_id']] = r['gene_name']

    print("  Pre-caching departments...")
    dept_cache = {}
    all_g = list(set(gene_cache.values()))
    for i in range(0, len(all_g), 2000):
        batch = all_g[i:i+2000]
        cur.execute("SELECT gene_name, all_departments FROM gene_department_map WHERE gene_name = ANY(%s)", (batch,))
        for r in cur.fetchall():
            dept_cache[r['gene_name']] = r['all_departments'] or ['Unknown']

    print("  Pre-caching phenotypes...")
    pheno_cache = {}
    for i in range(0, len(all_g), 2000):
        batch = all_g[i:i+2000]
        cur.execute("SELECT gene_name, phenotype_category FROM gene_phenotype_map WHERE gene_name = ANY(%s)", (batch,))
        for r in cur.fetchall():
            if r['gene_name'] not in pheno_cache:
                pheno_cache[r['gene_name']] = []
            pheno_cache[r['gene_name']].append(r['phenotype_category'])

    print("  Loading dept priors...")
    cur.execute("SELECT department, phenotype_category, prior_weight FROM dept_phenotype_priors")
    dp = defaultdict(list)
    for r in cur.fetchall():
        dp[r['department']].append((r['phenotype_category'], float(r['prior_weight'])))

    conn.close()

    state = {
        "drug_data": drug_data,
        "token_to_proteins": {k: list(v) for k, v in token_to_proteins.items()},
        "protein_to_tokens": {k: list(v) for k, v in protein_to_tokens.items()},
        "gene_cache": gene_cache,
        "dept_cache": dept_cache,
        "pheno_cache": pheno_cache,
        "dept_priors": dict(dp),
        "n_edges": n_edges,
    }

    outpath = "/tmp/module8_phase1_state.pkl"
    with open(outpath, "wb") as f:
        pickle.dump(state, f)
    print(f"\n  Saved state to {outpath} ({os.path.getsize(outpath) / 1e6:.1f} MB)")
    print(f"  Phase 1 elapsed: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
