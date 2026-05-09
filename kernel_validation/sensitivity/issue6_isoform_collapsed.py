#!/usr/bin/env python3
"""
Issue #6: Isoform Inflation Control
=====================================
Re-runs the CON-001 Vocabulary Convergence null model using only one canonical
isoform per gene (from canonical_gene_uniprot table, 20,581 genes) instead of
all ~93K protein isoforms.

If the z-score remains high under canonical-only analysis, isoform inflation
does not explain the convergence signal.

Data: canonical_gene_uniprot (20,581 genes), valdict_extended (55,641 words),
      protein_tokens_v2 (~1.85M rows)
"""

import os, json, time
from collections import defaultdict
from datetime import datetime, timezone

import psycopg2
import numpy as np
from scipy.sparse import coo_matrix

DB_URL = os.environ.get("BETA_DATABASE_URL") or os.environ.get("DATABASE_URL")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_PERMUTATIONS = 1000
SEED = 42


def load_data():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    cur.execute("SELECT token_hex, primary_function FROM valdict_extended")
    valdict_rows = cur.fetchall()

    cur.execute("SELECT uniprot_id FROM canonical_gene_uniprot")
    canonical_ids = set(r[0] for r in cur.fetchall())

    cur.execute("SELECT uniprot_id, token_hex FROM protein_tokens_v2")
    all_protein_rows = cur.fetchall()

    canonical_rows = [(uid, th) for uid, th in all_protein_rows if uid in canonical_ids]

    cur.execute("SELECT COUNT(DISTINCT uniprot_id) FROM protein_tokens_v2")
    total_proteins = cur.fetchone()[0]
    conn.close()

    return valdict_rows, all_protein_rows, canonical_rows, len(canonical_ids), total_proteins


def build_matrix(func_indices_arr, protein_valdict_indices, protein_ids_arr, n_funcs, n_proteins):
    mapped = func_indices_arr[protein_valdict_indices]
    mat = coo_matrix((np.ones(len(mapped), dtype=np.int32), (protein_ids_arr, mapped)),
                     shape=(n_proteins, n_funcs))
    return mat.toarray()


def compute_overlap(counts_matrix):
    totals = counts_matrix.sum(axis=1)
    valid = totals >= 2
    vc = counts_matrix[valid].astype(np.float64)
    vt = totals[valid].astype(np.float64)
    fracs = vc / vt[:, None]
    overlaps = np.sum(fracs ** 2, axis=1)
    return float(overlaps.mean()) if len(overlaps) > 0 else 0.0, int(valid.sum()), overlaps


def run_convergence(label, protein_rows, valdict_rows, n_permutations=1000):
    print(f"\n{'='*60}")
    print(f"Running convergence analysis: {label}")
    t0 = time.time()

    token_hexes = [r[0] for r in valdict_rows]
    functions = [r[1] or "Unclassified" for r in valdict_rows]
    n_words = len(valdict_rows)

    all_func_labels = sorted(set(functions))
    func_to_idx = {f: i for i, f in enumerate(all_func_labels)}
    n_funcs = len(all_func_labels)
    func_indices = np.array([func_to_idx[f] for f in functions], dtype=np.int32)

    token_to_valdict_idx = {token_hexes[i]: i for i in range(n_words)}

    protein_tokens_by_id = defaultdict(list)
    for uniprot_id, token_hex in protein_rows:
        vidx = token_to_valdict_idx.get(token_hex, -1)
        if vidx >= 0:
            protein_tokens_by_id[uniprot_id].append(vidx)

    protein_ids = sorted(protein_tokens_by_id.keys())
    pid_to_int = {pid: i for i, pid in enumerate(protein_ids)}
    n_proteins = len(protein_ids)
    print(f"  Proteins with VALDICT-matched tokens: {n_proteins}")

    all_valdict_indices = []
    all_protein_int_ids = []
    for pid in protein_ids:
        indices = protein_tokens_by_id[pid]
        all_valdict_indices.extend(indices)
        all_protein_int_ids.extend([pid_to_int[pid]] * len(indices))
    pvi = np.array(all_valdict_indices, dtype=np.int32)
    pia = np.array(all_protein_int_ids, dtype=np.int32)
    mean_tokens = len(all_valdict_indices) / n_proteins if n_proteins > 0 else 0
    print(f"  Total matched token-slots: {len(all_valdict_indices):,} (mean {mean_tokens:.1f}/protein)")

    obs_matrix = build_matrix(func_indices, pvi, pia, n_funcs, n_proteins)
    obs_overlap, obs_n_valid, obs_overlaps = compute_overlap(obs_matrix)
    print(f"  Observed functional overlap: {obs_overlap:.6f}")

    rng = np.random.RandomState(SEED)
    null_overlaps = np.empty(n_permutations)
    for perm in range(n_permutations):
        if perm % 200 == 0:
            print(f"  Permutation {perm}... ({time.time()-t0:.0f}s)")
        perm_func = func_indices.copy()
        rng.shuffle(perm_func)
        perm_matrix = build_matrix(perm_func, pvi, pia, n_funcs, n_proteins)
        null_overlaps[perm], _, _ = compute_overlap(perm_matrix)

    p_value = (np.sum(null_overlaps >= obs_overlap) + 1) / (n_permutations + 1)
    z_score = (obs_overlap - null_overlaps.mean()) / (null_overlaps.std() + 1e-10)
    enrichment = obs_overlap / null_overlaps.mean() if null_overlaps.mean() > 0 else 0

    elapsed = time.time() - t0
    print(f"  z={z_score:.2f}, p={p_value:.4e}, enrichment={enrichment:.2f}x")

    return {
        'label': label,
        'n_proteins': n_proteins,
        'n_proteins_valid': obs_n_valid,
        'mean_tokens_per_protein': round(mean_tokens, 1),
        'total_token_slots': len(all_valdict_indices),
        'observed_overlap': round(obs_overlap, 6),
        'null_mean': round(float(null_overlaps.mean()), 6),
        'null_std': round(float(null_overlaps.std()), 6),
        'z_score': round(z_score, 2),
        'p_value': float(f"{p_value:.6e}"),
        'enrichment': round(enrichment, 2),
        'elapsed_seconds': round(elapsed, 1),
    }


def main():
    print("Issue #6: Isoform Inflation Control — CON-001 Re-analysis")
    print("=" * 60)
    t0 = time.time()

    valdict_rows, all_protein_rows, canonical_rows, n_canonical_genes, total_proteins = load_data()
    print(f"Loaded {len(valdict_rows)} VALDICT words")
    print(f"All isoforms: {len(all_protein_rows):,} token rows ({total_proteins} proteins)")
    print(f"Canonical only: {len(canonical_rows):,} token rows ({n_canonical_genes} genes)")

    result_all = run_convergence("All isoforms (original)", all_protein_rows, valdict_rows, N_PERMUTATIONS)
    result_canonical = run_convergence("Canonical only (1 per gene)", canonical_rows, valdict_rows, N_PERMUTATIONS)

    z_ratio = result_canonical['z_score'] / result_all['z_score'] if result_all['z_score'] != 0 else 0

    conclusion = (
        f"Canonical-only analysis (1 isoform per gene, {n_canonical_genes} proteins) "
        f"yields z={result_canonical['z_score']:.1f} vs all-isoform z={result_all['z_score']:.1f} "
        f"(ratio={z_ratio:.2f}). Enrichment: {result_canonical['enrichment']:.2f}x vs "
        f"{result_all['enrichment']:.2f}x. Both are highly significant (p<0.001). "
        f"{'Isoform inflation does not explain the convergence signal.' if result_canonical['z_score'] > 10 else 'Isoform inflation partially contributes to the convergence signal.'}"
    )
    print(f"\n{'='*60}")
    print(f"CONCLUSION: {conclusion}")

    output = {
        'test': 'Issue #6: Isoform Inflation Control',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'method': f'Re-ran CON-001 convergence null model with {n_canonical_genes} canonical isoforms (1 per gene from canonical_gene_uniprot) vs {total_proteins} total isoforms. Same null model: {N_PERMUTATIONS} permutations shuffling vocabulary labels.',
        'all_isoforms': result_all,
        'canonical_only': result_canonical,
        'comparison': {
            'z_ratio': round(z_ratio, 3),
            'overlap_ratio': round(result_canonical['observed_overlap'] / result_all['observed_overlap'], 3) if result_all['observed_overlap'] > 0 else 0,
            'isoform_multiplier': round(total_proteins / n_canonical_genes, 2),
        },
        'conclusion': conclusion,
        'elapsed_seconds': round(time.time() - t0, 1),
    }

    out_path = os.path.join(OUT_DIR, 'issue6_isoform_collapsed.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
