#!/usr/bin/env python3
"""
GO-Free Spectral Test — Definitive response to Reviewer Concerns 2 & 6.
========================================================================

APPROACH:
  Instead of discretizing tokens into K clusters (which loses structure),
  we use the CONTINUOUS spectral coordinates of the token co-occurrence
  graph as the measurement space.

  Each eigenvector of the normalized Laplacian captures a natural mode of
  variation in token co-occurrence patterns. These are the intrinsic
  coordinates of the dispatch graph — no GO labels, no arbitrary clustering.

  For each gene g:
    spectral_profile[k] = mean of eigenvector_k across g's tokens

  Then test collinearity: within-primitive vs across-primitive cosine
  similarity of spectral profiles.

  If the algebra is real, genes sharing a primitive should have aligned
  spectral profiles — because the algebra is in the graph topology,
  not in the GO labels we project onto it.

MULTIPLE DIMENSIONALITIES:
  We test K = 5, 10, 15, 22, 30 eigenvectors. If collinearity survives
  across all K, the algebra is coordinate-free.

CONTROLS:
  1. Shuffled primitives: randomly assign genes to pseudo-primitives
  2. Random embedding: replace spectral eigenvectors with random vectors
  3. GO-spectral concordance: CCA/correlation between GO profiles and
     spectral profiles to quantify how much GO structure the graph captures

Usage:
    python3 -u validation/knockout/go_free_spectral_test.py
"""

import csv
import json
import os
import pickle
import sys
import time
import numpy as np
from collections import defaultdict, Counter
from scipy import sparse
from scipy.sparse.linalg import eigsh

STATE_PATH = "/tmp/module8_full_state.pkl"
VOCAB_PATH = "server/data/human/vocabulary.csv"
PRIMITIVES_PATH = "beta_transfer/genome_primitives.csv"
OUTPUT_PATH = "validation/knockout/go_free_dimension_results.json"

VALID_DEPARTMENTS = sorted([
    "Apoptosis", "Cell adhesion", "Cell cycle", "Chromatin", "Cytoskeleton",
    "DNA repair", "GTPase", "Immune", "Ion channel", "Kinase", "Methylation",
    "Nuc acid bind", "Phosphatase", "Protein folding", "Proteolysis",
    "RNA processing", "Signaling", "Structural", "Transcription",
    "Translation", "Transport", "Ubiquitin",
])
D2I = {d: i for i, d in enumerate(VALID_DEPARTMENTS)}

K_VALUES = [5, 10, 15, 22, 30]
MIN_TOKEN_CARRIERS = 5
N_SHUFFLE_CONTROLS = 20


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def load_data():
    print("[LOAD] Loading dispatch graph state and metadata...")
    t0 = time.time()

    with open(STATE_PATH, "rb") as f:
        state = pickle.load(f)
    ptt = state["ptt"]
    ttp = state["ttp"]
    gene_cache = state["gene_cache"]

    vocab_dept = {}
    with open(VOCAB_PATH) as f:
        for row in csv.DictReader(f):
            vocab_dept[row["word_hex"].replace("0x", "").upper()] = row["primary_function"]

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
    for p in raw_prims:
        ds = [d for d in p["function_sequence"].split("|") if d in D2I]
        if not ds:
            continue
        search = "|".join(ds)
        carriers = [uid for uid, seq in protein_dept_seqs.items() if search in seq]
        if len(carriers) >= 20:
            primitives.append({"search": search, "n_carriers": len(carriers)})

    gene_to_uids = defaultdict(list)
    for uid, g in gene_cache.items():
        if g:
            gene_to_uids[g].append(uid)

    prim_to_genes = defaultdict(list)
    for p in primitives:
        carriers = [uid for uid, seq in protein_dept_seqs.items() if p["search"] in seq]
        for uid in carriers:
            g = gene_cache.get(uid)
            if g:
                prim_to_genes[p["search"]].append(g)
    for k in prim_to_genes:
        prim_to_genes[k] = list(set(prim_to_genes[k]))

    testable = {p: genes for p, genes in prim_to_genes.items() if len(genes) >= 5}

    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Proteins: {len(ptt):,}")
    print(f"  Tokens: {len(ttp):,}")
    print(f"  Testable primitives: {len(testable)}")

    return {
        "ptt": ptt, "ttp": ttp, "gene_cache": gene_cache,
        "gene_to_uids": gene_to_uids, "vocab_dept": vocab_dept,
        "primitives": primitives, "testable": testable,
        "prim_to_genes": prim_to_genes,
        "protein_dept_seqs": protein_dept_seqs,
    }


def compute_token_spectral_embedding(ttp, max_K, rng_seed=42):
    """
    Compute spectral embedding of tokens using the token-token
    co-occurrence graph (tokens sharing proteins).

    Returns: dict mapping token -> array of spectral coordinates (max_K dims)
    """
    print(f"\n[SPECTRAL] Computing {max_K}-dimensional spectral embedding of tokens...")
    t0 = time.time()

    tok_list = [tok for tok, carriers in ttp.items() if len(carriers) >= MIN_TOKEN_CARRIERS]
    n_toks = len(tok_list)
    tok_to_idx = {tok: i for i, tok in enumerate(tok_list)}
    print(f"  Tokens with >= {MIN_TOKEN_CARRIERS} carriers: {n_toks:,}")

    all_proteins = set()
    for tok in tok_list:
        all_proteins.update(ttp[tok])
    prot_list = sorted(all_proteins)
    prot_to_idx = {p: i for i, p in enumerate(prot_list)}
    n_prots = len(prot_list)

    rows, cols, vals = [], [], []
    for ti, tok in enumerate(tok_list):
        for uid in ttp[tok]:
            pi = prot_to_idx.get(uid)
            if pi is not None:
                rows.append(ti)
                cols.append(pi)
                vals.append(1.0)

    B = sparse.csr_matrix((vals, (rows, cols)), shape=(n_toks, n_prots))
    print(f"  Incidence matrix B: {B.shape}, nnz={B.nnz:,}")

    W = B @ B.T
    W.setdiag(0)
    W.eliminate_zeros()
    print(f"  Token co-occurrence matrix W: nnz={W.nnz:,}")

    degrees = np.array(W.sum(axis=1)).flatten()
    degrees = np.maximum(degrees, 1e-10)
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
    W_norm = D_inv_sqrt @ W @ D_inv_sqrt

    n_eig = min(max_K + 1, n_toks - 1)
    print(f"  Computing {n_eig} largest eigenvectors of normalized adjacency...")
    eigenvalues, eigenvectors = eigsh(W_norm, k=n_eig, which='LM', tol=1e-3, maxiter=500)

    sort_idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    print(f"  Top eigenvalues: {', '.join(f'{ev:.4f}' for ev in eigenvalues[:min(10, len(eigenvalues))])}")

    embedding = eigenvectors[:, 1:max_K+1]

    token_embedding = {}
    for i, tok in enumerate(tok_list):
        token_embedding[tok] = embedding[i].copy()

    elapsed = time.time() - t0
    print(f"  Spectral embedding computed in {elapsed:.1f}s")

    return token_embedding, tok_list, eigenvalues


def compute_spectral_profiles(ptt, gene_to_uids, token_embedding, K, testable):
    """
    For each gene, compute its spectral profile as the mean spectral
    coordinate across all its tokens.
    """
    relevant_genes = set()
    for genes in testable.values():
        relevant_genes.update(genes)

    profiles = {}
    for gene in relevant_genes:
        uids = gene_to_uids.get(gene, [])
        if not uids:
            continue

        gene_tokens = set()
        for uid in uids:
            gene_tokens.update(ptt.get(uid, []))

        vecs = []
        for tok in gene_tokens:
            emb = token_embedding.get(tok)
            if emb is not None:
                vecs.append(emb[:K])

        if len(vecs) >= 2:
            profiles[gene] = np.mean(vecs, axis=0)

    return profiles


def compute_go_profiles(ptt, gene_to_uids, vocab_dept, testable):
    """
    Compute the standard GO-based disruption profiles for comparison.
    """
    relevant_genes = set()
    for genes in testable.values():
        relevant_genes.update(genes)

    dept_tok_counts = defaultdict(int)
    for tok, d in vocab_dept.items():
        if d in D2I:
            dept_tok_counts[d] += 1

    profiles = {}
    for gene in relevant_genes:
        uids = gene_to_uids.get(gene, [])
        if not uids:
            continue

        gene_tokens = set()
        for uid in uids:
            gene_tokens.update(ptt.get(uid, []))

        profile = np.zeros(22)
        for tok in gene_tokens:
            d = vocab_dept.get(tok.upper())
            if d and d in D2I:
                profile[D2I[d]] += 1

        for di in range(22):
            dept = VALID_DEPARTMENTS[di]
            if dept_tok_counts[dept] > 0:
                profile[di] /= dept_tok_counts[dept]

        profiles[gene] = profile

    return profiles


def compute_collinearity_d(profiles_dict, testable, rng_seed=42):
    rng = np.random.RandomState(rng_seed)
    all_genes = list(profiles_dict.keys())

    within_cos = []
    across_cos = []

    for prim, genes in testable.items():
        vecs = [profiles_dict[g] for g in genes
                if g in profiles_dict and np.linalg.norm(profiles_dict[g]) > 1e-10]
        if len(vecs) < 3:
            continue

        for i in range(len(vecs)):
            for j in range(i + 1, min(i + 10, len(vecs))):
                within_cos.append(cosine_sim(vecs[i], vecs[j]))

        rand = rng.choice(all_genes, size=min(len(genes), 50), replace=False)
        rvecs = [profiles_dict[g] for g in rand
                 if g in profiles_dict and np.linalg.norm(profiles_dict[g]) > 1e-10]
        for i in range(len(vecs)):
            for j in range(min(10, len(rvecs))):
                across_cos.append(cosine_sim(vecs[i], rvecs[j]))

    if not within_cos or not across_cos:
        return {"d": 0, "within_mean": 0, "across_mean": 0}

    wc = np.array(within_cos)
    ac = np.array(across_cos)
    pooled = np.sqrt((wc.var() + ac.var()) / 2)
    d = (wc.mean() - ac.mean()) / pooled if pooled > 0 else 0

    return {
        "within_mean": round(float(wc.mean()), 4),
        "across_mean": round(float(ac.mean()), 4),
        "delta": round(float(wc.mean() - ac.mean()), 4),
        "d": round(float(d), 4),
        "n_within": len(within_cos),
        "n_across": len(across_cos),
    }


def shuffled_primitive_control(profiles_dict, testable, n_shuffles=20):
    """
    Randomly reassign genes to pseudo-primitives of the same sizes.
    This tests whether the collinearity is specific to real primitives
    or would arise from any grouping of genes.
    """
    all_genes = list(profiles_dict.keys())
    shuffled_ds = []

    for si in range(n_shuffles):
        rng = np.random.RandomState(si + 3000)
        fake_testable = {}
        for prim, genes in testable.items():
            n = len(genes)
            fake_genes = list(rng.choice(all_genes, size=min(n, len(all_genes)), replace=False))
            fake_testable[prim] = fake_genes

        result = compute_collinearity_d(profiles_dict, fake_testable, rng_seed=si + 4000)
        shuffled_ds.append(result["d"])

    arr = np.array(shuffled_ds)
    return {
        "mean_d": round(float(arr.mean()), 4),
        "std_d": round(float(arr.std()), 4),
        "range": [round(float(arr.min()), 4), round(float(arr.max()), 4)],
    }


def random_embedding_control(ptt, gene_to_uids, testable, K, n_controls=10):
    """
    Replace spectral embedding with random vectors.
    Collinearity should vanish.
    """
    all_genes = set()
    for genes in testable.values():
        all_genes.update(genes)

    random_ds = []
    for ri in range(n_controls):
        rng = np.random.RandomState(ri + 5000)

        profiles = {}
        for gene in all_genes:
            uids = gene_to_uids.get(gene, [])
            if not uids:
                continue
            gene_tokens = set()
            for uid in uids:
                gene_tokens.update(ptt.get(uid, []))
            if len(gene_tokens) >= 2:
                vecs = [rng.randn(K) for _ in range(len(gene_tokens))]
                profiles[gene] = np.mean(vecs, axis=0)

        result = compute_collinearity_d(profiles, testable, rng_seed=ri + 6000)
        random_ds.append(result["d"])

    arr = np.array(random_ds)
    return {
        "mean_d": round(float(arr.mean()), 4),
        "std_d": round(float(arr.std()), 4),
        "range": [round(float(arr.min()), 4), round(float(arr.max()), 4)],
    }


def spectral_go_correlation(ptt, gene_to_uids, token_embedding, vocab_dept,
                              testable, K):
    """
    Compute correlation between spectral profiles and GO profiles.
    Shows how much of GO's structure the graph topology captures.
    """
    spectral_profiles = compute_spectral_profiles(
        ptt, gene_to_uids, token_embedding, K, testable
    )
    go_profiles = compute_go_profiles(ptt, gene_to_uids, vocab_dept, testable)

    common_genes = sorted(set(spectral_profiles.keys()) & set(go_profiles.keys()))
    if len(common_genes) < 50:
        return {"error": "too_few_common_genes", "n": len(common_genes)}

    S = np.array([spectral_profiles[g] for g in common_genes])
    G = np.array([go_profiles[g] for g in common_genes])

    cosines = []
    for i in range(len(common_genes)):
        c = cosine_sim(S[i], G[i])
        cosines.append(c)

    col_corrs = []
    for k in range(min(K, S.shape[1])):
        for d in range(G.shape[1]):
            r = np.corrcoef(S[:, k], G[:, d])[0, 1]
            if not np.isnan(r):
                col_corrs.append(abs(r))

    return {
        "n_genes": len(common_genes),
        "mean_profile_cosine": round(float(np.mean(cosines)), 4),
        "max_column_correlation": round(float(max(col_corrs) if col_corrs else 0), 4),
        "mean_column_correlation": round(float(np.mean(col_corrs) if col_corrs else 0), 4),
    }


def pca_analysis(profiles_dict, primitives, gene_cache, ptt, vocab_dept, K):
    """
    PCA dimensionality analysis on primitive mean spectral profiles.
    """
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

    prim_vecs = []
    for p in primitives:
        carriers = [uid for uid, seq in protein_dept_seqs.items() if p["search"] in seq]
        carrier_genes = set()
        for uid in carriers:
            g = gene_cache.get(uid)
            if g and g in profiles_dict:
                carrier_genes.add(g)

        if len(carrier_genes) < 3:
            continue

        mean_profile = np.mean([profiles_dict[g] for g in carrier_genes], axis=0)
        if np.linalg.norm(mean_profile) > 1e-10:
            prim_vecs.append(mean_profile)

    if len(prim_vecs) < 5:
        return {"error": "too_few_primitives", "n": len(prim_vecs)}

    M = np.array(prim_vecs)
    M_centered = M - M.mean(axis=0)

    from scipy.linalg import svd
    U, S, Vt = svd(M_centered, full_matrices=False)
    cumvar = np.cumsum(S ** 2) / (S ** 2).sum()

    n_dims_90 = int(np.searchsorted(cumvar, 0.90) + 1)
    n_dims_95 = int(np.searchsorted(cumvar, 0.95) + 1)
    eff_5d = float(cumvar[min(4, len(cumvar)-1)])

    return {
        "n_primitives": len(prim_vecs),
        "cumulative_variance": [round(float(c), 4) for c in cumvar[:min(15, len(cumvar))]],
        "dims_for_90pct": n_dims_90,
        "dims_for_95pct": n_dims_95,
        "effective_dim_5pc": round(eff_5d, 4),
    }


def main():
    print("=" * 72)
    print("  GO-FREE SPECTRAL TEST")
    print("  Continuous spectral coordinates — no clustering, no GO input")
    print("  Response to Reviewer Concerns 2 & 6")
    print("=" * 72)

    data = load_data()
    ptt = data["ptt"]
    ttp = data["ttp"]
    gene_cache = data["gene_cache"]
    gene_to_uids = data["gene_to_uids"]
    testable = data["testable"]
    primitives = data["primitives"]
    vocab_dept = data["vocab_dept"]

    max_K = max(K_VALUES)
    token_embedding, tok_list, eigenvalues = compute_token_spectral_embedding(ttp, max_K)

    print(f"\n{'='*72}")
    print(f"  BASELINE: GO-based collinearity (22D department space)")
    print(f"{'='*72}")
    go_profiles = compute_go_profiles(ptt, gene_to_uids, vocab_dept, testable)
    go_collinearity = compute_collinearity_d(go_profiles, testable)
    print(f"  GO-based collinearity: d={go_collinearity['d']:+.4f}")
    print(f"  within={go_collinearity['within_mean']:.4f}, across={go_collinearity['across_mean']:.4f}")

    all_results = {}

    for K in K_VALUES:
        print(f"\n{'='*72}")
        print(f"  SPECTRAL TEST: K={K} eigenvectors (GO-free)")
        print(f"{'='*72}")

        profiles = compute_spectral_profiles(ptt, gene_to_uids, token_embedding, K, testable)
        print(f"  Computed {len(profiles)} spectral profiles")

        collinearity = compute_collinearity_d(profiles, testable)
        print(f"\n  COLLINEARITY (K={K}): d={collinearity['d']:+.4f}")
        print(f"    within={collinearity['within_mean']:.4f}, across={collinearity['across_mean']:.4f}")
        print(f"    Δ={collinearity['delta']:.4f}")

        print(f"\n  Running shuffled-primitive controls (n={N_SHUFFLE_CONTROLS})...")
        shuffle = shuffled_primitive_control(profiles, testable, n_shuffles=N_SHUFFLE_CONTROLS)
        print(f"    Shuffled primitives d: {shuffle['mean_d']:+.4f} ± {shuffle['std_d']:.4f}")

        z_vs_shuffle = (collinearity["d"] - shuffle["mean_d"]) / max(shuffle["std_d"], 1e-6)
        print(f"    Z vs shuffle: {z_vs_shuffle:+.2f}")

        print(f"\n  Running random-embedding controls (n=10)...")
        random_ctrl = random_embedding_control(ptt, gene_to_uids, testable, K, n_controls=10)
        print(f"    Random embedding d: {random_ctrl['mean_d']:+.4f} ± {random_ctrl['std_d']:.4f}")

        pca = pca_analysis(profiles, primitives, gene_cache, ptt, vocab_dept, K)
        if "dims_for_90pct" in pca:
            print(f"\n  PCA: 90% variance in {pca['dims_for_90pct']} dims, "
                  f"5 PCs capture {pca.get('effective_dim_5pc', 0):.1%}")

        corr = spectral_go_correlation(ptt, gene_to_uids, token_embedding, vocab_dept, testable, K)
        if "mean_column_correlation" in corr:
            print(f"  GO-spectral correlation: max_col={corr['max_column_correlation']:.4f}, "
                  f"mean_col={corr['mean_column_correlation']:.4f}")

        d_val = collinearity["d"]
        if d_val > 0.8:
            verdict = "STRONG_SURVIVAL"
        elif d_val > 0.5:
            verdict = "CLEAR_SURVIVAL"
        elif d_val > 0.3:
            verdict = "PARTIAL_SURVIVAL"
        else:
            verdict = "FAILURE"

        if z_vs_shuffle > 3.0 and d_val > 0.2:
            verdict_refined = verdict + "_SIGNIFICANT"
        elif z_vs_shuffle > 2.0 and d_val > 0.2:
            verdict_refined = verdict + "_MARGINAL"
        else:
            verdict_refined = verdict

        print(f"\n  === K={K} VERDICT: {verdict_refined} ===")

        all_results[f"K={K}"] = {
            "K": K,
            "collinearity": collinearity,
            "shuffled_primitive_control": shuffle,
            "random_embedding_control": random_ctrl,
            "z_vs_shuffle": round(float(z_vs_shuffle), 2),
            "pca": pca,
            "go_spectral_correlation": corr,
            "verdict": verdict_refined,
        }

        sys.stdout.flush()

    print(f"\n\n{'='*72}")
    print(f"  FINAL SUMMARY — GO-FREE SPECTRAL TEST")
    print(f"{'='*72}")

    print(f"\n  GO baseline: d={go_collinearity['d']:+.4f}")
    print()
    print(f"  {'K':>4s}  {'d_spec':>7s}  {'d_shuf':>7s}  {'d_rand':>7s}  "
          f"{'Z':>6s}  {'PCA_5D':>7s}  {'90%dim':>6s}  {'Verdict'}")
    print(f"  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*25}")

    verdicts = []
    for K in K_VALUES:
        r = all_results[f"K={K}"]
        c = r["collinearity"]
        s = r["shuffled_primitive_control"]
        rc = r["random_embedding_control"]
        p = r["pca"]

        pca_5d = f"{p.get('effective_dim_5pc', 0):.1%}" if isinstance(p, dict) and "effective_dim_5pc" in p else "N/A"
        pca_90 = str(p.get("dims_for_90pct", "N/A")) if isinstance(p, dict) else "N/A"

        print(f"  {K:4d}  {c['d']:+7.4f}  {s['mean_d']:+7.4f}  {rc['mean_d']:+7.4f}  "
              f"{r['z_vs_shuffle']:+6.1f}  {pca_5d:>7s}  {pca_90:>6s}  {r['verdict']}")
        verdicts.append(r["verdict"])

    survival_count = sum(1 for v in verdicts if "SURVIVAL" in v)
    significant_count = sum(1 for v in verdicts if "SIGNIFICANT" in v)

    print(f"\n  Algebra survives in {survival_count}/{len(verdicts)} K-values tested.")
    print(f"  Statistically significant in {significant_count}/{len(verdicts)}.")

    if significant_count == len(verdicts):
        overall = "DEFINITIVE_SURVIVAL"
    elif survival_count == len(verdicts):
        overall = "CONSISTENT_SURVIVAL"
    elif survival_count > len(verdicts) // 2:
        overall = "MAJORITY_SURVIVAL"
    elif survival_count > 0:
        overall = "PARTIAL_SURVIVAL"
    else:
        overall = "FAILURE"

    print(f"\n  OVERALL VERDICT: {overall}")

    go_d = go_collinearity["d"]
    best_spectral_d = max(all_results[f"K={K}"]["collinearity"]["d"] for K in K_VALUES)
    retention = best_spectral_d / go_d if go_d > 0 else 0
    print(f"  GO-based d: {go_d:+.4f}")
    print(f"  Best spectral d: {best_spectral_d:+.4f}")
    print(f"  Retention: {retention:.1%} of GO-based signal survives in GO-free space")

    output = {
        "experiment": "GO-free spectral test",
        "purpose": "Test whether collinearity survives in continuous spectral coordinates "
                   "derived purely from dispatch graph topology (no GO input)",
        "method": "Eigenvectors of normalized token co-occurrence graph as coordinate axes",
        "K_values_tested": K_VALUES,
        "go_baseline_d": go_collinearity["d"],
        "eigenvalues": [round(float(ev), 4) for ev in eigenvalues[:max_K+1]],
        "results_by_K": all_results,
        "survival_count": survival_count,
        "significant_count": significant_count,
        "overall_verdict": overall,
        "retention_vs_go": round(retention, 4),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Saved: {OUTPUT_PATH}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
