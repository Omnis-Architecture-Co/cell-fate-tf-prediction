#!/usr/bin/env python3
"""
GO-Free Dimension Test — the definitive response to Reviewer Concerns 2 & 6.
=============================================================================

PREMISE:
  The reviewer argues that the 22 GO-derived departments may impose correlated
  structure, making the apparent 5D algebra an artifact of GO's annotation
  hierarchy rather than a genuine property of the proteome's dispatch graph.

EXPERIMENT:
  Replace the 22 GO-derived departments with K unsupervised protein clusters
  derived *purely* from dispatch graph topology (no GO, no functional labels).
  Then re-run:
    1. Disruption profiles in K-dimensional unsupervised space
    2. Collinearity test (within-primitive vs across-primitive cosine similarity)
    3. PCA dimensionality analysis (does ~5D still capture structure?)

CLUSTERING METHOD (spectral clustering on projected protein-protein graph):
  - Build the protein × token binary incidence matrix B (93k × 125k)
  - Compute the protein-protein co-occurrence matrix W = B @ B^T
    (W_ij = number of shared tokens between proteins i and j)
  - Apply spectral clustering with K eigenvectors of the normalized Laplacian
  - This is the most defensible choice because:
    (a) It uses *only* graph topology — zero functional annotation
    (b) Spectral methods find communities that minimize normalized cut
    (c) The Laplacian eigenvectors form an orthogonal basis, avoiding
        collinearity artifacts that k-means on raw features might introduce
    (d) It's mathematically well-characterized (Ng-Jordan-Weiss 2001)

MULTIPLE K VALUES:
  We sweep K ∈ {10, 15, 22, 30, 50} to show the algebra survives
  re-dimensioning across a range of granularities. If collinearity d > 0.5
  at K=10, K=22, AND K=50, the GO objection is definitively dead.

CONTROLS:
  - Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) between
    unsupervised clusters and GO departments, to quantify how much the
    clusters recapitulate GO. If ARI ≈ 0 but collinearity survives, the
    algebra is independent of GO structure.
  - Shuffled-cluster control: randomly reassign proteins to K groups,
    preserving group sizes. Collinearity should vanish (d ≈ 0).

SUCCESS CRITERIA:
  - d > 0.8:  Strong survival — algebra is unambiguously real
  - d > 0.5:  Clear survival — GO objection is dead
  - d > 0.3:  Partial survival — algebra has GO-independent component
  - d < 0.2:  Failure — algebra was a GO artifact

MATHEMATICAL SUBTLETY:
  If unsupervised clusters happen to recapitulate GO (high ARI), that does NOT
  invalidate the test — it means GO captured real structure. The key is that
  the *method* used no GO input. We report ARI to let reviewers judge.

  The deeper subtlety: if the algebra's 5D PCA subspace is invariant under
  change of basis (GO → unsupervised), this is a coordinate-free property of
  the underlying manifold, not an artifact of labeling.

Usage:
    python3 -u validation/knockout/go_free_dimension_test.py
"""

import csv
import json
import os
import pickle
import sys
import time
import numpy as np
from collections import defaultdict, Counter
from scipy import sparse, stats
from scipy.sparse.linalg import eigsh
from scipy.linalg import svd

STATE_PATH = "/tmp/module8_full_state.pkl"
VOCAB_PATH = "server/data/human/vocabulary.csv"
GENE_DEPTS_PATH = "server/data/human/gene_departments.csv"
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
N_DEPTS = len(VALID_DEPARTMENTS)

K_VALUES = [10, 15, 22, 30, 50]
N_SHUFFLE_CONTROLS = 5
MIN_TOKEN_CARRIERS = 5


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def adjusted_rand_index(labels_true, labels_pred):
    from collections import Counter
    n = len(labels_true)
    if n == 0:
        return 0.0

    contingency = defaultdict(int)
    for t, p in zip(labels_true, labels_pred):
        contingency[(t, p)] += 1

    sum_comb_c = 0
    row_sums = Counter()
    col_sums = Counter()
    for (t, p), count in contingency.items():
        sum_comb_c += count * (count - 1) / 2
        row_sums[t] += count
        col_sums[p] += count

    sum_comb_a = sum(v * (v - 1) / 2 for v in row_sums.values())
    sum_comb_b = sum(v * (v - 1) / 2 for v in col_sums.values())

    total_comb = n * (n - 1) / 2
    expected = sum_comb_a * sum_comb_b / total_comb if total_comb > 0 else 0
    max_index = (sum_comb_a + sum_comb_b) / 2
    denom = max_index - expected

    if abs(denom) < 1e-12:
        return 0.0
    return (sum_comb_c - expected) / denom


def normalized_mutual_info(labels_true, labels_pred):
    n = len(labels_true)
    if n == 0:
        return 0.0

    ct = Counter(labels_true)
    cp = Counter(labels_pred)

    h_true = -sum((c / n) * np.log(c / n + 1e-15) for c in ct.values())
    h_pred = -sum((c / n) * np.log(c / n + 1e-15) for c in cp.values())

    contingency = defaultdict(int)
    for t, p in zip(labels_true, labels_pred):
        contingency[(t, p)] += 1

    mi = 0.0
    for (t, p), count in contingency.items():
        ptp = count / n
        pt = ct[t] / n
        pp = cp[p] / n
        if ptp > 0:
            mi += ptp * np.log(ptp / (pt * pp) + 1e-15)

    denom = np.sqrt(h_true * h_pred) if h_true > 0 and h_pred > 0 else 1.0
    return mi / denom


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

    gene_depts = {}
    with open(GENE_DEPTS_PATH) as f:
        for row in csv.DictReader(f):
            gene_depts[row["gene"]] = row["department"]

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
        "gene_depts": gene_depts, "gene_to_uids": gene_to_uids,
        "primitives": primitives, "testable": testable,
        "prim_to_genes": prim_to_genes, "vocab_dept": vocab_dept,
    }


def cluster_tokens(ttp, ptt, K, rng_seed=42):
    """
    Cluster TOKENS (not proteins) into K groups using spectral clustering
    on the token-token co-occurrence graph. This is the correct GO-free analog:
    departments are groupings of tokens, so the unsupervised replacement must
    also group tokens.

    Method:
      1. Filter tokens with >= MIN_TOKEN_CARRIERS carriers
      2. Build token × protein incidence matrix B
      3. Compute token co-occurrence: W = B @ B^T (tokens sharing proteins)
      4. Apply spectral clustering on the normalized Laplacian of W
      5. This finds natural communities of co-occurring tokens

    Tokens that appear together on the same proteins get grouped together.
    This is what GO departments actually capture — tokens with similar
    functional contexts. But here we derive it purely from graph topology.
    """
    print(f"\n  [TOKEN-CLUSTER K={K}] Spectral clustering on token co-occurrence...")
    t0 = time.time()

    tok_list = [tok for tok, carriers in ttp.items() if len(carriers) >= MIN_TOKEN_CARRIERS]
    n_toks = len(tok_list)
    print(f"    Tokens with >= {MIN_TOKEN_CARRIERS} carriers: {n_toks:,}")

    if n_toks > 15000:
        rng = np.random.RandomState(rng_seed)
        tok_degrees = np.array([len(ttp[tok]) for tok in tok_list])
        p = tok_degrees / tok_degrees.sum()
        selected_idx = rng.choice(n_toks, size=15000, replace=False, p=p)
        selected_idx.sort()
        core_toks = [tok_list[i] for i in selected_idx]
        remaining_toks = [tok_list[i] for i in range(n_toks) if i not in set(selected_idx)]
    else:
        core_toks = tok_list
        remaining_toks = []

    n_core = len(core_toks)
    tok_to_idx = {tok: i for i, tok in enumerate(core_toks)}

    all_proteins = set()
    for tok in core_toks:
        all_proteins.update(ttp[tok])
    prot_list = sorted(all_proteins)
    prot_to_idx = {p: i for i, p in enumerate(prot_list)}
    n_prots = len(prot_list)

    rows, cols, vals = [], [], []
    for ti, tok in enumerate(core_toks):
        for uid in ttp[tok]:
            pi = prot_to_idx.get(uid)
            if pi is not None:
                rows.append(ti)
                cols.append(pi)
                vals.append(1.0)

    B = sparse.csr_matrix((vals, (rows, cols)), shape=(n_core, n_prots))
    print(f"    Incidence matrix B: {B.shape}, nnz={B.nnz:,}")

    W = B @ B.T
    W.setdiag(0)
    W.eliminate_zeros()
    print(f"    Token co-occurrence matrix W: nnz={W.nnz:,}")

    degrees = np.array(W.sum(axis=1)).flatten()
    degrees = np.maximum(degrees, 1e-10)
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
    L_norm = sparse.eye(n_core) - D_inv_sqrt @ W @ D_inv_sqrt

    n_eig = min(K + 1, n_core - 1)
    print(f"    Computing {n_eig} smallest eigenvectors...")
    try:
        eigenvalues, eigenvectors = eigsh(L_norm, k=n_eig, which='SM', tol=1e-3, maxiter=300)
        Z = eigenvectors[:, 1:K+1] if eigenvectors.shape[1] > K else eigenvectors
    except Exception as e:
        print(f"    WARNING: eigsh failed ({e}), falling back to random projection + k-means")
        rng = np.random.RandomState(rng_seed)
        R = rng.randn(n_prots, K) / np.sqrt(K)
        Z = B.dot(R)

    row_norms = np.linalg.norm(Z, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-10)
    Z = Z / row_norms

    print(f"    Running k-means (K={K}) on spectral embedding...")
    labels = _kmeans(Z, K, rng_seed=rng_seed, max_iter=100)

    token_clusters = {}
    for i, tok in enumerate(core_toks):
        token_clusters[tok] = int(labels[i])

    if remaining_toks:
        core_cluster_carriers = defaultdict(lambda: defaultdict(int))
        for tok in core_toks:
            c = token_clusters[tok]
            for uid in ttp[tok]:
                core_cluster_carriers[uid][c] += 1

        for tok in remaining_toks:
            votes = np.zeros(K)
            for uid in ttp[tok]:
                for c, cnt in core_cluster_carriers.get(uid, {}).items():
                    votes[c] += cnt
            if votes.sum() > 0:
                token_clusters[tok] = int(np.argmax(votes))
            else:
                token_clusters[tok] = int(rng.randint(K))

    elapsed = time.time() - t0
    all_labels = list(token_clusters.values())
    sizes = Counter(all_labels)
    print(f"    Cluster sizes: min={min(sizes.values()):,} max={max(sizes.values()):,} "
          f"median={int(np.median(list(sizes.values()))):,}")
    print(f"    Token spectral clustering done in {elapsed:.1f}s")

    return token_clusters, tok_list


def _kmeans(X, K, rng_seed=42, max_iter=50):
    rng = np.random.RandomState(rng_seed)
    n = X.shape[0]

    init_indices = rng.choice(n, size=K, replace=False)
    centers = X[init_indices].copy()

    labels = np.zeros(n, dtype=int)

    for iteration in range(max_iter):
        dists = np.zeros((n, K))
        for k in range(K):
            diff = X - centers[k]
            dists[:, k] = np.sum(diff ** 2, axis=1)
        new_labels = np.argmin(dists, axis=1)

        if np.all(new_labels == labels):
            break
        labels = new_labels

        for k in range(K):
            mask = labels == k
            if mask.sum() > 0:
                centers[k] = X[mask].mean(axis=0)

    return labels


def compute_disruption_profiles_token_clusters(ptt, ttp, gene_to_uids, token_clusters, K,
                                                 testable, max_genes=2500):
    """
    Compute disruption profiles in K-dimensional unsupervised token-cluster space.

    For each gene g and each token-cluster c:
      profile[c] = (# of cluster-c tokens carried by gene g's proteins) /
                   (total # of tokens in cluster c)

    This is exactly analogous to the GO-based disruption profiles:
    departments group tokens by GO label; here we group tokens by
    unsupervised carrier-profile clustering.
    """
    print(f"\n  [PROFILES K={K}] Computing disruption profiles (token clusters)...")
    t0 = time.time()

    cluster_sizes = np.zeros(K)
    for tok, c in token_clusters.items():
        cluster_sizes[c] += 1

    relevant_genes = set()
    for genes in testable.values():
        relevant_genes.update(genes)
    relevant_genes = list(relevant_genes)[:max_genes]

    profiles = {}
    for gene in relevant_genes:
        uids = gene_to_uids.get(gene, [])
        if not uids:
            continue

        gene_tokens = set()
        for uid in uids:
            gene_tokens.update(ptt.get(uid, []))

        if not gene_tokens:
            continue

        profile = np.zeros(K)
        for tok in gene_tokens:
            c = token_clusters.get(tok)
            if c is not None:
                profile[c] += 1

        for c in range(K):
            if cluster_sizes[c] > 0:
                profile[c] /= cluster_sizes[c]

        profiles[gene] = profile

    elapsed = time.time() - t0
    print(f"    Computed {len(profiles)} profiles in {elapsed:.1f}s")
    return profiles


def compute_collinearity_d(profiles_dict, testable, rng_seed=42):
    """
    Compute within-primitive vs across-primitive cosine similarity
    and Cohen's d. This is the core collinearity metric.
    """
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
        return {"d": 0, "within_mean": 0, "across_mean": 0, "n_within": 0, "n_across": 0}

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


def pca_dimensionality_analysis(profiles_dict, primitives, testable, gene_cache,
                                 ptt, vocab_dept, K):
    """
    Compute the PCA spectrum of primitive profiles in the unsupervised K-D space.
    Check whether ~5 dimensions still capture the algebraic structure.
    """
    print(f"\n  [PCA K={K}] Dimensionality analysis...")
    t0 = time.time()

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
        print(f"    WARNING: Only {len(prim_vecs)} primitives with profiles — too few for PCA")
        return {"n_primitives": len(prim_vecs), "error": "too_few_primitives"}

    M = np.array(prim_vecs)
    M_centered = M - M.mean(axis=0)

    U, S, Vt = svd(M_centered, full_matrices=False)
    cumvar = np.cumsum(S ** 2) / (S ** 2).sum()

    n_dims_90 = int(np.searchsorted(cumvar, 0.90) + 1)
    n_dims_95 = int(np.searchsorted(cumvar, 0.95) + 1)

    print(f"    Primitives with profiles: {len(prim_vecs)}")
    print(f"    PCA spectrum (cumulative variance):")
    for i in range(min(10, len(cumvar))):
        print(f"      PC{i+1}: {cumvar[i]:.1%}")
    print(f"    Dimensions for 90% variance: {n_dims_90}")
    print(f"    Dimensions for 95% variance: {n_dims_95}")

    elapsed = time.time() - t0
    print(f"    PCA done in {elapsed:.1f}s")

    return {
        "n_primitives": len(prim_vecs),
        "singular_values": [round(float(s), 4) for s in S[:min(15, len(S))]],
        "cumulative_variance": [round(float(c), 4) for c in cumvar[:min(15, len(cumvar))]],
        "dims_for_90pct": n_dims_90,
        "dims_for_95pct": n_dims_95,
        "effective_dim_5pc": round(float(cumvar[min(4, len(cumvar)-1)]), 4),
    }


def go_cluster_concordance(token_clusters, vocab_dept, K):
    """
    Compute ARI and NMI between unsupervised token clusters and GO departments.
    Low ARI + surviving collinearity = algebra is GO-independent.
    """
    labels_go = []
    labels_cluster = []

    for tok, c in token_clusters.items():
        d = vocab_dept.get(tok.upper())
        if d and d in D2I:
            labels_go.append(d)
            labels_cluster.append(c)

    if len(labels_go) < 100:
        return {"ari": 0, "nmi": 0, "n_compared": len(labels_go)}

    ari = adjusted_rand_index(labels_go, labels_cluster)
    nmi = normalized_mutual_info(labels_go, labels_cluster)

    return {
        "ari": round(float(ari), 4),
        "nmi": round(float(nmi), 4),
        "n_compared": len(labels_go),
    }


def shuffled_cluster_control(ptt, ttp, gene_to_uids, token_clusters, tok_list, K,
                              testable, n_shuffles=5):
    """
    Randomly reassign tokens to K clusters (preserving sizes).
    Collinearity should vanish in shuffled assignments.
    """
    print(f"\n  [SHUFFLE K={K}] Running {n_shuffles} shuffled-token-cluster controls...")
    t0 = time.time()

    toks = list(token_clusters.keys())
    original_labels = [token_clusters[t] for t in toks]

    shuffled_ds = []
    for si in range(n_shuffles):
        rng = np.random.RandomState(si + 1000)
        shuffled = list(original_labels)
        rng.shuffle(shuffled)

        shuffled_tc = {t: shuffled[i] for i, t in enumerate(toks)}

        profiles = compute_disruption_profiles_token_clusters(
            ptt, ttp, gene_to_uids, shuffled_tc, K, testable, max_genes=1500
        )

        result = compute_collinearity_d(profiles, testable, rng_seed=si + 2000)
        shuffled_ds.append(result["d"])
        print(f"      Shuffle {si+1}/{n_shuffles}: d={result['d']:+.4f}")

    elapsed = time.time() - t0
    arr = np.array(shuffled_ds)
    print(f"    Shuffled controls done in {elapsed:.1f}s")
    print(f"    Shuffled d: mean={arr.mean():+.4f} ± {arr.std():.4f}, "
          f"range=[{arr.min():+.4f}, {arr.max():+.4f}]")

    return {
        "mean_d": round(float(arr.mean()), 4),
        "std_d": round(float(arr.std()), 4),
        "range": [round(float(arr.min()), 4), round(float(arr.max()), 4)],
        "all_ds": [round(float(d), 4) for d in shuffled_ds],
        "n_shuffles": n_shuffles,
    }


def run_single_K(K, data):
    """Run the full GO-free test for a single value of K."""
    print(f"\n{'='*72}")
    print(f"  GO-FREE DIMENSION TEST: K={K}")
    print(f"{'='*72}")

    ptt = data["ptt"]
    ttp = data["ttp"]
    gene_cache = data["gene_cache"]
    gene_to_uids = data["gene_to_uids"]
    testable = data["testable"]
    primitives = data["primitives"]
    vocab_dept = data["vocab_dept"]

    token_clusters, tok_list = cluster_tokens(ttp, ptt, K)

    concordance = go_cluster_concordance(token_clusters, vocab_dept, K)
    print(f"\n  GO concordance: ARI={concordance['ari']:.4f}, NMI={concordance['nmi']:.4f}")

    profiles = compute_disruption_profiles_token_clusters(
        ptt, ttp, gene_to_uids, token_clusters, K, testable
    )

    collinearity = compute_collinearity_d(profiles, testable)
    print(f"\n  COLLINEARITY (K={K}): d={collinearity['d']:+.4f}")
    print(f"    within={collinearity['within_mean']:.4f}, across={collinearity['across_mean']:.4f}")
    print(f"    Δ={collinearity['delta']:.4f}")

    pca = pca_dimensionality_analysis(profiles, primitives, testable,
                                       gene_cache, ptt, vocab_dept, K)

    shuffle = shuffled_cluster_control(
        ptt, ttp, gene_to_uids, token_clusters, tok_list, K, testable,
        n_shuffles=N_SHUFFLE_CONTROLS
    )

    if shuffle["mean_d"] != 0:
        z_vs_shuffle = (collinearity["d"] - shuffle["mean_d"]) / max(shuffle["std_d"], 1e-6)
    else:
        z_vs_shuffle = float("inf") if collinearity["d"] > 0 else 0

    d_val = collinearity["d"]
    if d_val > 0.8:
        verdict = "STRONG_SURVIVAL"
    elif d_val > 0.5:
        verdict = "CLEAR_SURVIVAL"
    elif d_val > 0.3:
        verdict = "PARTIAL_SURVIVAL"
    else:
        verdict = "FAILURE"

    print(f"\n  === K={K} VERDICT: {verdict} ===")
    print(f"    Collinearity d = {d_val:+.4f}")
    print(f"    Shuffled control d = {shuffle['mean_d']:+.4f}")
    print(f"    Z vs shuffle = {z_vs_shuffle:+.2f}")
    print(f"    GO concordance ARI = {concordance['ari']:.4f}")
    if pca and "dims_for_90pct" in pca:
        print(f"    PCA 90% dims = {pca['dims_for_90pct']}")
        print(f"    PCA 5-PC capture = {pca.get('effective_dim_5pc', 0):.1%}")

    return {
        "K": K,
        "collinearity": collinearity,
        "pca": pca,
        "go_concordance": concordance,
        "shuffled_control": shuffle,
        "z_vs_shuffle": round(float(z_vs_shuffle), 2),
        "verdict": verdict,
    }


def main():
    print("=" * 72)
    print("  GO-FREE DIMENSION TEST")
    print("  Response to Reviewer Concerns 2 & 6")
    print("  'Is the algebra a property of the proteome or of GO annotations?'")
    print("=" * 72)

    data = load_data()

    all_results = {}
    for K in K_VALUES:
        result = run_single_K(K, data)
        all_results[f"K={K}"] = result
        sys.stdout.flush()

    print(f"\n\n{'='*72}")
    print(f"  FINAL SUMMARY — GO-FREE DIMENSION TEST")
    print(f"{'='*72}")
    print(f"\n  {'K':>4s}  {'d_real':>7s}  {'d_shuf':>7s}  {'Z':>6s}  {'ARI':>6s}  {'NMI':>6s}  "
          f"{'PCA_5D':>7s}  {'90%_dim':>7s}  {'Verdict'}")
    print(f"  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*20}")

    verdicts = []
    for K in K_VALUES:
        r = all_results[f"K={K}"]
        c = r["collinearity"]
        s = r["shuffled_control"]
        g = r["go_concordance"]
        p = r["pca"]

        pca_5d = f"{p.get('effective_dim_5pc', 0):.1%}" if isinstance(p, dict) and "effective_dim_5pc" in p else "N/A"
        pca_90 = str(p.get("dims_for_90pct", "N/A")) if isinstance(p, dict) else "N/A"

        print(f"  {K:4d}  {c['d']:+7.4f}  {s['mean_d']:+7.4f}  {r['z_vs_shuffle']:+6.1f}  "
              f"{g['ari']:6.4f}  {g['nmi']:6.4f}  {pca_5d:>7s}  {pca_90:>7s}  {r['verdict']}")
        verdicts.append(r["verdict"])

    survival_count = sum(1 for v in verdicts if "SURVIVAL" in v)
    total = len(verdicts)

    print(f"\n  Algebra survives in {survival_count}/{total} K-values tested.")

    if survival_count == total:
        overall = "DEFINITIVE_SURVIVAL"
        print(f"\n  OVERALL VERDICT: DEFINITIVE SURVIVAL")
        print(f"  The commutative algebra is a coordinate-free property of the")
        print(f"  dispatch graph manifold, NOT an artifact of GO annotation structure.")
        print(f"  The reviewer's concern is fully addressed.")
    elif survival_count > total // 2:
        overall = "MAJORITY_SURVIVAL"
        print(f"\n  OVERALL VERDICT: MAJORITY SURVIVAL")
        print(f"  The algebra survives in most dimensionalities tested.")
        print(f"  The GO contribution is partial but the structure is largely real.")
    elif survival_count > 0:
        overall = "PARTIAL_SURVIVAL"
        print(f"\n  OVERALL VERDICT: PARTIAL SURVIVAL")
        print(f"  The algebra survives in some but not all dimensionalities.")
        print(f"  Further investigation needed.")
    else:
        overall = "FAILURE"
        print(f"\n  OVERALL VERDICT: FAILURE")
        print(f"  The algebra does not survive GO-free re-dimensioning.")
        print(f"  The reviewer's concern is validated.")

    output = {
        "experiment": "GO-free dimension test",
        "purpose": "Test whether the commutative algebra survives replacement of "
                   "22 GO-derived departments with unsupervised graph-topology clusters",
        "method": "Spectral clustering on protein co-occurrence graph (no GO input)",
        "K_values_tested": K_VALUES,
        "success_criterion": "Cohen's d > 0.5 for collinearity in unsupervised space",
        "results_by_K": all_results,
        "survival_count": survival_count,
        "total_K_tested": total,
        "overall_verdict": overall,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Saved: {OUTPUT_PATH}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
