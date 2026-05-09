#!/usr/bin/env python3
"""
GO-Free Permutation Test — Definitive response to Reviewer Concerns 2 & 6.
==========================================================================

THE SIMPLEST, MOST POWERFUL TEST:

If the algebra is "just an artifact of GO annotations," then ANY assignment
of tokens to 22 departments (preserving department sizes) should produce
comparable collinearity. If the algebra is real and GO captures genuine
biology, then the SPECIFIC token→department mapping matters, and randomly
shuffling it should destroy collinearity.

METHOD:
  1. Compute collinearity d in the real GO-based 22D space (baseline)
  2. Randomly permute which department each token belongs to, preserving
     department sizes (N_SHUFFLES times)
  3. Re-compute collinearity in each shuffled department space
  4. Compare: if d_real >> d_shuffled, GO labels carry real information
     that the algebra uses — the algebra is NOT circular

ADDITIONAL TESTS:
  A. Permutation test on primitives (shuffle gene→primitive assignments)
  B. Held-out department test: remove one department at a time, compute
     collinearity in the remaining 21D space
  C. Token-frequency partitioning: group tokens by carrier count (no GO)
     and test collinearity in this alternative space

Usage:
    python3 -u validation/knockout/go_free_permutation_test.py
"""

import csv
import json
import os
import pickle
import sys
import time
import numpy as np
from collections import defaultdict, Counter

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
N_DEPTS = len(VALID_DEPARTMENTS)

N_PERM_SHUFFLES = 100


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

    prim_to_genes = {}
    for p in primitives:
        carriers = [uid for uid, seq in protein_dept_seqs.items() if p["search"] in seq]
        genes = set()
        for uid in carriers:
            g = gene_cache.get(uid)
            if g:
                genes.add(g)
        prim_to_genes[p["search"]] = list(genes)

    testable = {p: genes for p, genes in prim_to_genes.items() if len(genes) >= 5}

    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Proteins: {len(ptt):,}, Tokens: {len(ttp):,}")
    print(f"  Testable primitives: {len(testable)}")

    return {
        "ptt": ptt, "ttp": ttp, "gene_cache": gene_cache,
        "gene_to_uids": gene_to_uids, "vocab_dept": vocab_dept,
        "testable": testable, "protein_dept_seqs": protein_dept_seqs,
    }


def compute_gene_tok_profiles(ptt, gene_to_uids, tok_dept_map, testable):
    """
    For each gene, compute its token-department profile:
    profile[d] = (# of gene's tokens in department d) / (total tokens in dept d)

    tok_dept_map: dict mapping token_hex -> department_index
    """
    dept_sizes = np.zeros(N_DEPTS)
    for tok, di in tok_dept_map.items():
        dept_sizes[di] += 1

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

        if not gene_tokens:
            continue

        profile = np.zeros(N_DEPTS)
        for tok in gene_tokens:
            di = tok_dept_map.get(tok)
            if di is not None:
                profile[di] += 1

        for di in range(N_DEPTS):
            if dept_sizes[di] > 0:
                profile[di] /= dept_sizes[di]

        if np.linalg.norm(profile) > 1e-12:
            profiles[gene] = profile

    return profiles


def compute_collinearity_d(profiles_dict, testable, rng_seed=42, max_pairs_per_prim=50):
    rng = np.random.RandomState(rng_seed)
    all_genes = list(profiles_dict.keys())
    if not all_genes:
        return {"d": 0.0, "within_mean": 0, "across_mean": 0, "n_prims_tested": 0}

    within_cos = []
    across_cos = []
    prims_tested = 0

    for prim, genes in testable.items():
        vecs = [profiles_dict[g] for g in genes
                if g in profiles_dict and np.linalg.norm(profiles_dict[g]) > 1e-10]
        if len(vecs) < 3:
            continue
        prims_tested += 1

        pairs = 0
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                within_cos.append(cosine_sim(vecs[i], vecs[j]))
                pairs += 1
                if pairs >= max_pairs_per_prim:
                    break
            if pairs >= max_pairs_per_prim:
                break

        n_rand = min(len(genes) * 2, 50)
        rand = rng.choice(all_genes, size=min(n_rand, len(all_genes)), replace=False)
        rvecs = [profiles_dict[g] for g in rand
                 if g in profiles_dict and np.linalg.norm(profiles_dict[g]) > 1e-10]
        pairs = 0
        for i in range(min(len(vecs), 20)):
            for j in range(min(len(rvecs), 20)):
                across_cos.append(cosine_sim(vecs[i], rvecs[j]))
                pairs += 1
                if pairs >= max_pairs_per_prim:
                    break
            if pairs >= max_pairs_per_prim:
                break

    if not within_cos or not across_cos:
        return {"d": 0.0, "within_mean": 0, "across_mean": 0, "n_prims_tested": 0}

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
        "n_prims_tested": prims_tested,
    }


def build_tok_dept_map(vocab_dept):
    """Build the real token→department mapping."""
    tok_dept_map = {}
    for tok, dept in vocab_dept.items():
        if dept in D2I:
            tok_dept_map[tok] = D2I[dept]
    return tok_dept_map


def shuffled_tok_dept_map(real_map, rng):
    """Randomly permute the token→department assignments, preserving dept sizes."""
    toks = list(real_map.keys())
    depts = [real_map[t] for t in toks]
    rng.shuffle(depts)
    return {t: d for t, d in zip(toks, depts)}


# ─── TEST A: Department Label Permutation ────────────────────────────

def test_department_permutation(ptt, gene_to_uids, vocab_dept, testable):
    """
    The core test: randomly permute tok→dept assignments and compare
    collinearity to the real assignments.
    """
    print(f"\n{'='*72}")
    print(f"  TEST A: Department Label Permutation (n={N_PERM_SHUFFLES})")
    print(f"  'Does the SPECIFIC tok→dept mapping matter?'")
    print(f"{'='*72}")

    real_map = build_tok_dept_map(vocab_dept)
    print(f"  Tokens with department labels: {len(real_map):,}")

    dept_counts = Counter(real_map.values())
    print(f"  Department sizes: min={min(dept_counts.values())}, max={max(dept_counts.values())}, "
          f"median={int(np.median(list(dept_counts.values())))}")

    print(f"\n  Computing REAL collinearity...")
    real_profiles = compute_gene_tok_profiles(ptt, gene_to_uids, real_map, testable)
    real_result = compute_collinearity_d(real_profiles, testable)
    print(f"  REAL d = {real_result['d']:+.4f}")
    print(f"    within={real_result['within_mean']:.4f}, across={real_result['across_mean']:.4f}")
    print(f"    Δ={real_result['delta']:.4f}, prims tested={real_result['n_prims_tested']}")

    print(f"\n  Computing SHUFFLED collinearity ({N_PERM_SHUFFLES} permutations)...")
    shuffled_ds = []
    for si in range(N_PERM_SHUFFLES):
        rng = np.random.RandomState(si + 7000)
        shuf_map = shuffled_tok_dept_map(real_map, rng)
        shuf_profiles = compute_gene_tok_profiles(ptt, gene_to_uids, shuf_map, testable)
        shuf_result = compute_collinearity_d(shuf_profiles, testable, rng_seed=si + 8000)
        shuffled_ds.append(shuf_result["d"])
        if (si + 1) % 20 == 0:
            arr = np.array(shuffled_ds)
            print(f"    [{si+1}/{N_PERM_SHUFFLES}] mean={arr.mean():+.4f} ± {arr.std():.4f}")

    arr = np.array(shuffled_ds)
    z_score = (real_result["d"] - arr.mean()) / max(arr.std(), 1e-8)
    p_value = float(np.mean(arr >= real_result["d"]))

    print(f"\n  RESULTS:")
    print(f"    Real d:     {real_result['d']:+.4f}")
    print(f"    Shuffled d: {arr.mean():+.4f} ± {arr.std():.4f}")
    print(f"    Range:      [{arr.min():+.4f}, {arr.max():+.4f}]")
    print(f"    Z-score:    {z_score:+.2f}")
    print(f"    p-value:    {p_value} ({sum(arr >= real_result['d'])}/{N_PERM_SHUFFLES} >= real)")

    retention = arr.mean() / real_result["d"] if real_result["d"] > 0 else float("inf")
    go_contribution = 1.0 - retention if retention < 1.0 else 0.0

    print(f"\n    Retention under shuffling: {retention:.1%}")
    print(f"    GO-specific contribution: {go_contribution:.1%}")

    if z_score > 5:
        verdict = "STRONG: GO labels carry major real information"
    elif z_score > 3:
        verdict = "CLEAR: GO labels carry significant information"
    elif z_score > 2:
        verdict = "MODERATE: GO labels contribute measurably"
    else:
        verdict = "WEAK: GO labels contribute little beyond random"

    print(f"    VERDICT: {verdict}")

    return {
        "real_d": real_result["d"],
        "real_collinearity": real_result,
        "shuffled_mean_d": round(float(arr.mean()), 4),
        "shuffled_std_d": round(float(arr.std()), 4),
        "shuffled_range": [round(float(arr.min()), 4), round(float(arr.max()), 4)],
        "z_score": round(float(z_score), 2),
        "p_value": p_value,
        "retention": round(retention, 4),
        "go_contribution": round(go_contribution, 4),
        "verdict": verdict,
    }


# ─── TEST B: Primitive Permutation ───────────────────────────────────

def test_primitive_permutation(ptt, gene_to_uids, vocab_dept, testable, n_shuffles=100):
    """
    Shuffle gene→primitive assignments (preserving sizes).
    If collinearity vanishes, the PRIMITIVES are carrying real signal.
    """
    print(f"\n{'='*72}")
    print(f"  TEST B: Primitive Permutation (n={n_shuffles})")
    print(f"  'Do the SPECIFIC primitive-gene assignments matter?'")
    print(f"{'='*72}")

    real_map = build_tok_dept_map(vocab_dept)
    real_profiles = compute_gene_tok_profiles(ptt, gene_to_uids, real_map, testable)
    real_result = compute_collinearity_d(real_profiles, testable)
    print(f"  Real d = {real_result['d']:+.4f}")

    all_genes = list(real_profiles.keys())
    shuffled_ds = []

    for si in range(n_shuffles):
        rng = np.random.RandomState(si + 9000)
        fake_testable = {}
        for prim, genes in testable.items():
            n = len(genes)
            fake_genes = list(rng.choice(all_genes, size=min(n, len(all_genes)), replace=False))
            fake_testable[prim] = fake_genes

        shuf_result = compute_collinearity_d(real_profiles, fake_testable, rng_seed=si + 10000)
        shuffled_ds.append(shuf_result["d"])
        if (si + 1) % 20 == 0:
            arr = np.array(shuffled_ds)
            print(f"    [{si+1}/{n_shuffles}] mean={arr.mean():+.4f} ± {arr.std():.4f}")

    arr = np.array(shuffled_ds)
    z_score = (real_result["d"] - arr.mean()) / max(arr.std(), 1e-8)
    p_value = float(np.mean(arr >= real_result["d"]))

    print(f"\n  RESULTS:")
    print(f"    Real d:     {real_result['d']:+.4f}")
    print(f"    Shuffled d: {arr.mean():+.4f} ± {arr.std():.4f}")
    print(f"    Z-score:    {z_score:+.2f}")
    print(f"    p-value:    {p_value}")

    return {
        "real_d": real_result["d"],
        "shuffled_mean_d": round(float(arr.mean()), 4),
        "shuffled_std_d": round(float(arr.std()), 4),
        "z_score": round(float(z_score), 2),
        "p_value": p_value,
    }


# ─── TEST C: Token-Frequency Binning (alternative non-GO dimensions) ─

def test_token_frequency_binning(ptt, ttp, gene_to_uids, testable, K=22):
    """
    Group tokens by their carrier count (no GO input).
    Equal-mass binning: sort tokens by # carriers, split into K bins.
    This is a completely GO-free 22D space based purely on network degree.
    """
    print(f"\n{'='*72}")
    print(f"  TEST C: Token-Frequency Binning (K={K}, no GO)")
    print(f"  'Does collinearity exist in a non-functional dimension space?'")
    print(f"{'='*72}")

    all_toks = [(tok, len(carriers)) for tok, carriers in ttp.items() if len(carriers) >= 5]
    all_toks.sort(key=lambda x: x[1])
    n = len(all_toks)
    print(f"  Tokens with >= 5 carriers: {n:,}")

    bin_size = n // K
    freq_map = {}
    for i, (tok, _) in enumerate(all_toks):
        bin_idx = min(i // bin_size, K - 1)
        freq_map[tok.upper()] = bin_idx

    bin_counts = Counter(freq_map.values())
    print(f"  Bin sizes: min={min(bin_counts.values())}, max={max(bin_counts.values())}")

    profiles = {}
    relevant_genes = set()
    for genes in testable.values():
        relevant_genes.update(genes)

    dept_sizes = np.zeros(K)
    for tok, bi in freq_map.items():
        dept_sizes[bi] += 1

    for gene in relevant_genes:
        uids = gene_to_uids.get(gene, [])
        if not uids:
            continue

        gene_tokens = set()
        for uid in uids:
            gene_tokens.update(ptt.get(uid, []))

        profile = np.zeros(K)
        for tok in gene_tokens:
            bi = freq_map.get(tok.upper())
            if bi is not None:
                profile[bi] += 1

        for bi in range(K):
            if dept_sizes[bi] > 0:
                profile[bi] /= dept_sizes[bi]

        if np.linalg.norm(profile) > 1e-12:
            profiles[gene] = profile

    result = compute_collinearity_d(profiles, testable)
    print(f"  Frequency-binned collinearity: d={result['d']:+.4f}")
    print(f"    within={result['within_mean']:.4f}, across={result['across_mean']:.4f}")

    shuffled_ds = []
    for si in range(20):
        rng = np.random.RandomState(si + 11000)
        toks = list(freq_map.keys())
        vals = list(freq_map.values())
        rng.shuffle(vals)
        shuf_map = {t: v for t, v in zip(toks, vals)}

        shuf_profiles = {}
        shuf_sizes = np.zeros(K)
        for tok, bi in shuf_map.items():
            shuf_sizes[bi] += 1

        for gene in relevant_genes:
            uids = gene_to_uids.get(gene, [])
            if not uids:
                continue
            gene_tokens = set()
            for uid in uids:
                gene_tokens.update(ptt.get(uid, []))
            profile = np.zeros(K)
            for tok in gene_tokens:
                bi = shuf_map.get(tok.upper())
                if bi is not None:
                    profile[bi] += 1
            for bi in range(K):
                if shuf_sizes[bi] > 0:
                    profile[bi] /= shuf_sizes[bi]
            if np.linalg.norm(profile) > 1e-12:
                shuf_profiles[gene] = profile

        shuf_result = compute_collinearity_d(shuf_profiles, testable, rng_seed=si + 12000)
        shuffled_ds.append(shuf_result["d"])

    arr = np.array(shuffled_ds)
    z_score = (result["d"] - arr.mean()) / max(arr.std(), 1e-8)
    print(f"  Shuffled frequency-bins d: {arr.mean():+.4f} ± {arr.std():.4f}")
    print(f"  Z vs shuffled: {z_score:+.2f}")

    return {
        "collinearity": result,
        "shuffled_mean_d": round(float(arr.mean()), 4),
        "shuffled_std_d": round(float(arr.std()), 4),
        "z_score": round(float(z_score), 2),
    }


# ─── TEST D: Held-out Department Test ────────────────────────────────

def test_held_out_departments(ptt, gene_to_uids, vocab_dept, testable):
    """
    Remove one department at a time and compute collinearity in the
    remaining 21D space. If any single department drives the signal,
    removing it will collapse d. If the algebra is distributed across
    departments, d should be robust to single-department removal.
    """
    print(f"\n{'='*72}")
    print(f"  TEST D: Held-Out Department Test")
    print(f"  'Is the collinearity driven by one dominant department?'")
    print(f"{'='*72}")

    real_map = build_tok_dept_map(vocab_dept)
    real_profiles = compute_gene_tok_profiles(ptt, gene_to_uids, real_map, testable)
    real_result = compute_collinearity_d(real_profiles, testable)
    print(f"  Full 22D d = {real_result['d']:+.4f}")

    results = {}
    for held_out_dept in VALID_DEPARTMENTS:
        held_out_idx = D2I[held_out_dept]

        reduced_profiles = {}
        for gene, profile in real_profiles.items():
            reduced = np.delete(profile, held_out_idx)
            if np.linalg.norm(reduced) > 1e-12:
                reduced_profiles[gene] = reduced

        result = compute_collinearity_d(reduced_profiles, testable)
        delta_d = result["d"] - real_result["d"]
        results[held_out_dept] = {
            "d": result["d"],
            "delta_d": round(delta_d, 4),
        }

    ds = [r["d"] for r in results.values()]
    print(f"\n  d values with one dept removed:")
    for dept in VALID_DEPARTMENTS:
        r = results[dept]
        arrow = "↓" if r["delta_d"] < -0.05 else "↑" if r["delta_d"] > 0.05 else "→"
        print(f"    {dept:20s}: d={r['d']:+.4f} ({r['delta_d']:+.4f}) {arrow}")

    print(f"\n  Range: [{min(ds):+.4f}, {max(ds):+.4f}]")
    print(f"  Mean: {np.mean(ds):+.4f}")
    print(f"  Most impactful removal: {min(results, key=lambda k: results[k]['d'])}")

    return {
        "full_d": real_result["d"],
        "per_department": results,
        "min_d": round(float(min(ds)), 4),
        "max_d": round(float(max(ds)), 4),
        "mean_d": round(float(np.mean(ds)), 4),
    }


def main():
    print("=" * 72)
    print("  GO-FREE PERMUTATION TEST")
    print("  'Is the algebra a property of GO annotations or of the proteome?'")
    print("  Definitive response to Reviewer Concerns 2 & 6")
    print("=" * 72)

    data = load_data()

    result_a = test_department_permutation(
        data["ptt"], data["gene_to_uids"], data["vocab_dept"], data["testable"]
    )

    result_b = test_primitive_permutation(
        data["ptt"], data["gene_to_uids"], data["vocab_dept"], data["testable"]
    )

    result_c = test_token_frequency_binning(
        data["ptt"], data["ttp"], data["gene_to_uids"], data["testable"]
    )

    result_d = test_held_out_departments(
        data["ptt"], data["gene_to_uids"], data["vocab_dept"], data["testable"]
    )

    print(f"\n\n{'='*72}")
    print(f"  COMBINED SUMMARY")
    print(f"{'='*72}")

    print(f"\n  Test A (Dept Permutation):")
    print(f"    Real d={result_a['real_d']:+.4f}, Shuffled d={result_a['shuffled_mean_d']:+.4f}")
    print(f"    Z={result_a['z_score']:+.2f}, p={result_a['p_value']}")
    print(f"    GO contributes {result_a['go_contribution']:.1%} of collinearity signal")

    print(f"\n  Test B (Primitive Permutation):")
    print(f"    Real d={result_b['real_d']:+.4f}, Shuffled d={result_b['shuffled_mean_d']:+.4f}")
    print(f"    Z={result_b['z_score']:+.2f}, p={result_b['p_value']}")

    print(f"\n  Test C (Frequency Binning):")
    freq_d = result_c['collinearity']['d']
    print(f"    Freq-binned d={freq_d:+.4f}, Shuffled d={result_c['shuffled_mean_d']:+.4f}")
    print(f"    Z={result_c['z_score']:+.2f}")

    print(f"\n  Test D (Held-out Departments):")
    print(f"    Full d={result_d['full_d']:+.4f}")
    print(f"    Range with one dept removed: [{result_d['min_d']:+.4f}, {result_d['max_d']:+.4f}]")

    print(f"\n  INTERPRETATION:")
    if result_a["z_score"] > 3 and result_b["z_score"] > 3:
        print(f"    ✓ The GO department labels carry statistically significant real information")
        print(f"      (Z={result_a['z_score']:.1f} for dept permutation, "
              f"Z={result_b['z_score']:.1f} for primitive permutation)")
        print(f"    ✓ The algebra is NOT an artifact of GO — it requires both:")
        print(f"      (a) the specific token→dept mapping (captured by GO)")
        print(f"      (b) the specific gene→primitive mapping (captured by byte-stream encoding)")
        print(f"    ✓ GO contributes real biological structure that the algebra leverages,")
        print(f"      but the algebra adds {result_a['retention']:.1%} beyond random dept labels")
    else:
        print(f"    Results require careful interpretation.")

    output = {
        "experiment": "GO-free permutation test",
        "purpose": "Determine whether the algebraic collinearity is an artifact of "
                   "GO annotations or reflects genuine proteome structure",
        "test_A_dept_permutation": result_a,
        "test_B_primitive_permutation": result_b,
        "test_C_frequency_binning": result_c,
        "test_D_held_out_departments": result_d,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Saved: {OUTPUT_PATH}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
