#!/usr/bin/env python3
"""
The Compiler Test — predicting knockout effects from program structure.
======================================================================

If the genome is a deterministic program with consistent primitives:
  1. Learn each primitive's canonical disruption signature from a TRAINING set of proteins
  2. For HELD-OUT proteins, predict disruption from program composition alone
  3. Compare predicted vs actual knockout profiles

Random 50/50 protein split with seed for reproducibility. Primitives learned
from one half of the proteome predicting the other half proves the program
structure encodes portable, deterministic computation.

Usage:
    python3 -u validation/knockout/lisp_compiler_test.py
"""

import csv
import json
import os
import pickle
import random
import sys
import time
from collections import defaultdict

import numpy as np
from scipy import sparse, stats

STATE_PATH = "/tmp/module8_full_state.pkl"
VOCAB_PATH = "server/data/human/vocabulary.csv"
PRIMITIVES_PATH = "beta_transfer/genome_primitives.csv"
GENE_DEPTS_PATH = "server/data/human/gene_departments.csv"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "lisp_compiler_results.json")

VALID_DEPARTMENTS = sorted([
    "Apoptosis", "Cell adhesion", "Cell cycle", "Chromatin", "Cytoskeleton",
    "DNA repair", "GTPase", "Immune", "Ion channel", "Kinase", "Methylation",
    "Nuc acid bind", "Phosphatase", "Protein folding", "Proteolysis",
    "RNA processing", "Signaling", "Structural", "Transcription",
    "Translation", "Transport", "Ubiquitin",
])
DEPT_TO_IDX = {d: i for i, d in enumerate(VALID_DEPARTMENTS)}
N_DEPTS = len(VALID_DEPARTMENTS)

SPLIT_SEED = 42


def load_state():
    print("[1/7] Loading dispatch graph state...")
    with open(STATE_PATH, "rb") as f:
        state = pickle.load(f)
    return state


def build_token_dept_map():
    vocab_dept = {}
    with open(VOCAB_PATH) as f:
        for row in csv.DictReader(f):
            stripped = row["word_hex"].replace("0x", "").upper()
            vocab_dept[stripped] = row["primary_function"]
    return vocab_dept


def build_protein_data(state, vocab_dept):
    print("[2/7] Building protein department sequences...")
    ptt = state["ptt"]
    protein_dept_lists = {}
    protein_dept_seqs = {}
    for uid, tokens in ptt.items():
        depts = []
        for tok in tokens:
            dept = vocab_dept.get(tok.upper())
            if dept and dept in DEPT_TO_IDX:
                depts.append(dept)
        if depts:
            compressed = []
            for d in depts:
                if not compressed or compressed[-1] != d:
                    compressed.append(d)
            protein_dept_lists[uid] = compressed
            protein_dept_seqs[uid] = "|".join(compressed)
    print(f"  {len(protein_dept_seqs)} proteins with dept sequences")
    return protein_dept_seqs, protein_dept_lists


def build_disruption_infrastructure(state):
    print("[3/7] Building disruption matrices...")
    t0 = time.time()
    ptt = state["ptt"]
    ttp = state["ttp"]
    gene_cache = state["gene_cache"]

    gene_depts = {}
    with open(GENE_DEPTS_PATH) as f:
        for row in csv.DictReader(f):
            gene_depts[row["gene"]] = row["department"]

    all_tokens = sorted(ttp.keys())
    all_proteins = sorted(ptt.keys())
    tok_to_idx = {t: i for i, t in enumerate(all_tokens)}
    uid_to_idx = {u: i for i, u in enumerate(all_proteins)}
    n_p, n_t = len(all_proteins), len(all_tokens)

    rows, cols = [], []
    for tok, uids in ttp.items():
        ti = tok_to_idx[tok]
        for uid in uids:
            rows.append(uid_to_idx[uid])
            cols.append(ti)
    P = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_p, n_t),
    )

    dept_token_disruption = np.zeros((N_DEPTS, n_t), dtype=np.float32)
    for di, dept in enumerate(VALID_DEPARTMENTS):
        mask = np.zeros(n_p, dtype=np.float32)
        for uid, idx in uid_to_idx.items():
            gene = gene_cache.get(uid)
            if gene and gene_depts.get(gene) == dept:
                mask[idx] = 1.0
        dept_token_disruption[di] = np.asarray(mask @ P).flatten()

    print(f"  Built in {time.time()-t0:.1f}s")
    return tok_to_idx, uid_to_idx, dept_token_disruption


def protein_dept_profile(uid, ptt, tok_to_idx, dept_token_disruption):
    tokens = ptt.get(uid, [])
    idxs = [tok_to_idx[t] for t in tokens if t in tok_to_idx]
    if not idxs:
        return np.zeros(N_DEPTS)
    return dept_token_disruption[:, np.array(idxs, dtype=np.int32)].sum(axis=1)


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def split_proteins(protein_dept_seqs):
    print("[4/7] Splitting proteins into train/test (50/50, seed=42)...")
    all_uids = sorted(protein_dept_seqs.keys())
    random.seed(SPLIT_SEED)
    random.shuffle(all_uids)
    mid = len(all_uids) // 2
    train_uids = set(all_uids[:mid])
    test_uids = set(all_uids[mid:])
    print(f"  Train: {len(train_uids)} proteins")
    print(f"  Test:  {len(test_uids)} proteins")
    return train_uids, test_uids


def compute_population_mean(train_uids, protein_dept_seqs, ptt, tok_to_idx,
                              dept_token_disruption):
    print("  Computing population mean profile from training set...")
    profiles = []
    for uid in train_uids:
        if uid in protein_dept_seqs:
            p = protein_dept_profile(uid, ptt, tok_to_idx, dept_token_disruption)
            profiles.append(p)
    pop_mean = np.mean(profiles, axis=0)
    pop_std = np.std(profiles, axis=0)
    pop_std[pop_std == 0] = 1.0
    print(f"  Population mean top depts: {', '.join(VALID_DEPARTMENTS[i] for i in np.argsort(-pop_mean)[:3])}")
    return pop_mean, pop_std


def learn_primitive_signatures(primitives, train_uids, protein_dept_seqs,
                                 ptt, tok_to_idx, dept_token_disruption,
                                 pop_mean, pop_std):
    print("[5/7] Learning primitive DEVIATION signatures from training set...")
    t0 = time.time()

    signatures = {}
    stats_out = []

    for pi, prim in enumerate(primitives):
        prim_seq = prim["function_sequence"]
        prim_depts = [d for d in prim_seq.split("|") if d in DEPT_TO_IDX]
        if not prim_depts:
            continue
        search_seq = "|".join(prim_depts)

        train_carriers = [uid for uid in train_uids
                          if uid in protein_dept_seqs and search_seq in protein_dept_seqs[uid]]

        if len(train_carriers) < 3:
            continue

        profiles = np.zeros((len(train_carriers), N_DEPTS))
        for i, uid in enumerate(train_carriers):
            profiles[i] = protein_dept_profile(uid, ptt, tok_to_idx, dept_token_disruption)

        mean_profile = profiles.mean(axis=0)
        deviation = (mean_profile - pop_mean) / pop_std

        signatures[prim_seq] = {
            "deviation": deviation,
            "n_train": len(train_carriers),
            "raw_mean": mean_profile,
        }

        top_pos = np.argsort(-deviation)[:2]
        top_neg = np.argsort(deviation)[:2]
        stats_out.append({
            "primitive": prim_seq,
            "n_train_carriers": len(train_carriers),
            "enriched": [f"{VALID_DEPARTMENTS[i]}({deviation[i]:+.2f})" for i in top_pos],
            "depleted": [f"{VALID_DEPARTMENTS[i]}({deviation[i]:+.2f})" for i in top_neg],
        })

    print(f"  Learned {len(signatures)} primitive deviation signatures ({time.time()-t0:.1f}s)")
    for s in sorted(stats_out, key=lambda x: -x["n_train_carriers"])[:10]:
        print(f"    {s['primitive'][:40]:<40s} N={s['n_train_carriers']:5d} "
              f"UP: {', '.join(s['enriched'])}  DOWN: {', '.join(s['depleted'])}")

    return signatures


def predict_and_evaluate(signatures, test_uids, protein_dept_seqs, protein_dept_lists,
                           ptt, tok_to_idx, dept_token_disruption, pop_mean, pop_std):
    print("[6/7] Predicting held-out protein disruption from program structure...")
    t0 = time.time()

    cosines = []
    correlations = []
    n_prims_used = []
    protein_details = []

    mean_only_cosines = []

    prim_seqs_sorted = sorted(signatures.keys(), key=len, reverse=True)

    tested = 0
    skipped_no_prim = 0

    test_uid_list = [uid for uid in test_uids if uid in protein_dept_seqs]

    for uid in test_uid_list:
        seq = protein_dept_seqs[uid]

        matched_prims = []
        for ps in prim_seqs_sorted:
            ps_depts = [d for d in ps.split("|") if d in DEPT_TO_IDX]
            ps_search = "|".join(ps_depts)
            if ps_search in seq:
                matched_prims.append(ps)

        if not matched_prims:
            skipped_no_prim += 1
            continue

        combined_deviation = np.zeros(N_DEPTS)
        for ps in matched_prims:
            combined_deviation += signatures[ps]["deviation"]
        combined_deviation /= len(matched_prims)

        predicted = pop_mean + combined_deviation * pop_std

        actual = protein_dept_profile(uid, ptt, tok_to_idx, dept_token_disruption)

        cos = cosine_sim(predicted, actual)
        mean_cos = cosine_sim(pop_mean, actual)
        cosines.append(cos)
        mean_only_cosines.append(mean_cos)

        if np.std(predicted) > 0 and np.std(actual) > 0:
            r, p = stats.pearsonr(predicted, actual)
            correlations.append(r)
        else:
            correlations.append(0)

        n_prims_used.append(len(matched_prims))

        top_pred = VALID_DEPARTMENTS[np.argmax(predicted)]
        top_actual = VALID_DEPARTMENTS[np.argmax(actual)]

        protein_details.append({
            "uid": uid,
            "n_prims": len(matched_prims),
            "cosine": round(cos, 4),
            "mean_only_cosine": round(mean_cos, 4),
            "pearson_r": round(correlations[-1], 4),
            "top_pred": top_pred,
            "top_actual": top_actual,
            "top_match": top_pred == top_actual,
        })

        tested += 1
        if tested % 2000 == 0:
            elapsed = time.time() - t0
            cos_arr = np.array(cosines)
            mo_arr = np.array(mean_only_cosines)
            r_arr = np.array(correlations)
            top_match = np.mean([d["top_match"] for d in protein_details])
            mo_top = np.mean([VALID_DEPARTMENTS[np.argmax(pop_mean)] == d["top_actual"]
                              for d in protein_details])
            print(f"  [{tested:5d}] compiler={cos_arr.mean():.3f} mean_only={mo_arr.mean():.3f} "
                  f"top1={top_match:.1%}(vs mean_top1={mo_top:.1%}) "
                  f"r={r_arr.mean():.3f} ({elapsed:.0f}s)")
            sys.stdout.flush()

    cos_arr = np.array(cosines)
    mo_arr = np.array(mean_only_cosines)
    r_arr = np.array(correlations)
    top_matches = np.mean([d["top_match"] for d in protein_details])
    mean_only_top1 = VALID_DEPARTMENTS[np.argmax(pop_mean)]
    mean_top1_rate = np.mean([d["top_actual"] == mean_only_top1 for d in protein_details])

    print(f"\n  Tested: {tested} proteins ({skipped_no_prim} skipped, no matching primitives)")
    print(f"  Mean-only baseline cosine: {mo_arr.mean():.4f}")
    print(f"  Mean-only top-1 rate (always predict '{mean_only_top1}'): {mean_top1_rate:.1%}")
    return cos_arr, r_arr, mo_arr, top_matches, mean_top1_rate, n_prims_used, protein_details


def run_null_model(test_uids, protein_dept_seqs, ptt, tok_to_idx,
                     dept_token_disruption, n_to_test=5000):
    print("[  ] Running null model (random profile predictions)...")
    t0 = time.time()

    all_profiles = []
    test_list = [uid for uid in test_uids if uid in protein_dept_seqs][:n_to_test]
    for uid in test_list:
        p = protein_dept_profile(uid, ptt, tok_to_idx, dept_token_disruption)
        all_profiles.append(p)

    all_profiles = np.array(all_profiles)

    null_cosines = []
    np.random.seed(42)
    for i in range(len(all_profiles)):
        j = np.random.randint(len(all_profiles))
        while j == i:
            j = np.random.randint(len(all_profiles))
        cos = cosine_sim(all_profiles[i], all_profiles[j])
        null_cosines.append(cos)

    shuffled_cosines = []
    for i in range(len(all_profiles)):
        shuffled = all_profiles[i].copy()
        np.random.shuffle(shuffled)
        cos = cosine_sim(shuffled, all_profiles[i])
        shuffled_cosines.append(cos)

    null_arr = np.array(null_cosines)
    shuf_arr = np.array(shuffled_cosines)
    print(f"  Random-pair cosine: {null_arr.mean():.4f} ± {null_arr.std():.4f}")
    print(f"  Shuffled-self cosine: {shuf_arr.mean():.4f} ± {shuf_arr.std():.4f}")
    print(f"  ({time.time()-t0:.1f}s)")

    return {
        "random_pair_cosine_mean": round(float(null_arr.mean()), 4),
        "random_pair_cosine_std": round(float(null_arr.std()), 4),
        "shuffled_self_cosine_mean": round(float(shuf_arr.mean()), 4),
        "shuffled_self_cosine_std": round(float(shuf_arr.std()), 4),
    }


def main():
    state = load_state()
    vocab_dept = build_token_dept_map()
    protein_dept_seqs, protein_dept_lists = build_protein_data(state, vocab_dept)
    tok_to_idx, uid_to_idx, dept_token_disruption = build_disruption_infrastructure(state)
    train_uids, test_uids = split_proteins(protein_dept_seqs)
    ptt = state["ptt"]

    with open(PRIMITIVES_PATH) as f:
        primitives = list(csv.DictReader(f))

    pop_mean, pop_std = compute_population_mean(
        train_uids, protein_dept_seqs, ptt, tok_to_idx, dept_token_disruption
    )

    signatures = learn_primitive_signatures(
        primitives, train_uids, protein_dept_seqs,
        ptt, tok_to_idx, dept_token_disruption, pop_mean, pop_std
    )

    (cos_arr, r_arr, mo_arr, top_matches, mean_top1_rate,
     n_prims_used, protein_details) = predict_and_evaluate(
        signatures, test_uids, protein_dept_seqs, protein_dept_lists,
        ptt, tok_to_idx, dept_token_disruption, pop_mean, pop_std
    )

    null_results = run_null_model(
        test_uids, protein_dept_seqs, ptt, tok_to_idx, dept_token_disruption
    )

    print(f"\n{'='*72}")
    print(f"  THE COMPILER TEST — DEVIATION-BASED PREDICTION")
    print(f"{'='*72}")
    print(f"  Training: {len(train_uids)} proteins (random 50%)")
    print(f"  Testing:  {len(test_uids)} proteins (held-out 50%)")
    print(f"  Primitives with signatures: {len(signatures)}")
    print(f"  Test proteins predicted:    {len(cos_arr)}")
    print(f"")
    print(f"  === COMPILER vs MEAN-ONLY BASELINE ===")
    print(f"  Compiler cosine:            {cos_arr.mean():.4f} ± {cos_arr.std():.4f}")
    print(f"  Mean-only cosine:           {mo_arr.mean():.4f} ± {mo_arr.std():.4f}")
    print(f"  LIFT over mean-only:        {cos_arr.mean() - mo_arr.mean():+.4f}")
    print(f"  Compiler beats mean-only:   {np.mean(cos_arr > mo_arr):.1%} of proteins")
    print(f"")
    print(f"  === TOP-1 DEPARTMENT PREDICTION ===")
    print(f"  Compiler top-1 match:       {top_matches:.1%}")
    print(f"  Mean-only top-1 match:      {mean_top1_rate:.1%}")
    print(f"  Compiler lift:              {top_matches - mean_top1_rate:+.1%}")
    print(f"  Chance (1/22):              {1/22:.1%}")
    print(f"")
    print(f"  === CORRELATION ===")
    print(f"  Median cosine:              {np.median(cos_arr):.4f}")
    print(f"  Mean Pearson r:             {r_arr.mean():.4f}")
    print(f"  Cosine > 0.9:               {np.mean(cos_arr > 0.9):.1%}")
    print(f"  Cosine > 0.8:               {np.mean(cos_arr > 0.8):.1%}")
    print(f"")
    print(f"  === NULL BASELINES ===")
    print(f"  Random-pair cosine:         {null_results['random_pair_cosine_mean']:.4f}")
    print(f"  Shuffled-self cosine:       {null_results['shuffled_self_cosine_mean']:.4f}")
    print(f"")
    print(f"  === PRIMITIVES PER PROTEIN ===")
    prims_arr = np.array(n_prims_used)
    print(f"  Mean: {prims_arr.mean():.1f}, Median: {np.median(prims_arr):.0f}, "
          f"Max: {prims_arr.max()}")
    print(f"{'='*72}")

    compiler_lift = cos_arr.mean() - mo_arr.mean()
    compiler_wins = np.mean(cos_arr > mo_arr)
    top1_lift = top_matches - mean_top1_rate

    if compiler_lift > 0.01 and compiler_wins > 0.5:
        print(f"\n  ★ COMPILATION ADDS SIGNAL: Deviation-based primitive composition")
        print(f"    beats mean-only baseline by {compiler_lift:+.4f} cosine")
        print(f"    ({compiler_wins:.1%} of proteins improved)")
        if top1_lift > 0.05:
            print(f"    Top-1 dept prediction: {top_matches:.1%} vs {mean_top1_rate:.1%} (mean-only)")
    elif compiler_lift > 0:
        print(f"\n  Marginal lift: compiler beats mean-only by {compiler_lift:+.4f}")
    else:
        print(f"\n  No lift over mean-only baseline ({compiler_lift:+.4f})")

    output = {
        "split_seed": SPLIT_SEED,
        "split_method": "random_50_50_deviation",
        "n_train_proteins": len(train_uids),
        "n_test_proteins": len(test_uids),
        "n_primitives_with_signatures": len(signatures),
        "n_proteins_predicted": len(cos_arr),
        "compiler_cosine_mean": round(float(cos_arr.mean()), 4),
        "compiler_cosine_std": round(float(cos_arr.std()), 4),
        "compiler_cosine_median": round(float(np.median(cos_arr)), 4),
        "mean_only_cosine_mean": round(float(mo_arr.mean()), 4),
        "mean_only_cosine_std": round(float(mo_arr.std()), 4),
        "compiler_lift_over_mean": round(float(compiler_lift), 4),
        "frac_compiler_beats_mean": round(float(compiler_wins), 4),
        "pearson_r_mean": round(float(r_arr.mean()), 4),
        "compiler_top1_match": round(float(top_matches), 4),
        "mean_only_top1_match": round(float(mean_top1_rate), 4),
        "top1_lift": round(float(top1_lift), 4),
        "null_baselines": null_results,
        "prims_per_protein_mean": round(float(prims_arr.mean()), 2),
        "prims_per_protein_median": round(float(np.median(prims_arr)), 1),
        "top_predictions": sorted(protein_details, key=lambda x: -x["cosine"])[:50],
        "bottom_predictions": sorted(protein_details, key=lambda x: x["cosine"])[:50],
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
