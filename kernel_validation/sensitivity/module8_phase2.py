#!/usr/bin/env python3
"""Phase 2: Run shuffled cascades using saved Phase 1 state."""
import json, math, os, time, pickle, random
from collections import defaultdict
import numpy as np

np.random.seed(99)
random.seed(99)

N_SHUFFLES = 7
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


def compute_recall_at_k(predicted, ground_truth, k=10):
    gt_set = set(normalize_category(c) for c in ground_truth)
    pred_list = [normalize_category(c) for c in predicted[:k]]
    if not gt_set:
        return 0.0
    return len(gt_set & set(pred_list)) / len(gt_set)


def degree_preserving_shuffle(ttp, ptt, n_swaps):
    ttp = {t: set(ps) for t, ps in ttp.items()}
    ptt = {p: set(ts) for p, ts in ptt.items()}

    edges = [(p, t) for t, prots in ttp.items() for p in prots]
    n_edges = len(edges)
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

        ttp[t1].discard(p1); ttp[t1].add(p2)
        ttp[t2].discard(p2); ttp[t2].add(p1)
        ptt[p1].discard(t1); ptt[p1].add(t2)
        ptt[p2].discard(t2); ptt[p2].add(t1)
        edges[i1] = (p1, t2)
        edges[i2] = (p2, t1)
        successful += 1

    return ttp, ptt, successful


def run_cascade(target_gene, target_uid, target_tokens, ttp, ptt, gene_cache):
    target_tokens = set(target_tokens)
    in_cascade = {target_uid}
    frontier = {target_uid: target_tokens}
    hop_proteins = {}

    for hop in range(1, MAX_HOPS + 1):
        ftoks = set()
        for ts in frontier.values():
            ftoks.update(ts)
        if not ftoks:
            break

        thresh = MIN_SHARED_TOKENS
        if hop == 1 and len(target_tokens) < MIN_SHARED_TOKENS:
            thresh = max(1, len(target_tokens))

        protein_matched = defaultdict(set)
        for tok in ftoks:
            for uid in ttp.get(tok, []):
                if uid not in in_cascade:
                    protein_matched[uid].add(tok)

        scored = []
        for uid, matched in protein_matched.items():
            n_matched = len(matched)
            if n_matched >= thresh:
                scored.append((uid, n_matched, gene_cache.get(uid)))

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
        hop_proteins[hop] = hp

        frontier = {}
        for uid in hp:
            toks = set(ptt.get(uid, []))
            if toks:
                frontier[uid] = toks

    return hop_proteins


def score_cascade(hop_proteins, target_gene, dept_cache, pheno_cache, dept_priors):
    all_genes = set()
    for prots in hop_proteins.values():
        all_genes.update(v for v in prots.values() if v)
    all_genes.discard(target_gene)

    if not all_genes:
        return []

    phenotype_scores = defaultdict(float)
    for hop, prots in hop_proteins.items():
        hw = HOP_WEIGHTS.get(hop, 0.5 ** (hop - 1))
        dept_genes = defaultdict(set)
        for uid, gn in prots.items():
            if gn and gn != target_gene:
                for dept in dept_cache.get(gn, ['Unknown']):
                    dept_genes[dept].add(gn)

        for dept, genes_in_dept in dept_genes.items():
            for gn in genes_in_dept:
                for cat in pheno_cache.get(gn, []):
                    phenotype_scores[cat] += hw
            for cat, pw in dept_priors.get(dept, []):
                phenotype_scores[cat] += pw * hw * len(genes_in_dept) * 0.1

    ranked = sorted([(c, s) for c, s in phenotype_scores.items() if s >= 0.1], key=lambda x: -x[1])
    return [c for c, s in ranked]


def main():
    t0 = time.time()
    print("Loading Phase 1 state...")
    with open("/tmp/module8_phase1_state.pkl", "rb") as f:
        state = pickle.load(f)

    drug_data = state["drug_data"]
    ttp = state["token_to_proteins"]
    ptt = state["protein_to_tokens"]
    gene_cache = state["gene_cache"]
    dept_cache = state["dept_cache"]
    pheno_cache = state["pheno_cache"]
    dept_priors = state["dept_priors"]
    n_edges = state["n_edges"]

    print(f"  {len(drug_data)} drugs, {len(ttp)} tokens, {len(ptt)} proteins, {n_edges} edges")
    print(f"  Loaded in {time.time() - t0:.1f}s\n")

    shuffled_r10 = {drug: [] for drug in drug_data}

    for shuf_idx in range(N_SHUFFLES):
        t_shuf = time.time()
        ttp_shuf, ptt_shuf, n_swaps = degree_preserving_shuffle(ttp, ptt, n_swaps=n_edges // 2)
        print(f"Shuffle {shuf_idx+1}/{N_SHUFFLES} ({n_swaps} swaps, {time.time()-t_shuf:.1f}s)")

        for drug, dd in drug_data.items():
            hop_proteins = run_cascade(
                dd["gene"], dd["target_uniprot"], dd["target_tokens"],
                ttp_shuf, ptt_shuf, gene_cache
            )
            predicted = score_cascade(hop_proteins, dd["gene"], dept_cache, pheno_cache, dept_priors)
            r10 = compute_recall_at_k(predicted, dd["ground_truth"], k=10)
            shuffled_r10[drug].append(r10)
            cascade_sz = sum(len(hp) for hp in hop_proteins.values())
            print(f"  {drug:20s}: R@10={r10:.3f} (cascade={cascade_sz})")

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    results = {"drugs": {}, "summary": {}}
    for drug, dd in drug_data.items():
        real_r10 = dd["real_r_at_10"]
        shuf_vals = shuffled_r10[drug]
        mean_shuf = np.mean(shuf_vals)
        std_shuf = np.std(shuf_vals)
        delta = real_r10 - mean_shuf
        z = delta / std_shuf if std_shuf > 0 else 0
        p = sum(1 for s in shuf_vals if s >= real_r10) / len(shuf_vals) if shuf_vals else 1.0

        print(f"\n{drug} ({dd['expected']}):")
        print(f"  Real R@10:     {real_r10:.3f}")
        print(f"  Shuffled mean: {mean_shuf:.3f} ± {std_shuf:.3f}")
        print(f"  Δ:             {delta:+.3f}")
        print(f"  Z:             {z:.2f}")

        results["drugs"][drug] = {
            "gene": dd["gene"],
            "expected": dd["expected"],
            "cascade_size": dd["cascade_size"],
            "hop_sizes": dd["hop_sizes"],
            "real_r_at_10": round(real_r10, 3),
            "real_top10": dd["real_top10"],
            "ground_truth": dd["ground_truth"],
            "n_ground_truth": len(dd["ground_truth"]),
            "shuffled_mean_r10": round(float(mean_shuf), 3),
            "shuffled_std_r10": round(float(std_shuf), 3),
            "shuffled_r10_values": [round(v, 3) for v in shuf_vals],
            "delta_r10": round(float(delta), 3),
            "z_score": round(float(z), 2),
            "p_value": round(float(p), 4),
        }

    high_drugs = [d for d in drug_data if drug_data[d].get("expected") == "high"]
    low_drugs = [d for d in drug_data if drug_data[d].get("expected") == "low"]

    high_real = np.mean([drug_data[d]["real_r_at_10"] for d in high_drugs]) if high_drugs else 0
    high_shuf = np.mean([np.mean(shuffled_r10[d]) for d in high_drugs]) if high_drugs else 0
    low_real = np.mean([drug_data[d]["real_r_at_10"] for d in low_drugs]) if low_drugs else 0
    low_shuf = np.mean([np.mean(shuffled_r10[d]) for d in low_drugs]) if low_drugs else 0
    all_real = np.mean([drug_data[d]["real_r_at_10"] for d in drug_data])
    all_shuf = np.mean([np.mean(shuffled_r10[d]) for d in drug_data])

    print(f"\n{'='*60}")
    print(f"High-expected (n={len(high_drugs)}): real={high_real:.3f}, shuf={high_shuf:.3f}, Δ={high_real-high_shuf:+.3f}")
    print(f"Low-expected  (n={len(low_drugs)}):  real={low_real:.3f}, shuf={low_shuf:.3f}, Δ={low_real-low_shuf:+.3f}")
    print(f"All           (n={len(drug_data)}):  real={all_real:.3f}, shuf={all_shuf:.3f}, Δ={all_real-all_shuf:+.3f}")

    topology_dep = sum(1 for d in drug_data if results["drugs"][d]["delta_r10"] > 0.05)
    topology_indep = sum(1 for d in drug_data if abs(results["drugs"][d]["delta_r10"]) <= 0.05)

    if all_real - all_shuf < 0.02:
        verdict = "TOPOLOGY INDEPENDENT — signal primarily from static scoring tables"
    elif all_real - all_shuf > 0.10:
        verdict = "TOPOLOGY DEPENDENT — cascade wiring drives predictions"
    else:
        verdict = "MIXED — partial topology dependence"
    print(f"\nVERDICT: {verdict}")

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
        "topology_dependent_count": topology_dep,
        "topology_independent_count": topology_indep,
        "verdict": verdict,
    }

    elapsed = time.time() - t0
    results["elapsed_seconds"] = round(elapsed, 1)

    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "module8_graph_shuffle_results.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")
    print(f"Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
