#!/usr/bin/env python3
"""
Module 8 Full Dispatch Graph Shuffle Test (144 drugs, 100 permutations)
========================================================================
Final causal test for Paper 2. Tests whether Module 8's side-effect
predictions depend on the specific token-sharing graph topology.

Uses scipy sparse matrix BFS for ~100x speedup over pure-Python cascade.
Checkpoints results every 25 permutations.

Usage: python3 -u module8_full_shuffle.py
"""

import csv, json, math, os, pickle, random, re, sys, time
from collections import defaultdict

import numpy as np
from scipy import sparse, stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
STATE_PATH = "/tmp/module8_full_state.pkl"
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "module8_full_shuffle_results.json")

N_PERMUTATIONS = 100
CHECKPOINT_EVERY = 25
HOP_WEIGHTS = {1: 1.0, 2: 0.5, 3: 0.25, 4: 0.125, 5: 0.0625, 6: 0.03125}
MIN_SHARED = 2
MAX_PER_HOP = 150
MAX_HOPS = 6

GENE_REMAP = {"BCR-ABL1": "ABL1", "DNA": "ATM"}
DRUG_NAME_MAP = {
    "5-Fluorouracil": "Fluorouracil",
    "Valproic_Acid": "Valproate",
    "Propoxyphene": "Dextropropoxyphene",
}

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


def load_ground_truth():
    gt_path = os.path.join(PROJECT_ROOT, "pull_ground_truth_145.py")
    with open(gt_path) as f:
        src = f.read()
    start = src.find("CURATED_GROUND_TRUTH")
    block_start = src.find("{", start)
    depth = 0
    block_end = block_start
    for i in range(block_start, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                block_end = i + 1
                break
    gt_str = src[block_start:block_end]
    gt_str = gt_str.replace("'", '"')
    gt_str = re.sub(r",\s*}", "}", gt_str)
    gt_str = re.sub(r",\s*]", "]", gt_str)
    try:
        gt = json.loads(gt_str)
    except json.JSONDecodeError:
        import ast
        gt = ast.literal_eval(src[block_start:block_end])
    return gt


def load_drug_panel():
    csv_path = os.path.join(PROJECT_ROOT,
                            "attached_assets/drug_validation_panel_150_1773555284435.csv")
    drugs = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gene = row["Primary_Target_Gene"].strip()
            gene = GENE_REMAP.get(gene, gene)
            drugs.append({
                "drug": row["Drug_Name"].strip(),
                "gene": gene,
                "tier": row["Tier"].strip(),
                "mechanism": row["Mechanism_Class"].strip(),
            })
    return drugs


def load_state_from_db():
    """Load the full bipartite graph and all caches from the database."""
    import psycopg2
    from psycopg2.extras import RealDictCursor

    print("  Loading bipartite graph from database...")
    db_url = os.environ.get("BETA_DATABASE_URL", os.environ.get("DATABASE_URL"))
    conn = psycopg2.connect(db_url)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    t0 = time.time()
    cur.execute("SELECT token_hex, uniprot_id FROM protein_tokens_v2")
    rows = cur.fetchall()
    print(f"    Loaded {len(rows)} edges in {time.time()-t0:.1f}s")

    ttp = defaultdict(list)
    ptt = defaultdict(list)
    for r in rows:
        ttp[r["token_hex"]].append(r["uniprot_id"])
        ptt[r["uniprot_id"]].append(r["token_hex"])

    n_tokens = len(ttp)
    n_proteins = len(ptt)
    n_edges = len(rows)
    print(f"    Graph: {n_tokens} tokens, {n_proteins} proteins, {n_edges} edges")

    print("  Loading gene names...")
    t0 = time.time()
    cur.execute("SELECT uniprot_id, gene_name FROM protein_catalog")
    gene_cache = {}
    for r in cur.fetchall():
        gene_cache[r["uniprot_id"]] = r["gene_name"]
    print(f"    {len(gene_cache)} entries in {time.time()-t0:.1f}s")

    print("  Loading departments...")
    t0 = time.time()
    cur.execute("SELECT gene_name, all_departments FROM gene_department_map")
    dept_cache = {}
    for r in cur.fetchall():
        depts = r["all_departments"] or []
        dept_cache[r["gene_name"]] = depts if depts else ["Unknown"]
    print(f"    {len(dept_cache)} entries in {time.time()-t0:.1f}s")

    print("  Loading phenotypes...")
    t0 = time.time()
    cur.execute("SELECT gene_name, phenotype_category FROM gene_phenotype_map")
    pheno_cache = defaultdict(list)
    for r in cur.fetchall():
        pheno_cache[r["gene_name"]].append(r["phenotype_category"])
    print(f"    {len(pheno_cache)} entries in {time.time()-t0:.1f}s")

    print("  Loading dept priors...")
    cur.execute("SELECT department, phenotype_category, prior_weight FROM dept_phenotype_priors")
    dept_priors = defaultdict(list)
    for r in cur.fetchall():
        dept_priors[r["department"]].append(
            (r["phenotype_category"], float(r["prior_weight"]))
        )

    print("  Resolving drug targets...")
    cur.execute("""
        SELECT gene_name, uniprot_id, token_count
        FROM protein_catalog
        WHERE gene_name IS NOT NULL
        ORDER BY token_count DESC
    """)
    gene_to_uid = {}
    for r in cur.fetchall():
        gn = r["gene_name"]
        if gn not in gene_to_uid:
            gene_to_uid[gn] = r["uniprot_id"]

    conn.close()

    state = {
        "ttp": dict(ttp),
        "ptt": dict(ptt),
        "gene_cache": gene_cache,
        "dept_cache": dict(dept_cache),
        "pheno_cache": dict(pheno_cache),
        "dept_priors": dict(dept_priors),
        "gene_to_uid": gene_to_uid,
        "n_tokens": n_tokens,
        "n_proteins": n_proteins,
        "n_edges": n_edges,
    }

    with open(STATE_PATH, "wb") as f:
        pickle.dump(state, f)
    print(f"  State saved to {STATE_PATH} ({os.path.getsize(STATE_PATH)/1e6:.0f} MB)")
    return state


def build_sparse_matrix(ttp, ptt):
    """Build scipy sparse protein×token matrix and index maps."""
    all_tokens = sorted(ttp.keys())
    all_proteins = sorted(ptt.keys())

    tok_to_idx = {t: i for i, t in enumerate(all_tokens)}
    uid_to_idx = {u: i for i, u in enumerate(all_proteins)}
    idx_to_uid = {i: u for u, i in uid_to_idx.items()}
    idx_to_tok = {i: t for t, i in tok_to_idx.items()}

    rows, cols = [], []
    for tok, uids in ttp.items():
        ti = tok_to_idx[tok]
        for uid in uids:
            pi = uid_to_idx[uid]
            rows.append(pi)
            cols.append(ti)

    n_p = len(all_proteins)
    n_t = len(all_tokens)
    P = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_p, n_t),
    )

    return P, tok_to_idx, uid_to_idx, idx_to_uid, idx_to_tok


def sparse_bfs_cascade(target_uid_idx, target_tok_idxs, P, P_T, uid_to_idx, idx_to_uid, gene_cache, n_tokens):
    """Run BFS cascade using sparse matrix operations."""
    in_cascade = set([target_uid_idx])
    frontier_vec = np.zeros(n_tokens, dtype=np.float32)
    frontier_vec[target_tok_idxs] = 1.0

    hop_protein_idxs = {}

    for hop in range(1, MAX_HOPS + 1):
        shared_counts = np.asarray(P.dot(frontier_vec)).flatten()

        for pi in in_cascade:
            shared_counts[pi] = 0

        thresh = MIN_SHARED
        if hop == 1 and len(target_tok_idxs) < MIN_SHARED:
            thresh = max(1, len(target_tok_idxs))

        candidates = np.where(shared_counts >= thresh)[0]
        if len(candidates) == 0:
            break

        candidate_scores = shared_counts[candidates]
        sorted_order = np.argsort(-candidate_scores)

        seen_genes = set()
        selected = []
        for ci in sorted_order:
            pi = candidates[ci]
            uid = idx_to_uid[pi]
            gn = gene_cache.get(uid)
            if gn and gn in seen_genes:
                continue
            if gn:
                seen_genes.add(gn)
            selected.append(pi)
            in_cascade.add(pi)
            if len(selected) >= MAX_PER_HOP:
                break

        if not selected:
            break

        hop_protein_idxs[hop] = selected

        sel_vec = np.zeros(P.shape[0], dtype=np.float32)
        sel_vec[selected] = 1.0
        frontier_vec = np.asarray(P_T.dot(sel_vec)).flatten()
        frontier_vec = (frontier_vec > 0).astype(np.float32)

    return hop_protein_idxs


def score_cascade_fast(hop_protein_idxs, target_gene, idx_to_uid, gene_cache, dept_cache, pheno_cache, dept_priors):
    """Score cascade proteins to produce ranked side-effect predictions."""
    phenotype_scores = defaultdict(float)

    for hop, prot_idxs in hop_protein_idxs.items():
        hw = HOP_WEIGHTS.get(hop, 0.5 ** (hop - 1))
        dept_genes = defaultdict(set)
        for pi in prot_idxs:
            uid = idx_to_uid[pi]
            gn = gene_cache.get(uid)
            if gn and gn != target_gene:
                for dept in dept_cache.get(gn, ["Unknown"]):
                    dept_genes[dept].add(gn)

        for dept, genes_in_dept in dept_genes.items():
            for gn in genes_in_dept:
                for cat in pheno_cache.get(gn, []):
                    phenotype_scores[cat] += hw
            for cat, pw in dept_priors.get(dept, []):
                phenotype_scores[cat] += pw * hw * len(genes_in_dept) * 0.1

    ranked = sorted(
        [(c, s) for c, s in phenotype_scores.items() if s >= 0.1],
        key=lambda x: -x[1],
    )
    return [c for c, s in ranked]


def degree_preserving_shuffle(ttp, ptt, n_swaps):
    """Degree-preserving bipartite edge swaps. Returns new ttp, ptt."""
    ttp_s = {t: list(ps) for t, ps in ttp.items()}
    ptt_s = {p: set(ts) for p, ts in ptt.items()}

    edges = []
    for t, prots in ttp_s.items():
        for p in prots:
            edges.append((p, t))
    n = len(edges)

    ttp_set = {t: set(ps) for t, ps in ttp_s.items()}

    successful = 0
    attempts = 0
    max_attempts = n_swaps * 10

    while successful < n_swaps and attempts < max_attempts:
        attempts += 1
        i1 = random.randint(0, n - 1)
        i2 = random.randint(0, n - 1)
        if i1 == i2:
            continue
        p1, t1 = edges[i1]
        p2, t2 = edges[i2]
        if t1 == t2 or p1 == p2:
            continue
        if t2 in ptt_s[p1] or t1 in ptt_s[p2]:
            continue

        ttp_set[t1].discard(p1)
        ttp_set[t1].add(p2)
        ttp_set[t2].discard(p2)
        ttp_set[t2].add(p1)
        ptt_s[p1].discard(t1)
        ptt_s[p1].add(t2)
        ptt_s[p2].discard(t2)
        ptt_s[p2].add(t1)
        edges[i1] = (p1, t2)
        edges[i2] = (p2, t1)
        successful += 1

    ttp_out = {t: list(ps) for t, ps in ttp_set.items()}
    ptt_out = {p: list(ts) for p, ts in ptt_s.items()}
    return ttp_out, ptt_out, successful


def load_checkpoint():
    """Load the latest checkpoint if it exists."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoints = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("ckpt_") and f.endswith(".pkl")]
    )
    if not checkpoints:
        return None
    latest = checkpoints[-1]
    path = os.path.join(CHECKPOINT_DIR, latest)
    print(f"  Resuming from checkpoint: {latest}")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_checkpoint(perm_idx, shuffled_r10, drug_results):
    """Save checkpoint."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"ckpt_{perm_idx:04d}.pkl")
    data = {
        "perm_idx": perm_idx,
        "shuffled_r10": shuffled_r10,
        "drug_results": drug_results,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  Checkpoint saved: {path}")


def main():
    t_global = time.time()
    print("=" * 72)
    print("  MODULE 8 FULL DISPATCH GRAPH SHUFFLE TEST")
    print("  144 drugs × 100 permutations")
    print("=" * 72)

    # ==================================================================
    # STEP 1: Load or build state
    # ==================================================================
    print("\n[STEP 1] Loading state...")
    if os.path.exists(STATE_PATH):
        print(f"  Loading cached state from {STATE_PATH}")
        with open(STATE_PATH, "rb") as f:
            state = pickle.load(f)
        print(f"  Loaded: {state['n_tokens']} tokens, {state['n_proteins']} proteins, {state['n_edges']} edges")
    else:
        state = load_state_from_db()

    ttp = state["ttp"]
    ptt = state["ptt"]
    gene_cache = state["gene_cache"]
    dept_cache = state["dept_cache"]
    pheno_cache = state["pheno_cache"]
    dept_priors = state["dept_priors"]
    gene_to_uid = state["gene_to_uid"]

    # ==================================================================
    # STEP 2: Build sparse matrix
    # ==================================================================
    print("\n[STEP 2] Building sparse matrix...")
    t0 = time.time()
    P, tok_to_idx, uid_to_idx, idx_to_uid, idx_to_tok = build_sparse_matrix(ttp, ptt)
    P_T = P.T.tocsr()
    n_tokens = P.shape[1]
    n_proteins = P.shape[0]
    print(f"  Sparse matrix: {n_proteins}×{n_tokens}, {P.nnz} non-zeros, built in {time.time()-t0:.1f}s")

    # ==================================================================
    # STEP 3: Load drug panel + ground truth
    # ==================================================================
    print("\n[STEP 3] Loading drugs...")
    panel_drugs = load_drug_panel()
    ground_truth = load_ground_truth()

    drugs = []
    for d in panel_drugs:
        drug_name = d["drug"]
        gene = d["gene"]

        gt_name = drug_name
        if gt_name not in ground_truth:
            gt_name = DRUG_NAME_MAP.get(drug_name, drug_name)
        if gt_name not in ground_truth:
            gt_name_nound = drug_name.replace("_", " ")
            for gk in ground_truth:
                if gk.lower() == gt_name_nound.lower():
                    gt_name = gk
                    break

        gt_effects = ground_truth.get(gt_name, {}).get("known_effects", [])

        uid = gene_to_uid.get(gene)
        if not uid or uid not in uid_to_idx:
            continue

        target_idx = uid_to_idx[uid]
        target_tok_idxs = []
        for tok in ptt.get(uid, []):
            if tok in tok_to_idx:
                target_tok_idxs.append(tok_to_idx[tok])
        if not target_tok_idxs:
            continue

        drugs.append({
            "drug": drug_name,
            "gene": gene,
            "tier": d["tier"],
            "mechanism": d["mechanism"],
            "ground_truth": gt_effects,
            "target_uid": uid,
            "target_idx": target_idx,
            "target_tok_idxs": target_tok_idxs,
        })

    print(f"  Panel: {len(panel_drugs)} drugs")
    print(f"  Valid (gene in catalog + has tokens + has ground truth): {len(drugs)}")
    drugs_with_gt = [d for d in drugs if d["ground_truth"]]
    drugs_no_gt = [d for d in drugs if not d["ground_truth"]]
    print(f"  With ground truth effects: {len(drugs_with_gt)}")
    print(f"  Without ground truth (will run but cannot compute R@10): {len(drugs_no_gt)}")

    # ==================================================================
    # STEP 4: Run real cascades
    # ==================================================================
    print(f"\n[STEP 4] Running real cascades for {len(drugs)} drugs...")
    t0 = time.time()

    drug_results = {}
    for i, d in enumerate(drugs):
        hop_idxs = sparse_bfs_cascade(
            d["target_idx"], d["target_tok_idxs"],
            P, P_T, uid_to_idx, idx_to_uid, gene_cache, n_tokens,
        )
        predicted = score_cascade_fast(
            hop_idxs, d["gene"], idx_to_uid, gene_cache, dept_cache, pheno_cache, dept_priors,
        )
        cascade_size = sum(len(hp) for hp in hop_idxs.values())
        hop_sizes = {h: len(ps) for h, ps in sorted(hop_idxs.items())}

        if d["ground_truth"]:
            r10 = compute_recall_at_k(predicted, d["ground_truth"], k=10)
        else:
            r10 = None

        drug_results[d["drug"]] = {
            "gene": d["gene"],
            "tier": d["tier"],
            "mechanism": d["mechanism"],
            "cascade_size": cascade_size,
            "hop_sizes": hop_sizes,
            "real_top10": predicted[:10],
            "ground_truth": d["ground_truth"],
            "n_ground_truth": len(d["ground_truth"]),
            "real_r_at_10": round(r10, 4) if r10 is not None else None,
        }

        if (i + 1) % 20 == 0 or i == len(drugs) - 1:
            r10_str = f"{r10:.3f}" if r10 is not None else "N/A"
            print(f"  [{i+1}/{len(drugs)}] {d['drug']:25s} ({d['gene']:10s}): cascade={cascade_size}, R@10={r10_str}")

    real_elapsed = time.time() - t0
    evaluable = [d for d in drugs if d["ground_truth"]]
    mean_real = np.mean([drug_results[d["drug"]]["real_r_at_10"] for d in evaluable])
    print(f"\n  Real cascades complete in {real_elapsed:.1f}s")
    print(f"  Mean real R@10 ({len(evaluable)} evaluable drugs): {mean_real:.4f}")

    # ==================================================================
    # STEP 5: Shuffled permutations
    # ==================================================================
    ckpt = load_checkpoint()
    if ckpt:
        start_perm = ckpt["perm_idx"]
        shuffled_r10 = ckpt["shuffled_r10"]
        for drug_name, saved_data in ckpt["drug_results"].items():
            if drug_name in drug_results:
                drug_results[drug_name].update(saved_data)
        print(f"  Resuming from permutation {start_perm}")
    else:
        start_perm = 0
        shuffled_r10 = {d["drug"]: [] for d in evaluable}

    n_edges = state["n_edges"]
    swaps_per_perm = n_edges // 2

    print(f"\n[STEP 5] Running {N_PERMUTATIONS} shuffled permutations (starting from {start_perm})...")
    print(f"  {swaps_per_perm} edge swaps per permutation")

    for perm_idx in range(start_perm, N_PERMUTATIONS):
        random.seed(1000 + perm_idx)
        t_perm = time.time()

        ttp_shuf, ptt_shuf, n_swaps = degree_preserving_shuffle(ttp, ptt, swaps_per_perm)
        t_shuffle = time.time() - t_perm

        t_build = time.time()
        P_shuf, tok_idx_s, uid_idx_s, idx_uid_s, _ = build_sparse_matrix(ttp_shuf, ptt_shuf)
        P_T_shuf = P_shuf.T.tocsr()
        n_tok_shuf = P_shuf.shape[1]
        t_build_elapsed = time.time() - t_build

        t_cascade = time.time()
        perm_r10_sum = 0.0
        perm_r10_count = 0
        for d in drugs:
            if d["target_uid"] not in uid_idx_s:
                continue
            target_idx_s = uid_idx_s[d["target_uid"]]
            tok_idxs_s = []
            for tok in ptt_shuf.get(d["target_uid"], ptt.get(d["target_uid"], [])):
                if tok in tok_idx_s:
                    tok_idxs_s.append(tok_idx_s[tok])
            if not tok_idxs_s:
                continue

            hop_idxs = sparse_bfs_cascade(
                target_idx_s, tok_idxs_s,
                P_shuf, P_T_shuf, uid_idx_s, idx_uid_s, gene_cache, n_tok_shuf,
            )
            predicted = score_cascade_fast(
                hop_idxs, d["gene"], idx_uid_s, gene_cache, dept_cache, pheno_cache, dept_priors,
            )

            if d["ground_truth"]:
                r10 = compute_recall_at_k(predicted, d["ground_truth"], k=10)
                if d["drug"] in shuffled_r10:
                    shuffled_r10[d["drug"]].append(r10)
                perm_r10_sum += r10
                perm_r10_count += 1

        t_cascade_elapsed = time.time() - t_cascade
        total_perm = time.time() - t_perm

        if perm_r10_count > 0:
            mean_perm_r10 = perm_r10_sum / perm_r10_count
        else:
            mean_perm_r10 = 0

        print(f"  Perm {perm_idx+1:3d}/{N_PERMUTATIONS}: "
              f"shuffle={t_shuffle:.1f}s, build={t_build_elapsed:.1f}s, "
              f"cascade={t_cascade_elapsed:.1f}s, total={total_perm:.1f}s, "
              f"mean_R@10={mean_perm_r10:.3f} ({n_swaps} swaps)")

        if (perm_idx + 1) % CHECKPOINT_EVERY == 0:
            partial_drug_data = {}
            for d in evaluable:
                dn = d["drug"]
                if dn in shuffled_r10 and shuffled_r10[dn]:
                    partial_drug_data[dn] = {
                        "shuffled_r10_values": shuffled_r10[dn],
                    }
            save_checkpoint(perm_idx + 1, shuffled_r10, partial_drug_data)

    # ==================================================================
    # STEP 6: Compute statistics
    # ==================================================================
    print(f"\n[STEP 6] Computing statistics...")

    results = {"drugs": {}, "summary": {}, "method": {}}
    results["method"] = {
        "description": "Degree-preserving bipartite edge swap test on Module 8 token-sharing cascade graph",
        "n_permutations": N_PERMUTATIONS,
        "n_edges": n_edges,
        "n_tokens": state["n_tokens"],
        "n_proteins": state["n_proteins"],
        "swaps_per_permutation": swaps_per_perm,
        "cascade_params": {"MAX_HOPS": MAX_HOPS, "MAX_PER_HOP": MAX_PER_HOP, "MIN_SHARED": MIN_SHARED},
        "scoring": "hop-weighted phenotype + dept priors (firewall on target gene)",
        "metric": "R@10 (recall at k=10 with category synonym normalization)",
    }

    all_real_r10 = []
    all_shuf_means = []
    per_tier = defaultdict(lambda: {"real": [], "shuf": []})

    for d in evaluable:
        dn = d["drug"]
        dr = drug_results[dn]
        real_r10 = dr["real_r_at_10"]
        shuf_vals = shuffled_r10.get(dn, [])

        if not shuf_vals:
            continue

        mean_s = float(np.mean(shuf_vals))
        std_s = float(np.std(shuf_vals))
        delta = real_r10 - mean_s
        z = delta / std_s if std_s > 0 else 0
        p_emp = sum(1 for v in shuf_vals if v >= real_r10) / len(shuf_vals)

        results["drugs"][dn] = {
            **dr,
            "shuffled_mean_r10": round(mean_s, 4),
            "shuffled_std_r10": round(std_s, 4),
            "shuffled_r10_values": [round(v, 4) for v in shuf_vals],
            "delta_r10": round(delta, 4),
            "z_score": round(z, 2),
            "p_empirical": round(p_emp, 4),
            "n_permutations": len(shuf_vals),
        }

        all_real_r10.append(real_r10)
        all_shuf_means.append(mean_s)
        per_tier[d["tier"]]["real"].append(real_r10)
        per_tier[d["tier"]]["shuf"].append(mean_s)

    for dn in drug_results:
        if dn not in results["drugs"]:
            results["drugs"][dn] = drug_results[dn]

    n_eval = len(all_real_r10)
    if n_eval > 0:
        mean_real = float(np.mean(all_real_r10))
        mean_shuf = float(np.mean(all_shuf_means))
        delta_all = mean_real - mean_shuf

        deltas = [r - s for r, s in zip(all_real_r10, all_shuf_means)]

        if n_eval >= 2:
            t_stat, t_p = stats.ttest_rel(all_real_r10, all_shuf_means)
            try:
                w_stat, w_p = stats.wilcoxon(deltas)
            except ValueError:
                w_stat, w_p = 0, 1.0
        else:
            t_stat, t_p = 0, 1.0
            w_stat, w_p = 0, 1.0

        pos_delta = sum(1 for d in deltas if d > 0.05)
        neg_delta = sum(1 for d in deltas if d < -0.05)
        neutral = sum(1 for d in deltas if abs(d) <= 0.05)

        tier_summary = {}
        for tier in sorted(per_tier.keys()):
            td = per_tier[tier]
            tier_summary[tier] = {
                "n": len(td["real"]),
                "real_mean_r10": round(float(np.mean(td["real"])), 4),
                "shuf_mean_r10": round(float(np.mean(td["shuf"])), 4),
                "delta": round(float(np.mean(td["real"])) - float(np.mean(td["shuf"])), 4),
            }

        results["summary"] = {
            "n_drugs_evaluated": n_eval,
            "n_permutations": N_PERMUTATIONS,
            "real_mean_r10": round(mean_real, 4),
            "shuffled_mean_r10": round(mean_shuf, 4),
            "delta": round(delta_all, 4),
            "paired_t_statistic": round(float(t_stat), 4),
            "paired_t_p_value": round(float(t_p), 6),
            "wilcoxon_signed_rank_statistic": round(float(w_stat), 1),
            "wilcoxon_p_value": round(float(w_p), 6),
            "cohens_d": round(float(np.mean(deltas) / np.std(deltas)) if np.std(deltas) > 0 else 0, 4),
            "topology_positive_count": pos_delta,
            "topology_negative_count": neg_delta,
            "topology_neutral_count": neutral,
            "per_tier": tier_summary,
        }

        print(f"\n{'='*72}")
        print(f"  FINAL RESULTS ({n_eval} evaluable drugs, {N_PERMUTATIONS} permutations)")
        print(f"{'='*72}")
        print(f"  Real mean R@10:     {mean_real:.4f}")
        print(f"  Shuffled mean R@10: {mean_shuf:.4f}")
        print(f"  Δ (real - shuffled): {delta_all:+.4f}")
        print(f"  Paired t-test:      t={t_stat:.3f}, p={t_p:.6f}")
        print(f"  Wilcoxon:           W={w_stat:.0f}, p={w_p:.6f}")
        print(f"  Cohen's d:          {results['summary']['cohens_d']:.4f}")
        print(f"  Topology positive:  {pos_delta}")
        print(f"  Topology negative:  {neg_delta}")
        print(f"  Neutral:            {neutral}")
        for tier, ts in sorted(tier_summary.items()):
            print(f"  {tier} (n={ts['n']}): real={ts['real_mean_r10']:.4f}, shuf={ts['shuf_mean_r10']:.4f}, Δ={ts['delta']:+.4f}")

    elapsed = time.time() - t_global
    results["elapsed_seconds"] = round(elapsed, 1)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_PATH}")
    print(f"  Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
