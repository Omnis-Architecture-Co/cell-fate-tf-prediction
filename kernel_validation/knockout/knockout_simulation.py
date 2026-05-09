#!/usr/bin/env python3
"""
Full-Proteome Knockout Simulation — Sharded for Parallel Execution
===================================================================

Usage (parallel with 4 agents):
    python3 validation/knockout/knockout_simulation.py --shard 0 --total-shards 4
    python3 validation/knockout/knockout_simulation.py --shard 1 --total-shards 4
    python3 validation/knockout/knockout_simulation.py --shard 2 --total-shards 4
    python3 validation/knockout/knockout_simulation.py --shard 3 --total-shards 4

Single-process:
    python3 validation/knockout/knockout_simulation.py --shard 0 --total-shards 1

Resume after interruption:
    python3 validation/knockout/knockout_simulation.py --shard 0 --total-shards 4 --resume

Each shard saves results line-by-line to: results/shard_XXXX.jsonl
Progress updated every gene:            results/progress_XXXX.json

Null model: 10 degree-matched random token removals per gene.
Disease genes (50) get 100 null permutations for deeper statistics.
"""

import argparse
import csv
import json
import os
import pickle
import random
import sys
import time

import numpy as np
from collections import defaultdict
from scipy import sparse

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
STATE_PATH = "/tmp/module8_full_state.pkl"
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "gene_manifest.json")

NULL_PERMS_DEFAULT = 10
NULL_PERMS_DISEASE = 100

VALID_DEPARTMENTS = sorted([
    "Apoptosis", "Cell adhesion", "Cell cycle", "Chromatin", "Cytoskeleton",
    "DNA repair", "GTPase", "Immune", "Ion channel", "Kinase", "Methylation",
    "Nuc acid bind", "Phosphatase", "Protein folding", "Proteolysis",
    "RNA processing", "Signaling", "Structural", "Transcription",
    "Translation", "Transport", "Ubiquitin",
])


def find_dept_csv():
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "server", "data", "human", "gene_departments.csv"),
        "server/data/human/gene_departments.csv",
    ]
    for c in candidates:
        p = os.path.normpath(c)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("gene_departments.csv not found")


def load_gene_departments():
    gene_depts = {}
    with open(find_dept_csv()) as f:
        for row in csv.DictReader(f):
            gene_depts[row["gene"]] = row["department"]
    return gene_depts


def load_state():
    print("[1/3] Loading dispatch graph state...")
    t0 = time.time()
    with open(STATE_PATH, "rb") as f:
        state = pickle.load(f)
    ptt = state["ptt"]
    ttp = state["ttp"]
    gene_cache = state["gene_cache"]
    gene_to_uid = state["gene_to_uid"]
    print(f"  Loaded in {time.time()-t0:.1f}s: {len(ptt)} proteins, {len(ttp)} tokens")
    return ptt, ttp, gene_cache, gene_to_uid


def build_sparse_and_dept_index(ttp, ptt, gene_cache, gene_depts):
    print("[2/3] Building sparse matrix and department index...")
    t0 = time.time()
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

    dept_protein_idxs = defaultdict(list)
    valid_set = set(VALID_DEPARTMENTS)
    for uid, idx in uid_to_idx.items():
        gene = gene_cache.get(uid)
        if gene:
            dept = gene_depts.get(gene)
            if dept and dept in valid_set:
                dept_protein_idxs[dept].append(idx)

    dept_arrays = {}
    for dept, idxs in dept_protein_idxs.items():
        dept_arrays[dept] = np.array(idxs, dtype=np.int32)

    total_indexed = sum(len(a) for a in dept_arrays.values())

    print(f"  Matrix: {n_p}x{n_t}, {P.nnz} edges")
    print(f"  Department index: {len(dept_arrays)} depts, {total_indexed} proteins indexed")
    print(f"  Built in {time.time()-t0:.1f}s")

    return P, tok_to_idx, uid_to_idx, dept_arrays, n_p, n_t, total_indexed


def measure_disruption(P, token_idxs, dept_arrays, n_t, total_indexed):
    target_vec = np.zeros(n_t, dtype=np.float32)
    for ti in token_idxs:
        target_vec[ti] = 1.0

    affected = np.asarray(P.dot(target_vec)).flatten()

    total_link_loss_all = 0.0
    raw = {}
    for dept, idxs in dept_arrays.items():
        dept_affected = affected[idxs]
        n_affected = int(np.sum(dept_affected > 0))
        link_loss = float(np.sum(dept_affected))
        total_link_loss_all += link_loss
        raw[dept] = {
            "n_affected": n_affected,
            "n_total": len(idxs),
            "fraction": n_affected / len(idxs) if len(idxs) > 0 else 0.0,
            "link_loss": link_loss,
        }

    enrichment = {}
    for dept, idxs in dept_arrays.items():
        if total_link_loss_all > 0:
            observed_frac = raw[dept]["link_loss"] / total_link_loss_all
            expected_frac = len(idxs) / total_indexed
            enrichment[dept] = observed_frac / expected_frac if expected_frac > 0 else 0.0
        else:
            enrichment[dept] = 0.0

    return raw, enrichment, total_link_loss_all


def run_knockout(gene, gene_info, P, tok_to_idx, uid_to_idx,
                 dept_arrays, n_t, total_indexed, n_null_perms,
                 gene_depts, disease_gt):
    uid = gene_info["uid"]
    if uid not in uid_to_idx:
        return None

    pidx = uid_to_idx[uid]
    gene_token_idxs = list(P[pidx].indices)
    n_gene_tokens = len(gene_token_idxs)
    if n_gene_tokens == 0:
        return None

    real_raw, real_enr, real_total_loss = measure_disruption(
        P, gene_token_idxs, dept_arrays, n_t, total_indexed,
    )

    all_tok_range = list(range(n_t))
    null_raws = []
    null_enrs = []
    null_totals = []
    for _ in range(n_null_perms):
        rand_tokens = random.sample(all_tok_range, n_gene_tokens)
        nr, ne, nt = measure_disruption(P, rand_tokens, dept_arrays, n_t, total_indexed)
        null_raws.append(nr)
        null_enrs.append(ne)
        null_totals.append(nt)

    dept_stats = {}
    for dept in VALID_DEPARTMENTS:
        real_frac = real_raw.get(dept, {}).get("fraction", 0.0)
        real_loss = real_raw.get(dept, {}).get("link_loss", 0.0)
        real_aff = real_raw.get(dept, {}).get("n_affected", 0)
        real_tot = real_raw.get(dept, {}).get("n_total", 0)
        real_e = real_enr.get(dept, 0.0)

        null_fracs = [nr.get(dept, {}).get("fraction", 0.0) for nr in null_raws]
        null_enrich = [ne.get(dept, 0.0) for ne in null_enrs]

        frac_mean = float(np.mean(null_fracs))
        frac_std = float(np.std(null_fracs))
        frac_z = (real_frac - frac_mean) / frac_std if frac_std > 1e-10 else 0.0

        enr_mean = float(np.mean(null_enrich))
        enr_std = float(np.std(null_enrich))
        enr_z = (real_e - enr_mean) / enr_std if enr_std > 1e-10 else 0.0

        emp_p = (sum(1 for ne in null_enrich if ne >= real_e) + 1) / (len(null_enrich) + 1)

        dept_stats[dept] = {
            "real_fraction": round(real_frac, 6),
            "real_link_loss": round(real_loss, 2),
            "real_affected": real_aff,
            "real_total": real_tot,
            "null_frac_mean": round(frac_mean, 6),
            "null_frac_std": round(frac_std, 6),
            "disruption_z": round(frac_z, 4),
            "enrichment": round(real_e, 6),
            "null_enr_mean": round(enr_mean, 6),
            "null_enr_std": round(enr_std, 6),
            "enrichment_z": round(enr_z, 4),
            "empirical_p": round(emp_p, 6),
        }

    total_disruption = sum(d["real_link_loss"] for d in dept_stats.values())

    disrupt_ranking = sorted(dept_stats, key=lambda d: -dept_stats[d]["disruption_z"])
    enrich_ranking = sorted(dept_stats, key=lambda d: -dept_stats[d]["enrichment_z"])

    top_disrupt = disrupt_ranking[0]
    top_enrich = enrich_ranking[0]

    own_dept = gene_depts.get(gene, "Unknown")
    own_disrupt_z = dept_stats.get(own_dept, {}).get("disruption_z", 0.0)
    own_enrich_z = dept_stats.get(own_dept, {}).get("enrichment_z", 0.0)
    own_disrupt_rank = (disrupt_ranking.index(own_dept) + 1) if own_dept in disrupt_ranking else None
    own_enrich_rank = (enrich_ranking.index(own_dept) + 1) if own_dept in enrich_ranking else None

    null_total_mean = float(np.mean(null_totals))
    null_total_std = float(np.std(null_totals))
    total_z = (real_total_loss - null_total_mean) / null_total_std if null_total_std > 1e-10 else 0.0

    result = {
        "gene": gene,
        "uid": uid,
        "n_tokens": n_gene_tokens,
        "department": own_dept,
        "categories": gene_info["categories"],
        "chronos": gene_info.get("chronos"),
        "total_disruption": round(total_disruption, 2),
        "total_disruption_z": round(total_z, 4),
        "top_disrupted_dept": top_disrupt,
        "top_disrupted_z": dept_stats[top_disrupt]["disruption_z"],
        "top_enriched_dept": top_enrich,
        "top_enriched_z": dept_stats[top_enrich]["enrichment_z"],
        "own_dept_disrupt_z": round(own_disrupt_z, 4),
        "own_dept_disrupt_rank": own_disrupt_rank,
        "own_dept_enrich_z": round(own_enrich_z, 4),
        "own_dept_enrich_rank": own_enrich_rank,
        "n_null_perms": n_null_perms,
        "dept_stats": dept_stats,
    }

    if gene in disease_gt:
        gt_dept = disease_gt[gene]["department"]
        gt_disrupt_z = dept_stats.get(gt_dept, {}).get("disruption_z", 0.0)
        gt_enrich_z = dept_stats.get(gt_dept, {}).get("enrichment_z", 0.0)
        gt_disrupt_rank = (disrupt_ranking.index(gt_dept) + 1) if gt_dept in disrupt_ranking else None
        gt_enrich_rank = (enrich_ranking.index(gt_dept) + 1) if gt_dept in enrich_ranking else None
        result["disease_ground_truth"] = disease_gt[gene]
        result["disease_disrupt_concordant"] = (top_disrupt == gt_dept)
        result["disease_enrich_concordant"] = (top_enrich == gt_dept)
        result["disease_gt_disrupt_z"] = round(gt_disrupt_z, 4)
        result["disease_gt_enrich_z"] = round(gt_enrich_z, 4)
        result["disease_gt_disrupt_rank"] = gt_disrupt_rank
        result["disease_gt_enrich_rank"] = gt_enrich_rank

    return result


def main():
    parser = argparse.ArgumentParser(description="Knockout simulation (sharded)")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--total-shards", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    shard_tag = f"{args.shard:04d}"
    shard_file = os.path.join(RESULTS_DIR, f"shard_{shard_tag}.jsonl")
    progress_file = os.path.join(RESULTS_DIR, f"progress_{shard_tag}.json")

    with open(MANIFEST_PATH) as f:
        manifest_data = json.load(f)
    all_genes_dict = manifest_data["genes"]
    disease_gt = manifest_data.get("disease_ground_truth", {})

    all_genes = sorted(all_genes_dict.keys())
    shard_genes = [g for i, g in enumerate(all_genes) if i % args.total_shards == args.shard]

    print("=" * 72)
    print(f"  KNOCKOUT SIMULATION — Shard {args.shard}/{args.total_shards}")
    print(f"  Total genes: {len(all_genes)}, This shard: {len(shard_genes)}")
    n_disease = sum(1 for g in shard_genes if g in disease_gt)
    print(f"  Disease genes: {n_disease} (100 null perms)")
    print(f"  Other genes: {len(shard_genes) - n_disease} (10 null perms)")
    print("=" * 72)

    completed = set()
    if args.resume and os.path.exists(shard_file):
        with open(shard_file) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    completed.add(rec["gene"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"  Resuming: {len(completed)} genes already done")

    remaining = [g for g in shard_genes if g not in completed]
    print(f"  Remaining: {len(remaining)} genes")

    if not remaining:
        print("  Nothing to do — shard already complete!")
        return

    gene_depts = load_gene_departments()
    ptt, ttp, gene_cache, gene_to_uid = load_state()
    P, tok_to_idx, uid_to_idx, dept_arrays, n_p, n_t, total_indexed = \
        build_sparse_and_dept_index(ttp, ptt, gene_cache, gene_depts)

    print(f"[3/3] Running knockouts...")
    t_start = time.time()
    done_count = len(completed)
    total_shard = len(shard_genes)
    skipped = 0

    mode = "a" if args.resume and completed else "w"
    with open(shard_file, mode) as out_f:
        for i, gene in enumerate(remaining):
            gene_info = all_genes_dict[gene]
            is_disease = gene in disease_gt
            n_null = NULL_PERMS_DISEASE if is_disease else NULL_PERMS_DEFAULT

            t0 = time.time()
            result = run_knockout(
                gene, gene_info, P, tok_to_idx, uid_to_idx,
                dept_arrays, n_t, total_indexed, n_null,
                gene_depts, disease_gt,
            )
            elapsed = time.time() - t0

            if result is None:
                done_count += 1
                skipped += 1
                continue

            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            done_count += 1

            elapsed_total = time.time() - t_start
            rate = (i + 1) / elapsed_total if elapsed_total > 0 else 0
            eta = (len(remaining) - i - 1) / rate if rate > 0 else 0

            if done_count % 50 == 0 or is_disease or done_count == total_shard:
                tag = " [DISEASE]" if is_disease else ""
                print(
                    f"  [{done_count:5d}/{total_shard}] {gene:15s} "
                    f"tok={result['n_tokens']:3d} "
                    f"enr_top={result['top_enriched_dept']:15s} "
                    f"ez={result['top_enriched_z']:+6.2f} "
                    f"own_ez={result['own_dept_enrich_z']:+6.2f} "
                    f"total_z={result['total_disruption_z']:+6.2f} "
                    f"({elapsed:.2f}s) "
                    f"ETA={eta/60:.0f}m{tag}"
                )

            progress = {
                "shard": args.shard,
                "total_shards": args.total_shards,
                "done": done_count,
                "total": total_shard,
                "pct": round(done_count / total_shard * 100, 1),
                "elapsed_s": round(time.time() - t_start, 1),
                "rate_genes_per_s": round(rate, 2) if rate > 0 else 0,
                "eta_minutes": round(eta / 60, 1) if rate > 0 else 0,
                "skipped": skipped,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "status": "COMPLETE" if done_count >= total_shard else "RUNNING",
            }
            with open(progress_file, "w") as pf:
                json.dump(progress, pf, indent=2)

    elapsed_total = time.time() - t_start
    print(f"\n  Shard {args.shard} COMPLETE: {done_count} genes in {elapsed_total:.0f}s ({skipped} skipped)")
    print(f"  Results: {shard_file}")


if __name__ == "__main__":
    main()
