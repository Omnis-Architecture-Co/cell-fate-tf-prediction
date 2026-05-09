#!/usr/bin/env python3
"""
VAL-DICT-001 v3: Beta Database Validation of VALDICT001 (Extended Vocabulary)
==============================================================================
Uses beta database tables directly:
  - valdict_extended (55,641 words) instead of vocabulary.csv (1,932)
  - gene_department_map (69,280 rows) instead of gene_departments.csv (17,322)
  - protein_tokens_v2 (1.85M tokens, 93,465 proteins)
  - canonical_gene_uniprot (20,581 canonical isoforms)

Multi-label scoring, 10 seeds, threshold-based reporting.
Side-by-side comparison with v2 CSV results.
"""

import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(WORKSPACE, "validation")

SEEDS = list(range(42, 52))
WORD_THRESHOLDS = [2, 5, 10, 20, 50]
WORD_COUNT_BINS = [(2, 3), (4, 10), (11, 50), (51, 999999)]
BIN_LABELS = ["2-3", "4-10", "11-50", "50+"]

BETA_DB_URL = os.environ.get("BETA_DATABASE_URL", "")


def query_db(sql: str, params: list = None) -> list:
    import psycopg2
    conn = psycopg2.connect(BETA_DB_URL)
    cur = conn.cursor()
    cur.execute(sql, params or [])
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    cur.close()
    conn.close()
    return rows


def load_vocab_from_db() -> Dict[str, dict]:
    print("  Loading valdict_extended...")
    rows = query_db("""
        SELECT token_hex, normalized_hex, primary_function, n_proteins, total_proteins, confidence
        FROM valdict_extended
        WHERE primary_function IS NOT NULL AND primary_function != 'Unclassified'
    """)
    vocab = {}
    for r in rows:
        hex_norm = r["token_hex"].strip().lower()
        vocab[hex_norm] = {
            "hex_norm": hex_norm,
            "function": r["primary_function"],
            "n_proteins": int(r["n_proteins"]),
            "total_proteins": float(r["total_proteins"]),
            "confidence": float(r["confidence"]),
            "enrichment": float(r["confidence"]) * 100,
        }
    print(f"    Loaded {len(vocab)} classified words (from {len(rows)} rows)")
    return vocab


def load_tokens_from_db() -> Dict[str, List[str]]:
    print("  Loading protein_tokens_v2...")
    rows = query_db("SELECT uniprot_id, token_hex FROM protein_tokens_v2 ORDER BY uniprot_id, rank")
    protein_tokens = defaultdict(list)
    for r in rows:
        protein_tokens[r["uniprot_id"]].append(r["token_hex"].strip().lower())
    print(f"    Loaded {len(rows):,} tokens for {len(protein_tokens):,} proteins")
    return dict(protein_tokens)


def load_gt_from_db() -> Dict[str, Set[str]]:
    print("  Loading gene_department_map...")
    rows = query_db("""
        SELECT uniprot_id, primary_department, all_departments, confidence, source
        FROM gene_department_map
        WHERE primary_department IS NOT NULL
    """)
    gt = defaultdict(lambda: {"depts": set(), "confidence": 0.0, "source": ""})
    for r in rows:
        uid = r["uniprot_id"]
        depts = r["all_departments"] if r["all_departments"] else [r["primary_department"]]
        conf = float(r["confidence"]) if r["confidence"] else 0.5
        gt[uid]["depts"].update(depts)
        gt[uid]["confidence"] = max(gt[uid]["confidence"], conf)
        gt[uid]["source"] = r["source"] or gt[uid]["source"]
    gt_cache = {uid: info["depts"] for uid, info in gt.items() if info["depts"]}
    print(f"    Loaded {len(gt_cache):,} proteins with departments")
    return gt_cache


def load_canonical_map() -> Dict[str, str]:
    print("  Loading canonical_gene_uniprot...")
    rows = query_db("SELECT gene_name, uniprot_id FROM canonical_gene_uniprot")
    canonical = {r["gene_name"]: r["uniprot_id"] for r in rows}
    print(f"    Loaded {len(canonical):,} canonical mappings")
    return canonical


def compute_word_enrichment(
    protein_ids: Set[str],
    protein_tokens: Dict[str, List[str]],
    vocab_set: Set[str],
    vocab: Dict[str, dict],
    gt_cache: Dict[str, Set[str]],
) -> Dict[str, Dict[str, float]]:
    dept_proteins = defaultdict(set)
    all_in_split = set()
    for uid in protein_ids:
        depts = gt_cache.get(uid)
        if depts:
            for d in depts:
                dept_proteins[d].add(uid)
            all_in_split.add(uid)

    total_with_gt = len(all_in_split)
    if total_with_gt == 0:
        return {}
    dept_rates = {d: len(ps) / total_with_gt for d, ps in dept_proteins.items()}

    word_carriers = defaultdict(set)
    for uid in protein_ids:
        seen = set()
        for tok in protein_tokens.get(uid, []):
            if tok in vocab_set and tok not in seen:
                word_carriers[tok].add(uid)
                seen.add(tok)

    enrichments = {}
    for whex, carriers in word_carriers.items():
        if len(carriers) < 2:
            continue
        func = vocab[whex]["function"]
        carriers_gt = carriers & all_in_split
        if len(carriers_gt) < 2:
            continue
        carriers_in_dept = carriers_gt & dept_proteins.get(func, set())
        obs_rate = len(carriers_in_dept) / len(carriers_gt)
        exp_rate = dept_rates.get(func, 0)
        enrichments[whex] = {
            "enrichment": obs_rate / exp_rate if exp_rate > 0 else 0.0,
            "function": func,
        }
    return enrichments


def pearson_r(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    n = len(xs)
    if n < 3:
        return 0.0, 1.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0:
        return 0.0, 1.0
    r = max(-1.0, min(1.0, sxy / math.sqrt(sxx * syy)))
    if abs(r) >= 1.0:
        return r, 0.0
    t = r * math.sqrt((n - 2) / (1 - r * r))
    df = n - 2
    x_val = df / (df + t * t)
    a, b = df / 2.0, 0.5
    try:
        from math import lgamma, exp as mexp
        log_beta = lgamma(a) + lgamma(b) - lgamma(a + b)
        s = 0.0
        for k in range(200):
            num = 1.0
            for j in range(k):
                num *= (j + 1 - b) / (j + 1)
            s += num * (x_val ** (a + k)) / (a + k)
        ibeta = mexp(-log_beta) * s
        p = max(0.0, min(1.0, ibeta))
    except:
        p = 0.0
    return r, p


def predict_functions_batch(
    uids: List[str],
    protein_tokens: Dict[str, List[str]],
    vocab_set: Set[str],
    vocab: Dict[str, dict],
) -> Dict[str, Tuple[Optional[str], int]]:
    results = {}
    for uid in uids:
        dept_scores = defaultdict(float)
        n_vocab = 0
        tok_counts = Counter(protein_tokens.get(uid, []))
        for tok, count in tok_counts.items():
            if tok in vocab_set:
                w = vocab[tok]
                n_vocab += 1
                dept_scores[w["function"]] += w["confidence"] * count
        predicted = max(dept_scores, key=dept_scores.get) if dept_scores else None
        results[uid] = (predicted, n_vocab)
    return results


def run_single_seed(
    seed, all_uids, protein_tokens, vocab_set, vocab, gt_cache,
    dept_freq, most_common_dept, all_depts,
):
    rng = random.Random(seed)
    shuffled = list(all_uids)
    rng.shuffle(shuffled)
    mid = len(shuffled) // 2
    train_set = set(shuffled[:mid])
    test_set = set(shuffled[mid:])

    print(f"  Seed {seed}: enrichment...", end="", flush=True)
    train_e = compute_word_enrichment(train_set, protein_tokens, vocab_set, vocab, gt_cache)
    test_e = compute_word_enrichment(test_set, protein_tokens, vocab_set, vocab, gt_cache)

    common = set(train_e.keys()) & set(test_e.keys())
    tv = [train_e[w]["enrichment"] for w in common]
    ev = [test_e[w]["enrichment"] for w in common]
    r, p = pearson_r(tv, ev)
    reversed_n = sum(1 for w in common if (train_e[w]["enrichment"] > 1) != (test_e[w]["enrichment"] > 1))
    mean_train = sum(tv) / len(tv) if tv else 0
    mean_test = sum(ev) / len(ev) if ev else 0

    print(f" predictions...", end="", flush=True)
    test_uids = list(test_set)
    predictions = predict_functions_batch(test_uids, protein_tokens, vocab_set, vocab)

    dept_weights = [dept_freq.get(d, 0) for d in all_depts]

    bin_correct = {l: 0 for l in BIN_LABELS}
    bin_total = {l: 0 for l in BIN_LABELS}
    bin_tp = {l: defaultdict(int) for l in BIN_LABELS}
    bin_fp = {l: defaultdict(int) for l in BIN_LABELS}
    bin_fn = {l: defaultdict(int) for l in BIN_LABELS}
    random_correct_bin = {l: 0 for l in BIN_LABELS}
    freq_correct_bin = {l: 0 for l in BIN_LABELS}

    thresh_correct = {t: 0 for t in WORD_THRESHOLDS}
    thresh_total = {t: 0 for t in WORD_THRESHOLDS}
    thresh_random = {t: 0 for t in WORD_THRESHOLDS}
    thresh_freq = {t: 0 for t in WORD_THRESHOLDS}
    thresh_tp = {t: defaultdict(int) for t in WORD_THRESHOLDS}
    thresh_fp = {t: defaultdict(int) for t in WORD_THRESHOLDS}
    thresh_fn = {t: defaultdict(int) for t in WORD_THRESHOLDS}

    confusion = defaultdict(lambda: defaultdict(int))
    overall_c, overall_t, overall_rc, overall_fc = 0, 0, 0, 0

    for uid in test_uids:
        true_depts = gt_cache.get(uid)
        if not true_depts:
            continue
        predicted, n_words = predictions[uid]
        if predicted is None:
            continue

        is_correct = predicted in true_depts
        primary = sorted(true_depts)[0]
        rand_dept = random.Random(seed * 100000 + hash(uid)).choices(all_depts, weights=dept_weights)[0]
        rand_ok = rand_dept in true_depts
        freq_ok = most_common_dept in true_depts

        for i, (lo, hi) in enumerate(WORD_COUNT_BINS):
            if lo <= n_words <= hi:
                label = BIN_LABELS[i]
                bin_total[label] += 1
                bin_correct[label] += is_correct
                if is_correct:
                    bin_tp[label][predicted] += 1
                else:
                    bin_fp[label][predicted] += 1
                    for td in true_depts:
                        bin_fn[label][td] += 1
                random_correct_bin[label] += rand_ok
                freq_correct_bin[label] += freq_ok
                break

        for t in WORD_THRESHOLDS:
            if n_words >= t:
                thresh_total[t] += 1
                thresh_correct[t] += is_correct
                thresh_random[t] += rand_ok
                thresh_freq[t] += freq_ok
                if is_correct:
                    thresh_tp[t][predicted] += 1
                else:
                    thresh_fp[t][predicted] += 1
                    for td in true_depts:
                        thresh_fn[t][td] += 1

        confusion[primary][predicted] += 1
        overall_t += 1
        overall_c += is_correct
        overall_rc += rand_ok
        overall_fc += freq_ok

    def compute_f1s(tp_d, fp_d, fn_d):
        classes = set(tp_d.keys()) | set(fp_d.keys()) | set(fn_d.keys())
        f1s, supports = [], []
        for cls in classes:
            tp, fp, fn = tp_d.get(cls, 0), fp_d.get(cls, 0), fn_d.get(cls, 0)
            prec = tp / (tp + fp) if (tp + fp) else 0
            rec = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
            f1s.append(f1)
            supports.append(tp + fn)
        macro = sum(f1s) / len(f1s) if f1s else 0
        ts = sum(supports)
        weighted = sum(f * s for f, s in zip(f1s, supports)) / ts if ts else 0
        return macro, weighted

    bin_f1m, bin_f1w = {}, {}
    for label in BIN_LABELS:
        m, w = compute_f1s(bin_tp[label], bin_fp[label], bin_fn[label])
        bin_f1m[label] = m
        bin_f1w[label] = w

    thresh_f1m, thresh_f1w = {}, {}
    for t in WORD_THRESHOLDS:
        m, w = compute_f1s(thresh_tp[t], thresh_fp[t], thresh_fn[t])
        thresh_f1m[t] = m
        thresh_f1w[t] = w

    acc = overall_c / overall_t if overall_t else 0
    print(f" r={r:.4f} acc={acc:.4f} (n={overall_t})")

    result = {
        "seed": seed,
        "common_words": len(common),
        "pearson_r": r, "p_value": p,
        "mean_enrichment_train": mean_train, "mean_enrichment_test": mean_test,
        "reversed_predictions": reversed_n,
        "bin_accuracy": {l: bin_correct[l] / bin_total[l] if bin_total[l] else 0 for l in BIN_LABELS},
        "bin_total": {l: bin_total[l] for l in BIN_LABELS},
        "bin_f1_macro": bin_f1m, "bin_f1_weighted": bin_f1w,
        "bin_random": {l: random_correct_bin[l] / bin_total[l] if bin_total[l] else 0 for l in BIN_LABELS},
        "bin_freq": {l: freq_correct_bin[l] / bin_total[l] if bin_total[l] else 0 for l in BIN_LABELS},
        "overall_accuracy": acc,
        "overall_random": overall_rc / overall_t if overall_t else 0,
        "overall_freq": overall_fc / overall_t if overall_t else 0,
        "overall_total": overall_t,
        "thresh_accuracy": {str(t): thresh_correct[t] / thresh_total[t] if thresh_total[t] else 0 for t in WORD_THRESHOLDS},
        "thresh_total": {str(t): thresh_total[t] for t in WORD_THRESHOLDS},
        "thresh_random": {str(t): thresh_random[t] / thresh_total[t] if thresh_total[t] else 0 for t in WORD_THRESHOLDS},
        "thresh_freq": {str(t): thresh_freq[t] / thresh_total[t] if thresh_total[t] else 0 for t in WORD_THRESHOLDS},
        "thresh_f1_macro": {str(t): thresh_f1m[t] for t in WORD_THRESHOLDS},
        "thresh_f1_weighted": {str(t): thresh_f1w[t] for t in WORD_THRESHOLDS},
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }
    if seed == 42:
        result["train_enrichments"] = {w: train_e[w]["enrichment"] for w in common}
        result["test_enrichments"] = {w: test_e[w]["enrichment"] for w in common}
    return result


def generate_scatter_plot(train_e, test_e, r_val, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        words = sorted(train_e.keys())
        xs = [train_e[w] for w in words]
        ys = [test_e[w] for w in words]
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.scatter(xs, ys, alpha=0.3, s=8, c="#2563eb", edgecolors="none")
        mx = min(max(max(xs), max(ys)) * 1.05, 50)
        ax.plot([0, mx], [0, mx], "k--", alpha=0.3)
        ax.set_xlabel("Training Set Enrichment", fontsize=12)
        ax.set_ylabel("Test Set Enrichment", fontsize=12)
        ax.set_title(f"VALDICT Extended Held-Out Enrichment (r = {r_val:.3f}, n = {len(words):,})", fontsize=12)
        ax.set_xlim(0, mx); ax.set_ylim(0, mx); ax.set_aspect("equal")
        ax.text(0.05, 0.95, f"n = {len(words):,} words\nPearson r = {r_val:.3f}",
                transform=ax.transAxes, fontsize=11, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
        print(f"  Saved: {out_path}")
    except Exception as e:
        print(f"  Warning: scatter plot failed: {e}")


def generate_accuracy_curve(all_uids, protein_tokens, vocab_set, vocab, gt_cache, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        predictions = predict_functions_batch(all_uids, protein_tokens, vocab_set, vocab)
        data = []
        for uid in all_uids:
            true_depts = gt_cache.get(uid)
            if not true_depts:
                continue
            pred, nw = predictions[uid]
            if pred is None or nw < 1:
                continue
            data.append((nw, 1 if pred in true_depts else 0))
        data.sort()
        thresholds = sorted(set(wc for wc, _ in data))
        cx, cy, cn = [], [], []
        for t in thresholds:
            sub = [(wc, c) for wc, c in data if wc >= t]
            if len(sub) < 5:
                continue
            cx.append(t)
            cy.append(sum(c for _, c in sub) / len(sub))
            cn.append(len(sub))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [3, 1]})
        ax1.plot(cx, cy, color="#2563eb", linewidth=2)
        ax1.set_ylabel("Prediction Accuracy (multi-label)", fontsize=12)
        ax1.set_title("VALDICT Extended: Accuracy vs Min Word Count", fontsize=13)
        ax1.set_ylim(0, 1.05)
        ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
        ax1.grid(True, alpha=0.2)
        for t in WORD_THRESHOLDS:
            sub = [(wc, c) for wc, c in data if wc >= t]
            if sub:
                acc = sum(c for _, c in sub) / len(sub)
                ax1.axvline(x=t, color="orange", linestyle=":", alpha=0.5)
                ax1.annotate(f">={t}: {acc:.1%} (n={len(sub):,})", xy=(t, acc), fontsize=8,
                             xytext=(t + 1, acc + 0.03),
                             bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))
        ax2.plot(cx, cn, color="#dc2626", linewidth=1.5)
        ax2.set_xlabel("Minimum Word Count", fontsize=12)
        ax2.set_ylabel("Sample Size", fontsize=12)
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.2)
        ax2.set_xlim(ax1.get_xlim())
        plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
        print(f"  Saved: {out_path}")
    except Exception as e:
        print(f"  Warning: accuracy curve failed: {e}")


def mean_std(vals):
    m = sum(vals) / len(vals) if vals else 0
    s = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 0
    return m, s


def main():
    t0 = time.time()
    print("=" * 70)
    print("VAL-DICT-001 v3: Beta Database Validation (Extended Vocabulary)")
    print("=" * 70)
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")

    print("\n[1] Loading data from beta database...")
    vocab = load_vocab_from_db()
    vocab_set = set(vocab.keys())

    protein_tokens = load_tokens_from_db()
    all_uids = sorted(protein_tokens.keys())

    gt_cache = load_gt_from_db()
    canonical = load_canonical_map()

    proteins_with_gt = sum(1 for u in all_uids if u in gt_cache)
    proteins_without_gt = len(all_uids) - proteins_with_gt
    multi_label = sum(1 for u in all_uids if u in gt_cache and len(gt_cache[u]) > 1)
    single_label = proteins_with_gt - multi_label
    mean_labels = sum(len(gt_cache[u]) for u in all_uids if u in gt_cache) / proteins_with_gt if proteins_with_gt else 0

    print(f"\n  Data loaded:")
    print(f"    Vocabulary: {len(vocab):,} classified words")
    print(f"    Proteins: {len(all_uids):,}")
    print(f"    Ground truth: {proteins_with_gt:,} ({proteins_with_gt/len(all_uids)*100:.1f}%)")
    print(f"      Single-label: {single_label:,} | Multi-label: {multi_label:,} | Mean: {mean_labels:.2f}")
    print(f"    Canonical isoforms: {len(canonical):,}")

    toks_with = [len(protein_tokens[u]) for u in all_uids if u in gt_cache]
    toks_without = [len(protein_tokens[u]) for u in all_uids if u not in gt_cache]
    mean_with = sum(toks_with) / len(toks_with) if toks_with else 0
    mean_without = sum(toks_without) / len(toks_without) if toks_without else 0

    all_dept_labels = set()
    for ds in gt_cache.values():
        all_dept_labels.update(ds)
    dept_counts = Counter()
    for u in all_uids:
        if u in gt_cache:
            for d in gt_cache[u]:
                dept_counts[d] += 1
    all_depts = sorted(all_dept_labels)
    total_dept = sum(dept_counts.values())
    dept_freq = {d: c / total_dept for d, c in dept_counts.items()}
    most_common_dept = dept_counts.most_common(1)[0][0]
    print(f"    Departments: {len(all_depts)}")
    print(f"    Most common: {most_common_dept} ({dept_freq[most_common_dept]:.1%})")

    predictions_all = predict_functions_batch(all_uids, protein_tokens, vocab_set, vocab)
    sorted_wcs = sorted(predictions_all[u][1] for u in all_uids)
    median_wc = sorted_wcs[len(sorted_wcs) // 2]
    zero_words = sum(1 for wc in sorted_wcs if wc == 0)
    print(f"    Median vocab words/protein: {median_wc}")
    print(f"    Proteins with 0 vocab words: {zero_words:,} ({zero_words/len(all_uids)*100:.1f}%)")
    print(f"    Coverage at thresholds:")
    for t in WORD_THRESHOLDS:
        n = sum(1 for wc in sorted_wcs if wc >= t)
        n_gt = sum(1 for u in all_uids if predictions_all[u][1] >= t and u in gt_cache)
        print(f"      >={t}: {n:,} proteins ({n/len(all_uids)*100:.1f}%), {n_gt:,} with GT")

    print(f"\n[2] Well-known protein check...")
    for gene in ["BRCA1", "TP53", "EGFR", "KRAS", "MYC"]:
        can_uid = canonical.get(gene)
        if can_uid and can_uid in protein_tokens:
            pred, nw = predictions_all[can_uid]
            true = gt_cache.get(can_uid, set())
            match = "CORRECT" if pred in true else "WRONG"
            print(f"    {gene} ({can_uid}): {len(protein_tokens[can_uid])} tokens, {nw} vocab words, pred={pred}, true={true}, {match}")
        else:
            print(f"    {gene}: canonical={can_uid}, not in tokens")

    print(f"\n[3] Running {len(SEEDS)} held-out splits...")
    seed_results = []
    for seed in SEEDS:
        result = run_single_seed(seed, all_uids, protein_tokens, vocab_set, vocab,
                                 gt_cache, dept_freq, most_common_dept, all_depts)
        seed_results.append(result)

    rs = [s["pearson_r"] for s in seed_results]
    mean_r, std_r = mean_std(rs)
    accs = [s["overall_accuracy"] for s in seed_results]
    mean_acc, std_acc = mean_std(accs)

    print(f"\n  CORRELATION: r = {mean_r:.4f} +/- {std_r:.4f}")
    print(f"  ACCURACY: {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"\n  THRESHOLD RESULTS:")
    for t in WORD_THRESHOLDS:
        ta = [s["thresh_accuracy"][str(t)] for s in seed_results]
        tt = [s["thresh_total"][str(t)] for s in seed_results]
        tr = [s["thresh_random"][str(t)] for s in seed_results]
        tf = [s["thresh_freq"][str(t)] for s in seed_results]
        m_a, s_a = mean_std(ta)
        m_t, _ = mean_std(tt)
        m_r2, _ = mean_std(tr)
        m_f, _ = mean_std(tf)
        print(f"    >={t}: acc={m_a:.4f}+/-{s_a:.4f} (n~{m_t:.0f}) | random={m_r2:.4f} freq={m_f:.4f}")

    print("\n[4] Generating plots...")
    s42 = next((s for s in seed_results if s["seed"] == 42), seed_results[0])
    if s42.get("train_enrichments"):
        generate_scatter_plot(s42["train_enrichments"], s42["test_enrichments"],
                              s42["pearson_r"], os.path.join(OUT_DIR, "VAL-DICT-001_v3_beta_enrichment_scatter.png"))
    generate_accuracy_curve(all_uids, protein_tokens, vocab_set, vocab, gt_cache,
                            os.path.join(OUT_DIR, "VAL-DICT-001_v3_beta_accuracy_curve.png"))

    print("\n[5] Confusion matrix...")
    import csv
    merged_cm = defaultdict(lambda: defaultdict(int))
    for s in seed_results:
        for td, preds in s["confusion"].items():
            for pd, c in preds.items():
                merged_cm[td][pd] += c
    cm_depts = sorted(set(list(merged_cm.keys()) + [d for v in merged_cm.values() for d in v]))
    cm_path = os.path.join(OUT_DIR, "VAL-DICT-001_v3_beta_confusion_matrix.csv")
    with open(cm_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\predicted"] + cm_depts)
        for td in cm_depts:
            w.writerow([td] + [merged_cm[td].get(pd, 0) for pd in cm_depts])
    print(f"  Saved: {cm_path}")

    print("\n[6] Building output files...")

    v2_results_path = os.path.join(OUT_DIR, "VAL-DICT-001_holdout_results.json")
    v2_comparison = {}
    if os.path.exists(v2_results_path):
        with open(v2_results_path) as f:
            v2 = json.load(f)
        v2_comparison = {
            "v2_csv_pearson_r": v2["holdout_correlation"]["mean_pearson_r"],
            "v2_csv_accuracy": v2["function_prediction"]["overall_accuracy_multi_mean"],
            "v2_csv_vocab_size": v2["data_summary"]["vocabulary_words_total"],
            "v2_csv_gt_coverage": v2["data_summary"]["ground_truth_coverage_pct"],
        }

    results = {
        "validation_id": "VAL-DICT-001",
        "version": "v3_beta_database",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "v2_6bit",
        "data_source": "beta_database",
        "seeds": SEEDS, "n_seeds": len(SEEDS),
        "data_summary": {
            "total_proteins": len(all_uids),
            "proteins_with_gt": proteins_with_gt,
            "gt_coverage_pct": round(proteins_with_gt / len(all_uids) * 100, 1),
            "single_label": single_label,
            "multi_label": multi_label,
            "mean_labels": round(mean_labels, 2),
            "vocabulary_size": len(vocab),
            "departments": len(all_depts),
            "most_common_dept": most_common_dept,
            "median_vocab_words": median_wc,
            "zero_word_proteins": zero_words,
            "zero_word_pct": round(zero_words / len(all_uids) * 100, 1),
            "mean_tokens_with_gt": round(mean_with, 1),
            "mean_tokens_without_gt": round(mean_without, 1),
        },
        "holdout_correlation": {
            "mean_pearson_r": round(mean_r, 4),
            "std_pearson_r": round(std_r, 4),
            "per_seed_r": [round(r, 4) for r in rs],
            "mean_common_words": round(sum(s["common_words"] for s in seed_results) / len(seed_results), 1),
            "mean_reversed": round(sum(s["reversed_predictions"] for s in seed_results) / len(seed_results), 1),
        },
        "function_prediction": {
            "scoring": "multi-label",
            "overall_accuracy_mean": round(mean_acc, 4),
            "overall_accuracy_std": round(std_acc, 4),
            "overall_random_baseline": round(sum(s["overall_random"] for s in seed_results) / len(seed_results), 4),
            "overall_freq_baseline": round(sum(s["overall_freq"] for s in seed_results) / len(seed_results), 4),
            "by_threshold": {},
            "by_bin": {},
        },
        "comparison_to_v2_csv": v2_comparison,
    }

    for t in WORD_THRESHOLDS:
        ta = [s["thresh_accuracy"][str(t)] for s in seed_results]
        tt = [s["thresh_total"][str(t)] for s in seed_results]
        tr = [s["thresh_random"][str(t)] for s in seed_results]
        tf = [s["thresh_freq"][str(t)] for s in seed_results]
        tf1m = [s["thresh_f1_macro"][str(t)] for s in seed_results]
        tf1w = [s["thresh_f1_weighted"][str(t)] for s in seed_results]
        m_a, s_a = mean_std(ta)
        m_t, _ = mean_std(tt)
        m_r2, _ = mean_std(tr)
        m_f, _ = mean_std(tf)
        m_f1m, _ = mean_std(tf1m)
        m_f1w, _ = mean_std(tf1w)
        n_total = sum(1 for u in all_uids if predictions_all[u][1] >= t)
        results["function_prediction"]["by_threshold"][f">={t}"] = {
            "accuracy": {"mean": round(m_a, 4), "std": round(s_a, 4)},
            "f1_macro": round(m_f1m, 4), "f1_weighted": round(m_f1w, 4),
            "random": round(m_r2, 4), "freq": round(m_f, 4),
            "n_per_split": round(m_t), "n_total": n_total,
            "coverage_pct": round(n_total / len(all_uids) * 100, 1),
        }

    for label in BIN_LABELS:
        ba = [s["bin_accuracy"][label] for s in seed_results]
        bt = [s["bin_total"][label] for s in seed_results]
        bf1m = [s["bin_f1_macro"][label] for s in seed_results]
        bf1w = [s["bin_f1_weighted"][label] for s in seed_results]
        bra = [s["bin_random"][label] for s in seed_results]
        bfa = [s["bin_freq"][label] for s in seed_results]
        m_a, s_a = mean_std(ba)
        m_t, _ = mean_std(bt)
        m_f1m, _ = mean_std(bf1m)
        m_f1w, _ = mean_std(bf1w)
        m_r2, _ = mean_std(bra)
        m_f, _ = mean_std(bfa)
        results["function_prediction"]["by_bin"][label] = {
            "accuracy": {"mean": round(m_a, 4), "std": round(s_a, 4)},
            "f1_macro": round(m_f1m, 4), "f1_weighted": round(m_f1w, 4),
            "random": round(m_r2, 4), "freq": round(m_f, 4),
            "n_per_split": round(m_t),
        }

    with open(os.path.join(OUT_DIR, "VAL-DICT-001_v3_beta_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    md = []
    md.append("# VAL-DICT-001 v3: Beta Database Validation Report\n")
    md.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    md.append(f"**Source:** Beta database (valdict_extended + gene_department_map)")
    md.append(f"**Seeds:** {SEEDS[0]}-{SEEDS[-1]} ({len(SEEDS)} splits)")
    md.append(f"**Scoring:** Multi-label\n")

    md.append("## Side-by-Side Comparison: CSV vs Beta DB\n")
    md.append("| Metric | v2 (CSV) | v3 (Beta DB) |")
    md.append("|--------|----------|--------------|")
    md.append(f"| Vocabulary | 1,932 words | {len(vocab):,} words |")
    md.append(f"| GT coverage | {v2_comparison.get('v2_csv_gt_coverage', '?')}% | {proteins_with_gt/len(all_uids)*100:.1f}% |")
    md.append(f"| Pearson r | {v2_comparison.get('v2_csv_pearson_r', '?')} | {mean_r:.4f} |")
    md.append(f"| Overall accuracy | {v2_comparison.get('v2_csv_accuracy', '?')} | {mean_acc:.4f} |")
    md.append(f"| Zero-word proteins | ~32% | {zero_words/len(all_uids)*100:.1f}% |")
    md.append(f"| Departments | 22 | {len(all_depts)} |\n")

    md.append("## Data Summary\n")
    md.append(f"| Metric | Value |")
    md.append(f"|--------|-------|")
    md.append(f"| Total proteins | {len(all_uids):,} |")
    md.append(f"| With ground truth | {proteins_with_gt:,} ({proteins_with_gt/len(all_uids)*100:.1f}%) |")
    md.append(f"| Multi-label | {multi_label:,} (mean {mean_labels:.2f} labels) |")
    md.append(f"| Vocabulary (classified) | {len(vocab):,} |")
    md.append(f"| Median vocab words/protein | {median_wc} |")
    md.append(f"| Zero-word proteins | {zero_words:,} ({zero_words/len(all_uids)*100:.1f}%) |")
    md.append(f"| Departments | {len(all_depts)} |\n")

    md.append("## Part 1: Held-Out Correlation\n")
    md.append(f"**Pearson r = {mean_r:.4f} +/- {std_r:.4f}**\n")
    md.append(f"Per-seed: {', '.join(f'{r:.4f}' for r in rs)}\n")

    md.append("## Part 2: Function Prediction by Threshold\n")
    md.append("| Min Words | Accuracy | Macro F1 | Weighted F1 | Random | Freq | n/split | Total | Coverage |")
    md.append("|-----------|----------|----------|-------------|--------|------|---------|-------|----------|")
    for t in WORD_THRESHOLDS:
        d = results["function_prediction"]["by_threshold"][f">={t}"]
        md.append(f"| >={t} | {d['accuracy']['mean']:.4f}+/-{d['accuracy']['std']:.4f} | {d['f1_macro']:.4f} | {d['f1_weighted']:.4f} | {d['random']:.4f} | {d['freq']:.4f} | ~{d['n_per_split']} | {d['n_total']:,} | {d['coverage_pct']}% |")

    md.append("\n### By Word Count Bin\n")
    md.append("| Bin | Accuracy | Macro F1 | Random | Freq | n/split |")
    md.append("|-----|----------|----------|--------|------|---------|")
    for label in BIN_LABELS:
        d = results["function_prediction"]["by_bin"][label]
        md.append(f"| {label} | {d['accuracy']['mean']:.4f}+/-{d['accuracy']['std']:.4f} | {d['f1_macro']:.4f} | {d['random']:.4f} | {d['freq']:.4f} | ~{d['n_per_split']} |")

    md.append(f"\n## Baselines\n")
    md.append(f"- Random (proportional): {results['function_prediction']['overall_random_baseline']:.4f}")
    md.append(f"- Frequency (always {most_common_dept}): {results['function_prediction']['overall_freq_baseline']:.4f}\n")

    md.append("## Provenance\n")
    md.append("- **Source:** Beta database (BETA_DATABASE_URL)")
    md.append(f"- **Tables:** valdict_extended ({len(vocab):,} words), gene_department_map ({proteins_with_gt:,} proteins), protein_tokens_v2 ({len(all_uids):,} proteins)")
    md.append(f"- **Canonical isoforms:** {len(canonical):,} from canonical_gene_uniprot")
    md.append("")

    with open(os.path.join(OUT_DIR, "VAL-DICT-001_v3_beta_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"COMPLETE in {elapsed:.0f}s")
    print(f"  r = {mean_r:.4f} +/- {std_r:.4f}")
    print(f"  accuracy = {mean_acc:.4f} +/- {std_acc:.4f}")
    for t in WORD_THRESHOLDS:
        d = results["function_prediction"]["by_threshold"][f">={t}"]
        print(f"  >={t}: {d['accuracy']['mean']:.4f}+/-{d['accuracy']['std']:.4f} (n={d['n_total']:,}, {d['coverage_pct']}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
