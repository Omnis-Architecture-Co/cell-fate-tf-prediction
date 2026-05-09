#!/usr/bin/env python3
"""
VAL-DICT-001 v2: Publication-Ready Held-Out Validation of VALDICT001 Vocabulary
================================================================================
Multi-label scoring + threshold-based accuracy reporting.

Changes from v1:
  - Multi-label ground truth: prediction correct if it matches ANY department
  - Accuracy reported at word-count thresholds: >=2, >=5, >=10, >=20
  - Coverage percentage reported for each threshold
  - Enrichment correlation unchanged (already valid)
"""

import csv
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(WORKSPACE, "server", "data", "human")
OUT_DIR = os.path.join(WORKSPACE, "validation")

TOKENS_FILE = os.path.join(DATA_DIR, "protein_tokens_v2_with_genes.csv")
VOCAB_FILE = os.path.join(DATA_DIR, "vocabulary.csv")
DEPTS_FILE = os.path.join(DATA_DIR, "gene_departments.csv")

SEEDS = list(range(42, 52))
WORD_THRESHOLDS = [2, 5, 10, 20]
WORD_COUNT_BINS = [(2, 3), (4, 10), (11, 50), (51, 999999)]
BIN_LABELS = ["2-3", "4-10", "11-50", "50+"]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_vocabulary() -> Dict[str, dict]:
    vocab = {}
    with open(VOCAB_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hex_norm = row["word_hex"].strip().lower().replace("0x", "")
            vocab[hex_norm] = {
                "hex": row["word_hex"].strip(),
                "hex_norm": hex_norm,
                "function": (row.get("primary_function") or "Unclassified").strip(),
                "enrichment": float(row.get("token_enrichment") or "0"),
                "occurrences": int(row.get("occurrences") or "0"),
                "carrier_proteins": int(row.get("carrier_proteins") or "0"),
                "byte_length": int(row.get("word_length") or "2"),
            }
    return vocab


def load_tokens_and_genes() -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    protein_tokens = defaultdict(list)
    uid_to_gene = {}
    with open(TOKENS_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["uniprot_id"].strip()
            tok = row["token_hex"].strip().lower()
            protein_tokens[uid].append(tok)
            if uid not in uid_to_gene:
                gn = row["gene_name"].strip()
                uid_to_gene[uid] = gn.split()[0] if gn else ""
    return dict(protein_tokens), uid_to_gene


def build_gt_cache_multilabel(all_uids: List[str], uid_to_gene: Dict[str, str]) -> Dict[str, Set[str]]:
    raw_gt = defaultdict(set)
    with open(DEPTS_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["gene"].strip()
            dept = row["department"].strip()
            if key and dept:
                raw_gt[key].add(dept)
    raw_gt_upper = {}
    for k, v in raw_gt.items():
        ku = k.upper()
        if ku not in raw_gt_upper:
            raw_gt_upper[ku] = set()
        raw_gt_upper[ku].update(v)

    cache = {}
    for uid in all_uids:
        if uid in raw_gt:
            cache[uid] = raw_gt[uid]
        else:
            gene = uid_to_gene.get(uid, "")
            if gene:
                if gene in raw_gt:
                    cache[uid] = raw_gt[gene]
                elif gene.upper() in raw_gt_upper:
                    cache[uid] = raw_gt_upper[gene.upper()]
    return cache


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
        if func == "Unclassified":
            continue
        carriers_gt = carriers & all_in_split
        if len(carriers_gt) < 2:
            continue
        carriers_in_dept = carriers_gt & dept_proteins.get(func, set())
        obs_rate = len(carriers_in_dept) / len(carriers_gt)
        exp_rate = dept_rates.get(func, 0)
        enrichments[whex] = {
            "enrichment": obs_rate / exp_rate if exp_rate > 0 else 0.0,
            "n_carriers": len(carriers),
            "observed_rate": obs_rate,
            "expected_rate": exp_rate,
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
) -> Dict[str, Tuple[Optional[str], int, Dict[str, float]]]:
    results = {}
    for uid in uids:
        dept_scores = defaultdict(float)
        n_vocab_words = 0
        tok_counts = Counter(protein_tokens.get(uid, []))
        for tok, count in tok_counts.items():
            if tok in vocab_set:
                w = vocab[tok]
                func = w["function"]
                if func != "Unclassified":
                    n_vocab_words += 1
                    dept_scores[func] += w["enrichment"] * count
        predicted = max(dept_scores, key=dept_scores.get) if dept_scores else None
        top_3 = sorted(dept_scores.items(), key=lambda x: -x[1])[:3]
        results[uid] = (predicted, n_vocab_words, dict(top_3))
    return results


def run_single_seed(
    seed: int,
    all_uids: List[str],
    protein_tokens: Dict[str, List[str]],
    vocab_set: Set[str],
    vocab: Dict[str, dict],
    gt_cache: Dict[str, Set[str]],
    dept_freq: Dict[str, float],
    most_common_dept: str,
    all_depts: List[str],
) -> dict:
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

    bin_correct_single = {l: 0 for l in BIN_LABELS}
    bin_correct_multi = {l: 0 for l in BIN_LABELS}
    bin_total = {l: 0 for l in BIN_LABELS}
    bin_tp = {l: defaultdict(int) for l in BIN_LABELS}
    bin_fp = {l: defaultdict(int) for l in BIN_LABELS}
    bin_fn = {l: defaultdict(int) for l in BIN_LABELS}
    random_correct = {l: 0 for l in BIN_LABELS}
    freq_correct = {l: 0 for l in BIN_LABELS}
    confusion = defaultdict(lambda: defaultdict(int))

    thresh_correct = {t: 0 for t in WORD_THRESHOLDS}
    thresh_total = {t: 0 for t in WORD_THRESHOLDS}
    thresh_random = {t: 0 for t in WORD_THRESHOLDS}
    thresh_freq = {t: 0 for t in WORD_THRESHOLDS}
    thresh_tp = {t: defaultdict(int) for t in WORD_THRESHOLDS}
    thresh_fp = {t: defaultdict(int) for t in WORD_THRESHOLDS}
    thresh_fn = {t: defaultdict(int) for t in WORD_THRESHOLDS}

    overall_c_s, overall_c_m, overall_t = 0, 0, 0
    overall_rc, overall_fc = 0, 0

    for uid in test_uids:
        true_depts = gt_cache.get(uid)
        if not true_depts:
            continue
        predicted, n_words, _ = predictions[uid]
        if predicted is None:
            continue

        is_correct_multi = predicted in true_depts
        primary_dept = sorted(true_depts)[0]
        is_correct_single = predicted == primary_dept

        rand_dept = random.Random(seed * 100000 + hash(uid)).choices(all_depts, weights=dept_weights)[0]
        rand_correct_flag = rand_dept in true_depts
        freq_correct_flag = most_common_dept in true_depts

        for i, (lo, hi) in enumerate(WORD_COUNT_BINS):
            if lo <= n_words <= hi:
                label = BIN_LABELS[i]
                bin_total[label] += 1
                bin_correct_single[label] += is_correct_single
                bin_correct_multi[label] += is_correct_multi
                if is_correct_multi:
                    bin_tp[label][predicted] += 1
                else:
                    bin_fp[label][predicted] += 1
                    for td in true_depts:
                        bin_fn[label][td] += 1
                confusion[primary_dept][predicted] += 1
                random_correct[label] += rand_correct_flag
                freq_correct[label] += freq_correct_flag
                break

        for t in WORD_THRESHOLDS:
            if n_words >= t:
                thresh_total[t] += 1
                thresh_correct[t] += is_correct_multi
                thresh_random[t] += rand_correct_flag
                thresh_freq[t] += freq_correct_flag
                if is_correct_multi:
                    thresh_tp[t][predicted] += 1
                else:
                    thresh_fp[t][predicted] += 1
                    for td in true_depts:
                        thresh_fn[t][td] += 1

        overall_t += 1
        overall_c_s += is_correct_single
        overall_c_m += is_correct_multi
        overall_rc += rand_correct_flag
        overall_fc += freq_correct_flag

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

    bin_f1_macro, bin_f1_weighted = {}, {}
    for label in BIN_LABELS:
        m, w = compute_f1s(bin_tp[label], bin_fp[label], bin_fn[label])
        bin_f1_macro[label] = m
        bin_f1_weighted[label] = w

    thresh_f1_macro, thresh_f1_weighted = {}, {}
    for t in WORD_THRESHOLDS:
        m, w = compute_f1s(thresh_tp[t], thresh_fp[t], thresh_fn[t])
        thresh_f1_macro[t] = m
        thresh_f1_weighted[t] = w

    acc_m = overall_c_m / overall_t if overall_t else 0
    print(f" r={r:.4f} acc_multi={acc_m:.4f} acc_single={overall_c_s/overall_t:.4f}" if overall_t else " (no predictions)")

    result = {
        "seed": seed,
        "train_size": len(train_set), "test_size": len(test_set),
        "common_words": len(common),
        "pearson_r": r, "p_value": p,
        "mean_enrichment_train": mean_train, "mean_enrichment_test": mean_test,
        "reversed_predictions": reversed_n,
        "bin_accuracy_single": {l: bin_correct_single[l] / bin_total[l] if bin_total[l] else 0 for l in BIN_LABELS},
        "bin_accuracy_multi": {l: bin_correct_multi[l] / bin_total[l] if bin_total[l] else 0 for l in BIN_LABELS},
        "bin_total": {l: bin_total[l] for l in BIN_LABELS},
        "bin_f1_macro": bin_f1_macro, "bin_f1_weighted": bin_f1_weighted,
        "random_accuracy": {l: random_correct[l] / bin_total[l] if bin_total[l] else 0 for l in BIN_LABELS},
        "freq_accuracy": {l: freq_correct[l] / bin_total[l] if bin_total[l] else 0 for l in BIN_LABELS},
        "overall_accuracy_single": overall_c_s / overall_t if overall_t else 0,
        "overall_accuracy_multi": overall_c_m / overall_t if overall_t else 0,
        "overall_random_accuracy": overall_rc / overall_t if overall_t else 0,
        "overall_freq_accuracy": overall_fc / overall_t if overall_t else 0,
        "overall_total": overall_t,
        "thresh_accuracy": {str(t): thresh_correct[t] / thresh_total[t] if thresh_total[t] else 0 for t in WORD_THRESHOLDS},
        "thresh_total": {str(t): thresh_total[t] for t in WORD_THRESHOLDS},
        "thresh_random": {str(t): thresh_random[t] / thresh_total[t] if thresh_total[t] else 0 for t in WORD_THRESHOLDS},
        "thresh_freq": {str(t): thresh_freq[t] / thresh_total[t] if thresh_total[t] else 0 for t in WORD_THRESHOLDS},
        "thresh_f1_macro": {str(t): thresh_f1_macro[t] for t in WORD_THRESHOLDS},
        "thresh_f1_weighted": {str(t): thresh_f1_weighted[t] for t in WORD_THRESHOLDS},
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
        ax.scatter(xs, ys, alpha=0.4, s=15, c="#2563eb", edgecolors="none")
        mx = max(max(xs), max(ys)) * 1.05
        ax.plot([0, mx], [0, mx], "k--", alpha=0.3)
        ax.set_xlabel("Training Set Enrichment", fontsize=12)
        ax.set_ylabel("Test Set Enrichment", fontsize=12)
        ax.set_title(f"VALDICT001 Held-Out Enrichment Correlation (r = {r_val:.3f})", fontsize=13)
        ax.set_xlim(0, mx); ax.set_ylim(0, mx); ax.set_aspect("equal")
        ax.text(0.05, 0.95, f"n = {len(words)} words\nPearson r = {r_val:.3f}",
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
            pred, nw, _ = predictions[uid]
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
        ax1.set_title("VALDICT001 Function Prediction: Accuracy vs Min Word Count", fontsize=13)
        ax1.set_ylim(0, 1.05)
        ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="50%")
        ax1.grid(True, alpha=0.2)
        for t in WORD_THRESHOLDS:
            sub = [(wc, c) for wc, c in data if wc >= t]
            if sub:
                acc = sum(c for _, c in sub) / len(sub)
                ax1.axvline(x=t, color="orange", linestyle=":", alpha=0.5)
                ax1.annotate(f">={t}: {acc:.1%} (n={len(sub)})", xy=(t, acc), fontsize=9,
                             xytext=(t + 2, acc + 0.03),
                             bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))

        ax2.plot(cx, cn, color="#dc2626", linewidth=1.5)
        ax2.set_xlabel("Minimum Vocabulary Word Count", fontsize=12)
        ax2.set_ylabel("Sample Size (n)", fontsize=12)
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
    print("VAL-DICT-001 v2: VALDICT001 Held-Out Validation (Multi-Label)")
    print("=" * 70)
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")

    print("\n[1] Input file hashes...")
    hashes = {
        "protein_tokens_v2_with_genes.csv": sha256_file(TOKENS_FILE),
        "vocabulary.csv": sha256_file(VOCAB_FILE),
        "gene_departments.csv": sha256_file(DEPTS_FILE),
    }
    for name, h in hashes.items():
        print(f"  {name}: {h[:16]}...")
    with open(os.path.join(OUT_DIR, "VAL-DICT-001_input_hashes.json"), "w") as f:
        json.dump(hashes, f, indent=2)

    print("\n[2] Loading data...")
    vocab = load_vocabulary()
    vocab_set = set(vocab.keys())
    print(f"  Vocabulary: {len(vocab)} words")

    protein_tokens, uid_to_gene = load_tokens_and_genes()
    all_uids = sorted(protein_tokens.keys())
    print(f"  Proteins: {len(all_uids):,}")

    gt_cache = build_gt_cache_multilabel(all_uids, uid_to_gene)
    proteins_with_gt = len(gt_cache)
    proteins_without_gt = len(all_uids) - proteins_with_gt
    multi_label_count = sum(1 for ds in gt_cache.values() if len(ds) > 1)
    single_label_count = sum(1 for ds in gt_cache.values() if len(ds) == 1)
    mean_labels = sum(len(ds) for ds in gt_cache.values()) / proteins_with_gt if proteins_with_gt else 0
    print(f"  Ground truth: {proteins_with_gt:,} proteins ({proteins_with_gt/len(all_uids)*100:.1f}% coverage)")
    print(f"    Single-label: {single_label_count:,} | Multi-label: {multi_label_count:,} | Mean labels: {mean_labels:.2f}")

    toks_with = [len(protein_tokens[u]) for u in all_uids if u in gt_cache]
    toks_without = [len(protein_tokens[u]) for u in all_uids if u not in gt_cache]
    mean_with = sum(toks_with) / len(toks_with) if toks_with else 0
    mean_without = sum(toks_without) / len(toks_without) if toks_without else 0

    all_dept_labels = set()
    for ds in gt_cache.values():
        all_dept_labels.update(ds)
    dept_counts = Counter()
    for ds in gt_cache.values():
        for d in ds:
            dept_counts[d] += 1
    all_depts = sorted(all_dept_labels)
    total_gt = sum(dept_counts.values())
    dept_freq = {d: c / total_gt for d, c in dept_counts.items()}
    most_common_dept = dept_counts.most_common(1)[0][0]
    classified_vocab = sum(1 for w in vocab.values() if w["function"] != "Unclassified")
    print(f"  Departments: {len(all_depts)}, most common: {most_common_dept} ({dept_freq[most_common_dept]:.1%})")
    print(f"  Classified vocab: {classified_vocab}/{len(vocab)}")

    predictions_all = predict_functions_batch(all_uids, protein_tokens, vocab_set, vocab)
    word_count_dist = Counter()
    for uid in all_uids:
        _, nw, _ = predictions_all[uid]
        word_count_dist[nw] += 1
    sorted_wcs = [nw for uid in all_uids for _, nw, _ in [predictions_all[uid]]]
    sorted_wcs.sort()
    median_wc = sorted_wcs[len(sorted_wcs) // 2]
    print(f"  Median classified vocab words/protein: {median_wc}")
    print(f"  Coverage at thresholds:")
    for t in WORD_THRESHOLDS:
        n_above = sum(1 for wc in sorted_wcs if wc >= t)
        n_above_gt = sum(1 for uid in all_uids if predictions_all[uid][1] >= t and uid in gt_cache)
        print(f"    >={t} words: {n_above:,} proteins ({n_above/len(all_uids)*100:.1f}%), {n_above_gt:,} with GT")

    print(f"\n[3] Running {len(SEEDS)} held-out splits...")
    seed_results = []
    for seed in SEEDS:
        result = run_single_seed(seed, all_uids, protein_tokens, vocab_set, vocab,
                                 gt_cache, dept_freq, most_common_dept, all_depts)
        seed_results.append(result)

    rs = [s["pearson_r"] for s in seed_results]
    mean_r, std_r = mean_std(rs)
    accs_m = [s["overall_accuracy_multi"] for s in seed_results]
    accs_s = [s["overall_accuracy_single"] for s in seed_results]
    mean_acc_m, std_acc_m = mean_std(accs_m)
    mean_acc_s, std_acc_s = mean_std(accs_s)

    print(f"\n  CORRELATION: r = {mean_r:.4f} +/- {std_r:.4f}")
    print(f"  ACCURACY (multi-label): {mean_acc_m:.4f} +/- {std_acc_m:.4f}")
    print(f"  ACCURACY (single-label): {mean_acc_s:.4f} +/- {std_acc_s:.4f}")
    print(f"\n  THRESHOLD RESULTS (multi-label):")
    for t in WORD_THRESHOLDS:
        ta = [s["thresh_accuracy"][str(t)] for s in seed_results]
        tt = [s["thresh_total"][str(t)] for s in seed_results]
        tr = [s["thresh_random"][str(t)] for s in seed_results]
        tf = [s["thresh_freq"][str(t)] for s in seed_results]
        m_a, s_a = mean_std(ta)
        m_t, _ = mean_std(tt)
        m_r2, _ = mean_std(tr)
        m_f, _ = mean_std(tf)
        print(f"    >={t} words: acc={m_a:.4f}+/-{s_a:.4f} (n~{m_t:.0f}) | random={m_r2:.4f} freq={m_f:.4f}")

    print("\n[4] Generating plots...")
    s42 = next((s for s in seed_results if s["seed"] == 42), seed_results[0])
    if s42.get("train_enrichments"):
        generate_scatter_plot(s42["train_enrichments"], s42["test_enrichments"],
                              s42["pearson_r"], os.path.join(OUT_DIR, "VAL-DICT-001_enrichment_scatter.png"))
    generate_accuracy_curve(all_uids, protein_tokens, vocab_set, vocab, gt_cache,
                            os.path.join(OUT_DIR, "VAL-DICT-001_accuracy_curve.png"))

    print("\n[5] Confusion matrix...")
    merged_cm = defaultdict(lambda: defaultdict(int))
    for s in seed_results:
        for td, preds in s["confusion"].items():
            for pd, c in preds.items():
                merged_cm[td][pd] += c
    cm_depts = sorted(set(list(merged_cm.keys()) + [d for v in merged_cm.values() for d in v]))
    cm_path = os.path.join(OUT_DIR, "VAL-DICT-001_confusion_matrix.csv")
    with open(cm_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\predicted"] + cm_depts)
        for td in cm_depts:
            w.writerow([td] + [merged_cm[td].get(pd, 0) for pd in cm_depts])
    print(f"  Saved: {cm_path}")

    print("\n[6] Building output files...")
    results = {
        "validation_id": "VAL-DICT-001",
        "version": "v2_multilabel",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "v2_6bit",
        "seeds": SEEDS, "n_seeds": len(SEEDS),
        "input_files": hashes,
        "data_summary": {
            "total_proteins": len(all_uids),
            "proteins_with_ground_truth": proteins_with_gt,
            "proteins_without_ground_truth": proteins_without_gt,
            "ground_truth_coverage_pct": round(proteins_with_gt / len(all_uids) * 100, 1),
            "single_label_proteins": single_label_count,
            "multi_label_proteins": multi_label_count,
            "mean_labels_per_protein": round(mean_labels, 2),
            "mean_tokens_with_gt": round(mean_with, 1),
            "mean_tokens_without_gt": round(mean_without, 1),
            "median_classified_vocab_words": median_wc,
            "vocabulary_words_total": len(vocab),
            "vocabulary_classified": classified_vocab,
            "departments": len(all_depts),
            "department_distribution": {d: {"count": dept_counts[d], "pct": round(dept_freq[d] * 100, 1)} for d in all_depts},
            "most_common_department": most_common_dept,
        },
        "holdout_correlation": {
            "mean_pearson_r": round(mean_r, 4),
            "std_pearson_r": round(std_r, 4),
            "per_seed_r": [round(r, 4) for r in rs],
            "per_seed_p_value": [s["p_value"] for s in seed_results],
            "mean_common_words": round(sum(s["common_words"] for s in seed_results) / len(seed_results), 1),
            "mean_enrichment_train": round(sum(s["mean_enrichment_train"] for s in seed_results) / len(seed_results), 3),
            "mean_enrichment_test": round(sum(s["mean_enrichment_test"] for s in seed_results) / len(seed_results), 3),
            "mean_reversed_predictions": round(sum(s["reversed_predictions"] for s in seed_results) / len(seed_results), 1),
        },
        "function_prediction": {
            "scoring": "multi-label (correct if predicted dept is ANY of protein's assigned departments)",
            "overall_accuracy_multi_mean": round(mean_acc_m, 4),
            "overall_accuracy_multi_std": round(std_acc_m, 4),
            "overall_accuracy_single_mean": round(mean_acc_s, 4),
            "overall_accuracy_single_std": round(std_acc_s, 4),
            "by_word_count_bin": {},
            "by_word_count_threshold": {},
            "baselines": {
                "random_overall": round(sum(s["overall_random_accuracy"] for s in seed_results) / len(seed_results), 4),
                "frequency_overall": round(sum(s["overall_freq_accuracy"] for s in seed_results) / len(seed_results), 4),
                "frequency_department": most_common_dept,
            },
        },
        "circularity_note": (
            "gene_departments.csv was derived from GO annotations. The vocabulary word functions "
            "in vocabulary.csv were also assigned via GO enrichment analysis. The held-out 50/50 "
            "split ensures no protein's annotations influenced its own prediction within each "
            "split, but the vocabulary and ground truth share the same upstream GO source. "
            "An independent functional annotation (Reactome, KEGG) would provide stronger evidence."
        ),
    }

    for label in BIN_LABELS:
        ba_m = [s["bin_accuracy_multi"][label] for s in seed_results]
        ba_s = [s["bin_accuracy_single"][label] for s in seed_results]
        bf1m = [s["bin_f1_macro"][label] for s in seed_results]
        bf1w = [s["bin_f1_weighted"][label] for s in seed_results]
        bra = [s["random_accuracy"][label] for s in seed_results]
        bfa = [s["freq_accuracy"][label] for s in seed_results]
        bt = [s["bin_total"][label] for s in seed_results]
        m_am, s_am = mean_std(ba_m)
        m_as, s_as = mean_std(ba_s)
        m_f1m, _ = mean_std(bf1m)
        m_f1w, _ = mean_std(bf1w)
        m_r2, _ = mean_std(bra)
        m_f, _ = mean_std(bfa)
        m_t, _ = mean_std(bt)
        results["function_prediction"]["by_word_count_bin"][label] = {
            "accuracy_multi": {"mean": round(m_am, 4), "std": round(s_am, 4)},
            "accuracy_single": {"mean": round(m_as, 4), "std": round(s_as, 4)},
            "f1_macro": round(m_f1m, 4), "f1_weighted": round(m_f1w, 4),
            "random_baseline": round(m_r2, 4), "freq_baseline": round(m_f, 4),
            "n_proteins_mean": round(m_t),
        }

    for t in WORD_THRESHOLDS:
        ta = [s["thresh_accuracy"][str(t)] for s in seed_results]
        tt = [s["thresh_total"][str(t)] for s in seed_results]
        tr = [s["thresh_random"][str(t)] for s in seed_results]
        tf = [s["thresh_freq"][str(t)] for s in seed_results]
        tf1m = [s["thresh_f1_macro"][str(t)] for s in seed_results]
        tf1w = [s["thresh_f1_weighted"][str(t)] for s in seed_results]
        m_a, s_a = mean_std(ta)
        m_t2, _ = mean_std(tt)
        m_r2, _ = mean_std(tr)
        m_f, _ = mean_std(tf)
        m_f1m, _ = mean_std(tf1m)
        m_f1w, _ = mean_std(tf1w)
        n_total_proteome = sum(1 for uid in all_uids if predictions_all[uid][1] >= t)
        results["function_prediction"]["by_word_count_threshold"][f">={t}"] = {
            "accuracy_multi": {"mean": round(m_a, 4), "std": round(s_a, 4)},
            "f1_macro": round(m_f1m, 4), "f1_weighted": round(m_f1w, 4),
            "random_baseline": round(m_r2, 4), "freq_baseline": round(m_f, 4),
            "n_proteins_per_test_split": round(m_t2),
            "n_total_proteome": n_total_proteome,
            "proteome_coverage_pct": round(n_total_proteome / len(all_uids) * 100, 1),
        }

    with open(os.path.join(OUT_DIR, "VAL-DICT-001_holdout_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    md = []
    md.append("# VAL-DICT-001 v2: VALDICT001 Held-Out Validation Report\n")
    md.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    md.append(f"**Pipeline:** v2_6bit | **Seeds:** {SEEDS[0]}-{SEEDS[-1]} ({len(SEEDS)} splits)")
    md.append(f"**Scoring:** Multi-label (prediction correct if it matches ANY assigned department)\n")

    md.append("## Data Summary\n")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| Total proteins | {len(all_uids):,} |")
    md.append(f"| With ground truth | {proteins_with_gt:,} ({proteins_with_gt/len(all_uids)*100:.1f}%) |")
    md.append(f"| Single-label | {single_label_count:,} |")
    md.append(f"| Multi-label | {multi_label_count:,} (mean {mean_labels:.2f} labels/protein) |")
    md.append(f"| Vocabulary words | {len(vocab)} ({classified_vocab} classified) |")
    md.append(f"| Median vocab words/protein | {median_wc} |")
    md.append(f"| Departments | {len(all_depts)} |\n")

    md.append("## Part 1: Held-Out Vocabulary Correlation\n")
    md.append(f"**Pearson r = {mean_r:.4f} +/- {std_r:.4f}** (mean +/- std, {len(SEEDS)} seeds)\n")
    md.append(f"Per-seed: {', '.join(f'{r:.4f}' for r in rs)}\n")
    md.append(f"- Common words: {results['holdout_correlation']['mean_common_words']:.0f}")
    md.append(f"- Mean enrichment: train={results['holdout_correlation']['mean_enrichment_train']:.3f}, test={results['holdout_correlation']['mean_enrichment_test']:.3f}")
    md.append(f"- Reversed predictions: {results['holdout_correlation']['mean_reversed_predictions']:.1f}\n")

    md.append("## Part 2: Function Prediction Accuracy\n")
    md.append(f"**Overall (multi-label): {mean_acc_m:.4f} +/- {std_acc_m:.4f}**")
    md.append(f"**Overall (single-label): {mean_acc_s:.4f} +/- {std_acc_s:.4f}**\n")

    md.append("### By Word Count Threshold (with coverage)\n")
    md.append("| Min Words | Accuracy | Macro F1 | Weighted F1 | Random | Freq | n/split | Total | Coverage |")
    md.append("|-----------|----------|----------|-------------|--------|------|---------|-------|----------|")
    for t in WORD_THRESHOLDS:
        d = results["function_prediction"]["by_word_count_threshold"][f">={t}"]
        md.append(f"| >={t} | {d['accuracy_multi']['mean']:.4f}+/-{d['accuracy_multi']['std']:.4f} | {d['f1_macro']:.4f} | {d['f1_weighted']:.4f} | {d['random_baseline']:.4f} | {d['freq_baseline']:.4f} | ~{d['n_proteins_per_test_split']} | {d['n_total_proteome']:,} | {d['proteome_coverage_pct']}% |")

    md.append("\n### By Word Count Bin\n")
    md.append("| Bin | Accuracy (multi) | Accuracy (single) | Macro F1 | Random | Freq | n/split |")
    md.append("|-----|-----------------|-------------------|----------|--------|------|---------|")
    for label in BIN_LABELS:
        d = results["function_prediction"]["by_word_count_bin"][label]
        md.append(f"| {label} | {d['accuracy_multi']['mean']:.4f}+/-{d['accuracy_multi']['std']:.4f} | {d['accuracy_single']['mean']:.4f}+/-{d['accuracy_single']['std']:.4f} | {d['f1_macro']:.4f} | {d['random_baseline']:.4f} | {d['freq_baseline']:.4f} | ~{d['n_proteins_mean']} |")

    md.append("\n## Part 3: Baselines\n")
    md.append(f"- Random (proportional): {results['function_prediction']['baselines']['random_overall']:.4f}")
    md.append(f"- Frequency (always {most_common_dept}): {results['function_prediction']['baselines']['frequency_overall']:.4f}\n")

    md.append("## Part 4: Provenance\n")
    md.append("### Input File Hashes (SHA-256)\n")
    for name, h in hashes.items():
        md.append(f"- `{name}`: `{h}`")
    md.append(f"\n### Ground Truth Coverage\n")
    md.append(f"{proteins_with_gt:,} of {len(all_uids):,} proteins ({proteins_with_gt/len(all_uids)*100:.1f}%) had functional annotations.")
    md.append(f"Of these, {multi_label_count:,} ({multi_label_count/proteins_with_gt*100:.1f}%) have multiple department assignments (mean {mean_labels:.2f}).")
    md.append(f"Annotated proteins have more tokens on average ({mean_with:.1f} vs {mean_without:.1f}).\n")
    md.append("### Circularity Safeguard\n")
    md.append(results["circularity_note"])
    md.append("")

    with open(os.path.join(OUT_DIR, "VAL-DICT-001_holdout_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"COMPLETE in {elapsed:.0f}s")
    print(f"  r = {mean_r:.4f} +/- {std_r:.4f}")
    print(f"  accuracy (multi) = {mean_acc_m:.4f} +/- {std_acc_m:.4f}")
    print(f"  accuracy (single) = {mean_acc_s:.4f} +/- {std_acc_s:.4f}")
    for t in WORD_THRESHOLDS:
        d = results["function_prediction"]["by_word_count_threshold"][f">={t}"]
        print(f"  >={t} words: {d['accuracy_multi']['mean']:.4f}+/-{d['accuracy_multi']['std']:.4f} (n={d['n_total_proteome']:,}, {d['proteome_coverage_pct']}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
