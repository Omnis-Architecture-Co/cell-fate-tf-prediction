#!/usr/bin/env python3
"""
VAL-DICT-001 v5: Human-Only, Gene-Name Join, Confidence-Filtered GT
====================================================================
Builds on v4 with label-level confidence filtering:
  - Each department label for a protein is kept only if its source confidence >= threshold
  - Proteins left with zero labels after filtering are excluded from evaluation
  - Reports results at multiple confidence thresholds (0.0, 0.3, 0.5, 0.7)
  - Primary run uses >= 0.5 threshold

This removes low-confidence convergence-derived catch-all labels (primarily
Chromatin/Cytoskeleton from omnis_convergence source) while keeping all
expert-curated (api source, confidence=1.0) and high-quality heuristic labels.
"""

import csv
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(WORKSPACE, "validation")
PREFIX = "VAL-DICT-001_v5_filtered"

SEEDS = list(range(42, 52))
WORD_THRESHOLDS = [2, 5, 10, 20, 50]
WORD_COUNT_BINS = [(2, 3), (4, 10), (11, 50), (51, 999999)]
BIN_LABELS = ["2-3", "4-10", "11-50", "50+"]
CONF_THRESHOLDS = [0.0, 0.3, 0.5, 0.7]
PRIMARY_CONF_THRESHOLD = 0.5

BETA_DB_URL = os.environ.get("BETA_DATABASE_URL", "")

_conn = None

def get_conn():
    global _conn
    if _conn is None or _conn.closed:
        import psycopg2
        _conn = psycopg2.connect(BETA_DB_URL)
    return _conn

def query_db(sql, params=None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(sql, params or [])
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    cur.close()
    return rows


def load_human_entries():
    print("  Loading complete_human_proteome...")
    rows = query_db("SELECT entry FROM complete_human_proteome")
    entries = {r["entry"] for r in rows}
    print(f"    {len(entries):,} entries")
    return entries


def load_vocab():
    print("  Loading valdict_extended...")
    rows = query_db("""
        SELECT token_hex, primary_function, n_proteins, total_proteins, confidence
        FROM valdict_extended
        WHERE primary_function IS NOT NULL AND primary_function != 'Unclassified'
    """)
    vocab = {}
    for r in rows:
        h = r["token_hex"].strip().lower()
        vocab[h] = {
            "hex_norm": h,
            "function": r["primary_function"],
            "n_proteins": int(r["n_proteins"]),
            "total_proteins": float(r["total_proteins"]),
            "confidence": float(r["confidence"]),
            "enrichment": float(r["confidence"]) * 100,
        }
    print(f"    {len(vocab):,} classified words")
    return vocab


def load_tokens_human(human_entries):
    print("  Loading protein_tokens_v2 (human only)...")
    rows = query_db("SELECT uniprot_id, token_hex FROM protein_tokens_v2 ORDER BY uniprot_id, rank")
    pt = defaultdict(list)
    for r in rows:
        if r["uniprot_id"] in human_entries:
            pt[r["uniprot_id"]].append(r["token_hex"].strip().lower())
    print(f"    {len(pt):,} human proteins")
    return dict(pt)


def load_gene_map():
    print("  Loading gene_name map...")
    rows = query_db("SELECT uniprot_id, gene_name FROM protein_encoding_v2")
    return {r["uniprot_id"]: r["gene_name"] for r in rows}


def load_gt_with_confidence(uid_to_gene, human_uids):
    """Returns per-protein, per-department confidence: {uid: {dept: max_confidence}}"""
    print("  Loading gene_department_map with per-label confidence...")

    rows = query_db("""
        SELECT gene_name, primary_department, all_departments, confidence, source
        FROM gene_department_map
        WHERE primary_department IS NOT NULL
    """)

    gene_dept_conf = defaultdict(lambda: defaultdict(float))
    gene_dept_source = defaultdict(lambda: defaultdict(str))
    for r in rows:
        gn = r["gene_name"]
        conf = float(r["confidence"]) if r["confidence"] else 0.0
        source = r["source"] or ""
        depts = r["all_departments"] if r["all_departments"] else [r["primary_department"]]
        for d in depts:
            if conf > gene_dept_conf[gn][d]:
                gene_dept_conf[gn][d] = conf
                gene_dept_source[gn][d] = source

    rows2 = query_db("""
        SELECT uniprot_id, primary_department, all_departments, confidence, source
        FROM gene_department_map
        WHERE primary_department IS NOT NULL
    """)
    uid_dept_conf = defaultdict(lambda: defaultdict(float))
    uid_dept_source = defaultdict(lambda: defaultdict(str))
    for r in rows2:
        uid = r["uniprot_id"]
        conf = float(r["confidence"]) if r["confidence"] else 0.0
        source = r["source"] or ""
        depts = r["all_departments"] if r["all_departments"] else [r["primary_department"]]
        for d in depts:
            if conf > uid_dept_conf[uid][d]:
                uid_dept_conf[uid][d] = conf
                uid_dept_source[uid][d] = source

    human_set = set(human_uids)
    gt_full = {}
    for uid in human_uids:
        merged = defaultdict(float)
        merged_src = defaultdict(str)

        gene = uid_to_gene.get(uid)
        if gene and gene in gene_dept_conf:
            for d, c in gene_dept_conf[gene].items():
                if c > merged[d]:
                    merged[d] = c
                    merged_src[d] = gene_dept_source[gene][d]

        if uid in uid_dept_conf:
            for d, c in uid_dept_conf[uid].items():
                if c > merged[d]:
                    merged[d] = c
                    merged_src[d] = uid_dept_source[uid][d]

        if merged:
            gt_full[uid] = dict(merged)

    print(f"    {len(gt_full):,} proteins with any department labels")
    return gt_full


def filter_gt_by_confidence(gt_full, threshold):
    gt = {}
    for uid, dept_confs in gt_full.items():
        filtered = {d for d, c in dept_confs.items() if c >= threshold}
        if filtered:
            gt[uid] = filtered
    return gt


def load_canonical():
    rows = query_db("SELECT gene_name, uniprot_id FROM canonical_gene_uniprot")
    return {r["gene_name"]: r["uniprot_id"] for r in rows}


def pearson_r(xs, ys):
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


def predict_batch(uids, protein_tokens, vocab_set, vocab):
    results = {}
    for uid in uids:
        scores = defaultdict(float)
        nv = 0
        tc = Counter(protein_tokens.get(uid, []))
        for tok, cnt in tc.items():
            if tok in vocab_set:
                w = vocab[tok]
                nv += 1
                scores[w["function"]] += w["confidence"] * cnt
        pred = max(scores, key=scores.get) if scores else None
        results[uid] = (pred, nv)
    return results


def compute_enrichment(protein_ids, protein_tokens, vocab_set, vocab, gt_cache):
    dept_proteins = defaultdict(set)
    all_gt = set()
    for uid in protein_ids:
        depts = gt_cache.get(uid)
        if depts:
            for d in depts:
                dept_proteins[d].add(uid)
            all_gt.add(uid)
    total = len(all_gt)
    if total == 0:
        return {}
    dept_rates = {d: len(ps) / total for d, ps in dept_proteins.items()}

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
        cgt = carriers & all_gt
        if len(cgt) < 2:
            continue
        cin = cgt & dept_proteins.get(func, set())
        obs = len(cin) / len(cgt)
        exp = dept_rates.get(func, 0)
        enrichments[whex] = {"enrichment": obs / exp if exp > 0 else 0.0, "function": func}
    return enrichments


def run_seed(seed, all_uids, protein_tokens, vocab_set, vocab, gt_cache,
             dept_freq, most_common, all_depts):
    rng = random.Random(seed)
    shuffled = list(all_uids)
    rng.shuffle(shuffled)
    mid = len(shuffled) // 2
    train = set(shuffled[:mid])
    test = set(shuffled[mid:])

    print(f"  Seed {seed}: enrichment...", end="", flush=True)
    te = compute_enrichment(train, protein_tokens, vocab_set, vocab, gt_cache)
    ee = compute_enrichment(test, protein_tokens, vocab_set, vocab, gt_cache)
    common = set(te.keys()) & set(ee.keys())
    tv = [te[w]["enrichment"] for w in common]
    ev = [ee[w]["enrichment"] for w in common]
    r, p = pearson_r(tv, ev)
    rev = sum(1 for w in common if (te[w]["enrichment"] > 1) != (ee[w]["enrichment"] > 1))

    print(f" pred...", end="", flush=True)
    test_uids = list(test)
    preds = predict_batch(test_uids, protein_tokens, vocab_set, vocab)
    dw = [dept_freq.get(d, 0) for d in all_depts]

    bc = {l: 0 for l in BIN_LABELS}
    bt = {l: 0 for l in BIN_LABELS}
    btp = {l: defaultdict(int) for l in BIN_LABELS}
    bfp = {l: defaultdict(int) for l in BIN_LABELS}
    bfn = {l: defaultdict(int) for l in BIN_LABELS}
    brc = {l: 0 for l in BIN_LABELS}
    bfc = {l: 0 for l in BIN_LABELS}

    tc2 = {t: 0 for t in WORD_THRESHOLDS}
    tt = {t: 0 for t in WORD_THRESHOLDS}
    trc = {t: 0 for t in WORD_THRESHOLDS}
    tfc = {t: 0 for t in WORD_THRESHOLDS}
    ttp = {t: defaultdict(int) for t in WORD_THRESHOLDS}
    tfp = {t: defaultdict(int) for t in WORD_THRESHOLDS}
    tfn = {t: defaultdict(int) for t in WORD_THRESHOLDS}

    cm = defaultdict(lambda: defaultdict(int))
    oc, ot, orc, ofc = 0, 0, 0, 0

    for uid in test_uids:
        td = gt_cache.get(uid)
        if not td:
            continue
        pr, nw = preds[uid]
        if pr is None:
            continue

        ok = pr in td
        primary = sorted(td)[0]
        rd = random.Random(seed * 100000 + hash(uid)).choices(all_depts, weights=dw)[0]
        rok = rd in td
        fok = most_common in td

        for i, (lo, hi) in enumerate(WORD_COUNT_BINS):
            if lo <= nw <= hi:
                lb = BIN_LABELS[i]
                bt[lb] += 1; bc[lb] += ok
                if ok: btp[lb][pr] += 1
                else:
                    bfp[lb][pr] += 1
                    for x in td: bfn[lb][x] += 1
                brc[lb] += rok; bfc[lb] += fok
                break

        for th in WORD_THRESHOLDS:
            if nw >= th:
                tt[th] += 1; tc2[th] += ok; trc[th] += rok; tfc[th] += fok
                if ok: ttp[th][pr] += 1
                else:
                    tfp[th][pr] += 1
                    for x in td: tfn[th][x] += 1

        cm[primary][pr] += 1
        ot += 1; oc += ok; orc += rok; ofc += fok

    def f1s(tp_d, fp_d, fn_d):
        cls = set(tp_d) | set(fp_d) | set(fn_d)
        f1l, sl = [], []
        for c in cls:
            tp, fp, fn = tp_d.get(c, 0), fp_d.get(c, 0), fn_d.get(c, 0)
            pr2 = tp / (tp + fp) if tp + fp else 0
            re = tp / (tp + fn) if tp + fn else 0
            f1l.append(2 * pr2 * re / (pr2 + re) if pr2 + re else 0)
            sl.append(tp + fn)
        macro = sum(f1l) / len(f1l) if f1l else 0
        ts = sum(sl)
        weighted = sum(a * b for a, b in zip(f1l, sl)) / ts if ts else 0
        return macro, weighted

    acc = oc / ot if ot else 0
    print(f" r={r:.4f} acc={acc:.4f} (n={ot})")

    res = {
        "seed": seed, "common_words": len(common),
        "pearson_r": r, "p_value": p, "reversed": rev,
        "bin_acc": {l: bc[l] / bt[l] if bt[l] else 0 for l in BIN_LABELS},
        "bin_n": {l: bt[l] for l in BIN_LABELS},
        "bin_f1m": {l: f1s(btp[l], bfp[l], bfn[l])[0] for l in BIN_LABELS},
        "bin_f1w": {l: f1s(btp[l], bfp[l], bfn[l])[1] for l in BIN_LABELS},
        "bin_rand": {l: brc[l] / bt[l] if bt[l] else 0 for l in BIN_LABELS},
        "bin_freq": {l: bfc[l] / bt[l] if bt[l] else 0 for l in BIN_LABELS},
        "overall_acc": acc,
        "overall_rand": orc / ot if ot else 0,
        "overall_freq": ofc / ot if ot else 0,
        "overall_n": ot,
        "thresh_acc": {str(t): tc2[t] / tt[t] if tt[t] else 0 for t in WORD_THRESHOLDS},
        "thresh_n": {str(t): tt[t] for t in WORD_THRESHOLDS},
        "thresh_rand": {str(t): trc[t] / tt[t] if tt[t] else 0 for t in WORD_THRESHOLDS},
        "thresh_freq": {str(t): tfc[t] / tt[t] if tt[t] else 0 for t in WORD_THRESHOLDS},
        "thresh_f1m": {str(t): f1s(ttp[t], tfp[t], tfn[t])[0] for t in WORD_THRESHOLDS},
        "thresh_f1w": {str(t): f1s(ttp[t], tfp[t], tfn[t])[1] for t in WORD_THRESHOLDS},
        "confusion": {k: dict(v) for k, v in cm.items()},
    }
    if seed == 42:
        res["train_e"] = {w: te[w]["enrichment"] for w in common}
        res["test_e"] = {w: ee[w]["enrichment"] for w in common}
    return res


def make_scatter(train_e, test_e, r_val, path):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        words = sorted(train_e)
        xs, ys = [train_e[w] for w in words], [test_e[w] for w in words]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(xs, ys, alpha=0.3, s=8, c="#2563eb", edgecolors="none")
        mx = min(max(max(xs), max(ys)) * 1.05, 50)
        ax.plot([0, mx], [0, mx], "k--", alpha=0.3)
        ax.set_xlabel("Training Enrichment"); ax.set_ylabel("Test Enrichment")
        ax.set_title(f"VALDICT001 Human, Conf>=0.5 (r={r_val:.3f}, n={len(words):,})")
        ax.set_xlim(0, mx); ax.set_ylim(0, mx); ax.set_aspect("equal")
        ax.text(0.05, 0.95, f"n={len(words):,}\nr={r_val:.3f}", transform=ax.transAxes,
                fontsize=11, va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  Warning: scatter failed: {e}")


def make_accuracy_curve(all_uids, protein_tokens, vocab_set, vocab, gt_cache, path):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        preds = predict_batch(all_uids, protein_tokens, vocab_set, vocab)
        data = []
        for uid in all_uids:
            td = gt_cache.get(uid)
            if not td: continue
            pr, nw = preds[uid]
            if pr is None or nw < 1: continue
            data.append((nw, 1 if pr in td else 0))
        data.sort()
        ths = sorted(set(wc for wc, _ in data))
        cx, cy, cn = [], [], []
        for t in ths:
            sub = [(w, c) for w, c in data if w >= t]
            if len(sub) < 5: continue
            cx.append(t); cy.append(sum(c for _, c in sub) / len(sub)); cn.append(len(sub))

        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [3, 1]})
        a1.plot(cx, cy, color="#2563eb", linewidth=2)
        a1.set_ylabel("Accuracy (multi-label)"); a1.set_ylim(0, 1.05)
        a1.set_title("VALDICT001 Human, Conf>=0.5: Accuracy vs Min Words")
        a1.axhline(y=0.5, color="gray", ls="--", alpha=0.3); a1.grid(True, alpha=0.2)
        for t in WORD_THRESHOLDS:
            sub = [(w, c) for w, c in data if w >= t]
            if sub:
                a = sum(c for _, c in sub) / len(sub)
                a1.axvline(x=t, color="orange", ls=":", alpha=0.5)
                a1.annotate(f">={t}: {a:.1%} (n={len(sub):,})", xy=(t, a), fontsize=8,
                            xytext=(t+1, a+0.03),
                            bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.8))
        a2.plot(cx, cn, color="#dc2626", lw=1.5)
        a2.set_xlabel("Min Word Count"); a2.set_ylabel("Sample Size"); a2.set_yscale("log")
        a2.grid(True, alpha=0.2); a2.set_xlim(a1.get_xlim())
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  Warning: curve failed: {e}")


def ms(vals):
    m = sum(vals) / len(vals) if vals else 0
    s = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 0
    return m, s


def run_at_threshold(conf_thresh, all_uids, protein_tokens, vocab_set, vocab, gt_full,
                     canonical, uid_to_gene, full_run=False):
    gt = filter_gt_by_confidence(gt_full, conf_thresh)
    n_with_gt = sum(1 for u in all_uids if u in gt)
    multi = sum(1 for u in all_uids if u in gt and len(gt[u]) > 1)
    single = n_with_gt - multi
    mean_lab = sum(len(gt[u]) for u in all_uids if u in gt) / n_with_gt if n_with_gt else 0

    all_dept_labels = set()
    for ds in gt.values():
        all_dept_labels.update(ds)
    dc = Counter()
    for u in all_uids:
        if u in gt:
            for d in gt[u]: dc[d] += 1
    all_depts = sorted(all_dept_labels)
    total_d = sum(dc.values())
    df = {d: c / total_d for d, c in dc.items()}
    mc = dc.most_common(1)[0][0] if dc else "?"
    mc_pct = df.get(mc, 0)

    preds_all = predict_batch(all_uids, protein_tokens, vocab_set, vocab)
    wcs = sorted(preds_all[u][1] for u in all_uids)
    median_wc = wcs[len(wcs) // 2]
    zero_w = sum(1 for w in wcs if w == 0)

    summary = {
        "conf_threshold": conf_thresh,
        "proteins_with_gt": n_with_gt,
        "gt_coverage_pct": round(n_with_gt / len(all_uids) * 100, 1),
        "single_label": single, "multi_label": multi,
        "mean_labels": round(mean_lab, 2),
        "departments": len(all_depts),
        "most_common": mc, "most_common_pct": round(mc_pct * 100, 1),
        "dept_top5": [(d, c, round(c/total_d*100, 1)) for d, c in dc.most_common(5)],
    }

    print(f"\n  conf>={conf_thresh}: GT={n_with_gt:,} ({n_with_gt/len(all_uids)*100:.1f}%), "
          f"depts={len(all_depts)}, top={mc} ({mc_pct:.1%}), "
          f"single={single:,} multi={multi:,} mean={mean_lab:.2f}")

    if not full_run:
        seed_results = []
        for seed in [42, 43, 44]:
            res = run_seed(seed, all_uids, protein_tokens, vocab_set, vocab, gt,
                           df, mc, all_depts)
            seed_results.append(res)
        rs = [s["pearson_r"] for s in seed_results]
        accs = [s["overall_acc"] for s in seed_results]
        mr, sr = ms(rs)
        ma, sa = ms(accs)
        summary["pearson_r"] = round(mr, 4)
        summary["accuracy"] = round(ma, 4)
        summary["freq_baseline"] = round(sum(s["overall_freq"] for s in seed_results) / len(seed_results), 4)
        summary["rand_baseline"] = round(sum(s["overall_rand"] for s in seed_results) / len(seed_results), 4)
        summary["n_seeds"] = 3
        for t in WORD_THRESHOLDS:
            ta = [s["thresh_acc"][str(t)] for s in seed_results]
            m, _ = ms(ta)
            summary[f"acc_ge{t}"] = round(m, 4)
        return summary, None, None, None

    print(f"\n  Running full {len(SEEDS)} seeds at conf>={conf_thresh}...")
    seed_results = []
    for seed in SEEDS:
        res = run_seed(seed, all_uids, protein_tokens, vocab_set, vocab, gt,
                       df, mc, all_depts)
        seed_results.append(res)
    return summary, seed_results, gt, {
        "all_depts": all_depts, "dept_freq": df, "most_common": mc,
        "dept_counts": dc, "preds_all": preds_all, "median_wc": median_wc,
        "zero_words": zero_w,
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("VAL-DICT-001 v5: Human-Only, Gene-Join, Confidence-Filtered GT")
    print("=" * 70)
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")

    print("\n[1] Loading data...")
    human_entries = load_human_entries()
    vocab = load_vocab()
    vocab_set = set(vocab.keys())
    protein_tokens = load_tokens_human(human_entries)
    all_uids = sorted(protein_tokens.keys())
    uid_to_gene = load_gene_map()
    gt_full = load_gt_with_confidence(uid_to_gene, all_uids)
    canonical = load_canonical()

    print(f"\n  Base data: {len(all_uids):,} human proteins, {len(vocab):,} vocab words")
    print(f"  Proteins with any GT: {len(gt_full):,}")

    print("\n[2] Confidence threshold sweep (3 seeds each)...")
    sweep_results = {}
    for ct in CONF_THRESHOLDS:
        summary, _, _, _ = run_at_threshold(ct, all_uids, protein_tokens, vocab_set, vocab,
                                            gt_full, canonical, uid_to_gene, full_run=False)
        sweep_results[ct] = summary

    print(f"\n  === SWEEP SUMMARY ===")
    print(f"  {'Conf':>5} | {'GT':>7} | {'Cov%':>5} | {'Depts':>5} | {'Top Dept':>20} | {'Top%':>5} | {'Acc':>6} | {'Freq':>6} | {'Acc/Freq':>8}")
    for ct in CONF_THRESHOLDS:
        s = sweep_results[ct]
        ratio = s["accuracy"] / s["freq_baseline"] if s["freq_baseline"] > 0 else 0
        print(f"  {ct:>5.1f} | {s['proteins_with_gt']:>7,} | {s['gt_coverage_pct']:>5.1f} | {s['departments']:>5} | "
              f"{s['most_common']:>20} | {s['most_common_pct']:>5.1f} | {s['accuracy']:>6.4f} | {s['freq_baseline']:>6.4f} | {ratio:>8.3f}")

    print(f"\n[3] Full run at conf >= {PRIMARY_CONF_THRESHOLD}...")
    summary, seed_results, gt_cache, meta = run_at_threshold(
        PRIMARY_CONF_THRESHOLD, all_uids, protein_tokens, vocab_set, vocab,
        gt_full, canonical, uid_to_gene, full_run=True)

    rs = [s["pearson_r"] for s in seed_results]
    mr, sr = ms(rs)
    accs = [s["overall_acc"] for s in seed_results]
    ma, sa = ms(accs)
    all_depts = meta["all_depts"]
    df = meta["dept_freq"]
    mc = meta["most_common"]
    dc = meta["dept_counts"]
    preds_all = meta["preds_all"]

    print(f"\n  CORRELATION: r = {mr:.4f} +/- {sr:.4f}")
    print(f"  ACCURACY: {ma:.4f} +/- {sa:.4f}")
    print(f"\n  THRESHOLD RESULTS:")
    for t in WORD_THRESHOLDS:
        ta = [s["thresh_acc"][str(t)] for s in seed_results]
        tn = [s["thresh_n"][str(t)] for s in seed_results]
        tr = [s["thresh_rand"][str(t)] for s in seed_results]
        tf = [s["thresh_freq"][str(t)] for s in seed_results]
        m_a, s_a = ms(ta)
        m_n, _ = ms(tn)
        m_r, _ = ms(tr)
        m_f, _ = ms(tf)
        print(f"    >={t}: acc={m_a:.4f}+/-{s_a:.4f} (n~{m_n:.0f}) | rand={m_r:.4f} freq={m_f:.4f} | lift={m_a/m_f:.2f}x" if m_f > 0 else f"    >={t}: acc={m_a:.4f}+/-{s_a:.4f} (n~{m_n:.0f})")

    print(f"\n[4] Well-known proteins (conf>={PRIMARY_CONF_THRESHOLD})...")
    for gene in ["BRCA1", "TP53", "EGFR", "KRAS", "MYC", "INS", "APOE", "PTEN", "RB1", "CDK2"]:
        cu = canonical.get(gene)
        if cu and cu in protein_tokens:
            pr, nw = preds_all[cu]
            true = gt_cache.get(cu, set())
            m = "CORRECT" if pr and pr in true else ("NO_GT" if not true else "WRONG")
            gt_detail = gt_full.get(cu, {})
            confs = {d: gt_detail.get(d, 0) for d in true} if true else {}
            print(f"    {gene} ({cu}): {nw} words, pred={pr}, true={true}, {m}")
            if confs:
                print(f"      label confs: {confs}")
        else:
            found = [u for u in all_uids if (uid_to_gene.get(u) or "").split()[:1] == [gene]]
            if found:
                uid = found[0]
                pr, nw = preds_all[uid]
                true = gt_cache.get(uid, set())
                m = "CORRECT" if pr and pr in true else ("NO_GT" if not true else "WRONG")
                print(f"    {gene} ({uid} via gene): {nw} words, pred={pr}, true={true}, {m}")
            else:
                print(f"    {gene}: not found")

    print("\n[5] Generating plots...")
    s42 = next((s for s in seed_results if s["seed"] == 42), seed_results[0])
    if s42.get("train_e"):
        make_scatter(s42["train_e"], s42["test_e"], s42["pearson_r"],
                     os.path.join(OUT_DIR, f"{PREFIX}_enrichment_scatter.png"))
    make_accuracy_curve(all_uids, protein_tokens, vocab_set, vocab, gt_cache,
                        os.path.join(OUT_DIR, f"{PREFIX}_accuracy_curve.png"))

    print("\n[6] Confusion matrix...")
    merged = defaultdict(lambda: defaultdict(int))
    for s in seed_results:
        for td, ps in s["confusion"].items():
            for pd, c in ps.items():
                merged[td][pd] += c
    cmd = sorted(set(list(merged.keys()) + [d for v in merged.values() for d in v]))
    with open(os.path.join(OUT_DIR, f"{PREFIX}_confusion_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\predicted"] + cmd)
        for td in cmd:
            w.writerow([td] + [merged[td].get(pd, 0) for pd in cmd])
    print(f"  Saved confusion matrix")

    print("\n[7] Building output files...")

    v2p = os.path.join(OUT_DIR, "VAL-DICT-001_holdout_results.json")
    v3p = os.path.join(OUT_DIR, "VAL-DICT-001_v3_beta_results.json")
    v4p = os.path.join(OUT_DIR, "VAL-DICT-001_v4_human_results.json")
    prev = {}
    for label, path, rkey, akey in [
        ("v2_csv", v2p, "holdout_correlation.mean_pearson_r", "function_prediction.overall_accuracy_multi_mean"),
        ("v3_beta_uid", v3p, "holdout_correlation.mean_pearson_r", "function_prediction.overall_accuracy_mean"),
        ("v4_human_gene", v4p, "holdout_correlation.mean_pearson_r", "function_prediction.overall_accuracy_mean"),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            def get_nested(obj, key):
                for k in key.split("."):
                    obj = obj.get(k, {})
                return obj
            prev[label] = {
                "pearson_r": get_nested(d, rkey),
                "accuracy": get_nested(d, akey),
                "gt_coverage_pct": d.get("data_summary", {}).get("gt_coverage_pct",
                                   d.get("data_summary", {}).get("ground_truth_coverage_pct", "?")),
            }

    n_with_gt = sum(1 for u in all_uids if u in gt_cache)
    multi = sum(1 for u in all_uids if u in gt_cache and len(gt_cache[u]) > 1)
    single = n_with_gt - multi
    mean_lab = sum(len(gt_cache[u]) for u in all_uids if u in gt_cache) / n_with_gt if n_with_gt else 0

    results = {
        "validation_id": "VAL-DICT-001",
        "version": "v5_filtered",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "v2_6bit",
        "species_filter": "Homo sapiens",
        "gt_join": "gene_name via protein_encoding_v2",
        "gt_confidence_threshold": PRIMARY_CONF_THRESHOLD,
        "seeds": SEEDS, "n_seeds": len(SEEDS),
        "data_summary": {
            "human_proteins": len(all_uids),
            "proteins_with_gt": n_with_gt,
            "gt_coverage_pct": round(n_with_gt / len(all_uids) * 100, 1),
            "single_label": single, "multi_label": multi,
            "mean_labels": round(mean_lab, 2),
            "vocabulary_size": len(vocab),
            "departments": len(all_depts),
            "most_common_dept": mc,
            "most_common_pct": round(df[mc] * 100, 1),
            "median_vocab_words": meta["median_wc"],
            "zero_word_proteins": meta["zero_words"],
        },
        "confidence_sweep": {str(ct): sweep_results[ct] for ct in CONF_THRESHOLDS},
        "holdout_correlation": {
            "mean_pearson_r": round(mr, 4), "std_pearson_r": round(sr, 4),
            "per_seed_r": [round(r, 4) for r in rs],
        },
        "function_prediction": {
            "scoring": "multi-label",
            "overall_accuracy_mean": round(ma, 4), "overall_accuracy_std": round(sa, 4),
            "overall_random": round(sum(s["overall_rand"] for s in seed_results) / len(seed_results), 4),
            "overall_freq": round(sum(s["overall_freq"] for s in seed_results) / len(seed_results), 4),
            "by_threshold": {}, "by_bin": {},
        },
        "comparison": prev,
    }

    for t in WORD_THRESHOLDS:
        ta = [s["thresh_acc"][str(t)] for s in seed_results]
        tn = [s["thresh_n"][str(t)] for s in seed_results]
        tr = [s["thresh_rand"][str(t)] for s in seed_results]
        tf = [s["thresh_freq"][str(t)] for s in seed_results]
        tf1m = [s["thresh_f1m"][str(t)] for s in seed_results]
        tf1w = [s["thresh_f1w"][str(t)] for s in seed_results]
        m_a, s_a = ms(ta); m_n, _ = ms(tn); m_r, _ = ms(tr); m_f, _ = ms(tf)
        m_f1m, _ = ms(tf1m); m_f1w, _ = ms(tf1w)
        n_tot = sum(1 for u in all_uids if preds_all[u][1] >= t)
        results["function_prediction"]["by_threshold"][f">={t}"] = {
            "accuracy": {"mean": round(m_a, 4), "std": round(s_a, 4)},
            "f1_macro": round(m_f1m, 4), "f1_weighted": round(m_f1w, 4),
            "random": round(m_r, 4), "freq": round(m_f, 4),
            "lift_over_freq": round(m_a / m_f, 2) if m_f > 0 else 0,
            "lift_over_random": round(m_a / m_r, 2) if m_r > 0 else 0,
            "n_per_split": round(m_n), "n_total": n_tot,
            "coverage_pct": round(n_tot / len(all_uids) * 100, 1),
        }

    for lb in BIN_LABELS:
        ba = [s["bin_acc"][lb] for s in seed_results]
        bn = [s["bin_n"][lb] for s in seed_results]
        bf1m = [s["bin_f1m"][lb] for s in seed_results]
        bf1w = [s["bin_f1w"][lb] for s in seed_results]
        bra = [s["bin_rand"][lb] for s in seed_results]
        bfa = [s["bin_freq"][lb] for s in seed_results]
        m_a, s_a = ms(ba); m_n, _ = ms(bn)
        m_f1m, _ = ms(bf1m); m_f1w, _ = ms(bf1w)
        m_r, _ = ms(bra); m_f, _ = ms(bfa)
        results["function_prediction"]["by_bin"][lb] = {
            "accuracy": {"mean": round(m_a, 4), "std": round(s_a, 4)},
            "f1_macro": round(m_f1m, 4), "f1_weighted": round(m_f1w, 4),
            "random": round(m_r, 4), "freq": round(m_f, 4),
            "n_per_split": round(m_n),
        }

    with open(os.path.join(OUT_DIR, f"{PREFIX}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    md = []
    md.append("# VAL-DICT-001 v5: Confidence-Filtered Validation Report\n")
    md.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    md.append(f"**Species:** Homo sapiens only")
    md.append(f"**GT Join:** gene_name (protein_encoding_v2 -> gene_department_map)")
    md.append(f"**GT Confidence:** >= {PRIMARY_CONF_THRESHOLD}")
    md.append(f"**Seeds:** {SEEDS[0]}-{SEEDS[-1]} ({len(SEEDS)} splits)")
    md.append(f"**Scoring:** Multi-label\n")

    md.append("## Confidence Threshold Sweep\n")
    md.append("| Conf >= | GT proteins | Coverage | Depts | Top Dept | Top % | Accuracy | Freq Baseline | Lift |")
    md.append("|---------|-------------|----------|-------|----------|-------|----------|---------------|------|")
    for ct in CONF_THRESHOLDS:
        s = sweep_results[ct]
        lift = s["accuracy"] / s["freq_baseline"] if s["freq_baseline"] > 0 else 0
        md.append(f"| {ct:.1f} | {s['proteins_with_gt']:,} | {s['gt_coverage_pct']}% | {s['departments']} | {s['most_common']} | {s['most_common_pct']}% | {s['accuracy']:.4f} | {s['freq_baseline']:.4f} | {lift:.2f}x |")

    md.append(f"\n## Why Confidence Filtering Matters\n")
    md.append("The `gene_department_map` has three sources with very different quality:")
    md.append("- **api** (25,580 rows): UniProt-curated, confidence=1.0")
    md.append("- **omnis_convergence** (15,154 rows): Algorithm-derived, avg confidence=0.36")
    md.append("- **heuristic** (946 rows): Rule-based, avg confidence=0.70")
    md.append("")
    md.append("Without filtering, Chromatin (conf=0.35 median) and Cytoskeleton (conf=0.37 median)")
    md.append("dominate as low-confidence catch-all labels, inflating the frequency baseline to ~49%")
    md.append("and making it impossible for any classifier to beat random-by-frequency.\n")

    md.append("## Four-Way Comparison\n")
    md.append("| Metric | v2 (CSV) | v3 (Beta, uid) | v4 (Human, gene) | v5 (Human, gene, conf>=0.5) |")
    md.append("|--------|----------|----------------|------------------|----------------------------|")
    v2r = prev.get("v2_csv", {}); v3r = prev.get("v3_beta_uid", {}); v4r = prev.get("v4_human_gene", {})
    md.append(f"| Vocabulary | 1,932 | 55,641 | 55,641 | {len(vocab):,} |")
    md.append(f"| Species | All | All | Human | Human |")
    md.append(f"| GT join | gene (CSV) | uid | gene | gene |")
    md.append(f"| GT conf filter | none | none | none | >= {PRIMARY_CONF_THRESHOLD} |")
    md.append(f"| GT coverage | {v2r.get('gt_coverage_pct','?')}% | {v3r.get('gt_coverage_pct','?')}% | {v4r.get('gt_coverage_pct','?')}% | {n_with_gt/len(all_uids)*100:.1f}% |")
    md.append(f"| Pearson r | {v2r.get('pearson_r','?')} | {v3r.get('pearson_r','?')} | {v4r.get('pearson_r','?')} | {mr:.4f} |")
    md.append(f"| Overall accuracy | {v2r.get('accuracy','?')} | {v3r.get('accuracy','?')} | {v4r.get('accuracy','?')} | {ma:.4f} |")
    md.append(f"| Departments | 22 | 32 | 32 | {len(all_depts)} |\n")

    md.append(f"## Data Summary (conf >= {PRIMARY_CONF_THRESHOLD})\n")
    md.append(f"| Metric | Value |")
    md.append(f"|--------|-------|")
    md.append(f"| Human proteins | {len(all_uids):,} |")
    md.append(f"| With GT (conf>={PRIMARY_CONF_THRESHOLD}) | {n_with_gt:,} ({n_with_gt/len(all_uids)*100:.1f}%) |")
    md.append(f"| Single-label | {single:,} | Multi-label | {multi:,} (mean {mean_lab:.2f}) |")
    md.append(f"| Vocabulary | {len(vocab):,} classified words |")
    md.append(f"| Departments | {len(all_depts)} |")
    md.append(f"| Most common | {mc} ({df[mc]:.1%}) |\n")

    md.append("### Department Distribution\n")
    md.append("| Dept | Count | % |")
    md.append("|------|-------|---|")
    total_d = sum(dc.values())
    for d, c in dc.most_common():
        md.append(f"| {d} | {c:,} | {c/total_d*100:.1f}% |")

    md.append(f"\n## Part 1: Held-Out Correlation\n")
    md.append(f"**Pearson r = {mr:.4f} +/- {sr:.4f}**\n")
    md.append(f"Per-seed: {', '.join(f'{r:.4f}' for r in rs)}\n")

    md.append("## Part 2: Function Prediction by Threshold\n")
    md.append("| Min Words | Accuracy | F1 Macro | F1 Weighted | Random | Freq | Lift/Freq | Lift/Rand | n/split | Total | Coverage |")
    md.append("|-----------|----------|----------|-------------|--------|------|-----------|-----------|---------|-------|----------|")
    for t in WORD_THRESHOLDS:
        d = results["function_prediction"]["by_threshold"][f">={t}"]
        md.append(f"| >={t} | {d['accuracy']['mean']:.4f}+/-{d['accuracy']['std']:.4f} | {d['f1_macro']:.4f} | {d['f1_weighted']:.4f} | {d['random']:.4f} | {d['freq']:.4f} | {d['lift_over_freq']:.2f}x | {d['lift_over_random']:.2f}x | ~{d['n_per_split']} | {d['n_total']:,} | {d['coverage_pct']}% |")

    md.append("\n### By Word Count Bin\n")
    md.append("| Bin | Accuracy | F1 Macro | Random | Freq | n/split |")
    md.append("|-----|----------|----------|--------|------|---------|")
    for lb in BIN_LABELS:
        d = results["function_prediction"]["by_bin"][lb]
        md.append(f"| {lb} | {d['accuracy']['mean']:.4f}+/-{d['accuracy']['std']:.4f} | {d['f1_macro']:.4f} | {d['random']:.4f} | {d['freq']:.4f} | ~{d['n_per_split']} |")

    md.append(f"\n## Baselines\n")
    md.append(f"- Random (proportional): {results['function_prediction']['overall_random']:.4f}")
    md.append(f"- Frequency (always {mc}): {results['function_prediction']['overall_freq']:.4f}\n")

    md.append("## Methodology Notes\n")
    md.append("- **Confidence filtering** removes low-quality GT labels at the label level, not protein level")
    md.append("- A protein with Chromatin (conf=0.3) + DNA repair (conf=0.9) keeps only DNA repair at conf>=0.5")
    md.append("- This is standard ML practice: do not evaluate against uncertain ground truth")
    md.append("- All filtering is transparent and reproducible via the confidence threshold parameter\n")

    md.append("## Provenance\n")
    md.append("- **Source:** Beta database (BETA_DATABASE_URL)")
    md.append(f"- **Tables:** valdict_extended, gene_department_map (conf>={PRIMARY_CONF_THRESHOLD}), protein_tokens_v2 (human), complete_human_proteome")
    md.append("")

    with open(os.path.join(OUT_DIR, f"{PREFIX}_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"COMPLETE in {elapsed:.0f}s")
    print(f"  Species: Human only ({len(all_uids):,})")
    print(f"  GT conf >= {PRIMARY_CONF_THRESHOLD}: {n_with_gt:,} ({n_with_gt/len(all_uids)*100:.1f}%)")
    print(f"  r = {mr:.4f} +/- {sr:.4f}")
    print(f"  accuracy = {ma:.4f} +/- {sa:.4f}")
    for t in WORD_THRESHOLDS:
        d = results["function_prediction"]["by_threshold"][f">={t}"]
        print(f"  >={t}: {d['accuracy']['mean']:.4f}+/-{d['accuracy']['std']:.4f} (n={d['n_total']:,}, {d['coverage_pct']}%) lift={d['lift_over_freq']:.2f}x freq")
    print(f"{'='*70}")

    try: get_conn().close()
    except: pass


if __name__ == "__main__":
    main()
