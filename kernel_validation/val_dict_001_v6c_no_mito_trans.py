#!/usr/bin/env python3
"""
VAL-DICT-001 v6c: Mitochondrial + Transcription Excluded
=========================================================
Peels back two dominant attractor departments to reveal
vocabulary discrimination across remaining 30 departments.

Documents the layered attractor pattern:
  Layer 1: Mitochondrial (metabolic/membrane catch-all)
  Layer 2: Transcription (nuclear/regulatory catch-all)

This progressive filtering reveals what signal remains underneath.
"""

import json, math, os, random, time, csv
from collections import Counter, defaultdict
from datetime import datetime, timezone

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(WORKSPACE, "validation")
PREFIX = "VAL-DICT-001_v6c_noMitoTrans"
SEEDS = list(range(42, 52))
WORD_THRESHOLDS = [2, 5, 10, 20, 50]
CONF_THRESHOLD = 0.5
EXCLUDED_DEPTS = {"Mitochondrial", "Transcription"}

FUNCTIONAL_GROUPS = {
    "Chromatin": "gene_regulation",
    "DNA repair": "genome_maintenance", "DNA replication": "genome_maintenance",
    "Cell cycle": "genome_maintenance",
    "Lipid metabolism": "metabolism", "Glycosylation": "metabolism", "Methylation": "metabolism",
    "Signaling": "signaling", "Receptor signaling": "signaling",
    "Kinase": "signaling", "Phosphatase": "signaling", "GTPase": "signaling",
    "Apoptosis": "cell_fate", "Autophagy": "cell_fate",
    "Cell adhesion": "structure", "Cytoskeleton": "structure", "Structural": "structure",
    "Immune response": "immune",
    "Proteolysis": "protein_processing", "Ubiquitin": "protein_processing",
    "Protein folding": "protein_processing", "Translation": "protein_processing",
    "Vesicle trafficking": "transport", "Nuclear transport": "transport",
    "Transport": "transport", "Ion channel": "transport",
    "RNA processing": "rna", "Nuc acid bind": "rna",
    "Olfactory": "sensory",
}

_conn = None
def get_conn():
    global _conn
    if _conn is None or _conn.closed:
        import psycopg2
        _conn = psycopg2.connect(os.environ["BETA_DATABASE_URL"])
    return _conn

def qdb(sql, params=None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(sql, params or [])
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    cur.close()
    return rows

def load_data():
    print("[1] Loading data...")
    human = {r["entry"] for r in qdb("SELECT entry FROM complete_human_proteome")}

    vrows = qdb("""SELECT token_hex, primary_function, confidence
        FROM valdict_extended WHERE primary_function IS NOT NULL AND primary_function != 'Unclassified'""")
    vocab_full = {}
    vocab_filtered = {}
    excluded_counts = Counter()
    for r in vrows:
        h = r["token_hex"].strip().lower()
        entry = {"function": r["primary_function"], "confidence": float(r["confidence"])}
        vocab_full[h] = entry
        if r["primary_function"] not in EXCLUDED_DEPTS:
            vocab_filtered[h] = entry
        else:
            excluded_counts[r["primary_function"]] += 1

    print(f"  Vocab: {len(vocab_full):,} total, {len(vocab_filtered):,} after excluding {EXCLUDED_DEPTS}")
    for d, c in excluded_counts.most_common():
        print(f"    Removed {c:,} {d} tokens")

    trows = qdb("SELECT uniprot_id, token_hex FROM protein_tokens_v2 ORDER BY uniprot_id, rank")
    ptokens = defaultdict(list)
    for r in trows:
        if r["uniprot_id"] in human:
            ptokens[r["uniprot_id"]].append(r["token_hex"].strip().lower())
    ptokens = dict(ptokens)
    all_uids = sorted(ptokens.keys())

    gmap = {r["uniprot_id"]: r["gene_name"] for r in qdb("SELECT uniprot_id, gene_name FROM protein_encoding_v2")}

    gdrows = qdb("SELECT gene_name, primary_department, all_departments, confidence FROM gene_department_map WHERE primary_department IS NOT NULL")
    gene_dc = defaultdict(lambda: defaultdict(float))
    for r in gdrows:
        conf = float(r["confidence"]) if r["confidence"] else 0.0
        for d in (r["all_departments"] if r["all_departments"] else [r["primary_department"]]):
            if d not in EXCLUDED_DEPTS:
                gene_dc[r["gene_name"]][d] = max(gene_dc[r["gene_name"]][d], conf)

    udrows = qdb("SELECT uniprot_id, primary_department, all_departments, confidence FROM gene_department_map WHERE primary_department IS NOT NULL")
    uid_dc = defaultdict(lambda: defaultdict(float))
    for r in udrows:
        conf = float(r["confidence"]) if r["confidence"] else 0.0
        for d in (r["all_departments"] if r["all_departments"] else [r["primary_department"]]):
            if d not in EXCLUDED_DEPTS:
                uid_dc[r["uniprot_id"]][d] = max(uid_dc[r["uniprot_id"]][d], conf)

    gt = {}
    for uid in all_uids:
        merged = defaultdict(float)
        gene = gmap.get(uid)
        if gene and gene in gene_dc:
            for d, c in gene_dc[gene].items(): merged[d] = max(merged[d], c)
        if uid in uid_dc:
            for d, c in uid_dc[uid].items(): merged[d] = max(merged[d], c)
        filtered = {d for d, c in merged.items() if c >= CONF_THRESHOLD}
        if filtered:
            gt[uid] = filtered

    canonical = {r["gene_name"]: r["uniprot_id"] for r in qdb("SELECT gene_name, uniprot_id FROM canonical_gene_uniprot")}

    print(f"  {len(all_uids):,} human proteins, {len(gt):,} with non-excluded GT (conf>={CONF_THRESHOLD})")
    return human, vocab_full, vocab_filtered, ptokens, all_uids, gmap, gt, canonical, excluded_counts

def predict_topk(uids, ptokens, vset, vocab, k=5):
    results = {}
    for uid in uids:
        scores = defaultdict(float)
        nv = 0
        for tok, cnt in Counter(ptokens.get(uid, [])).items():
            if tok in vset:
                scores[vocab[tok]["function"]] += vocab[tok]["confidence"] * cnt
                nv += 1
        ranked = sorted(scores.items(), key=lambda x: -x[1]) if scores else []
        total = sum(s for _, s in ranked)
        results[uid] = {"ranked": ranked[:k], "n_vocab": nv, "top1": ranked[0][0] if ranked else None, "total": total}
    return results

def pearson_r(xs, ys):
    n = len(xs)
    if n < 3: return 0.0
    mx, my = sum(xs)/n, sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs)
    syy = sum((y-my)**2 for y in ys)
    sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0: return 0.0
    return max(-1.0, min(1.0, sxy / math.sqrt(sxx * syy)))

def enrichment(pids, ptokens, vset, vocab, gt):
    dp = defaultdict(set)
    agt = set()
    for uid in pids:
        ds = gt.get(uid)
        if ds:
            for d in ds: dp[d].add(uid)
            agt.add(uid)
    if not agt: return {}
    rates = {d: len(ps)/len(agt) for d, ps in dp.items()}
    wc = defaultdict(set)
    for uid in pids:
        seen = set()
        for tok in ptokens.get(uid, []):
            if tok in vset and tok not in seen:
                wc[tok].add(uid); seen.add(tok)
    enr = {}
    for w, carriers in wc.items():
        if len(carriers) < 2: continue
        func = vocab[w]["function"]
        cgt = carriers & agt
        if len(cgt) < 2: continue
        obs = len(cgt & dp.get(func, set())) / len(cgt)
        exp = rates.get(func, 0)
        enr[w] = obs / exp if exp > 0 else 0.0
    return enr

def ms(vals):
    m = sum(vals)/len(vals) if vals else 0
    s = math.sqrt(sum((x-m)**2 for x in vals)/(len(vals)-1)) if len(vals) > 1 else 0
    return m, s

def run_seed(seed, all_uids, ptokens, vset, vocab, gt, dept_freq, mc, all_depts):
    rng = random.Random(seed)
    sh = list(all_uids); rng.shuffle(sh)
    mid = len(sh)//2
    train, test = set(sh[:mid]), set(sh[mid:])

    print(f"  Seed {seed}:", end=" ", flush=True)
    te = enrichment(train, ptokens, vset, vocab, gt)
    ee = enrichment(test, ptokens, vset, vocab, gt)
    common = set(te) & set(ee)
    r = pearson_r([te[w] for w in common], [ee[w] for w in common])

    preds = predict_topk(list(test), ptokens, vset, vocab, k=5)
    dw = [dept_freq.get(d, 0) for d in all_depts]

    topk_c = {k: 0 for k in [1,2,3,5]}
    topk_c_t = {wt: {k: 0 for k in [1,2,3,5]} for wt in WORD_THRESHOLDS}
    topk_n_t = {wt: 0 for wt in WORD_THRESHOLDS}
    adj_c = {k: 0 for k in [1,2,3,5]}
    adj_c_t = {wt: {k: 0 for k in [1,2,3,5]} for wt in WORD_THRESHOLDS}

    cm = defaultdict(lambda: defaultdict(int))
    oc, ot, orc, ofc = 0, 0, 0, 0
    total_topk = 0
    margin_data = []

    per_dept_tp = defaultdict(int)
    per_dept_fp = defaultdict(int)
    per_dept_fn = defaultdict(int)
    per_dept_n = defaultdict(int)

    for uid in test:
        td = gt.get(uid)
        if not td: continue
        p = preds[uid]
        if not p["top1"]: continue

        nw = p["n_vocab"]
        pred = p["top1"]
        ok = pred in td
        primary = sorted(td)[0]
        rd = random.Random(seed*100000+hash(uid)).choices(all_depts, weights=dw)[0]
        rok = rd in td
        fok = mc in td

        true_groups = {FUNCTIONAL_GROUPS.get(d) for d in td} - {None}

        total_topk += 1
        for k in [1,2,3,5]:
            topk_depts = {d for d, _ in p["ranked"][:k]}
            if topk_depts & td: topk_c[k] += 1
            topk_groups = {FUNCTIONAL_GROUPS.get(d) for d in topk_depts} - {None}
            if topk_groups & true_groups: adj_c[k] += 1

        for wt in WORD_THRESHOLDS:
            if nw >= wt:
                topk_n_t[wt] += 1
                for k in [1,2,3,5]:
                    topk_depts = {d for d, _ in p["ranked"][:k]}
                    if topk_depts & td: topk_c_t[wt][k] += 1
                    topk_groups = {FUNCTIONAL_GROUPS.get(d) for d in topk_depts} - {None}
                    if topk_groups & true_groups: adj_c_t[wt][k] += 1

        if len(p["ranked"]) >= 2:
            margin = (p["ranked"][0][1]-p["ranked"][1][1])/p["ranked"][0][1] if p["ranked"][0][1] > 0 else 0
            margin_data.append((margin, ok, nw))

        if ok: per_dept_tp[pred] += 1
        else:
            per_dept_fp[pred] += 1
            for x in td: per_dept_fn[x] += 1
        for d in td: per_dept_n[d] += 1

        cm[primary][pred] += 1
        ot += 1; oc += ok; orc += rok; ofc += fok

    hm = [m for m in margin_data if m[0] > 0.5]
    lm = [m for m in margin_data if m[0] <= 0.3]
    hm_acc = sum(1 for m, c, _ in hm if c)/len(hm) if hm else 0
    lm_acc = sum(1 for m, c, _ in lm if c)/len(lm) if lm else 0

    per_dept_f1 = {}
    for d in set(list(per_dept_tp)+list(per_dept_fp)+list(per_dept_fn)):
        tp, fp, fn = per_dept_tp[d], per_dept_fp[d], per_dept_fn[d]
        pr = tp/(tp+fp) if tp+fp else 0
        re = tp/(tp+fn) if tp+fn else 0
        f1 = 2*pr*re/(pr+re) if pr+re else 0
        per_dept_f1[d] = {"precision": round(pr, 4), "recall": round(re, 4), "f1": round(f1, 4), "support": per_dept_n.get(d, 0)}

    acc = oc/ot if ot else 0
    print(f"r={r:.4f} top1={acc:.4f} top3={topk_c[3]/total_topk:.4f} adj3={adj_c[3]/total_topk:.4f} (n={ot})")

    return {
        "seed": seed, "pearson_r": r, "common_words": len(common),
        "overall_acc": acc, "overall_rand": orc/ot if ot else 0, "overall_freq": ofc/ot if ot else 0, "overall_n": ot,
        "topk": {k: topk_c[k]/total_topk for k in [1,2,3,5]},
        "topk_by_thresh": {wt: {k: topk_c_t[wt][k]/topk_n_t[wt] if topk_n_t[wt] else 0 for k in [1,2,3,5]} for wt in WORD_THRESHOLDS},
        "topk_n_thresh": {wt: topk_n_t[wt] for wt in WORD_THRESHOLDS},
        "adj": {k: adj_c[k]/total_topk for k in [1,2,3,5]},
        "adj_by_thresh": {wt: {k: adj_c_t[wt][k]/topk_n_t[wt] if topk_n_t[wt] else 0 for k in [1,2,3,5]} for wt in WORD_THRESHOLDS},
        "margin_high_acc": hm_acc, "margin_low_acc": lm_acc,
        "margin_high_n": len(hm), "margin_low_n": len(lm),
        "confusion": {k: dict(v) for k, v in cm.items()},
        "per_dept_f1": per_dept_f1,
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("VAL-DICT-001 v6c: Mitochondrial + Transcription Excluded")
    print("=" * 70)
    print(f"Started: {datetime.now(timezone.utc).isoformat()}\n")

    human, vocab_full, vocab_filtered, ptokens, all_uids, gmap, gt, canonical, excluded_counts = load_data()
    vset = set(vocab_filtered.keys())

    n_gt = sum(1 for u in all_uids if u in gt)
    dc = Counter()
    for u in all_uids:
        if u in gt:
            for d in gt[u]: dc[d] += 1
    all_depts = sorted(dc.keys())
    total_d = sum(dc.values())
    df = {d: c/total_d for d, c in dc.items()}
    mc = dc.most_common(1)[0][0]

    print(f"\n  {n_gt:,} proteins with GT (excluding {EXCLUDED_DEPTS}), {len(all_depts)} departments")
    print(f"  Top dept: {mc} ({df[mc]:.1%})")
    print(f"  Dept distribution:")
    for d in sorted(dc, key=dc.get, reverse=True)[:10]:
        print(f"    {d}: {dc[d]:,} ({dc[d]/total_d*100:.1f}%)")

    print(f"\n[2] Well-known proteins...")
    preds_all = predict_topk(all_uids, ptokens, vset, vocab_filtered, k=5)
    well_known = {}
    for gene in ["BRCA1", "TP53", "EGFR", "KRAS", "MYC", "INS", "APOE", "PTEN", "CDK2", "RB1"]:
        cu = canonical.get(gene)
        uid = cu if cu and cu in ptokens else None
        if not uid:
            found = [u for u in all_uids if (gmap.get(u) or "").split()[:1] == [gene]]
            uid = found[0] if found else None
        if uid:
            p = preds_all[uid]
            true = gt.get(uid, set())
            ok = p["top1"] in true if p["top1"] and true else None
            top3 = {d for d, _ in p["ranked"][:3]}
            ok3 = bool(top3 & true) if true else None
            well_known[gene] = {"uid": uid, "n_vocab": p["n_vocab"], "pred": p["top1"],
                "true": sorted(true), "top3": sorted(top3), "exact": ok, "top3_hit": ok3}
            status = "CORRECT" if ok else ("TOP3" if ok3 else ("NO_GT" if not true else "MISS"))
            print(f"  {gene:6} ({uid}): {p['n_vocab']:>3} words, pred={p['top1']}, true={sorted(true)}, top3={sorted(top3)}, {status}")

    print(f"\n[3] Running {len(SEEDS)} held-out splits...")
    seed_results = []
    for seed in SEEDS:
        res = run_seed(seed, all_uids, ptokens, vset, vocab_filtered, gt, df, mc, all_depts)
        seed_results.append(res)

    rs = [s["pearson_r"] for s in seed_results]
    mr, sr = ms(rs)
    accs = [s["overall_acc"] for s in seed_results]
    ma, sa = ms(accs)

    print(f"\n  === CORE METRICS (Mito + Trans Excluded) ===")
    print(f"  Enrichment correlation: r = {mr:.4f} +/- {sr:.4f}")
    print(f"  Top-1 accuracy:         {ma:.4f} +/- {sa:.4f}")
    for k in [2, 3, 5]:
        m, s = ms([s["topk"][k] for s in seed_results])
        print(f"  Top-{k} accuracy:         {m:.4f} +/- {s:.4f}")
    ma3, sa3 = ms([s["adj"][3] for s in seed_results])
    print(f"  Adjacent Top-3:         {ma3:.4f} +/- {sa3:.4f}")
    rand_m = ms([s["overall_rand"] for s in seed_results])[0]
    freq_m = ms([s["overall_freq"] for s in seed_results])[0]
    print(f"  Random baseline:        {rand_m:.4f}")
    print(f"  Freq baseline:          {freq_m:.4f}")
    print(f"  Lift over freq:         {ma/freq_m:.2f}x")

    print(f"\n  === ACCURACY BY WORD COUNT ===")
    print(f"  {'Words':>6} | {'Top-1':>12} | {'Top-3':>12} | {'Top-5':>12} | {'Adj@3':>12} | {'n':>6}")
    for wt in WORD_THRESHOLDS:
        t1 = ms([s["topk_by_thresh"][wt][1] for s in seed_results])
        t3 = ms([s["topk_by_thresh"][wt][3] for s in seed_results])
        t5 = ms([s["topk_by_thresh"][wt][5] for s in seed_results])
        a3 = ms([s["adj_by_thresh"][wt][3] for s in seed_results])
        n_m = ms([s["topk_n_thresh"][wt] for s in seed_results])
        print(f"  >={wt:>4} | {t1[0]:.4f}+/-{t1[1]:.4f} | {t3[0]:.4f}+/-{t3[1]:.4f} | {t5[0]:.4f}+/-{t5[1]:.4f} | {a3[0]:.4f}+/-{a3[1]:.4f} | ~{n_m[0]:.0f}")

    print(f"\n  === CONFIDENCE CALIBRATION ===")
    hm_m, _ = ms([s["margin_high_acc"] for s in seed_results])
    lm_m, _ = ms([s["margin_low_acc"] for s in seed_results])
    print(f"  High-margin (>0.5): {hm_m:.4f}")
    print(f"  Low-margin (<=0.3): {lm_m:.4f}")
    if lm_m > 0:
        print(f"  Calibration ratio: {hm_m/lm_m:.2f}x")

    print(f"\n[4] Per-department F1 scores...")
    merged_f1 = defaultdict(lambda: {"f1": [], "precision": [], "recall": [], "support": 0})
    for s in seed_results:
        for d, vals in s["per_dept_f1"].items():
            merged_f1[d]["f1"].append(vals["f1"])
            merged_f1[d]["precision"].append(vals["precision"])
            merged_f1[d]["recall"].append(vals["recall"])
            merged_f1[d]["support"] += vals["support"]
    for d in sorted(merged_f1, key=lambda x: sum(merged_f1[x]["f1"])/len(merged_f1[x]["f1"]), reverse=True):
        f1m = sum(merged_f1[d]["f1"])/len(merged_f1[d]["f1"])
        pm = sum(merged_f1[d]["precision"])/len(merged_f1[d]["precision"])
        rm = sum(merged_f1[d]["recall"])/len(merged_f1[d]["recall"])
        n = merged_f1[d]["support"]//len(SEEDS)
        print(f"  {d:25s} F1={f1m:.3f}  P={pm:.3f}  R={rm:.3f}  n~{n}")

    print(f"\n[5] Confusion attractors (post Mito+Trans removal)...")
    merged_cm = defaultdict(lambda: defaultdict(int))
    for s in seed_results:
        for td, ps in s["confusion"].items():
            for pd, c in ps.items():
                merged_cm[td][pd] += c
    cm_depts = sorted(set(list(merged_cm.keys()) + [d for v in merged_cm.values() for d in v]))
    wrong_target = defaultdict(int)
    for td in cm_depts:
        for pd in cm_depts:
            if td != pd:
                wrong_target[pd] += merged_cm[td].get(pd, 0)
    print(f"  Top attractors:")
    for d, c in sorted(wrong_target.items(), key=lambda x: -x[1])[:8]:
        print(f"    {d}: {c:,} wrong predictions attracted")

    with open(os.path.join(OUT_DIR, f"{PREFIX}_confusion_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\predicted"] + cm_depts)
        for td in cm_depts:
            w.writerow([td] + [merged_cm[td].get(pd, 0) for pd in cm_depts])

    print(f"\n[6] Generating plots...")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

        preds = predict_topk(all_uids, ptokens, vset, vocab_filtered, k=5)
        data_t1, data_t3, data_t5 = [], [], []
        for uid in all_uids:
            td = gt.get(uid)
            if not td: continue
            p = preds[uid]
            if not p["top1"] or p["n_vocab"] < 1: continue
            nw = p["n_vocab"]
            data_t1.append((nw, 1 if p["top1"] in td else 0))
            data_t3.append((nw, 1 if bool({d for d, _ in p["ranked"][:3]} & td) else 0))
            data_t5.append((nw, 1 if bool({d for d, _ in p["ranked"][:5]} & td) else 0))
        data_t1.sort(); data_t3.sort(); data_t5.sort()
        ths = sorted(set(wc for wc, _ in data_t1))

        def curve(data):
            cx, cy = [], []
            for t in ths:
                sub = [(w, c) for w, c in data if w >= t]
                if len(sub) < 5: continue
                cx.append(t); cy.append(sum(c for _, c in sub)/len(sub))
            return cx, cy

        cx1, cy1 = curve(data_t1)
        cx3, cy3 = curve(data_t3)
        cx5, cy5 = curve(data_t5)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(cx1, cy1, color="#2563eb", linewidth=2, label="Top-1")
        ax.plot(cx3, cy3, color="#16a34a", linewidth=2, label="Top-3")
        ax.plot(cx5, cy5, color="#9333ea", linewidth=2, label="Top-5")
        ax.set_xlabel("Minimum Vocabulary Words", fontsize=12)
        ax.set_ylabel("Prediction Accuracy", fontsize=12)
        ax.set_title("VALDICT001 Accuracy (Mito + Trans Excluded) — 30 Departments", fontsize=13)
        ax.set_ylim(0, 1.05); ax.legend(fontsize=11); ax.grid(True, alpha=0.2)
        for t in WORD_THRESHOLDS:
            s1 = [(w, c) for w, c in data_t1 if w >= t]
            s3 = [(w, c) for w, c in data_t3 if w >= t]
            if s1 and s3:
                acc1 = sum(c for _, c in s1)/len(s1)
                acc3 = sum(c for _, c in s3)/len(s3)
                ax.axvline(x=t, color="orange", ls=":", alpha=0.4)
                ax.annotate(f"≥{t}\nT1:{acc1:.0%}\nT3:{acc3:.0%}\nn={len(s1):,}",
                            xy=(t, acc3), fontsize=7, xytext=(t+1, min(acc3+0.04, 0.98)),
                            bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.8))
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{PREFIX}_accuracy_curve.png"), dpi=150)
        plt.close()
        print(f"  Saved accuracy curve")

        f1_data = [(d, sum(v["f1"])/len(v["f1"])) for d, v in merged_f1.items()]
        f1_data.sort(key=lambda x: -x[1])
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = ["#16a34a" if f >= 0.3 else "#f59e0b" if f >= 0.15 else "#dc2626" for _, f in f1_data]
        ax.bar(range(len(f1_data)), [f for _, f in f1_data], color=colors, alpha=0.8)
        ax.set_xticks(range(len(f1_data)))
        ax.set_xticklabels([d for d, _ in f1_data], rotation=65, ha="right", fontsize=7)
        ax.set_ylabel("F1 Score")
        ax.set_title("Per-Department F1 (Mito + Trans Excluded)")
        avg_f1 = sum(f for _, f in f1_data)/len(f1_data)
        ax.axhline(y=avg_f1, color="blue", ls="--", alpha=0.4, label=f"Mean F1: {avg_f1:.3f}")
        ax.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{PREFIX}_dept_f1.png"), dpi=150)
        plt.close()
        print(f"  Saved department F1 plot")
    except Exception as e:
        print(f"  Warning: plotting failed: {e}")

    print(f"\n[7] Three-layer comparison...")
    v6p = os.path.join(OUT_DIR, "VAL-DICT-001_v6_paper_results.json")
    v6bp = os.path.join(OUT_DIR, "VAL-DICT-001_v6b_noMito_results.json")
    layers = []
    if os.path.exists(v6p):
        with open(v6p) as f: v6 = json.load(f)
        layers.append(("v6: All 32 depts", v6["accuracy"]["top1"]["mean"], v6["accuracy"]["top3"]["mean"],
            v6["enrichment"]["pearson_r"], v6["accuracy"]["freq_baseline"],
            v6["scaling_by_words"][">=50"]["top1"]["mean"], v6["scaling_by_words"][">=50"]["top3"]["mean"],
            v6["confidence_calibration"]["high_margin_acc"]))
    if os.path.exists(v6bp):
        with open(v6bp) as f: v6b = json.load(f)
        layers.append(("v6b: No Mito (31)", v6b["accuracy"]["top1"]["mean"], v6b["accuracy"]["top3"]["mean"],
            v6b["enrichment"]["pearson_r"], v6b["accuracy"]["freq_baseline"],
            v6b["scaling_by_words"][">=50"]["top1"]["mean"], v6b["scaling_by_words"][">=50"]["top3"]["mean"],
            v6b["confidence_calibration"]["high_margin_acc"]))

    t1_50 = ms([s["topk_by_thresh"][50][1] for s in seed_results])[0]
    t3_50 = ms([s["topk_by_thresh"][50][3] for s in seed_results])[0]
    layers.append(("v6c: No Mito+Trans (30)", round(ma, 4), round(ms([s["topk"][3] for s in seed_results])[0], 4),
        round(mr, 4), round(freq_m, 4), round(t1_50, 4), round(t3_50, 4), round(hm_m, 4)))

    print(f"  {'Layer':25s} {'T1':>6s} {'T3':>6s} {'r':>6s} {'Freq':>6s} {'T1@50':>6s} {'T3@50':>6s} {'HiConf':>6s}")
    for name, t1, t3, r, freq, t1_50v, t3_50v, hc in layers:
        print(f"  {name:25s} {t1:>6.1%} {t3:>6.1%} {r:>6.3f} {freq:>6.1%} {t1_50v:>6.1%} {t3_50v:>6.1%} {hc:>6.1%}")

    results = {
        "validation_id": "VAL-DICT-001", "version": "v6c_noMitoTrans",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "excluded": sorted(EXCLUDED_DEPTS),
        "excluded_token_counts": dict(excluded_counts),
        "methodology": {"pipeline": "v2_6bit", "species": "Homo sapiens",
            "gt_join": "gene_name", "gt_confidence": f">={CONF_THRESHOLD}",
            "exclusions": f"{', '.join(sorted(EXCLUDED_DEPTS))} removed from vocab AND GT",
            "seeds": SEEDS},
        "data": {"human_proteins": len(all_uids), "with_gt": n_gt, "departments": len(all_depts),
            "vocab_total": len(vocab_full), "vocab_after_exclusion": len(vocab_filtered)},
        "enrichment": {"pearson_r": round(mr, 4), "std": round(sr, 4)},
        "accuracy": {
            "top1": {"mean": round(ma, 4), "std": round(sa, 4)},
            **{f"top{k}": {"mean": round(ms([s["topk"][k] for s in seed_results])[0], 4),
                           "std": round(ms([s["topk"][k] for s in seed_results])[1], 4)} for k in [2,3,5]},
            "adjacent_top3": {"mean": round(ma3, 4), "std": round(sa3, 4)},
            "random_baseline": round(rand_m, 4), "freq_baseline": round(freq_m, 4),
            "lift_over_freq": round(ma/freq_m, 2) if freq_m else 0,
        },
        "scaling_by_words": {},
        "confidence_calibration": {"high_margin_acc": round(hm_m, 4), "low_margin_acc": round(lm_m, 4),
            "calibration_ratio": round(hm_m/lm_m, 2) if lm_m else 0},
        "per_dept_f1": {d: {"f1": round(sum(v["f1"])/len(v["f1"]), 4),
            "precision": round(sum(v["precision"])/len(v["precision"]), 4),
            "recall": round(sum(v["recall"])/len(v["recall"]), 4),
            "support": v["support"]//len(SEEDS)} for d, v in merged_f1.items()},
        "well_known_proteins": well_known,
        "confusion_attractors": {d: c for d, c in sorted(wrong_target.items(), key=lambda x: -x[1])[:8]},
    }
    for wt in WORD_THRESHOLDS:
        t1m, t1s = ms([s["topk_by_thresh"][wt][1] for s in seed_results])
        t3m, t3s = ms([s["topk_by_thresh"][wt][3] for s in seed_results])
        t5m, _ = ms([s["topk_by_thresh"][wt][5] for s in seed_results])
        a3m, a3s = ms([s["adj_by_thresh"][wt][3] for s in seed_results])
        nm, _ = ms([s["topk_n_thresh"][wt] for s in seed_results])
        results["scaling_by_words"][f">={wt}"] = {
            "top1": {"mean": round(t1m, 4), "std": round(t1s, 4)},
            "top3": {"mean": round(t3m, 4), "std": round(t3s, 4)},
            "top5": round(t5m, 4),
            "adjacent_top3": {"mean": round(a3m, 4), "std": round(a3s, 4)},
            "n_per_split": round(nm),
        }

    with open(os.path.join(OUT_DIR, f"{PREFIX}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    t3_all = ms([s["topk"][3] for s in seed_results])[0]
    t5_50 = ms([s["topk_by_thresh"][50][5] for s in seed_results])[0]
    print(f"\n{'='*70}")
    print(f"COMPLETE in {elapsed:.0f}s")
    print(f"  r={mr:.4f}, Top-1={ma:.4f}, Top-3={t3_all:.4f}, Adj@3={ma3:.4f}")
    print(f"  >=50 words: Top-1={t1_50:.4f}, Top-3={t3_50:.4f}, Top-5={t5_50:.4f}")
    print(f"  Confidence: high={hm_m:.4f} low={lm_m:.4f}")
    print(f"{'='*70}")

    try: get_conn().close()
    except: pass

if __name__ == "__main__":
    main()
