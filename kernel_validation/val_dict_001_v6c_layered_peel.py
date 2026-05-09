#!/usr/bin/env python3
"""
VAL-DICT-001 v6c: Progressive Attractor Peel
=============================================
Removes dominant attractor departments one layer at a time.
At each layer, identifies the next biggest attractor and removes it.
Documents the full attractor hierarchy for future vocabulary refinement.

Layers:
  L0: All 32 departments (baseline)
  L1: Remove Mitochondrial
  L2: Remove Transcription
  L3: Remove next attractor (discovered at runtime)
  L4: Remove next attractor (discovered at runtime)
"""

import json, math, os, random, time, csv
from collections import Counter, defaultdict
from datetime import datetime, timezone

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(WORKSPACE, "validation")
PREFIX = "VAL-DICT-001_v6c_layered_peel"
SEEDS = list(range(42, 52))
WORD_THRESHOLDS = [2, 5, 10, 20, 50]
CONF_THRESHOLD = 0.5
MAX_LAYERS = 5

FUNCTIONAL_GROUPS = {
    "Transcription": "gene_regulation", "Chromatin": "gene_regulation",
    "DNA repair": "genome_maintenance", "DNA replication": "genome_maintenance",
    "Cell cycle": "genome_maintenance",
    "Mitochondrial": "metabolism", "Lipid metabolism": "metabolism",
    "Glycosylation": "metabolism", "Methylation": "metabolism",
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

def ms(vals):
    m = sum(vals)/len(vals) if vals else 0
    s = math.sqrt(sum((x-m)**2 for x in vals)/(len(vals)-1)) if len(vals) > 1 else 0
    return m, s

def pearson_r(xs, ys):
    n = len(xs)
    if n < 3: return 0.0
    mx, my = sum(xs)/n, sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs)
    syy = sum((y-my)**2 for y in ys)
    sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0: return 0.0
    return max(-1.0, min(1.0, sxy / math.sqrt(sxx * syy)))

def load_raw_data():
    print("[0] Loading raw data...")
    human = {r["entry"] for r in qdb("SELECT entry FROM complete_human_proteome")}

    vrows = qdb("""SELECT token_hex, primary_function, confidence
        FROM valdict_extended WHERE primary_function IS NOT NULL AND primary_function != 'Unclassified'""")
    vocab_all = {}
    for r in vrows:
        h = r["token_hex"].strip().lower()
        vocab_all[h] = {"function": r["primary_function"], "confidence": float(r["confidence"])}
    print(f"  {len(vocab_all):,} vocab words")

    trows = qdb("SELECT uniprot_id, token_hex FROM protein_tokens_v2 ORDER BY uniprot_id, rank")
    ptokens = defaultdict(list)
    for r in trows:
        if r["uniprot_id"] in human:
            ptokens[r["uniprot_id"]].append(r["token_hex"].strip().lower())
    ptokens = dict(ptokens)
    all_uids = sorted(ptokens.keys())
    print(f"  {len(all_uids):,} human proteins")

    gmap = {r["uniprot_id"]: r["gene_name"] for r in qdb("SELECT uniprot_id, gene_name FROM protein_encoding_v2")}

    gdrows = qdb("SELECT gene_name, primary_department, all_departments, confidence FROM gene_department_map WHERE primary_department IS NOT NULL")
    gene_dc_raw = []
    for r in gdrows:
        conf = float(r["confidence"]) if r["confidence"] else 0.0
        for d in (r["all_departments"] if r["all_departments"] else [r["primary_department"]]):
            gene_dc_raw.append((r["gene_name"], d, conf))

    udrows = qdb("SELECT uniprot_id, primary_department, all_departments, confidence FROM gene_department_map WHERE primary_department IS NOT NULL")
    uid_dc_raw = []
    for r in udrows:
        conf = float(r["confidence"]) if r["confidence"] else 0.0
        for d in (r["all_departments"] if r["all_departments"] else [r["primary_department"]]):
            uid_dc_raw.append((r["uniprot_id"], d, conf))

    return human, vocab_all, ptokens, all_uids, gmap, gene_dc_raw, uid_dc_raw

def build_filtered(vocab_all, ptokens, all_uids, gmap, gene_dc_raw, uid_dc_raw, excluded):
    vocab = {h: v for h, v in vocab_all.items() if v["function"] not in excluded}

    gene_dc = defaultdict(lambda: defaultdict(float))
    for gn, d, conf in gene_dc_raw:
        if d not in excluded:
            gene_dc[gn][d] = max(gene_dc[gn][d], conf)

    uid_dc = defaultdict(lambda: defaultdict(float))
    for uid, d, conf in uid_dc_raw:
        if d not in excluded:
            uid_dc[uid][d] = max(uid_dc[uid][d], conf)

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

    return vocab, gt

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
        results[uid] = {"ranked": ranked[:k], "n_vocab": nv, "top1": ranked[0][0] if ranked else None}
    return results

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

def run_layer(layer_name, excluded, vocab_all, ptokens, all_uids, gmap, gene_dc_raw, uid_dc_raw):
    print(f"\n{'='*60}")
    print(f"  LAYER: {layer_name}")
    print(f"  Excluded: {sorted(excluded) if excluded else 'None'}")
    print(f"{'='*60}")

    vocab, gt = build_filtered(vocab_all, ptokens, all_uids, gmap, gene_dc_raw, uid_dc_raw, excluded)
    vset = set(vocab.keys())

    n_gt = sum(1 for u in all_uids if u in gt)
    dc = Counter()
    for u in all_uids:
        if u in gt:
            for d in gt[u]: dc[d] += 1
    all_depts = sorted(dc.keys())
    total_d = sum(dc.values())
    df = {d: c/total_d for d, c in dc.items()}
    mc = dc.most_common(1)[0][0]

    print(f"  {len(vocab):,} vocab words, {n_gt:,} proteins with GT, {len(all_depts)} depts")
    print(f"  Top dept: {mc} ({df[mc]:.1%})")

    all_seed_results = []
    for seed in SEEDS:
        rng = random.Random(seed)
        sh = list(all_uids); rng.shuffle(sh)
        mid = len(sh)//2
        train, test = set(sh[:mid]), set(sh[mid:])

        te = enrichment(train, ptokens, vset, vocab, gt)
        ee = enrichment(test, ptokens, vset, vocab, gt)
        common = set(te) & set(ee)
        r = pearson_r([te[w] for w in common], [ee[w] for w in common])

        preds = predict_topk(list(test), ptokens, vset, vocab, k=5)
        dw = [df.get(d, 0) for d in all_depts]

        topk_c = {k: 0 for k in [1,2,3,5]}
        topk_c_t = {wt: {k: 0 for k in [1,2,3,5]} for wt in WORD_THRESHOLDS}
        topk_n_t = {wt: 0 for wt in WORD_THRESHOLDS}
        adj_c = {k: 0 for k in [1,2,3,5]}

        cm = defaultdict(lambda: defaultdict(int))
        oc, ot, ofc = 0, 0, 0
        total_topk = 0
        margin_data = []

        for uid in test:
            td = gt.get(uid)
            if not td: continue
            p = preds[uid]
            if not p["top1"]: continue

            nw = p["n_vocab"]
            pred = p["top1"]
            ok = pred in td
            primary = sorted(td)[0]
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
                        topk_depts_w = {d for d, _ in p["ranked"][:k]}
                        if topk_depts_w & td: topk_c_t[wt][k] += 1

            if len(p["ranked"]) >= 2:
                margin = (p["ranked"][0][1]-p["ranked"][1][1])/p["ranked"][0][1] if p["ranked"][0][1] > 0 else 0
                margin_data.append((margin, ok))

            cm[primary][pred] += 1
            ot += 1; oc += ok; ofc += fok

        hm = [m for m in margin_data if m[0] > 0.5]
        lm = [m for m in margin_data if m[0] <= 0.3]
        hm_acc = sum(1 for m, c in hm if c)/len(hm) if hm else 0
        lm_acc = sum(1 for m, c in lm if c)/len(lm) if lm else 0

        acc = oc/ot if ot else 0
        all_seed_results.append({
            "seed": seed, "pearson_r": r, "overall_acc": acc,
            "overall_freq": ofc/ot if ot else 0, "overall_n": ot,
            "topk": {k: topk_c[k]/total_topk if total_topk else 0 for k in [1,2,3,5]},
            "topk_by_thresh": {wt: {k: topk_c_t[wt][k]/topk_n_t[wt] if topk_n_t[wt] else 0 for k in [1,2,3,5]} for wt in WORD_THRESHOLDS},
            "topk_n_thresh": {wt: topk_n_t[wt] for wt in WORD_THRESHOLDS},
            "adj3": adj_c[3]/total_topk if total_topk else 0,
            "margin_high_acc": hm_acc, "margin_low_acc": lm_acc,
            "confusion": {k: dict(v) for k, v in cm.items()},
        })

    mr, sr = ms([s["pearson_r"] for s in all_seed_results])
    ma, sa = ms([s["overall_acc"] for s in all_seed_results])
    t3m, t3s = ms([s["topk"][3] for s in all_seed_results])
    a3m, _ = ms([s["adj3"] for s in all_seed_results])
    freq_m = ms([s["overall_freq"] for s in all_seed_results])[0]
    hm_m, _ = ms([s["margin_high_acc"] for s in all_seed_results])
    lm_m, _ = ms([s["margin_low_acc"] for s in all_seed_results])

    print(f"\n  r={mr:.4f}±{sr:.4f}  Top-1={ma:.4f}±{sa:.4f}  Top-3={t3m:.4f}  Adj@3={a3m:.4f}")
    print(f"  Freq={freq_m:.4f}  Lift={ma/freq_m:.2f}x  HiConf={hm_m:.4f}  LoConf={lm_m:.4f}")

    wt_results = {}
    for wt in WORD_THRESHOLDS:
        t1w = ms([s["topk_by_thresh"][wt][1] for s in all_seed_results])
        t3w = ms([s["topk_by_thresh"][wt][3] for s in all_seed_results])
        t5w = ms([s["topk_by_thresh"][wt][5] for s in all_seed_results])
        nw = ms([s["topk_n_thresh"][wt] for s in all_seed_results])
        wt_results[wt] = {"t1": t1w, "t3": t3w, "t5": t5w, "n": nw}
        print(f"  >={wt:>3}: T1={t1w[0]:.1%}  T3={t3w[0]:.1%}  T5={t5w[0]:.1%}  n~{nw[0]:.0f}")

    merged_cm = defaultdict(lambda: defaultdict(int))
    for s in all_seed_results:
        for td, ps in s["confusion"].items():
            for pd, c in ps.items():
                merged_cm[td][pd] += c

    wrong_target = defaultdict(int)
    cm_depts = sorted(set(list(merged_cm.keys()) + [d for v in merged_cm.values() for d in v]))
    for td in cm_depts:
        for pd in cm_depts:
            if td != pd:
                wrong_target[pd] += merged_cm[td].get(pd, 0)

    next_attractor = max(wrong_target, key=wrong_target.get) if wrong_target else None
    print(f"\n  Next attractor: {next_attractor} ({wrong_target.get(next_attractor, 0):,} wrong predictions)")
    top5_attractors = sorted(wrong_target.items(), key=lambda x: -x[1])[:5]
    for d, c in top5_attractors:
        print(f"    {d}: {c:,}")

    return {
        "layer": layer_name, "excluded": sorted(excluded),
        "n_depts": len(all_depts), "n_vocab": len(vocab), "n_gt": n_gt,
        "pearson_r": round(mr, 4), "r_std": round(sr, 4),
        "top1": round(ma, 4), "top1_std": round(sa, 4),
        "top3": round(t3m, 4), "top3_std": round(t3s, 4),
        "adj3": round(a3m, 4),
        "freq_baseline": round(freq_m, 4),
        "lift": round(ma/freq_m, 2) if freq_m else 0,
        "hi_conf": round(hm_m, 4), "lo_conf": round(lm_m, 4),
        "cal_ratio": round(hm_m/lm_m, 2) if lm_m > 0 else 0,
        "scaling": {f">={wt}": {"top1": round(wt_results[wt]["t1"][0], 4),
            "top3": round(wt_results[wt]["t3"][0], 4),
            "top5": round(wt_results[wt]["t5"][0], 4),
            "n": round(wt_results[wt]["n"][0])} for wt in WORD_THRESHOLDS},
        "next_attractor": next_attractor,
        "top_attractors": {d: c for d, c in top5_attractors},
    }, next_attractor


def main():
    t0 = time.time()
    print("=" * 70)
    print("VAL-DICT-001 v6c: Progressive Attractor Peel")
    print("=" * 70)
    print(f"Started: {datetime.now(timezone.utc).isoformat()}\n")

    human, vocab_all, ptokens, all_uids, gmap, gene_dc_raw, uid_dc_raw = load_raw_data()

    excluded = set()
    all_layers = []
    known_attractors = ["Mitochondrial", "Transcription"]

    for i in range(MAX_LAYERS):
        if i < len(known_attractors):
            removing = known_attractors[i] if i > 0 else None
        else:
            removing = all_layers[-1]["next_attractor"] if all_layers else None

        if i == 0:
            layer_name = f"L0: All departments"
        else:
            if removing:
                excluded.add(removing)
            layer_name = f"L{i}: -{', '.join(sorted(excluded))}"

        result, next_att = run_layer(layer_name, excluded, vocab_all, ptokens, all_uids, gmap, gene_dc_raw, uid_dc_raw)
        all_layers.append(result)

        if i >= len(known_attractors) and next_att:
            known_attractors.append(next_att)

    print(f"\n\n{'='*70}")
    print(f"PROGRESSIVE PEEL SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Layer':40s} {'Depts':>5s} {'Top-1':>7s} {'Top-3':>7s} {'Adj@3':>7s} {'T3@50':>7s} {'HiConf':>7s} {'Lift':>6s} {'Next Attractor':>20s}")
    for layer in all_layers:
        t3_50 = layer["scaling"][">=50"]["top3"]
        print(f"{layer['layer']:40s} {layer['n_depts']:>5d} {layer['top1']:>7.1%} {layer['top3']:>7.1%} {layer['adj3']:>7.1%} {t3_50:>7.1%} {layer['hi_conf']:>7.1%} {layer['lift']:>5.1f}x {layer['next_attractor']:>20s}")

    print(f"\nAccuracy scaling at >=50 words across layers:")
    print(f"{'Layer':40s} {'Top-1':>7s} {'Top-3':>7s} {'Top-5':>7s} {'n':>6s}")
    for layer in all_layers:
        s = layer["scaling"][">=50"]
        print(f"{layer['layer']:40s} {s['top1']:>7.1%} {s['top3']:>7.1%} {s['top5']:>7.1%} {s['n']:>6.0f}")

    print(f"\nAttractor hierarchy discovered:")
    for i, layer in enumerate(all_layers):
        if layer["next_attractor"]:
            att = layer["next_attractor"]
            cnt = layer["top_attractors"].get(att, 0)
            print(f"  Layer {i} → {att} ({cnt:,} wrong predictions absorbed)")

    with open(os.path.join(OUT_DIR, f"{PREFIX}_results.json"), "w") as f:
        json.dump({"validation_id": "VAL-DICT-001", "version": "v6c_layered_peel",
            "timestamp": datetime.now(timezone.utc).isoformat(), "layers": all_layers}, f, indent=2)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        layer_names = [l["layer"].split(":")[0] for l in all_layers]
        t1s = [l["top1"] for l in all_layers]
        t3s = [l["top3"] for l in all_layers]
        a3s = [l["adj3"] for l in all_layers]
        x = range(len(all_layers))

        axes[0].plot(x, t1s, "o-", color="#2563eb", lw=2, label="Top-1")
        axes[0].plot(x, t3s, "s-", color="#16a34a", lw=2, label="Top-3")
        axes[0].plot(x, a3s, "^-", color="#9333ea", lw=2, label="Adj@3")
        axes[0].set_xticks(list(x)); axes[0].set_xticklabels(layer_names, fontsize=9)
        axes[0].set_ylabel("Accuracy"); axes[0].set_title("Overall Accuracy by Layer")
        axes[0].legend(); axes[0].grid(True, alpha=0.2); axes[0].set_ylim(0, 0.8)

        t1_50 = [l["scaling"][">=50"]["top1"] for l in all_layers]
        t3_50 = [l["scaling"][">=50"]["top3"] for l in all_layers]
        t5_50 = [l["scaling"][">=50"]["top5"] for l in all_layers]
        axes[1].plot(x, t1_50, "o-", color="#2563eb", lw=2, label="Top-1 @≥50")
        axes[1].plot(x, t3_50, "s-", color="#16a34a", lw=2, label="Top-3 @≥50")
        axes[1].plot(x, t5_50, "^-", color="#9333ea", lw=2, label="Top-5 @≥50")
        axes[1].set_xticks(list(x)); axes[1].set_xticklabels(layer_names, fontsize=9)
        axes[1].set_ylabel("Accuracy"); axes[1].set_title("Accuracy @≥50 Words by Layer")
        axes[1].legend(); axes[1].grid(True, alpha=0.2); axes[1].set_ylim(0.4, 1.05)

        hcs = [l["hi_conf"] for l in all_layers]
        lcs = [l["lo_conf"] for l in all_layers]
        lifts = [l["lift"] for l in all_layers]
        axes[2].bar([i-0.15 for i in x], hcs, 0.3, color="#16a34a", alpha=0.8, label="High conf")
        axes[2].bar([i+0.15 for i in x], lcs, 0.3, color="#dc2626", alpha=0.8, label="Low conf")
        ax2 = axes[2].twinx()
        ax2.plot(x, lifts, "D-", color="#f59e0b", lw=2, label="Lift over freq")
        ax2.set_ylabel("Lift (x)", color="#f59e0b")
        axes[2].set_xticks(list(x)); axes[2].set_xticklabels(layer_names, fontsize=9)
        axes[2].set_ylabel("Accuracy"); axes[2].set_title("Confidence Calibration by Layer")
        axes[2].legend(loc="upper left"); ax2.legend(loc="upper right")
        axes[2].grid(True, alpha=0.2)

        for ax in axes:
            for i, l in enumerate(all_layers):
                if l.get("next_attractor"):
                    ax.annotate(f"-{l['next_attractor']}", xy=(i, ax.get_ylim()[0]),
                        fontsize=6, rotation=45, ha="left", va="bottom", color="red", alpha=0.7)

        plt.suptitle("VALDICT001: Progressive Attractor Peel Analysis", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{PREFIX}_summary.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSaved summary plot")
    except Exception as e:
        print(f"Warning: plotting failed: {e}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")
    print(f"{'='*70}")

    try: get_conn().close()
    except: pass


if __name__ == "__main__":
    main()
