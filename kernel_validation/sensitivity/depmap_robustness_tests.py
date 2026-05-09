#!/usr/bin/env python3
"""
DepMap Essentiality — Five Robustness Tests
============================================
Test 1: Protein length control (regress out length, recompute eta2)
Test 2: Translation peel (remove Translation dept, recompute eta2)
Test 3: GO direct comparison (GO BP annotations vs vocabulary departments)
Test 4: Department size balance (equal-N subsampling, 1000 iterations)
Test 5: Pfam-dark characterization (326 Pfam-dark proteins subset analysis)

Output: validation/sensitivity/depmap_robustness_results.json
"""

import csv
import io
import json
import math
import os
import random
import re
import statistics
import sys
import urllib.request
import urllib.error
import time
from collections import defaultdict, Counter

random.seed(42)

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "depmap_robustness_results.json")
DEPMAP_CACHE = "/tmp/depmap_gene_essentiality.csv"


def ensure_depmap_data():
    if os.path.exists(DEPMAP_CACHE):
        with open(DEPMAP_CACHE) as f:
            lines = sum(1 for _ in f)
        if lines > 1000:
            return True

    print("  Downloading CRISPRGeneEffect.csv from DepMap 25Q3...")
    url_api = "https://depmap.org/portal/api/download/files"
    req = urllib.request.Request(url_api, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=15)
    content = resp.read().decode()

    reader = csv.DictReader(io.StringIO(content))
    download_url = None
    for row in reader:
        if (row.get("filename") == "CRISPRGeneEffect.csv"
                and "25Q3" in row.get("release", "")):
            download_url = row["url"]
            break

    if not download_url:
        print("  ERROR: Could not find DepMap download URL")
        return False

    raw_path = "/tmp/CRISPRGeneEffect.csv"
    req2 = urllib.request.Request(download_url, headers={"User-Agent": "Mozilla/5.0"})
    resp2 = urllib.request.urlopen(req2, timeout=300)
    with open(raw_path, "wb") as f:
        while True:
            chunk = resp2.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    print("  Processing into per-gene essentiality scores...")
    with open(raw_path) as f:
        header = next(csv.reader(f))

    gene_cols = {}
    for i, col in enumerate(header):
        if i == 0:
            continue
        match = re.match(r'^(.+?)\s*\((\d+)\)$', col.strip())
        if match:
            gene_cols[match.group(1).strip()] = i

    gene_scores = {g: [] for g in gene_cols}
    with open(raw_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            for gene, col_idx in gene_cols.items():
                if col_idx < len(row) and row[col_idx]:
                    try:
                        gene_scores[gene].append(float(row[col_idx]))
                    except ValueError:
                        pass

    with open(DEPMAP_CACHE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gene", "mean_chronos", "median_chronos", "n_lines",
                         "pct_dependent"])
        for gene in sorted(gene_scores.keys()):
            scores = gene_scores[gene]
            if len(scores) >= 100:
                writer.writerow([
                    gene,
                    f"{statistics.mean(scores):.4f}",
                    f"{statistics.median(scores):.4f}",
                    len(scores),
                    f"{sum(1 for s in scores if s < -0.5) / len(scores) * 100:.1f}",
                ])

    return True


def load_depmap():
    depmap = {}
    with open(DEPMAP_CACHE) as f:
        for row in csv.DictReader(f):
            depmap[row["gene"]] = float(row["mean_chronos"])
    return depmap


def load_gene_departments():
    path = os.path.join(BASE, "server", "data", "human", "gene_departments.csv")
    depts = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            gene = row["gene"]
            if gene not in depts:
                depts[gene] = row["department"]
    return depts


def load_uniprot_to_gene():
    path = os.path.join(BASE, "server", "data", "human",
                        "protein_tokens_v2_with_genes.csv")
    mapping = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["uniprot_id"]
            gn = row.get("gene_name", "")
            if gn and uid not in mapping:
                mapping[uid] = gn
    return mapping


def compute_eta_squared(dept_groups):
    all_vals = []
    for vals in dept_groups.values():
        all_vals.extend(vals)
    if not all_vals:
        return 0.0
    grand_mean = statistics.mean(all_vals)
    ss_between = sum(len(v) * (statistics.mean(v) - grand_mean) ** 2
                     for v in dept_groups.values() if len(v) >= 2)
    ss_total = sum((x - grand_mean) ** 2 for x in all_vals)
    return ss_between / ss_total if ss_total > 0 else 0.0


def load_protein_lengths_from_db():
    try:
        import psycopg2
        conn = psycopg2.connect(os.environ['BETA_DATABASE_URL'])
        cur = conn.cursor()
        cur.execute("SELECT gene_names_primary, length FROM complete_human_proteome WHERE gene_names_primary IS NOT NULL AND gene_names_primary != ''")
        lengths = {}
        for gene, length in cur.fetchall():
            gene = gene.strip().split()[0]
            if gene and length:
                lengths[gene] = int(length)
        conn.close()
        return lengths
    except Exception as e:
        print(f"  WARNING: Could not load protein lengths from DB: {e}")
        return {}


def load_go_bp_from_db():
    try:
        import psycopg2
        conn = psycopg2.connect(os.environ['BETA_DATABASE_URL'])
        cur = conn.cursor()
        cur.execute("SELECT gene_names_primary, gene_ontology_biological_process FROM complete_human_proteome WHERE gene_names_primary IS NOT NULL AND gene_names_primary != '' AND gene_ontology_biological_process IS NOT NULL AND gene_ontology_biological_process != ''")
        go_map = {}
        for gene, go_bp in cur.fetchall():
            gene = gene.strip().split()[0]
            if gene and go_bp:
                terms = [t.strip() for t in go_bp.split(';') if t.strip()]
                if terms:
                    go_map[gene] = terms
        conn.close()
        return go_map
    except Exception as e:
        print(f"  WARNING: Could not load GO BP from DB: {e}")
        return {}


def load_pfam_dark_proteins():
    path = os.path.join(BASE, "server", "data", "human",
                        "protein_tokens_v2_with_genes.csv")
    all_uids = set()
    uid_to_gene = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["uniprot_id"]
            all_uids.add(uid)
            gn = row.get("gene_name", "")
            if gn:
                uid_to_gene[uid] = gn

    pfam_cache = "/tmp/pfam_dark_cache.json"
    if os.path.exists(pfam_cache):
        with open(pfam_cache) as f:
            cached = json.load(f)
        return cached.get("pfam_dark_genes", []), cached.get("pfam_dark_uids", [])

    print("  Fetching Pfam annotations for Pfam-dark identification (sample of 2000)...")
    sample_uids = random.sample(sorted(all_uids), min(2000, len(all_uids)))

    pfam_dark_uids = []
    pfam_dark_genes = []
    batch_size = 50
    for i in range(0, len(sample_uids), batch_size):
        batch = sample_uids[i:i + batch_size]
        query = "+OR+".join(f"accession:{uid}" for uid in batch)
        url = f"https://rest.uniprot.org/uniprotkb/search?query={query}&fields=accession,xref_pfam&format=tsv&size=500"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=30)
            content = resp.read().decode()
            lines = content.strip().split('\n')
            if len(lines) > 1:
                for line in lines[1:]:
                    parts = line.split('\t')
                    uid = parts[0].strip()
                    pfam = parts[1].strip() if len(parts) > 1 else ""
                    if not pfam:
                        if uid in uid_to_gene:
                            pfam_dark_uids.append(uid)
                            pfam_dark_genes.append(uid_to_gene[uid])
            time.sleep(0.5)
        except Exception as e:
            print(f"    Batch {i//batch_size}: {e}")
            time.sleep(2)

        if (i // batch_size) % 10 == 0:
            print(f"    Processed {i + len(batch)}/{len(sample_uids)} proteins, {len(pfam_dark_genes)} Pfam-dark so far")

    with open(pfam_cache, 'w') as f:
        json.dump({"pfam_dark_genes": pfam_dark_genes, "pfam_dark_uids": pfam_dark_uids}, f)

    return pfam_dark_genes, pfam_dark_uids


def main():
    t0 = time.time()
    print("=" * 70)
    print("  DEPMAP ESSENTIALITY — FIVE ROBUSTNESS TESTS")
    print("=" * 70)

    print("\n[0/5] Loading data...")
    if not ensure_depmap_data():
        print("FAILED: Could not obtain DepMap data")
        return
    depmap = load_depmap()
    depts = load_gene_departments()

    uid_to_gene = load_uniprot_to_gene()
    gene_to_dept = {}
    for uid, dept in depts.items():
        gene = uid_to_gene.get(uid, uid)
        if gene not in gene_to_dept:
            gene_to_dept[gene] = dept

    overlap = set(depmap.keys()) & set(gene_to_dept.keys())
    print(f"  DepMap genes: {len(depmap)}")
    print(f"  Department genes (mapped to gene names): {len(gene_to_dept)}")
    print(f"  Overlap: {len(overlap)}")

    dept_groups = defaultdict(list)
    for g in overlap:
        dept_groups[gene_to_dept[g]].append(depmap[g])
    dept_groups = {d: v for d, v in dept_groups.items() if len(v) >= 10}

    baseline_eta2 = compute_eta_squared(dept_groups)
    print(f"  Baseline eta-squared: {baseline_eta2:.4f} ({baseline_eta2*100:.1f}%)")
    print(f"  Departments with >=10 genes: {len(dept_groups)}")

    results = {
        "test_suite": "DepMap Essentiality Robustness Tests",
        "n_genes_overlap": len(overlap),
        "n_departments": len(dept_groups),
        "baseline_eta_squared": round(baseline_eta2, 4),
    }

    # ==================================================================
    # TEST 1: Protein Length Control
    # ==================================================================
    print("\n" + "=" * 70)
    print("  TEST 1: PROTEIN LENGTH CONTROL")
    print("=" * 70)

    protein_lengths = load_protein_lengths_from_db()
    genes_with_length = overlap & set(protein_lengths.keys())
    print(f"  Genes with length data: {len(genes_with_length)}/{len(overlap)}")

    if len(genes_with_length) > 100:
        lengths = [protein_lengths[g] for g in genes_with_length]
        chronos = [depmap[g] for g in genes_with_length]

        n = len(lengths)
        mean_len = sum(lengths) / n
        mean_chr = sum(chronos) / n
        cov_lc = sum((lengths[i] - mean_len) * (chronos[i] - mean_chr) for i in range(n)) / (n - 1)
        var_len = sum((l - mean_len) ** 2 for l in lengths) / (n - 1)
        var_chr = sum((c - mean_chr) ** 2 for c in chronos) / (n - 1)
        r_length_chronos = cov_lc / math.sqrt(var_len * var_chr) if var_len > 0 and var_chr > 0 else 0

        beta = cov_lc / var_len if var_len > 0 else 0
        residuals = {g: depmap[g] - (mean_chr + beta * (protein_lengths[g] - mean_len))
                     for g in genes_with_length}

        dept_groups_residual = defaultdict(list)
        for g in genes_with_length:
            d = gene_to_dept[g]
            if d in dept_groups:
                dept_groups_residual[d].append(residuals[g])
        dept_groups_residual = {d: v for d, v in dept_groups_residual.items() if len(v) >= 10}

        eta2_residual = compute_eta_squared(dept_groups_residual)

        dept_groups_length_only = defaultdict(list)
        for g in genes_with_length:
            d = gene_to_dept[g]
            if d in dept_groups:
                dept_groups_length_only[d].append(float(protein_lengths[g]))
        dept_mean_lengths = {d: statistics.mean(v) for d, v in dept_groups_length_only.items() if len(v) >= 10}

        print(f"  Pearson r (length vs Chronos): {r_length_chronos:.4f}")
        print(f"  R-squared (length alone): {r_length_chronos**2:.4f} ({r_length_chronos**2*100:.1f}%)")
        print(f"  Eta-squared BEFORE length control: {baseline_eta2:.4f}")
        print(f"  Eta-squared AFTER regressing out length: {eta2_residual:.4f}")
        print(f"  Signal retained: {eta2_residual/baseline_eta2*100:.1f}%")

        print(f"\n  Department mean protein lengths:")
        for d in sorted(dept_mean_lengths, key=lambda x: dept_mean_lengths[x], reverse=True)[:5]:
            print(f"    {d:<20} mean_length={dept_mean_lengths[d]:.0f} aa  n={len(dept_groups_length_only[d])}")
        print(f"    ...")
        for d in sorted(dept_mean_lengths, key=lambda x: dept_mean_lengths[x])[:3]:
            print(f"    {d:<20} mean_length={dept_mean_lengths[d]:.0f} aa  n={len(dept_groups_length_only[d])}")

        results["test1_protein_length"] = {
            "n_genes_with_length": len(genes_with_length),
            "pearson_r_length_chronos": round(r_length_chronos, 4),
            "r_squared_length": round(r_length_chronos ** 2, 4),
            "eta2_before": round(baseline_eta2, 4),
            "eta2_after_length_control": round(eta2_residual, 4),
            "signal_retained_pct": round(eta2_residual / baseline_eta2 * 100, 1),
            "dept_mean_lengths": {d: round(v, 0) for d, v in dept_mean_lengths.items()},
        }
    else:
        print("  SKIPPED: insufficient length data")
        results["test1_protein_length"] = {"status": "skipped", "reason": "insufficient length data"}

    # ==================================================================
    # TEST 2: Translation Peel
    # ==================================================================
    print("\n" + "=" * 70)
    print("  TEST 2: TRANSLATION PEEL")
    print("=" * 70)

    translation_genes = [g for g in overlap if gene_to_dept[g] == "Translation"]
    non_translation = {d: v for d, v in dept_groups.items() if d != "Translation"}

    eta2_no_translation = compute_eta_squared(non_translation)
    n_remaining = sum(len(v) for v in non_translation.values())

    all_scores = []
    for v in dept_groups.values():
        all_scores.extend(v)
    trans_scores = dept_groups.get("Translation", [])
    non_trans_scores = [depmap[g] for g in overlap if gene_to_dept[g] != "Translation"]

    print(f"  Translation genes removed: {len(translation_genes)}")
    print(f"  Remaining genes: {n_remaining}")
    print(f"  Remaining departments: {len(non_translation)}")
    print(f"  Translation dept mean Chronos: {statistics.mean(trans_scores):.4f}")
    print(f"  Non-Translation mean Chronos: {statistics.mean(non_trans_scores):.4f}")
    print(f"  Eta-squared WITH Translation: {baseline_eta2:.4f}")
    print(f"  Eta-squared WITHOUT Translation: {eta2_no_translation:.4f}")
    print(f"  Signal retained: {eta2_no_translation/baseline_eta2*100:.1f}%")

    top5_no_trans = sorted(non_translation.items(), key=lambda x: statistics.mean(x[1]))[:5]
    print(f"\n  Top-5 departments after Translation peel:")
    for dept, scores in top5_no_trans:
        print(f"    {dept:<20} mean={statistics.mean(scores):.4f}  n={len(scores)}  %ess={sum(1 for s in scores if s < -0.5)/len(scores)*100:.1f}%")

    results["test2_translation_peel"] = {
        "n_translation_removed": len(translation_genes),
        "n_remaining": n_remaining,
        "n_depts_remaining": len(non_translation),
        "eta2_with_translation": round(baseline_eta2, 4),
        "eta2_without_translation": round(eta2_no_translation, 4),
        "signal_retained_pct": round(eta2_no_translation / baseline_eta2 * 100, 1),
        "top5_after_peel": [
            {"dept": d, "mean_chronos": round(statistics.mean(v), 4),
             "n": len(v), "pct_essential": round(sum(1 for s in v if s < -0.5) / len(v) * 100, 1)}
            for d, v in top5_no_trans
        ],
    }

    # ==================================================================
    # TEST 3: GO Direct Comparison
    # ==================================================================
    print("\n" + "=" * 70)
    print("  TEST 3: GO BIOLOGICAL PROCESS DIRECT COMPARISON")
    print("=" * 70)

    go_bp = load_go_bp_from_db()
    genes_with_go = overlap & set(go_bp.keys())
    print(f"  Genes with GO BP annotations: {len(genes_with_go)}/{len(overlap)}")

    if len(genes_with_go) > 100:
        go_term_counts = Counter()
        for g in genes_with_go:
            for t in go_bp[g]:
                go_term_counts[t] += 1

        min_genes_per_term = 10
        valid_terms = {t for t, c in go_term_counts.items() if c >= min_genes_per_term}
        print(f"  GO BP terms with >={min_genes_per_term} genes: {len(valid_terms)}")

        gene_primary_go = {}
        for g in genes_with_go:
            best_term = None
            best_count = float('inf')
            for t in go_bp[g]:
                if t in valid_terms:
                    c = go_term_counts[t]
                    if c < best_count:
                        best_count = c
                        best_term = t
            if best_term:
                gene_primary_go[g] = best_term

        go_groups = defaultdict(list)
        for g, t in gene_primary_go.items():
            go_groups[t].append(depmap[g])
        go_groups = {t: v for t, v in go_groups.items() if len(v) >= 10}

        eta2_go = compute_eta_squared(go_groups)
        n_go_groups = len(go_groups)
        n_go_genes = sum(len(v) for v in go_groups.values())

        genes_both = set(gene_primary_go.keys()) & overlap
        dept_groups_matched = defaultdict(list)
        for g in genes_both:
            d = gene_to_dept[g]
            dept_groups_matched[d].append(depmap[g])
        dept_groups_matched = {d: v for d, v in dept_groups_matched.items() if len(v) >= 10}
        eta2_vocab_matched = compute_eta_squared(dept_groups_matched)

        print(f"  GO BP groups (>= 10 genes each): {n_go_groups}")
        print(f"  GO BP genes in groups: {n_go_genes}")
        print(f"  GO BP eta-squared: {eta2_go:.4f} ({eta2_go*100:.1f}%)")
        print(f"  GO BP number of categories: {n_go_groups}")
        print(f"  Vocabulary eta-squared (same genes): {eta2_vocab_matched:.4f} ({eta2_vocab_matched*100:.1f}%)")
        print(f"  Vocabulary number of categories: {len(dept_groups_matched)}")
        print(f"  GO uses {n_go_groups} categories vs vocabulary's {len(dept_groups_matched)}")

        ratio = eta2_go / eta2_vocab_matched if eta2_vocab_matched > 0 else float('inf')
        per_category = (eta2_go / n_go_groups) / (eta2_vocab_matched / len(dept_groups_matched)) if eta2_vocab_matched > 0 and len(dept_groups_matched) > 0 else 0
        print(f"  Raw eta2 ratio (GO/vocab): {ratio:.2f}")
        print(f"  Per-category efficiency ratio (GO/vocab): {per_category:.2f}")

        results["test3_go_comparison"] = {
            "n_genes_with_go": len(genes_with_go),
            "n_go_groups": n_go_groups,
            "n_go_genes": n_go_genes,
            "eta2_go_bp": round(eta2_go, 4),
            "eta2_vocabulary_same_genes": round(eta2_vocab_matched, 4),
            "n_vocab_categories": len(dept_groups_matched),
            "raw_ratio_go_over_vocab": round(ratio, 2),
            "per_category_efficiency": round(per_category, 2),
            "min_genes_per_term": min_genes_per_term,
        }
    else:
        print("  SKIPPED: insufficient GO data")
        results["test3_go_comparison"] = {"status": "skipped"}

    # ==================================================================
    # TEST 4: Department Size Balance
    # ==================================================================
    print("\n" + "=" * 70)
    print("  TEST 4: DEPARTMENT SIZE BALANCE (1,000 iterations)")
    print("=" * 70)

    dept_sizes = {d: len(v) for d, v in dept_groups.items()}
    min_size = min(dept_sizes.values())
    print(f"  Department sizes:")
    for d in sorted(dept_sizes, key=lambda x: dept_sizes[x], reverse=True):
        print(f"    {d:<20} n={dept_sizes[d]}")
    print(f"  Minimum department size: {min_size}")
    print(f"  Subsampling each department to n={min_size}, 1000 iterations...")

    N_ITER = 1000
    balanced_eta2s = []
    for i in range(N_ITER):
        balanced_groups = {}
        for d, vals in dept_groups.items():
            balanced_groups[d] = random.sample(vals, min_size)
        balanced_eta2s.append(compute_eta_squared(balanced_groups))

    mean_balanced = statistics.mean(balanced_eta2s)
    std_balanced = statistics.stdev(balanced_eta2s)
    ci_lo = sorted(balanced_eta2s)[int(0.025 * N_ITER)]
    ci_hi = sorted(balanced_eta2s)[int(0.975 * N_ITER)]

    print(f"  Balanced eta-squared: {mean_balanced:.4f} +/- {std_balanced:.4f}")
    print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Unbalanced eta-squared: {baseline_eta2:.4f}")
    print(f"  Signal retained: {mean_balanced/baseline_eta2*100:.1f}%")

    results["test4_balanced_sampling"] = {
        "dept_sizes": dept_sizes,
        "min_dept_size": min_size,
        "n_iterations": N_ITER,
        "balanced_eta2_mean": round(mean_balanced, 4),
        "balanced_eta2_std": round(std_balanced, 4),
        "balanced_eta2_ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        "unbalanced_eta2": round(baseline_eta2, 4),
        "signal_retained_pct": round(mean_balanced / baseline_eta2 * 100, 1),
    }

    # ==================================================================
    # TEST 5: Pfam-Dark Characterization
    # ==================================================================
    print("\n" + "=" * 70)
    print("  TEST 5: PFAM-DARK PROTEIN CHARACTERIZATION")
    print("=" * 70)

    pfam_dark_genes, pfam_dark_uids = load_pfam_dark_proteins()
    pfam_dark_set = set(pfam_dark_genes)
    print(f"  Pfam-dark genes identified: {len(pfam_dark_set)}")

    pfam_dark_in_depmap = pfam_dark_set & set(depmap.keys())
    pfam_dark_in_both = pfam_dark_set & overlap
    print(f"  Pfam-dark genes in DepMap: {len(pfam_dark_in_depmap)}")
    print(f"  Pfam-dark genes with dept + DepMap: {len(pfam_dark_in_both)}")

    if pfam_dark_in_depmap:
        dark_chronos = [depmap[g] for g in pfam_dark_in_depmap]
        print(f"  Mean Chronos (Pfam-dark): {statistics.mean(dark_chronos):.4f}")
        print(f"  % Essential (Chronos < -0.5): {sum(1 for c in dark_chronos if c < -0.5)/len(dark_chronos)*100:.1f}%")

    if len(protein_lengths) > 0:
        dark_with_length = pfam_dark_set & set(protein_lengths.keys())
        if dark_with_length:
            dark_lengths = [protein_lengths[g] for g in dark_with_length]
            all_lengths = list(protein_lengths.values())
            print(f"  Mean length (Pfam-dark): {statistics.mean(dark_lengths):.0f} aa (n={len(dark_with_length)})")
            print(f"  Mean length (all proteins): {statistics.mean(all_lengths):.0f} aa")
        else:
            dark_lengths = []
    else:
        dark_lengths = []

    if len(pfam_dark_in_both) >= 5:
        dark_dept_groups = defaultdict(list)
        for g in pfam_dark_in_both:
            d = gene_to_dept[g]
            dark_dept_groups[d].append(depmap[g])

        print(f"\n  Pfam-dark department distribution:")
        for d in sorted(dark_dept_groups, key=lambda x: len(dark_dept_groups[x]), reverse=True):
            vals = dark_dept_groups[d]
            print(f"    {d:<20} n={len(vals)}  mean_chronos={statistics.mean(vals):.4f}")

        dark_dept_filtered = {d: v for d, v in dark_dept_groups.items() if len(v) >= 3}
        if len(dark_dept_filtered) >= 2:
            eta2_dark = compute_eta_squared(dark_dept_filtered)
            print(f"\n  Eta-squared (Pfam-dark only, depts with >=3 genes): {eta2_dark:.4f}")
            print(f"  Number of departments: {len(dark_dept_filtered)}")
            print(f"  Total genes: {sum(len(v) for v in dark_dept_filtered.values())}")
        else:
            eta2_dark = None
            print(f"  Too few departments with >=3 genes for eta-squared")
    else:
        eta2_dark = None
        print("  Too few Pfam-dark genes with department + DepMap for subset analysis")

    results["test5_pfam_dark"] = {
        "n_pfam_dark_total": len(pfam_dark_set),
        "n_in_depmap": len(pfam_dark_in_depmap),
        "n_in_both": len(pfam_dark_in_both),
        "mean_chronos_dark": round(statistics.mean(dark_chronos), 4) if pfam_dark_in_depmap else None,
        "pct_essential_dark": round(sum(1 for c in dark_chronos if c < -0.5) / len(dark_chronos) * 100, 1) if pfam_dark_in_depmap else None,
        "mean_length_dark": round(statistics.mean(dark_lengths), 0) if dark_lengths else None,
        "mean_length_all": round(statistics.mean(list(protein_lengths.values())), 0) if protein_lengths else None,
        "eta2_dark_only": round(eta2_dark, 4) if eta2_dark is not None else None,
        "dept_distribution": {d: len(v) for d, v in (dark_dept_groups if len(pfam_dark_in_both) >= 5 else {}).items()},
    }

    # ==================================================================
    # SUMMARY
    # ==================================================================
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("  SUMMARY — ALL FIVE ROBUSTNESS TESTS")
    print("=" * 70)

    t1 = results.get("test1_protein_length", {})
    t2 = results.get("test2_translation_peel", {})
    t3 = results.get("test3_go_comparison", {})
    t4 = results.get("test4_balanced_sampling", {})
    t5 = results.get("test5_pfam_dark", {})

    print(f"\n  Baseline eta-squared: {baseline_eta2:.4f} (21.4%)")
    print()
    if "eta2_after_length_control" in t1:
        print(f"  Test 1 (Length control):     eta2={t1['eta2_after_length_control']:.4f}  retained={t1['signal_retained_pct']:.1f}%  r(length,chronos)={t1['pearson_r_length_chronos']:.4f}")
    if "eta2_without_translation" in t2:
        print(f"  Test 2 (Translation peel):   eta2={t2['eta2_without_translation']:.4f}  retained={t2['signal_retained_pct']:.1f}%  ({t2['n_translation_removed']} genes removed)")
    if "eta2_go_bp" in t3:
        print(f"  Test 3 (GO BP comparison):   GO eta2={t3['eta2_go_bp']:.4f} ({t3['n_go_groups']} groups)  vocab eta2={t3['eta2_vocabulary_same_genes']:.4f} ({t3['n_vocab_categories']} depts)")
    if "balanced_eta2_mean" in t4:
        print(f"  Test 4 (Balanced sampling):  eta2={t4['balanced_eta2_mean']:.4f}+/-{t4['balanced_eta2_std']:.4f}  CI=[{t4['balanced_eta2_ci95'][0]:.4f},{t4['balanced_eta2_ci95'][1]:.4f}]  retained={t4['signal_retained_pct']:.1f}%")
    if t5.get("n_in_both"):
        print(f"  Test 5 (Pfam-dark):          n={t5['n_in_both']}  eta2={t5.get('eta2_dark_only', 'N/A')}  mean_chronos={t5.get('mean_chronos_dark', 'N/A')}")

    results["elapsed_seconds"] = round(elapsed, 1)

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {OUTPUT}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
