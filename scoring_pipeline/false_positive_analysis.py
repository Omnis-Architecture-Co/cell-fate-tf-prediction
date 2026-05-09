#!/usr/bin/env python3
"""
False Positive Characterization: For each cocktail, examine the top-50 ranked genes.
Classify non-cocktail genes in top-50 as:
  1. Program neighbors — share ≥1 program with a known cocktail factor
  2. Known co-factors — in published cocktails for the same target (from other cocktails)
  3. Novel predictions — neither of the above; potentially testable hypotheses

Also computes: department overlap, marker overlap, and mean program Jaccard similarity
with known factors.

Uses precomputed raw component cache from run_null_tests.py stage1.
"""

import json
import os
import sys
import csv
import pickle
import time
import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(__file__))
from vm_cocktail_predictor import CocktailPredictor
from validate_77_cocktails import (
    get_calibrated_weights,
    TARGET_CELL_MAP as T1, SOURCE_CELL_MAP as S1, ALIAS_MAP as A1,
    FAMILY_MAP as F1, parse_factors as pf1,
)
from validate_extended_cocktails import (
    TARGET_CELL_MAP as T2, SOURCE_CELL_MAP as S2, ALIAS_MAP as A2,
    FAMILY_MAP as F2, parse_factors as pf2,
)
from validate_set3_cocktails import (
    TARGET_CELL_MAP as T3, SOURCE_CELL_MAP as S3, ALIAS_MAP as A3,
    FAMILY_MAP as F3, parse_factors as pf3,
)

BASE = os.path.dirname(__file__)
ROOT = os.path.dirname(BASE)
CACHE_PATH = os.path.join(BASE, '.null_test_cache.pkl')

CSV_77 = os.path.join(ROOT, 'attached_assets', 'published_reprogramming_cocktails_1774053006993.csv')
CSV_EXT = os.path.join(ROOT, 'attached_assets', 'additional_reprogramming_cocktails_1774066435911.csv')
CSV_S3 = os.path.join(ROOT, 'attached_assets', 'validation_set3_cocktails_1774073324946.csv')

TOP_K = 50
COMPONENTS = ['w_dir', 'w_act', 'w_frac', 'w_enr', 'w_gtex', 'w_tau', 'w_pheno', 'w_kern', 'w_temp']


def build_cocktails(csv_path, target_map, source_map, alias_map, parse_fn,
                    family_map, source_col='source_cell'):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    cocktails = []
    for row in rows:
        cid = row.get('cocktail_id', row.get('id', ''))
        target_cell = row['target_cell']
        source_cell = row.get(source_col, row.get('source', ''))
        gene_factors = parse_fn(row['factors'])
        if not gene_factors:
            continue
        target_types = target_map.get(target_cell)
        source_type = source_map.get(source_cell)
        if target_types is None or source_type is None:
            continue
        family = family_map.get(target_cell, 'Other')
        cocktails.append({
            'cid': cid, 'target_cell': target_cell, 'source_cell': source_cell,
            'target_types': target_types, 'source_type': source_type,
            'factors': gene_factors, 'family': family,
        })
    return cocktails


def rank_with_weights(genes, vectors, weights):
    src_pen = weights['src_pen']
    directional = np.maximum(vectors[:, 0] - src_pen * vectors[:, 1], 0)
    w_vec = np.array([weights[c] for c in COMPONENTS], dtype=np.float32)
    comp_vals = np.column_stack([directional, vectors[:, 2:]])
    composites = comp_vals @ w_vec
    order = np.argsort(-composites)
    ranked = [(genes[idx], float(composites[idx])) for idx in order]
    return ranked


def build_all_known_factors_by_target(all_cocktails):
    """For each target cell type, collect all known factors from all cocktails."""
    target_factors = defaultdict(set)
    for cocktails in all_cocktails.values():
        for c in cocktails:
            for tt in c['target_types']:
                target_factors[tt].update(c['factors'])
    return target_factors


def compute_program_jaccard(gene_a, gene_b, gene_progs):
    progs_a = gene_progs.get(gene_a, set())
    progs_b = gene_progs.get(gene_b, set())
    if not progs_a or not progs_b:
        return 0.0
    return len(progs_a & progs_b) / len(progs_a | progs_b)


def classify_top_k(ranked_genes, cocktail_factors, target_factors_all,
                   predictor, target_types, k=TOP_K):
    """Classify top-k genes relative to known cocktail factors."""
    gene_progs = predictor.gene_progs
    factor_set = set(cocktail_factors)
    factor_programs = set()
    for f in factor_set:
        factor_programs |= gene_progs.get(f, set())

    all_known_for_target = set()
    for tt in target_types:
        all_known_for_target |= target_factors_all.get(tt, set())

    results = []
    for rank_idx, (gene, score) in enumerate(ranked_genes[:k]):
        rank = rank_idx + 1
        pctile = rank / len(ranked_genes) * 100
        gene_programs = gene_progs.get(gene, set())

        is_known = gene in factor_set
        shared_programs = len(gene_programs & factor_programs) if gene_programs else 0
        is_program_neighbor = shared_programs > 0 and not is_known
        is_cofactor = gene in all_known_for_target and not is_known

        max_jaccard = 0.0
        best_factor = None
        for f in factor_set:
            j = compute_program_jaccard(gene, f, gene_progs)
            if j > max_jaccard:
                max_jaccard = j
                best_factor = f

        if is_known:
            category = 'known_factor'
        elif is_cofactor:
            category = 'known_cofactor'
        elif is_program_neighbor:
            category = 'program_neighbor'
        else:
            category = 'novel_prediction'

        dept = predictor.gene_dept.get(gene, 'Unknown') if hasattr(predictor, 'gene_dept') else 'Unknown'
        tier = predictor.tf_tier.get(gene, 0)

        results.append({
            'rank': rank,
            'gene': gene,
            'score': score,
            'pctile': pctile,
            'category': category,
            'shared_programs': shared_programs,
            'max_jaccard': max_jaccard,
            'closest_factor': best_factor,
            'tier': tier,
        })

    return results


def run_analysis():
    t0 = time.time()

    print("Loading predictor...")
    predictor = CocktailPredictor()
    predictor.load()

    print("Loading precomputed cache...")
    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    raw_cache = cache['raw_cache']

    print("Building cocktail lists...")
    all_cocktails = {
        'Set 1': build_cocktails(CSV_77, T1, S1, A1, pf1, F1),
        'Set 2': build_cocktails(CSV_EXT, T2, S2, A2, pf2, F2),
        'Set 3': build_cocktails(CSV_S3, T3, S3, A3, pf3, F3, source_col='source'),
    }
    for s, c in all_cocktails.items():
        print(f"  {s}: {len(c)} cocktails")

    target_factors_all = build_all_known_factors_by_target(all_cocktails)

    print(f"\nAnalyzing top-{TOP_K} genes per cocktail...")

    all_results = {}
    aggregate = {'known_factor': 0, 'known_cofactor': 0, 'program_neighbor': 0, 'novel_prediction': 0}
    total_fp = 0
    total_cocktails = 0
    family_stats = defaultdict(lambda: defaultdict(int))
    jaccard_by_category = defaultdict(list)
    cofactor_genes = Counter()
    neighbor_genes = Counter()
    novel_genes = Counter()

    for set_name, cocktails in all_cocktails.items():
        set_results = []
        for c in cocktails:
            key = (c['source_type'], tuple(sorted(c['target_types'])))
            if key not in raw_cache:
                continue

            genes, vectors = raw_cache[key]
            w = get_calibrated_weights(c['target_types'])
            ranked = rank_with_weights(genes, vectors, w)

            classified = classify_top_k(
                ranked, c['factors'], target_factors_all,
                predictor, c['target_types']
            )

            cats = Counter(r['category'] for r in classified)
            for cat, cnt in cats.items():
                aggregate[cat] += cnt
                family_stats[c['family']][cat] += cnt

            for r in classified:
                jaccard_by_category[r['category']].append(r['max_jaccard'])
                if r['category'] == 'known_cofactor':
                    cofactor_genes[r['gene']] += 1
                elif r['category'] == 'program_neighbor':
                    neighbor_genes[r['gene']] += 1
                elif r['category'] == 'novel_prediction':
                    novel_genes[r['gene']] += 1

            n_fp = sum(1 for r in classified if r['category'] != 'known_factor')
            total_fp += n_fp
            total_cocktails += 1

            set_results.append({
                'cocktail_id': c['cid'],
                'target': c['target_cell'],
                'source': c['source_cell'],
                'family': c['family'],
                'n_factors': len(c['factors']),
                'factors_in_top50': sum(1 for r in classified if r['category'] == 'known_factor'),
                'cofactors': sum(1 for r in classified if r['category'] == 'known_cofactor'),
                'program_neighbors': sum(1 for r in classified if r['category'] == 'program_neighbor'),
                'novel': sum(1 for r in classified if r['category'] == 'novel_prediction'),
                'top_genes': classified,
            })

        all_results[set_name] = set_results

    total_slots = total_cocktails * TOP_K
    n_known = aggregate['known_factor']
    n_cofactor = aggregate['known_cofactor']
    n_neighbor = aggregate['program_neighbor']
    n_novel = aggregate['novel_prediction']
    n_fp_total = n_cofactor + n_neighbor + n_novel

    print(f"\n{'='*100}")
    print(f"  FALSE POSITIVE CHARACTERIZATION SUMMARY")
    print(f"  (Top-{TOP_K} genes per cocktail, {total_cocktails} cocktails)")
    print(f"{'='*100}")
    print(f"\n  Total top-{TOP_K} slots:  {total_slots}")
    print(f"  Known factors:         {n_known:>5d}  ({n_known/total_slots*100:.1f}%)")
    print(f"  ─── 'False positives': {n_fp_total:>5d}  ({n_fp_total/total_slots*100:.1f}%)")
    print(f"    Known co-factors:    {n_cofactor:>5d}  ({n_cofactor/n_fp_total*100:.1f}% of FP)")
    print(f"    Program neighbors:   {n_neighbor:>5d}  ({n_neighbor/n_fp_total*100:.1f}% of FP)")
    print(f"    Novel predictions:   {n_novel:>5d}  ({n_novel/n_fp_total*100:.1f}% of FP)")

    print(f"\n  Mean Jaccard similarity to closest known factor:")
    for cat in ['known_factor', 'known_cofactor', 'program_neighbor', 'novel_prediction']:
        vals = jaccard_by_category[cat]
        if vals:
            print(f"    {cat:<22s}: {np.mean(vals):.4f} ± {np.std(vals):.4f} (n={len(vals)})")

    print(f"\n  Per-family breakdown:")
    print(f"  {'Family':<20s} {'Known':>8s} {'Co-fact':>8s} {'Neighbor':>8s} {'Novel':>8s} {'%Expl':>8s}")
    print(f"  {'─'*60}")
    for fam in sorted(family_stats.keys()):
        fs = family_stats[fam]
        total_fam = sum(fs.values())
        explained = fs['known_factor'] + fs['known_cofactor'] + fs['program_neighbor']
        pct = explained / total_fam * 100 if total_fam > 0 else 0
        print(f"  {fam:<20s} {fs['known_factor']:>8d} {fs['known_cofactor']:>8d} "
              f"{fs['program_neighbor']:>8d} {fs['novel_prediction']:>8d} {pct:>7.1f}%")

    print(f"\n  Most frequent 'false positive' genes (appearing in multiple cocktails):")
    print(f"\n  Top co-factors (known in other cocktails for same target):")
    for gene, cnt in cofactor_genes.most_common(15):
        print(f"    {gene:<12s}: {cnt} cocktails")
    print(f"\n  Top program neighbors:")
    for gene, cnt in neighbor_genes.most_common(15):
        print(f"    {gene:<12s}: {cnt} cocktails")
    print(f"\n  Top novel predictions:")
    for gene, cnt in novel_genes.most_common(15):
        print(f"    {gene:<12s}: {cnt} cocktails")

    out_dir = os.path.join(BASE, 'v2_submission')
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'table_s9_false_positive_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['family', 'known_factors', 'known_cofactors', 'program_neighbors',
                     'novel_predictions', 'total', 'pct_explained'])
        for fam in sorted(family_stats.keys()):
            fs = family_stats[fam]
            total_fam = sum(fs.values())
            explained = fs['known_factor'] + fs['known_cofactor'] + fs['program_neighbor']
            pct = explained / total_fam * 100 if total_fam > 0 else 0
            w.writerow([fam, fs['known_factor'], fs['known_cofactor'],
                        fs['program_neighbor'], fs['novel_prediction'], total_fam, f"{pct:.1f}"])

    detail_rows = []
    for set_name, results in all_results.items():
        for cr in results:
            for g in cr['top_genes']:
                detail_rows.append({
                    'set': set_name, 'cocktail_id': cr['cocktail_id'],
                    'target': cr['target'], 'family': cr['family'],
                    'rank': g['rank'], 'gene': g['gene'], 'score': f"{g['score']:.6f}",
                    'pctile': f"{g['pctile']:.2f}", 'category': g['category'],
                    'shared_programs': g['shared_programs'], 'max_jaccard': f"{g['max_jaccard']:.4f}",
                    'closest_factor': g['closest_factor'] or '',
                })

    with open(os.path.join(out_dir, 'table_s10_top50_detail.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        w.writeheader()
        w.writerows(detail_rows)

    summary = {
        'total_cocktails': total_cocktails,
        'top_k': TOP_K,
        'total_slots': total_slots,
        'aggregate': aggregate,
        'family_stats': {k: dict(v) for k, v in family_stats.items()},
        'jaccard_means': {cat: float(np.mean(vals)) for cat, vals in jaccard_by_category.items() if vals},
        'top_cofactors': cofactor_genes.most_common(20),
        'top_neighbors': neighbor_genes.most_common(20),
        'top_novel': novel_genes.most_common(20),
    }
    with open(os.path.join(BASE, 'false_positive_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    generate_figure(aggregate, family_stats, jaccard_by_category, total_cocktails)

    print(f"\n  Analysis completed in {time.time()-t0:.1f}s")


def generate_figure(aggregate, family_stats, jaccard_by_category, total_cocktails):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    fig_dir = os.path.join(BASE, 'v2_submission', 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    C_GREEN = "#1B7837"
    C_BLUE = "#2166AC"
    C_ORANGE = "#E08214"
    C_RED = "#B2182B"
    C_GRAY = "#969696"
    cat_colors = {
        'known_factor': C_GREEN,
        'known_cofactor': C_BLUE,
        'program_neighbor': C_ORANGE,
        'novel_prediction': C_RED,
    }
    cat_labels = {
        'known_factor': 'Known factor',
        'known_cofactor': 'Known co-factor\n(same target)',
        'program_neighbor': 'Program neighbor\n(shared pathway)',
        'novel_prediction': 'Novel prediction',
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    ax.text(-0.08, 1.05, "a", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    cats = ['known_factor', 'known_cofactor', 'program_neighbor', 'novel_prediction']
    sizes = [aggregate[c] for c in cats]
    colors = [cat_colors[c] for c in cats]
    labels = [cat_labels[c] for c in cats]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct='%1.1f%%', colors=colors,
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight('bold')
    ax.legend(wedges, labels, loc='center left', bbox_to_anchor=(-0.25, 0.5), fontsize=9)
    total = sum(sizes)
    ax.set_title(f"Top-{TOP_K} gene classification\n({total:,} slots across {total_cocktails} cocktails)")

    ax = axes[1]
    ax.text(-0.08, 1.05, "b", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    families = sorted(family_stats.keys())
    y_pos = np.arange(len(families))
    left = np.zeros(len(families))
    for cat in cats:
        vals = [family_stats[fam].get(cat, 0) for fam in families]
        ax.barh(y_pos, vals, left=left, color=cat_colors[cat], edgecolor='white',
                label=cat_labels[cat].replace('\n', ' '))
        left += np.array(vals)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(families, fontsize=8)
    ax.set_xlabel("Count")
    ax.set_title("Classification by cell type family")
    ax.legend(fontsize=7, loc='lower right')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[2]
    ax.text(-0.08, 1.05, "c", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    box_data = []
    box_labels = []
    box_colors = []
    for cat in cats:
        vals = jaccard_by_category.get(cat, [])
        if vals:
            box_data.append(vals)
            box_labels.append(cat_labels[cat].replace('\n', ' '))
            box_colors.append(cat_colors[cat])

    bp = ax.boxplot(box_data, vert=True, patch_artist=True, widths=0.6,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(box_labels, fontsize=8, rotation=15, ha='right')
    ax.set_ylabel("Jaccard similarity to closest known factor")
    ax.set_title("Program overlap with known factors")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig10_false_positive_characterization.png")
    fig.savefig(f"{fig_dir}/fig10_false_positive_characterization.pdf")
    plt.close(fig)
    print(f"  Fig 10 saved to {fig_dir}/")


if __name__ == '__main__':
    run_analysis()
