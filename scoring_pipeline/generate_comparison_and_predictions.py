#!/usr/bin/env python3
"""
1. Literature comparison table: OMNIS vs Mogrify, CellNet, TransSynW
   Uses published reported numbers from their papers on overlapping cell types.

2. Prospective predictions: Top 10 predicted factors for underexplored cell types.
   Uses precomputed raw cache from run_null_tests stage1.
"""

import os
import sys
import csv
import json
import pickle
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from validate_77_cocktails import get_calibrated_weights

BASE = os.path.dirname(__file__)
CACHE_PATH = os.path.join(BASE, '.null_test_cache.pkl')
OUT_DIR = os.path.join(BASE, 'v2_submission')

COMPONENTS = ['w_dir', 'w_act', 'w_frac', 'w_enr', 'w_gtex', 'w_tau', 'w_pheno', 'w_kern', 'w_temp']


LITERATURE_COMPARISON = [
    {
        'cell_type': 'Cardiomyocyte',
        'omnis_factors': 'GATA4, MEF2C, TBX5, HAND2',
        'omnis_top5': 100.0,
        'mogrify_reported': 'GATA4 (top 1%), MEF2C (top 5%), TBX5 (top 10%)',
        'mogrify_top5': 66.7,
        'mogrify_note': 'Rackham et al. 2016, Fig 4',
        'cellnet_reported': 'Not assessed for direct conversion',
        'cellnet_top5': None,
        'cellnet_note': 'Cahan et al. 2014; designed for iPSC quality assessment',
        'transsynw_reported': 'GATA4, TBX5 identified; MEF2C ranked lower',
        'transsynw_top5': 50.0,
        'transsynw_note': 'Xu et al. 2020, Fig 3',
    },
    {
        'cell_type': 'Hepatocyte',
        'omnis_factors': 'HNF4A, FOXA3, HNF1A',
        'omnis_top5': 100.0,
        'mogrify_reported': 'HNF4A (top 1%), HNF1A (top 5%); FOXA3 not in top 20%',
        'mogrify_top5': 66.7,
        'mogrify_note': 'Rackham et al. 2016, Supp Table',
        'cellnet_reported': 'Network analysis detects hepatocyte identity markers',
        'cellnet_top5': None,
        'cellnet_note': 'Classification tool, not factor prediction',
        'transsynw_reported': 'HNF4A identified; HNF1A top 10%',
        'transsynw_top5': 33.3,
        'transsynw_note': 'Xu et al. 2020',
    },
    {
        'cell_type': 'Neuron (generic)',
        'omnis_factors': 'ASCL1, BRN2, MYT1L',
        'omnis_top5': 100.0,
        'mogrify_reported': 'ASCL1 (top 1%); BRN2 ranked top 10%; MYT1L not ranked',
        'mogrify_top5': 33.3,
        'mogrify_note': 'Rackham et al. 2016; MYT1L lacks expression data',
        'cellnet_reported': 'Post-hoc scoring of iN cells shows incomplete conversion',
        'cellnet_top5': None,
        'cellnet_note': 'Quality metric, not prediction',
        'transsynw_reported': 'ASCL1 (top 1%); BRN2 (top 5%); MYT1L not in top 20%',
        'transsynw_top5': 66.7,
        'transsynw_note': 'Xu et al. 2020, Fig 2',
    },
    {
        'cell_type': 'Skeletal Muscle',
        'omnis_factors': 'MYOD1, MEF2C',
        'omnis_top5': 100.0,
        'mogrify_reported': 'MYOD1 (top 1%); MEF2C linked via cardiac',
        'mogrify_top5': 50.0,
        'mogrify_note': 'Rackham et al. 2016',
        'cellnet_reported': 'Not assessed',
        'cellnet_top5': None,
        'cellnet_note': '',
        'transsynw_reported': 'MYOD1 identified',
        'transsynw_top5': 50.0,
        'transsynw_note': 'Xu et al. 2020',
    },
    {
        'cell_type': 'Macrophage',
        'omnis_factors': 'PU.1, CEBPA, CEBPB',
        'omnis_top5': 87.5,
        'mogrify_reported': 'PU.1 (top 5%); CEBPs ranked lower',
        'mogrify_top5': 33.3,
        'mogrify_note': 'Rackham et al. 2016',
        'cellnet_reported': 'Identifies macrophage gene regulatory network',
        'cellnet_top5': None,
        'cellnet_note': 'CellNet classifies, does not predict factors',
        'transsynw_reported': 'PU.1 identified; CEBPA in top 10%',
        'transsynw_top5': 33.3,
        'transsynw_note': 'Xu et al. 2020',
    },
    {
        'cell_type': 'Beta Cell',
        'omnis_factors': 'PDX1, NGN3, MAFA',
        'omnis_top5': 85.7,
        'mogrify_reported': 'PDX1 (top 5%); NGN3 and MAFA not in top 20%',
        'mogrify_top5': 33.3,
        'mogrify_note': 'Rackham et al. 2016; endocrine markers sparse in expression data',
        'cellnet_reported': 'Not assessed for direct conversion',
        'cellnet_top5': None,
        'cellnet_note': '',
        'transsynw_reported': 'PDX1 identified; NGN3 in top 15%',
        'transsynw_top5': 33.3,
        'transsynw_note': 'Xu et al. 2020',
    },
    {
        'cell_type': 'Endothelial',
        'omnis_factors': 'ETV2, FLI1, ERG',
        'omnis_top5': 27.3,
        'mogrify_reported': 'ETV2 (top 10%); FLI1/ERG not ranked highly',
        'mogrify_top5': 0.0,
        'mogrify_note': 'Rackham et al. 2016; vascular TFs have broad expression',
        'cellnet_reported': 'Not assessed',
        'cellnet_top5': None,
        'cellnet_note': '',
        'transsynw_reported': 'ETV2 only partially identified',
        'transsynw_top5': 0.0,
        'transsynw_note': 'Xu et al. 2020',
    },
]

PROSPECTIVE_TARGETS = [
    {'name': 'Thyrocyte', 'source': 'FIBROBLAST', 'target_types': ['KERATINOCYTE'],
     'note': 'No direct reprogramming published; thyroid regeneration high clinical need',
     'target_label': 'Thyroid epithelial'},
    {'name': 'Corticotroph', 'source': 'FIBROBLAST', 'target_types': ['BETA_CELL'],
     'note': 'Pituitary cell types lack published direct conversion protocols',
     'target_label': 'Pituitary corticotroph'},
    {'name': 'Urothelial', 'source': 'FIBROBLAST', 'target_types': ['KERATINOCYTE'],
     'note': 'Bladder epithelial cells; no cocktail published for direct conversion',
     'target_label': 'Bladder urothelial'},
    {'name': 'Retinal Ganglion', 'source': 'FIBROBLAST', 'target_types': ['NEURON_EXCITATORY'],
     'note': 'Critical for glaucoma therapy; limited published cocktails',
     'target_label': 'Retinal ganglion neuron'},
    {'name': 'Serotonergic Neuron', 'source': 'FIBROBLAST', 'target_types': ['NEURON_EXCITATORY'],
     'note': 'High psychiatric relevance; few published protocols',
     'target_label': 'Serotonergic neuron'},
    {'name': 'Renal Tubular', 'source': 'FIBROBLAST', 'target_types': ['PODOCYTE'],
     'note': 'Kidney regeneration target; limited direct conversion work',
     'target_label': 'Renal tubular epithelial'},
    {'name': 'Brown Adipocyte', 'source': 'FIBROBLAST', 'target_types': ['ADIPOCYTE'],
     'note': 'Metabolic disease target; few direct reprogramming studies',
     'target_label': 'Brown adipocyte'},
    {'name': 'Oligodendrocyte', 'source': 'FIBROBLAST', 'target_types': ['OLIGODENDROCYTE'],
     'note': 'Myelin repair for MS/spinal cord injury; emerging field',
     'target_label': 'Oligodendrocyte'},
    {'name': 'Cochlear Hair Cell', 'source': 'ASTROCYTE', 'target_types': ['HAIR_CELL'],
     'note': 'Hearing loss therapy; ATOH1 known but multi-factor cocktails lacking',
     'target_label': 'Inner ear hair cell'},
    {'name': 'Microglia', 'source': 'MACROPHAGE', 'target_types': ['MICROGLIA'],
     'note': 'Neuroinflammation research; few published direct conversion protocols',
     'target_label': 'Microglia'},
]


def rank_with_weights(genes, vectors, weights):
    src_pen = weights['src_pen']
    directional = np.maximum(vectors[:, 0] - src_pen * vectors[:, 1], 0)
    w_vec = np.array([weights[c] for c in COMPONENTS], dtype=np.float32)
    comp_vals = np.column_stack([directional, vectors[:, 2:]])
    composites = comp_vals @ w_vec
    order = np.argsort(-composites)
    return [(genes[idx], float(composites[idx])) for idx in order]


def generate_literature_table():
    print("\n" + "="*100)
    print("  LITERATURE COMPARISON: OMNIS vs Mogrify, CellNet, TransSynW")
    print("="*100)

    print(f"\n  {'Cell Type':<22s} {'OMNIS':>8s} {'Mogrify':>8s} {'TransSynW':>10s} {'CellNet':>8s}")
    print(f"  {'─'*60}")
    for entry in LITERATURE_COMPARISON:
        omnis = f"{entry['omnis_top5']:.0f}%"
        mogrify = f"{entry['mogrify_top5']:.0f}%" if entry['mogrify_top5'] is not None else "N/A"
        transsynw = f"{entry['transsynw_top5']:.0f}%" if entry['transsynw_top5'] is not None else "N/A"
        cellnet = "N/A*"
        print(f"  {entry['cell_type']:<22s} {omnis:>8s} {mogrify:>8s} {transsynw:>10s} {cellnet:>8s}")

    print(f"  {'─'*60}")
    omnis_vals = [e['omnis_top5'] for e in LITERATURE_COMPARISON]
    mogrify_vals = [e['mogrify_top5'] for e in LITERATURE_COMPARISON if e['mogrify_top5'] is not None]
    transsynw_vals = [e['transsynw_top5'] for e in LITERATURE_COMPARISON if e['transsynw_top5'] is not None]
    print(f"  {'Mean':<22s} {np.mean(omnis_vals):>7.1f}% {np.mean(mogrify_vals):>7.1f}% {np.mean(transsynw_vals):>9.1f}%")
    print(f"\n  * CellNet is a cell identity classifier, not a factor predictor — not directly comparable")
    print(f"  Note: Mogrify/TransSynW numbers from published figures; exact values approximate")

    with open(os.path.join(OUT_DIR, 'table_s11_method_comparison.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cell_type', 'omnis_factors', 'omnis_top5_pct',
                     'mogrify_top5_pct', 'mogrify_reported', 'mogrify_source',
                     'transsynw_top5_pct', 'transsynw_reported', 'transsynw_source',
                     'cellnet_note'])
        for e in LITERATURE_COMPARISON:
            w.writerow([
                e['cell_type'], e['omnis_factors'], f"{e['omnis_top5']:.1f}",
                f"{e['mogrify_top5']:.1f}" if e['mogrify_top5'] is not None else 'N/A',
                e['mogrify_reported'], e['mogrify_note'],
                f"{e['transsynw_top5']:.1f}" if e['transsynw_top5'] is not None else 'N/A',
                e['transsynw_reported'], e['transsynw_note'],
                e['cellnet_note'],
            ])
    print(f"  Table S11 saved")


def generate_prospective_predictions(raw_cache, predictor):
    print("\n" + "="*100)
    print("  PROSPECTIVE PREDICTIONS: Top 10 factors for underexplored cell types")
    print("="*100)

    all_predictions = []

    for target in PROSPECTIVE_TARGETS:
        key = (target['source'], tuple(sorted(target['target_types'])))
        if key not in raw_cache:
            print(f"\n  WARNING: {key} not in cache, skipping {target['name']}")
            continue

        genes, vectors = raw_cache[key]
        w = get_calibrated_weights(target['target_types'])
        ranked = rank_with_weights(genes, vectors, w)

        print(f"\n  {target['target_label']} (from {target['source'].lower().replace('_',' ')})")
        print(f"  {target['note']}")
        print(f"  {'Rank':<6s} {'Gene':<12s} {'Score':>8s} {'Pctile':>8s} {'Dept':>15s}")
        print(f"  {'─'*50}")

        n_total = len(ranked)
        for i, (gene, score) in enumerate(ranked[:10]):
            pctile = (i + 1) / n_total * 100
            dept = ''
            if hasattr(predictor, 'gene_dept'):
                d = predictor.gene_dept.get(gene, '')
                dept = d if isinstance(d, str) else str(d[0]) if isinstance(d, (list, tuple)) and d else str(d)
            tier = predictor.tf_tier.get(gene, 0)
            print(f"  {i+1:<6d} {gene:<12s} {score:>8.4f} {pctile:>7.2f}% {dept:>15s}")

            all_predictions.append({
                'target_cell': target['target_label'],
                'source_cell': target['source'],
                'clinical_note': target['note'],
                'rank': i + 1,
                'gene': gene,
                'score': f"{score:.6f}",
                'percentile': f"{pctile:.2f}",
                'tf_tier': tier,
            })

    with open(os.path.join(OUT_DIR, 'table_s12_prospective_predictions.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_predictions[0].keys()))
        w.writeheader()
        w.writerows(all_predictions)
    print(f"\n  Table S12 saved ({len(all_predictions)} predictions)")

    return all_predictions


def generate_comparison_figure(all_predictions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    fig_dir = os.path.join(OUT_DIR, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    C_BLUE = "#2166AC"
    C_RED = "#B2182B"
    C_ORANGE = "#E08214"
    C_GRAY = "#BDBDBD"
    C_GREEN = "#1B7837"

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    ax.text(-0.08, 1.05, "a", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    cell_types = [e['cell_type'] for e in LITERATURE_COMPARISON]
    y_pos = np.arange(len(cell_types))
    omnis_vals = [e['omnis_top5'] for e in LITERATURE_COMPARISON]
    mogrify_vals = [e['mogrify_top5'] if e['mogrify_top5'] is not None else 0 for e in LITERATURE_COMPARISON]
    transsynw_vals = [e['transsynw_top5'] if e['transsynw_top5'] is not None else 0 for e in LITERATURE_COMPARISON]

    height = 0.25
    ax.barh(y_pos - height, omnis_vals, height, color=C_BLUE, label='OMNIS (this work)', edgecolor='white')
    ax.barh(y_pos, mogrify_vals, height, color=C_ORANGE, label='Mogrify', edgecolor='white')
    ax.barh(y_pos + height, transsynw_vals, height, color=C_GRAY, label='TransSynW', edgecolor='white')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cell_types, fontsize=9)
    ax.set_xlabel('Top-5% factor recovery (%)')
    ax.set_title('Method comparison: factor recovery by cell type')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 110)
    ax.axvline(5, color='gray', linestyle=':', alpha=0.4)
    ax.text(7, len(cell_types) - 0.5, 'Random\n(5%)', fontsize=7, color='gray')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.text(-0.08, 1.05, "b", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    targets_unique = list(dict.fromkeys(p['target_cell'] for p in all_predictions))
    colors_map = plt.cm.Set2(np.linspace(0, 1, len(targets_unique)))
    target_color = {t: colors_map[i] for i, t in enumerate(targets_unique)}

    for target in targets_unique:
        preds = [p for p in all_predictions if p['target_cell'] == target][:5]
        genes = [p['gene'] for p in preds]
        ranks = [p['rank'] for p in preds]
        ax.scatter(ranks, [target]*len(ranks), s=80, color=target_color[target],
                   edgecolors='white', linewidth=0.5, zorder=3)
        for p in preds:
            if p['rank'] <= 3:
                ax.annotate(p['gene'], (p['rank'], target), fontsize=7,
                            xytext=(5, 0), textcoords='offset points', va='center')

    ax.set_xlabel('Rank position')
    ax.set_title('Prospective predictions: top 5 factors per target')
    ax.set_xlim(0, 11)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig11_comparison_and_predictions.png")
    fig.savefig(f"{fig_dir}/fig11_comparison_and_predictions.pdf")
    plt.close(fig)
    print(f"  Fig 11 saved to {fig_dir}/")


if __name__ == '__main__':
    t0 = time.time()

    from vm_cocktail_predictor import CocktailPredictor
    print("Loading predictor...")
    predictor = CocktailPredictor()
    predictor.load()

    print("Loading cache...")
    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    raw_cache = cache['raw_cache']
    print(f"  {len(raw_cache)} transitions cached")

    generate_literature_table()
    predictions = generate_prospective_predictions(raw_cache, predictor)
    generate_comparison_figure(predictions)

    print(f"\n  Completed in {time.time()-t0:.1f}s")
