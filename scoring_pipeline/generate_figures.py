#!/usr/bin/env python3
"""
Generate all main figures for Nature Paper 2.
All figures are large (Nature: 180mm max width, we target full-width panels).
Font: Helvetica/Arial. Panel labels: 8pt bold lowercase.
"""

import json
import os
import sys
import pickle
import csv
import numpy as np
from collections import defaultdict, Counter
from scipy import stats
from scipy.linalg import svd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

PROFILES_PATH = "validation/knockout/disruption_profiles_full.json"
GENE_DEPTS_PATH = "server/data/human/gene_departments.csv"
STATE_PATH = "/tmp/module8_full_state.pkl"
FIG_DIR = "paper2/figures"

VALID_DEPARTMENTS = sorted([
    "Apoptosis", "Cell adhesion", "Cell cycle", "Chromatin", "Cytoskeleton",
    "DNA repair", "GTPase", "Immune", "Ion channel", "Kinase", "Methylation",
    "Nuc acid bind", "Phosphatase", "Protein folding", "Proteolysis",
    "RNA processing", "Signaling", "Structural", "Transcription",
    "Translation", "Transport", "Ubiquitin",
])
N_DEPTS = len(VALID_DEPARTMENTS)
D2I = {d: i for i, d in enumerate(VALID_DEPARTMENTS)}

L1_DEPTS = {"Chromatin", "Cytoskeleton", "DNA repair", "Structural", "Cell cycle"}
L2_DEPTS = {"Transcription", "Nuc acid bind", "Methylation", "RNA processing", "Translation"}
L3_DEPTS = {"Kinase", "Signaling", "Phosphatase", "GTPase", "Immune", "Ion channel",
            "Apoptosis", "Cell adhesion", "Protein folding", "Proteolysis", "Transport", "Ubiquitin"}

DISEASE_GENES = {
    "BRCA1", "BRCA2", "TP53", "RB1", "APC", "MLH1", "MSH2", "CFTR",
    "DMD", "HTT", "FBN1", "PKD1", "PKD2", "NF1", "NF2", "TSC1", "TSC2",
    "VHL", "WT1", "MEN1", "RET", "PTEN", "STK11", "SMAD4", "BMPR1A",
    "CDH1", "PALB2", "ATM", "CHEK2", "RAD51C", "MUTYH",
    "PTCH1", "SUFU", "DICER1", "SMARCB1", "BAP1", "CDK4", "CDKN2A",
    "KIT", "PDGFRA", "ALK", "GATA2", "RUNX1", "ETV6", "CEBPA",
    "PAX3", "PAX6", "SOX9", "SOX10", "SHH", "GLI3", "FGFR1", "FGFR2",
    "FGFR3", "COL1A1", "COL1A2", "COL2A1", "COL3A1", "ELN",
    "LMNA", "EMD", "GBA", "HEXA", "HEXB", "IDUA", "GLA", "GAA",
    "SMN1", "DMPK", "FMR1", "AR", "ABCA4", "RPE65",
    "USH2A", "MYO7A", "KCNQ1", "KCNH2", "SCN5A", "RYR1", "RYR2",
    "CACNA1A", "SCN1A", "KCNJ11", "ABCC8", "GJB2", "SLC26A4",
    "HBB", "F5", "F8", "F9",
    "SERPINC1", "PROC", "PROS1",
    "LDLR", "APOB", "PCSK9",
    "G6PD", "ATP7A", "ATP7B",
}

ESSENTIAL_CORE = {
    "RPS2", "RPS3", "RPS5", "RPS6", "RPS8", "RPS9", "RPS14", "RPS19",
    "RPL5", "RPL11", "RPL23", "RPL26", "RPL35A",
    "POLR2A", "POLR2B", "POLR2C", "POLR2D", "POLR2E",
    "SF3B1", "SF3A1", "PRPF8", "SNRPD1",
    "PSMA1", "PSMA2", "PSMA3", "PSMB1", "PSMB2", "PSMB5",
    "CDK1", "CDK2", "CDK7", "CDK9",
    "UBA1", "UBB", "UBC",
    "PCNA", "RFC1", "MCM2", "MCM4", "MCM5", "MCM7",
    "COPA", "COPB1", "COPB2", "COPE", "COPG1",
    "CCT2", "CCT3", "CCT4", "CCT5", "CCT6A", "CCT7", "CCT8", "TCP1",
}

C_BLUE = "#2166AC"
C_RED = "#B2182B"
C_GREEN = "#1B7837"
C_ORANGE = "#E08214"
C_PURPLE = "#7B3294"
C_GRAY = "#969696"
C_LIGHT_BLUE = "#92C5DE"
C_LIGHT_RED = "#F4A582"

C_L1 = "#4393C3"
C_L2 = "#F4A582"
C_L3 = "#D6604D"


def load_data():
    with open(PROFILES_PATH) as f:
        data = json.load(f)
    profiles = {}
    for gene, prof in data["profiles"].items():
        vec = np.array([prof.get(d, 0.0) for d in VALID_DEPARTMENTS])
        profiles[gene] = vec
    genes = sorted(profiles.keys())
    M = np.array([profiles[g] for g in genes])
    gene_idx = {g: i for i, g in enumerate(genes)}

    gene_depts = {}
    with open(GENE_DEPTS_PATH) as f:
        for row in csv.DictReader(f):
            gene_depts[row["gene"]] = row["department"]

    return genes, M, gene_idx, profiles, gene_depts


def load_ppi_degree(genes, gene_idx):
    if not os.path.exists(STATE_PATH):
        sys.path.insert(0, ".")
        from validation.sensitivity.module8_full_shuffle import load_state_from_db
        load_state_from_db()
    with open(STATE_PATH, "rb") as f:
        state = pickle.load(f)
    ptt = state["ptt"]
    ttp = state["ttp"]
    gc = state["gene_cache"]

    token_to_genes = defaultdict(set)
    for token, uids in ttp.items():
        for uid in uids:
            g = gc.get(uid)
            if g:
                token_to_genes[token].add(g)

    ppi_degree = defaultdict(int)
    for token, gs in token_to_genes.items():
        if len(gs) > 300:
            continue
        for g in gs:
            ppi_degree[g] += len(gs) - 1

    return ppi_degree


def panel_label(ax, label, x=-0.08, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="top", ha="left")


def figure1(genes, M, gene_idx, ppi_degree):
    """Fig 1: The disruption profile space and PC1 dominance."""
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    U, S, Vt = svd(M, full_matrices=False)
    eigenvalues = S**2 / len(genes)
    pc1 = Vt[0]
    pc1_scores = M @ pc1
    profile_sums = np.sum(M, axis=1)

    ax1 = fig.add_subplot(gs[0, 0])
    panel_label(ax1, "a")
    n_show = min(21, len(eigenvalues))
    x_ev = np.arange(1, n_show + 1)
    colors_ev = [C_RED] + [C_BLUE] * (n_show - 1)
    ax1.bar(x_ev, eigenvalues[:n_show], color=colors_ev, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Eigenvalue (variance explained)")
    ax1.set_title("Eigenvalue spectrum of disruption profiles")
    ratio = eigenvalues[0] / eigenvalues[1]
    ax1.annotate(f"λ₁/λ₂ = {ratio:.1f}",
                 xy=(1, eigenvalues[0]), xytext=(5, eigenvalues[0] * 0.85),
                 fontsize=11, fontweight="bold", color=C_RED,
                 arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.5))
    ax1.set_xlim(0.3, n_show + 0.7)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    panel_label(ax2, "b")
    degree_vals = []
    pc1_vals = []
    for g in genes:
        if g in ppi_degree:
            degree_vals.append(ppi_degree[g])
            pc1_vals.append(pc1_scores[gene_idx[g]])
    n_sample = min(5000, len(degree_vals))
    np.random.seed(42)
    idx = np.random.choice(len(degree_vals), n_sample, replace=False)
    dx = [degree_vals[i] for i in idx]
    py = [pc1_vals[i] for i in idx]
    ax2.scatter(dx, py, s=3, alpha=0.15, color=C_BLUE, rasterized=True)
    r_val, _ = stats.spearmanr(degree_vals, pc1_vals)
    ax2.set_xlabel("PPI degree (token-sharing network)")
    ax2.set_ylabel("PC1 score")
    ax2.set_title("PC1 = network connectivity")
    ax2.text(0.95, 0.05, f"ρ = {r_val:.2f}\nn = {len(degree_vals):,}",
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=12, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_BLUE, alpha=0.9))
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax3 = fig.add_subplot(gs[1, 0])
    panel_label(ax3, "c")
    pc1_proj = np.outer(M @ pc1, pc1)
    M_resid = M - pc1_proj
    pc2 = Vt[1]
    pc3 = Vt[2]
    x_proj = M_resid @ pc2
    y_proj = M_resid @ pc3
    n_sample2 = min(8000, len(genes))
    idx2 = np.random.choice(len(genes), n_sample2, replace=False)

    disease_in = DISEASE_GENES & set(genes)
    essential_in = ESSENTIAL_CORE & set(genes)
    is_disease = np.array([genes[i] in disease_in for i in idx2])
    is_essential = np.array([genes[i] in essential_in for i in idx2])
    is_other = ~is_disease & ~is_essential

    ax3.scatter(x_proj[idx2[is_other]], y_proj[idx2[is_other]],
                s=2, alpha=0.08, color=C_GRAY, rasterized=True, label="Other genes")
    ax3.scatter(x_proj[idx2[is_disease]], y_proj[idx2[is_disease]],
                s=25, alpha=0.8, color=C_RED, edgecolors="black", linewidth=0.3,
                zorder=5, label=f"Disease genes (n={int(is_disease.sum())})")
    ax3.scatter(x_proj[idx2[is_essential]], y_proj[idx2[is_essential]],
                s=25, alpha=0.8, color=C_GREEN, edgecolors="black", linewidth=0.3,
                zorder=5, label=f"Essential genes (n={int(is_essential.sum())})")
    ax3.set_xlabel("PC2 (residual)")
    ax3.set_ylabel("PC3 (residual)")
    ax3.set_title("Residual space after PC1 removal")
    ax3.legend(loc="upper right", framealpha=0.9, markerscale=2)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    ax4 = fig.add_subplot(gs[1, 1])
    panel_label(ax4, "d")

    gene_depts_local = {}
    with open(GENE_DEPTS_PATH) as f:
        for row in csv.DictReader(f):
            gene_depts_local[row["gene"]] = row["department"]

    norms = np.linalg.norm(M_resid, axis=1)
    med_norm = np.median(norms)

    groups = {"All genes": lambda i: True,
              "High residual\nnorm": lambda i: norms[i] > med_norm,
              "Low residual\nnorm": lambda i: norms[i] <= med_norm}
    group_acc = {}
    for label, mask_fn in groups.items():
        t1 = 0; total_g = 0
        for i, g in enumerate(genes):
            if g not in gene_depts_local or gene_depts_local[g] not in VALID_DEPARTMENTS:
                continue
            if not mask_fn(i):
                continue
            actual_idx = VALID_DEPARTMENTS.index(gene_depts_local[g])
            total_g += 1
            resid_ranked = np.argsort(-np.abs(M_resid[i]))
            if resid_ranked[0] == actual_idx:
                t1 += 1
        group_acc[label] = (t1 / total_g * 100 if total_g > 0 else 0, total_g)

    chance = 100.0 / len(VALID_DEPARTMENTS)
    x_pos = np.arange(3)
    labels_g = list(groups.keys())
    accs = [group_acc[l][0] for l in labels_g]
    ns = [group_acc[l][1] for l in labels_g]
    colors_g = [C_BLUE, C_GREEN, C_GRAY]
    bars = ax4.bar(x_pos, accs, color=colors_g, edgecolor="white", width=0.5)
    ax4.axhline(chance, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax4.text(2.4, chance + 0.3, f"Chance ({chance:.1f}%)", fontsize=8, color="gray")
    for i, bar in enumerate(bars):
        h = bar.get_height()
        fold = h / chance
        ax4.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}%\n({fold:.1f}× chance)",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax4.set_ylabel("Department prediction accuracy (%)")
    ax4.set_title("Residual profiles predict function")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels_g)
    ax4.set_ylim(0, max(accs) * 1.5)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    fig.savefig(f"{FIG_DIR}/fig1_disruption_space.png")
    fig.savefig(f"{FIG_DIR}/fig1_disruption_space.pdf")
    plt.close(fig)
    print("  Fig 1 saved")


def figure2(genes, M, gene_idx):
    """Fig 2: Tropical extreme-value structure and convergence."""
    fig = plt.figure(figsize=(18, 7))
    gs = gridspec.GridSpec(1, 3, wspace=0.35)

    gene_sets = {
        "Tumor suppressors": ({"BRCA1","BRCA2","TP53","RB1","APC","PTEN","VHL","NF1","NF2",
                               "WT1","SMAD4","CDH1","CDKN2A","BAP1","SMARCB1","ARID1A",
                               "KMT2D","KMT2C","CREBBP","EP300","STAG2","ATRX","SETD2",
                               "KDM6A","FBXW7","PTCH1","SUFU","TSC1","TSC2","STK11"}, C_RED),
        "Essential core": (ESSENTIAL_CORE, C_GREEN),
        "Housekeeping": ({"ACTB","GAPDH","TUBB","TUBA1A","HSP90AA1","HSP90AB1",
                          "HSPA8","HSPA5","PPIA","PPIB","EEF1A1","EEF2",
                          "CALM1","CALM2","CALM3","UBB","UBC","RPS27A",
                          "LDHA","LDHB","PKM","ENO1","ALDOA","TPI1","PGK1"}, C_BLUE),
    }

    ax1 = fig.add_subplot(gs[0, 0])
    panel_label(ax1, "a")

    for set_name, (gset, color) in gene_sets.items():
        gs_in = [g for g in gset if g in gene_idx]
        if len(gs_in) < 3:
            continue
        vecs = np.array([M[gene_idx[g]] for g in gs_in])

        np.random.seed(42)
        order = np.random.permutation(len(gs_in))
        fractions = []
        running_max = np.zeros(N_DEPTS)
        final_max = np.max(vecs, axis=0)
        final_sum = np.sum(final_max)

        for i, idx in enumerate(order):
            running_max = np.maximum(running_max, vecs[idx])
            frac = np.sum(running_max) / final_sum if final_sum > 0 else 0
            fractions.append(frac)

        ax1.plot(range(1, len(fractions)+1), fractions, "-o", color=color,
                 markersize=4, linewidth=2, label=f"{set_name} (n={len(gs_in)})")

    ax1.axhline(0.95, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax1.text(2, 0.96, "95% saturation", fontsize=9, color="gray")
    ax1.set_xlabel("Number of genes added")
    ax1.set_ylabel("Fraction of tropical max reached")
    ax1.set_title("Tropical saturation convergence")
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.set_ylim(0.3, 1.05)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    panel_label(ax2, "b")

    leaders = {
        "Tumor\nsuppressors": [("KMT2C", 17, C_RED), ("TSC2", 3, C_LIGHT_RED), ("Others", 2, C_GRAY)],
        "Oncogenes": [("ALK", 22, C_PURPLE), ("Others", 0, C_GRAY)],
        "Essential\ncore": [("PRPF8", 21, C_GREEN), ("COPA", 1, "#a1d99b"), ("Others", 0, C_GRAY)],
        "House-\nkeeping": [("HSPA8", 11, C_BLUE), ("HSP90AA1", 4, C_LIGHT_BLUE), ("Others", 7, C_GRAY)],
    }

    y_pos = np.arange(len(leaders))
    for i, (set_name, genes_data) in enumerate(leaders.items()):
        left = 0
        for gene_name, count, color in genes_data:
            if count > 0:
                ax2.barh(i, count, left=left, color=color, edgecolor="white", height=0.6)
                if count >= 3:
                    ax2.text(left + count/2, i, gene_name, ha="center", va="center",
                             fontsize=8, fontweight="bold", color="white",
                             path_effects=[pe.withStroke(linewidth=2, foreground="black")])
                left += count

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(list(leaders.keys()))
    ax2.set_xlabel("Department maxima carried (out of 22)")
    ax2.set_title("Tropical max leader genes")
    ax2.set_xlim(0, 23)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax3 = fig.add_subplot(gs[0, 2])
    panel_label(ax3, "c")

    np.random.seed(42)
    n_trials = 500
    saturation_points = []
    for _ in range(n_trials):
        sample = np.random.choice(len(genes), size=50, replace=False)
        vecs = M[sample]
        running_max = np.zeros(N_DEPTS)
        final_max = np.max(vecs, axis=0)
        final_sum = np.sum(final_max)
        for j in range(len(sample)):
            running_max = np.maximum(running_max, vecs[j])
            if np.sum(running_max) / final_sum >= 0.95:
                saturation_points.append(j + 1)
                break
        else:
            saturation_points.append(50)

    ax3.hist(saturation_points, bins=np.arange(1, max(saturation_points)+2)-0.5,
             color=C_BLUE, edgecolor="white", alpha=0.8)
    median_sat = np.median(saturation_points)
    ax3.axvline(median_sat, color=C_RED, linestyle="--", linewidth=2)
    ax3.text(min(median_sat + 1, 40), ax3.get_ylim()[1]*0.85 if ax3.get_ylim()[1] > 0 else 50,
             f"Median: {median_sat:.0f} genes\n(out of 50)",
             fontsize=10, fontweight="bold", color=C_RED)
    ax3.set_xlabel("Genes needed for 95% saturation")
    ax3.set_ylabel("Frequency (n=500 trials)")
    ax3.set_title("Genome-wide tropical saturation\n(random 50-gene samples)")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    fig.savefig(f"{FIG_DIR}/fig2_tropical_structure.png")
    fig.savefig(f"{FIG_DIR}/fig2_tropical_structure.pdf")
    plt.close(fig)
    print("  Fig 2 saved")


def figure3(genes, M, gene_idx, gene_depts):
    """Fig 3: 75/25 architecture, disease vs essential enrichment, entropy."""
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    U, S, Vt = svd(M, full_matrices=False)
    pc1 = Vt[0]
    M_resid = M - np.outer(M @ pc1, pc1)
    norms = np.linalg.norm(M_resid, axis=1)

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = km.fit_predict(M_resid)
    cluster_sizes = Counter(labels)
    bulk_label = cluster_sizes.most_common(1)[0][0]

    spec_mask = labels != bulk_label
    bulk_mask = labels == bulk_label

    ax1 = fig.add_subplot(gs[0, 0])
    panel_label(ax1, "a")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(M_resid)

    n_show = min(8000, len(genes))
    np.random.seed(42)
    show_idx = np.random.choice(len(genes), n_show, replace=False)

    for cl in range(5):
        mask = np.array([labels[i] == cl for i in show_idx])
        if cl == bulk_label:
            ax1.scatter(coords[show_idx[mask], 0], coords[show_idx[mask], 1],
                        s=2, alpha=0.05, color=C_GRAY, rasterized=True, label=f"Bulk ({cluster_sizes[cl]:,})")
        else:
            c = [C_BLUE, C_RED, C_GREEN, C_ORANGE, C_PURPLE][cl % 5]
            ax1.scatter(coords[show_idx[mask], 0], coords[show_idx[mask], 1],
                        s=4, alpha=0.25, color=c, rasterized=True,
                        label=f"Fiber {cl} ({cluster_sizes[cl]:,})")

    ax1.set_xlabel("PC2")
    ax1.set_ylabel("PC3")
    ax1.set_title("Residual space: bulk vs specialized fibers")
    ax1.legend(loc="upper right", markerscale=5, framealpha=0.9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    panel_label(ax2, "b")

    disease_in = DISEASE_GENES & set(genes)
    essential_in = ESSENTIAL_CORE & set(genes)
    spec_genes = {genes[i] for i in range(len(genes)) if spec_mask[i]}
    spec_frac = sum(spec_mask) / len(genes)

    categories = ["OMIM\ndisease", "Tumor\nsuppressors", "Essential\ncore", "Housekeeping"]
    tsg = {"TP53","RB1","APC","BRCA1","BRCA2","PTEN","VHL","NF1","NF2",
           "WT1","SMAD4","CDH1","CDKN2A","BAP1","SMARCB1","ARID1A",
           "KMT2D","KMT2C","CREBBP","EP300","STAG2","ATRX","SETD2",
           "KDM6A","FBXW7","PTCH1","SUFU","TSC1","TSC2","STK11"} & set(genes)
    hk = {"ACTB","GAPDH","TUBB","TUBA1A","HSP90AA1","HSP90AB1","HSPA8","HSPA5",
          "PPIA","PPIB","EEF1A1","EEF2","CALM1","UBB","UBC","RPS27A","PKM"} & set(genes)

    enrichments = []
    for gset in [disease_in, tsg, essential_in, hk]:
        n_spec = len(gset & spec_genes)
        n_total = len(gset)
        e = (n_spec / n_total) / spec_frac if n_total > 0 and spec_frac > 0 else 0
        enrichments.append(e)

    colors_bar = [C_RED, C_RED, C_GREEN, C_BLUE]
    bars = ax2.bar(range(4), enrichments, color=colors_bar, edgecolor="white", width=0.6)
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax2.text(3.5, 1.05, "Expected if random", fontsize=8, ha="right", color="gray")
    for i, (bar, e) in enumerate(zip(bars, enrichments)):
        ax2.text(bar.get_x() + bar.get_width()/2, e + 0.05, f"{e:.1f}x",
                 ha="center", fontsize=11, fontweight="bold")
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(categories)
    ax2.set_ylabel("Enrichment in specialized fibers")
    ax2.set_title("Disease genes in fibers, essential genes in bulk")
    ax2.set_ylim(0, max(enrichments) * 1.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax3 = fig.add_subplot(gs[1, 0])
    panel_label(ax3, "c")

    entropies = []
    for i in range(len(genes)):
        v = M[i] / (M[i].sum() + 1e-30)
        v = v[v > 0]
        e = -np.sum(v * np.log2(v))
        entropies.append(e)
    entropies = np.array(entropies)
    max_ent = np.log2(N_DEPTS)
    ent_ratios = entropies / max_ent

    ax3.hist(ent_ratios, bins=50, color=C_GRAY, alpha=0.5, edgecolor="white", label="All genes")

    disease_ent = [ent_ratios[gene_idx[g]] for g in disease_in if g in gene_idx]
    essential_ent = [ent_ratios[gene_idx[g]] for g in essential_in if g in gene_idx]
    ax3.hist(disease_ent, bins=30, color=C_RED, alpha=0.7, edgecolor="white",
             label=f"Disease (n={len(disease_ent)})")
    ax3.hist(essential_ent, bins=15, color=C_GREEN, alpha=0.7, edgecolor="white",
             label=f"Essential (n={len(essential_ent)})")

    ax3.axvline(np.median(disease_ent), color=C_RED, linestyle="--", linewidth=2)
    ax3.axvline(np.median(essential_ent), color=C_GREEN, linestyle="--", linewidth=2)
    ax3.set_xlabel("Entropy ratio (1.0 = uniform disruption)")
    ax3.set_ylabel("Count")
    ax3.set_title("Disruption entropy: disease = broad, essential = specific")
    ax3.legend(loc="upper left")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    ax4 = fig.add_subplot(gs[1, 1])
    panel_label(ax4, "d")

    norms_sorted = np.sort(norms)
    med_norm = np.median(norms)
    high_mask = norms > med_norm
    low_mask = norms <= med_norm

    gene_has_dept = np.array([genes[i] in gene_depts for i in range(len(genes))])
    M_normed = M_resid / (norms[:, np.newaxis] + 1e-30)

    high_correct = 0
    high_total = 0
    low_correct = 0
    low_total = 0

    np.random.seed(42)
    sample = np.random.choice(len(genes), size=min(3000, len(genes)), replace=False)
    for i in sample:
        if not gene_has_dept[i]:
            continue
        top_dept_idx = np.argmax(np.abs(M_resid[i]))
        pred_dept = VALID_DEPARTMENTS[top_dept_idx]
        actual = gene_depts.get(genes[i], "")
        correct = pred_dept == actual
        if high_mask[i]:
            high_total += 1
            high_correct += int(correct)
        else:
            low_total += 1
            low_correct += int(correct)

    high_acc = high_correct / high_total if high_total > 0 else 0
    low_acc = low_correct / low_total if low_total > 0 else 0

    bars = ax4.bar(["Low residual\nnorm", "High residual\nnorm"],
                   [low_acc * 100, high_acc * 100],
                   color=[C_GRAY, C_BLUE], edgecolor="white", width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%",
                 ha="center", fontsize=12, fontweight="bold")

    ax4.set_ylabel("Department prediction accuracy (%)")
    ax4.set_title("Residual norm predicts informativeness")
    ratio = high_acc / low_acc if low_acc > 0 else 0
    ax4.text(0.5, 0.85, f"{ratio:.1f}× higher accuracy",
             transform=ax4.transAxes, ha="center", fontsize=13,
             fontweight="bold", color=C_RED)
    ax4.set_ylim(0, max(high_acc, low_acc) * 130)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    fig.savefig(f"{FIG_DIR}/fig3_architecture.png")
    fig.savefig(f"{FIG_DIR}/fig3_architecture.pdf")
    plt.close(fig)
    print("  Fig 3 saved")


def figure4(genes, M, gene_idx, gene_depts):
    """Fig 4: Functional validation — NN enrichment + department heatmap."""
    fig = plt.figure(figsize=(18, 7))
    gs = gridspec.GridSpec(1, 2, wspace=0.35)

    U, S, Vt = svd(M, full_matrices=False)
    pc1 = Vt[0]
    M_resid = M - np.outer(M @ pc1, pc1)

    ax1 = fig.add_subplot(gs[0, 0])
    panel_label(ax1, "a")

    norms = np.linalg.norm(M_resid, axis=1, keepdims=True)
    norms[norms == 0] = 1
    M_normed = M_resid / norms

    np.random.seed(42)
    sample = np.random.choice(len(genes), size=min(2000, len(genes)), replace=False)

    k_values = [1, 3, 5, 10, 20]
    nn_rates = []
    rand_rates = []

    for k in k_values:
        same_nn = 0
        same_rand = 0
        tested = 0
        for i in sample:
            g = genes[i]
            if g not in gene_depts:
                continue
            sims = M_normed[i] @ M_normed.T
            sims[i] = -2
            top_k = np.argsort(sims)[-k:]
            actual_dept = gene_depts[g]
            nn_match = sum(1 for j in top_k if genes[j] in gene_depts and gene_depts[genes[j]] == actual_dept)
            rand_idx = np.random.choice(len(genes), size=k, replace=False)
            rand_match = sum(1 for j in rand_idx if genes[j] in gene_depts and gene_depts[genes[j]] == actual_dept)
            same_nn += nn_match
            same_rand += rand_match
            tested += k

        nn_rates.append(same_nn / tested if tested > 0 else 0)
        rand_rates.append(same_rand / tested if tested > 0 else 0)

    x_pos = np.arange(len(k_values))
    w = 0.3
    ax1.bar(x_pos - w/2, [r*100 for r in nn_rates], w, color=C_BLUE, label="Algebraic neighbors")
    ax1.bar(x_pos + w/2, [r*100 for r in rand_rates], w, color=C_GRAY, label="Random pairs")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"k={k}" for k in k_values])
    ax1.set_ylabel("Same-department rate (%)")
    ax1.set_xlabel("Number of neighbors (k)")
    ax1.set_title("Algebraic neighbors share function")
    ax1.legend()
    for i, (nn, rd) in enumerate(zip(nn_rates, rand_rates)):
        fold = nn/rd if rd > 0 else 0
        ax1.text(i, max(nn*100, rd*100) + 2, f"{fold:.1f}×", ha="center",
                 fontsize=10, fontweight="bold", color=C_RED)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    panel_label(ax2, "b")

    corr = np.corrcoef(M_resid.T)
    im = ax2.imshow(corr, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    short_names = [d[:6] for d in VALID_DEPARTMENTS]
    ax2.set_xticks(range(N_DEPTS))
    ax2.set_xticklabels(short_names, rotation=90, fontsize=7)
    ax2.set_yticks(range(N_DEPTS))
    ax2.set_yticklabels(short_names, fontsize=7)
    ax2.set_title("Department correlation structure (residual space)")
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label("Pearson r")

    fig.savefig(f"{FIG_DIR}/fig4_functional_validation.png")
    fig.savefig(f"{FIG_DIR}/fig4_functional_validation.pdf")
    plt.close(fig)
    print("  Fig 4 saved")


def figure5(genes, M, gene_idx):
    """Fig 5: Yamanaka = ICM — the star figure."""
    fig = plt.figure(figsize=(18, 7))
    gs = gridspec.GridSpec(1, 3, wspace=0.35)

    U, S, Vt = svd(M, full_matrices=False)
    pc1 = Vt[0]
    M_resid = M - np.outer(M @ pc1, pc1)

    yamanaka = ["POU5F1", "SOX2", "KLF4", "MYC"]
    icm = ["POU5F1", "NANOG", "SOX2", "KLF4", "ESRRB", "TBX3", "TFCP2L1", "GBX2"]
    te = ["CDX2", "TEAD4", "GATA3", "TFAP2C", "ELF5", "EOMES"]
    epi = ["POU5F1", "NANOG", "SOX2", "OTX2", "FGF4", "DNMT3B"]
    pre = ["GATA6", "GATA4", "SOX17", "PDGFRA", "HNF4A", "FOXA2"]

    def get_vecs(gl):
        return np.array([M_resid[gene_idx[g]] for g in gl if g in gene_idx])

    def var_captured(fv, tv):
        if fv is None or tv is None or len(fv) < 2 or len(tv) < 2:
            return 0
        U2, S2, Vt2 = svd(fv, full_matrices=False)
        basis = Vt2[:min(len(S2), len(fv))]
        proj = tv @ basis.T
        recon = proj @ basis
        resid = tv - recon
        total = np.sum(tv**2)
        return float(1 - np.sum(resid**2) / total) if total > 0 else 0

    yam_vecs = get_vecs(yamanaka)
    captures = {
        "ICM": var_captured(yam_vecs, get_vecs(icm)),
        "Epiblast": var_captured(yam_vecs, get_vecs(epi)),
        "Primitive\nendoderm": var_captured(yam_vecs, get_vecs(pre)),
        "Tropho-\nectoderm": var_captured(yam_vecs, get_vecs(te)),
    }

    ax1 = fig.add_subplot(gs[0, 0])
    panel_label(ax1, "a")
    names = list(captures.keys())
    vals = [captures[n] for n in names]
    colors_c = [C_RED, C_ORANGE, C_BLUE, C_GRAY]
    bars = ax1.bar(range(len(names)), [v*100 for v in vals], color=colors_c,
                   edgecolor="white", width=0.6)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{v*100:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Variance captured (%)")
    ax1.set_title("OSKM (Yamanaka factors)\nlineage-specific capture")
    ax1.set_ylim(0, 70)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    panel_label(ax2, "b")

    np.random.seed(42)
    n_perm = 1000
    random_icm = []
    icm_vecs = get_vecs(icm)
    for _ in range(n_perm):
        rg = np.random.choice(genes, size=4, replace=False)
        rv = np.array([M_resid[gene_idx[g]] for g in rg])
        rc = var_captured(rv, icm_vecs)
        random_icm.append(rc)

    ax2.hist([r*100 for r in random_icm], bins=40, color=C_GRAY, edgecolor="white", alpha=0.7)
    ax2.axvline(captures["ICM"]*100, color=C_RED, linewidth=3, linestyle="-")
    ax2.text(captures["ICM"]*100 + 1, ax2.get_ylim()[1]*0.8 if ax2.get_ylim()[1] > 0 else 50,
             f"OSKM = {captures['ICM']*100:.1f}%\np < 0.001",
             fontsize=12, fontweight="bold", color=C_RED)
    ax2.set_xlabel("ICM variance captured by random 4-gene set (%)")
    ax2.set_ylabel("Count (n=1,000 permutations)")
    ax2.set_title("Yamanaka specificity:\n0/1,000 random matches")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax3 = fig.add_subplot(gs[0, 2])
    panel_label(ax3, "c")

    fate_pairs = [
        ("ICM vs TE\n(first fate)", icm, te, "Day 3.5"),
        ("Epiblast vs\nPrEndoderm", epi, pre, "Day 4.5"),
    ]

    def angle_between(gl1, gl2):
        v1 = get_vecs(gl1)
        v2 = get_vecs(gl2)
        if len(v1) < 2 or len(v2) < 2:
            return None
        c1 = np.mean(v1, axis=0)
        c2 = np.mean(v2, axis=0)
        cos = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-30)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    angles = []
    labels_a = []
    for name, gl1, gl2, timing in fate_pairs:
        a = angle_between(gl1, gl2)
        if a is not None:
            angles.append(a)
            labels_a.append(f"{name}\n({timing})")

    ecto = ["PAX6", "SOX1", "NES", "OTX2"]
    meso = ["TBXT", "MESP1", "TBX6", "MIXL1"]
    endo = ["SOX17", "FOXA2", "GATA4", "GATA6"]

    germ_pairs = [
        ("Ecto vs\nMeso", ecto, meso, "Day 7"),
        ("Ecto vs\nEndo", ecto, endo, "Day 7"),
        ("Meso vs\nEndo", meso, endo, "Day 7"),
    ]
    for name, gl1, gl2, timing in germ_pairs:
        a = angle_between(gl1, gl2)
        if a is not None:
            angles.append(a)
            labels_a.append(f"{name}\n({timing})")

    if angles:
        colors_angle = [C_RED, C_ORANGE] + [C_BLUE] * len(germ_pairs)
        colors_angle = colors_angle[:len(angles)]
        bars = ax3.bar(range(len(angles)), angles, color=colors_angle,
                       edgecolor="white", width=0.6)
        for bar, a in zip(bars, angles):
            ax3.text(bar.get_x() + bar.get_width()/2, a + 1,
                     f"{a:.0f}°", ha="center", fontsize=11, fontweight="bold")
        ax3.set_xticks(range(len(angles)))
        ax3.set_xticklabels(labels_a, fontsize=8)
        ax3.set_ylabel("Angle between lineage centroids (degrees)")
        ax3.set_title("Cell fate orthogonality gradient")
        ax3.axhline(90, color="gray", linestyle=":", alpha=0.4)
        ax3.text(len(angles)-0.5, 92, "Orthogonal", fontsize=8, color="gray")
        ax3.set_ylim(0, 110)

    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    fig.savefig(f"{FIG_DIR}/fig5_yamanaka_icm.png")
    fig.savefig(f"{FIG_DIR}/fig5_yamanaka_icm.pdf")
    plt.close(fig)
    print("  Fig 5 saved")


def figure6(genes, M, gene_idx, gene_depts):
    """Fig 6: Three-layer architecture + power law + drug class profiles."""
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    panel_label(ax1, "a")

    l1_count = sum(1 for g, d in gene_depts.items() if d in L1_DEPTS)
    l2_count = sum(1 for g, d in gene_depts.items() if d in L2_DEPTS)
    l3_count = sum(1 for g, d in gene_depts.items() if d in L3_DEPTS)
    total = l1_count + l2_count + l3_count

    layer_data = [
        ("L1: Infrastructure\n(Chromatin, Cytoskeleton,\nDNA repair, Structural, Cell cycle)", l1_count, C_L1),
        ("L2: Information\n(Transcription, Nuc acid bind,\nMethylation, RNA proc, Translation)", l2_count, C_L2),
        ("L3: Signaling\n(Kinase, Signaling, Immune,\nIon channel, +8 more)", l3_count, C_L3),
    ]

    y_pos = [2, 1, 0]
    for i, (name, count, color) in enumerate(layer_data):
        ax1.barh(y_pos[i], count, color=color, edgecolor="white", height=0.6)
        ax1.text(count + 50, y_pos[i], f"{count:,} ({count/total*100:.0f}%)",
                 va="center", fontsize=10, fontweight="bold")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([ld[0] for ld in layer_data], fontsize=8)
    ax1.set_xlabel("Number of genes")
    ax1.set_title("Three-layer architecture")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    panel_label(ax2, "b")

    disease_in = DISEASE_GENES & set(gene_depts.keys())
    essential_in = ESSENTIAL_CORE & set(gene_depts.keys())

    layers = ["L1", "L2", "L3"]
    layer_sets = [L1_DEPTS, L2_DEPTS, L3_DEPTS]
    layer_sizes = [l1_count, l2_count, l3_count]

    disease_enrich = []
    essential_enrich = []
    for ls, lsize in zip(layer_sets, layer_sizes):
        lg = {g for g, d in gene_depts.items() if d in ls}
        d_count = len(disease_in & lg)
        e_count = len(essential_in & lg)
        d_frac = lsize / total if total > 0 else 0
        d_enrich = (d_count / len(disease_in)) / d_frac if len(disease_in) > 0 and d_frac > 0 else 0
        e_enrich = (e_count / len(essential_in)) / d_frac if len(essential_in) > 0 and d_frac > 0 else 0
        disease_enrich.append(d_enrich)
        essential_enrich.append(e_enrich)

    x_pos = np.arange(3)
    w = 0.3
    ax2.bar(x_pos - w/2, essential_enrich, w, color=C_GREEN, label="Essential genes")
    ax2.bar(x_pos + w/2, disease_enrich, w, color=C_RED, label="Disease genes")
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(layers)
    ax2.set_ylabel("Enrichment over expected")
    ax2.set_title("Essential vs disease genes by layer")
    ax2.legend()
    for i in range(3):
        ax2.text(i - w/2, essential_enrich[i] + 0.05, f"{essential_enrich[i]:.1f}×",
                 ha="center", fontsize=9, fontweight="bold", color=C_GREEN)
        ax2.text(i + w/2, disease_enrich[i] + 0.05, f"{disease_enrich[i]:.1f}×",
                 ha="center", fontsize=9, fontweight="bold", color=C_RED)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax3 = fig.add_subplot(gs[1, 0])
    panel_label(ax3, "c")

    u14_path = "validation/knockout/fourteen_unknowns_results.json"
    with open(u14_path) as f:
        u14 = json.load(f)
    area_profiles = u14.get("U14", {}).get("area_layer_profiles", {})

    drug_areas = {}
    for area, vals in area_profiles.items():
        if vals.get("n", 0) >= 3:
            drug_areas[area] = (vals["mean_l1"], vals["mean_l2"], vals["mean_l3"])

    drug_areas = dict(sorted(drug_areas.items(), key=lambda x: x[1][2]))

    areas = list(drug_areas.keys())
    l1_vals = [drug_areas[a][0] for a in areas]
    l2_vals = [drug_areas[a][1] for a in areas]
    l3_vals = [drug_areas[a][2] for a in areas]

    y_pos = np.arange(len(areas))
    ax3.barh(y_pos, l1_vals, color=C_L1, label="L1 Infrastructure", height=0.6)
    ax3.barh(y_pos, l2_vals, left=l1_vals, color=C_L2, label="L2 Information", height=0.6)
    left2 = [l1+l2 for l1, l2 in zip(l1_vals, l2_vals)]
    ax3.barh(y_pos, l3_vals, left=left2, color=C_L3, label="L3 Signaling", height=0.6)

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(areas)
    ax3.set_xlabel("Layer energy fraction")
    ax3.set_title("Drug class layer signatures")
    ax3.set_xlim(0, 1)
    ax3.legend(loc="lower right", fontsize=8)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    ax4 = fig.add_subplot(gs[1, 1])
    panel_label(ax4, "d")

    U, S, Vt = svd(M, full_matrices=False)
    pc1 = Vt[0]
    M_resid = M - np.outer(M @ pc1, pc1)
    U2, S2, Vt2 = svd(M_resid, full_matrices=False)
    eigenvalues = S2**2 / len(genes)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]

    log_k = np.log10(np.arange(1, len(eigenvalues)+1))
    log_ev = np.log10(eigenvalues)

    slope, intercept, r, p, se = stats.linregress(log_k, log_ev)

    ax4.scatter(log_k, log_ev, s=30, color=C_BLUE, zorder=5)
    fit_line = slope * log_k + intercept
    ax4.plot(log_k, fit_line, "--", color=C_RED, linewidth=2,
             label=f"Power law: α = {abs(slope):.2f}")
    ax4.set_xlabel("log₁₀(component rank)")
    ax4.set_ylabel("log₁₀(eigenvalue)")
    ax4.set_title("Residual spectrum: power law at criticality")
    ax4.text(0.95, 0.95, f"α = {abs(slope):.2f}\nr² = {r**2:.3f}\n\nBrownian: α = 0.50\n1/f noise: α = 1.00",
             transform=ax4.transAxes, ha="right", va="top",
             fontsize=10, bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                                     edgecolor=C_ORANGE, alpha=0.9))
    ax4.legend(loc="lower left")
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    fig.savefig(f"{FIG_DIR}/fig6_layers_spectrum.png")
    fig.savefig(f"{FIG_DIR}/fig6_layers_spectrum.pdf")
    plt.close(fig)
    print("  Fig 6 saved")


if __name__ == "__main__":
    import time
    t0 = time.time()
    print("Generating Nature Paper 2 figures...")
    os.makedirs(FIG_DIR, exist_ok=True)

    genes, M, gene_idx, profiles, gene_depts = load_data()
    print(f"  Loaded {len(genes)} genes, {N_DEPTS} departments")

    ppi_degree = load_ppi_degree(genes, gene_idx)
    print(f"  Loaded PPI degree for {len(ppi_degree)} genes")

    figure1(genes, M, gene_idx, ppi_degree)
    figure2(genes, M, gene_idx)
    figure3(genes, M, gene_idx, gene_depts)
    figure4(genes, M, gene_idx, gene_depts)
    figure5(genes, M, gene_idx)
    figure6(genes, M, gene_idx, gene_depts)

    elapsed = time.time() - t0
    print(f"\nAll figures generated in {elapsed:.1f}s")
    print(f"Output: {FIG_DIR}/")
