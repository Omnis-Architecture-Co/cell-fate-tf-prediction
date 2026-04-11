# OMNIS Biological Simulator — Paper 2 Zenodo Supplementary Package
### "Kernel-based prediction of cell fate reprogramming factors from protein interaction disruption profiles"
**OMNIS Architecture Co. | Jasmine Levy**
Generated: 2026-04-11

---

## Package Contents

### Tables/
Eighteen supplementary data tables in CSV format.

| File | Description |
|---|---|
| TableS01_set1_factor_rankings.csv | Set 1 validation: per-factor percentile rankings (77 cocktails, 167 factors) |
| TableS02_set2_factor_rankings.csv | Set 2 validation: per-factor percentile rankings (64 cocktails, 133 factors) |
| TableS03_set1_combinations.csv | Set 1: all 75 published cocktails with PubMed IDs, scoring status, and factor ranks |
| TableS04_set2_combinations.csv | Set 2: all 63 cocktails with PubMed IDs, scoring status, and factor ranks |
| TableS05_set3_combinations.csv | Set 3 (blind): all 32 cocktails with scoring status — held-out set, scored without parameter adjustment |
| TableS06_family_breakdown.csv | Per-family top-5% and top-10% recall across all three validation sets |
| TableS07_component_ablation.csv | Component ablation: top-5% recall when each of the 9 scoring components is removed |
| TableS08_single_component_baselines.csv | Single-component baselines: top-5% recall for each component used alone |
| TableS09_strict_holdout_and_overlap.csv | Strict factor holdout analysis and cross-set factor overlap counts |
| TableS10_false_positive_characterization.csv | Top-50 non-cocktail gene breakdown by family (6,700 gene-slots examined) |
| TableS11_method_comparison.csv | Method comparison: OMNIS vs Mogrify vs TransSynW per cell type |
| TableS12_prospective_predictions.csv | Prospective predictions: top 10 candidate factors for 10 underexplored cell types |
| TableS13_set3_factor_rankings.csv | Set 3 blind validation: per-factor percentile rankings (33 cocktails, 49 factors) |
| TableS14_cell_type_markers.csv | Cell type marker database: 36 cell types, 524 marker genes with confidence levels and associated GTEx tissues |
| TableS18_mogrify_headtohead.csv | Mogrify head-to-head: per-factor ranks for OMNIS vs Mogrify on 4 cell type conversions. OMNIS 10/12 (83%) vs Mogrify 7/12 (58%) |
| TableS20_genomic_programs.csv | Genomic programs: 1,891 programs across 24 chromosomes from the combined protein_program_map |
| TableS21_reference_gene_lists.csv | Reference gene lists used in validation: ESSENTIAL_CORE (54), OMIM_disease (113), tumor_suppressor (36), housekeeping (34), oncogene (37) |
| TableS22_vascular_component_decomposition.csv | Vascular decomposition: all 9 per-component scores for 17 vascular/endothelial genes |

**Cocktail table notes (S03–S05):**
- `scoring_status`: "scored" = all factors in the 3,166-gene pool; "skipped: [reason]" = unmappable target, non-TF factors, or novel protocol
- `factor_ranks`: formatted as GENE:#rank(percentile%), e.g. GATA4:#3(0.1%)
- `top5_pct_recovery`: fraction of in-pool factors landing in the top 5% of ranked TFs

**S21 note:** Counts in the manuscript text (99/30/25 for OMIM/TSG/HK) reflect the intersection of these source lists with the DepMap 25Q3 tested gene universe (10,819 genes with ≥100 cell lines measured). The full source lists are provided here.

---

### Figures/
Seven supplementary figures (PNG at 300 DPI + vector PDF), numbered to match the manuscript.

| File | Description |
|---|---|
| SF1_set3_blind_validation_detail.{png,pdf} | **Set 3 blind validation detail.** Per-cocktail factor rank distributions for the 32 held-out cocktails scored without parameter revisitation. |
| SF2_ppi_shuffle_null_control.{png,pdf} | **PPI shuffle null control.** Performance when protein interaction edges are randomly rewired, preserving degree distribution (100-permutation null; z = 19.6–43.2 vs observed). |
| SF3_mogrify_headtohead.{png,pdf} | **Mogrify head-to-head comparison.** Per-factor rank percentiles for OMNIS vs Mogrify across 4 cell type conversions (Heart, Macrophage, Myoblast, Hepatocyte). OMNIS 10/12 (83%) vs Mogrify 7/12 (58%) at matched N. Full data in TableS18. |
| SF4_sensitivity_holdout.{png,pdf} | **Sensitivity and holdout analyses** (4-panel). (a) Two-tier vs single-tier weight system (+4–8pp advantage). (b) Kernel-only architecture ablation stack. (c) Strict factor holdout with 95% CI: Set 2 64.2%, Set 3 30.8% (n = 13). (d) Vascular synthetic rescue: ETV2/FLI1/ERG with median phenotype score; family top-5% 25% → 100%. |
| SF5_validation_performance.{png,pdf} | **Validation performance across all sets and families** (4-panel). (a) Recall vs rank threshold (0.5–20%) for Sets 1–3 and combined. (b) Cumulative factor rank distribution vs random. (c) Per-family top-5% recall heatmap (17 families). (d) Per-family factor percentile scatter. |
| SF6_statistical_controls.{png,pdf} | **Statistical controls** (4-panel). (a) Component ablation: mean top-5% drop per removed component (GTEx −18.1pp, Phenotype −12.7pp, …). (b) Single-component baselines vs full model (69.3%). (c) Gene-program permutation null (z = 31–43). (d) Propensity-matched null histogram: observed 71.1% vs null 13.6% ± 1.6% (z = 35, p < 0.005). |
| SF7_scoring_pipeline_methods.{png,pdf} | **Scoring pipeline methods diagram.** End-to-end workflow from protein interaction network to composite rank score. |

---

## Validation Summary

| Set | Cocktails (total) | Scored | Top-1% | Top-5% | Top-10% |
|---|---|---|---|---|---|
| Set 1 (training validation) | 75 | 57 | 37.7% | 77.8% | 83.2% |
| Set 2 (extended validation) | 63 | 52 | 33.8% | 64.7% | 76.7% |
| Set 3 (blind held-out) | 32 | 25 | 24.5% | 65.3% | 73.5% |
| **Combined** | **170** | **134** | **34.4%** | **71.1%** | **79.4%** |

Pool size: 3,166 Tier 2+ transcription factors  
Propensity-matched null: 13.6% ± 1.6% (real model 71.1%; z = 35, p < 0.005)

---

## Data Sources and Reproducibility

- Validation code: `paper2/vm_cocktail_predictor.py`, `paper2/validate_*.py`
- Table generation: `generate_paper2_verified_tables.py`, `export_paper2_supp_tables.py`
- Figure generation: `paper2/generate_supp_figures.py`
- DepMap essentiality data: DepMap 25Q3 CRISPRGeneEffect.csv (1,186 cancer cell lines)
- Cell type markers: CellMarker, PanglaoDB, and primary literature (see TableS14)
- Genomic programs: internal `protein_program_map` + `protein_program_map_v2` (see TableS20)
- Reference gene lists: `validation/knockout/community_validations.py` (see TableS21)

All source code is available in the associated GitHub repository.
