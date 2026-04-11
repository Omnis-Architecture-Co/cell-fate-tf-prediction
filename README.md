# OMNIS Biological Simulator — Paper 2 Zenodo Supplementary Package
### "Kernel-based prediction of cell fate reprogramming factors from protein interaction disruption profiles"
**OMNIS Architecture Co. | Jasmine Levy**
Generated: 2026-04-11

---

## Package Contents

### Tables/
Twenty-three supplementary data tables in CSV format, numbered S1–S23 to match the manuscript.

| File | Description |
|---|---|
| TableS01_functional_department_definitions.csv | 22 functional departments defined by the genomic kernel vocabulary: department name, word count, % of vocabulary, mean enrichment |
| TableS02_gene_protein_set.csv | Full gene/protein set: 17,322 genes with their assigned functional department (GO-derived classification) |
| TableS03_set1_combinations.csv | Set 1: all 75 published cocktails with PubMed IDs, scoring status, and per-factor ranks |
| TableS04_set2_combinations.csv | Set 2: all 63 cocktails with PubMed IDs, scoring status, and per-factor ranks |
| TableS05_set3_combinations.csv | Set 3 (blind held-out): all 32 cocktails with scoring status — scored without parameter adjustment |
| TableS06_set1_factor_rankings.csv | Set 1: per-factor percentile rankings among 3,166 Tier 2+ TFs (167 testable factors) |
| TableS07_set2_factor_rankings.csv | Set 2: per-factor percentile rankings (133 testable factors) |
| TableS08_set3_factor_rankings.csv | Set 3: per-factor percentile rankings, blind held-out (49 testable factors) |
| TableS09_aggregate_summary.csv | Aggregate performance summary: top-1/2/5/10/20% recall for Sets 1, 2, 3, and combined |
| TableS10_family_breakdown.csv | Per-family top-5% and top-10% recall across all three validation sets (17 cell type families) |
| TableS11_two_tier_weights.csv | Two-tier component weight vectors: Tier A (expression-dominant) and Tier B (phenotype-dominant) for all 9 components |
| TableS12_component_ablation.csv | Component ablation: top-5% recall drop when each of the 9 components is removed (GTEx −18.1pp, Phenotype −12.7pp, …) |
| TableS13_single_component_baselines.csv | Single-component baselines: top-5% recall for each component used alone vs. full model (69.3%) |
| TableS14_cell_type_markers.csv | Cell type marker database: 36 cell types, 524 marker genes with confidence levels and GTEx tissue associations |
| TableS15_strict_factor_holdout.csv | Strict factor holdout: full vs. strict recall for Sets 2 and 3 with shared factors excluded; 41 shared factors listed |
| TableS16_false_positive_characterization.csv | False positive characterization: per-family breakdown of top-50 non-cocktail genes (known factors, cofactors, program neighbors, novel predictions) |
| TableS17_top50_prediction_detail.csv | Top-50 prediction detail: 6,700 gene-level entries across all cocktails with rank, score, percentile, and category |
| TableS18_method_comparison.csv | Method comparison (Mogrify head-to-head): per-factor ranks for OMNIS vs Mogrify on 4 cell type conversions. OMNIS 10/12 (83%) vs Mogrify 7/12 (58%) |
| TableS19_prospective_predictions.csv | Prospective predictions: top 10 candidate TFs for 10 underexplored cell types (100 predictions total) |
| TableS20_genomic_programs.csv | Genomic programs: 1,891 programs across 24 chromosomes from the protein_program_map |
| TableS21_reference_gene_lists.csv | Reference gene lists: ESSENTIAL_CORE (54), OMIM_disease (113), tumor_suppressor (36), housekeeping (34), oncogene (37) from community_validations.py |
| TableS22_vascular_component_decomposition.csv | Vascular decomposition: all 9 per-component scores for 17 vascular/endothelial and comparison genes |
| TableS23_phenotype_annotation_audit.csv | Phenotype annotation audit: OMIM/HPO coverage for each gene in S22; explains vascular family annotation gap (2/9 vascular genes covered vs. 7/17 overall) |

**Cocktail table notes (S3–S5):**
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
