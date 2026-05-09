# The Genomic Kernel Predicts Cell Fate Transcription Factors

**Jasmine Levy, OMNIS Architecture Co.**

Companion code and data for:

- **Paper 2:** "The genomic kernel predicts cell fate transcription factors" ([Zenodo](https://doi.org/10.5281/zenodo.XXXXX))
- **Paper 1:** "A deterministic computational kernel encoded in the human genome" ([bioRxiv](https://doi.org/10.64898/2026.04.12.718009))

---

## Repository Contents

### Tables/

Twenty-two supplementary data tables (S1-S19, S21-S23) in CSV format, numbered to match the manuscript. TableS20 (genomic programs, 55 MB) is available on Zenodo due to file size.

| File | Description |
|---|---|
| TableS01 | Functional department definitions: 22 departments with word counts and enrichment |
| TableS02 | Gene/protein set: 17,322 genes with assigned functional departments |
| TableS03 | Set 1 combinations: 75 published cocktails with PubMed IDs and per-factor ranks |
| TableS04 | Set 2 combinations: 63 cocktails with scoring status and per-factor ranks |
| TableS05 | Set 3 combinations: 32 blind held-out cocktails |
| TableS06 | Set 1 factor rankings: percentile ranks among 3,166 Tier 2+ TFs |
| TableS07 | Set 2 factor rankings |
| TableS08 | Set 3 factor rankings (blind held-out) |
| TableS09 | Aggregate performance summary: top-1/2/5/10/20% recall across all sets |
| TableS10 | Per-family breakdown: 17 cell type families |
| TableS11 | Two-tier component weight vectors (Tier A and Tier B) |
| TableS12 | Component ablation: top-5% recall drop per removed component |
| TableS13 | Single-component baselines versus full model |
| TableS14 | Cell type marker database: 36 cell types, 524 markers |
| TableS15 | Strict factor holdout analysis |
| TableS16 | False positive characterization: top-50 non-cocktail gene breakdown |
| TableS17 | Top-50 prediction detail: 6,700 gene-level entries |
| TableS18 | Method comparison: OMNIS versus Mogrify head-to-head |
| TableS19 | Prospective predictions: top 10 candidates for 10 underexplored lineages |
| TableS21 | Reference gene lists (essential, OMIM, tumor suppressor, housekeeping, oncogene) |
| TableS22 | Vascular component decomposition: per-component scores |
| TableS23 | Phenotype annotation audit: OMIM/HPO coverage analysis |

### Figures/

Seven supplementary figures in PDF and PNG format.

| File | Description |
|---|---|
| SF1 | Set 3 blind validation detail: per-cocktail factor rank distributions |
| SF2 | PPI shuffle null control: 100-permutation degree-preserving shuffle |
| SF3 | Mogrify head-to-head comparison across 4 cell types |
| SF4 | Sensitivity and holdout analyses (4-panel) |
| SF5 | Validation performance across all sets and families (4-panel) |
| SF6 | Statistical controls: ablation, baselines, permutation nulls (4-panel) |
| SF7 | Scoring pipeline methods diagram |

### scoring_pipeline/

Core code for Paper 2. The nine-component scoring function that predicts cell fate transcription factors from protein interaction network structure.

| Script | Description |
|---|---|
| vm_cocktail_predictor.py | Core scoring function: 9-component composite score with two-tier adaptive weighting |
| validate_77_cocktails.py | Set 1 validation (75 combinations, 167 testable factors) |
| validate_extended_cocktails.py | Set 2 validation (63 combinations, 133 testable factors) |
| validate_set3_cocktails.py | Set 3 blind validation (32 combinations, 49 testable factors) |
| calibrate_weights.py | Two-tier weight optimization (Tier A: expression-dominant, Tier B: phenotype-dominant) |
| run_null_tests.py | Propensity-matched null and permutation tests |
| run_propensity_null_tier2plus.py | Propensity null against 3,166 Tier 2+ TF pool |
| false_positive_analysis.py | Top-50 false positive characterization |
| statistical_validation.py | Statistical tests (binomial, KS, one-sample t) |
| generate_figures.py | Main figure generation |
| generate_supp_figures.py | Supplementary figure generation |
| generate_v2_submission.py | Supplementary table generation |
| generate_comparison_and_predictions.py | Mogrify comparison and prospective predictions |
| reviewer_response_tests.py | Additional robustness tests |
| scoring_function_tuning.py | Scoring function parameter analysis |
| run_holdout_3166.py | Holdout validation at Tier 2+ pool size |
| verify_baseline.py | Baseline verification |

### kernel_validation/

Validation suite for Paper 1. Tests the structural properties of the genomic kernel described in the companion paper.

**Root scripts:**

| Script | Description |
|---|---|
| run_all_validations.py | Master runner for the full validation suite |
| val_enc_001_encoding_null_model.py | Encoding null model: random sequences fail vocabulary extraction |
| val_con_001_convergence_null_model.py | Convergence null: shuffled vocabularies fail functional coherence |
| val_dict_001_v6_consolidated.py | Dictionary validation: functional department prediction accuracy |
| val_net_001_dispatch_hub_null.py | Dispatch hub null: random networks fail relay architecture |
| val_prm_001_primitive_recurrence.py | Primitive recurrence: token frequency distributions versus null |
| val_xsp_001_cross_species_tau.py | Cross-species conservation of kernel structure |
| val_peel_addendum.py | Progressive department removal analysis |

**kernel_validation/knockout/** — Disruption profile validation (29 scripts): gene knockout simulation, community validations against DepMap essentiality data, LISP-like computational architecture tests, disease gene ground truth, algebraic structure analyses.

**kernel_validation/sensitivity/** — Robustness analyses (24 scripts): PPI graph shuffle, DepMap essentiality, Pfam domain comparison, isoform stability, parameter sensitivity, chromosomal independence, full pipeline negative controls.

---

## Validation Summary

| Set | Cocktails | Scored | Top-1% | Top-5% | Top-10% |
|---|---|---|---|---|---|
| Set 1 (calibration) | 75 | 57 | 37.7% | 77.8% | 83.2% |
| Set 2 (extended) | 63 | 52 | 33.8% | 64.7% | 76.7% |
| Set 3 (blind held-out) | 32 | 25 | 24.5% | 65.3% | 73.5% |
| **Combined** | **170** | **134** | **34.4%** | **71.1%** | **79.4%** |

Pool size: 3,166 Tier 2+ transcription factors. Propensity-matched null: 13.6% +/- 1.6% (Z = 35, p < 0.005).

---

## Related Repositories

- **Kernel code:** [obs-kernel](https://github.com/Omnis-Architecture-Co/obs-kernel)

## License

MIT License. See [LICENSE](LICENSE).

## Contact

Jasmine Levy — Jasmine.Levy@OmnisArchitecture.com
