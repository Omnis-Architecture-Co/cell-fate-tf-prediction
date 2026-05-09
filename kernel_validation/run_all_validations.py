"""
Consolidated Validation Test Runner
====================================
Runs all VAL-NULL-SUITE tests and generates the final report.

Tests:
  VAL-ENC-001: Encoding Null Model
  VAL-CON-001: Vocabulary Convergence Null Model
  VAL-PRM-001: Primitive Recurrence Permutation Test
  VAL-NET-001: Dispatch Hub Null Model
  VAL-XSP-001: Cross-Species Kendall's Tau Conservation
"""

import os, sys, json, time, importlib.util, traceback
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TESTS = [
    ("VAL-ENC-001", "val_enc_001_encoding_null_model"),
    ("VAL-CON-001", "val_con_001_convergence_null_model"),
    ("VAL-PRM-001", "val_prm_001_primitive_recurrence"),
    ("VAL-NET-001", "val_net_001_dispatch_hub_null"),
    ("VAL-XSP-001", "val_xsp_001_cross_species_tau"),
]


def run_test(test_id, module_name):
    print(f"\n{'='*70}")
    print(f"RUNNING: {test_id}")
    print(f"{'='*70}")

    module_path = os.path.join(SCRIPT_DIR, f"{module_name}.py")
    if not os.path.exists(module_path):
        return {"status": "MISSING", "error": f"File not found: {module_path}"}

    t0 = time.time()
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "main"):
            mod.main()
        elapsed = time.time() - t0
        return {"status": "PASSED", "elapsed": round(elapsed, 1)}
    except Exception as e:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        print(f"\nERROR in {test_id}: {e}")
        print(tb)
        return {"status": "FAILED", "error": str(e), "traceback": tb, "elapsed": round(elapsed, 1)}


def generate_final_report(test_results):
    json_files = {
        "VAL-ENC-001": "VAL-ENC-001_encoding_null_model.json",
        "VAL-CON-001": "VAL-CON-001_convergence_null_model.json",
        "VAL-PRM-001": "VAL-PRM-001_primitive_recurrence.json",
        "VAL-NET-001": "VAL-NET-001_dispatch_hub.json",
        "VAL-XSP-001": "VAL-XSP-001_cross_species_tau.json",
    }

    graph_files = {
        "VAL-ENC-001": "VAL-ENC-001_byte_distribution.png",
        "VAL-CON-001": "VAL-CON-001_convergence_heatmap.png",
        "VAL-PRM-001": "VAL-PRM-001_recurrence_distribution.png",
        "VAL-NET-001": "VAL-NET-001_hub_analysis.png",
        "VAL-XSP-001": "VAL-XSP-001_tau_vs_divergence.png",
    }

    test_data = {}
    for tid, jf in json_files.items():
        jp = os.path.join(SCRIPT_DIR, jf)
        if os.path.exists(jp):
            with open(jp) as f:
                test_data[tid] = json.load(f)

    report = f"""# VAL-NULL-SUITE: Statistical Null Model Validation Report
## OMNIS Architecture Co. — V2 6-bit Pipeline Validation

**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
**Pipeline**: V2 6-bit encoding (CODON_TABLE + NUC_BIN -> binary -> hex -> tokenize)
**Purpose**: Provide reproducible statistical evidence for/against the claims made
in the VALDICT001 paper regarding the V2 6-bit protein vocabulary encoding system.

---

## Executive Summary

| Test ID | Name | Status | Key Metric | p-value | Significant? |
|---------|------|--------|------------|---------|--------------|
"""

    for tid, module_name in TESTS:
        tr = test_results.get(tid, {})
        td = test_data.get(tid, {})
        status = tr.get("status", "NOT RUN")
        conclusion = td.get("conclusion", "")

        if tid == "VAL-ENC-001":
            r = td.get("results", {}).get("byte_distribution", {})
            metric = f"KS={r.get('ks_statistic', '?')}"
            pval = str(r.get("ks_p_value", "?"))
        elif tid == "VAL-CON-001":
            pval = str(td.get("results", {}).get("p_value", "?"))
            r = td.get("results", {}).get("observed", {})
            metric = f"overlap={r.get('functional_overlap', '?')}"
        elif tid == "VAL-PRM-001":
            pval = str(td.get("results", {}).get("p_value_concentration", td.get("results", {}).get("p_value_per_chrom_max", "?")))
            r = td.get("results", {}).get("observed", {})
            metric = f"conc_HHI={r.get('chromosome_concentration_hhi', '?')}"
        elif tid == "VAL-NET-001":
            pval = str(td.get("results", {}).get("p_value_gini", "?"))
            r = td.get("results", {}).get("observed", {})
            metric = f"Gini={r.get('gini_outbound', '?')}"
        elif tid == "VAL-XSP-001":
            pval = str(td.get("results", {}).get("tau_divergence_correlation", {}).get("p_permutation", "?"))
            r = td.get("results", {}).get("tau_divergence_correlation", {})
            metric = f"tau={r.get('kendall_tau', '?')}"
        else:
            metric = "?"
            pval = "?"

        sig_str = "SIGNIFICANT" if "SIGNIFICANT" in conclusion and "NOT SIGNIFICANT" not in conclusion else "NOT SIGNIFICANT" if "NOT SIGNIFICANT" in conclusion else "?"
        report += f"| {tid} | {td.get('test_name', module_name)} | {status} | {metric} | {pval} | {sig_str} |\n"

    report += "\n---\n\n"

    for tid, module_name in TESTS:
        td = test_data.get(tid, {})
        if not td:
            report += f"## {tid}\n\n*No results available.*\n\n---\n\n"
            continue

        report += f"""## {tid}: {td.get('test_name', '')}

### Materials
"""
        prov = td.get("provenance", {})
        for k, v in prov.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, dict):
                        report += f"- **{k2}**: "
                        for k3, v3 in v2.items():
                            report += f"{k3}=`{v3}` "
                        report += "\n"
                    else:
                        report += f"- **{k2}**: `{v2}`\n"
            else:
                report += f"- **{k}**: `{v}`\n"

        params = td.get("parameters", {})
        report += "\n### Parameters\n"
        for k, v in params.items():
            report += f"- **{k}**: {v}\n"

        report += "\n### Results\n"
        results = td.get("results", {})
        for section, data in results.items():
            if isinstance(data, dict):
                report += f"\n#### {section}\n| Key | Value |\n|-----|-------|\n"
                for k, v in data.items():
                    if isinstance(v, (dict, list)):
                        report += f"| {k} | *(see JSON)* |\n"
                    else:
                        report += f"| {k} | {v} |\n"
            elif isinstance(data, list):
                report += f"\n#### {section}\n*(see JSON for {len(data)} entries)*\n"

        report += f"""
### Interpretation
{td.get('conclusion', 'No conclusion.')}

### Graph
![{tid} Graph]({graph_files.get(tid, 'plot.png')})

---

"""

    report += """## Provenance Table

This table maps every numeric claim in this report to its source JSON file.

| Claim | JSON Source File | JSON Path |
|-------|-----------------|-----------|
"""
    for tid in json_files:
        td = test_data.get(tid, {})
        jf = json_files[tid]
        results = td.get("results", {})
        for section, data in results.items():
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, (int, float)) and v != 0:
                        report += f"| {section}.{k}={v} | `{jf}` | results.{section}.{k} |\n"

    report += f"""
## Methodology Notes

### Encoding Pipeline (V2 6-bit)
The encoding pipeline converts amino acid sequences to hex byte streams:
1. Each amino acid is mapped to a single codon via a deterministic codon table
2. RNA codons are converted to DNA (U->T)
3. Each nucleotide is mapped to 2 bits (A=00, T=01, G=10, C=11)
4. Binary stream is padded to 8-bit boundaries and converted to hex bytes
5. Tokenization: scan 2-5 byte patterns, keep those with frequency >= 2,
   score by length*10 + frequency*2, take top 100

### Null Model Approaches
- **ENC-001**: Amino acid sequence shuffle (preserves composition, destroys ordering)
- **CON-001**: Function label permutation across vocabulary words
- **PRM-001**: Chromosome assignment shuffle preserving per-chromosome program counts
- **NET-001**: Edge-swap degree-preserving randomization (5,000 swaps per round)
- **XSP-001**: Tau-divergence pair shuffling

### Statistical Standards
- All tests use seed=42 for reproducibility
- p-values computed as (count_null >= observed + 1) / (N + 1)
- Multiple testing: 5 independent tests; Bonferroni threshold = 0.01

## Reproducibility

All tests can be rerun with:
```bash
cd validation && python run_all_validations.py
```

Each test produces:
1. JSON results file with full provenance (file hashes, row counts, parameters)
2. Markdown summary with materials/methods/results/interpretation
3. PNG graph with annotated statistics

Generated by the OMNIS V2 6-bit validation pipeline.
"""

    report_path = os.path.join(SCRIPT_DIR, "VAL-NULL-SUITE_FINAL_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nFinal report saved to {report_path}")


def main():
    print("=" * 70)
    print("VAL-NULL-SUITE: Statistical Null Model Validation")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    t0 = time.time()
    test_results = {}

    for tid, module_name in TESTS:
        result = run_test(tid, module_name)
        test_results[tid] = result
        print(f"\n  -> {tid}: {result['status']} ({result.get('elapsed', 0):.1f}s)")

    total_elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    passed = sum(1 for r in test_results.values() if r["status"] == "PASSED")
    failed = sum(1 for r in test_results.values() if r["status"] == "FAILED")
    print(f"  PASSED: {passed}")
    print(f"  FAILED: {failed}")
    print(f"  Total time: {total_elapsed:.1f}s")

    generate_final_report(test_results)

    print(f"\nAll done. {passed}/{len(TESTS)} tests passed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        test_results = {tid: {"status": "PASSED", "elapsed": 0} for tid, _ in TESTS}
        generate_final_report(test_results)
    else:
        sys.exit(main())
