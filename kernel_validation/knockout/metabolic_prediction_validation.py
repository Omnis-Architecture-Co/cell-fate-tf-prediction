#!/usr/bin/env python3
"""
Layer 2d — Metabolic Liability Predictor Validation
====================================================
9-drug CYP metabolism recall test + inhibition flag verification.

Validates:
  1. CYP substrate prediction recall (expected enzyme relationships)
  2. True negative handling (renal clearance drugs)
  3. CYP inhibition risk detection (mibefradil CYP3A4+CYP2D6)

Ground truth: FDA drug labels, Rendic & Guengerich 2015, Zanger & Schwab 2013.
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from metabolic_liability_predictor import predict_metabolic_liabilities

DRUG_SMILES = {
    "Olaparib": {
        "smiles": "O=C(C1CC1)N1CCc2nc(=O)c3ccccc3n2CC1c1cc(F)ccc1",
        "target": "PARP1",
        "expected_cyp": ["CYP3A4"],
        "expected_clearance": "hepatic",
    },
    "Tamoxifen": {
        "smiles": "CC(/C=C/c1ccc(OCCN(C)C)cc1)=C(\\c1ccccc1)c1ccccc1",
        "target": "ESR1",
        "expected_cyp": ["CYP2D6", "CYP3A4"],
        "expected_clearance": "hepatic",
    },
    "Imatinib": {
        "smiles": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
        "target": "ABL1",
        "expected_cyp": ["CYP3A4", "CYP2D6"],
        "expected_clearance": "hepatic",
    },
    "Thalidomide": {
        "smiles": "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1",
        "target": "CRBN",
        "expected_cyp": [],
        "expected_clearance": "renal/hydrolysis",
    },
    "Metformin": {
        "smiles": "CN(C)C(=N)NC(=N)N",
        "target": "PRKAB1",
        "expected_cyp": [],
        "expected_clearance": "renal",
    },
    "Pioglitazone": {
        "smiles": "O=C1NC(=O)C(Cc2ccc(OCCc3ccccn3)cc2)S1",
        "target": "PPARG",
        "expected_cyp": ["CYP2C8", "CYP3A4"],
        "expected_clearance": "hepatic",
    },
    "Celecoxib": {
        "smiles": "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
        "target": "PTGS2",
        "expected_cyp": ["CYP2C9"],
        "expected_clearance": "hepatic",
    },
    "Atorvastatin": {
        "smiles": "CC(C)c1n(CC[C@@H](O)C[C@@H](O)CC(=O)O)c(-c2ccccc2)c(C(=O)Nc2ccccc2)c1-c1ccc(F)cc1",
        "target": "HMGCR",
        "expected_cyp": ["CYP3A4"],
        "expected_clearance": "hepatic",
    },
    "Mibefradil": {
        "smiles": "C(=O)(OC(C)C)CCN1CCC(C(c2cc(F)ccc2)OC)CC1",
        "target": "CACNA1G",
        "expected_cyp": ["CYP3A4", "CYP2D6"],
        "expected_clearance": "hepatic",
        "expected_inhibition": ["CYP3A4", "CYP2D6"],
    },
}


def run_validation():
    results = {
        "timestamp": datetime.now().isoformat(),
        "module": "Layer 2d — Metabolic Liability Predictor",
        "drugs_tested": len(DRUG_SMILES),
        "drug_results": {},
        "summary": {},
    }

    total_expected = 0
    total_found = 0
    total_inhibition_expected = 0
    total_inhibition_found = 0
    true_negative_correct = 0
    true_negative_total = 0
    failures = []

    for drug_name, info in DRUG_SMILES.items():
        prediction = predict_metabolic_liabilities(info["smiles"])
        predicted_enzymes = [e["gene_symbol"] for e in prediction["predicted_enzymes"]]
        inhibition_flags = [r["gene_symbol"] for r in prediction["cyp_inhibition_risk"]]

        expected = info["expected_cyp"]
        found = [e for e in expected if e in predicted_enzymes]
        missed = [e for e in expected if e not in predicted_enzymes]

        total_expected += len(expected)
        total_found += len(found)

        if not expected:
            true_negative_total += 1
            major_cyps = [e for e in predicted_enzymes if e.startswith("CYP")]
            if not major_cyps:
                true_negative_correct += 1
            else:
                failures.append(f"{drug_name}: false positive CYP prediction(s): {', '.join(major_cyps)}")

        if "expected_inhibition" in info:
            for inh_cyp in info["expected_inhibition"]:
                total_inhibition_expected += 1
                if inh_cyp in inhibition_flags:
                    total_inhibition_found += 1
                else:
                    failures.append(f"{drug_name}: missing {inh_cyp} inhibition flag")

        if missed:
            failures.append(f"{drug_name}: missed substrate {', '.join(missed)}")

        status = "PASS"
        if missed:
            status = "FAIL"
        elif not expected and not [e for e in predicted_enzymes if e.startswith("CYP")]:
            status = "PASS (true negative)"

        drug_result = {
            "status": status,
            "target": info["target"],
            "expected_cyp": expected,
            "predicted_enzymes": predicted_enzymes[:8],
            "found": found,
            "missed": missed,
            "mw": prediction["molecular_properties"]["molecular_weight"],
            "logp": prediction["molecular_properties"]["logp"],
        }

        if "expected_inhibition" in info:
            drug_result["expected_inhibition"] = info["expected_inhibition"]
            drug_result["detected_inhibition"] = inhibition_flags

        results["drug_results"][drug_name] = drug_result

        mark = "✓" if status.startswith("PASS") else "✗"
        print(f"  {mark} {drug_name:18s} target={info['target']:8s}  "
              f"expected={','.join(expected) or '(none)':20s}  "
              f"predicted={','.join(predicted_enzymes[:5]) or '(none)'}")
        if missed:
            print(f"    ↳ MISSED: {', '.join(missed)}")
        if "expected_inhibition" in info:
            inh_status = "✓" if all(c in inhibition_flags for c in info["expected_inhibition"]) else "✗"
            print(f"    ↳ Inhibition [{inh_status}]: expected={','.join(info['expected_inhibition'])}  "
                  f"detected={','.join(inhibition_flags)}")

    recall = total_found / total_expected * 100 if total_expected else 0
    inh_recall = total_inhibition_found / total_inhibition_expected * 100 if total_inhibition_expected else 0

    results["summary"] = {
        "substrate_recall": f"{total_found}/{total_expected} = {recall:.1f}%",
        "inhibition_recall": f"{total_inhibition_found}/{total_inhibition_expected} = {inh_recall:.1f}%",
        "cyp_true_negative_accuracy": f"{true_negative_correct}/{true_negative_total} (no major CYP predicted for renal-clearance drugs)",
        "failures": failures,
        "overall_pass": recall >= 85.0 and inh_recall >= 80.0 and len(failures) == 0,
    }

    print(f"\n{'='*60}")
    print(f"  Substrate recall:     {results['summary']['substrate_recall']}")
    print(f"  Inhibition recall:    {results['summary']['inhibition_recall']}")
    print(f"  CYP true negatives:   {results['summary']['cyp_true_negative_accuracy']}")
    print(f"  Overall:              {'PASS' if results['summary']['overall_pass'] else 'FAIL'}")
    if failures:
        print(f"  Failures:")
        for f in failures:
            print(f"    - {f}")
    print(f"{'='*60}")

    output_path = os.path.join(os.path.dirname(__file__), "metabolic_prediction_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {output_path}")

    return results["summary"]["overall_pass"]


if __name__ == "__main__":
    passed = run_validation()
    sys.exit(0 if passed else 1)
