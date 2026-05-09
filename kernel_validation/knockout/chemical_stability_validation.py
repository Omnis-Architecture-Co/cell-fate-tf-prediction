#!/usr/bin/env python3
"""
Chemical Stability & Formulation Risk Module — Validation Suite
===============================================================
Tests the three engines against well-documented historical cases:

1. Ranitidine — NDMA nitrosamine contamination (FDA recall 2020)
2. Nifedipine — Photodegradation (light-sensitive DHP)
3. Aspirin — Ester hydrolysis + excipient incompatibility
4. Omeprazole — Thioether oxidation (sulfoxide → sulfone)
5. Amlodipine — Maillard reaction with lactose
6. Penicillin G — Beta-lactam hydrolysis

Each test validates:
 - Correct degradation pathway identification
 - Correct severity assignment
 - Excipient compatibility flagging where applicable
 - Overall risk rating consistency
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from chemical_stability import (
    predict_degradation_pathways,
    predict_excipient_incompatibilities,
    predict_el_risks,
    assess_formulation_risk,
    RDKIT_AVAILABLE,
)

VALIDATION_CASES = {
    'ranitidine': {
        'smiles': 'CN/C(=C\\[S](=O)Cc1ccc(CN(C)C)o1)NC#N',
        'drug_name': 'Ranitidine',
        'expected_degradation_types': ['nitrosamine', 'oxidation'],
        'required_findings': [
            {'pathway_type': 'nitrosamine', 'reason': 'Dimethylamine/secondary amine → NDMA precursor (FDA recall 2020)'},
            {'pathway_type': 'oxidation', 'reason': 'Thioether oxidation (S → sulfoxide)'},
        ],
        'expected_min_risk': 'CRITICAL',
        'notes': 'Withdrawn worldwide 2020 due to NDMA contamination. Dimethylamine + nitrite → NDMA.',
    },
    'nifedipine': {
        'smiles': 'COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1[N+](=O)[O-]',
        'drug_name': 'Nifedipine',
        'expected_degradation_types': ['photolysis', 'hydrolysis'],
        'required_findings': [
            {'pathway_type': 'photolysis', 'reason': 'Nitroaromatic photo-reduction + DHP photo-oxidation'},
            {'pathway_type': 'hydrolysis', 'reason': 'Ester hydrolysis of methyl ester groups'},
        ],
        'expected_min_risk': 'HIGH',
        'notes': 'Classic photolabile drug. Must be stored in light-protective packaging (amber).',
    },
    'aspirin': {
        'smiles': 'CC(=O)Oc1ccccc1C(=O)O',
        'drug_name': 'Aspirin',
        'expected_degradation_types': ['hydrolysis'],
        'required_findings': [
            {'pathway_type': 'hydrolysis', 'reason': 'Ester hydrolysis → salicylic acid + acetic acid'},
        ],
        'expected_excipient_flags': ['transesterification'],
        'expected_min_risk': 'HIGH',
        'notes': 'Acetyl ester easily hydrolyzed. Vinegar smell = acetic acid = degraded aspirin.',
    },
    'omeprazole': {
        'smiles': 'COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1',
        'drug_name': 'Omeprazole',
        'expected_degradation_types': ['oxidation'],
        'required_findings': [
            {'pathway_type': 'oxidation', 'reason': 'Thioether → sulfoxide → sulfone oxidation'},
        ],
        'expected_min_risk': 'MODERATE',
        'notes': 'PPI class. Sulfoxide already present; further oxidation to sulfone is key degradant. MODERATE is correct since sulfoxide is already the active form.',
    },
    'amlodipine': {
        'smiles': 'CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl',
        'drug_name': 'Amlodipine',
        'expected_degradation_types': ['hydrolysis'],
        'required_findings': [
            {'pathway_type': 'hydrolysis', 'reason': 'Ester hydrolysis of ethyl/methyl ester'},
        ],
        'expected_excipient_flags': ['maillard_reaction'],
        'expected_min_risk': 'HIGH',
        'notes': 'Amlodipine besylate + lactose → Maillard browning (primary amine + reducing sugar).',
    },
    'penicillin_g': {
        'smiles': 'CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)Cc3ccccc3)C(=O)O)C',
        'drug_name': 'Penicillin G',
        'expected_degradation_types': ['hydrolysis'],
        'required_findings': [
            {'pathway_type': 'hydrolysis', 'reason': 'Beta-lactam ring hydrolysis'},
        ],
        'expected_min_risk': 'HIGH',
        'notes': 'Classic beta-lactam instability. Lactam ring opening → penicilloic acid.',
    },
    'caffeine_negative_control': {
        'smiles': 'Cn1c(=O)c2c(ncn2C)n(C)c1=O',
        'drug_name': 'Caffeine (negative control)',
        'expected_degradation_types': [],
        'required_findings': [],
        'expected_max_risk': 'MODERATE',
        'expected_min_risk': 'LOW',
        'notes': 'Stable xanthine — no esters, no primary amines, no thioethers, no photolabile groups. Should score LOW or MODERATE.',
        'is_negative_control': True,
    },
}


def run_validation():
    print("=" * 80)
    print("CHEMICAL STABILITY MODULE — VALIDATION SUITE")
    print("=" * 80)
    print(f"RDKit available: {RDKIT_AVAILABLE}")
    print()

    if not RDKIT_AVAILABLE:
        print("FATAL: RDKit not available. Cannot run validation.")
        return False

    results = {}
    total_checks = 0
    passed_checks = 0
    failed_checks = 0

    for case_name, case in VALIDATION_CASES.items():
        print(f"\n{'─' * 60}")
        print(f"TEST: {case['drug_name']} ({case_name})")
        print(f"SMILES: {case['smiles'][:60]}...")
        print(f"Notes: {case['notes']}")
        print(f"{'─' * 60}")

        result = assess_formulation_risk(case['smiles'], drug_name=case['drug_name'])
        results[case_name] = result

        if not result.get('success'):
            print(f"  FAIL: assess_formulation_risk returned success=False: {result.get('error')}")
            failed_checks += 1
            total_checks += 1
            continue

        total_checks += 1
        passed_checks += 1
        print(f"  [OK] Module returned success=True")

        deg = result.get('degradation_assessment', {})
        exc = result.get('excipient_assessment', {})
        el = result.get('el_assessment', {})

        found_pathway_types = set()
        for finding in deg.get('degradation_pathways', []):
            found_pathway_types.add(finding['pathway_type'])

        for req in case.get('required_findings', []):
            total_checks += 1
            req_type = req['pathway_type']
            if req_type in found_pathway_types:
                passed_checks += 1
                print(f"  [OK] Found required pathway: {req_type} — {req['reason']}")
            else:
                failed_checks += 1
                print(f"  [FAIL] Missing required pathway: {req_type} — {req['reason']}")
                print(f"         Found pathways: {found_pathway_types}")

        overall_risk = result.get('overall_formulation_risk', 'LOW')
        expected_min = case.get('expected_min_risk', 'LOW')
        expected_max = case.get('expected_max_risk')
        severity_rank = {'CRITICAL': 4, 'HIGH': 3, 'MODERATE': 2, 'LOW': 1}
        total_checks += 1
        meets_min = severity_rank.get(overall_risk, 0) >= severity_rank.get(expected_min, 0)
        meets_max = expected_max is None or severity_rank.get(overall_risk, 0) <= severity_rank.get(expected_max, 4)
        if meets_min and meets_max:
            passed_checks += 1
            if expected_max:
                print(f"  [OK] Overall risk: {overall_risk} (within range: {expected_min} to {expected_max})")
            else:
                print(f"  [OK] Overall risk: {overall_risk} (meets minimum: {expected_min})")
        else:
            failed_checks += 1
            if expected_max:
                print(f"  [FAIL] Overall risk: {overall_risk} (expected range: {expected_min} to {expected_max})")
            else:
                print(f"  [FAIL] Overall risk: {overall_risk} (expected minimum: {expected_min})")

        for expected_exc_flag in case.get('expected_excipient_flags', []):
            total_checks += 1
            found_exc_types = [f['interaction_type'] for f in exc.get('excipient_incompatibilities', [])]
            if expected_exc_flag in found_exc_types:
                passed_checks += 1
                print(f"  [OK] Found excipient incompatibility: {expected_exc_flag}")
            else:
                failed_checks += 1
                print(f"  [FAIL] Missing excipient incompatibility: {expected_exc_flag}")
                print(f"         Found: {found_exc_types}")

        deg_count = deg.get('total_vulnerabilities', 0)
        exc_count = exc.get('total_incompatibilities', 0)
        print(f"  Summary: {deg_count} degradation pathways, {exc_count} excipient incompatibilities, risk={overall_risk}")

    print(f"\n{'=' * 80}")
    print(f"VALIDATION RESULTS")
    print(f"{'=' * 80}")
    print(f"Total checks:  {total_checks}")
    print(f"Passed:        {passed_checks}")
    print(f"Failed:        {failed_checks}")
    recall = passed_checks / total_checks * 100 if total_checks > 0 else 0
    print(f"Pass rate:     {recall:.1f}%")
    print()

    for case_name, case in VALIDATION_CASES.items():
        r = results.get(case_name, {})
        risk = r.get('overall_formulation_risk', 'N/A')
        deg_n = r.get('degradation_assessment', {}).get('total_vulnerabilities', 0)
        exc_n = r.get('excipient_assessment', {}).get('total_incompatibilities', 0)
        print(f"  {case['drug_name']:20s} | risk={risk:10s} | {deg_n} degradation | {exc_n} excipient")

    output_path = os.path.join(os.path.dirname(__file__), 'chemical_stability_validation_results.json')
    with open(output_path, 'w') as f:
        json.dump({
            'validation_date': str(__import__('datetime').datetime.now()),
            'total_checks': total_checks,
            'passed': passed_checks,
            'failed': failed_checks,
            'pass_rate': recall,
            'cases': {k: {
                'drug_name': VALIDATION_CASES[k]['drug_name'],
                'overall_risk': v.get('overall_formulation_risk'),
                'degradation_count': v.get('degradation_assessment', {}).get('total_vulnerabilities', 0),
                'excipient_count': v.get('excipient_assessment', {}).get('total_incompatibilities', 0),
                'success': v.get('success', False),
            } for k, v in results.items()},
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return failed_checks == 0


if __name__ == '__main__':
    success = run_validation()
    sys.exit(0 if success else 1)
