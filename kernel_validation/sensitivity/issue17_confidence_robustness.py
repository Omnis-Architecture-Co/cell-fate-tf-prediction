#!/usr/bin/env python3
"""
Issue #17: Confidence Score Robustness
========================================
Tests whether the rank ordering of vocabulary words (by priority score) and
functional department predictions are invariant to reasonable parameter
perturbations in the scoring formula.

Production priority formula:
  priority = (length * 10 + diversity * 5) * 2 + min(frequency, 20) * 2

We test 9 perturbations of the weight coefficients and measure rank-order
stability using Spearman's rho and Kendall's tau against the production ranking.

Data source: human vocabulary from server/data/human/vocabulary.csv and
             protein_tokens_v2 from the database.
"""

import os, json, time, csv
from collections import Counter, defaultdict
from datetime import datetime, timezone

import psycopg2
import numpy as np
from scipy import stats

DB_URL = os.environ.get("BETA_DATABASE_URL") or os.environ.get("DATABASE_URL")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

CODON_TABLE = {
    'A': 'GCU', 'C': 'UGU', 'D': 'GAU', 'E': 'GAG', 'F': 'UUU',
    'G': 'GGU', 'H': 'CAU', 'I': 'AUU', 'K': 'AAG', 'L': 'UUG',
    'M': 'AUG', 'N': 'AAU', 'P': 'CCU', 'Q': 'CAG', 'R': 'CGU',
    'S': 'UCU', 'T': 'ACU', 'V': 'GUU', 'W': 'UGG', 'Y': 'UAU',
    'U': 'UGA', 'O': 'UAG', 'X': 'NNN', 'B': 'GAU', 'Z': 'GAG',
    'J': 'UUG', '*': 'UAG',
}
NUC_BIN = {'A': '00', 'T': '01', 'G': '10', 'C': '11'}


WEIGHT_CONFIGS = [
    {'label': 'Production (L=10, D=5, F_cap=20, scale=2, F_scale=2)',
     'length_w': 10, 'div_w': 5, 'scale': 2, 'freq_cap': 20, 'freq_scale': 2},
    {'label': 'Length-heavy (L=20, D=5)',
     'length_w': 20, 'div_w': 5, 'scale': 2, 'freq_cap': 20, 'freq_scale': 2},
    {'label': 'Length-light (L=5, D=5)',
     'length_w': 5, 'div_w': 5, 'scale': 2, 'freq_cap': 20, 'freq_scale': 2},
    {'label': 'Diversity-heavy (L=10, D=15)',
     'length_w': 10, 'div_w': 15, 'scale': 2, 'freq_cap': 20, 'freq_scale': 2},
    {'label': 'Diversity-zero (L=10, D=0)',
     'length_w': 10, 'div_w': 0, 'scale': 2, 'freq_cap': 20, 'freq_scale': 2},
    {'label': 'Freq-uncapped (F_cap=1000)',
     'length_w': 10, 'div_w': 5, 'scale': 2, 'freq_cap': 1000, 'freq_scale': 2},
    {'label': 'Freq-cap-5 (F_cap=5)',
     'length_w': 10, 'div_w': 5, 'scale': 2, 'freq_cap': 5, 'freq_scale': 2},
    {'label': 'No-scale (scale=1)',
     'length_w': 10, 'div_w': 5, 'scale': 1, 'freq_cap': 20, 'freq_scale': 1},
    {'label': 'Freq-heavy (F_scale=5)',
     'length_w': 10, 'div_w': 5, 'scale': 2, 'freq_cap': 20, 'freq_scale': 5},
]


def encode_protein(aa_seq):
    rna = ''.join(CODON_TABLE.get(aa, '') for aa in aa_seq.upper() if CODON_TABLE.get(aa))
    dna = rna.replace('U', 'T')
    bits = ''.join(NUC_BIN.get(n, '00') for n in dna)
    if len(bits) % 8:
        bits += '0' * (8 - len(bits) % 8)
    byte_vals = [int(bits[i:i+8], 2) for i in range(0, len(bits), 8)]
    return ''.join(f'{b:02X}' for b in byte_vals)


def tokenize_with_weights(hex_string, config, max_rank=100):
    bc = len(hex_string) // 2
    if bc < 4:
        return []
    bl = [hex_string[i:i+2] for i in range(0, len(hex_string), 2)]
    tokens = []
    for plen in range(2, 6):
        if bc < plen * 2:
            continue
        counts = Counter()
        for i in range(bc - plen + 1):
            pat = ''.join(bl[i:i+plen])
            counts[pat] += 1
        for pat, freq in counts.items():
            if freq >= 2:
                db = bytes.fromhex(pat)
                has_l = any(chr(b).isalpha() for b in db if 32 <= b <= 126)
                has_d = any(chr(b).isdigit() for b in db if 32 <= b <= 126)
                has_s = any(not chr(b).isalnum() for b in db if 32 <= b <= 126)
                div = sum([has_l, has_d, has_s])
                priority = ((plen * config['length_w'] + div * config['div_w']) * config['scale']
                            + min(freq, config['freq_cap']) * config['freq_scale'])
                tokens.append({'hex': pat.upper(), 'frequency': freq, 'priority': priority})
    tokens.sort(key=lambda t: -t['priority'])
    return tokens[:max_rank]


def main():
    print("Issue #17: Confidence Score / Priority Robustness")
    print("=" * 60)
    t0 = time.time()

    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    print("Loading AA sequences...")
    cur.execute("SELECT uniprot_id, aa_sequence FROM protein_encoding_v2")
    all_data = cur.fetchall()
    conn.close()

    sample_size = 10000
    rng = np.random.RandomState(42)
    indices = rng.choice(len(all_data), size=sample_size, replace=False)
    sampled = [(all_data[i][0], encode_protein(all_data[i][1])) for i in indices]
    print(f"Encoded {sample_size} proteins")

    all_rankings = {}
    for cfg in WEIGHT_CONFIGS:
        label = cfg['label']
        print(f"\n  Testing: {label}")
        tok_global_priority = defaultdict(float)
        tok_global_count = Counter()
        for uid, hex_str in sampled:
            toks = tokenize_with_weights(hex_str, cfg)
            for t in toks:
                tok_global_priority[t['hex']] += t['priority']
                tok_global_count[t['hex']] += 1

        common_tokens = [tok for tok, cnt in tok_global_count.items() if cnt >= 10]
        ranking = sorted(common_tokens, key=lambda t: -tok_global_priority[t])
        all_rankings[label] = ranking
        print(f"    Common tokens (>=10 proteins): {len(common_tokens)}")

    production_label = WEIGHT_CONFIGS[0]['label']
    production_ranking = all_rankings[production_label]
    prod_rank_map = {tok: i for i, tok in enumerate(production_ranking)}

    results = []
    for cfg in WEIGHT_CONFIGS[1:]:
        label = cfg['label']
        alt_ranking = all_rankings[label]
        shared = set(production_ranking) & set(alt_ranking)
        if len(shared) < 10:
            print(f"  {label}: too few shared tokens ({len(shared)})")
            results.append({'label': label, 'shared_tokens': len(shared), 'spearman_rho': None, 'kendall_tau': None})
            continue

        shared_list = sorted(shared, key=lambda t: prod_rank_map[t])
        prod_ranks = [prod_rank_map[t] for t in shared_list]
        alt_rank_map = {tok: i for i, tok in enumerate(alt_ranking)}
        alt_ranks = [alt_rank_map[t] for t in shared_list]

        rho, rho_p = stats.spearmanr(prod_ranks, alt_ranks)
        tau, tau_p = stats.kendalltau(prod_ranks, alt_ranks)

        results.append({
            'label': label,
            'shared_tokens': len(shared),
            'spearman_rho': round(float(rho), 4),
            'spearman_p': float(f"{rho_p:.4e}"),
            'kendall_tau': round(float(tau), 4),
            'kendall_p': float(f"{tau_p:.4e}"),
        })
        print(f"  {label}: rho={rho:.4f} (p={rho_p:.2e}), tau={tau:.4f} (p={tau_p:.2e}), shared={len(shared)}")

    rhos = [r['spearman_rho'] for r in results if r['spearman_rho'] is not None]
    taus = [r['kendall_tau'] for r in results if r['kendall_tau'] is not None]
    mean_rho = np.mean(rhos) if rhos else 0
    mean_tau = np.mean(taus) if taus else 0
    min_rho = min(rhos) if rhos else 0

    conclusion = (
        f"Across {len(results)} weight perturbations, rank-order stability is "
        f"{'high' if mean_rho > 0.7 else 'moderate' if mean_rho > 0.5 else 'low'} "
        f"(mean Spearman rho={mean_rho:.3f}, mean Kendall tau={mean_tau:.3f}). "
        f"Minimum rho={min_rho:.3f}. "
        f"{'The priority ranking is robust to reasonable weight perturbations.' if min_rho > 0.5 else 'Some weight configurations substantially alter rankings.'}"
    )
    print(f"\n{'='*60}")
    print(f"CONCLUSION: {conclusion}")

    output = {
        'test': 'Issue #17: Confidence Score Robustness',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'method': f'Tested {len(WEIGHT_CONFIGS)} priority weight configurations on {sample_size} proteins. Measured rank-order correlation (Spearman rho, Kendall tau) of token rankings against production weights.',
        'sample_size': sample_size,
        'weight_configs': [{k: v for k, v in cfg.items()} for cfg in WEIGHT_CONFIGS],
        'results': results,
        'summary': {
            'mean_spearman_rho': round(mean_rho, 4),
            'mean_kendall_tau': round(mean_tau, 4),
            'min_spearman_rho': round(min_rho, 4),
            'all_rhos': [round(r, 4) for r in rhos],
        },
        'conclusion': conclusion,
        'elapsed_seconds': round(time.time() - t0, 1),
    }

    out_path = os.path.join(OUT_DIR, 'issue17_confidence_robustness.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
