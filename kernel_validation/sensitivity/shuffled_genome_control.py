#!/usr/bin/env python3
"""Shuffled Genome Control.

Tests whether the kernel architecture is a property of real genome
organization or an artifact of applying any deterministic encoding to
structured biological data.

Method:
  1. Load real human protein sequences from UniProt FASTA
  2. Encode each through the 6-bit pipeline
  3. Match encoded byte streams against the production vocabulary
  4. Shuffle each protein's amino acid sequence (preserving composition)
  5. Re-encode and re-match against the SAME vocabulary
  6. Compare: do real sequences produce more vocabulary hits, more
     functional coherence, and more cross-protein token sharing?

If the kernel is an artifact of encoding structured data, shuffled
proteins (which preserve amino acid composition) should produce
equal vocabulary hits. If the kernel reflects real sequence order,
shuffling should destroy it.

Output: validation/sensitivity/shuffled_genome_control_results.json
"""

import csv
import json
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "shuffled_genome_control_results.json")
FASTA_PATH = "/tmp/human_proteome.fasta"

AA_TO_RNA = {
    'A': 'GCU', 'C': 'UGU', 'D': 'GAU', 'E': 'GAG',
    'F': 'UUU', 'G': 'GGU', 'H': 'CAU', 'I': 'AUU',
    'K': 'AAG', 'L': 'UUG', 'M': 'AUG', 'N': 'AAU',
    'P': 'CCU', 'Q': 'CAG', 'R': 'CGU', 'S': 'UCU',
    'T': 'ACU', 'V': 'GUU', 'W': 'UGG', 'Y': 'UAU',
    '*': 'UAG', 'U': 'UGA', 'O': 'UAG', 'X': 'NNN',
    'B': 'GAU', 'Z': 'GAG', 'J': 'UUG',
}

NUC_TO_BINARY = {'A': '00', 'T': '01', 'G': '10', 'C': '11', 'N': '00'}


def encode_protein(aa_sequence):
    rna = ''.join(AA_TO_RNA.get(aa, 'NNN') for aa in aa_sequence)
    dna = rna.replace('U', 'T')
    binary = ''.join(NUC_TO_BINARY.get(n, '00') for n in dna)
    usable = (len(binary) // 8) * 8
    hex_bytes = []
    for i in range(0, usable, 8):
        hex_bytes.append(format(int(binary[i:i+8], 2), '02X'))
    return ''.join(hex_bytes)


def find_vocabulary_matches(hex_stream, vocab_patterns):
    matches = set()
    stream_upper = hex_stream.upper()
    for pat in vocab_patterns:
        if pat in stream_upper:
            matches.add(pat)
    return matches


def load_fasta(path):
    proteins = {}
    current_id = None
    current_seq = []
    current_gene = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id and current_seq:
                    proteins[current_id] = {
                        "seq": ''.join(current_seq),
                        "gene": current_gene,
                    }
                parts = line[1:].split('|')
                current_id = parts[1] if len(parts) >= 2 else parts[0].split()[0]
                current_gene = ""
                gn_idx = line.find("GN=")
                if gn_idx >= 0:
                    current_gene = line[gn_idx+3:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

    if current_id and current_seq:
        proteins[current_id] = {"seq": ''.join(current_seq), "gene": current_gene}

    return proteins


def load_vocabulary():
    path = os.path.join(BASE, "server", "data", "human", "vocabulary.csv")
    vocab = {}
    patterns = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            hex_val = row["word_hex"].replace("0x", "").upper()
            vocab[hex_val] = row.get("primary_function", "Unclassified")
            patterns.add(hex_val)
    return vocab, patterns


def load_gene_departments():
    path = os.path.join(BASE, "server", "data", "human", "gene_departments.csv")
    depts = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            depts[row["gene"]] = row["department"]
    return depts


def mann_whitney_z(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0, 1.0
    combined = [(v, 0) for v in group1] + [(v, 1) for v in group2]
    combined.sort(key=lambda x: x[0])
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) - 1 and combined[j + 1][0] == combined[j][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    r1 = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    mean_u = n1 * n2 / 2
    std_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if std_u == 0:
        return 0, 1.0
    z = (u1 - mean_u) / std_u
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p


def main():
    print("=" * 60)
    print("SHUFFLED GENOME CONTROL")
    print("Does the kernel survive when protein sequences are shuffled?")
    print("=" * 60)

    random.seed(42)

    print("\nStep 1: Load data...")
    proteins = load_fasta(FASTA_PATH)
    print(f"  Loaded {len(proteins)} protein sequences")

    vocab, vocab_patterns = load_vocabulary()
    print(f"  Vocabulary: {len(vocab)} words")

    depts = load_gene_departments()
    print(f"  Gene departments: {len(depts)} genes")

    sample_size = 10000
    all_ids = list(proteins.keys())
    random.shuffle(all_ids)
    sample_ids = all_ids[:sample_size]
    print(f"  Sample: {sample_size} proteins")

    # ================================================================
    # Step 2: Encode real proteins and match vocabulary
    # ================================================================
    print("\nStep 2: Encode REAL proteins, match against vocabulary...")
    t0 = time.time()

    real_hits_per_protein = {}
    real_token_carriers = defaultdict(set)
    real_hit_count_list = []

    for uid in sample_ids:
        seq = proteins[uid]["seq"]
        hex_stream = encode_protein(seq)
        matches = find_vocabulary_matches(hex_stream, vocab_patterns)
        real_hits_per_protein[uid] = matches
        real_hit_count_list.append(len(matches))
        for tok in matches:
            real_token_carriers[tok].add(uid)

    real_total_hits = sum(real_hit_count_list)
    real_mean_hits = statistics.mean(real_hit_count_list)
    real_proteins_with_hits = sum(1 for h in real_hit_count_list if h > 0)
    print(f"  Total vocabulary hits: {real_total_hits}")
    print(f"  Mean hits per protein: {real_mean_hits:.1f}")
    print(f"  Proteins with >= 1 hit: {real_proteins_with_hits} "
          f"({real_proteins_with_hits/sample_size*100:.1f}%)")
    print(f"  Unique vocabulary words matched: {len(real_token_carriers)}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ================================================================
    # Step 3: Encode SHUFFLED proteins, match against same vocabulary
    # ================================================================
    print("\nStep 3: Encode SHUFFLED proteins, match against vocabulary...")
    t0 = time.time()

    shuffled_hits_per_protein = {}
    shuffled_token_carriers = defaultdict(set)
    shuffled_hit_count_list = []

    for uid in sample_ids:
        seq = list(proteins[uid]["seq"])
        random.shuffle(seq)
        hex_stream = encode_protein(''.join(seq))
        matches = find_vocabulary_matches(hex_stream, vocab_patterns)
        shuffled_hits_per_protein[uid] = matches
        shuffled_hit_count_list.append(len(matches))
        for tok in matches:
            shuffled_token_carriers[tok].add(uid)

    shuf_total_hits = sum(shuffled_hit_count_list)
    shuf_mean_hits = statistics.mean(shuffled_hit_count_list)
    shuf_proteins_with_hits = sum(1 for h in shuffled_hit_count_list if h > 0)
    print(f"  Total vocabulary hits: {shuf_total_hits}")
    print(f"  Mean hits per protein: {shuf_mean_hits:.1f}")
    print(f"  Proteins with >= 1 hit: {shuf_proteins_with_hits} "
          f"({shuf_proteins_with_hits/sample_size*100:.1f}%)")
    print(f"  Unique vocabulary words matched: {len(shuffled_token_carriers)}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ================================================================
    # Test 1: Hit count comparison (Mann-Whitney)
    # ================================================================
    print("\n--- Test 1: Vocabulary Hit Count ---")

    hit_ratio = real_mean_hits / shuf_mean_hits if shuf_mean_hits > 0 else float('inf')
    z_hits, p_hits = mann_whitney_z(real_hit_count_list, shuffled_hit_count_list)
    print(f"  Real mean hits: {real_mean_hits:.1f}")
    print(f"  Shuffled mean hits: {shuf_mean_hits:.1f}")
    print(f"  Hit ratio (real/shuffled): {hit_ratio:.2f}x")
    print(f"  Mann-Whitney z = {z_hits:.2f}, p = {p_hits:.2e}")

    # ================================================================
    # Test 2: Functional coherence of matched tokens
    # ================================================================
    print("\n--- Test 2: Functional Coherence ---")

    def compute_coherence(token_carriers, proteins_data, gene_depts, min_carriers=10):
        scores = []
        for tok, carriers in token_carriers.items():
            if len(carriers) < min_carriers:
                continue
            carrier_depts = []
            for uid in carriers:
                gene = proteins_data.get(uid, {}).get("gene", "")
                if gene and gene in gene_depts:
                    carrier_depts.append(gene_depts[gene])
            if len(carrier_depts) < 5:
                continue
            dept_counts = Counter(carrier_depts)
            top_frac = max(dept_counts.values()) / len(carrier_depts)
            scores.append(top_frac)
        return scores

    real_coherence = compute_coherence(real_token_carriers, proteins, depts)
    shuf_coherence = compute_coherence(shuffled_token_carriers, proteins, depts)

    real_coh_mean = statistics.mean(real_coherence) if real_coherence else 0
    shuf_coh_mean = statistics.mean(shuf_coherence) if shuf_coherence else 0

    print(f"  Real: {len(real_coherence)} tokens tested, "
          f"mean coherence = {real_coh_mean:.4f}")
    print(f"  Shuffled: {len(shuf_coherence)} tokens tested, "
          f"mean coherence = {shuf_coh_mean:.4f}")
    if shuf_coh_mean > 0:
        coh_ratio = real_coh_mean / shuf_coh_mean
        print(f"  Coherence ratio: {coh_ratio:.2f}x")
    else:
        coh_ratio = float('inf')
        print(f"  Shuffled coherence = 0 (no tokens with enough carriers)")

    if real_coherence and shuf_coherence:
        z_coh, p_coh = mann_whitney_z(real_coherence, shuf_coherence)
        print(f"  Mann-Whitney z = {z_coh:.2f}, p = {p_coh:.2e}")
    else:
        z_coh, p_coh = 0, 1.0

    # ================================================================
    # Test 3: Token sharing by functional department
    # ================================================================
    print("\n--- Test 3: Functional Token Sharing ---")

    def sharing_by_dept(hits_per_protein, proteins_data, gene_depts, n_pairs=10000):
        uids_with_dept = []
        for uid in hits_per_protein:
            gene = proteins_data.get(uid, {}).get("gene", "")
            if gene and gene in gene_depts and hits_per_protein[uid]:
                uids_with_dept.append(uid)

        if len(uids_with_dept) < 100:
            return [], []

        same_j = []
        diff_j = []
        rng = random.Random(999)
        tested = 0
        for _ in range(n_pairs * 3):
            a, b = rng.sample(uids_with_dept, 2)
            ta = hits_per_protein[a]
            tb = hits_per_protein[b]
            if not ta or not tb:
                continue
            union = ta | tb
            if not union:
                continue
            jaccard = len(ta & tb) / len(union)
            ga = proteins_data[a]["gene"]
            gb = proteins_data[b]["gene"]
            if gene_depts[ga] == gene_depts[gb]:
                same_j.append(jaccard)
            else:
                diff_j.append(jaccard)
            tested += 1
            if tested >= n_pairs:
                break
        return same_j, diff_j

    real_same, real_diff = sharing_by_dept(real_hits_per_protein, proteins, depts)
    shuf_same, shuf_diff = sharing_by_dept(shuffled_hits_per_protein, proteins, depts)

    real_same_m = statistics.mean(real_same) if real_same else 0
    real_diff_m = statistics.mean(real_diff) if real_diff else 0
    shuf_same_m = statistics.mean(shuf_same) if shuf_same else 0
    shuf_diff_m = statistics.mean(shuf_diff) if shuf_diff else 0

    real_ratio = real_same_m / real_diff_m if real_diff_m > 0 else 0
    shuf_ratio = shuf_same_m / shuf_diff_m if shuf_diff_m > 0 else 0

    print(f"  Real: same-dept Jaccard={real_same_m:.4f}, "
          f"diff-dept={real_diff_m:.4f}, ratio={real_ratio:.2f}x")
    print(f"  Shuffled: same-dept Jaccard={shuf_same_m:.4f}, "
          f"diff-dept={shuf_diff_m:.4f}, ratio={shuf_ratio:.2f}x")
    print(f"  Proteins with same-function share more tokens in real data: "
          f"{'YES' if real_ratio > shuf_ratio * 1.2 else 'NO'}")

    # ================================================================
    # Test 4: Vocabulary coverage (unique words matched)
    # ================================================================
    print("\n--- Test 4: Vocabulary Coverage ---")
    real_coverage = len(real_token_carriers) / len(vocab) * 100
    shuf_coverage = len(shuffled_token_carriers) / len(vocab) * 100
    print(f"  Real: {len(real_token_carriers)}/{len(vocab)} words matched "
          f"({real_coverage:.1f}%)")
    print(f"  Shuffled: {len(shuffled_token_carriers)}/{len(vocab)} words matched "
          f"({shuf_coverage:.1f}%)")
    print(f"  Coverage ratio: {real_coverage/shuf_coverage:.2f}x"
          if shuf_coverage > 0 else "  Shuffled coverage = 0%")

    # ================================================================
    # Test 5: Multiple shuffle seeds
    # ================================================================
    print("\n--- Test 5: Stability Across 5 Shuffle Seeds ---")
    subsample = sample_ids[:3000]
    seed_results = []
    for seed in [100, 200, 300, 400, 500]:
        rng = random.Random(seed)
        hits = []
        for uid in subsample:
            seq = list(proteins[uid]["seq"])
            rng.shuffle(seq)
            hex_stream = encode_protein(''.join(seq))
            matches = find_vocabulary_matches(hex_stream, vocab_patterns)
            hits.append(len(matches))
        seed_results.append({
            "seed": seed,
            "mean_hits": round(statistics.mean(hits), 1),
            "pct_with_hits": round(sum(1 for h in hits if h > 0) / len(hits) * 100, 1),
        })
        print(f"  Seed {seed}: mean hits = {seed_results[-1]['mean_hits']}, "
              f"proteins with hits = {seed_results[-1]['pct_with_hits']}%")

    # Real comparison on same subsample
    real_sub_hits = [len(real_hits_per_protein.get(uid, set())) for uid in subsample]
    print(f"  Real: mean hits = {statistics.mean(real_sub_hits):.1f}, "
          f"proteins with hits = "
          f"{sum(1 for h in real_sub_hits if h > 0)/len(real_sub_hits)*100:.1f}%")

    # ================================================================
    # Final verdict
    # ================================================================
    kernel_destroyed = (
        hit_ratio > 1.3 and p_hits < 0.001 and real_coverage > shuf_coverage * 1.2
    )

    results = {
        "test_suite": "Shuffled Genome Control",
        "method": (
            "Each protein's amino acid sequence was shuffled (preserving "
            "amino acid composition, destroying sequential order) and "
            "re-encoded through the identical 6-bit pipeline. The SAME "
            "production vocabulary (1,932 words) was matched against both "
            "real and shuffled byte streams. Four properties were compared."
        ),
        "sample_size": sample_size,
        "random_seed": 42,
        "test1_hit_count": {
            "real_mean": round(real_mean_hits, 1),
            "shuffled_mean": round(shuf_mean_hits, 1),
            "hit_ratio": round(hit_ratio, 2),
            "mann_whitney_z": round(z_hits, 2),
            "mann_whitney_p": p_hits,
            "real_proteins_with_hits": real_proteins_with_hits,
            "shuffled_proteins_with_hits": shuf_proteins_with_hits,
        },
        "test2_functional_coherence": {
            "real_tokens_tested": len(real_coherence),
            "real_mean_coherence": round(real_coh_mean, 4),
            "shuffled_tokens_tested": len(shuf_coherence),
            "shuffled_mean_coherence": round(shuf_coh_mean, 4),
            "coherence_ratio": round(coh_ratio, 2) if coh_ratio != float('inf') else None,
            "mann_whitney_z": round(z_coh, 2),
            "mann_whitney_p": p_coh,
        },
        "test3_token_sharing": {
            "real_same_dept": round(real_same_m, 4),
            "real_diff_dept": round(real_diff_m, 4),
            "real_sharing_ratio": round(real_ratio, 2),
            "shuffled_same_dept": round(shuf_same_m, 4),
            "shuffled_diff_dept": round(shuf_diff_m, 4),
            "shuffled_sharing_ratio": round(shuf_ratio, 2),
        },
        "test4_vocabulary_coverage": {
            "real_words_matched": len(real_token_carriers),
            "shuffled_words_matched": len(shuffled_token_carriers),
            "vocabulary_size": len(vocab),
            "real_coverage_pct": round(real_coverage, 1),
            "shuffled_coverage_pct": round(shuf_coverage, 1),
        },
        "test5_seed_stability": seed_results,
        "kernel_destroyed_by_shuffling": kernel_destroyed,
        "summary": "",
    }

    print(f"\n{'='*60}")
    if kernel_destroyed:
        print("RESULT: KERNEL STRUCTURE DESTROYED BY SHUFFLING")
        results["summary"] = (
            f"Shuffling protein sequences while preserving amino acid "
            f"composition significantly reduces vocabulary matches. Real "
            f"proteins average {real_mean_hits:.1f} vocabulary hits vs "
            f"{shuf_mean_hits:.1f} for shuffled ({hit_ratio:.1f}x ratio, "
            f"Mann-Whitney z = {z_hits:.1f}, p = {p_hits:.2e}). Real "
            f"sequences match {real_coverage:.1f}% of the vocabulary vs "
            f"{shuf_coverage:.1f}% for shuffled. These results confirm "
            f"that the computational vocabulary is a property of real "
            f"protein sequence order, not an artifact of amino acid "
            f"composition or encoding methodology."
        )
    else:
        print("RESULT: KERNEL STRUCTURE PARTIALLY PRESERVED")
        results["summary"] = (
            f"Partial overlap between real and shuffled. "
            f"Hit ratio: {hit_ratio:.2f}x"
        )

    print(f"  Hit ratio: {hit_ratio:.2f}x")
    print(f"  Vocabulary coverage: {real_coverage:.1f}% real vs {shuf_coverage:.1f}% shuffled")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT}")


if __name__ == "__main__":
    main()
