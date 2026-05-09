"""
VAL-ENC-001: Encoding Null Model
=================================
Tests whether the V2 6-bit encoding pipeline produces token/byte
distributions that are distinguishable from random (shuffled) sequences.

Method:
  1. Sample 1,000 proteins from complete_human_proteome (DB)
  2. Shuffle each protein's amino acid sequence (preserve length + composition)
  3. Encode both real and shuffled through the V2 6-bit pipeline
  4. Compare byte frequency distributions (KS test, chi-square)
  5. Compare token vocabulary hit rates

Input: BETA_DATABASE_URL (complete_human_proteome table)
       server/data/human/vocabulary.csv (1,932 words)
Output: validation/VAL-ENC-001_encoding_null_model.json
        validation/VAL-ENC-001_summary.md
        validation/VAL-ENC-001_byte_distribution.png
"""

import os, sys, json, random, hashlib, time
from collections import Counter
from datetime import datetime, timezone

import psycopg2
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DB_URL = os.environ.get("BETA_DATABASE_URL") or os.environ.get("DATABASE_URL")
SAMPLE_SIZE = 1000
SEED = 42
VOCAB_PATH = os.path.join(os.path.dirname(__file__), "..", "server", "data", "human", "vocabulary.csv")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

CODON_TABLE = {
    "A": "GCU", "C": "UGU", "D": "GAU", "E": "GAA", "F": "UUU",
    "G": "GGU", "H": "CAU", "I": "AUU", "K": "AAA", "L": "CUU",
    "M": "AUG", "N": "AAU", "P": "CCU", "Q": "CAA", "R": "CGU",
    "S": "UCU", "T": "ACU", "V": "GUU", "W": "UGG", "Y": "UAU",
}
NUC_BIN = {"A": "00", "T": "01", "G": "10", "C": "11"}


def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def file_rows(path):
    with open(path) as f:
        return sum(1 for _ in f) - 1


def encode_sequence(aa_seq):
    cleaned = "".join(c for c in aa_seq.upper() if c in CODON_TABLE)
    rna = "".join(CODON_TABLE.get(aa, "NNN") for aa in cleaned)
    dna = rna.replace("U", "T")
    binary = "".join(NUC_BIN.get(n, "00") for n in dna)
    if len(binary) % 8:
        binary += "0" * (8 - len(binary) % 8)
    hex_str = ""
    for i in range(0, len(binary), 8):
        hex_str += format(int(binary[i:i+8], 2), "02X")
    return hex_str


def tokenize_hex(hex_stream):
    byte_list = [hex_stream[i:i+2] for i in range(0, len(hex_stream), 2)]
    tokens = []
    for pat_len in range(2, 6):
        counts = Counter()
        for i in range(len(byte_list) - pat_len + 1):
            pat = "".join(byte_list[i:i+pat_len])
            counts[pat] += 1
        for pat, freq in counts.items():
            if freq >= 2:
                tokens.append(("0x" + pat, pat_len, freq))
    tokens.sort(key=lambda x: (-x[1] * 10 - min(x[2], 20) * 2))
    return [t[0] for t in tokens[:100]]


def load_vocabulary():
    vocab_set = set()
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH) as f:
            next(f)
            for line in f:
                parts = line.split(",")
                if parts:
                    vocab_set.add(parts[0].strip())
    return vocab_set


def fetch_proteins():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM complete_human_proteome WHERE sequence IS NOT NULL AND LENGTH(sequence) >= 50")
    total_available = cur.fetchone()[0]
    cur.execute("SELECT entry, gene_names_primary, sequence FROM complete_human_proteome WHERE sequence IS NOT NULL AND LENGTH(sequence) >= 50 ORDER BY random() LIMIT %s", (SAMPLE_SIZE * 2,))
    rows = cur.fetchall()
    conn.close()
    rng = random.Random(SEED)
    rng.shuffle(rows)
    return rows[:SAMPLE_SIZE], total_available


def main():
    print("VAL-ENC-001: Encoding Null Model")
    print("=" * 60)

    t0 = time.time()
    vocab = load_vocabulary()
    vocab_hash = file_hash(VOCAB_PATH)
    vocab_rows = file_rows(VOCAB_PATH)
    print(f"Loaded vocabulary: {len(vocab)} words (md5={vocab_hash}, {vocab_rows} rows)")

    proteins, total_available = fetch_proteins()
    print(f"Sampled {len(proteins)} proteins from DB (total available: {total_available})")

    real_bytes = Counter()
    shuf_bytes = Counter()
    real_vocab_hits = []
    shuf_vocab_hits = []
    real_token_counts = []
    shuf_token_counts = []
    real_entropy_list = []
    shuf_entropy_list = []

    rng = random.Random(SEED)

    for i, (entry, gene, seq) in enumerate(proteins):
        if i % 200 == 0:
            print(f"  Processing protein {i+1}/{len(proteins)}...")

        real_hex = encode_sequence(seq)
        for j in range(0, len(real_hex), 2):
            real_bytes[real_hex[j:j+2]] += 1

        real_tokens = tokenize_hex(real_hex)
        real_token_counts.append(len(real_tokens))
        hits = sum(1 for t in real_tokens if t in vocab)
        real_vocab_hits.append(hits)

        byte_vals = [int(real_hex[j:j+2], 16) for j in range(0, len(real_hex), 2)]
        if len(byte_vals) > 1:
            probs = np.array(list(Counter(byte_vals).values()), dtype=float)
            probs /= probs.sum()
            real_entropy_list.append(-np.sum(probs * np.log2(probs + 1e-15)))

        shuffled = list(seq)
        rng.shuffle(shuffled)
        shuffled_seq = "".join(shuffled)

        shuf_hex = encode_sequence(shuffled_seq)
        for j in range(0, len(shuf_hex), 2):
            shuf_bytes[shuf_hex[j:j+2]] += 1

        shuf_tokens = tokenize_hex(shuf_hex)
        shuf_token_counts.append(len(shuf_tokens))
        shuf_hits = sum(1 for t in shuf_tokens if t in vocab)
        shuf_vocab_hits.append(shuf_hits)

        byte_vals_s = [int(shuf_hex[j:j+2], 16) for j in range(0, len(shuf_hex), 2)]
        if len(byte_vals_s) > 1:
            probs_s = np.array(list(Counter(byte_vals_s).values()), dtype=float)
            probs_s /= probs_s.sum()
            shuf_entropy_list.append(-np.sum(probs_s * np.log2(probs_s + 1e-15)))

    all_byte_keys = sorted(set(real_bytes.keys()) | set(shuf_bytes.keys()))
    real_dist = np.array([real_bytes.get(k, 0) for k in all_byte_keys], dtype=float)
    shuf_dist = np.array([shuf_bytes.get(k, 0) for k in all_byte_keys], dtype=float)

    real_dist_norm = real_dist / real_dist.sum()
    shuf_dist_norm = shuf_dist / shuf_dist.sum()

    ks_stat, ks_p = stats.ks_2samp(
        np.repeat(np.arange(len(all_byte_keys)), real_dist.astype(int)),
        np.repeat(np.arange(len(all_byte_keys)), shuf_dist.astype(int)),
    )

    mask = (real_dist > 0) | (shuf_dist > 0)
    chi2_real = real_dist[mask]
    chi2_shuf = shuf_dist[mask]
    expected = (chi2_real + chi2_shuf) / 2
    chi2_stat = np.sum((chi2_real - expected) ** 2 / (expected + 1e-10)) + \
                np.sum((chi2_shuf - expected) ** 2 / (expected + 1e-10))
    chi2_df = int(mask.sum()) - 1
    chi2_p = 1.0 - stats.chi2.cdf(chi2_stat, chi2_df)

    vocab_t, vocab_t_p = stats.ttest_ind(real_vocab_hits, shuf_vocab_hits)
    entropy_t, entropy_t_p = stats.ttest_ind(real_entropy_list, shuf_entropy_list)

    elapsed = time.time() - t0

    results = {
        "test_id": "VAL-ENC-001",
        "test_name": "Encoding Null Model",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "sample_size": len(proteins),
            "seed": SEED,
            "vocabulary_size": len(vocab),
            "pipeline_version": "v2_6bit",
            "encoding": "AA → codon (deterministic) → DNA → binary → hex → tokenize (2-5 byte, freq≥2, top 100)",
        },
        "provenance": {
            "vocabulary_file": VOCAB_PATH,
            "vocabulary_md5": vocab_hash,
            "vocabulary_rows": vocab_rows,
            "database": "BETA_DATABASE_URL",
            "table": "complete_human_proteome",
            "total_proteins_available": total_available,
            "proteins_sampled": len(proteins),
        },
        "results": {
            "byte_distribution": {
                "ks_statistic": round(ks_stat, 6),
                "ks_p_value": float(f"{ks_p:.6e}"),
                "chi2_statistic": round(chi2_stat, 2),
                "chi2_df": chi2_df,
                "chi2_p_value": float(f"{chi2_p:.6e}"),
                "distinct_byte_values_real": int((real_dist > 0).sum()),
                "distinct_byte_values_shuffled": int((shuf_dist > 0).sum()),
                "total_bytes_real": int(real_dist.sum()),
                "total_bytes_shuffled": int(shuf_dist.sum()),
            },
            "vocabulary_hits": {
                "mean_real": round(float(np.mean(real_vocab_hits)), 2),
                "mean_shuffled": round(float(np.mean(shuf_vocab_hits)), 2),
                "std_real": round(float(np.std(real_vocab_hits)), 2),
                "std_shuffled": round(float(np.std(shuf_vocab_hits)), 2),
                "t_statistic": round(float(vocab_t), 4),
                "t_p_value": float(f"{vocab_t_p:.6e}"),
                "ratio_real_over_shuffled": round(float(np.mean(real_vocab_hits)) / max(float(np.mean(shuf_vocab_hits)), 0.01), 2),
            },
            "entropy": {
                "mean_real": round(float(np.mean(real_entropy_list)), 4),
                "mean_shuffled": round(float(np.mean(shuf_entropy_list)), 4),
                "t_statistic": round(float(entropy_t), 4),
                "t_p_value": float(f"{entropy_t_p:.6e}"),
            },
            "token_counts": {
                "mean_real": round(float(np.mean(real_token_counts)), 2),
                "mean_shuffled": round(float(np.mean(shuf_token_counts)), 2),
            },
        },
        "conclusion": "",
        "elapsed_seconds": round(elapsed, 1),
    }

    sig_byte = ks_p < 0.001
    sig_vocab = vocab_t_p < 0.001
    results["conclusion"] = (
        f"Byte distribution: {'SIGNIFICANT' if sig_byte else 'NOT SIGNIFICANT'} "
        f"(KS p={ks_p:.2e}, chi2 p={chi2_p:.2e}). "
        f"Vocabulary hits: {'SIGNIFICANT' if sig_vocab else 'NOT SIGNIFICANT'} "
        f"(real {np.mean(real_vocab_hits):.1f} vs shuffled {np.mean(shuf_vocab_hits):.1f}, "
        f"t={vocab_t:.2f}, p={vocab_t_p:.2e}). "
        f"Entropy: mean real {np.mean(real_entropy_list):.4f} vs shuffled {np.mean(shuf_entropy_list):.4f} "
        f"(p={entropy_t_p:.2e})."
    )

    json_path = os.path.join(OUT_DIR, "VAL-ENC-001_encoding_null_model.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("VAL-ENC-001: Encoding Null Model\n1,000 proteins real vs amino-acid-shuffled", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    x = np.arange(len(all_byte_keys))
    ax.bar(x - 0.2, real_dist_norm, 0.4, label="Real", alpha=0.7, color="#2196F3")
    ax.bar(x + 0.2, shuf_dist_norm, 0.4, label="Shuffled", alpha=0.7, color="#FF9800")
    ax.set_xlabel("Byte value (hex)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Byte Distribution (KS={ks_stat:.4f}, p={ks_p:.2e})")
    ax.legend()
    step = max(1, len(all_byte_keys) // 20)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([all_byte_keys[i] for i in range(0, len(all_byte_keys), step)], rotation=45, fontsize=6)

    ax = axes[0, 1]
    diff = real_dist_norm - shuf_dist_norm
    colors = ["#2196F3" if d > 0 else "#FF9800" for d in diff]
    ax.bar(x, diff, color=colors, alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Byte value index")
    ax.set_ylabel("Real − Shuffled frequency")
    ax.set_title("Byte Frequency Difference (blue=real higher)")

    ax = axes[1, 0]
    ax.hist(real_vocab_hits, bins=30, alpha=0.6, label=f"Real (μ={np.mean(real_vocab_hits):.1f})", color="#2196F3")
    ax.hist(shuf_vocab_hits, bins=30, alpha=0.6, label=f"Shuffled (μ={np.mean(shuf_vocab_hits):.1f})", color="#FF9800")
    ax.set_xlabel("Vocabulary hits per protein")
    ax.set_ylabel("Count")
    ax.set_title(f"Vocabulary Hit Rate (t={vocab_t:.2f}, p={vocab_t_p:.2e})")
    ax.legend()

    ax = axes[1, 1]
    ax.hist(real_entropy_list, bins=30, alpha=0.6, label=f"Real (μ={np.mean(real_entropy_list):.2f})", color="#2196F3")
    ax.hist(shuf_entropy_list, bins=30, alpha=0.6, label=f"Shuffled (μ={np.mean(shuf_entropy_list):.2f})", color="#FF9800")
    ax.set_xlabel("Shannon entropy (bits)")
    ax.set_ylabel("Count")
    ax.set_title(f"Byte Entropy Distribution (p={entropy_t_p:.2e})")
    ax.legend()

    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, "VAL-ENC-001_byte_distribution.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Plot saved to {png_path}")

    r = results["results"]
    md = f"""# VAL-ENC-001: Encoding Null Model

## Purpose
Test whether the V2 6-bit encoding pipeline produces byte and token distributions
distinguishable from randomly shuffled amino acid sequences. A shuffled sequence
preserves length and AA composition but destroys biological ordering.

## Method
1. Sample {len(proteins)} proteins from the human proteome (≥50 aa, {total_available} available)
2. For each protein, shuffle its amino acid sequence (preserve length and composition)
3. Encode both real and shuffled through the V2 6-bit pipeline:
   AA → deterministic codon → DNA → 2-bit binary → hex bytes → tokenize (2-5 byte patterns, freq≥2, top 100)
4. Compare byte frequency distributions (KS test, chi-square)
5. Compare vocabulary hit rates (t-test) and byte entropy

## Materials
| Item | Value |
|------|-------|
| Proteins sampled | {len(proteins)} / {total_available} available |
| Vocabulary file | `server/data/human/vocabulary.csv` |
| Vocabulary MD5 | `{vocab_hash}` |
| Vocabulary rows | {vocab_rows} |
| Vocabulary words | {len(vocab)} |
| Pipeline | V2 6-bit |
| Random seed | {SEED} |
| Source table | complete_human_proteome (BETA DB) |

## Results

### Byte Distribution
| Metric | Value |
|--------|-------|
| KS statistic | {r['byte_distribution']['ks_statistic']:.6f} |
| KS p-value | {r['byte_distribution']['ks_p_value']:.2e} |
| Chi-square | {r['byte_distribution']['chi2_statistic']:.2f} (df={r['byte_distribution']['chi2_df']}) |
| Chi-square p | {r['byte_distribution']['chi2_p_value']:.2e} |
| Distinct bytes (real) | {r['byte_distribution']['distinct_byte_values_real']} |
| Distinct bytes (shuffled) | {r['byte_distribution']['distinct_byte_values_shuffled']} |
| Total bytes (real) | {r['byte_distribution']['total_bytes_real']:,} |
| Total bytes (shuffled) | {r['byte_distribution']['total_bytes_shuffled']:,} |

### Vocabulary Hit Rate
| Metric | Real | Shuffled |
|--------|------|----------|
| Mean hits/protein | {r['vocabulary_hits']['mean_real']} | {r['vocabulary_hits']['mean_shuffled']} |
| Std | {r['vocabulary_hits']['std_real']} | {r['vocabulary_hits']['std_shuffled']} |
| t-statistic | {r['vocabulary_hits']['t_statistic']} | |
| t p-value | {r['vocabulary_hits']['t_p_value']:.2e} | |
| Ratio (real/shuffled) | {r['vocabulary_hits']['ratio_real_over_shuffled']}× | |

### Entropy
| Metric | Real | Shuffled |
|--------|------|----------|
| Mean entropy | {r['entropy']['mean_real']:.4f} | {r['entropy']['mean_shuffled']:.4f} |
| t p-value | {r['entropy']['t_p_value']:.2e} | |

## Interpretation
{results['conclusion']}

The encoding uses a deterministic codon table (one codon per AA), so the hex byte
distribution is entirely determined by amino acid composition and ordering. Shuffling
preserves composition but destroys local sequence context. This test establishes
whether the pipeline's signal comes from biological sequence ordering or merely
from amino acid composition.

## Graph
![Byte Distribution](VAL-ENC-001_byte_distribution.png)

## Provenance
| Claim | Source |
|-------|--------|
| {len(proteins)} proteins sampled | complete_human_proteome, BETA_DATABASE_URL |
| {len(vocab)} vocabulary words | `{VOCAB_PATH}` (MD5: `{vocab_hash}`) |
| KS={r['byte_distribution']['ks_statistic']:.6f} | Computed from {r['byte_distribution']['total_bytes_real']:,} real + {r['byte_distribution']['total_bytes_shuffled']:,} shuffled bytes |

## Runtime
{results['elapsed_seconds']:.1f} seconds
"""
    md_path = os.path.join(OUT_DIR, "VAL-ENC-001_summary.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Summary saved to {md_path}")
    print(f"\n{results['conclusion']}")


if __name__ == "__main__":
    main()
