"""
kmer_eval.py
============
k-mer frequency Jensen-Shannon divergence between the real AMPs and every
generated set (model variants AND trivial baselines).

This metric captures SEQUENCE-ORDER structure, which amino-acid composition
and the physicochemical random-forest features cannot see. It is the main
reason the bigram-Markov baseline is worth including: it matches composition
almost perfectly, so any advantage the generator shows here is an advantage
in learned order, not in residue frequencies.

Change vs. the previous version
-------------------------------
Sample-size correction. The real set is much larger than a 2,000-sequence
generated set, and the k-mer space is large (20^4 = 160,000 for k=4), so the
smaller set is necessarily sparser and its JS divergence is inflated for
purely statistical reasons. We now subsample the real set to an equal number
of k-mer tokens and bootstrap, reporting mean +/- s.d. The uncorrected
whole-set value is also reported as `js_{k}mer_raw` for transparency -- do
not use it for between-method comparison.

Usage
-----
  python kmer_eval.py --real AMPuniqmoree5_clean_rmdup.fasta \
      --gen_dir ablation_results --out_dir ablation_results --ks 2 3 4
"""

import os
import argparse

import pandas as pd

from common import load_clean_fasta, read_fasta, clean_sequences, kmer_js, n_kmer_tokens

TARGETS = [
    ("Generated (GRU, with special)", "generated_with_special.fasta"),
    ("Generated (GRU, no special)",   "generated_no_special.fasta"),
    ("Generated (GRU)",               "generated_GRU.fasta"),
    ("Generated (LSTM)",              "generated_LSTM.fasta"),
    ("Baseline: random",              "generated_baseline_random.fasta"),
    ("Baseline: freq_matched",        "generated_baseline_freq_matched.fasta"),
    ("Baseline: training_sample",     "generated_baseline_training_sample.fasta"),
    ("Baseline: bigram_markov",       "generated_baseline_bigram_markov.fasta"),
]


def discover_targets(gen_dir):
    """
    Named groups first, then any other generated_*.fasta in the directory
    (the temperature-sweep outputs, for example), so a new arm is evaluated
    without editing this file.
    """
    import glob
    seen = {f for _, f in TARGETS}
    out = list(TARGETS)
    for p in sorted(glob.glob(os.path.join(gen_dir, "generated_*.fasta"))):
        f = os.path.basename(p)
        if f in seen:
            continue
        stem = f[len("generated_"):-len(".fasta")]
        label = (f"Generated (GRU, T={stem[1:]})"
                 if stem.startswith("T") and stem[1:2].isdigit()
                 else f"Generated ({stem})")
        out.append((label, f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="Real AMP FASTA.")
    ap.add_argument("--gen_dir", default="./ablation_results")
    ap.add_argument("--out_dir", default="./ablation_results")
    ap.add_argument("--ks", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--n_bootstrap", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading real AMPs...")
    real = load_clean_fasta(args.real)

    rows = []
    for name, fname in discover_targets(args.gen_dir):
        path = os.path.join(args.gen_dir, fname)
        if not os.path.exists(path):
            print(f"  [skip] {fname} not found")
            continue

        raw_seqs = read_fasta(path)
        gen = clean_sequences(raw_seqs)
        if not gen:
            # An explicit zero row, not a silent skip -- see ablation_rf.py.
            print(f"  {name:32s} 0/{len(raw_seqs)} valid -- reported as zero")
            row = {"group": name, "n": 0, "n_generated": len(raw_seqs)}
            for k in args.ks:
                row[f"js_{k}mer_mean"] = float("nan")
                row[f"js_{k}mer_sd"] = float("nan")
                row[f"js_{k}mer_raw"] = float("nan")
                row[f"js_{k}mer"] = "no valid sequences"
                row[f"n_tokens_{k}mer"] = 0
            rows.append(row)
            continue

        row = {"group": name, "n": len(gen), "n_generated": len(raw_seqs)}
        parts = []
        for k in args.ks:
            mean, sd, raw = kmer_js(real, gen, k,
                                    n_bootstrap=args.n_bootstrap, seed=args.seed)
            row[f"js_{k}mer_mean"] = round(mean, 4)
            row[f"js_{k}mer_sd"] = round(sd, 4)
            row[f"js_{k}mer_raw"] = round(raw, 4)
            row[f"js_{k}mer"] = f"{mean:.4f} ± {sd:.4f}"
            row[f"n_tokens_{k}mer"] = n_kmer_tokens(gen, k)
            parts.append(f"{k}mer={mean:.4f}±{sd:.4f}")
        rows.append(row)
        print(f"  {name:32s} n={len(gen):>5d}  " + "  ".join(parts))

    if not rows:
        print("\nNo generated FASTA files found. Run ablation.py and "
              "baseline_generation.py first.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "kmer_divergence.csv"), index=False)
    print(f"\nSaved: {args.out_dir}/kmer_divergence.csv")
    print("Note: values are sample-size corrected. Lower is closer to the "
          "real k-mer distribution; the copy oracle (training_sample) marks "
          "the attainable floor.")


if __name__ == "__main__":
    main()
