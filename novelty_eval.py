"""
novelty_eval.py
===============
NOVELTY / MEMORISATION screen: maximum sequence identity of every generated
sequence to the training set.

Why this file exists
--------------------
In the previous version the identity calculation lived inside
baseline_generation.py and was therefore applied ONLY to the trivial
baselines. The manuscript nevertheless listed
  "(v) maximum sequence identity to the training set to rule out memorization"
among the metrics -- a claim with no supporting number for the actual
generator. This script applies one identical procedure to every generated
set, model variants included.

Three further corrections
-------------------------
1. The metric is now Smith-Waterman local identity (BLOSUM62, BLAST-like gap
   costs) via Biopython, not `difflib.SequenceMatcher.ratio()`. The latter is
   a generic string-similarity heuristic and should not be called "sequence
   identity" in a biological venue. If Biopython is missing the script falls
   back to difflib and labels the metric accordingly in the output.
2. Every generated sequence is screened against the FULL training set (a
   3-mer containment prefilter narrows the candidates before alignment),
   rather than 200 sampled queries against 5,000 sampled references with an
   early break -- which made the reported "global maximum" a subsample
   maximum.
3. A distribution is reported, not a single number: mean, median, global
   maximum, the fraction above 90% identity, and the fraction of exact
   copies.

Wording for the manuscript: prefer "quantify the degree of memorisation"
over "rule out memorisation". A screen can bound memorisation; it cannot
rule it out.

Usage
-----
  python novelty_eval.py --real AMPuniqmoree5_clean_rmdup.fasta \
      --gen_dir ablation_results --out_dir ablation_results
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (load_clean_fasta, read_fasta, clean_sequences,
                    max_identity_to_reference, aligner_kind)

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
    ap.add_argument("--real", required=True, help="Training-set AMP FASTA.")
    ap.add_argument("--gen_dir", default="./ablation_results")
    ap.add_argument("--out_dir", default="./ablation_results")
    ap.add_argument("--max_query", type=int, default=0,
                    help="Cap on queries per set (0 = all). Use only if "
                         "runtime is prohibitive, and say so in the caption.")
    ap.add_argument("--top_k", type=int, default=60,
                    help="Reference candidates aligned per query after the "
                         "k-mer prefilter.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Loading training set...")
    real = load_clean_fasta(args.real)
    print(f"Identity metric: {aligner_kind()}\n")

    rows, per_seq = [], []
    for name, fname in discover_targets(args.gen_dir):
        path = os.path.join(args.gen_dir, fname)
        if not os.path.exists(path):
            print(f"  [skip] {fname} not found")
            continue

        raw_seqs = read_fasta(path)
        gen = clean_sequences(raw_seqs)
        if not gen:
            # An explicit zero row, not a silent skip -- see ablation_rf.py.
            print(f"  {name}: 0/{len(raw_seqs)} valid -- reported as zero")
            rows.append({
                "group": name, "n_query": 0, "n_generated": len(raw_seqs),
                "n_reference": len(real), "metric": aligner_kind(),
                "mean_max_identity": float("nan"),
                "median_max_identity": float("nan"),
                "global_max_identity": float("nan"),
                "frac_identity_gt_0.90": float("nan"),
                "frac_exact_copy": float("nan"),
            })
            continue

        if args.max_query and len(gen) > args.max_query:
            idx = rng.choice(len(gen), args.max_query, replace=False)
            gen = [gen[i] for i in idx]

        print(f"  {name} (n={len(gen)}) ...")
        res = max_identity_to_reference(gen, real, top_k=args.top_k)

        rows.append({
            "group": name,
            "n_query": len(gen),
            "n_generated": len(raw_seqs),
            "n_reference": len(real),
            "metric": res["metric"],
            "mean_max_identity": round(res["mean_max"], 4),
            "median_max_identity": round(res["median_max"], 4),
            "global_max_identity": round(res["global_max"], 4),
            "frac_identity_gt_0.90": round(res["frac_gt_0.90"], 4),
            "frac_exact_copy": round(res["frac_eq_1.00"], 4),
        })
        for v in res["identities"]:
            per_seq.append({"group": name, "max_identity": v})

        print(f"     mean={rows[-1]['mean_max_identity']}  "
              f"median={rows[-1]['median_max_identity']}  "
              f"max={rows[-1]['global_max_identity']}  "
              f">0.90={rows[-1]['frac_identity_gt_0.90']}  "
              f"exact={rows[-1]['frac_exact_copy']}")

    if not rows:
        print("\nNo generated FASTA files found. Run ablation.py and "
              "baseline_generation.py first.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "novelty_identity.csv"), index=False)
    pd.DataFrame(per_seq).to_csv(
        os.path.join(args.out_dir, "novelty_identity_per_sequence.csv"),
        index=False)

    print("\n[Novelty / memorisation screen]")
    print(df.to_string(index=False))

    # Distribution plot -- far more informative than a single summary number.
    psd = pd.DataFrame(per_seq)
    if psd.empty:
        print("\nNo per-sequence identities to plot.")
        return
    groups = [g for g in df["group"] if (psd["group"] == g).any()]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    data = [psd.loc[psd["group"] == g, "max_identity"].values for g in groups]
    parts = ax.violinplot(data, showmedians=True, widths=0.85)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    ax.axhline(0.90, color="red", linestyle="--", linewidth=1,
               label="90% identity")
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(groups, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Maximum identity to any training sequence")
    ax.set_ylim(0, 1.05)
    ax.set_title("Novelty screen: per-sequence maximum identity to the "
                 "training set", fontsize=11)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "novelty_identity.png"), dpi=300)
    plt.close()

    print(f"\nSaved: {args.out_dir}/novelty_identity.csv, "
          f"novelty_identity_per_sequence.csv, novelty_identity.png")
    print("The copy oracle (training_sample) should sit at 1.0 -- it is the "
          "positive control for this screen.")


if __name__ == "__main__":
    main()
