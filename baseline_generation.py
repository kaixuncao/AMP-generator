"""
baseline_generation.py
======================
GENERATION BASELINES -- simple/trivial models to compare the AMP generator
against.

Directly supports the submission checklist item
    "Baseline comparisons to simple/trivial models ... are provided."
(generation half; the classifier half lives in ablation_rf.py).

Baselines
---------
  1. random           : uniform over the 20 canonical amino acids
  2. freq_matched     : residues drawn at the training-set frequencies
  3. training_sample  : sequences copied from the training set  -- COPY ORACLE
  4. bigram_markov    : first-order Markov chain over residues

All baselines draw their lengths from the empirical AMP length distribution,
so no comparison is confounded by length.

On baseline 3
-------------
`training_sample` is NOT a competitor to be beaten. It is a copy oracle: it
defines the attainable CEILING on every distribution-matching metric (JS
divergences go to ~0) and the FLOOR on novelty (identity = 1.0 by
construction). It calibrates the range within which the other methods should
be read. Say so explicitly in the manuscript, otherwise a reader will notice
that the "best" baseline is plagiarism and wonder why it was presented as a
rival.

Changes vs. the previous version
--------------------------------
* Multiple seeds, reported as mean +/- s.d.
* Shared cleaning rules and metric implementations (common.py).
* k-mer JS added, with sample-size correction.
* The identity/memorisation screen moved to novelty_eval.py so that the SAME
  procedure is applied to the model-generated sequences too -- previously it
  was run on baselines only, leaving the manuscript's memorisation claim
  about the actual generator unsupported.

Usage
-----
  python baseline_generation.py --fasta AMPuniqmoree5_clean_rmdup.fasta \
      --out_dir ablation_results --n_generate 2000 --seeds 3
"""

import os
import argparse
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from common import (
    CANONICAL_AAS, LengthSampler, load_clean_fasta, write_fasta,
    validity_rate, diversity, js_length_divergence, kmer_js,
    aa_composition_l1, aggregate_seeds,
)


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------

def baseline_random(length_sampler, n, rng):
    lengths = length_sampler.sample(n, rng)
    idx = [rng.integers(0, len(CANONICAL_AAS), size=int(L)) for L in lengths]
    return ["".join(CANONICAL_AAS[i] for i in ii) for ii in idx]


def baseline_freq_matched(length_sampler, aa_freq, n, rng):
    aas = list(aa_freq.keys())
    probs = np.array([aa_freq[a] for a in aas], dtype=float)
    probs /= probs.sum()
    lengths = length_sampler.sample(n, rng)
    return ["".join(rng.choice(aas, size=int(L), p=probs)) for L in lengths]


def baseline_training_sample(real_seqs, n, rng):
    """Copy oracle -- see module docstring."""
    idx = rng.integers(0, len(real_seqs), size=n)
    return [real_seqs[i] for i in idx]


def baseline_bigram_markov(real_seqs, length_sampler, n, rng):
    trans = defaultdict(Counter)
    starts = Counter()
    for s in real_seqs:
        if len(s) < 2:
            continue
        starts[s[0]] += 1
        for a, b in zip(s[:-1], s[1:]):
            trans[a][b] += 1

    start_aas = list(starts)
    start_probs = np.array([starts[a] for a in start_aas], float)
    start_probs /= start_probs.sum()

    trans_probs = {}
    for a, counter in trans.items():
        nxt = list(counter)
        p = np.array([counter[b] for b in nxt], float)
        trans_probs[a] = (nxt, p / p.sum())

    out = []
    for L in length_sampler.sample(n, rng):
        seq = [rng.choice(start_aas, p=start_probs)]
        for _ in range(int(L) - 1):
            prev = seq[-1]
            if prev in trans_probs:
                nxt, p = trans_probs[prev]
                seq.append(rng.choice(nxt, p=p))
            else:
                seq.append(rng.choice(CANONICAL_AAS))
        out.append("".join(seq))
    return out


BASELINES = {
    "random": "Uniform random over 20 canonical residues",
    "freq_matched": "Residues drawn at training-set frequencies",
    "training_sample": "Copied from the training set (COPY ORACLE)",
    "bigram_markov": "First-order Markov chain over residues",
}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True, help="Real AMP FASTA.")
    ap.add_argument("--out_dir", default="./ablation_results")
    ap.add_argument("--n_generate", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=42)
    ap.add_argument("--ks", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--n_bootstrap", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [args.seed0 + i for i in range(args.seeds)]

    print("Loading data...")
    real = load_clean_fasta(args.fasta)
    real_lengths = [len(s) for s in real]

    aa_counter = Counter()
    for s in real:
        aa_counter.update(s)
    total = sum(aa_counter.values())
    aa_freq = {a: aa_counter.get(a, 0) / total for a in CANONICAL_AAS}

    length_sampler = LengthSampler(real_lengths)
    n = args.n_generate
    print(f"\nGenerating {n} sequences per baseline x {len(seeds)} seed(s)\n")

    per_seed = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        sets = {
            "random":          baseline_random(length_sampler, n, rng),
            "freq_matched":    baseline_freq_matched(length_sampler, aa_freq, n, rng),
            "training_sample": baseline_training_sample(real, n, rng),
            "bigram_markov":   baseline_bigram_markov(real, length_sampler, n, rng),
        }

        for name, seqs in sets.items():
            # Only the first seed's FASTA feeds the downstream scripts, so that
            # every evaluation stage looks at exactly the same sequences.
            if seed == seeds[0]:
                write_fasta(seqs,
                            os.path.join(args.out_dir,
                                         f"generated_baseline_{name}.fasta"),
                            prefix=name)

            lengths = [len(s) for s in seqs]
            row = {
                "method": name, "seed": seed, "n": len(seqs),
                "validity_rate": round(validity_rate(seqs), 4),
                "diversity": round(diversity(seqs), 4),
                "mean_len": round(float(np.mean(lengths)), 2),
                "std_len": round(float(np.std(lengths)), 2),
                "js_length": round(js_length_divergence(real_lengths, lengths), 4),
                "aa_L1": round(aa_composition_l1(real, seqs), 4),
            }
            for k in args.ks:
                m, _, _ = kmer_js(real, seqs, k, n_bootstrap=args.n_bootstrap,
                                  seed=seed)
                row[f"js_{k}mer"] = round(m, 4)
            per_seed.append(row)

            print(f"  [seed {seed}] {name:16s} validity={row['validity_rate']} "
                  f"div={row['diversity']} JSlen={row['js_length']} "
                  f"aaL1={row['aa_L1']} "
                  + " ".join(f"JS{k}={row[f'js_{k}mer']}" for k in args.ks))

    df = pd.DataFrame(per_seed)
    df.to_csv(os.path.join(args.out_dir, "baseline_generation_per_seed.csv"),
              index=False)

    metrics = ["validity_rate", "diversity", "mean_len", "std_len",
               "js_length", "aa_L1"] + [f"js_{k}mer" for k in args.ks]
    agg = aggregate_seeds(df, ["method"], metrics)
    agg["description"] = agg["method"].map(BASELINES)
    agg.to_csv(os.path.join(args.out_dir, "baseline_generation.csv"), index=False)

    print(f"\nAggregated over {len(seeds)} seed(s):")
    show = ["method", "n_seeds"] + [f"{m}_fmt" for m in
                                    ["validity_rate", "diversity", "js_length",
                                     "aa_L1", f"js_{args.ks[0]}mer"]]
    print(agg[show].to_string(index=False))
    print(f"\nSaved: {args.out_dir}/baseline_generation.csv "
          f"(+ _per_seed.csv, + FASTA files from seed {seeds[0]})")
    print("Reminder: 'training_sample' is a copy oracle, not a competitor.")


if __name__ == "__main__":
    main()
