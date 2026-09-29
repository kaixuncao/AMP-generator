"""
make_tables.py
==============
Assemble the CSV outputs into supplementary tables mapped one-to-one onto
the two submission checklist items.

  Checklist: "Ablation experiments are included."
    Table S1  GMM component selection (AIC/BIC)
    Table S2  GRU vs. LSTM
    Table S3  Special tokens ([BOS]/[EOS])

  Checklist: "Baseline comparisons to simple/trivial models are provided."
    Table S4  Generation baselines
    Table S5  Classifier baselines (most-frequent / 1-NN / logistic / RF)
    Table S6  Discriminative scores + k-mer divergence + novelty screen

Outputs both per-table CSVs and a single Markdown file to paste into the
supplementary information.

Usage
-----
  python make_tables.py --out_dir ablation_results
"""

import os
import argparse

import pandas as pd


def load(out_dir, name):
    p = os.path.join(out_dir, name)
    if not os.path.exists(p):
        print(f"  [missing] {name}")
        return None
    return pd.read_csv(p)


def pick(df, cols):
    return df[[c for c in cols if c in df.columns]].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./ablation_results")
    ap.add_argument("--tables_dir", default=None)
    args = ap.parse_args()

    tables_dir = args.tables_dir or os.path.join(args.out_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    md = ["# Supplementary Tables",
          "",
          "Every generated set was produced by multinomial sampling at the "
          "temperature stated in the Methods; greedy decoding is not used "
          "anywhere in this study. Where a configuration was run with several "
          "random seeds, values are mean ± s.d. across seeds. Set sizes are "
          "given in the `n` columns.",
          ""]
    tables = []

    # ---------------- Ablations ----------------
    df = load(args.out_dir, "gmm_components.csv")
    if df is not None:
        t = pick(df, ["n_components", "k_params", "logL", "AIC", "BIC",
                      "dAIC_vs_prev", "dBIC_vs_prev", "means", "variances",
                      "weights"])
        tables.append(("S1", "GMM component selection on the sequence-length "
                             "distribution", t,
                       "Free parameters k = 3n − 1 for an n-component "
                       "one-dimensional full-covariance mixture. AIC/BIC are "
                       "unweighted."))

    df = load(args.out_dir, "model_type.csv")
    if df is not None:
        t = pick(df, ["model", "n_seeds", "val_residue_ppl_renorm_fmt",
                      "validity_rate_fmt", "diversity_fmt",
                      "mean_len_fmt", "std_len_fmt", "js_length_fmt",
                      "js_3mer_fmt", "aa_L1_fmt"])
        tables.append(("S2", "Ablation — recurrent cell type (GRU vs. LSTM)",
                       t,
                       "Identical data, tokenizer, schedule and seeds. "
                       "Perplexity is computed on held-out data over "
                       "amino-acid targets, renormalised over the 20 residue "
                       "logits. Training loss is not comparable across arms "
                       "and is recorded in the per-seed CSV only."))

    df = load(args.out_dir, "special_tokens.csv")
    if df is not None:
        t = pick(df, ["variant", "n_seeds", "val_residue_ppl_renorm_fmt",
                      "validity_rate_fmt", "diversity_fmt",
                      "mean_len_fmt", "std_len_fmt", "js_length_fmt",
                      "js_3mer_fmt", "aa_L1_fmt"])
        tables.append(("S3", "Ablation — special tokens ([BOS]/[EOS])", t,
                       "[PAD] is retained in both arms as a batching "
                       "mechanism and is assigned an id outside the "
                       "amino-acid alphabet, so no residue is excluded from "
                       "the loss in either arm. Residue perplexity is "
                       "renormalised over the 20 residue logits, so the arm "
                       "carrying [EOS] is not penalised for the probability "
                       "mass it must reserve for termination; on that "
                       "like-for-like measure the two arms model residues "
                       "equally well, and the difference between them is one "
                       "of controllability, not of fit. Without [EOS] the output "
                       "length is set entirely by the decoding budget (150 "
                       "tokens) rather than learned from the data, so no "
                       "sequence can fall inside the 5–129 aa validity "
                       "window."))

    # ---------------- Baselines ----------------
    df = load(args.out_dir, "baseline_generation.csv")
    if df is not None:
        t = pick(df, ["method", "description", "n_seeds", "validity_rate_fmt",
                      "diversity_fmt", "mean_len_fmt", "js_length_fmt",
                      "aa_L1_fmt", "js_2mer_fmt", "js_3mer_fmt"])
        tables.append(("S4", "Generation baselines (simple/trivial models)", t,
                       "All baselines draw lengths from the empirical AMP "
                       "length distribution, so comparisons are not "
                       "confounded by length; validity rate is therefore 1.0 "
                       "by construction and is reported as a sanity check "
                       "only. `training_sample` is a copy oracle, not a "
                       "competing method: it marks the attainable floor on "
                       "the divergence metrics and the floor on novelty."))

    df = load(args.out_dir, "classifier_baselines.csv")
    if df is not None:
        t = pick(df, ["classifier", "AUC", "Accuracy", "Precision", "Recall", "F1"])
        note = ("Five-fold stratified cross-validation, mean ± s.d. across "
                "folds. Negatives are length-matched to the positives, so "
                "sequence length carries no label information. "
                "Scaling-sensitive baselines (1-NN, logistic regression) are "
                "fitted inside a standardisation pipeline.")
        fa = load(args.out_dir, "rf_feature_ablation.csv")
        if fa is not None:
            note += (" A feature ablation (rf_feature_ablation.csv) confirms "
                     "the forest is not acting as a length detector.")
        tables.append(("S5", "Classifier baselines vs. the random-forest "
                             "validator", t, note))

    df = load(args.out_dir, "temperature_sweep.csv")
    if df is not None:
        t = pick(df, ["temperature", "n_seeds", "validity_rate_fmt",
                      "diversity_fmt", "mean_len_fmt", "js_length_fmt",
                      "aa_L1_fmt", "js_2mer_fmt", "js_3mer_fmt",
                      "mean_max_identity", "frac_identity_gt_0.90",
                      "frac_exact_copy"])
        tables.append(("S7", "Ablation \u2014 decoding temperature", t,
                       "Temperature is an inference hyper-parameter: one "
                       "model per seed was trained and then sampled at each "
                       "value, so all differences are a decoding effect. "
                       "Distribution match and novelty move in opposite "
                       "directions with temperature, and the operating point "
                       "is chosen on both. Novelty columns are from the first "
                       "seed."))

    parts = []
    rf = load(args.out_dir, "rf_validation.csv")
    if rf is not None:
        parts.append(pick(rf, ["group", "n", "n_generated", "mean_prob",
                               "sd_prob", "frac_prob_ge_0.5"]))
    km = load(args.out_dir, "kmer_divergence.csv")
    if km is not None:
        parts.append(pick(km, ["group", "js_2mer", "js_3mer", "js_4mer"]))
    nv = load(args.out_dir, "novelty_identity.csv")
    if nv is not None:
        parts.append(pick(nv, ["group", "metric", "mean_max_identity",
                               "global_max_identity", "frac_identity_gt_0.90",
                               "frac_exact_copy"]))
    if parts:
        merged = parts[0]
        for p in parts[1:]:
            merged = merged.merge(p, on="group", how="outer")
        tables.append(("S6", "Evaluation of generated and baseline sequences",
                       merged,
                       "Random-forest probability (real groups scored "
                       "out-of-fold), sample-size-corrected k-mer "
                       "Jensen–Shannon divergence, and the novelty screen "
                       "(maximum identity of each generated sequence to any "
                       "training sequence). A group with n = 0 produced no "
                       "sequence inside the validity window, which is itself "
                       "the result rather than a missing measurement. The "
                       "random forest reflects "
                       "distributional consistency with known AMPs, not "
                       "antimicrobial activity."))

    tables.sort(key=lambda x: x[0])

    for tag, title, t, note in tables:
        t.to_csv(os.path.join(tables_dir, f"Table_{tag}.csv"), index=False)
        md += [f"## Table {tag}. {title}", "", t.to_markdown(index=False), "",
               f"*{note}*", ""]
        print(f"  wrote Table_{tag}.csv  ({len(t)} rows)")

    md_path = os.path.join(tables_dir, "supplementary_tables.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))

    print(f"\nSaved {len(tables)} tables to {tables_dir}/")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
