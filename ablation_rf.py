"""
ablation_rf.py
==============
CLASSIFIER BASELINES + discriminative scoring of generated sequences.

Directly supports the submission checklist item
    "Baseline comparisons to simple/trivial models (for example,
     1-nearest neighbour, random forest, most frequent class) are provided."

What this script produces
-------------------------
  classifier_baselines.csv : most-frequent-class / 1-NN / logistic regression
                             / random forest, 5-fold stratified CV,
                             mean +/- s.d. for AUC, accuracy, precision,
                             recall, F1.
  rf_feature_ablation.csv  : RF with all features / without length /
                             length only. Shows the classifier is not a
                             length detector.
  rf_feature_importance.csv: impurity AND permutation importance.
  rf_validation.csv/.png   : RF AMP-probability for real AMP, real non-AMP,
                             model-generated and every trivial baseline.

Changes vs. the previous version
--------------------------------
* Negatives are LENGTH-MATCHED to the positives by default. Without this,
  `length` can separate the classes almost trivially (AMPs <=129 aa, the
  non-AMP pool was cleaned to <=200 aa), and the headline AUC means nothing.
  This is the single most common reviewer objection to this kind of analysis.
* 1-NN and logistic regression now sit inside a StandardScaler pipeline.
  Previously they were fed raw features spanning several orders of magnitude
  (length ~10^2 vs. composition ~10^-2), which handicapped them and made the
  random forest look better than it is. A baseline comparison is only
  meaningful if the baselines are given a fair chance.
* Reference scores for real AMP / real non-AMP come from OUT-OF-FOLD
  predictions. Previously the reference bar was scored by a model that had
  been trained on those very sequences, biasing the comparison in favour of
  the real-AMP reference.
* 5-fold CV with mean +/- s.d. instead of a single 80/20 split.
* Shared cleaning rules with every other script (common.py).

Honest framing for the manuscript
---------------------------------
This classifier is trained on the same positive pool the generator was
trained on, using compositional features that the generator explicitly
learns to reproduce. A high AMP probability therefore measures DISTRIBUTIONAL
CONSISTENCY with known AMPs -- it is not evidence of antimicrobial activity,
and the model is not an independent validator. State this in the limitations.

Usage
-----
  python ablation_rf.py --pos_fasta AMPuniqmoree5_clean_rmdup.fasta \
      --neg_fasta namp_clean.fasta --out_dir ablation_results
"""

import os
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score)
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from common import load_clean_fasta, read_fasta, clean_sequences, featurize


# --------------------------------------------------------------------------
# Length matching
# --------------------------------------------------------------------------

def length_matched_subsets(pos, neg, rng, bin_width=1):
    """
    Return (pos_sub, neg_sub) with IDENTICAL length distributions.

    For every length bin we keep min(n_pos, n_neg) sequences from each class.
    Length then carries zero information about the label by construction, so
    any remaining discriminative signal must come from composition and
    physicochemistry.
    """
    def bucket(seqs):
        d = defaultdict(list)
        for s in seqs:
            d[len(s) // bin_width].append(s)
        return d

    bp, bn = bucket(pos), bucket(neg)
    pos_sub, neg_sub = [], []
    for b in sorted(set(bp) & set(bn)):
        m = min(len(bp[b]), len(bn[b]))
        pos_sub += list(rng.choice(bp[b], size=m, replace=False))
        neg_sub += list(rng.choice(bn[b], size=m, replace=False))
    return pos_sub, neg_sub


# --------------------------------------------------------------------------
# Classifier zoo
# --------------------------------------------------------------------------

def make_classifiers(n_estimators, seed):
    """
    The three trivial/simple baselines named in the checklist, plus the
    random forest. Scaling-sensitive models are wrapped in a pipeline.
    """
    return {
        "MostFrequentClass": DummyClassifier(strategy="most_frequent"),
        "1-NearestNeighbour": Pipeline([
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=1, n_jobs=-1)),
        ]),
        "LogisticRegression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=n_estimators, n_jobs=-1,
            random_state=seed, class_weight="balanced"),
    }


def cv_evaluate(clf, X, y, cv):
    """Per-fold metrics -> mean +/- s.d."""
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        m = clf
        from sklearn.base import clone
        m = clone(clf)
        m.fit(X[tr], y[tr])
        proba = m.predict_proba(X[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "fold": fold,
            "AUC": roc_auc_score(y[te], proba) if len(set(y[te])) > 1 else np.nan,
            "Accuracy": accuracy_score(y[te], pred),
            "Precision": precision_score(y[te], pred, zero_division=0),
            "Recall": recall_score(y[te], pred, zero_division=0),
            "F1": f1_score(y[te], pred, zero_division=0),
        })
    df = pd.DataFrame(rows)
    out = {}
    for m in ["AUC", "Accuracy", "Precision", "Recall", "F1"]:
        out[f"{m}_mean"] = round(float(df[m].mean()), 4)
        out[f"{m}_sd"] = round(float(df[m].std(ddof=1)), 4)
        out[m] = f"{df[m].mean():.4f} ± {df[m].std(ddof=1):.4f}"
    return out, df


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_fasta", required=True, help="Real AMP FASTA.")
    ap.add_argument("--neg_fasta", required=True, help="Non-AMP FASTA.")
    ap.add_argument("--out_dir", default="./ablation_results")
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_length_match", action="store_true",
                    help="Disable length matching (NOT recommended; kept only "
                         "so the confound can be demonstrated).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Loading sequences (identical cleaning rules as every other script)...")
    pos = load_clean_fasta(args.pos_fasta)
    neg = load_clean_fasta(args.neg_fasta)
    print(f"  positives (AMP)     : {len(pos)}")
    print(f"  negatives (non-AMP) : {len(neg)}")
    if len(neg) < 100:
        raise ValueError(f"Only {len(neg)} usable non-AMP sequences.")

    if args.no_length_match:
        n = min(len(pos), len(neg))
        pos_s = list(rng.choice(pos, n, replace=False))
        neg_s = list(rng.choice(neg, n, replace=False))
        print(f"  [!] length matching DISABLED -- n={n} per class")
    else:
        pos_s, neg_s = length_matched_subsets(pos, neg, rng)
        print(f"  length-matched: n={len(pos_s)} per class")
        print(f"  mean length  : pos={np.mean([len(s) for s in pos_s]):.2f}  "
              f"neg={np.mean([len(s) for s in neg_s]):.2f}")
    if len(pos_s) < 100:
        raise ValueError("Too few length-matched pairs; widen bin_width or "
                         "supply a broader non-AMP pool.")

    print("\nFeaturizing...")
    X_pos = featurize(pos_s, verbose=True)
    X_neg = featurize(neg_s, verbose=True)
    X_df = pd.concat([X_pos, X_neg], ignore_index=True)
    X = X_df.values
    y = np.array([1] * len(X_pos) + [0] * len(X_neg))
    cv = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                         random_state=args.seed)

    # ---- Classifier baselines (the checklist item) ----
    print(f"\n[Classifier baselines, {args.n_folds}-fold stratified CV]")
    clfs = make_classifiers(args.n_estimators, args.seed)
    rows, fold_frames = [], []
    for name, clf in clfs.items():
        summary, folds = cv_evaluate(clf, X, y, cv)
        rows.append({"classifier": name,
                     **{k: v for k, v in summary.items() if "_" not in k},
                     **{k: v for k, v in summary.items() if "_" in k}})
        folds["classifier"] = name
        fold_frames.append(folds)
        print(f"  {name:>20s}  AUC={summary['AUC']}  Acc={summary['Accuracy']}  "
              f"F1={summary['F1']}")

    pd.DataFrame(rows).to_csv(
        os.path.join(args.out_dir, "classifier_baselines.csv"), index=False)
    pd.concat(fold_frames, ignore_index=True).to_csv(
        os.path.join(args.out_dir, "classifier_baselines_per_fold.csv"),
        index=False)

    # ---- Feature ablation: is the RF just reading length? ----
    print("\n[RF feature ablation]")
    feat_rows = []
    for tag, cols in [
        ("all_features", list(X_df.columns)),
        ("without_length", [c for c in X_df.columns if c != "length"]),
        ("length_only", ["length"]),
    ]:
        if not cols or (tag != "all_features" and "length" not in X_df.columns):
            continue
        Xa = X_df[cols].values
        rf_a = RandomForestClassifier(n_estimators=args.n_estimators, n_jobs=-1,
                                      random_state=args.seed,
                                      class_weight="balanced")
        summary, _ = cv_evaluate(rf_a, Xa, y, cv)
        feat_rows.append({"feature_set": tag, "n_features": len(cols),
                          "AUC": summary["AUC"], "Accuracy": summary["Accuracy"],
                          "F1": summary["F1"]})
        print(f"  {tag:>16s} ({len(cols):>2d} feats)  AUC={summary['AUC']}")
    pd.DataFrame(feat_rows).to_csv(
        os.path.join(args.out_dir, "rf_feature_ablation.csv"), index=False)

    # ---- Final RF + importances ----
    print("\nFitting final random forest on all matched data...")
    rf = RandomForestClassifier(n_estimators=args.n_estimators, n_jobs=-1,
                                random_state=args.seed, class_weight="balanced")
    rf.fit(X, y)

    tr_idx, te_idx = next(iter(cv.split(X, y)))
    rf_pi = RandomForestClassifier(n_estimators=args.n_estimators, n_jobs=-1,
                                   random_state=args.seed,
                                   class_weight="balanced").fit(X[tr_idx], y[tr_idx])
    pi = permutation_importance(rf_pi, X[te_idx], y[te_idx], n_repeats=10,
                                random_state=args.seed, n_jobs=-1)
    imp = pd.DataFrame({
        "feature": X_df.columns,
        "impurity_importance": rf.feature_importances_,
        "permutation_importance_mean": pi.importances_mean,
        "permutation_importance_sd": pi.importances_std,
    }).sort_values("permutation_importance_mean", ascending=False)
    imp.to_csv(os.path.join(args.out_dir, "rf_feature_importance.csv"), index=False)
    print("\nTop 10 features (permutation importance):")
    print(imp.head(10).to_string(index=False))

    # ---- Out-of-fold reference scores for the real sequences ----
    print("\nComputing out-of-fold probabilities for the real sequences...")
    oof = cross_val_predict(
        RandomForestClassifier(n_estimators=args.n_estimators, n_jobs=-1,
                               random_state=args.seed, class_weight="balanced"),
        X, y, cv=cv, method="predict_proba")[:, 1]

    def summarize_probs(p, tag, n):
        p = np.asarray(p, float)
        return {"group": tag, "n": n,
                "mean_prob": round(float(p.mean()), 4),
                "median_prob": round(float(np.median(p)), 4),
                "sd_prob": round(float(p.std()), 4),
                "frac_prob_ge_0.5": round(float((p >= 0.5).mean()), 4)}

    rows = [
        summarize_probs(oof[y == 1], "Real AMP (out-of-fold)", int((y == 1).sum())),
        summarize_probs(oof[y == 0], "Real non-AMP (out-of-fold)", int((y == 0).sum())),
    ]

    # ---- Score every generated set found in out_dir ----
    def score_fasta(path, tag):
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            return None
        raw = read_fasta(path)
        seqs = clean_sequences(raw)
        if not seqs:
            # Do NOT silently drop this group. A set with zero valid sequences
            # is a RESULT (this is what the no-[EOS] arm produces: every
            # sequence runs to the 150-token decoding budget and so falls
            # outside the 5-129 aa window). Reporting it as an explicit zero
            # keeps the ablation visible in the table instead of leaving a
            # hole a reader would read as "not run".
            print(f"  {tag}: 0/{len(raw)} sequences valid -- reported as zero")
            return {"group": tag, "n": 0, "n_generated": len(raw),
                    "mean_prob": float("nan"), "median_prob": float("nan"),
                    "sd_prob": float("nan"), "frac_prob_ge_0.5": float("nan")}
        p = rf.predict_proba(featurize(seqs).values)[:, 1]
        print(f"  scored {tag} ({len(seqs)}/{len(raw)} valid)")
        r = summarize_probs(p, tag, len(seqs))
        r["n_generated"] = len(raw)
        return r

    print("\nScoring generated sets...")
    targets = [
        ("generated_with_special.fasta", "Generated (GRU, with special)"),
        ("generated_no_special.fasta", "Generated (GRU, no special)"),
        ("generated_GRU.fasta", "Generated (GRU)"),
        ("generated_LSTM.fasta", "Generated (LSTM)"),
        ("generated_baseline_random.fasta", "Baseline: random"),
        ("generated_baseline_freq_matched.fasta", "Baseline: freq_matched"),
        ("generated_baseline_training_sample.fasta", "Baseline: training_sample"),
        ("generated_baseline_bigram_markov.fasta", "Baseline: bigram_markov"),
    ]
    # Any other generated_*.fasta (e.g. the temperature sweep) is picked up
    # automatically, so a new arm needs no edit here.
    import glob
    known = {f for f, _ in targets}
    for p_ in sorted(glob.glob(os.path.join(args.out_dir, "generated_*.fasta"))):
        f = os.path.basename(p_)
        if f in known:
            continue
        stem = f[len("generated_"):-len(".fasta")]
        tag = (f"Generated (GRU, T={stem[1:]})"
               if stem.startswith("T") and stem[1:2].isdigit()
               else f"Generated ({stem})")
        targets.append((f, tag))

    for fname, tag in targets:
        r = score_fasta(os.path.join(args.out_dir, fname), tag)
        if r:
            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "rf_validation.csv"), index=False)
    print("\n[RF probability by group]")
    print(df.to_string(index=False))

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(11, 5.5))
    palette = {
        "Real AMP (out-of-fold)": "#2E7D32",
        "Real non-AMP (out-of-fold)": "#C62828",
        "Generated (GRU, with special)": "#1565C0",
        "Generated (GRU)": "#1565C0",
        "Generated (LSTM)": "#0277BD",
        "Generated (GRU, no special)": "#EF6C00",
    }
    fallback = ["#7B1FA2", "#00838F", "#5D4037", "#455A64", "#9E9E9E"]
    colors, j = [], 0
    for gname in df["group"]:
        c = palette.get(gname)
        if c is None:
            c = fallback[j % len(fallback)]
            j += 1
        colors.append(c)

    bars = ax.bar(range(len(df)), df["mean_prob"].fillna(0.0),
                  yerr=df["sd_prob"].fillna(0.0),
                  capsize=3, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["group"], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Random-forest probability of being an AMP")
    ax.set_ylim(0, 1.12)
    ax.set_title("Discriminative scoring of generated and baseline sequences\n"
                 "(length-matched negatives; real groups scored out-of-fold)",
                 fontsize=11)
    for b, v in zip(bars, df["mean_prob"]):
        label = "no valid\nsequences" if pd.isna(v) else f"{v:.3f}"
        ax.text(b.get_x() + b.get_width() / 2, (0.0 if pd.isna(v) else v) + 0.03,
                label, ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "rf_validation.png"), dpi=300)
    plt.close()

    print(f"\nSaved to {args.out_dir}/: classifier_baselines.csv, "
          f"rf_feature_ablation.csv, rf_feature_importance.csv, "
          f"rf_validation.csv, rf_validation.png")
    print("Reminder: this classifier measures distributional consistency with "
          "known AMPs, not antimicrobial activity.")


if __name__ == "__main__":
    main()
