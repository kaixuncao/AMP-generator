"""
ablation.py
===========
ABLATION EXPERIMENTS for the AMP-generator pipeline.

Directly supports the submission checklist item
    "Ablation experiments are included."

Three ablations:
  1. gmm_components : 1/2/3/4-component GMMs on the length distribution,
                      compared by AIC and BIC.
  2. model_type     : GRU vs. LSTM, identical data / tokenizer / schedule.
  3. special_tokens : with vs. without [BOS]/[EOS].

What changed relative to the previous version
---------------------------------------------
* PAD no longer collides with alanine (see common.SimpleTokenizer).
* Every configuration is run with several random seeds; all metrics are
  reported as mean +/- s.d. A single-seed comparison cannot separate a real
  architectural difference from run-to-run noise.
* A held-out validation split is used, and the reported quantity is
  residue-only validation perplexity, renormalised over the 20 residue
  logits. Raw training loss was not comparable across arms (different
  vocabularies, different prediction tasks); even residue-target-only NLL
  still penalises the [EOS]-carrying arm for the mass it must reserve for
  termination. The renormalised figure conditions on "the next token is a
  residue", which is the identical problem in both arms.
* Length-distribution and k-mer JS divergence are now actually computed for
  every ablation arm (they were defined but never called before).
* "weighted AIC/BIC" is now plain AIC/BIC, because no weights were ever
  passed. Optional true weighting is still available via --length_weights.
* Generation is batched and masks [PAD]/[BOS] out of the softmax instead of
  the previous `continue` statement, which re-fed the same input token and
  silently wasted the decoding budget.
* Decoding is always stochastic (multinomial). Greedy decoding is never
  used anywhere in this suite -- do NOT state otherwise in the manuscript.

Usage
-----
  python ablation.py --fasta AMPuniqmoree5_clean_rmdup.fasta \
      --out_dir ablation_results \
      --run gmm_components model_type special_tokens \
      --seeds 3
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from common import (
    CANONICAL_AAS, MIN_LEN, MAX_LEN, SimpleTokenizer,
    load_clean_fasta, write_fasta, set_seed,
    validity_rate, diversity, js_length_divergence, kmer_js,
    aa_composition_l1, aggregate_seeds, max_identity_to_reference,
)

VAL_SPLIT_SEED = 12345   # fixed, independent of the model seed
VAL_FRACTION = 0.10


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class SequenceDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_len=256):
        self.samples = []
        for seq in sequences:
            ids = tokenizer.encode(seq)
            if len(ids) < 2:
                continue
            self.samples.append(ids[:max_len])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ids = self.samples[i]
        return ids[:-1], ids[1:]


def make_collate(pad_id):
    def collate(batch):
        xs, ys = zip(*batch)
        m = max(len(x) for x in xs)
        xs_out = [list(x) + [pad_id] * (m - len(x)) for x in xs]
        ys_out = [list(y) + [pad_id] * (m - len(y)) for y in ys]
        return (torch.tensor(xs_out, dtype=torch.long),
                torch.tensor(ys_out, dtype=torch.long))
    return collate


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class SampleModel(nn.Module):
    """GRU / LSTM next-token predictor."""

    def __init__(self, vocab_size, pad_id, embed_dim=128, hidden_dim=256,
                 num_layers=2, rnn_type="GRU", dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        rnn_cls = nn.GRU if rnn_type == "GRU" else nn.LSTM
        self.rnn = rnn_cls(embed_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        out, hidden = self.rnn(self.embedding(x), hidden)
        return self.fc(out), hidden

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------

def residue_only_nll(model, loader, tokenizer, device):
    """
    Negative log-likelihood over AMINO-ACID targets only, in two forms.

    `full`    -- softmax over the whole vocabulary, scored on residue targets.
    `renorm`  -- softmax renormalised over the 20 residue logits alone.

    Why both. Restricting to residue TARGETS is not sufficient on its own: a
    model that carries [EOS] must reserve probability mass for it at every
    position, so its residue probabilities are structurally smaller and its
    `full` perplexity is inflated relative to a model that cannot terminate.
    That is a property of the vocabulary, not of modelling quality, and it
    would hand the no-special-token arm an unearned advantage.

    The `renorm` figure removes it by conditioning on "the next token is a
    residue", which is exactly the same prediction problem in both arms. USE
    `renorm` FOR THE ABLATION TABLE and treat `full` as diagnostic only.

    This also means a raw training-loss comparison between the two arms is
    meaningless, and should not appear in the manuscript as evidence either
    way.
    """
    model.eval()
    n_aa = len(tokenizer.aa_vocab)
    tot_full, tot_re, count = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_y = y.reshape(-1)
            mask = flat_y < n_aa
            if not bool(mask.any()):
                continue

            sel_logits = flat_logits[mask]
            sel_y = flat_y[mask]

            tot_full += float(F.cross_entropy(sel_logits, sel_y,
                                              reduction="sum"))
            tot_re += float(F.cross_entropy(sel_logits[:, :n_aa], sel_y,
                                            reduction="sum"))
            count += int(mask.sum())

    c = max(count, 1)
    nll_full, nll_re = tot_full / c, tot_re / c
    return (nll_full, float(np.exp(min(nll_full, 50))),
            nll_re, float(np.exp(min(nll_re, 50))))


def train_model(train_seqs, val_seqs, tokenizer, rnn_type="GRU", epochs=15,
                batch_size=64, lr=1e-3, max_len=256, device="cpu", seed=42,
                verbose=True):
    set_seed(seed)
    collate = make_collate(tokenizer.pad_id)

    tr_loader = DataLoader(SequenceDataset(train_seqs, tokenizer, max_len),
                           batch_size=batch_size, shuffle=True, collate_fn=collate)
    va_loader = DataLoader(SequenceDataset(val_seqs, tokenizer, max_len),
                           batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = SampleModel(tokenizer.vocab_size, tokenizer.pad_id,
                        rnn_type=rnn_type).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            count += x.size(0)

        tr_loss = total / max(count, 1)
        va_nll, va_ppl, va_nll_re, va_ppl_re = residue_only_nll(
            model, va_loader, tokenizer, device)
        history.append({"epoch": epoch, "train_loss": tr_loss,
                        "val_residue_nll": va_nll, "val_residue_ppl": va_ppl,
                        "val_residue_nll_renorm": va_nll_re,
                        "val_residue_ppl_renorm": va_ppl_re})
        if verbose:
            print(f"    epoch {epoch:>2}/{epochs} | train_loss={tr_loss:.4f} "
                  f"| val_ppl={va_ppl:.3f} | val_ppl_renorm={va_ppl_re:.3f}")

    return model, history


# --------------------------------------------------------------------------
# Generation -- batched, always stochastic
# --------------------------------------------------------------------------

@torch.no_grad()
def generate_sequences(model, tokenizer, n=2000, max_len=150, device="cpu",
                       temperature=1.5, batch_size=256, seed=0):
    """
    Multinomial sampling from the softmax at the given temperature.

    Greedy decoding is deliberately not implemented: it collapses every
    generation to one high-confidence sequence and makes diversity
    meaningless. The manuscript must not claim greedy decoding was used.

    [PAD] and [BOS] are masked out of the distribution so they can never be
    emitted. Without [EOS] (the no-special-token arm) sequences necessarily
    run to the full decoding budget -- that is exactly the effect under test.
    """
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    forbidden = torch.tensor(tokenizer.forbidden_ids, dtype=torch.long)

    results = []
    remaining = n
    while remaining > 0:
        b = min(batch_size, remaining)
        remaining -= b

        if tokenizer.use_special_tokens:
            cur = torch.full((b, 1), tokenizer.bos_id, dtype=torch.long, device=device)
            tokens = [[] for _ in range(b)]
        else:
            start = torch.randint(0, len(tokenizer.aa_vocab), (b, 1), generator=g)
            cur = start.to(device)
            tokens = [[int(start[i, 0])] for i in range(b)]

        hidden = None
        alive = torch.ones(b, dtype=torch.bool)

        for _ in range(max_len - (0 if tokenizer.use_special_tokens else 1)):
            logits, hidden = model(cur, hidden)
            last = logits[:, -1, :].float()
            last[:, forbidden.to(last.device)] = -float("inf")
            probs = torch.softmax(last / temperature, dim=-1).cpu()
            nxt = torch.multinomial(probs, 1, generator=g).squeeze(1)

            if tokenizer.eos_id is not None:
                alive = alive & (nxt != tokenizer.eos_id)
            for i in range(b):
                if alive[i]:
                    tokens[i].append(int(nxt[i]))
            if not bool(alive.any()):
                break
            cur = nxt.unsqueeze(1).to(device)

        results.extend(tokenizer.decode(t) for t in tokens)

    return results


# --------------------------------------------------------------------------
# Metric battery applied to every generated set
# --------------------------------------------------------------------------

def evaluate_generated(gen, real_seqs, real_lengths, ks=(2, 3), n_bootstrap=10):
    lengths = [len(s) for s in gen]
    row = {
        "n": len(gen),
        "unique": len(set(gen)),
        "diversity": round(diversity(gen), 4),
        "validity_rate": round(validity_rate(gen), 4),
        "mean_len": round(float(np.mean(lengths)), 2) if lengths else 0.0,
        "std_len": round(float(np.std(lengths)), 2) if lengths else 0.0,
        "js_length": round(js_length_divergence(real_lengths, lengths), 4),
        "aa_L1": round(aa_composition_l1(real_seqs, gen), 4),
    }
    for k in ks:
        m, _, _ = kmer_js(real_seqs, gen, k, n_bootstrap=n_bootstrap)
        row[f"js_{k}mer"] = round(m, 4)
    return row


# --------------------------------------------------------------------------
# Ablation 1: number of GMM components
# --------------------------------------------------------------------------

def run_gmm_components(real_lengths, weights=None, components=(1, 2, 3, 4),
                       out_dir=".", seed=42):
    """
    AIC/BIC model selection on the length distribution.

    Free parameters for an n-component 1-D full-covariance mixture:
        n means + n variances + (n - 1) weights = 3n - 1
    """
    os.makedirs(out_dir, exist_ok=True)
    X = np.asarray(real_lengths, dtype=float).reshape(-1, 1)
    w = np.ones(len(X)) if weights is None else np.asarray(weights, float).ravel()
    weighted = not np.allclose(w, w[0])

    def fit(n):
        gm = GaussianMixture(n_components=n, covariance_type="full",
                             random_state=seed, max_iter=1000, n_init=5)
        if not weighted:
            gm.fit(X)
        else:
            counts = np.maximum(((w / w.sum()) * 100000).astype(int), 1)
            gm.fit(np.repeat(X, counts, axis=0))
        return gm

    rows, fitted = [], {}
    for n in components:
        gm = fit(n)
        fitted[n] = gm
        logL = float(np.sum(w * gm.score_samples(X)))
        k = 3 * n - 1
        n_eff = float(w.sum())
        aic = 2 * k - 2 * logL
        bic = k * np.log(max(n_eff, 1.0)) - 2 * logL
        order = np.argsort(gm.means_.ravel())
        rows.append({
            "n_components": n, "k_params": k,
            "logL": round(logL, 4), "AIC": round(aic, 4), "BIC": round(bic, 4),
            "means": ";".join(f"{v:.4f}" for v in gm.means_.ravel()[order]),
            "variances": ";".join(f"{v:.4f}" for v in gm.covariances_.ravel()[order]),
            "weights": ";".join(f"{v:.4f}" for v in gm.weights_.ravel()[order]),
            "weighting": "weighted" if weighted else "unweighted",
        })
        print(f"[GMM] n={n}  AIC={aic:.2f}  BIC={bic:.2f}  "
              f"means={np.round(gm.means_.ravel()[order], 2)}  "
              f"weights={np.round(gm.weights_.ravel()[order], 3)}")

    df = pd.DataFrame(rows)
    df["dAIC_vs_prev"] = df["AIC"].diff().round(2)
    df["dBIC_vs_prev"] = df["BIC"].diff().round(2)
    df.to_csv(os.path.join(out_dir, "gmm_components.csv"), index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(X.ravel(), bins=60, density=True, alpha=0.25, color="gray",
             label="Empirical")
    grid = np.linspace(X.min() - 1, X.max() + 1, 1000).reshape(-1, 1)
    for n in components:
        plt.plot(grid.ravel(), np.exp(fitted[n].score_samples(grid)),
                 linewidth=1.8, label=f"GMM n={n}")
    plt.xlabel("Sequence length (aa)")
    plt.ylabel("Density")
    plt.title("GMM component selection on the AMP length distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gmm_components.png"), dpi=300)
    plt.close()

    print(f"[GMM] lowest BIC at n_components = "
          f"{int(df.loc[df['BIC'].idxmin(), 'n_components'])}")
    return df


# --------------------------------------------------------------------------
# Ablations 2 & 3: generator variants
# --------------------------------------------------------------------------

def run_variant_ablation(kind, train_seqs, val_seqs, real_seqs, out_dir,
                         epochs, max_len, n_generate, device, temperature,
                         seeds, save_fasta_seed):
    """kind = 'model_type' (GRU/LSTM) or 'special_tokens' (with/without)."""
    os.makedirs(out_dir, exist_ok=True)
    real_lengths = [len(s) for s in real_seqs]

    if kind == "model_type":
        variants = [("GRU", dict(rnn_type="GRU", use_special=True)),
                    ("LSTM", dict(rnn_type="LSTM", use_special=True))]
        label = "model"
    else:
        variants = [("with_special", dict(rnn_type="GRU", use_special=True)),
                    ("no_special", dict(rnn_type="GRU", use_special=False))]
        label = "variant"

    per_seed = []
    for tag, cfg in variants:
        for seed in seeds:
            print(f"\n[{kind}] variant={tag}  seed={seed}")
            tok = SimpleTokenizer(use_special_tokens=cfg["use_special"])
            model, hist = train_model(train_seqs, val_seqs, tok,
                                      rnn_type=cfg["rnn_type"], epochs=epochs,
                                      max_len=max_len, device=device, seed=seed)

            if seed == save_fasta_seed:
                torch.save(model.state_dict(),
                           os.path.join(out_dir, f"model_{tag}.pt"))

            gen = generate_sequences(model, tok, n=n_generate, max_len=150,
                                     device=device, temperature=temperature,
                                     seed=seed)

            if seed == save_fasta_seed:
                write_fasta(gen, os.path.join(out_dir, f"generated_{tag}.fasta"),
                            prefix=tag)

            row = {label: tag, "seed": seed,
                   "n_params": model.n_params(),
                   "train_loss": round(hist[-1]["train_loss"], 4),
                   "val_residue_nll": round(hist[-1]["val_residue_nll"], 4),
                   "val_residue_ppl": round(hist[-1]["val_residue_ppl"], 4),
                   "val_residue_nll_renorm":
                       round(hist[-1]["val_residue_nll_renorm"], 4),
                   "val_residue_ppl_renorm":
                       round(hist[-1]["val_residue_ppl_renorm"], 4)}
            row.update(evaluate_generated(gen, real_seqs, real_lengths))
            per_seed.append(row)

            print(f"    -> validity={row['validity_rate']} "
                  f"diversity={row['diversity']} mean_len={row['mean_len']} "
                  f"js_length={row['js_length']} js_3mer={row['js_3mer']}")

            pd.DataFrame(hist).to_csv(
                os.path.join(out_dir, f"history_{kind}_{tag}_seed{seed}.csv"),
                index=False)

    df = pd.DataFrame(per_seed)
    df.to_csv(os.path.join(out_dir, f"{kind}_per_seed.csv"), index=False)

    metrics = ["train_loss", "val_residue_nll", "val_residue_ppl",
               "val_residue_nll_renorm", "val_residue_ppl_renorm",
               "diversity", "validity_rate", "mean_len", "std_len",
               "js_length", "aa_L1", "js_2mer", "js_3mer"]
    agg = aggregate_seeds(df, [label], metrics)
    agg.to_csv(os.path.join(out_dir, f"{kind}.csv"), index=False)

    print(f"\n[{kind}] aggregated over {len(seeds)} seed(s):")
    show = [label, "n_seeds"] + [f"{m}_fmt" for m in
                                 ["val_residue_ppl_renorm", "validity_rate",
                                  "diversity", "mean_len", "js_length", "js_3mer"]]
    print(agg[show].to_string(index=False))
    return agg


# --------------------------------------------------------------------------
# Ablation 4: sampling temperature (INFERENCE-ONLY)
# --------------------------------------------------------------------------

def run_temperature_sweep(train_seqs, val_seqs, real_seqs, out_dir, epochs,
                          max_len, n_generate, device, temperatures, seeds,
                          save_fasta_seed, novelty_top_k=40,
                          skip_novelty=False):
    """
    Sweep the decoding temperature for the chosen architecture (GRU, with
    special tokens). One model is trained per seed and then sampled at every
    temperature, so differences are purely a decoding effect.

    Why this belongs in the paper
    -----------------------------
    Temperature is an INFERENCE hyper-parameter, not part of the model. It was
    previously fixed at 1.5 without justification, and T > 1 flattens the
    softmax, which degrades amino-acid composition and k-mer agreement with
    the real distribution. Reporting the sweep both justifies the operating
    point and answers the obvious reviewer question ("why 1.5?").

    Novelty is included because temperature trades off against memorisation
    in the opposite direction to distribution match: lower T concentrates
    probability on high-likelihood strings, which are more often near-copies
    of training sequences. The operating point must be chosen on both axes,
    not on distribution match alone.
    """
    os.makedirs(out_dir, exist_ok=True)
    real_lengths = [len(s) for s in real_seqs]

    per_seed = []
    for seed in seeds:
        print(f"\n[temperature] training GRU (with special), seed={seed}")
        tok = SimpleTokenizer(use_special_tokens=True)
        model, hist = train_model(train_seqs, val_seqs, tok, rnn_type="GRU",
                                  epochs=epochs, max_len=max_len,
                                  device=device, seed=seed)

        for T in temperatures:
            gen = generate_sequences(model, tok, n=n_generate, max_len=150,
                                     device=device, temperature=T, seed=seed)

            if seed == save_fasta_seed:
                write_fasta(gen,
                            os.path.join(out_dir, f"generated_T{T:g}.fasta"),
                            prefix=f"T{T:g}")

            row = {"temperature": T, "seed": seed,
                   "val_residue_ppl_renorm":
                       round(hist[-1]["val_residue_ppl_renorm"], 4)}
            row.update(evaluate_generated(gen, real_seqs, real_lengths))

            # Novelty on the first seed only -- it is the expensive step and
            # the effect of temperature on memorisation is large relative to
            # seed-to-seed variation.
            if seed == save_fasta_seed and not skip_novelty:
                valid = [s for s in gen
                         if MIN_LEN <= len(s) <= MAX_LEN]
                if valid:
                    nov = max_identity_to_reference(
                        valid, real_seqs, top_k=novelty_top_k,
                        verbose=False)
                    row["mean_max_identity"] = round(nov["mean_max"], 4)
                    row["frac_identity_gt_0.90"] = round(nov["frac_gt_0.90"], 4)
                    row["frac_exact_copy"] = round(nov["frac_eq_1.00"], 4)

            per_seed.append(row)
            print(f"    T={T:<4g} validity={row['validity_rate']:.4f} "
                  f"div={row['diversity']:.4f} JSlen={row['js_length']:.4f} "
                  f"JS2={row['js_2mer']:.4f} JS3={row['js_3mer']:.4f} "
                  f"aaL1={row['aa_L1']:.4f}"
                  + (f" exact={row['frac_exact_copy']:.4f}"
                     if "frac_exact_copy" in row else ""))

    df = pd.DataFrame(per_seed)
    df.to_csv(os.path.join(out_dir, "temperature_sweep_per_seed.csv"),
              index=False)

    metrics = ["diversity", "validity_rate", "mean_len", "std_len",
               "js_length", "aa_L1", "js_2mer", "js_3mer"]
    agg = aggregate_seeds(df, ["temperature"], metrics)

    # Carry the single-seed novelty columns through unaggregated.
    nov_cols = [c for c in ["mean_max_identity", "frac_identity_gt_0.90",
                            "frac_exact_copy"] if c in df.columns]
    if nov_cols:
        nov = (df.dropna(subset=nov_cols[:1])
                 .groupby("temperature")[nov_cols].first().reset_index())
        agg = agg.merge(nov, on="temperature", how="left")

    agg.to_csv(os.path.join(out_dir, "temperature_sweep.csv"), index=False)

    print(f"\n[temperature] aggregated over {len(seeds)} seed(s):")
    show = ["temperature", "n_seeds"] + \
           [f"{m}_fmt" for m in ["validity_rate", "diversity", "js_length",
                                 "aa_L1", "js_2mer", "js_3mer"]] + nov_cols
    print(agg[[c for c in show if c in agg.columns]].to_string(index=False))
    print("\nChoose the operating point on BOTH axes: distribution match "
          "(js_*, aa_L1) and novelty (frac_exact_copy). They move in "
          "opposite directions.")
    return agg


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Ablation study for the AMP generator.")
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out_dir", default="./ablation_results")
    ap.add_argument("--run", nargs="+",
                    choices=["gmm_components", "model_type", "special_tokens",
                             "temperature"],
                    default=["gmm_components", "model_type", "special_tokens",
                             "temperature"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--n_generate", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=1.5,
                    help="Temperature used for the model_type and "
                         "special_tokens ablations.")
    ap.add_argument("--temperatures", nargs="+", type=float,
                    default=[0.8, 1.0, 1.2, 1.5],
                    help="Temperatures for the temperature ablation.")
    ap.add_argument("--components", nargs="+", type=int,
                    default=[1, 2, 3, 4, 5, 6, 7, 8],
                    help="GMM component counts to compare. The range must be "
                         "wide enough to show where BIC actually stops "
                         "improving; stopping at 4 cannot support a claim "
                         "about the minimum.")
    ap.add_argument("--skip_sweep_novelty", action="store_true")
    ap.add_argument("--seeds", type=int, default=3,
                    help="Number of random seeds per configuration.")
    ap.add_argument("--seed0", type=int, default=42,
                    help="First seed; subsequent seeds are seed0+1, seed0+2, ...")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [args.seed0 + i for i in range(args.seeds)]

    print(f"Device: {device}")
    print(f"Seeds: {seeds}")
    print(f"Sampling temperature: {args.temperature} (stochastic sampling only)")
    print(f"Validity window: {MIN_LEN}-{MAX_LEN} aa over "
          f"{len(CANONICAL_AAS)} canonical residues\n")

    print("Loading data...")
    seqs = load_clean_fasta(args.fasta)
    if len(seqs) < 100:
        raise ValueError(f"Only {len(seqs)} sequences after cleaning.")
    lengths = [len(s) for s in seqs]

    rng = np.random.default_rng(VAL_SPLIT_SEED)
    perm = rng.permutation(len(seqs))
    n_val = max(int(len(seqs) * VAL_FRACTION), 1)
    val_seqs = [seqs[i] for i in perm[:n_val]]
    train_seqs = [seqs[i] for i in perm[n_val:]]
    print(f"  train={len(train_seqs)}  val={len(val_seqs)} "
          f"(fixed split, seed {VAL_SPLIT_SEED})\n")

    if "gmm_components" in args.run:
        run_gmm_components(lengths, components=tuple(args.components),
                           out_dir=args.out_dir)

    for kind in ["model_type", "special_tokens"]:
        if kind in args.run:
            run_variant_ablation(
                kind, train_seqs, val_seqs, seqs, args.out_dir,
                epochs=args.epochs, max_len=args.max_len,
                n_generate=args.n_generate, device=device,
                temperature=args.temperature, seeds=seeds,
                save_fasta_seed=seeds[0])

    if "temperature" in args.run:
        run_temperature_sweep(
            train_seqs, val_seqs, seqs, args.out_dir,
            epochs=args.epochs, max_len=args.max_len,
            n_generate=args.n_generate, device=device,
            temperatures=args.temperatures, seeds=seeds,
            save_fasta_seed=seeds[0],
            skip_novelty=args.skip_sweep_novelty)

    print(f"\nAll ablations done. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
