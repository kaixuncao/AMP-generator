"""
common.py
=========
Shared utilities for the AMP-generator ablation / baseline suite.

This module is the SINGLE SOURCE OF TRUTH for:
  * FASTA I/O and sequence cleaning  (identical rules in every script)
  * the tokenizer                    (PAD id can never collide with an AA)
  * distribution metrics             (length JS, k-mer JS, AA composition L1)
  * novelty / memorisation metrics   (alignment-based sequence identity)
  * physicochemical featurisation    (used by the discriminative validator)

Rationale for centralising: in the previous version each script applied a
different length filter (max_len = None / 129 / 200), which silently broke
the claim that "all experiments used the same cleaned dataset".
"""

from __future__ import annotations

import os
import random
from collections import Counter, defaultdict

import numpy as np

# --------------------------------------------------------------------------
# Global constants -- change here and everywhere follows
# --------------------------------------------------------------------------

CANONICAL_AAS = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(CANONICAL_AAS)

MIN_LEN = 5      # inclusive
MAX_LEN = 129    # inclusive -- validity criterion used throughout

# Kyte-Doolittle hydropathy
KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Net charge at ~pH 7
CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------
# FASTA I/O
# --------------------------------------------------------------------------

def read_fasta(path):
    seqs, buf = [], []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if buf:
                    seqs.append("".join(buf))
                    buf = []
            else:
                buf.append(line.strip().upper())
        if buf:
            seqs.append("".join(buf))
    return seqs


def write_fasta(seqs, path, prefix="seq"):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">{prefix}_{i}\n{s}\n")


def clean_sequences(seqs, min_len=MIN_LEN, max_len=MAX_LEN):
    """The one and only cleaning rule: length window + canonical alphabet."""
    out = []
    for s in seqs:
        s = s.strip().upper()
        if not (min_len <= len(s) <= max_len):
            continue
        if any(a not in AA_SET for a in s):
            continue
        out.append(s)
    return out


def load_clean_fasta(path, min_len=MIN_LEN, max_len=MAX_LEN, verbose=True):
    raw = read_fasta(path)
    seqs = clean_sequences(raw, min_len, max_len)
    if verbose:
        print(f"  {os.path.basename(path)}: {len(raw)} read -> "
              f"{len(seqs)} kept ({min_len}-{max_len} aa, canonical only)")
    return seqs


# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------

class SimpleTokenizer:
    """
    Residue-level tokenizer.

    CRITICAL FIX vs. the previous version
    -------------------------------------
    Previously, when use_special_tokens=False the vocabulary was exactly the
    20 amino acids and `pad_id` defaulted to 0 -- which is alanine. That made
    `nn.Embedding(padding_idx=0)` freeze alanine's embedding at zero and made
    `CrossEntropyLoss(ignore_index=0)` drop EVERY alanine target from the
    loss (~8-9% of all residues). The no-special-token model was therefore
    crippled in a way unrelated to [BOS]/[EOS], and its lower training loss
    was largely an artefact of scoring fewer tokens.

    Here the 20 amino acids ALWAYS occupy ids 0-19 in both variants, and
    [PAD] is a reserved id outside the amino-acid alphabet. [PAD] is a
    batching mechanism, not a modelling choice, so it is retained in both
    variants; the ablation isolates [BOS]/[EOS], which is the substantive
    question. Report it that way in the manuscript.

    Because ids 0-19 mean the same residue in both variants, validation
    negative log-likelihood restricted to residue positions is directly
    comparable across them.
    """

    def __init__(self, use_special_tokens: bool = True):
        self.use_special_tokens = use_special_tokens
        self.aa_vocab = list(CANONICAL_AAS)

        if use_special_tokens:
            self.vocab = self.aa_vocab + ["[PAD]", "[BOS]", "[EOS]"]
        else:
            self.vocab = self.aa_vocab + ["[PAD]"]

        self.token2id = {t: i for i, t in enumerate(self.vocab)}
        self.id2token = {i: t for t, i in self.token2id.items()}

        self.aa_ids = [self.token2id[a] for a in self.aa_vocab]   # 0..19
        self.pad_id = self.token2id["[PAD]"]                      # 20
        self.bos_id = self.token2id.get("[BOS]")                  # 21 or None
        self.eos_id = self.token2id.get("[EOS]")                  # 22 or None

        # ids that must never be emitted during sampling
        self.forbidden_ids = [self.pad_id]
        if self.bos_id is not None:
            self.forbidden_ids.append(self.bos_id)

    def encode(self, seq):
        ids = [self.token2id[a] for a in seq if a in self.token2id]
        if self.use_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids):
        return "".join(
            self.id2token[int(i)] for i in ids
            if 0 <= int(i) < len(self.aa_vocab)
        )

    @property
    def vocab_size(self):
        return len(self.vocab)


# --------------------------------------------------------------------------
# Length sampling
# --------------------------------------------------------------------------

class LengthSampler:
    """Empirical length distribution of the real AMPs."""

    def __init__(self, lengths):
        self.lengths = np.asarray(lengths)
        self.unique, counts = np.unique(self.lengths, return_counts=True)
        self.probs = counts / counts.sum()

    def sample(self, n, rng):
        return rng.choice(self.unique, size=n, p=self.probs)


# --------------------------------------------------------------------------
# Distribution metrics
# --------------------------------------------------------------------------

def _js(p, q):
    """Jensen-Shannon divergence, base 2, on two count vectors."""
    eps = 1e-12
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        return float(np.sum(a * np.log2(a / b)))

    return float(np.sqrt(max(0.5 * _kl(p, m) + 0.5 * _kl(q, m), 0.0)))


def validity_rate(seqs, min_len=MIN_LEN, max_len=MAX_LEN):
    """
    Fraction of generated sequences inside the length window and over the
    canonical alphabet.

    NOTE for the manuscript: this is a sanity check, not a discriminative
    metric. Every baseline that draws lengths from the empirical AMP length
    distribution attains 1.0 by construction. Call it 'validity rate', not
    'success rate', so readers do not mistake 1.0 for an achievement.
    """
    if not seqs:
        return 0.0
    ok = sum(1 for s in seqs
             if min_len <= len(s) <= max_len and all(a in AA_SET for a in s))
    return ok / len(seqs)


def diversity(seqs):
    return len(set(seqs)) / len(seqs) if seqs else 0.0


def js_length_divergence(real_lengths, gen_lengths, bins=100, value_range=(0, 200)):
    hist_r, edges = np.histogram(real_lengths, bins=bins, range=value_range)
    hist_g, _ = np.histogram(gen_lengths, bins=edges)
    return _js(hist_r, hist_g)


def aa_composition(seqs):
    c = Counter()
    total = 0
    for s in seqs:
        for a in s:
            c[a] += 1
            total += 1
    return np.array([c.get(a, 0) / max(total, 1) for a in CANONICAL_AAS])


def aa_composition_l1(real_seqs, gen_seqs):
    return float(np.abs(aa_composition(real_seqs) - aa_composition(gen_seqs)).sum())


def kmer_counts(seqs, k):
    c = Counter()
    for s in seqs:
        for i in range(len(s) - k + 1):
            c[s[i:i + k]] += 1
    return c


def _kmer_js_from_counts(c1, c2):
    keys = set(c1) | set(c2)
    p = np.array([c1.get(kk, 0) for kk in keys], dtype=float)
    q = np.array([c2.get(kk, 0) for kk in keys], dtype=float)
    return _js(p, q)


def n_kmer_tokens(seqs, k):
    return sum(max(len(s) - k + 1, 0) for s in seqs)


def kmer_js(real_seqs, gen_seqs, k, n_bootstrap=20, seed=0):
    """
    k-mer frequency JS divergence with SAMPLE-SIZE CORRECTION.

    The real set is typically much larger than a 2,000-sequence generated
    set. Because the k-mer space is large (20^4 = 160,000 for k=4), the
    smaller set is necessarily sparser, which inflates JS for reasons that
    have nothing to do with model quality. We therefore subsample the real
    set to the same number of k-mer tokens as the generated set and
    bootstrap over that subsample.

    Returns (mean, sd, raw_full_set_value).
    """
    raw = _kmer_js_from_counts(kmer_counts(real_seqs, k), kmer_counts(gen_seqs, k))

    budget = n_kmer_tokens(gen_seqs, k)
    if budget == 0:
        return float("nan"), float("nan"), raw

    gen_c = kmer_counts(gen_seqs, k)
    rng = np.random.default_rng(seed)
    vals = []
    idx_all = np.arange(len(real_seqs))

    for _ in range(n_bootstrap):
        rng.shuffle(idx_all)
        picked, tokens = [], 0
        for i in idx_all:
            s = real_seqs[i]
            picked.append(s)
            tokens += max(len(s) - k + 1, 0)
            if tokens >= budget:
                break
        vals.append(_kmer_js_from_counts(kmer_counts(picked, k), gen_c))

    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, raw


# --------------------------------------------------------------------------
# Novelty / memorisation: alignment-based sequence identity
# --------------------------------------------------------------------------

_ALIGNER = None
_ALIGNER_KIND = None


def _get_aligner():
    """
    Prefer Smith-Waterman local alignment (BLOSUM62, BLAST-like gap costs).

    The previous version used difflib.SequenceMatcher.ratio(), which is a
    generic string-similarity heuristic, NOT biological sequence identity.
    Reporting it as 'sequence identity' in a bioinformatics venue invites a
    request to redo the analysis. If Biopython is unavailable we fall back to
    difflib but relabel the metric so the distinction stays visible.
    """
    global _ALIGNER, _ALIGNER_KIND
    if _ALIGNER_KIND is not None:
        return _ALIGNER, _ALIGNER_KIND

    try:
        from Bio.Align import PairwiseAligner, substitution_matrices
        al = PairwiseAligner()
        al.mode = "local"
        al.substitution_matrix = substitution_matrices.load("BLOSUM62")
        al.open_gap_score = -11
        al.extend_gap_score = -1
        _ALIGNER, _ALIGNER_KIND = al, "smith_waterman_blosum62"
    except Exception:
        print("  [warn] Biopython not found -- falling back to difflib ratio.")
        print("         Install with `pip install biopython` for true "
              "Smith-Waterman identity.")
        _ALIGNER, _ALIGNER_KIND = None, "difflib_ratio"

    return _ALIGNER, _ALIGNER_KIND


def aligner_kind():
    return _get_aligner()[1]


def pairwise_identity(a: str, b: str) -> float:
    """
    Sequence identity = (identical aligned residues) / (length of the shorter
    sequence). Using the shorter sequence as denominator is the conservative
    choice for a memorisation screen: a generated peptide fully contained in
    a longer training sequence scores 1.0.
    """
    al, kind = _get_aligner()
    if kind == "difflib_ratio":
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a, b).ratio()

    try:
        aln = al.align(a, b)[0]
    except Exception:
        return 0.0

    matches = 0
    for (a0, a1), (b0, b1) in zip(aln.aligned[0], aln.aligned[1]):
        for x, y in zip(a[a0:a1], b[b0:b1]):
            if x == y:
                matches += 1
    return matches / max(min(len(a), len(b)), 1)


def _kmer_index(seqs, k=3):
    idx = defaultdict(set)
    for i, s in enumerate(seqs):
        for j in range(len(s) - k + 1):
            idx[s[j:j + k]].add(i)
    return idx


def max_identity_to_reference(query_seqs, ref_seqs, top_k=60, k_prefilter=3,
                              verbose=True, report_every=200):
    """
    For EVERY query sequence, the maximum identity against the FULL reference
    set. Exhaustive alignment is O(n_query * n_ref); we keep it exact-enough
    by first ranking reference sequences with a cheap 3-mer containment
    prefilter and then aligning only the top_k candidates.

    Previously this was computed on 200 sampled generated sequences against
    5,000 sampled training sequences, with an early `break` at 0.99 -- so the
    reported "global max identity" was a subsample maximum, not a global one.
    It was also never applied to the GRU-generated sequences at all, leaving
    the manuscript's memorisation claim unsupported.

    Returns dict with the full per-query vector plus summary statistics.
    """
    if not query_seqs or not ref_seqs:
        return {"identities": [], "mean_max": float("nan"),
                "median_max": float("nan"), "global_max": float("nan"),
                "frac_gt_0.90": float("nan"), "frac_eq_1.00": float("nan"),
                "metric": aligner_kind()}

    ref_index = _kmer_index(ref_seqs, k_prefilter)
    ref_kmer_n = [max(len(s) - k_prefilter + 1, 1) for s in ref_seqs]

    identities = []
    for qi, q in enumerate(query_seqs):
        q_kmers = {q[j:j + k_prefilter] for j in range(len(q) - k_prefilter + 1)}

        shared = Counter()
        for km in q_kmers:
            for ri in ref_index.get(km, ()):
                shared[ri] += 1

        if shared:
            cand = sorted(
                shared.keys(),
                key=lambda ri: shared[ri] / min(max(len(q_kmers), 1), ref_kmer_n[ri]),
                reverse=True,
            )[:top_k]
        else:
            cand = list(range(min(top_k, len(ref_seqs))))

        best = 0.0
        for ri in cand:
            v = pairwise_identity(q, ref_seqs[ri])
            if v > best:
                best = v
                if best >= 1.0:
                    break
        identities.append(best)

        if verbose and (qi + 1) % report_every == 0:
            print(f"      identity {qi + 1}/{len(query_seqs)} "
                  f"(running mean {np.mean(identities):.3f})")

    arr = np.asarray(identities, dtype=float)
    return {
        "identities": identities,
        "mean_max": float(arr.mean()),
        "median_max": float(np.median(arr)),
        "global_max": float(arr.max()),
        "frac_gt_0.90": float((arr > 0.90).mean()),
        "frac_eq_1.00": float((arr >= 0.999).mean()),
        "metric": aligner_kind(),
    }


# --------------------------------------------------------------------------
# Physicochemical featurisation (discriminative validator)
# --------------------------------------------------------------------------

def net_charge(seq):
    return float(sum(CHARGE.get(a, 0.0) for a in seq))


def gravy(seq):
    return float(np.mean([KD.get(a, 0.0) for a in seq])) if seq else 0.0


def hydrophobic_moment(seq, angle=100.0):
    s = np.radians(angle)
    sin_sum = cos_sum = 0.0
    for i, a in enumerate(seq):
        h = KD.get(a, 0.0)
        sin_sum += h * np.sin(i * s)
        cos_sum += h * np.cos(i * s)
    return float(np.sqrt(sin_sum ** 2 + cos_sum ** 2) / max(len(seq), 1))


def aromaticity(seq):
    return sum(1 for a in seq if a in "FWY") / max(len(seq), 1)


def featurize(seqs, include_length=True, verbose=False):
    import pandas as pd

    rows = []
    for i, s in enumerate(seqs):
        f = []
        if include_length:
            f.append(len(s))
        f += [net_charge(s), gravy(s), hydrophobic_moment(s), aromaticity(s)]
        n = max(len(s), 1)
        f += [s.count(a) / n for a in CANONICAL_AAS]
        rows.append(f)
        if verbose and (i + 1) % 5000 == 0:
            print(f"    featurized {i + 1}/{len(seqs)}")

    cols = (["length"] if include_length else []) + \
           ["charge", "gravy", "hydrophobic_moment", "aromaticity"] + \
           [f"aa_{a}" for a in CANONICAL_AAS]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------
# Aggregation helper for multi-seed experiments
# --------------------------------------------------------------------------

def aggregate_seeds(df, group_cols, metric_cols):
    """mean +/- sd across seeds, plus a preformatted 'mean ± sd' string."""
    import pandas as pd

    g = df.groupby(group_cols)
    out = g[metric_cols].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join([c for c in col if c]).rstrip("_")
                   for col in out.columns.to_flat_index()]
    out["n_seeds"] = g.size().values

    for m in metric_cols:
        mu, sd = out[f"{m}_mean"], out[f"{m}_std"].fillna(0.0)
        out[f"{m}_fmt"] = [f"{a:.4f} ± {b:.4f}" for a, b in zip(mu, sd)]
    return out
