#!/usr/bin/env bash
# run_all.sh -- full ablation + baseline suite, in dependency order.
#
#   bash run_all.sh AMPuniqmoree5_clean_rmdup.fasta namp_clean.fasta
#
# Stage order matters: ablation.py and baseline_generation.py write the FASTA
# files that the three evaluation stages consume.

set -euo pipefail

POS="${1:?usage: bash run_all.sh <amp.fasta> <namp.fasta> [out_dir]}"
NEG="${2:?usage: bash run_all.sh <amp.fasta> <namp.fasta> [out_dir]}"
OUT="${3:-ablation_results}"

SEEDS="${SEEDS:-3}"
NGEN="${NGEN:-2000}"
EPOCHS="${EPOCHS:-15}"
TEMP="${TEMP:-1.5}"                       # used by the GRU/LSTM and token ablations
TEMPS="${TEMPS:-0.8 1.0 1.2 1.5}"         # swept in the temperature ablation
COMPONENTS="${COMPONENTS:-1 2 3 4 5 6 7 8}"
KS="${KS:-2 3 4 5 6}"

mkdir -p "$OUT"

echo "=============================================================="
echo " Stage 1/5  Ablations (GMM / GRU-LSTM / special tokens / temperature)"
echo "=============================================================="
python ablation.py \
    --fasta "$POS" \
    --out_dir "$OUT" \
    --run gmm_components model_type special_tokens temperature \
    --epochs "$EPOCHS" \
    --n_generate "$NGEN" \
    --temperature "$TEMP" \
    --temperatures $TEMPS \
    --components $COMPONENTS \
    --seeds "$SEEDS"

echo
echo "=============================================================="
echo " Stage 2/5  Generation baselines (simple/trivial models)"
echo "=============================================================="
python baseline_generation.py \
    --fasta "$POS" \
    --out_dir "$OUT" \
    --n_generate "$NGEN" \
    --seeds "$SEEDS"

echo
echo "=============================================================="
echo " Stage 3/5  Classifier baselines + discriminative scoring"
echo "=============================================================="
python ablation_rf.py \
    --pos_fasta "$POS" \
    --neg_fasta "$NEG" \
    --out_dir "$OUT"

echo
echo "=============================================================="
echo " Stage 4/5  k-mer divergence and novelty screen"
echo "=============================================================="
python kmer_eval.py --real "$POS" --gen_dir "$OUT" --out_dir "$OUT" --ks $KS
python novelty_eval.py --real "$POS" --gen_dir "$OUT" --out_dir "$OUT"

echo
echo "=============================================================="
echo " Stage 5/5  Supplementary tables"
echo "=============================================================="
python make_tables.py --out_dir "$OUT"

echo
echo "Done. Tables in $OUT/tables/, figures in $OUT/"
