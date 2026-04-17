#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
cd "$ROOT_DIR"

DATA_PATH=./data/datasets/fineweb10B_sp1024
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
VOCAB_SIZE=1024
MAX_WALLCLOCK_SECONDS=600
VAL_LOSS_EVERY=0
TRAIN_LOG_EVERY=200

run_trial() {
  local run_id="$1"
  shift
  env \
    RUN_ID="$run_id" \
    DATA_PATH="$DATA_PATH" \
    TOKENIZER_PATH="$TOKENIZER_PATH" \
    VOCAB_SIZE="$VOCAB_SIZE" \
    MAX_WALLCLOCK_SECONDS="$MAX_WALLCLOCK_SECONDS" \
    VAL_LOSS_EVERY="$VAL_LOSS_EVERY" \
    TRAIN_LOG_EVERY="$TRAIN_LOG_EVERY" \
    "$@" \
    torchrun --standalone --nproc_per_node=1 train_gpt.py
}

run_trial baseline_sp1024_single_h100
run_trial lower_lr_sp1024_single_h100 \
  MATRIX_LR=0.02 \
  SCALAR_LR=0.02 \
  TIED_EMBED_LR=0.03
run_trial seq4096_lowerlr_single_h100 \
  TRAIN_SEQ_LEN=4096 \
  TRAIN_BATCH_TOKENS=393216 \
  MATRIX_LR=0.02 \
  SCALAR_LR=0.02 \
  TIED_EMBED_LR=0.03 \
  MUON_MOMENTUM=0.99 \
  MUON_MOMENTUM_WARMUP_STEPS=1500 \
  MUON_MOMENTUM_WARMUP_START=0.92 \
  WARMDOWN_ITERS=3000
