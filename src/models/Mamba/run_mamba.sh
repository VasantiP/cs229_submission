#!/bin/bash
set -e

TRAIN_CSV=/tmp/emb_bootstrap_train_chunks_newsplit.csv
TEST_CSV=/tmp/emb_bootstrap_test_chunks_newsplit.csv
LOCAL_CACHE=/home/jupyter/raw_chunk_cache/all
OUTPUT_DIR=/home/jupyter/runs_mamba
RUN_NAME=${1:-"run_$(date +%Y%m%d_%H%M%S)"}

cd "$(dirname "$0")"

echo "Training:"
python train_mamba.py \
    --train-csv    $TRAIN_CSV \
    --test-csv     $TEST_CSV \
    --local-cache  $LOCAL_CACHE \
    --output-dir   $OUTPUT_DIR \
    --run-name     $RUN_NAME \
    --pooling      attention \
    --epochs       80 \
    --peak-lr      1e-4 \
    --warmup       2 \
    --patience     6 \
    --micro-bs     16 \
    --d-model      128 \
    --d-inner      128 \
    --n-layers     2 \
    --d-state      16 \
    --dropout      0.3

echo "Done. Outputs in $OUTPUT_DIR/$RUN_NAME/"
