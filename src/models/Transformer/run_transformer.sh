#!/bin/bash
set -e

TRAIN_CSV=/tmp/emb_bootstrap_train_chunks_newsplit.csv
TEST_CSV=/tmp/emb_bootstrap_test_chunks_newsplit.csv
LOCAL_CACHE=/home/jupyter/raw_chunk_cache/all
OUTPUT_DIR=/home/jupyter/runs_transformer
RUN_NAME=${1:-"run_$(date +%Y%m%d_%H%M%S)"}

cd "$(dirname "$0")"

echo "Training:"
python train_resattn_transformer.py \
    --train-csv    $TRAIN_CSV \
    --test-csv     $TEST_CSV \
    --local-cache  $LOCAL_CACHE \
    --output-dir   $OUTPUT_DIR \
    --run-name     $RUN_NAME \
    --epochs       80 \
    --peak-lr      3e-4 \
    --warmup       8 \
    --patience     20 \
    --micro-bs     16 \
    --d-model      128 \
    --d-inner      128 \
    --n-heads      4 \
    --n-layers     2 \
    --dim-ff       256 \
    --dropout      0.3 \
    --layer-drop   0.1 \
    --alibi

echo "Evaluation:"
python analyze_transformer.py \
    --checkpoint   $OUTPUT_DIR/$RUN_NAME/checkpoint_best.pt \
    --test-csv     $TEST_CSV \
    --local-cache  $LOCAL_CACHE \
    --n-shuffles   5

echo "Residue attention analysis:"
python analyze_residue_mapping.py \
    --checkpoint   $OUTPUT_DIR/$RUN_NAME/checkpoint_best.pt \
    --test-csv     $TEST_CSV \
    --local-cache  $LOCAL_CACHE

echo "Done. Outputs in $OUTPUT_DIR/$RUN_NAME/"
