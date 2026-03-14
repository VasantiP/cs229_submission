#!/bin/bash
set -e

TRAIN_RECEPTORS=cs229_submission/data/processed/train_receptors_v3_deduped.csv
TEST_RECEPTORS=cs229_submission/data/processed/test_receptors_v3_deduped.csv

cd "$(dirname "$0")"

echo "Training:"
python train_summary.py \
    --train-receptors  $TRAIN_RECEPTORS \
    --test-receptors   $TEST_RECEPTORS \
    --model            lr \
    --penalty          l2 \
    --tune \
    --plot-curves \
    --plot-roc \
    --plot-importance \
    --plot-features

echo "Done."
