#!/bin/bash
set -e

TRAIN_RECEPTORS=cs229-md-prediction/data/processed/train_receptors_v3_deduped.csv
TEST_RECEPTORS=cs229-md-prediction/data/processed/test_receptors_v3_deduped.csv

cd "$(dirname "$0")"

echo "Training:"
python train_frame_mlp.py \
    --train-receptors  $TRAIN_RECEPTORS \
    --test-receptors   $TEST_RECEPTORS \
    --hidden           15 \
    --dropout          0.1 \
    --pool             mean+std \
    --epochs           1000 \
    --lr               1e-3 \
    --wd               1e-3 \
    --patience         100 \
    --plot-curves \
    --plot-roc

echo "Done."
