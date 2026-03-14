#!/bin/bash
set -e

PRECOMPUTED_EV=notebooks/bootstrap_tica_eigenvalues.csv
PLOT_OUT=src/models/TICA_threshold/tica_plot.png

cd "$(dirname "$0")/../../.."

echo "Running TICA eigenvalue classifier ..."
python src/models/TICA_threshold/tica_threshold_classifier.py \
    --load-ev  $PRECOMPUTED_EV \
    --n-ev     2 \
    --n-ev-list 1 2 3 5 10 \
    --plot     $PLOT_OUT \
    --latex

echo "Done. Plot saved to $PLOT_OUT"
