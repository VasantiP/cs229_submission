#!/bin/bash
# run_parallel_extraction.sh
# 
# Simple script to run feature extraction in parallel across N cores/sessions

METADATA_CSV="data/processed/data_processed_v3_base_dataset_deduped.csv"
GCS_MOUNT="/home/jupyter/gcs_mount"  # GCS FUSE mount point
OUTPUT_DIR="data/processed_v4"
N_PARTITIONS=4  # Adjust based on available cores/sessions
SKIP_EXISTING=true # Set to false to reprocess everything

echo "Starting parallel feature extraction with $N_PARTITIONS partitions..."

# Option 1: Background processes (simple, works everywhere)
for i in $(seq 0 $(($N_PARTITIONS - 1))); do
    echo "Starting partition $i..."

    CMD="python src/data/extract_parallel.py \
        --metadata_csv $METADATA_CSV \
        --partition $i \
        --n_partitions $N_PARTITIONS \
        --gcs_mount $GCS_MOUNT \
        --output_dir $OUTPUT_DIR"

        # Add skip_existing flag if enabled
        if [ "$SKIP_EXISTING" = true ]; then
            CMD="$CMD --skip_existing"
        fi

        $CMD > logs/partition_${i}.log 2>&1 &
done

echo "All $N_PARTITIONS processes started in background"
echo "Monitor progress with: tail -f logs/partition_*.log"
echo "Wait for completion with: wait"
echo ""
echo "To wait for all to finish, run: wait"

# Uncomment this to wait for all processes:
# wait
# echo "All partitions complete!"