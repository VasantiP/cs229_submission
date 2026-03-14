"""
Generate a single receptor-grouped train/test split.

Splits so that no receptor appears in both train and test,
but receptors from the same family CAN appear in both.

Usage:
    python generate_receptor_split.py \
        --metadata_csv data/metadata/metadata.csv \
        --output_dir data/metadata/splits/receptor_split \
        --test_frac 0.2
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def main(args):
    df = pd.read_csv(args.metadata_csv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Total: {len(df)} trajectories, {df['receptor'].nunique()} receptors")
    print(f"Multi: {df['y'].sum():.0f}, Single: {(df['y']==0).sum():.0f}\n")

    # Shuffle receptors and split
    unique_receptors = df['receptor'].unique()
    np.random.seed(args.seed)
    np.random.shuffle(unique_receptors)

    n_test = max(1, int(len(unique_receptors) * args.test_frac))
    test_receptors = set(unique_receptors[:n_test])
    train_receptors = set(unique_receptors[n_test:])

    test_mask = df['receptor'].isin(test_receptors)
    train_df = df[~test_mask]
    test_df = df[test_mask]

    # Save
    train_df.to_csv(output_dir / "train_chunks.csv", index=False)
    test_df.to_csv(output_dir / "test_chunks.csv", index=False)

    print(f"Train: {len(train_df)} trajectories, {len(train_receptors)} receptors "
          f"(multi={train_df['y'].sum():.0f})")
    print(f"Test:  {len(test_df)} trajectories, {len(test_receptors)} receptors "
          f"(multi={test_df['y'].sum():.0f})")

    # Show which families appear in both
    train_families = set(train_df['family'].unique())
    test_families = set(test_df['family'].unique())
    shared = train_families & test_families
    print(f"\nFamilies in both train & test: {len(shared)}/{len(test_families)}")

    print(f"\nSaved to {output_dir}")
    print(f"\nTo train:")
    print(f"  python train_tcn.py \\")
    print(f"    --train_csv {output_dir}/protein_level_train.csv \\")
    print(f"    --test_csv {output_dir}/protein_level_test.csv \\")
    print(f"    --npy_dir <your_npy_dir>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)