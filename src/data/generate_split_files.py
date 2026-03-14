"""
Generate train/test split CSVs for all experiment types.

Uses family_mapping.py as the canonical source of truth for family assignments.
All splits operate at the TRAJECTORY level (one row = one trajectory).

Hold out entire protein families for testing

Usage:
    # Cross-protein (default holdout: Peptide, Adenosine, Serotonin):
    python generate_splits.py cross-protein \
        --metadata_csv metadata_gpcr.csv \
        --trajectory_pct 50 --feature_type scalar
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

import sys
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent / "utils"))
from family_mapping import assign_family


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_metadata(path):
    """Load metadata CSV and ensure family column exists."""
    df = pd.read_csv(path)

    if "family" not in df.columns:
        if "receptor" not in df.columns:
            raise ValueError("CSV must have 'receptor' or 'family' column")
        print("Applying family mapping...")
        df["family"] = df["receptor"].apply(assign_family)
        df = df[df["family"].notna()]  # drop non-GPCR

    print(f"Loaded {len(df)} trajectories, {df['family'].nunique()} families")
    print(f"Labels: {df['y'].value_counts().to_dict()}")
    return df


def resolve_feature_dir(feature_base_dir, trajectory_pct, feature_type):
    """
    Build the feature directory path following project convention:
        data/processed_v4/features_50pct/scalar/
        data/processed_v4/features_50pct/tica/projections/
    """
    pct_str = f"{trajectory_pct}pct"
    if feature_type == "scalar":
        return Path(feature_base_dir) / f"features_{pct_str}" / "scalar"
    elif feature_type == "tica":
        return Path(feature_base_dir) / f"features_{pct_str}" / "tica" / "projections"
    elif feature_type == "combined":
        return Path(feature_base_dir) / f"combined_features" / f"combined_{pct_str}"
    elif feature_type == "sanity_check":
        return Path(feature_base_dir) / f"features_{pct_str}" / "sanity_check_features"
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")


def make_chunk_filename(row):
    """
    Construct NPY filename from metadata row.
    Convention: {receptor_with_tilde_replaced}_sim{simID}_rep{rep}.npy
    """
    receptor = row["receptor"].replace("~", "_")
    sim_id = row.get("simID", row.get("sim_id", 0))
    rep = row.get("rep", 1)
    return f"{receptor}_sim{int(sim_id)}_rep{int(rep)}.npy"


def make_traj_id(row):
    """Unique trajectory identifier (for deduplication / grouping)."""
    receptor = row["receptor"].replace("~", "_")
    sim_id = row.get("simID", row.get("sim_id", 0))
    rep = row.get("rep", 1)
    return f"{receptor}_sim{int(sim_id)}_rep{int(rep)}"


def add_derived_columns(df):
    """Add chunk_file and traj_id columns if not present."""
    if "chunk_file" not in df.columns:
        df["chunk_file"] = df.apply(make_chunk_filename, axis=1)
    if "traj_id" not in df.columns:
        df["traj_id"] = df.apply(make_traj_id, axis=1)
    return df


def filter_existing_npy(df, feature_dir):
    """Filter to only rows whose NPY file exists. Returns (filtered_df, n_missing)."""
    feature_dir = Path(feature_dir)
    print(f"Filtering rows based on NPY files in {feature_dir}")
    if not feature_dir.exists():
        print(f"  WARNING: {feature_dir} does not exist")
        return df, 0

    exists_mask = df["chunk_file"].apply(lambda f: (feature_dir / f).exists())
    n_missing = (~exists_mask).sum()
    filtered = df[exists_mask].copy()

    if n_missing > 0:
        missing_files = df[~exists_mask]["chunk_file"].tolist()
        print(f"  {len(filtered)}/{len(df)} NPY files found ({n_missing} missing)")
        for f in missing_files[:5]:
            print(f"    missing: {f}")
        if n_missing > 5:
            print(f"    ... and {n_missing - 5} more")
    else:
        print(f"  All {len(filtered)} NPY files found")

    return filtered, n_missing


def print_class_dist(df, label=""):
    """Print class distribution."""
    n_multi = int(df["y"].sum())
    n_single = int((df["y"] == 0).sum())
    print(f"  {label}: {n_multi} multi, {n_single} single ({len(df)} total)")


# ══════════════════════════════════════════════════════════════════════════════
# Cross-protein splits
# ══════════════════════════════════════════════════════════════════════════════

def generate_cross_protein(df, holdout_families, feature_dir, output_dir):
    """Hold out entire protein families for testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_mask = df["family"].isin(holdout_families)
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()

    # Filter to existing NPY files
    print("\nFiltering train set:")
    train_df, _ = filter_existing_npy(train_df, feature_dir)
    print("Filtering test set:")
    test_df, _ = filter_existing_npy(test_df, feature_dir)

    print(f"\nCross-protein split:")
    print_class_dist(train_df, "Train")
    print_class_dist(test_df, f"Test ({', '.join(holdout_families)})")

    train_path = output_dir / "cross_protein_train.csv"
    test_path = output_dir / "cross_protein_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"\n  Saved: {train_path}")
    print(f"  Saved: {test_path}")

    return train_path, test_path


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate train/test splits for MD prediction experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── Shared arguments ──────────────────────────────────────────────────
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--metadata_csv", required=True,
                        help="Path to metadata CSV (e.g. metadata_gpcr.csv)")
    shared.add_argument("--feature_base_dir", default="data/processed_v4",
                        help="Base dir containing features_XXpct/ dirs")
    shared.add_argument("--feature_type", default="scalar",
                        choices=["scalar", "tica", "combined", "sanity_check"],)
    shared.add_argument("--trajectory_pct", type=int, default=50,
                        help="Trajectory percentage for feature extraction")
    shared.add_argument("--output_dir", default="splits/")

    # ── Cross-protein ─────────────────────────────────────────────────────
    cp = subparsers.add_parser("cross-protein", parents=[shared],
                                help="Hold out protein families for testing")
    cp.add_argument("--holdout_families", nargs="+",
                    default=["Peptide", "Adenosine", "Serotonin"])

    args = parser.parse_args()
    df = load_metadata(args.metadata_csv)
    df = add_derived_columns(df)

    feature_dir = resolve_feature_dir(
        args.feature_base_dir, args.trajectory_pct, args.feature_type
    )

    generate_cross_protein(df, args.holdout_families, feature_dir,
                                args.output_dir)

if __name__ == "__main__":
    main()