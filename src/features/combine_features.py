"""
Combine multiple feature sets into single NPY files for TCN training.

Concatenates scalar (3-dim) + graph (20-dim) + structural (35-dim)
= 58 features per frame. Also generates the metadata CSV pointing
to the combined files.

Handles:
- Different naming conventions across feature directories
- Variable trajectory lengths (slices to desired fraction)
- Missing features for some trajectories (skips them)

Usage:
    python combine_features.py \
        --metadata data/metadata/metadata_unified.csv \
        --scalar_dir data/processed_v4/features_50pct/scalar \
        --graph_dir data/processed_v4/graph_embeddings \
        --structural_dir data/processed_v4/structural_embeddings \
        --output_dir data/processed_v4/combined_features \
        --frac 0.5 \
        --output_csv data/processed_v4/combined_50pct.csv

    # Multiple fractions at once:
    python combine_features.py \
        --metadata data/metadata/metadata_unified.csv \
        --scalar_dir data/processed_v4/features_50pct/scalar \
        --graph_dir data/processed_v4/graph_embeddings \
        --structural_dir data/processed_v4/structural_embeddings \
        --output_dir data/processed_v4/combined_features \
        --fracs 50 70 90
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def find_feature_file(receptor, rep, simID, feature_dir, patterns=None):
    """
    Find the NPY file for a trajectory in a feature directory.
    Tries multiple naming conventions.
    """
    feature_dir = Path(feature_dir)
    receptor_clean = str(receptor).replace('~', '_')
    rep = int(float(rep))
    simID = str(int(float(simID)))

    # All naming patterns used across the project
    candidates = [
        f"{receptor_clean}_rep{rep}_{simID}.npy",         # graph/structural
        f"{receptor_clean}_sim{simID}_rep{rep}.npy",       # scalar/tica
        f"early_ts_{receptor_clean}_rep{rep}_{simID}.npy", # legacy
    ]

    if patterns:
        candidates = patterns + candidates

    for name in candidates:
        path = feature_dir / name
        if path.exists():
            return path

    return None


def combine_features_for_trajectory(scalar_path, graph_path, structural_path, sanity_check_path=None,
                                      frac=1.0):
    """
    Load and concatenate features from multiple sources.

    Each input: (n_frames, n_features)
    Output: (n_frames_used, total_features)

    For graph/structural (full trajectory), slices to frac.
    For scalar (already sliced), uses as-is.
    """
    features = []
    n_frames_list = []

    # Scalar features (may already be sliced to a fraction)
    if scalar_path and scalar_path.exists():
        scalar = np.load(scalar_path)
        if scalar.ndim == 1:
            # Single vector — expand to (1, n_features) for consistency
            scalar = scalar.reshape(1, -1)
        features.append(('scalar', scalar))
        n_frames_list.append(len(scalar))

    # Graph features (full trajectory — slice to frac)
    if graph_path and graph_path.exists():
        graph = np.load(graph_path)
        n_use = max(1, int(len(graph) * frac))
        graph = graph[:n_use]
        features.append(('graph', graph))
        n_frames_list.append(len(graph))

    # Structural features (full trajectory — slice to frac)
    if structural_path and structural_path.exists():
        structural = np.load(structural_path)
        n_use = max(1, int(len(structural) * frac))
        structural = structural[:n_use]
        features.append(('structural', structural))
        n_frames_list.append(len(structural))

    # Sanity Check features (full trajectory — slice to frac)
    if sanity_check_path and sanity_check_path.exists():
        sanity_check = np.load(sanity_check_path)
        n_use = max(1, int(len(sanity_check) * frac))
        sanity_check = sanity_check[:n_use]
        features.append(('sanity_check', sanity_check))
        n_frames_list.append(len(sanity_check))

    if not features:
        return None, {}

    # Align frame counts — truncate to shortest
    min_frames = min(n_frames_list)

    aligned = []
    dims = {}
    for name, feat in features:
        aligned.append(feat[:min_frames])
        dims[name] = feat.shape[1]

    combined = np.concatenate(aligned, axis=1).astype(np.float32)
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)

    return combined, dims


def main():
    parser = argparse.ArgumentParser(
        description="Combine feature sets into single NPY files for TCN")

    parser.add_argument('--metadata', required=True,
        help='Unified metadata CSV')
    parser.add_argument('--scalar_dir', type=str, default=None,
        help='Directory with scalar feature NPYs')
    parser.add_argument('--graph_dir', type=str, default=None,
        help='Directory with graph embedding NPYs')
    parser.add_argument('--structural_dir', type=str, default=None,
        help='Directory with structural feature NPYs')
    parser.add_argument('--sanity_check_dir', type=str, default=None,
        help='Directory with sanity check feature NPYs (optional)')
    parser.add_argument('--output_dir', required=True,
        help='Output directory for combined NPYs')
    parser.add_argument('--frac', type=float, default=0.5,
        help='Trajectory fraction for graph/structural (scalar assumed pre-sliced)')
    parser.add_argument('--fracs', type=int, nargs='+', default=None,
        help='Multiple fractions to process (e.g., 50 70 90)')
    parser.add_argument('--output_csv', type=str, default=None,
        help='Output metadata CSV (default: output_dir/metadata.csv)')

    args = parser.parse_args()

    fracs = [args.frac] if args.fracs is None else [f/100.0 for f in args.fracs]

    df = pd.read_csv(args.metadata)
    print(f"Metadata: {len(df)} trajectories")

    for frac in fracs:
        pct = int(frac * 100)
        out_dir = Path(args.output_dir) / f"combined_{pct}pct"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Scalar dir might be fraction-specific
        scalar_dir = args.scalar_dir
        if scalar_dir and '{pct}' in scalar_dir:
            scalar_dir = scalar_dir.replace('{pct}', str(pct))

        print(f"\n{'='*60}")
        print(f"Processing {pct}% trajectory fraction")
        print(f"{'='*60}")
        print(f"  Scalar dir:     {scalar_dir}")
        print(f"  Graph dir:      {args.graph_dir}")
        print(f"  Structural dir: {args.structural_dir}")
        print(f"  Output dir:     {out_dir}")

        results = []
        n_success = 0
        n_skip = 0
        feature_dims = None

        for idx, row in df.iterrows():
            receptor = row['receptor']
            rep = row['rep']
            simID = row['simID']

            # Find feature files
            scalar_path = find_feature_file(
                receptor, rep, simID, scalar_dir) if scalar_dir else None
            graph_path = find_feature_file(
                receptor, rep, simID, args.graph_dir) if args.graph_dir else None
            structural_path = find_feature_file(
                receptor, rep, simID, args.structural_dir) if args.structural_dir else None
            sanity_check_path = find_feature_file(
                receptor, rep, simID, args.sanity_check_dir) if args.sanity_check_dir else None

            # Combine
            combined, dims = combine_features_for_trajectory(
                scalar_path, graph_path, structural_path, sanity_check_path=sanity_check_path, frac=frac
            )

            if combined is None:
                n_skip += 1
                continue

            if feature_dims is None:
                feature_dims = dims
                total_dim = combined.shape[1]
                print(f"\n  Feature dimensions: {dims}")
                print(f"  Total: {total_dim} features per frame")

            # Save combined NPY (matches generate_splits.py convention)
            out_name = f"{str(receptor).replace('~','_')}_sim{int(float(simID))}_rep{int(float(rep))}.npy"
            out_path = out_dir / out_name
            np.save(out_path, combined)

            results.append({
                'receptor': receptor,
                'rep': rep,
                'simID': simID,
                'y': row['y'],
                'family': row.get('family', 'Unknown'),
                'chunk_file': out_name,
                'n_frames': combined.shape[0],
                'n_features': combined.shape[1],
            })
            n_success += 1

        # Save metadata CSV
        results_df = pd.DataFrame(results)
        csv_path = args.output_csv if args.output_csv else str(out_dir / "metadata.csv")
        if len(fracs) > 1:
            csv_path = str(out_dir / "metadata.csv")
        results_df.to_csv(csv_path, index=False)

        print(f"\n  Success: {n_success}, Skipped: {n_skip}")
        print(f"  Saved: {csv_path}")
        print(f"  Multi: {results_df['y'].sum():.0f}, "
              f"Single: {(results_df['y']==0).sum()}")

    # Print feature index reference
    if feature_dims:
        print(f"\n{'='*60}")
        print("FEATURE INDEX REFERENCE")
        print(f"{'='*60}")
        offset = 0
        for name, dim in feature_dims.items():
            print(f"  {name}: columns {offset}-{offset+dim-1} ({dim} features)")
            offset += dim
        print(f"  Total: {offset} features")
        print(f"\n  Use --feature_cols in train_tcn.py to select subsets")


if __name__ == "__main__":
    main()