"""
Compute per-frame dynamics features from CA coordinates using sliding windows.

Outputs (n_frames, n_features) NPY files compatible with the TCN pipeline.

Features per frame (computed over a sliding window centered on that frame):
  - Per-residue RMSF (binned into N_BINS regions) → N_BINS features
  - Per-residue displacement mean (binned) → N_BINS features
  - Per-residue autocorrelation lag-1 (binned) → N_BINS features
  - Per-residue autocorrelation lag-5 (binned) → N_BINS features
  - Global summaries (mean/std/max of each feature type) → 4 * 3 = 12 features

Total: N_BINS * 4 + 12 features per frame (default N_BINS=10 → 52 features)

Usage:
    python compute_perframe_autocorr.py \
        --metadata_csv data/metadata/metadatacsv \
        --ca_coord_dir ../gcs_mount/data/ca_coords \
        --output_dir data/processed_v4/features_50pct/autocorr \
        --frac 0.5 --window 50 --n_bins 10
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import time
from multiprocessing import Pool, cpu_count


def find_npy(receptor, rep, simID, npy_dir):
    """Find CA coordinate NPY file."""
    npy_dir = Path(npy_dir)
    receptor_clean = str(receptor).replace('~', '_')
    rep = int(float(rep))
    simID = int(float(simID))
    for name in [
        f"{receptor_clean}_rep{rep}_{simID}.npy",
        f"{receptor_clean}_sim{simID}_rep{rep}.npy",
    ]:
        p = npy_dir / name
        if p.exists():
            return p
    return None


def kabsch_align(mobile, reference):
    """Align mobile to reference. Both (n_atoms, 3)."""
    mobile_c = mobile - mobile.mean(axis=0)
    ref_c = reference - reference.mean(axis=0)
    H = mobile_c.T @ ref_c
    V, _, Wt = np.linalg.svd(H)
    if np.linalg.det(V @ Wt) < 0:
        V[:, -1] *= -1
    return mobile_c @ (V @ Wt)


def compute_perframe_features(ca_coords, skip_eq=100, frac=0.5,
                                window=50, n_bins=10):
    """
    Compute per-frame dynamics features using sliding windows.

    Args:
        ca_coords: (n_frames_total, n_residues, 3)
        skip_eq: frames to skip for equilibration
        frac: fraction of trajectory to use
        window: sliding window size (frames)
        n_bins: number of bins for residue positions

    Returns:
        features: (n_output_frames, n_features) array
        feature_names: list of feature names
    """
    n_total = ca_coords.shape[0]
    start = min(skip_eq, n_total - 1)
    n_use = max(1, int((n_total - start) * frac))
    frames = ca_coords[start:start + n_use].astype(np.float64)

    n_frames, n_res, _ = frames.shape
    half_w = window // 2

    # Align all frames to first frame
    ref = frames[0].copy()
    aligned = np.zeros_like(frames)
    for i in range(n_frames):
        aligned[i] = kabsch_align(frames[i], ref)

    # Precompute per-residue distance from mean for autocorrelation
    global_mean = aligned.mean(axis=0)  # (n_res, 3)
    dist_from_mean = np.sqrt(((aligned - global_mean) ** 2).sum(axis=2))  # (n_frames, n_res)

    # Bin edges for residue positions
    bin_edges = np.linspace(0, n_res, n_bins + 1).astype(int)

    # Build feature names
    feature_names = []
    for feat in ['rmsf', 'disp_mean', 'autocorr1', 'autocorr5']:
        for b in range(n_bins):
            feature_names.append(f"{feat}_bin{b}")
    for feat in ['rmsf', 'disp_mean', 'autocorr1', 'autocorr5']:
        for stat in ['mean', 'std', 'max']:
            feature_names.append(f"{feat}_{stat}")

    n_features = len(feature_names)
    output = np.zeros((n_frames, n_features), dtype=np.float32)

    for t in range(n_frames):
        # Window bounds
        w_start = max(0, t - half_w)
        w_end = min(n_frames, t + half_w + 1)
        w_frames = aligned[w_start:w_end]  # (w_len, n_res, 3)
        w_len = len(w_frames)

        if w_len < 3:
            continue

        # Per-residue RMSF in window
        w_mean = w_frames.mean(axis=0)
        rmsf = np.sqrt(((w_frames - w_mean) ** 2).sum(axis=2).mean(axis=0))  # (n_res,)

        # Per-residue displacement mean in window
        displacements = np.sqrt(((w_frames[1:] - w_frames[:-1]) ** 2).sum(axis=2))  # (w_len-1, n_res)
        disp_mean = displacements.mean(axis=0)  # (n_res,)

        # Per-residue autocorrelation (lag-1) in window
        w_dist = dist_from_mean[w_start:w_end]  # (w_len, n_res)
        autocorr1 = np.zeros(n_res)
        autocorr5 = np.zeros(n_res)
        for r in range(n_res):
            series = w_dist[:, r]
            if series.std() > 1e-10 and len(series) > 2:
                autocorr1[r] = np.corrcoef(series[:-1], series[1:])[0, 1]
            if series.std() > 1e-10 and len(series) > 6:
                autocorr5[r] = np.corrcoef(series[:-5], series[5:])[0, 1]

        # Bin all per-residue features
        feat_idx = 0
        for feat_vals in [rmsf, disp_mean, autocorr1, autocorr5]:
            for b in range(n_bins):
                bin_vals = feat_vals[bin_edges[b]:bin_edges[b+1]]
                if len(bin_vals) > 0:
                    output[t, feat_idx] = np.nanmean(bin_vals)
                feat_idx += 1

        # Global summaries
        for feat_vals in [rmsf, disp_mean, autocorr1, autocorr5]:
            output[t, feat_idx] = np.nanmean(feat_vals)
            output[t, feat_idx + 1] = np.nanstd(feat_vals)
            output[t, feat_idx + 2] = np.nanmax(feat_vals)
            feat_idx += 3

    # Replace NaNs
    output = np.nan_to_num(output, nan=0.0)

    return output, feature_names


def process_one_trajectory(task):
    """Process a single trajectory. Called by Pool.map."""
    row, ca_coord_dir, output_dir, skip_eq, frac, window, n_bins = task

    receptor = row['receptor']
    rep = row['rep']
    simID = row['simID']

    chunk_file = row.get('chunk_file', '')
    if chunk_file:
        out_name = chunk_file
    else:
        receptor_clean = str(receptor).replace('~', '_')
        out_name = f"{receptor_clean}_sim{int(simID)}_rep{int(rep)}.npy"

    out_path = Path(output_dir) / out_name

    if out_path.exists():
        return ('skip', receptor)

    ca_path = find_npy(receptor, rep, simID, ca_coord_dir)
    if ca_path is None:
        return ('fail', f"{receptor}: NPY not found")

    try:
        ca_coords = np.load(ca_path)
        features, _ = compute_perframe_features(
            ca_coords, skip_eq=skip_eq, frac=frac,
            window=window, n_bins=n_bins,
        )
        np.save(out_path, features)
        return ('ok', f"{receptor} {features.shape}")
    except Exception as e:
        return ('fail', f"{receptor}: {e}")


def main(args):
    df = pd.read_csv(args.metadata_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_workers = args.workers if args.workers > 0 else cpu_count()
    print(f"Trajectories: {len(df)}")
    print(f"Window: {args.window}, Bins: {args.n_bins}, Frac: {args.frac}")
    print(f"Workers: {n_workers}")

    # Build task list
    tasks = []
    for _, row in df.iterrows():
        tasks.append((
            row.to_dict(), args.ca_coord_dir, str(output_dir),
            args.skip_eq, args.frac, args.window, args.n_bins
        ))

    # Run in parallel
    t0 = time.time()
    n_ok = n_fail = n_skip = 0

    with Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_one_trajectory, tasks)):
            status, msg = result
            if status == 'ok':
                n_ok += 1
            elif status == 'fail':
                n_fail += 1
                print(f"  FAIL: {msg}")
            else:
                n_skip += 1

            if (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(tasks)}] {elapsed:.0f}s "
                      f"({n_ok} ok, {n_skip} skip, {n_fail} fail)")

    elapsed = time.time() - t0
    print(f"\nDone: {n_ok} success, {n_skip} skipped, {n_fail} failed in {elapsed:.0f}s")
    print(f"Output: {output_dir}")
    print(f"Features per frame: {args.n_bins * 4 + 12}")

    # Save feature names
    _, names = compute_perframe_features(
        np.zeros((200, 300, 3)), skip_eq=0, frac=1.0,
        window=args.window, n_bins=args.n_bins
    )
    with open(output_dir / "feature_names.txt", "w") as f:
        for n in names:
            f.write(n + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--ca_coord_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--frac", type=float, default=0.5)
    parser.add_argument("--skip_eq", type=int, default=100)
    parser.add_argument("--window", type=int, default=50,
                        help="Sliding window size in frames")
    parser.add_argument("--n_bins", type=int, default=10,
                        help="Number of bins for residue positions")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0 = all cores)")
    args = parser.parse_args()
    main(args)