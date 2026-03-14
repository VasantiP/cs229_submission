"""
TCN Sanity Check: Per-frame features derived from label-generating pipeline.

Replicates the exact labeling pipeline (TICA + k-means on full trajectory)
and extracts per-frame features that BY CONSTRUCTION encode the label.
If the TCN can't classify these, the model/training is broken.

Pipeline per trajectory:
    1. Load CA coords (n_frames, n_residues, 3)
    2. Skip first 100 equilibration frames
    3. Kabsch-align to first post-equilibration frame
    4. Flatten to (T, 3N)
    5. TICA (lag=10, dim=2) → per-frame 2D projections
    6. K-means (k=2) in TICA space
    7. Per-frame features:
        - TICA projection (2 features)
        - Distance to centroid 0 (1 feature)
        - Distance to centroid 1 (1 feature)
        - Silhouette sample score (1 feature)
        - Cluster assignment as float (1 feature)
        - Running dominant cluster fraction (1 feature)
        Total: 7 features per frame

Usage:
    # Timing test
    python sanity_check_features.py timing_test \
        --coord_file data/processed_v4/ca_coords/some_receptor.npy

    # Batch processing
    python sanity_check_features.py batch \
        --coord_dir data/processed_v4/ca_coords \
        --output_dir data/processed_v4/sanity_check_features \
        --n_workers 8

    # Then train TCN:
    python train_tcn.py \
        --train_csv splits/sanity_cross_protein_train.csv \
        --test_csv splits/sanity_cross_protein_test.csv \
        --npy_dir data/processed_v4/sanity_check_features \
        --batch_size 1 --num_workers 0 \
        --output_dir results/tcn_sanity_check
"""

import numpy as np
from pathlib import Path
import argparse
import time
from multiprocessing import Pool, cpu_count

try:
    import pyemma.coordinates as coor
    HAS_PYEMMA = True
except ImportError:
    HAS_PYEMMA = False
    print("WARNING: pyemma not installed. Install: pip install pyemma")

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


# ══════════════════════════════════════════════════════════════════════════════
# Kabsch alignment (matches data_processing.ipynb exactly)
# ══════════════════════════════════════════════════════════════════════════════

def kabsch_align_centered(mobile, reference):
    """
    Kabsch algorithm: align mobile to reference, both centered.
    Returns (mobile_aligned_centered, reference_centered).
    """
    mobile_centered = mobile - mobile.mean(axis=0)
    reference_centered = reference - reference.mean(axis=0)

    covariance = mobile_centered.T @ reference_centered
    V, _, Wt = np.linalg.svd(covariance)

    if np.linalg.det(V @ Wt) < 0:
        V[:, -1] *= -1

    rotation = V @ Wt
    mobile_aligned_centered = mobile_centered @ rotation
    return mobile_aligned_centered, reference_centered


# ══════════════════════════════════════════════════════════════════════════════
# Core: compute sanity check features for one trajectory
# ══════════════════════════════════════════════════════════════════════════════

def compute_sanity_features(ca_coords, skip_eq=100, tica_lag=10, tica_dim=2,
                             n_clusters=2, random_state=0):
    """
    Replicate the labeling pipeline and extract per-frame features.

    Args:
        ca_coords: (n_frames, n_residues, 3) array of CA coordinates
        skip_eq: number of equilibration frames to skip
        tica_lag: TICA lag time
        tica_dim: TICA output dimensions
        n_clusters: number of k-means clusters
        random_state: random seed for k-means

    Returns:
        features: (T, 7) array of per-frame features
        info: dict with label-generating metrics
    """
    if not HAS_PYEMMA:
        raise ImportError("pyemma required for TICA")

    n_frames = ca_coords.shape[0]
    start = min(skip_eq, n_frames - 1)

    # Step 1: Align to first post-equilibration frame (Kabsch)
    ref = ca_coords[start].astype(np.float64)
    aligned = []
    for i in range(start, n_frames):
        mobile = ca_coords[i].astype(np.float64)
        mobile_aligned, _ = kabsch_align_centered(mobile, ref)
        aligned.append(mobile_aligned)

    aligned = np.array(aligned, dtype=np.float64)  # (T, n_residues, 3)
    T = aligned.shape[0]

    # Flatten to (T, 3N) — matches notebook convention
    X_flat = aligned.reshape(T, -1)

    # Step 2: TICA
    if T <= tica_lag + 1:
        raise ValueError(f"Not enough frames for TICA: T={T}, lag={tica_lag}")

    # pyemma.coordinates.tica expects list of arrays or single array
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.bool = bool  # pyemma compatibility
        tica = coor.tica(X_flat, lag=tica_lag, dim=tica_dim)
        traj_tica = tica.get_output()[0]  # (T, tica_dim)

    # Step 3: K-means clustering in TICA space
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    cluster_labels = kmeans.fit_predict(traj_tica)
    centroids = kmeans.cluster_centers_  # (n_clusters, tica_dim)

    # Step 4: Per-frame features

    # 4a: TICA projections (2 features)
    tica_proj = traj_tica  # (T, 2)

    # 4b: Distance to each centroid (2 features)
    dist_to_c0 = np.linalg.norm(traj_tica - centroids[0], axis=1, keepdims=True)
    dist_to_c1 = np.linalg.norm(traj_tica - centroids[1], axis=1, keepdims=True)

    # 4c: Silhouette sample scores (1 feature)
    unique_labels = np.unique(cluster_labels)
    if len(unique_labels) >= 2:
        sil_samples = silhouette_samples(traj_tica, cluster_labels).reshape(-1, 1)
    else:
        sil_samples = np.zeros((T, 1))

    # 4d: Cluster assignment (1 feature)
    cluster_assign = cluster_labels.astype(np.float32).reshape(-1, 1)

    # 4e: Running dominant cluster fraction (1 feature)
    # At each frame t, what fraction of frames 0..t are in the dominant cluster
    running_dominant = np.zeros((T, 1), dtype=np.float32)
    for t in range(T):
        counts = np.bincount(cluster_labels[:t+1], minlength=n_clusters)
        running_dominant[t] = counts.max() / counts.sum()

    # Concatenate all features
    features = np.concatenate([
        tica_proj,          # 2: TICA projections
        dist_to_c0,         # 1: distance to centroid 0
        dist_to_c1,         # 1: distance to centroid 1
        sil_samples,        # 1: silhouette sample score
        cluster_assign,     # 1: cluster assignment
        running_dominant,   # 1: running dominant cluster fraction
    ], axis=1).astype(np.float32)

    # Also compute the label-generating metrics for validation
    counts = np.bincount(cluster_labels, minlength=n_clusters)
    fractions = counts / counts.sum()
    dominant_frac = float(np.max(fractions))

    if len(unique_labels) >= 2:
        from sklearn.metrics import silhouette_score
        sil_score = float(silhouette_score(traj_tica, cluster_labels))
    else:
        sil_score = np.nan

    info = {
        'T_frames': T,
        'n_features': features.shape[1],
        'dominant_cluster_frac': dominant_frac,
        'silhouette_score': sil_score,
        'cluster_counts': counts.tolist(),
    }

    return features, info


# ══════════════════════════════════════════════════════════════════════════════
# Processing functions
# ══════════════════════════════════════════════════════════════════════════════

def process_one(args):
    """Process a single trajectory (for multiprocessing)."""
    coord_file, output_dir = args
    coord_file = Path(coord_file)
    output_dir = Path(output_dir)

    stem = coord_file.stem
    out_path = output_dir / f"{stem}.npy"

    if out_path.exists():
        return stem, 'skip', None

    try:
        ca_coords = np.load(coord_file)
        features, info = compute_sanity_features(ca_coords)
        np.save(out_path, features)
        return stem, 'ok', info
    except Exception as e:
        return stem, 'error', str(e)


def batch_process(coord_dir, output_dir, n_workers=1):
    """Process all trajectories in coord_dir."""
    coord_dir = Path(coord_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(coord_dir.glob("*.npy"))
    print(f"Found {len(npy_files)} coordinate files in {coord_dir}")

    tasks = [(str(f), str(output_dir)) for f in npy_files]

    n_ok = n_skip = n_err = 0
    t0 = time.time()

    if n_workers <= 1:
        results = [process_one(t) for t in tasks]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(process_one, tasks)

    for stem, status, info in results:
        if status == 'ok':
            n_ok += 1
            if info:
                print(f"  {stem}: {info['T_frames']} frames, "
                      f"dominant={info['dominant_cluster_frac']:.3f}, "
                      f"silhouette={info['silhouette_score']:.3f}")
        elif status == 'skip':
            n_skip += 1
        else:
            n_err += 1
            print(f"  ERROR {stem}: {info}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s: {n_ok} ok, {n_skip} skipped, {n_err} errors")


def timing_test(coord_file):
    """Time the processing of a single trajectory."""
    ca_coords = np.load(coord_file)
    print(f"Timing test: {coord_file}")
    print(f"  Shape: {ca_coords.shape}")

    t0 = time.time()
    features, info = compute_sanity_features(ca_coords)
    elapsed = time.time() - t0

    print(f"  Output: {features.shape}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Dominant cluster frac: {info['dominant_cluster_frac']:.3f}")
    print(f"  Silhouette score: {info['silhouette_score']:.3f}")
    print(f"  Cluster counts: {info['cluster_counts']}")
    print(f"\n  Estimated for 242 trajectories (1 worker): {elapsed * 242 / 60:.1f} min")
    print(f"  Estimated for 242 trajectories (8 workers): {elapsed * 242 / 8 / 60:.1f} min")

    # Feature names for reference
    print(f"\n  Feature columns:")
    names = ['tica_0', 'tica_1', 'dist_centroid_0', 'dist_centroid_1',
             'silhouette_sample', 'cluster_label', 'running_dominant_frac']
    for i, name in enumerate(names):
        vals = features[:, i]
        print(f"    [{i}] {name}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
              f"min={vals.min():.4f}, max={vals.max():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compute TCN sanity check features from full-trajectory TICA+clustering")

    subparsers = parser.add_subparsers(dest='command', required=True)

    # Timing test
    tt = subparsers.add_parser('timing_test')
    tt.add_argument('--coord_file', required=True)

    # Batch processing
    bp = subparsers.add_parser('batch')
    bp.add_argument('--coord_dir', required=True)
    bp.add_argument('--output_dir', required=True)
    bp.add_argument('--n_workers', type=int, default=min(cpu_count(), 8))

    args = parser.parse_args()

    if args.command == 'timing_test':
        timing_test(args.coord_file)
    elif args.command == 'batch':
        batch_process(args.coord_dir, args.output_dir, args.n_workers)


if __name__ == "__main__":
    main()