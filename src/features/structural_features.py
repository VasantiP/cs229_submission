"""
Residue-level structural features from CA coordinates.

Computes per-frame features that capture LOCAL structural changes,
unlike the global graph statistics which average everything away.

Features computed per frame:
  1. Windowed RMSF (per-residue flexibility in sliding window)
     → stats: mean, std, max, skew of per-residue RMSF
     → also: fraction of "flexible" residues (RMSF > threshold)
  2. Contact map evolution (which contacts are forming/breaking)
     → stats: contact gain/loss rate, contact map entropy
  3. Contact order (avg sequence separation of contacts)
     → captures whether contacts are local (helical) or long-range (tertiary)
  4. Distance matrix statistics
     → radius of gyration from CA, asphericity, max extent
  5. Hinge detection (from RMSF gradient)
     → number and location of flexibility transitions

All features are aggregated into a fixed-size vector regardless of
protein size (different # residues across GPCRs).

Usage:
    python structural_features.py \
        --coord_dir data/processed_v4/ca_coords \
        --output_dir data/processed_v4/structural_embeddings \
        --n_workers 8

Output: (n_frames, n_features) NPY per trajectory, same format as graph embeddings.
"""

import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import skew, kurtosis
import argparse
import time


# ══════════════════════════════════════════════════════════════════════════════
# Feature computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_rmsf_windowed(coords, window=50):
    """
    Compute per-residue RMSF over a sliding window centered at each frame.

    Args:
        coords: (n_frames, n_residues, 3)
        window: number of frames in sliding window

    Returns:
        rmsf: (n_frames, n_residues) — per-residue RMSF at each frame
    """
    n_frames, n_res, _ = coords.shape
    half_w = window // 2
    rmsf = np.zeros((n_frames, n_res), dtype=np.float32)

    for i in range(n_frames):
        start = max(0, i - half_w)
        end = min(n_frames, i + half_w + 1)
        window_coords = coords[start:end]  # (w, n_res, 3)
        mean_pos = window_coords.mean(axis=0)  # (n_res, 3)
        deviations = window_coords - mean_pos[np.newaxis]  # (w, n_res, 3)
        rmsf[i] = np.sqrt((deviations ** 2).sum(axis=2).mean(axis=0))

    return rmsf


def compute_contact_maps(coords, cutoff=10.0):
    """
    Compute binary contact maps for each frame.

    Args:
        coords: (n_frames, n_residues, 3)
        cutoff: distance cutoff in Angstroms

    Returns:
        contacts: (n_frames, n_residues, n_residues) boolean
    """
    n_frames, n_res, _ = coords.shape
    contacts = np.zeros((n_frames, n_res, n_res), dtype=np.bool_)

    for i in range(n_frames):
        dist = squareform(pdist(coords[i]))
        c = dist < cutoff
        # Zero out diagonal and i,i+1
        np.fill_diagonal(c, False)
        for j in range(n_res - 1):
            c[j, j+1] = False
            c[j+1, j] = False
        contacts[i] = c

    return contacts


def frame_features(coords_frame, coords_prev_frame, rmsf_frame,
                    contact_current, contact_prev, contact_ref,
                    cutoff=10.0):
    """
    Compute all features for a single frame.

    Returns: fixed-size feature vector (same size regardless of n_residues)
    """
    n_res = len(coords_frame)
    features = []

    # ── 1. RMSF statistics (7 features) ──
    rmsf = rmsf_frame
    features.extend([
        rmsf.mean(),
        rmsf.std(),
        rmsf.max(),
        rmsf.min(),
        np.median(rmsf),
        skew(rmsf) if len(rmsf) > 2 else 0.0,
        # Fraction of "flexible" residues (RMSF > 2x median)
        (rmsf > 2 * np.median(rmsf)).mean(),
    ])

    # ── 2. RMSF spatial distribution (5 features) ──
    # Split protein into 5 equal segments, report RMSF per segment
    segment_size = max(1, n_res // 5)
    for seg in range(5):
        start = seg * segment_size
        end = min((seg + 1) * segment_size, n_res)
        features.append(rmsf[start:end].mean())

    # ── 3. Hinge detection from RMSF gradient (3 features) ──
    if len(rmsf) > 3:
        rmsf_gradient = np.abs(np.diff(rmsf))
        threshold = rmsf_gradient.mean() + 2 * rmsf_gradient.std()
        n_hinges = (rmsf_gradient > threshold).sum()
        features.extend([
            float(n_hinges),
            rmsf_gradient.mean(),
            rmsf_gradient.max(),
        ])
    else:
        features.extend([0.0, 0.0, 0.0])

    # ── 4. Contact map statistics (6 features) ──
    n_contacts = contact_current.sum() / 2  # symmetric
    max_possible = (n_res * (n_res - 1)) / 2 - (n_res - 1)
    contact_density = n_contacts / max(max_possible, 1)

    # Contact order: avg sequence separation of contacts
    contact_pairs = np.argwhere(np.triu(contact_current, k=2))
    if len(contact_pairs) > 0:
        seq_separations = np.abs(contact_pairs[:, 0] - contact_pairs[:, 1])
        contact_order = seq_separations.mean() / n_res  # normalized
        contact_order_std = seq_separations.std() / n_res
        # Fraction long-range (|i-j| > 12, beyond 3 helical turns)
        frac_long_range = (seq_separations > 12).mean()
    else:
        contact_order = 0.0
        contact_order_std = 0.0
        frac_long_range = 0.0

    features.extend([
        float(n_contacts),
        contact_density,
        contact_order,
        contact_order_std,
        frac_long_range,
        float(n_res),  # for normalization downstream
    ])

    # ── 5. Contact evolution (5 features) ──
    if contact_prev is not None:
        contacts_gained = (contact_current & ~contact_prev).sum() / 2
        contacts_lost = (~contact_current & contact_prev).sum() / 2
        contacts_stable = (contact_current & contact_prev).sum() / 2
        total_prev = max(contact_prev.sum() / 2, 1)
        turnover_rate = (contacts_gained + contacts_lost) / max(n_contacts + total_prev, 1)
    else:
        contacts_gained = 0
        contacts_lost = 0
        contacts_stable = n_contacts
        turnover_rate = 0.0

    # Deviation from reference (first frame) contact map
    if contact_ref is not None:
        ref_deviation = (contact_current != contact_ref).sum() / 2
        ref_deviation_frac = ref_deviation / max(max_possible, 1)
    else:
        ref_deviation_frac = 0.0

    features.extend([
        float(contacts_gained),
        float(contacts_lost),
        turnover_rate,
        float(contacts_stable),
        ref_deviation_frac,
    ])

    # ── 6. Distance / shape features (6 features) ──
    dist_matrix = squareform(pdist(coords_frame))
    upper_tri = dist_matrix[np.triu_indices(n_res, k=1)]

    # Radius of gyration from CA
    centroid = coords_frame.mean(axis=0)
    rg = np.sqrt(((coords_frame - centroid) ** 2).sum(axis=1).mean())

    # Max extent
    max_dist = upper_tri.max()

    # Asphericity (from inertia tensor eigenvalues)
    centered = coords_frame - centroid
    inertia = (centered.T @ centered) / n_res
    eigvals = np.sort(np.linalg.eigvalsh(inertia))[::-1]
    asphericity = eigvals[0] - 0.5 * (eigvals[1] + eigvals[2])

    features.extend([
        rg,
        max_dist,
        asphericity,
        upper_tri.mean(),
        upper_tri.std(),
        # End-to-end distance (N-term to C-term CA)
        np.linalg.norm(coords_frame[0] - coords_frame[-1]),
    ])

    # ── 7. Displacement from previous frame (3 features) ──
    if coords_prev_frame is not None:
        displacement = np.linalg.norm(coords_frame - coords_prev_frame, axis=1)
        features.extend([
            displacement.mean(),
            displacement.max(),
            displacement.std(),
        ])
    else:
        features.extend([0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)


# Feature names for documentation
FEATURE_NAMES = [
    # RMSF stats (7)
    'rmsf_mean', 'rmsf_std', 'rmsf_max', 'rmsf_min', 'rmsf_median',
    'rmsf_skew', 'frac_flexible',
    # RMSF spatial (5)
    'rmsf_seg1', 'rmsf_seg2', 'rmsf_seg3', 'rmsf_seg4', 'rmsf_seg5',
    # Hinge (3)
    'n_hinges', 'rmsf_gradient_mean', 'rmsf_gradient_max',
    # Contact stats (6)
    'n_contacts', 'contact_density', 'contact_order', 'contact_order_std',
    'frac_long_range', 'n_residues',
    # Contact evolution (5)
    'contacts_gained', 'contacts_lost', 'contact_turnover', 'contacts_stable',
    'ref_contact_deviation',
    # Shape (6)
    'radius_of_gyration', 'max_extent', 'asphericity',
    'avg_pairwise_dist', 'std_pairwise_dist', 'end_to_end_dist',
    # Displacement (3)
    'displacement_mean', 'displacement_max', 'displacement_std',
]

N_FEATURES = len(FEATURE_NAMES)  # should be 35


# ══════════════════════════════════════════════════════════════════════════════
# Process one trajectory
# ══════════════════════════════════════════════════════════════════════════════

def embed_trajectory(coord_file, cutoff=10.0, rmsf_window=50):
    """
    Compute structural features for all frames of a trajectory.

    Args:
        coord_file: path to (n_frames, n_residues, 3) NPY
        cutoff: contact distance cutoff
        rmsf_window: sliding window size for RMSF

    Returns:
        (n_frames, N_FEATURES) array
    """
    coords = np.load(coord_file)  # (n_frames, n_res, 3)
    n_frames, n_res, _ = coords.shape

    # Precompute RMSF for all frames
    rmsf_all = compute_rmsf_windowed(coords, window=rmsf_window)

    # Precompute contact maps (memory-intensive for large proteins)
    # For 300 residues × 2500 frames: ~225 MB as bool, manageable
    if n_res <= 500 and n_frames <= 5000:
        contacts_all = compute_contact_maps(coords, cutoff=cutoff)
        precomputed_contacts = True
    else:
        # Too large, compute per-frame
        precomputed_contacts = False
        contacts_all = None

    # Reference contact map (first frame)
    if precomputed_contacts:
        contact_ref = contacts_all[0]
    else:
        dist = squareform(pdist(coords[0]))
        contact_ref = dist < cutoff
        np.fill_diagonal(contact_ref, False)
        for j in range(n_res - 1):
            contact_ref[j, j+1] = False
            contact_ref[j+1, j] = False

    embeddings = []

    for i in range(n_frames):
        if precomputed_contacts:
            contact_current = contacts_all[i]
            contact_prev = contacts_all[i-1] if i > 0 else None
        else:
            dist = squareform(pdist(coords[i]))
            contact_current = dist < cutoff
            np.fill_diagonal(contact_current, False)
            for j in range(n_res - 1):
                contact_current[j, j+1] = False
                contact_current[j+1, j] = False
            if i > 0:
                dist_prev = squareform(pdist(coords[i-1]))
                contact_prev = dist_prev < cutoff
                np.fill_diagonal(contact_prev, False)
                for j in range(n_res - 1):
                    contact_prev[j, j+1] = False
                    contact_prev[j+1, j] = False
            else:
                contact_prev = None

        feats = frame_features(
            coords_frame=coords[i],
            coords_prev_frame=coords[i-1] if i > 0 else None,
            rmsf_frame=rmsf_all[i],
            contact_current=contact_current,
            contact_prev=contact_prev,
            contact_ref=contact_ref,
            cutoff=cutoff,
        )
        embeddings.append(feats)

    return np.array(embeddings, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Batch processing (parallel)
# ══════════════════════════════════════════════════════════════════════════════

def _process_one(args):
    """Worker for parallel processing."""
    file_idx, total, coord_file, out_file, cutoff, rmsf_window = args

    coord_file = Path(coord_file)
    out_file = Path(out_file)

    if out_file.exists():
        return ('skip', coord_file.stem)

    try:
        t0 = time.time()
        embeddings = embed_trajectory(
            str(coord_file), cutoff=cutoff, rmsf_window=rmsf_window
        )
        np.save(out_file, embeddings)
        elapsed = time.time() - t0
        print(f"[{file_idx+1}/{total}] {coord_file.stem}: "
              f"{embeddings.shape} ({elapsed:.1f}s)")
        return ('success', coord_file.stem)
    except Exception as e:
        print(f"[{file_idx+1}/{total}] {coord_file.stem}: ERROR - {e}")
        import traceback
        traceback.print_exc()
        return ('fail', coord_file.stem)


def batch_process(coord_dir, output_dir, cutoff=10.0, rmsf_window=50,
                   n_workers=None):
    """Process all CA coordinate files in parallel."""
    from multiprocessing import Pool, cpu_count

    if n_workers is None:
        n_workers = min(cpu_count(), 8)

    coord_dir = Path(coord_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(coord_dir.glob("*.npy"))
    print(f"Found {len(npy_files)} coordinate files")
    print(f"Features per frame: {N_FEATURES}")
    print(f"Contact cutoff: {cutoff}Å, RMSF window: {rmsf_window} frames")
    print(f"Workers: {n_workers}\n")

    tasks = []
    for i, f in enumerate(npy_files):
        out_file = output_dir / f.name
        tasks.append((i, len(npy_files), str(f), str(out_file),
                       cutoff, rmsf_window))

    t0 = time.time()
    if n_workers == 1:
        results = [_process_one(t) for t in tasks]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(_process_one, tasks)

    counts = {'success': 0, 'fail': 0, 'skip': 0}
    for status, _ in results:
        counts[status] += 1

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Complete in {elapsed/60:.1f} minutes")
    print(f"  Success: {counts['success']}, Skip: {counts['skip']}, "
          f"Fail: {counts['fail']}")
    print(f"Output: {output_dir}")


def timing_test(coord_file, cutoff=10.0, rmsf_window=50):
    """Time one trajectory to estimate batch duration."""
    coords = np.load(coord_file)
    n_frames = coords.shape[0]
    n_res = coords.shape[1]
    print(f"Timing test: {coord_file}")
    print(f"  Shape: {coords.shape}")
    print(f"  Features: {N_FEATURES}")

    # Test on subset
    n_test = min(50, n_frames)
    test_coords = coords[:n_test]

    t0 = time.time()

    # RMSF
    rmsf = compute_rmsf_windowed(test_coords, window=min(rmsf_window, n_test))

    # Contacts
    contacts = compute_contact_maps(test_coords, cutoff=cutoff)

    # Features
    for i in range(n_test):
        feats = frame_features(
            test_coords[i],
            test_coords[i-1] if i > 0 else None,
            rmsf[i],
            contacts[i],
            contacts[i-1] if i > 0 else None,
            contacts[0],
            cutoff=cutoff,
        )

    elapsed = time.time() - t0
    per_frame = elapsed / n_test

    print(f"\n  Time: {per_frame*1000:.1f}ms/frame")
    print(f"  Per trajectory ({n_frames} frames): {per_frame * n_frames:.0f}s")
    print(f"  All 242 trajectories (1 worker): "
          f"{per_frame * n_frames * 242 / 3600:.1f} hours")
    print(f"  All 242 trajectories (8 workers): "
          f"{per_frame * n_frames * 242 / 3600 / 8:.1f} hours")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compute structural features from CA coordinates")

    subparsers = parser.add_subparsers(dest='command')

    # batch
    p_batch = subparsers.add_parser('batch',
        help='Process all trajectories')
    p_batch.add_argument('--coord_dir', required=True)
    p_batch.add_argument('--output_dir', required=True)
    p_batch.add_argument('--cutoff', type=float, default=10.0)
    p_batch.add_argument('--rmsf_window', type=int, default=50)
    p_batch.add_argument('--n_workers', type=int, default=None)

    # timing
    p_time = subparsers.add_parser('timing_test',
        help='Time one file')
    p_time.add_argument('--coord_file', required=True)
    p_time.add_argument('--cutoff', type=float, default=10.0)
    p_time.add_argument('--rmsf_window', type=int, default=50)

    # single
    p_single = subparsers.add_parser('single',
        help='Process one trajectory')
    p_single.add_argument('--coord_file', required=True)
    p_single.add_argument('--output', required=True)
    p_single.add_argument('--cutoff', type=float, default=10.0)
    p_single.add_argument('--rmsf_window', type=int, default=50)

    args = parser.parse_args()

    if args.command == 'batch':
        batch_process(args.coord_dir, args.output_dir,
                       cutoff=args.cutoff, rmsf_window=args.rmsf_window,
                       n_workers=args.n_workers)
    elif args.command == 'timing_test':
        timing_test(args.coord_file, cutoff=args.cutoff,
                     rmsf_window=args.rmsf_window)
    elif args.command == 'single':
        emb = embed_trajectory(args.coord_file, cutoff=args.cutoff,
                                rmsf_window=args.rmsf_window)
        np.save(args.output, emb)
        print(f"Saved: {args.output} {emb.shape}")
    else:
        parser.print_help()
        print(f"\nFeature names ({N_FEATURES} total):")
        for i, name in enumerate(FEATURE_NAMES):
            print(f"  {i:2d}. {name}")


if __name__ == "__main__":
    main()