"""
Extract CA coordinates and compute graph embeddings for all GPCRmd trajectories.

Uses the deduped CSV which has exact traj_file and top_file columns.
Extracts FULL trajectories — slice to desired fraction at training time.

Usage:
    # Step 1: Extract CA coordinates (fast, ~1 min/trajectory)
    python extract_and_embed.py extract_coords \
        --metadata data/processed/data_processed_v3_base_dataset_deduped.csv \
        --traj_dir ../gcs_mount/data/trajectories \
        --top_dir ../gcs_mount/data/topologies \
        --output_dir data/processed_v4/ca_coords \
        --n_workers 4

    # Step 2: Compute graph embeddings (handcrafted: ~2 min/traj, node2vec: ~30 min/traj)
    python extract_and_embed.py embed \
        --coord_dir data/processed_v4/ca_coords \
        --output_dir data/processed_v4/graph_embeddings \
        --mode fast \
        --n_workers 8

    # Step 3 (at training time): just load and slice
    #   emb = np.load('graph_embeddings/receptor_rep1_85.npy')  # (2500, 20)
    #   early = emb[:int(len(emb) * 0.5)]  # first 50%

Dependencies:
    pip install MDAnalysis networkx numpy pandas scipy
    # For node2vec mode: pip install node2vec
"""

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import argparse
import time
import warnings
from scipy.spatial.distance import pdist, squareform


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Extract CA coordinates
# ══════════════════════════════════════════════════════════════════════════════

def _extract_one_coord(args):
    """Worker function for parallel CA coordinate extraction."""
    import MDAnalysis as mda

    idx, total, receptor, rep, simID, traj_path, top_path, output_file, ca_selection = args

    out_name = Path(output_file).stem

    if Path(output_file).exists():
        return ('skip', out_name)

    if not Path(traj_path).exists():
        return ('missing_traj', out_name)
    if not Path(top_path).exists():
        return ('missing_top', out_name)

    try:
        t0 = time.time()
        u = mda.Universe(str(top_path), str(traj_path))
        ca = u.select_atoms(ca_selection)
        n_frames = len(u.trajectory)
        n_res = len(ca)

        coords = np.zeros((n_frames, n_res, 3), dtype=np.float32)
        for i, ts in enumerate(u.trajectory):
            coords[i] = ca.positions

        np.save(output_file, coords)
        elapsed = time.time() - t0
        print(f"[{idx+1}/{total}] {out_name}: {coords.shape} ({elapsed:.1f}s)")
        return ('success', out_name)

    except Exception as e:
        print(f"[{idx+1}/{total}] {out_name}: ERROR - {e}")
        return ('fail', out_name)


def extract_all_coords(metadata_csv, traj_dir, top_dir, output_dir,
                        ca_selection='name CA', n_workers=None):
    """
    Extract CA coordinates for all trajectories in the metadata CSV.
    Saves full trajectory as (n_frames, n_residues, 3) NPY.

    Uses multiprocessing for parallel extraction.
    Set n_workers=1 for sequential (easier debugging).
    """
    from multiprocessing import Pool, cpu_count

    if n_workers is None:
        n_workers = min(cpu_count(), 8)  # cap at 8 to avoid I/O bottleneck

    df = pd.read_csv(metadata_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = Path(traj_dir)
    top_dir = Path(top_dir)

    # Build task list
    tasks = []
    for idx, row in df.iterrows():
        receptor = row['receptor']
        rep = int(row['rep'])
        simID = int(row['simID'])
        traj_file = row['traj_file']
        top_file = row['top_file']

        out_name = f"{receptor.replace('~', '_')}_rep{rep}_{simID}"
        output_file = str(output_dir / f"{out_name}.npy")
        traj_path = str(traj_dir / traj_file)
        top_path = str(top_dir / top_file)

        tasks.append((idx, len(df), receptor, rep, simID,
                       traj_path, top_path, output_file, ca_selection))

    print(f"Extracting CA coords for {len(tasks)} trajectories "
          f"using {n_workers} workers")
    print(f"Output: {output_dir}\n")

    t0 = time.time()

    if n_workers == 1:
        results = [_extract_one_coord(t) for t in tasks]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(_extract_one_coord, tasks)

    # Summarize
    counts = {'success': 0, 'fail': 0, 'skip': 0,
              'missing_traj': 0, 'missing_top': 0}
    for status, name in results:
        counts[status] = counts.get(status, 0) + 1

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Extract complete in {elapsed/60:.1f} minutes:")
    print(f"  Success: {counts['success']}")
    print(f"  Skipped (exists): {counts['skip']}")
    print(f"  Missing traj: {counts['missing_traj']}")
    print(f"  Missing top: {counts['missing_top']}")
    print(f"  Failed: {counts['fail']}")
    print(f"Output: {output_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Contact graph construction
# ══════════════════════════════════════════════════════════════════════════════

def frame_to_contact_graph(ca_positions, cutoff=10.0):
    """
    CA positions → contact graph.
    Nodes = residues, edges = pairs within cutoff (skip i,i+1).
    """
    n_res = len(ca_positions)
    dist_matrix = squareform(pdist(ca_positions))

    G = nx.Graph()
    G.add_nodes_from(range(n_res))

    for i in range(n_res):
        for j in range(i + 2, n_res):
            if dist_matrix[i, j] < cutoff:
                G.add_edge(i, j, weight=dist_matrix[i, j])

    return G


# ══════════════════════════════════════════════════════════════════════════════
# Step 2a: Handcrafted graph features
# ══════════════════════════════════════════════════════════════════════════════

def handcrafted_features(G):
    """Graph statistics → 20-dim vector per frame. FAST version using numpy."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes == 0:
        return np.zeros(20, dtype=np.float32)

    degrees = np.array([d for _, d in G.degree()])

    # Fast: degree-based stats (O(n))
    density = nx.density(G)

    # Fast: edge weight stats from edge list (O(e))
    if n_edges > 0:
        weights = np.array([G[u][v].get('weight', 1.0) for u, v in G.edges()])
        w_mean, w_std, w_min, w_max = (
            weights.mean(), weights.std(), weights.min(), weights.max()
        )
    else:
        w_mean = w_std = w_min = w_max = 0.0

    # Fast: number of connected components via NetworkX (optimized in C)
    n_components = nx.number_connected_components(G)

    # Fraction of nodes in largest connected component
    if n_components > 0:
        largest_cc_size = max(len(c) for c in nx.connected_components(G))
        lcc_frac = largest_cc_size / n_nodes
    else:
        lcc_frac = 0.0

    # Fast: transitivity (global clustering, implemented in C in NetworkX)
    transitivity = nx.transitivity(G)

    # Fast: degree assortativity
    try:
        assortativity = nx.degree_assortativity_coefficient(G)
    except (ValueError, nx.NetworkXError):
        assortativity = 0.0

    return np.array([
        degrees.mean(), degrees.std(), float(degrees.max()), float(degrees.min()),
        density, float(n_components), lcc_frac, transitivity, assortativity,
        w_mean, w_std, w_min, w_max,
        float(np.percentile(degrees, 10)),
        float(np.percentile(degrees, 25)),
        float(np.percentile(degrees, 50)),
        float(np.percentile(degrees, 75)),
        float(np.percentile(degrees, 90)),
        float(n_edges),
        float(n_edges) / max(n_nodes, 1),  # avg degree (redundant w/ mean but explicit)
    ], dtype=np.float32)


def fast_contact_features(ca_positions, cutoff=10.0):
    """
    FAST alternative: compute features directly from distance matrix
    without building a NetworkX graph at all.

    Returns 20-dim vector per frame. ~100x faster than graph-based.
    """
    n_res = len(ca_positions)
    dist_matrix = squareform(pdist(ca_positions))

    # Build binary contact matrix (skip i, i+1 backbone neighbors)
    contact = (dist_matrix < cutoff).astype(np.float32)
    np.fill_diagonal(contact, 0)
    for i in range(n_res - 1):
        contact[i, i+1] = 0
        contact[i+1, i] = 0

    # Degree = row sums of contact matrix
    degrees = contact.sum(axis=1)
    n_edges = int(degrees.sum() / 2)

    # Contact distance stats (only for pairs in contact)
    contact_distances = dist_matrix[np.triu(contact, k=2) > 0]
    if len(contact_distances) > 0:
        cd_mean = contact_distances.mean()
        cd_std = contact_distances.std()
        cd_min = contact_distances.min()
        cd_max = contact_distances.max()
    else:
        cd_mean = cd_std = cd_min = cd_max = 0.0

    # Full distance matrix stats (captures overall shape)
    upper_tri = dist_matrix[np.triu_indices(n_res, k=2)]
    dist_mean = upper_tri.mean()
    dist_std = upper_tri.std()

    # Density
    max_edges = (n_res * (n_res - 1)) / 2 - (n_res - 1)  # excluding i,i+1
    density = n_edges / max(max_edges, 1)

    # Approximate clustering: for each node, fraction of neighbor pairs
    # that are also connected (sampled for speed)
    n_sample = min(50, n_res)
    sample_idx = np.linspace(0, n_res-1, n_sample, dtype=int)
    clustering_samples = []
    for i in sample_idx:
        neighbors = np.where(contact[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            clustering_samples.append(0.0)
            continue
        # Count edges among neighbors
        neighbor_contacts = contact[np.ix_(neighbors, neighbors)]
        triangles = neighbor_contacts.sum() / 2
        possible = k * (k - 1) / 2
        clustering_samples.append(triangles / possible)
    avg_clustering = np.mean(clustering_samples)

    return np.array([
        degrees.mean(), degrees.std(), degrees.max(), degrees.min(),
        density, avg_clustering,
        cd_mean, cd_std, cd_min, cd_max,
        dist_mean, dist_std,
        float(np.percentile(degrees, 10)),
        float(np.percentile(degrees, 25)),
        float(np.percentile(degrees, 50)),
        float(np.percentile(degrees, 75)),
        float(np.percentile(degrees, 90)),
        float(np.percentile(degrees, 95)),
        float(n_edges),
        float(n_edges) / max(n_res, 1),
    ], dtype=np.float32)


# Feature names for reference
HANDCRAFTED_FEATURE_NAMES = [
    'degree_mean', 'degree_std', 'degree_max', 'degree_min',
    'density', 'n_components', 'lcc_fraction', 'transitivity',
    'assortativity',
    'edge_weight_mean', 'edge_weight_std', 'edge_weight_min', 'edge_weight_max',
    'degree_p10', 'degree_p25', 'degree_p50', 'degree_p75', 'degree_p90',
    'n_edges', 'avg_degree',
]

FAST_FEATURE_NAMES = [
    'degree_mean', 'degree_std', 'degree_max', 'degree_min',
    'density', 'avg_clustering',
    'contact_dist_mean', 'contact_dist_std', 'contact_dist_min', 'contact_dist_max',
    'all_dist_mean', 'all_dist_std',
    'degree_p10', 'degree_p25', 'degree_p50', 'degree_p75', 'degree_p90', 'degree_p95',
    'n_edges', 'avg_degree',
]


# ══════════════════════════════════════════════════════════════════════════════
# Step 2b: Node2Vec embedding (mean-pooled)
# ══════════════════════════════════════════════════════════════════════════════

def node2vec_features(G, dimensions=32, walk_length=80, num_walks=10,
                       window=10, seed=42):
    """Node2Vec → mean-pool → (dimensions,) vector per frame."""
    from node2vec import Node2Vec as N2V

    if G.number_of_edges() == 0:
        return np.zeros(dimensions, dtype=np.float32)

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n2v = N2V(G, dimensions=dimensions, walk_length=walk_length,
                   num_walks=num_walks, seed=seed, quiet=True)
        model = n2v.fit(window=window, min_count=1, batch_words=4)

    node_embs = np.array([model.wv[str(n)] for n in G.nodes()])
    return node_embs.mean(axis=0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Embed all trajectories
# ══════════════════════════════════════════════════════════════════════════════

def _embed_one_trajectory(args):
    """Worker function for parallel graph embedding."""
    file_idx, total, npy_file, out_file, mode, cutoff, dimensions = args

    npy_file = Path(npy_file)
    out_file = Path(out_file)

    if out_file.exists():
        return ('skip', npy_file.stem)

    try:
        coords = np.load(npy_file)
        n_frames = coords.shape[0]

        embeddings = []
        t0 = time.time()

        for i in range(n_frames):
            if mode == 'fast':
                # Skip graph construction entirely — pure numpy
                emb = fast_contact_features(coords[i], cutoff=cutoff)
            elif mode == 'handcrafted':
                G = frame_to_contact_graph(coords[i], cutoff=cutoff)
                emb = handcrafted_features(G)
            elif mode == 'node2vec':
                G = frame_to_contact_graph(coords[i], cutoff=cutoff)
                emb = node2vec_features(G, dimensions=dimensions)
            elif mode == 'both':
                G = frame_to_contact_graph(coords[i], cutoff=cutoff)
                hc = handcrafted_features(G)
                n2v = node2vec_features(G, dimensions=dimensions)
                emb = np.concatenate([hc, n2v])
            else:
                raise ValueError(f"Unknown mode: {mode}")

            embeddings.append(emb)

        embeddings = np.array(embeddings, dtype=np.float32)
        np.save(out_file, embeddings)

        elapsed = time.time() - t0
        print(f"[{file_idx+1}/{total}] {npy_file.stem}: "
              f"{embeddings.shape} ({elapsed:.1f}s)")
        return ('success', npy_file.stem)

    except Exception as e:
        print(f"[{file_idx+1}/{total}] {npy_file.stem}: ERROR - {e}")
        return ('fail', npy_file.stem)


def embed_all(coord_dir, output_dir, mode='handcrafted', cutoff=10.0,
              dimensions=32, n_workers=None):
    """
    Compute graph embeddings for all CA coordinate NPY files.

    Each input file: (n_frames, n_residues, 3)
    Each output file: (n_frames, n_features)

    Uses multiprocessing for parallel embedding.
    Set n_workers=1 for sequential (easier debugging).
    """
    from multiprocessing import Pool, cpu_count

    if n_workers is None:
        n_workers = min(cpu_count(), 8)

    coord_dir = Path(coord_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(coord_dir.glob("*.npy"))
    print(f"Found {len(npy_files)} coordinate files in {coord_dir}")
    print(f"Mode: {mode}, cutoff: {cutoff}Å, workers: {n_workers}", end="")
    if 'node2vec' in mode:
        print(f", dimensions: {dimensions}")
    else:
        print()

    # Build task list
    tasks = []
    for i, npy_file in enumerate(npy_files):
        out_file = output_dir / npy_file.name
        tasks.append((i, len(npy_files), str(npy_file), str(out_file),
                       mode, cutoff, dimensions))

    total_t0 = time.time()

    if n_workers == 1:
        results = [_embed_one_trajectory(t) for t in tasks]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(_embed_one_trajectory, tasks)

    counts = {'success': 0, 'fail': 0, 'skip': 0}
    for status, name in results:
        counts[status] = counts.get(status, 0) + 1

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"Embedding complete in {total_elapsed/60:.1f} minutes:")
    print(f"  Success: {counts['success']}")
    print(f"  Skipped: {counts['skip']}")
    print(f"  Failed: {counts['fail']}")
    print(f"Output: {output_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Training-time loader (slice to desired fraction)
# ══════════════════════════════════════════════════════════════════════════════

def load_graph_embedding(npy_path, frac=0.5):
    """
    Load graph embedding and slice to desired trajectory fraction.

    Args:
        npy_path: path to (n_frames, n_features) NPY
        frac: fraction of trajectory to use (0.0-1.0)

    Returns:
        (n_frames_early, n_features) array
    """
    emb = np.load(npy_path)
    n_use = int(len(emb) * frac)
    return emb[:n_use]


# ══════════════════════════════════════════════════════════════════════════════
# Quick timing test on one trajectory
# ══════════════════════════════════════════════════════════════════════════════

def timing_test(coord_file, cutoff=10.0):
    """
    Run timing test on one trajectory to estimate batch time.
    """
    coords = np.load(coord_file)
    n_frames = coords.shape[0]
    n_trajs = 242
    print(f"Timing test: {coord_file}")
    print(f"  Shape: {coords.shape}")

    # Test fast mode (pure numpy, no graph construction)
    n_test = min(100, n_frames)
    t0 = time.time()
    for i in range(n_test):
        emb = fast_contact_features(coords[i], cutoff=cutoff)
    fast_time = (time.time() - t0) / n_test
    print(f"\n  FAST (numpy): {fast_time*1000:.1f}ms/frame, {emb.shape[0]} features")
    print(f"    Per trajectory ({n_frames} frames): {fast_time * n_frames:.0f}s")
    print(f"    All {n_trajs} trajectories (1 worker): "
          f"{fast_time * n_frames * n_trajs / 3600:.1f} hours")
    print(f"    All {n_trajs} trajectories (8 workers): "
          f"{fast_time * n_frames * n_trajs / 3600 / 8:.1f} hours")

    # Test handcrafted (NetworkX graph-based)
    n_test = min(20, n_frames)
    t0 = time.time()
    for i in range(n_test):
        G = frame_to_contact_graph(coords[i], cutoff=cutoff)
        emb = handcrafted_features(G)
    hc_time = (time.time() - t0) / n_test
    print(f"\n  Handcrafted (NetworkX): {hc_time*1000:.1f}ms/frame, {emb.shape[0]} features")
    print(f"    Per trajectory ({n_frames} frames): {hc_time * n_frames:.0f}s")
    print(f"    All {n_trajs} trajectories (1 worker): "
          f"{hc_time * n_frames * n_trajs / 3600:.1f} hours")
    print(f"    All {n_trajs} trajectories (8 workers): "
          f"{hc_time * n_frames * n_trajs / 3600 / 8:.1f} hours")

    # Node2Vec
    try:
        n_test = min(3, n_frames)
        t0 = time.time()
        for i in range(n_test):
            G = frame_to_contact_graph(coords[i], cutoff=cutoff)
            emb = node2vec_features(G, dimensions=32)
        n2v_time = (time.time() - t0) / n_test
        print(f"\n  Node2Vec (32-dim): {n2v_time*1000:.1f}ms/frame, "
              f"{emb.shape[0]} features")
        print(f"    Per trajectory ({n_frames} frames): {n2v_time * n_frames:.0f}s")
        print(f"    All {n_trajs} trajectories (8 workers): "
              f"{n2v_time * n_frames * n_trajs / 3600 / 8:.1f} hours")
    except ImportError:
        print("\n  Node2Vec: not installed (pip install node2vec)")

    print(f"\n  → Recommendation: use --mode fast")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Extract CA coords and compute graph embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. extract_coords  → (n_frames, n_residues, 3) NPY per trajectory
  2. embed           → (n_frames, n_features) NPY per trajectory
  3. timing_test     → estimate batch processing time

At training time, slice to fraction:
  emb = np.load('embedding.npy')[:int(2500 * 0.5)]  # first 50%
        """)

    subparsers = parser.add_subparsers(dest='command')

    # extract_coords
    p_extract = subparsers.add_parser('extract_coords',
        help='Extract CA coordinates from raw trajectories')
    p_extract.add_argument('--metadata', required=True,
        help='Path to deduped CSV with traj_file and top_file columns')
    p_extract.add_argument('--traj_dir', required=True,
        help='Directory containing .xtc/.dcd trajectory files')
    p_extract.add_argument('--top_dir', required=True,
        help='Directory containing .psf/.pdb/.gro topology files')
    p_extract.add_argument('--output_dir', required=True,
        help='Output directory for CA coordinate NPY files')
    p_extract.add_argument('--n_workers', type=int, default=None,
        help='Number of parallel workers (default: min(cpu_count, 8))')

    # embed
    p_embed = subparsers.add_parser('embed',
        help='Compute graph embeddings from CA coordinates')
    p_embed.add_argument('--coord_dir', required=True,
        help='Directory with CA coordinate NPY files')
    p_embed.add_argument('--output_dir', required=True,
        help='Output directory for embedding NPY files')
    p_embed.add_argument('--mode', default='fast',
        choices=['fast', 'handcrafted', 'node2vec', 'both'],
        help='fast=pure numpy (recommended), handcrafted=NetworkX, node2vec=slow')
    p_embed.add_argument('--cutoff', type=float, default=10.0,
        help='Contact distance cutoff in Angstroms (default: 10.0)')
    p_embed.add_argument('--dimensions', type=int, default=32,
        help='Node2Vec embedding dim (default: 32)')
    p_embed.add_argument('--n_workers', type=int, default=None,
        help='Number of parallel workers (default: min(cpu_count, 8))')

    # timing_test
    p_time = subparsers.add_parser('timing_test',
        help='Run timing test on one coordinate file')
    p_time.add_argument('--coord_file', required=True,
        help='One CA coordinate NPY file to test')
    p_time.add_argument('--cutoff', type=float, default=10.0)

    args = parser.parse_args()

    if args.command == 'extract_coords':
        extract_all_coords(
            args.metadata, args.traj_dir, args.top_dir, args.output_dir,
            n_workers=args.n_workers
        )
    elif args.command == 'embed':
        embed_all(
            args.coord_dir, args.output_dir,
            mode=args.mode, cutoff=args.cutoff, dimensions=args.dimensions,
            n_workers=args.n_workers
        )
    elif args.command == 'timing_test':
        timing_test(args.coord_file, cutoff=args.cutoff)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()