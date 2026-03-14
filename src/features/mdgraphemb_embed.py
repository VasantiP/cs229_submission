"""
MDGraphEmb → TCN Pipeline Wrapper
===================================

Uses MDGraphEmb's ProteinDataLoader to convert MD trajectories into
contact graphs, then computes Node2Vec embeddings per frame, mean-pooled
across residues to produce a fixed-size time series for the TCN.

Output: NPY file with shape (n_frames, embedding_dim) — directly
compatible with the existing TCN training pipeline.

Usage:
    # Single trajectory
    python mdgraphemb_embed.py \
        --topology data/raw/receptor.pdb \
        --trajectory data/raw/receptor.xtc \
        --output data/processed_v4/features_50pct/graph/receptor_sim100_rep1.npy \
        --dimensions 32 \
        --frac 0.5

    # Batch mode
    python mdgraphemb_embed.py \
        --batch_csv splits/metadata_gpcr.csv \
        --traj_dir data/raw \
        --output_dir data/processed_v4/features_50pct/graph \
        --dimensions 32 \
        --frac 0.5

    # From pre-extracted CA coordinates (no MDAnalysis needed)
    python mdgraphemb_embed.py \
        --coord_dir data/ca_coords_50pct \
        --output_dir data/processed_v4/features_50pct/graph \
        --dimensions 32

Dependencies:
    pip install MDAnalysis networkx node2vec numpy pandas gensim

Notes:
    - MDGraphEmb repo must be cloned and on your Python path
    - Node2Vec trains fresh per frame (no pretrained weights)
    - Mean-pooling across residues gives cross-protein comparable embeddings
    - Default 32-dim is a good balance of richness vs TCN overfitting risk
"""

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import argparse
import time
import sys
import os
import warnings

# ══════════════════════════════════════════════════════════════════════════════
# Node2Vec embedding (self-contained, no MDGraphEmb dependency needed)
# ══════════════════════════════════════════════════════════════════════════════

try:
    from node2vec import Node2Vec
    HAS_NODE2VEC = True
except ImportError:
    HAS_NODE2VEC = False
    print("WARNING: node2vec not installed. Install: pip install node2vec")


def frame_to_contact_graph(ca_positions, cutoff=10.0):
    """
    Convert CA positions to a contact graph (same approach as MDGraphEmb).

    Each residue is a node. Edges connect residue pairs whose CA atoms
    are within the cutoff distance, excluding sequential neighbors (i, i±1).
    Edge weights are the CA-CA distances.
    """
    from scipy.spatial.distance import pdist, squareform

    n_res = len(ca_positions)
    dist_matrix = squareform(pdist(ca_positions))

    G = nx.Graph()
    G.add_nodes_from(range(n_res))

    for i in range(n_res):
        for j in range(i + 2, n_res):  # skip i, i+1 backbone neighbors
            if dist_matrix[i, j] < cutoff:
                G.add_edge(i, j, weight=dist_matrix[i, j])

    return G


def node2vec_embed_graph(G, dimensions=32, walk_length=80, num_walks=10,
                          window=10, min_count=1, batch_words=4, seed=42):
    """
    Compute Node2Vec embeddings for a graph and mean-pool across nodes.

    Returns: (dimensions,) array — one vector per frame.
    """
    if G.number_of_edges() == 0:
        return np.zeros(dimensions, dtype=np.float32)

    # Node2Vec needs connected graph
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n2v = Node2Vec(
            G, dimensions=dimensions, walk_length=walk_length,
            num_walks=num_walks, seed=seed, quiet=True
        )
        model = n2v.fit(
            window=window, min_count=min_count, batch_words=batch_words
        )

    # Mean pool node embeddings → graph-level embedding
    node_embeddings = np.array([model.wv[str(n)] for n in G.nodes()])
    graph_embedding = node_embeddings.mean(axis=0).astype(np.float32)

    return graph_embedding


# ══════════════════════════════════════════════════════════════════════════════
# Also compute handcrafted graph features (fast fallback / complement)
# ══════════════════════════════════════════════════════════════════════════════

def handcrafted_graph_features(G):
    """
    Fast graph statistics (no training). Returns ~20-dim vector.
    Can be concatenated with Node2Vec for richer representation.
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes == 0:
        return np.zeros(20, dtype=np.float32)

    degrees = np.array([d for _, d in G.degree()])
    density = nx.density(G)
    n_components = nx.number_connected_components(G)
    avg_clustering = nx.average_clustering(G)
    transitivity = nx.transitivity(G)

    try:
        assortativity = nx.degree_assortativity_coefficient(G)
    except (ValueError, nx.NetworkXError):
        assortativity = 0.0

    # Edge weight stats
    if n_edges > 0:
        weights = np.array([G[u][v].get('weight', 1.0) for u, v in G.edges()])
        w_mean, w_std, w_min = weights.mean(), weights.std(), weights.min()
    else:
        w_mean = w_std = w_min = 0.0

    # Centrality on largest CC
    if n_components > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc)
        closeness = np.array(list(nx.closeness_centrality(subG).values()))
        close_mean, close_std = closeness.mean(), closeness.std()
        lcc_frac = len(largest_cc) / n_nodes
    else:
        close_mean = close_std = lcc_frac = 0.0

    features = np.array([
        degrees.mean(), degrees.std(), degrees.max(), degrees.min(),
        density, float(n_components), avg_clustering, transitivity,
        assortativity, lcc_frac,
        close_mean, close_std,
        w_mean, w_std, w_min,
        np.percentile(degrees, 25), np.percentile(degrees, 50),
        np.percentile(degrees, 75), np.percentile(degrees, 90),
        float(n_edges),
    ], dtype=np.float32)

    return features


# ══════════════════════════════════════════════════════════════════════════════
# Process trajectory: MDAnalysis path
# ══════════════════════════════════════════════════════════════════════════════

def embed_trajectory_mda(topology, trajectory, frac=0.5, stride=1,
                          cutoff=10.0, dimensions=32, mode='node2vec',
                          ca_selection='name CA', verbose=True):
    """
    Load trajectory via MDAnalysis, build contact graphs, embed each frame.

    Args:
        mode: 'node2vec', 'handcrafted', or 'both' (concatenated)

    Returns:
        (n_frames, n_features) array
    """
    import MDAnalysis as mda

    u = mda.Universe(topology, trajectory)
    n_total = len(u.trajectory)
    n_use = int(n_total * frac)
    ca_atoms = u.select_atoms(ca_selection)

    if verbose:
        print(f"  {n_total} total frames, using {n_use} ({frac*100:.0f}%)")
        print(f"  {len(ca_atoms)} CA atoms, cutoff={cutoff}Å, "
              f"mode={mode}, dim={dimensions}")

    frame_indices = range(0, n_use, stride)
    embeddings = []
    t0 = time.time()

    for i, fi in enumerate(frame_indices):
        u.trajectory[fi]
        ca_pos = ca_atoms.positions
        G = frame_to_contact_graph(ca_pos, cutoff=cutoff)

        if mode == 'node2vec':
            emb = node2vec_embed_graph(G, dimensions=dimensions)
        elif mode == 'handcrafted':
            emb = handcrafted_graph_features(G)
        elif mode == 'both':
            n2v = node2vec_embed_graph(G, dimensions=dimensions)
            hc = handcrafted_graph_features(G)
            emb = np.concatenate([n2v, hc])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        embeddings.append(emb)

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(list(frame_indices)) - i - 1) / max(rate, 0.01)
            print(f"    Frame {i+1}/{len(list(frame_indices))} "
                  f"({rate:.1f} f/s, ~{eta:.0f}s left)")

    embeddings = np.array(embeddings, dtype=np.float32)

    if verbose:
        print(f"  Done: {embeddings.shape} in {time.time()-t0:.1f}s")

    return embeddings


# ══════════════════════════════════════════════════════════════════════════════
# Process trajectory: MDGraphEmb path (uses their ProteinDataLoader)
# ══════════════════════════════════════════════════════════════════════════════

def embed_trajectory_mdgraphemb(topology, trajectory, dimensions=32,
                                 mode='node2vec', verbose=True):
    """
    Use MDGraphEmb's ProteinDataLoader to build graphs, then embed.

    Note: MDGraphEmb loads ALL frames (no frac support). If you need
    partial trajectories, use embed_trajectory_mda() instead or
    pre-truncate your trajectory file.
    """
    try:
        from config import Config
        from protein_data_loader import ProteinDataLoader
    except ImportError:
        raise ImportError(
            "MDGraphEmb not found. Clone it and add to sys.path:\n"
            "  git clone https://github.com/FerdoosHN/MDGraphEMB.git\n"
            "  sys.path.append('MDGraphEMB')"
        )

    config = Config(trajectory, topology, "", "", "", "")
    config.DIMENSIONS = dimensions
    config.NUM_WALKS = 10
    config.WALK_LENGTH = 80

    if verbose:
        print(f"  Loading graphs via MDGraphEmb...")

    loader = ProteinDataLoader(config)
    graphs = loader.get_graphs()

    if verbose:
        print(f"  {len(graphs)} frames loaded, embedding...")

    embeddings = []
    t0 = time.time()

    for i, G in enumerate(graphs):
        if mode == 'node2vec':
            emb = node2vec_embed_graph(
                G, dimensions=dimensions,
                walk_length=config.WALK_LENGTH,
                num_walks=config.NUM_WALKS
            )
        elif mode == 'handcrafted':
            emb = handcrafted_graph_features(G)
        elif mode == 'both':
            n2v = node2vec_embed_graph(
                G, dimensions=dimensions,
                walk_length=config.WALK_LENGTH,
                num_walks=config.NUM_WALKS
            )
            hc = handcrafted_graph_features(G)
            emb = np.concatenate([n2v, hc])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        embeddings.append(emb)

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"    Frame {i+1}/{len(graphs)} ({rate:.1f} f/s)")

    embeddings = np.array(embeddings, dtype=np.float32)

    if verbose:
        print(f"  Done: {embeddings.shape} in {time.time()-t0:.1f}s")

    return embeddings


# ══════════════════════════════════════════════════════════════════════════════
# Process from pre-extracted CA coordinate NPY files
# ══════════════════════════════════════════════════════════════════════════════

def embed_from_coords_npy(coord_npy, cutoff=10.0, dimensions=32,
                           mode='node2vec', verbose=True):
    """
    If you already have CA coordinates as NPY (n_frames, n_residues, 3),
    skip MDAnalysis entirely.
    """
    coords = np.load(coord_npy)
    if coords.ndim != 3:
        raise ValueError(f"Expected (frames, residues, 3), got {coords.shape}")

    n_frames = coords.shape[0]
    embeddings = []
    t0 = time.time()

    for i in range(n_frames):
        G = frame_to_contact_graph(coords[i], cutoff=cutoff)

        if mode == 'node2vec':
            emb = node2vec_embed_graph(G, dimensions=dimensions)
        elif mode == 'handcrafted':
            emb = handcrafted_graph_features(G)
        elif mode == 'both':
            n2v = node2vec_embed_graph(G, dimensions=dimensions)
            hc = handcrafted_graph_features(G)
            emb = np.concatenate([n2v, hc])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        embeddings.append(emb)

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"    Frame {i+1}/{n_frames} ({rate:.1f} f/s)")

    embeddings = np.array(embeddings, dtype=np.float32)

    if verbose:
        print(f"  Result: {embeddings.shape} in {time.time()-t0:.1f}s")

    return embeddings


# ══════════════════════════════════════════════════════════════════════════════
# Batch processing
# ══════════════════════════════════════════════════════════════════════════════

def batch_embed(metadata_csv, traj_dir, output_dir, frac=0.5, stride=1,
                cutoff=10.0, dimensions=32, mode='node2vec',
                use_mdgraphemb=False, verbose=True):
    """
    Batch process all trajectories in metadata CSV.

    IMPORTANT: You must customize find_traj_files() below to match your
    directory structure for raw trajectory/topology file locations.
    """
    df = pd.read_csv(metadata_csv)
    if 'fold' in df.columns:
        df = df.drop_duplicates(subset=['chunk_file']).reset_index(drop=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = Path(traj_dir)

    n_success = 0
    n_fail = 0
    n_skip = 0

    for idx, row in df.iterrows():
        traj_id = row.get('traj_id', '')
        output_file = output_dir / f"{traj_id}.npy"

        if output_file.exists():
            n_skip += 1
            continue

        # ── CUSTOMIZE THIS: find topology + trajectory files ──
        receptor = str(row.get('receptor', '')).replace('~', '/')
        simID = str(row.get('simID', ''))
        rep = str(row.get('rep', ''))

        topology, trajectory = find_traj_files(traj_dir, receptor, simID, rep)

        if topology is None or trajectory is None:
            if verbose:
                print(f"[{idx+1}/{len(df)}] SKIP (files not found): {traj_id}")
            n_fail += 1
            continue

        print(f"\n[{idx+1}/{len(df)}] {traj_id}")

        try:
            if use_mdgraphemb:
                embeddings = embed_trajectory_mdgraphemb(
                    str(topology), str(trajectory),
                    dimensions=dimensions, mode=mode, verbose=verbose
                )
            else:
                embeddings = embed_trajectory_mda(
                    str(topology), str(trajectory),
                    frac=frac, stride=stride, cutoff=cutoff,
                    dimensions=dimensions, mode=mode, verbose=verbose
                )

            np.save(output_file, embeddings)
            print(f"  Saved: {output_file} {embeddings.shape}")
            n_success += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            n_fail += 1

    print(f"\n{'='*60}")
    print(f"Batch: {n_success} success, {n_fail} failed, {n_skip} skipped")
    print(f"Output: {output_dir}")


def batch_embed_from_coords(coord_dir, output_dir, cutoff=10.0,
                              dimensions=32, mode='node2vec', verbose=True):
    """
    Batch process pre-extracted CA coordinate NPY files.
    Files should have shape (n_frames, n_residues, 3).
    """
    coord_dir = Path(coord_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(coord_dir.glob("*_coords.npy"))
    if not npy_files:
        npy_files = sorted(coord_dir.glob("*.npy"))

    print(f"Found {len(npy_files)} coordinate files")

    for i, f in enumerate(npy_files):
        stem = f.stem.replace('_coords', '')
        output_file = output_dir / f"{stem}.npy"

        if output_file.exists():
            continue

        print(f"\n[{i+1}/{len(npy_files)}] {stem}")

        try:
            embeddings = embed_from_coords_npy(
                f, cutoff=cutoff, dimensions=dimensions,
                mode=mode, verbose=verbose
            )
            np.save(output_file, embeddings)
        except Exception as e:
            print(f"  ERROR: {e}")


def find_traj_files(traj_dir, receptor, simID, rep):
    """
    Find topology and trajectory files for a given simulation.

    *** CUSTOMIZE THIS FOR YOUR DIRECTORY STRUCTURE ***

    Common GPCRmd patterns:
        data/raw/{receptor_pdb}/topology.pdb
        data/raw/{receptor_pdb}/sim{ID}_rep{rep}.xtc

    Returns: (topology_path, trajectory_path) or (None, None)
    """
    traj_dir = Path(traj_dir)

    # Try various directory patterns
    search_dirs = [
        traj_dir,
        traj_dir / receptor.split('/')[0] if '/' in receptor else traj_dir,
        traj_dir / f"sim{simID}",
    ]

    for d in search_dirs:
        if not d.exists():
            continue

        # Find topology
        top = None
        for pattern in [f"*{simID}*.pdb", f"*{simID}*.gro", "*.pdb", "*.gro"]:
            matches = list(d.glob(pattern))
            if matches:
                top = matches[0]
                break

        # Find trajectory
        traj = None
        for pattern in [f"*sim{simID}*rep{rep}*.xtc",
                        f"*{simID}*rep{rep}*.xtc",
                        f"*{simID}*.xtc",
                        f"*sim{simID}*rep{rep}*.dcd"]:
            matches = list(d.glob(pattern))
            if matches:
                traj = matches[0]
                break

        if top and traj:
            return top, traj

    return None, None


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Graph embedding pipeline for MD trajectories → TCN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  node2vec     - Node2Vec random walks, mean-pooled (default, 32-dim)
  handcrafted  - Graph statistics only (20-dim, very fast)
  both         - Concatenation of node2vec + handcrafted (52-dim)

Examples:
  # Single trajectory
  python mdgraphemb_embed.py \\
      --topology receptor.pdb --trajectory receptor.xtc \\
      --output graph_emb.npy --mode node2vec --frac 0.5

  # Batch from raw trajectories
  python mdgraphemb_embed.py \\
      --batch_csv splits/metadata_gpcr.csv \\
      --traj_dir data/raw \\
      --output_dir data/processed_v4/features_50pct/graph

  # Batch from pre-extracted CA coordinates
  python mdgraphemb_embed.py \\
      --coord_dir data/ca_coords_50pct \\
      --output_dir data/processed_v4/features_50pct/graph

  # Use handcrafted features (very fast, no node2vec needed)
  python mdgraphemb_embed.py \\
      --topology receptor.pdb --trajectory receptor.xtc \\
      --output graph_emb.npy --mode handcrafted
        """)

    # Input modes
    parser.add_argument('--topology', type=str)
    parser.add_argument('--trajectory', type=str)
    parser.add_argument('--coord_npy', type=str,
                        help='Pre-extracted CA coords (n_frames, n_res, 3)')
    parser.add_argument('--batch_csv', type=str)
    parser.add_argument('--traj_dir', type=str)
    parser.add_argument('--coord_dir', type=str)

    # Output
    parser.add_argument('--output', type=str)
    parser.add_argument('--output_dir', type=str)

    # Parameters
    parser.add_argument('--mode', type=str, default='node2vec',
                        choices=['node2vec', 'handcrafted', 'both'])
    parser.add_argument('--dimensions', type=int, default=32,
                        help='Node2Vec embedding dim (default: 32)')
    parser.add_argument('--cutoff', type=float, default=10.0,
                        help='Contact distance cutoff in Angstroms')
    parser.add_argument('--frac', type=float, default=0.5,
                        help='Fraction of trajectory to use')
    parser.add_argument('--stride', type=int, default=1,
                        help='Frame stride')
    parser.add_argument('--use_mdgraphemb', action='store_true',
                        help='Use MDGraphEmb ProteinDataLoader (loads all frames)')
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()
    verbose = not args.quiet

    if args.coord_dir:
        batch_embed_from_coords(
            args.coord_dir, args.output_dir or 'graph_embeddings',
            cutoff=args.cutoff, dimensions=args.dimensions,
            mode=args.mode, verbose=verbose
        )

    elif args.batch_csv:
        batch_embed(
            args.batch_csv, args.traj_dir, args.output_dir or 'graph_embeddings',
            frac=args.frac, stride=args.stride, cutoff=args.cutoff,
            dimensions=args.dimensions, mode=args.mode,
            use_mdgraphemb=args.use_mdgraphemb, verbose=verbose
        )

    elif args.coord_npy:
        embeddings = embed_from_coords_npy(
            args.coord_npy, cutoff=args.cutoff, dimensions=args.dimensions,
            mode=args.mode, verbose=verbose
        )
        out = args.output or 'graph_embedding.npy'
        np.save(out, embeddings)
        print(f"Saved: {out} {embeddings.shape}")

    elif args.topology and args.trajectory:
        if args.use_mdgraphemb:
            embeddings = embed_trajectory_mdgraphemb(
                args.topology, args.trajectory,
                dimensions=args.dimensions, mode=args.mode, verbose=verbose
            )
        else:
            embeddings = embed_trajectory_mda(
                args.topology, args.trajectory,
                frac=args.frac, stride=args.stride, cutoff=args.cutoff,
                dimensions=args.dimensions, mode=args.mode, verbose=verbose
            )
        out = args.output or 'graph_embedding.npy'
        np.save(out, embeddings)
        print(f"Saved: {out} {embeddings.shape}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()