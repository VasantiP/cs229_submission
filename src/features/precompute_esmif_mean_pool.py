"""
precompute_esmif_mean_pool.py

Loads per-residue ESM-IF embeddings (T, N, 512), mean-pools over residues
to produce (T, 512), slices to the requested trajectory fraction, and saves
with the naming convention expected by train_tcn.py:
    {receptor_underscores}_sim{simID}_rep{rep}.npy

Usage:
    python precompute_esmif_mean_pool.py \
        --metadata  ../../data/metadata/metadata.csv \
        --esmif_dir ../../data/processed_v4/esm_if_embeddings \
        --out_dir   ../../data/processed_v4/features_50pct/esm_if_mean_pooled \
        --pct       50
"""

import argparse
import csv
from itertools import count
import os
import numpy as np
from pathlib import Path


def receptor_to_esmif_filename(receptor: str, rep: str, sim_id: str) -> str:
    """
    Converts metadata receptor field to the ESM-IF .npy filename.
    e.g. '5-hydroxytryptamine_receptor_1B~4IAQ_A', rep=1, simID=85
      -> 'esmif_emb-5-hydroxytryptamine_receptor_1B_4IAQ_A_rep1_85.npy'
    """
    base = receptor.replace("~", "_")
    return f"esmif_emb-{base}_rep{rep}_{sim_id}.npy"


def receptor_to_output_filename(receptor: str, rep: str, sim_id: str) -> str:
    """
    Produces the scalar_file-style name expected by train_tcn.py.
    e.g. 'Adenosine_receptor_A1~5UEN_Z', rep=1, simID=165
      -> 'Adenosine_receptor_A1_5UEN_Z_sim165_rep1.npy'
    """
    base = receptor.replace("~", "_")
    return f"{base}_sim{sim_id}_rep{rep}.npy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata",  required=True,
                        help="Path to metadata.csv")
    parser.add_argument("--esmif_dir", required=True,
                        help="Directory containing raw ESM-IF .npy files "
                             "(T, N, 512)")
    parser.add_argument("--out_dir",   required=True,
                        help="Output directory for mean-pooled files")
    parser.add_argument("--pct",       type=int, default=50,
                        help="Trajectory fraction to slice (e.g. 50 = first 50%%)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.metadata) as f:
        rows = list(csv.DictReader(f))

    missing, written = [], 0

    count = 0
    total = len(rows)
    for row in rows:
        count += 1
        receptor = row["receptor"]
        rep      = row["rep"]
        sim_id   = row["simID"]
        n_frames = int(row["n_frames_total"])

        src_name = receptor_to_esmif_filename(receptor, rep, sim_id)
        src_path = os.path.join(args.esmif_dir, src_name)

        dst_name = receptor_to_output_filename(receptor, rep, sim_id)
        dst_path = os.path.join(args.out_dir, dst_name)

        print(f"Processing {count}/{total}: {src_name}...", end="")
        if not os.path.exists(src_path):
            print(f"  -- MISSING {src_name}")
            missing.append(src_name)
            continue

        if os.path.exists(dst_path):
            print(f"  -- SKIPPING {dst_name} (already exists)")
            continue

        # Load (T, N, 512)
        try:
            emb = np.load(src_path)          # float32
            assert emb.ndim == 3 and emb.shape[-1] == 512, \
                f"Unexpected shape {emb.shape} for {src_name}"
        except Exception as e:
            print(f"  -- ERROR loading {src_name}: {e}")
            missing.append(src_name)
            continue

        # Slice to requested fraction
        n_keep = max(1, int(n_frames * args.pct / 100))
        emb = emb[:n_keep]               # (n_keep, N, 512)

        # Mean-pool over residues -> (n_keep, 512)
        emb_pooled = emb.mean(axis=1)    # float32

        np.save(dst_path, emb_pooled)
        print(f"  -- WRITTEN {dst_name} with shape {emb_pooled.shape}")
        written += 1

    print(f"Done. Written: {written} / {len(rows)}")
    if missing:
        print(f"Missing ESM-IF files ({len(missing)}):")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()