import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset


class ChunkDataset(Dataset):
    def __init__(self, meta_df, local_dir, max_frames=None, max_res=None):
        self.meta       = meta_df.reset_index(drop=True)
        self.local_dir  = Path(local_dir)
        self.max_frames = max_frames
        self.max_res    = max_res

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row   = self.meta.iloc[idx]
        fpath = self.local_dir / f"chunk_{row['traj_id']}_{int(row['chunk_id'])}.npy"
        arr   = np.load(fpath, mmap_mode="r")[:]
        if self.max_frames and arr.shape[0] > self.max_frames:
            arr = arr[:self.max_frames]
        if self.max_res and arr.shape[1] > self.max_res:
            arr = arr[:, :self.max_res]
        arr  = arr.astype(np.float32)
        mask = np.isfinite(arr.sum(-1))
        arr  = np.where(mask[..., None], arr, 0.0)
        return {
            "esmif":      torch.from_numpy(arr),
            "res_mask":   torch.from_numpy(mask),
            "frame_idxs": torch.arange(arr.shape[0], dtype=torch.long),
            "y":          float(row["y"]),
        }


def collate_fn(batch):
    T_max = max(b["esmif"].shape[0] for b in batch)
    R_max = max(b["esmif"].shape[1] for b in batch)
    B     = len(batch)
    D     = batch[0]["esmif"].shape[2]

    esmif      = torch.zeros(B, T_max, R_max, D)
    res_mask   = torch.zeros(B, T_max, R_max, dtype=torch.bool)
    time_mask  = torch.ones(B, T_max, dtype=torch.bool)
    frame_idxs = torch.zeros(B, T_max, dtype=torch.long)
    y          = torch.zeros(B)

    for i, b in enumerate(batch):
        T, R = b["esmif"].shape[:2]
        esmif[i, :T, :R]    = b["esmif"]
        res_mask[i, :T, :R] = b["res_mask"]
        time_mask[i, :T]    = False
        frame_idxs[i, :T]   = b["frame_idxs"]
        y[i]                = b["y"]

    return esmif, res_mask, time_mask, frame_idxs, y


def cache_check(meta_df, local_dir, label):
    local_dir = Path(local_dir)
    exists = [
        (local_dir / f"chunk_{row['traj_id']}_{int(row['chunk_id'])}.npy").exists()
        for _, row in meta_df.iterrows()
    ]
    n_ok   = sum(exists)
    n_drop = len(exists) - n_ok
    print(f"  [{label}] {n_ok} chunks cached locally")
    if n_drop:
        print(f"  [{label}] dropped {n_drop} missing")
    return meta_df[exists].reset_index(drop=True)
