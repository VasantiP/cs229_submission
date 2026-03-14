import os
import re
import pandas as pd
from pathlib import Path


BASE_DIR = Path("/home/jupyter/cs229-md-prediction")
DATA_DIR = BASE_DIR / "data/processed"

PROTEIN_TRAIN = DATA_DIR / "protein_level_train.csv"
PROTEIN_TEST  = DATA_DIR / "protein_level_test.csv"

LOCAL_CHUNK_DIRS = [
    Path("/home/jupyter/raw_chunk_cache/train"),
    Path("/home/jupyter/raw_chunk_cache/test"),
]
GCS_CHUNK_DIRS = [
    Path("/home/jupyter/gcs_mount/data/raw_esmif_chunks/train"),
    Path("/home/jupyter/gcs_mount/data/raw_esmif_chunks/test"),
]

OUT_TRAIN = Path("/tmp/emb_bootstrap_train_chunks_newsplit.csv")
OUT_TEST  = Path("/tmp/emb_bootstrap_test_chunks_newsplit.csv")

CHUNK_PATTERN = re.compile(r"^chunk_(.+)_(\d+)\.npy$")


def build_chunk_index(dirs):
    index = {}
    for d in dirs:
        if not d.exists():
            print(f"  [skip] {d}")
            continue
        for fname in os.listdir(d):
            m = CHUNK_PATTERN.match(fname)
            if not m:
                continue
            traj_id  = m.group(1)
            chunk_id = int(m.group(2))
            index.setdefault(traj_id, []).append((chunk_id, str(d / fname)))
    return index


def check_leakage(train_meta, test_meta):
    overlap = set(train_meta["traj_id"]) & set(test_meta["traj_id"])
    if overlap:
        print(f"WARNING: {len(overlap)} traj_ids appear in both splits!")
    else:
        print("Leakage check: clean\n")


def make_chunk_df(protein_df, chunk_index, label):
    rows, missing = [], []
    for _, row in protein_df.iterrows():
        tid = row["traj_id"]
        if tid not in chunk_index:
            missing.append(tid)
            continue
        for chunk_id, _ in sorted(chunk_index[tid]):
            rows.append({"traj_id": tid, "chunk_id": chunk_id, "y": int(row["y"])})

    if missing:
        preview = missing[:10]
        print(f"  [{label}] {len(missing)} traj_ids with no chunks: {preview}"
              + (f" ... +{len(missing)-10}" if len(missing) > 10 else ""))

    df = pd.DataFrame(rows)
    n_pos = (df["y"] == 1).sum()
    n_neg = (df["y"] == 0).sum()
    print(f"  [{label}] {len(df)} chunks / {df['traj_id'].nunique()} trajs  (neg={n_neg}, pos={n_pos})")
    return df


def main():
    print("Scanning local chunk cache ...")
    chunk_index = build_chunk_index(LOCAL_CHUNK_DIRS)
    if not chunk_index:
        print("  Nothing local, falling back to GCS mount ...")
        chunk_index = build_chunk_index(GCS_CHUNK_DIRS)

    n_trajs  = len(chunk_index)
    n_chunks = sum(len(v) for v in chunk_index.values())
    print(f"  {n_trajs} traj_ids, {n_chunks} chunks\n")

    train_meta = pd.read_csv(PROTEIN_TRAIN)
    test_meta  = pd.read_csv(PROTEIN_TEST)
    print(f"Train split: {len(train_meta)} trajs")
    print(f"Test split:  {len(test_meta)} trajs\n")

    check_leakage(train_meta, test_meta)

    train_chunks = make_chunk_df(train_meta, chunk_index, "train")
    test_chunks  = make_chunk_df(test_meta,  chunk_index, "test")

    train_chunks.to_csv(OUT_TRAIN, index=False)
    test_chunks.to_csv(OUT_TEST,   index=False)
    print(f"\nWrote {OUT_TRAIN}")
    print(f"Wrote {OUT_TEST}")


if __name__ == "__main__":
    main()
