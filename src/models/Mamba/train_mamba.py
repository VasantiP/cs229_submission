"""
Mamba SSM classifier for ESM-IF chunk embeddings (single vs multi-state).

Residue pooling strategies (--pooling):
  mean      : mean over valid residues -> Linear(512, d_model)
  meanvar   : [mean, std] concat -> Linear(1024, d_model)
  attention : learned attention scores over residues (same as ResAttnTransformer)
  conv      : Conv1d over residue dim -> mean pool -> LayerNorm
  resattn   : TransformerEncoder over residues + attention pool (ResAttnTransformer Stage 1)

Usage:
    python models/train_mamba.py --pooling attention --split newsplit
    python models/train_mamba.py --pooling meanvar   --split original
    python models/train_mamba.py --pooling conv \\
        --train-csv /tmp/my_train.csv --test-csv /tmp/my_test.csv \\
        --local-cache /home/jupyter/raw_chunk_cache/all

Requires:
    pip install mambapy           # pure PyTorch, no CUDA ext needed
    pip install mamba-ssm causal-conv1d  # faster CUDA kernels
"""

import argparse
import json
import math
import platform
import subprocess
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score,
    classification_report, confusion_matrix,
)

from mamba_model import MambaClassifier, MAMBA_AVAILABLE


SPLIT_PRESETS = {
    "newsplit": {
        "train_csv":   "/tmp/emb_bootstrap_train_chunks_newsplit.csv",
        "test_csv":    "/tmp/emb_bootstrap_test_chunks_newsplit.csv",
        "local_cache": "/home/jupyter/raw_chunk_cache/all",
    },
    "original": {
        "train_csv":   "/tmp/emb_bootstrap_train_chunks_original.csv",
        "test_csv":    "/tmp/emb_bootstrap_test_chunks_original.csv",
        "local_cache": "/home/jupyter/raw_chunk_cache/all",
    },
}


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
            sel = np.sort(np.random.choice(arr.shape[0], self.max_frames, replace=False))
            arr = arr[sel]
        if self.max_res and arr.shape[1] > self.max_res:
            arr = arr[:, :self.max_res]
        mask = np.isfinite(arr.astype(np.float32).sum(-1))
        arr  = arr.astype(np.float16)
        arr  = np.where(mask[..., None], arr, np.float16(0.0))
        return {
            "esmif":      torch.from_numpy(arr),
            "res_mask":   torch.from_numpy(mask),
            "frame_idxs": torch.arange(arr.shape[0], dtype=torch.long),
            "y":          float(row["y"]),
        }


def collate_fn(batch):
    T_max = max(b["esmif"].shape[0] for b in batch)
    R_max = max(b["esmif"].shape[1] for b in batch)
    B, D  = len(batch), batch[0]["esmif"].shape[2]

    esmif      = torch.zeros(B, T_max, R_max, D, dtype=torch.float16)
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
    n_ok, n_drop = sum(exists), len(exists) - sum(exists)
    print(f"  [{label}] {n_ok} usable chunks" + (f"  (dropped {n_drop} missing)" if n_drop else ""))
    return meta_df[exists].reset_index(drop=True)


def label_smooth_bce(logits, y, smooth, pos_weight=None):
    y_s = y * (1 - smooth) + 0.5 * smooth
    return nn.functional.binary_cross_entropy_with_logits(
        logits.squeeze(-1), y_s, pos_weight=pos_weight
    )


def cosine_lr(optimizer, step, total_steps, peak_lr, warmup_steps, min_lr=1e-6):
    if step < warmup_steps:
        lr = peak_lr * step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def compute_metrics(logits_list, y_list):
    logits = np.concatenate(logits_list)
    y      = np.concatenate(y_list)
    prob   = 1 / (1 + np.exp(-logits))
    pred   = (prob >= 0.5).astype(int)
    auc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan")
    f1  = f1_score(y, pred, average="macro", zero_division=0)
    sR  = recall_score(y, pred, pos_label=0, zero_division=0)
    mR  = recall_score(y, pred, pos_label=1, zero_division=0)
    acc = (pred == y.astype(int)).mean()
    return auc, f1, sR, mR, acc


def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _make_run_dir(args):
    name = args.run_name or f"{args.pooling}__{args.split}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, name


def save_config(run_dir, args, n_params, n_tr_pos, n_tr_neg, n_te_pos, n_te_neg):
    cfg = {
        "run_name":          run_dir.name,
        "timestamp":         datetime.now().isoformat(),
        "git_hash":          _git_hash(),
        "python":            platform.python_version(),
        "hostname":          platform.node(),
        **vars(args),
        "n_params":          n_params,
        "train_n_pos":       int(n_tr_pos),
        "train_n_neg":       int(n_tr_neg),
        "test_n_pos":        int(n_te_pos),
        "test_n_neg":        int(n_te_neg),
        "train_pos_weight":  round(n_tr_neg / max(n_tr_pos, 1), 4),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def save_results(run_dir, cfg, history, best_auc, best_epoch, report_str, confusion):
    results = {
        "run_name":              cfg["run_name"],
        "timestamp_end":         datetime.now().isoformat(),
        "best_auc":              round(best_auc, 6),
        "best_epoch":            best_epoch,
        "total_epochs":          len(history),
        "final_te_AUC":          round(history[-1]["te_AUC"], 6),
        "final_te_F1":           round(history[-1]["te_F1"], 6),
        "final_te_sR":           round(history[-1]["te_sR"], 6),
        "final_te_mR":           round(history[-1]["te_mR"], 6),
        "classification_report": report_str,
        "confusion_matrix":      confusion.tolist(),
        "pooling":               cfg["pooling"],
        "split":                 cfg["split"],
        "d_model":               cfg["d_model"],
        "n_layers":              cfg["n_layers"],
        "d_state":               cfg["d_state"],
        "dropout":               cfg["dropout"],
        "peak_lr":               cfg["peak_lr"],
        "n_params":              cfg["n_params"],
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved config.json + results.json -> {run_dir}")


class _Tee:
    def __init__(self, log_path):
        self._term = sys.__stdout__
        self._file = open(log_path, "a", buffering=1)

    def write(self, data):
        self._term.write(data)
        self._file.write(data)

    def flush(self):
        self._term.flush()
        self._file.flush()

    def close(self):
        self._file.close()


def train(args):
    if not MAMBA_AVAILABLE:
        raise RuntimeError(
            "No Mamba backend found. Install one of:\n"
            "  pip install mambapy\n"
            "  pip install mamba-ssm causal-conv1d"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Pooling: {args.pooling}  |  Split: {args.split or 'custom'}")

    train_meta = pd.read_csv(args.train_csv)
    test_meta  = pd.read_csv(args.test_csv)
    print(f"  Train: {len(train_meta)} chunks, Test: {len(test_meta)} chunks")

    if "esmif_emb_file" in train_meta.columns:
        from live_dataset import LiveChunkDataset
        train_ds = LiveChunkDataset(train_meta, args.max_frames, args.max_res)
        test_ds  = LiveChunkDataset(test_meta,  args.max_frames, args.max_res)
    else:
        train_meta = cache_check(train_meta, args.local_cache, "train")
        test_meta  = cache_check(test_meta,  args.local_cache, "test")
        train_ds = ChunkDataset(train_meta, args.local_cache, args.max_frames, args.max_res)
        test_ds  = ChunkDataset(test_meta,  args.local_cache, args.max_frames, args.max_res)

    eff_bs       = args.micro_bs * args.accum_steps
    n_tr_batches = math.ceil(len(train_meta) / args.micro_bs)
    print(f"  micro_bs={args.micro_bs}  eff_bs={eff_bs}  tr_batches={n_tr_batches}")

    train_loader = DataLoader(train_ds, batch_size=args.micro_bs, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.micro_bs, shuffle=False,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model = MambaClassifier(
        pooling=args.pooling,
        d_inner=args.d_inner, d_model=args.d_model,
        n_layers=args.n_layers, d_state=args.d_state,
        d_conv=args.d_conv, expand=args.expand,
        dropout=args.dropout, conv_kernel=args.conv_kernel,
        n_res_layers=args.n_res_layers, n_res_heads=args.n_res_heads,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nMambaClassifier[{args.pooling}]  {n_params:,} params")

    n_tr_neg   = (train_meta["y"] == 0).sum()
    n_tr_pos   = (train_meta["y"] == 1).sum()
    n_te_neg   = (test_meta["y"] == 0).sum()
    n_te_pos   = (test_meta["y"] == 1).sum()
    pw         = n_tr_neg / max(n_tr_pos, 1)
    pos_weight = torch.tensor([pw], device=device)
    print(f"Class balance: {n_tr_neg} neg / {n_tr_pos} pos  pos_weight={pw:.2f}")

    run_dir, _ = _make_run_dir(args)
    cfg        = save_config(run_dir, args, n_params, n_tr_pos, n_tr_neg, n_te_pos, n_te_neg)
    tee        = _Tee(run_dir / "train.log")
    sys.stdout = tee
    print(f"Run dir: {run_dir}")

    optimizer    = optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.wd)
    scaler       = torch.amp.GradScaler("cuda")
    warmup_steps = args.warmup * math.ceil(n_tr_batches / args.accum_steps)
    total_steps  = args.epochs * math.ceil(n_tr_batches / args.accum_steps)

    start_epoch    = 1
    best_auc       = 0.0
    best_epoch     = 1
    patience_count = 0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_auc    = ckpt.get("best_auc", 0.0)
        best_epoch  = ckpt.get("best_epoch", 1)
        print(f"Resumed from epoch {ckpt['epoch']}  best_auc={best_auc:.4f}")

    history = []
    print(f"\n{'Ep':>4} {'loss':>8} "
          f"| {'tr_acc':>6} {'tr_sR':>6} {'tr_mR':>6} {'tr_AUC':>7} "
          f"| {'te_acc':>6} {'te_sR':>6} {'te_mR':>6} {'te_AUC':>7}")
    print("-" * 85)

    step = (start_epoch - 1) * math.ceil(n_tr_batches / args.accum_steps)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_logits, tr_y = [], []
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Ep {epoch:>3}", leave=False, unit="batch")
        for batch_idx, (esmif, res_mask, time_mask, frame_idxs, y) in enumerate(pbar):
            esmif      = esmif.to(device)
            res_mask   = res_mask.to(device)
            time_mask  = time_mask.to(device)
            frame_idxs = frame_idxs.to(device)
            y          = y.to(device)

            with torch.amp.autocast("cuda"):
                logits = model(esmif, res_mask, time_mask, frame_idxs)
                loss   = label_smooth_bce(logits, y, args.label_smooth, pos_weight) / args.accum_steps

            scaler.scale(loss).backward()
            tr_loss += loss.item() * args.accum_steps
            tr_logits.append(logits.detach().float().cpu().numpy().reshape(-1))
            tr_y.append(y.cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item()*args.accum_steps:.4f}", refresh=False)

            if (batch_idx + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                cosine_lr(optimizer, step, total_steps, args.peak_lr, warmup_steps)
                step += 1

        tr_auc, _, tr_sR, tr_mR, tr_acc = compute_metrics(tr_logits, tr_y)

        model.eval()
        te_logits, te_y = [], []
        with torch.no_grad():
            for esmif, res_mask, time_mask, frame_idxs, y in test_loader:
                esmif      = esmif.to(device)
                res_mask   = res_mask.to(device)
                time_mask  = time_mask.to(device)
                frame_idxs = frame_idxs.to(device)
                with torch.amp.autocast("cuda"):
                    logits = model(esmif, res_mask, time_mask, frame_idxs)
                te_logits.append(logits.float().cpu().numpy().reshape(-1))
                te_y.append(y.numpy())

        te_auc, te_f1, te_sR, te_mR, te_acc = compute_metrics(te_logits, te_y)
        avg_loss = tr_loss / n_tr_batches

        history.append(dict(
            epoch=epoch, tr_loss=avg_loss,
            tr_acc=tr_acc, tr_sR=tr_sR, tr_mR=tr_mR, tr_AUC=tr_auc,
            te_acc=te_acc, te_sR=te_sR, te_mR=te_mR, te_AUC=te_auc, te_F1=te_f1,
        ))
        print(f"{epoch:>4} {avg_loss:>8.4f} "
              f"| tr: acc={tr_acc:.3f} sR={tr_sR:.3f} mR={tr_mR:.3f} AUC={tr_auc:.3f} "
              f"| te: acc={te_acc:.3f} sR={te_sR:.3f} mR={te_mR:.3f} AUC={te_auc:.3f}")

        ckpt = dict(epoch=epoch, model=model.state_dict(),
                    optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                    best_auc=best_auc, best_epoch=best_epoch,
                    run_name=run_dir.name, args=vars(args))
        torch.save(ckpt, run_dir / "checkpoint_latest.pt")

        if te_auc > best_auc:
            best_auc       = te_auc
            best_epoch     = epoch
            patience_count = 0
            torch.save(ckpt, run_dir / "checkpoint_best.pt")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        if epoch == 4 and best_auc < 0.6:
            print(f"\nTerminating: test AUC {best_auc:.4f} < 0.6 by epoch 4.")
            break

    print("\n" + "=" * 70)
    ckpt = torch.load(run_dir / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    te_logits, te_y = [], []
    with torch.no_grad():
        for esmif, res_mask, time_mask, frame_idxs, y in test_loader:
            esmif      = esmif.to(device)
            res_mask   = res_mask.to(device)
            time_mask  = time_mask.to(device)
            frame_idxs = frame_idxs.to(device)
            with torch.amp.autocast("cuda"):
                logits = model(esmif, res_mask, time_mask, frame_idxs)
            te_logits.append(logits.float().cpu().numpy().reshape(-1))
            te_y.append(y.numpy())

    all_logits = np.concatenate(te_logits)
    all_y      = np.concatenate(te_y)
    all_prob   = 1 / (1 + np.exp(-all_logits))
    all_pred   = (all_prob >= 0.5).astype(int)
    report_str = classification_report(all_y, all_pred, target_names=["single", "multi"])
    cm         = confusion_matrix(all_y, all_pred)
    print(report_str)
    print("Confusion matrix:")
    print(cm)
    print(f"Best test AUC: {best_auc:.4f}  (epoch {best_epoch})")

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    save_results(run_dir, cfg, history, best_auc, best_epoch, report_str, cm)
    print(f"\nAll outputs in: {run_dir}")
    sys.stdout = sys.__stdout__
    tee.close()


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--split",       choices=["newsplit", "original"], default="newsplit")
    p.add_argument("--train-csv",   default=None)
    p.add_argument("--test-csv",    default=None)
    p.add_argument("--local-cache", default=None)
    p.add_argument("--output-dir",  default="/home/jupyter/runs_mamba")
    p.add_argument("--run-name",    default=None)
    p.add_argument("--resume",      default=None)
    p.add_argument("--pooling",     choices=["mean", "meanvar", "attention", "conv", "resattn"],
                   default="attention")
    p.add_argument("--conv-kernel",  type=int,   default=5)
    p.add_argument("--n-res-layers", type=int,   default=1)
    p.add_argument("--n-res-heads",  type=int,   default=4)
    p.add_argument("--epochs",       type=int,   default=80)
    p.add_argument("--peak-lr",      type=float, default=1e-4)
    p.add_argument("--warmup",       type=int,   default=2)
    p.add_argument("--wd",           type=float, default=1e-2)
    p.add_argument("--patience",     type=int,   default=6)
    p.add_argument("--micro-bs",     type=int,   default=16)
    p.add_argument("--accum-steps",  type=int,   default=1)
    p.add_argument("--max-frames",   type=int,   default=32)
    p.add_argument("--max-res",      type=int,   default=None)
    p.add_argument("--label-smooth", type=float, default=0.1)
    p.add_argument("--d-inner",      type=int,   default=128)
    p.add_argument("--d-model",      type=int,   default=128)
    p.add_argument("--n-layers",     type=int,   default=2)
    p.add_argument("--d-state",      type=int,   default=16)
    p.add_argument("--d-conv",       type=int,   default=4)
    p.add_argument("--expand",       type=int,   default=2)
    p.add_argument("--dropout",      type=float, default=0.3)

    args = p.parse_args()
    preset = SPLIT_PRESETS[args.split]
    if args.train_csv   is None: args.train_csv   = preset["train_csv"]
    if args.test_csv    is None: args.test_csv     = preset["test_csv"]
    if args.local_cache is None: args.local_cache  = preset["local_cache"]
    return args


if __name__ == "__main__":
    train(parse_args())
