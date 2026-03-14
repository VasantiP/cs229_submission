import argparse
import os
import math
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from model import ResAttnTransformer
from dataset import ChunkDataset, collate_fn, cache_check
from utils import _Tee, label_smooth_bce, cosine_lr, compute_metrics


def train(args):
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir  = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(run_dir)

    tee = _Tee(run_dir / "train.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"run dir: {run_dir}")
    print(f"device:  {device}")

    train_meta = pd.read_csv(args.train_csv)
    test_meta  = pd.read_csv(args.test_csv)
    print(f"train: {len(train_meta)}  test: {len(test_meta)}")

    live = "esmif_emb_file" in train_meta.columns
    if live:
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
    n_te_batches = math.ceil(len(test_meta)  / args.micro_bs)
    print(f"micro_bs={args.micro_bs}  eff_bs={eff_bs}  "
          f"tr_batches={n_tr_batches}  te_batches={n_te_batches}")

    train_loader = DataLoader(train_ds, batch_size=args.micro_bs, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.micro_bs, shuffle=False,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)

    model = ResAttnTransformer(
        d_inner=args.d_inner, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, dim_ff=args.dim_ff, dropout=args.dropout,
        layer_drop=args.layer_drop, causal=args.causal, alibi=args.alibi,
        score_hidden=args.score_hidden,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    pos_mode = "alibi" if args.alibi else "pos_emb"
    print(f"ResAttnTransformer  {n_params:,} params  pos={pos_mode}  causal={args.causal}")

    n_neg      = (train_meta["y"] == 0).sum()
    n_pos      = (train_meta["y"] == 1).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"class balance: {n_neg} neg / {n_pos} pos  pos_weight={pos_weight.item():.2f}")

    optimizer    = optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.wd)
    scaler       = torch.amp.GradScaler("cuda")
    warmup_steps = args.warmup * math.ceil(n_tr_batches / args.accum_steps)
    total_steps  = args.epochs  * math.ceil(n_tr_batches / args.accum_steps)

    start_epoch = 1
    best_auc    = 0.0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt["model"]
        if any(k.startswith("transformer.layers.") for k in state):
            state = {k.replace("transformer.layers.", "layers.", 1): v for k, v in state.items()}
        model.load_state_dict(state)
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_auc    = ckpt.get("best_auc", 0.0)
        print(f"resumed from epoch {ckpt['epoch']}  best_auc={best_auc:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    patience_count = 0
    history        = []

    print(f"\n{'Ep':>4} {'tr_loss':>8} {'tr_sR':>6} {'tr_mR':>6} {'tr_AUC':>7} "
          f"{'te_sR':>6} {'te_mR':>6} {'te_AUC':>7} {'te_F1':>6}")
    print("-" * 70)

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

        tr_auc, _, tr_sR, tr_mR = compute_metrics(tr_logits, tr_y)

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

        te_auc, te_f1, te_sR, te_mR = compute_metrics(te_logits, te_y)
        avg_loss = tr_loss / n_tr_batches

        history.append(dict(epoch=epoch, tr_loss=avg_loss, tr_sR=tr_sR, tr_mR=tr_mR,
                            tr_AUC=tr_auc, te_sR=te_sR, te_mR=te_mR, te_AUC=te_auc, te_F1=te_f1))

        print(f"{epoch:>4} {avg_loss:>8.4f} {tr_sR:>6.3f} {tr_mR:>6.3f} {tr_auc:>7.3f} "
              f"{te_sR:>6.3f} {te_mR:>6.3f} {te_auc:>7.3f} {te_f1:>6.3f}")

        ckpt = dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict(), best_auc=best_auc, args=vars(args))
        torch.save(ckpt, Path(args.output_dir) / "checkpoint_latest.pt")

        if te_auc > best_auc:
            best_auc       = te_auc
            patience_count = 0
            torch.save(ckpt, Path(args.output_dir) / "checkpoint_best.pt")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    print("\n" + "=" * 70)
    ckpt = torch.load(Path(args.output_dir) / "checkpoint_best.pt", map_location=device)
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
    print(classification_report(all_y, all_pred, target_names=["single", "multi"]))
    print("confusion matrix:")
    print(confusion_matrix(all_y, all_pred))
    print(f"best test AUC: {best_auc:.4f}")

    pd.DataFrame(history).to_csv(Path(args.output_dir) / "history.csv", index=False)
    print(f"saved to {args.output_dir}")
    tee.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv",    default="/tmp/emb_bootstrap_train_chunks_newsplit.csv")
    p.add_argument("--test-csv",     default="/tmp/emb_bootstrap_test_chunks_newsplit.csv")
    p.add_argument("--local-cache",  default="/home/jupyter/raw_chunk_cache/all")
    p.add_argument("--gcs-mount",    default="/home/jupyter/gcs_mount")
    p.add_argument("--gcs-bucket",   default="cs229-central")
    p.add_argument("--output-dir",   default="/tmp/runs")
    p.add_argument("--resume",       default=None)
    p.add_argument("--epochs",       type=int,   default=80)
    p.add_argument("--peak-lr",      type=float, default=3e-4)
    p.add_argument("--warmup",       type=int,   default=8)
    p.add_argument("--wd",           type=float, default=1e-2)
    p.add_argument("--patience",     type=int,   default=20)
    p.add_argument("--micro-bs",     type=int,   default=16)
    p.add_argument("--accum-steps",  type=int,   default=1)
    p.add_argument("--max-frames",   type=int,   default=64)
    p.add_argument("--max-res",      type=int,   default=300)
    p.add_argument("--label-smooth", type=float, default=0.1)
    p.add_argument("--mixup-alpha",  type=float, default=0.3)
    p.add_argument("--d-inner",      type=int,   default=128)
    p.add_argument("--d-model",      type=int,   default=128)
    p.add_argument("--n-heads",      type=int,   default=4)
    p.add_argument("--n-layers",     type=int,   default=2)
    p.add_argument("--dim-ff",       type=int,   default=256)
    p.add_argument("--score-hidden", type=int,   default=32)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--layer-drop",   type=float, default=0.0)
    p.add_argument("--alibi",        action="store_true", default=False)
    p.add_argument("--causal",       action="store_true", default=False)
    p.add_argument("--run-name",     default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
