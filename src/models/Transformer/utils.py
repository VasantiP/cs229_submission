import sys
import math
import numpy as np

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, recall_score


class _Tee:
    def __init__(self, path):
        self._file   = open(path, "w", buffering=1)
        self._stdout = sys.stdout
        sys.stdout   = self
    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def close(self):
        sys.stdout = self._stdout
        self._file.close()
    def __enter__(self):  return self
    def __exit__(self, *_): self.close()


def mixup(esmif, res_mask, time_mask, frame_idxs, y, alpha):
    if alpha <= 0:
        return esmif, res_mask, time_mask, frame_idxs, y
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(esmif.size(0), device=esmif.device)
    return (lam * esmif + (1 - lam) * esmif[idx],
            res_mask, time_mask, frame_idxs,
            lam * y + (1 - lam) * y[idx])


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
    return lr


def compute_metrics(logits_list, y_list):
    logits = np.concatenate(logits_list)
    y      = np.concatenate(y_list)
    prob   = 1 / (1 + np.exp(-logits))
    pred   = (prob >= 0.5).astype(int)
    auc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan")
    f1  = f1_score(y, pred, average="macro", zero_division=0)
    sR  = recall_score(y, pred, pos_label=0, zero_division=0)
    mR  = recall_score(y, pred, pos_label=1, zero_division=0)
    return auc, f1, sR, mR
