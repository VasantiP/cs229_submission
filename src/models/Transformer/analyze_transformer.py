"""
analyze_transformer.py — Comprehensive analysis of a trained ResAttnTransformer.

Plots produced (saved to <run_dir>/analysis/plots/):
  00_roc_curves.png            — ROC curve(s) with AUC annotation
  01_attn_vs_distance.png      — mean attention weight vs. frame distance (ALiBi locality evidence)
  02_umap_frame_paths.png      — UMAP of per-frame residue-pooled embeddings as connected paths (n=40/class)
  03_shuffling_auroc.png       — AUROC normal vs. shuffled frames (temporal dynamics ablation)
  04_attn_entropy.png          — attention entropy H_t vs. relative frame position per layer
  05_training_curves.png       — loss, AUC, sR/mR per epoch (parsed from train.log)
  06_self_attention.png        — (T,T) self-attention heatmap per layer, per class
  07_beta_attention.png        — temporal pooling attention β per sample + mean per class
  08_alibi_bias.png            — ALiBi positional bias visualisation
  09_residue_attention.png     — mean residue attention α per class + Δα (multi − single)
  10_prob_calibration.png      — predicted probability distribution + calibration curve
  11_error_analysis.png        — FP / FN score distributions + confusion breakdown
  12_frame_order_ablation.png  — bar chart ΔAUC original vs shuffled vs reversed (full metrics)
  13_individual_attention.png  — per-sample β + self-attn for top-5 high-conf predictions
  14_intra_window_spread.png   — mean pairwise cosine distance per window by class
  15_half_mask_ablation.png    — AUC drop when first vs second half of frames masked
  16_no_pos_ablation.png       — AUC with vs without positional encodings

Usage:
  python models/analyze_transformer.py \\
      --checkpoint /home/jupyter/runs_transformer/<run>/checkpoint_best.pt

  python models/analyze_transformer.py \\
      --checkpoint .../checkpoint_best.pt \\
      --test-csv /tmp/emb_bootstrap_test_chunks_newsplit.csv \\
      --local-cache /home/jupyter/raw_chunk_cache/all \\
      --n-shuffles 5
"""

import argparse
import re
import sys
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.calibration import calibration_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))
from model import ResAttnTransformer
from dataset import ChunkDataset, collate_fn


def _metrics(probs, ys):
    preds = (probs >= 0.5).astype(int)
    auc   = roc_auc_score(ys, probs) if len(np.unique(ys)) > 1 else float("nan")
    acc   = accuracy_score(ys, preds)
    f1    = f1_score(ys, preds, zero_division=0)
    sr    = (preds[ys == 0] == 0).mean() if (ys == 0).any() else float("nan")
    mr    = (preds[ys == 1] == 1).mean() if (ys == 1).any() else float("nan")
    return dict(auc=auc, acc=acc, f1=f1, sR=sr, mR=mr)


def _fmt(d):
    return (f"AUC={d['auc']:.4f}  acc={d['acc']:.4f}  f1={d['f1']:.4f}"
            f"  sR={d['sR']:.4f}  mR={d['mR']:.4f}")


@torch.no_grad()
def _run_forward(model, batch, device, shuffle=False, reverse=False,
                 mask_half=None, zero_pos=False):
    esmif, res_mask, time_mask, frame_idxs, y = batch
    esmif      = esmif.to(device).float()
    res_mask   = res_mask.to(device)
    time_mask  = time_mask.to(device)
    frame_idxs = frame_idxs.to(device)
    B, T, R    = esmif.shape[:3]

    if shuffle:
        for b in range(B):
            T_v  = int((~time_mask[b]).sum().item())
            perm = torch.randperm(T_v, device=device)
            esmif[b, :T_v]    = esmif[b, :T_v][perm]
            res_mask[b, :T_v] = res_mask[b, :T_v][perm]
    elif reverse:
        for b in range(B):
            T_v = int((~time_mask[b]).sum().item())
            esmif[b, :T_v]    = esmif[b, :T_v].flip(0)
            res_mask[b, :T_v] = res_mask[b, :T_v].flip(0)

    x = esmif.view(B * T, R, -1)
    m = res_mask.view(B * T, R)
    x = model.res_norm(model.input_proj(x))
    scores = model.score_net(x).squeeze(-1).masked_fill(~m, float("-inf"))
    alpha  = torch.softmax(scores, dim=-1)

    alpha_bt   = alpha.view(B, T, R)
    valid_f    = (~time_mask).float().unsqueeze(-1)
    n_valid    = valid_f.sum(1).clamp(min=1)
    alpha_mean = (alpha_bt * valid_f).sum(1) / n_valid

    frame_emb    = (alpha.unsqueeze(-1) * model.value_proj(x)).sum(1).view(B, T, -1)
    frame_emb_np = frame_emb.detach().cpu().numpy()

    if mask_half is not None:
        T_valid = (~time_mask).sum(dim=1)              # (B,)
        time_mask = time_mask.clone()
        for b in range(B):
            T_b  = int(T_valid[b].item())
            half = max(1, T_b // 2)
            if mask_half == "first":
                time_mask[b, :half] = True
            else:
                time_mask[b, half:T_b] = True

    time_mask_np = time_mask.detach().cpu().numpy()

    h = frame_emb
    if zero_pos or shuffle:
        alibi = torch.zeros(B * model.n_heads, T, T, device=h.device)
    else:
        alibi = (ResAttnTransformer._alibi_bias(model.n_heads, T, h.device)
                 .unsqueeze(0).expand(B, -1, -1, -1)
                 .reshape(B * model.n_heads, T, T))

    tm_float  = torch.zeros(B, T, device=h.device).masked_fill(time_mask, float("-inf"))
    attn_maps = []
    for layer in model.layers:
        h_norm = layer.norm1(h)
        _, attn_w = layer.self_attn(
            h_norm, h_norm, h_norm,
            attn_mask=alibi,
            key_padding_mask=tm_float,
            need_weights=True,
            average_attn_weights=True,
        )
        attn_maps.append(attn_w.detach().cpu().numpy())
        h = layer(h, src_mask=alibi, src_key_padding_mask=tm_float)

    t_scores = model.time_score(h).squeeze(-1).masked_fill(time_mask, float("-inf"))
    beta     = torch.softmax(t_scores, dim=-1)
    pooled   = (beta.unsqueeze(-1) * h).sum(1)
    logit    = model.head(pooled)

    prob = torch.sigmoid(logit).squeeze(-1).detach().cpu().numpy()
    return (prob, y.numpy(),
            beta.detach().cpu().numpy(),
            attn_maps,
            alpha_mean.detach().cpu().numpy(),
            frame_emb_np,
            time_mask_np,
            pooled.detach().cpu().numpy())         # (B, d_model)


@torch.no_grad()
def collect_all(model, loader, device, shuffle=False, reverse=False, seed=0,
                mask_half=None, zero_pos=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    all_probs, all_y, all_beta, all_attn, all_alpha = [], [], [], [], []
    all_femb, all_tmask, all_pooled = [], [], []

    for batch in loader:
        prob, y, beta, attn, alpha, femb, tmask, pooled = _run_forward(
            model, batch, device, shuffle, reverse, mask_half, zero_pos)
        all_probs.append(prob)
        all_y.append(y)
        all_beta.append(beta)
        all_attn.append(attn)
        all_alpha.append(alpha)
        all_femb.append(femb)
        all_tmask.append(tmask)
        all_pooled.append(pooled)

    probs = np.concatenate(all_probs)
    ys    = np.concatenate(all_y)

    T_max  = max(b.shape[1] for b in all_beta)
    beta_p = np.zeros((len(probs), T_max))
    idx = 0
    for b in all_beta:
        beta_p[idx:idx+b.shape[0], :b.shape[1]] = b
        idx += b.shape[0]

    R_max   = max(a.shape[1] for a in all_alpha)
    alpha_p = np.zeros((len(probs), R_max))
    idx = 0
    for a in all_alpha:
        alpha_p[idx:idx+a.shape[0], :a.shape[1]] = a
        idx += a.shape[0]

    n_layers    = len(all_attn[0])
    attn_layers = []
    for li in range(n_layers):
        maps = [a[li] for a in all_attn]
        T_l  = max(m.shape[1] for m in maps)
        full = np.zeros((len(probs), T_l, T_l))
        idx2 = 0
        for m in maps:
            B_, t, _ = m.shape
            full[idx2:idx2+B_, :t, :t] = m
            idx2 += B_
        attn_layers.append(full)

    D       = all_femb[0].shape[2]
    T_fe    = max(f.shape[1] for f in all_femb)
    femb_p  = np.zeros((len(probs), T_fe, D))
    tmask_p = np.ones((len(probs), T_fe), dtype=bool)
    idx = 0
    for f, tm in zip(all_femb, all_tmask):
        B_, t, _ = f.shape
        femb_p[idx:idx+B_, :t, :]  = f
        tmask_p[idx:idx+B_, :t]    = tm
        idx += B_

    pooled_p = np.concatenate(all_pooled, axis=0)

    return probs, ys, beta_p, attn_layers, alpha_p, femb_p, tmask_p, pooled_p


def plot_training_curves(log_path, out_path):
    if not Path(log_path).exists():
        print(f"  train.log not found at {log_path}, skipping training curves.")
        return

    epochs, tr_loss, tr_auc, tr_sR, tr_mR = [], [], [], [], []
    te_auc, te_sR, te_mR, te_f1 = [], [], [], []

    pat = re.compile(
        r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
        r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    )
    with open(log_path) as f:
        for line in f:
            m = pat.match(line)
            if m:
                g = m.groups()
                epochs.append(int(g[0]))
                tr_loss.append(float(g[1]))
                tr_sR.append(float(g[2]));  tr_mR.append(float(g[3]))
                tr_auc.append(float(g[4]))
                te_sR.append(float(g[5]));  te_mR.append(float(g[6]))
                te_auc.append(float(g[7]));  te_f1.append(float(g[8]))

    if not epochs:
        print("  Could not parse epoch rows from train.log, skipping training curves.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs, tr_loss, color="steelblue", lw=2, label="train loss")
    axes[0].set_xlabel("Epoch");  axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss");  axes[0].legend()

    axes[1].plot(epochs, tr_auc, color="steelblue", lw=2, label="train AUC")
    axes[1].plot(epochs, te_auc, color="tomato",    lw=2, label="test AUC")
    axes[1].set_xlabel("Epoch");  axes[1].set_ylabel("AUC")
    axes[1].set_title("AUC");  axes[1].legend()

    axes[2].plot(epochs, tr_sR, "--", color="steelblue", lw=1.5, label="train sR")
    axes[2].plot(epochs, tr_mR, "-",  color="steelblue", lw=1.5, label="train mR")
    axes[2].plot(epochs, te_sR, "--", color="tomato",    lw=1.5, label="test sR")
    axes[2].plot(epochs, te_mR, "-",  color="tomato",    lw=1.5, label="test mR")
    axes[2].set_xlabel("Epoch");  axes[2].set_ylabel("Recall")
    axes[2].set_title("Single / Multi-state Recall");  axes[2].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {Path(out_path).name}")


def plot_self_attention(attn_layers, ys, max_frames, out_path):
    n_layers = len(attn_layers)
    fig, axes = plt.subplots(2, n_layers, figsize=(6 * n_layers, 10))
    if n_layers == 1:
        axes = axes.reshape(2, 1)

    classes = {"Single-state (y=0)": ys == 0, "Multi-state (y=1)": ys == 1}
    for row, (label, mask) in enumerate(classes.items()):
        for li in range(n_layers):
            A  = attn_layers[li][mask].mean(0)
            T_ = min(A.shape[0], max_frames)
            ax = axes[row, li]
            im = ax.imshow(A[:T_, :T_], aspect="auto", cmap="Blues", vmin=0)
            ax.set_title(f"Layer {li+1}  |  {label}", fontsize=9)
            ax.set_xlabel("Key frame");  ax.set_ylabel("Query frame")
            plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        "Self-attention maps (mean per class)\n"
        "Diagonal = local/temporal attention  |  Uniform = bag-of-frames"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {Path(out_path).name}")


def plot_beta(beta, ys, max_frames, out_path):
    sort_idx    = np.argsort(ys)
    beta_sorted = beta[sort_idx]
    y_sorted    = ys[sort_idx]
    boundary    = (y_sorted == 0).sum()
    T_vis       = min(beta.shape[1], max_frames)
    beta_vis    = beta_sorted[:, :T_vis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im = axes[0].imshow(beta_vis, aspect="auto", cmap="hot", vmin=0)
    axes[0].axhline(boundary - 0.5, color="cyan", lw=1.5)
    axes[0].set_xlabel("Frame index");  axes[0].set_ylabel("Sample")
    axes[0].set_title("Temporal attention β\n(top=single-state, bottom=multi-state)")
    plt.colorbar(im, ax=axes[0])

    for cls, lbl, col in [(0, "Single-state", "steelblue"), (1, "Multi-state", "tomato")]:
        axes[1].plot(beta[ys == cls, :T_vis].mean(0), label=lbl, color=col, lw=2)
    axes[1].set_xlabel("Frame index");  axes[1].set_ylabel("Mean β weight")
    axes[1].set_title("Mean temporal attention per class");  axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {Path(out_path).name}")


def plot_alibi_bias(model, max_frames, out_path):
    import torch as _torch
    T     = min(max_frames, 64)
    bias  = ResAttnTransformer._alibi_bias(model.n_heads, T,
                                           _torch.device("cpu")).numpy()  # (H, T, T)
    n_heads = bias.shape[0]
    n_heads = model.n_heads
    h_idx   = np.arange(1, n_heads + 1)
    slopes  = 2.0 ** (-8.0 * h_idx / n_heads)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(h_idx, slopes, color="steelblue")
    axes[0].set_xlabel("Head");  axes[0].set_ylabel("Slope")
    axes[0].set_title("ALiBi slopes per head")
    axes[0].set_xticks(h_idx)

    im = axes[1].imshow(bias[0], aspect="auto", cmap="Blues_r", origin="upper")
    axes[1].set_xlabel("Key frame");  axes[1].set_ylabel("Query frame")
    axes[1].set_title(f"ALiBi bias matrix — head 1 (slope={slopes[0]:.3f})")
    plt.colorbar(im, ax=axes[1])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {Path(out_path).name}")


def plot_residue_attention(alpha, ys, out_path):
    R       = alpha.shape[1]
    a_single = alpha[ys == 0].mean(0)   # (R,)
    a_multi  = alpha[ys == 1].mean(0)   # (R,)
    delta    = a_multi - a_single

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].bar(np.arange(R), a_single, color="steelblue", width=1.0, label="Single-state")
    axes[0].set_ylabel("Mean α");  axes[0].set_title("Residue attention — Single-state (y=0)")
    axes[0].legend()

    axes[1].bar(np.arange(R), a_multi, color="tomato", width=1.0, label="Multi-state")
    axes[1].set_ylabel("Mean α");  axes[1].set_title("Residue attention — Multi-state (y=1)")
    axes[1].legend()

    colors = ["tomato" if d > 0 else "steelblue" for d in delta]
    axes[2].bar(np.arange(R), delta, color=colors, width=1.0)
    axes[2].axhline(0, color="black", lw=0.8, ls="--")
    axes[2].set_xlabel("Residue index");  axes[2].set_ylabel("Δα (multi − single)")
    axes[2].set_title("Δ Residue attention  (red=more attended in multi-state)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {Path(out_path).name}")


def plot_calibration(probs, ys, out_path):
    preds   = (probs >= 0.5).astype(int)
    n_bins  = 10

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(probs[ys == 0], bins=40, alpha=0.7, color="steelblue", label="Single-state", density=True)
    axes[0].hist(probs[ys == 1], bins=40, alpha=0.7, color="tomato",    label="Multi-state",  density=True)
    axes[0].axvline(0.5, color="black", ls="--", lw=1)
    axes[0].set_xlabel("Predicted probability");  axes[0].set_ylabel("Density")
    axes[0].set_title("Predicted probability distribution");  axes[0].legend()

    frac_pos, mean_pred = calibration_curve(ys, probs, n_bins=n_bins, strategy="uniform")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    axes[1].plot(mean_pred, frac_pos, "o-", color="tomato", lw=2, label="Model")
    axes[1].set_xlabel("Mean predicted probability");  axes[1].set_ylabel("Fraction positive")
    axes[1].set_title("Calibration curve");  axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {Path(out_path).name}")


def plot_error_analysis(probs, ys, out_path):
    preds   = (probs >= 0.5).astype(int)
    tp_mask = (preds == 1) & (ys == 1)
    tn_mask = (preds == 0) & (ys == 0)
    fp_mask = (preds == 1) & (ys == 0)
    fn_mask = (preds == 0) & (ys == 1)
    cm      = confusion_matrix(ys, preds)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for mask, lbl, col, ls in [
        (tp_mask, f"TP ({tp_mask.sum()})", "tomato",    "-"),
        (tn_mask, f"TN ({tn_mask.sum()})", "steelblue", "-"),
        (fp_mask, f"FP ({fp_mask.sum()})", "orange",    "--"),
        (fn_mask, f"FN ({fn_mask.sum()})", "purple",    "--"),
    ]:
        if mask.sum() > 1:
            axes[0].hist(probs[mask], bins=20, alpha=0.6, color=col,
                         linestyle=ls, label=lbl, density=True)
    axes[0].axvline(0.5, color="black", ls="--", lw=1)
    axes[0].set_xlabel("Predicted probability");  axes[0].set_ylabel("Density")
    axes[0].set_title("Score distributions by outcome (TP/TN/FP/FN)")
    axes[0].legend(fontsize=8)

    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_xticks([0, 1]);  axes[1].set_xticklabels(["Pred Single", "Pred Multi"])
    axes[1].set_yticks([0, 1]);  axes[1].set_yticklabels(["True Single", "True Multi"])
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm[i, j]), ha="center", va="center",
                         fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
    axes[1].set_title("Confusion matrix");  plt.colorbar(im, ax=axes[1])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {Path(out_path).name}")


def plot_frame_order_ablation(orig_m, rev_m, shuf_m, shuf_std, out_path):
    metrics = ["auc", "f1", "sR", "mR"]
    labels  = ["AUC", "F1", "Single recall", "Multi recall"]
    x       = np.arange(len(metrics))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, [orig_m[k] for k in metrics], width, label="Original",  color="steelblue")
    ax.bar(x,         [rev_m[k]  for k in metrics], width, label="Reversed",  color="orange")
    ax.bar(x + width, [shuf_m[k] for k in metrics], width,
           yerr=[shuf_std[k] for k in metrics],
           label="Shuffled (mean±std)", color="tomato", capsize=4)

    ax.set_xticks(x);  ax.set_xticklabels(labels)
    ax.set_ylabel("Score");  ax.set_ylim(0, 1.05)
    ax.set_title("Frame order ablation\n(shuffled ≈ original → model is bag-of-frames)")
    ax.legend(fontsize=9);  ax.grid(axis="y", alpha=0.3)

    delta = orig_m["auc"] - shuf_m["auc"]
    ax.text(0.98, 0.02, f"ΔAUC (orig−shuffled) = {delta:+.4f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.6))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {Path(out_path).name}")


def plot_individual_attention_samples(beta, attn_layers, tmask_p, probs, ys,
                                      n_samples=5, out_path=None):
    multi_idx  = np.where((probs > 0.8)  & (ys == 1))[0]
    single_idx = np.where((probs < 0.2)  & (ys == 0))[0]

    multi_idx  = multi_idx[np.argsort(-probs[multi_idx])][:n_samples]
    single_idx = single_idx[np.argsort(probs[single_idx])][:n_samples]

    n_col = max(len(multi_idx), len(single_idx), 1)
    has_attn = len(attn_layers) > 0

    n_rows = 3 if has_attn else 2
    fig, axes = plt.subplots(n_rows, n_col, figsize=(3.5 * n_col, 3.5 * n_rows))
    if n_col == 1:
        axes = axes.reshape(n_rows, 1)

    def _plot_beta(ax, idx, color, row_label):
        valid = ~tmask_p[idx]
        b     = beta[idx][valid]
        T_v   = len(b)
        xpos   = np.arange(T_v)
        ax.bar(xpos, b, color=color, alpha=0.8, width=0.8)
        ax.axhline(1.0 / max(T_v, 1), color="gray", ls="--", lw=1.2, label="Uniform")
        ax.set_xlabel("Frame (within window)", fontsize=7)
        ax.set_ylabel("β weight", fontsize=7)
        ax.set_title(f"{row_label}\np={probs[idx]:.3f}  T={T_v}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(axis="y", alpha=0.25)

    for col, idx in enumerate(multi_idx):
        _plot_beta(axes[0, col], idx, "#a23b72", "Multi-state ✓")
    for col in range(len(multi_idx), n_col):
        axes[0, col].set_visible(False)

    for col, idx in enumerate(single_idx):
        _plot_beta(axes[1, col], idx, "#2e86ab", "Single-state ✓")
    for col in range(len(single_idx), n_col):
        axes[1, col].set_visible(False)

    if has_attn:
        all_idxs  = list(multi_idx) + list(single_idx)
        all_cols  = list(range(len(multi_idx))) + list(range(len(single_idx)))
        all_colors = ["#a23b72"] * len(multi_idx) + ["#2e86ab"] * len(single_idx)
        all_labels = ["Multi"] * len(multi_idx) + ["Single"] * len(single_idx)
        shown = set()
        for idx, col, col_color, lbl in zip(all_idxs, all_cols, all_colors, all_labels):
            if col in shown:
                col = col + len(multi_idx)
            if col >= n_col:
                continue
            shown.add(col)
            ax = axes[2, col]
            valid  = ~tmask_p[idx]
            T_v    = valid.sum()
            A      = attn_layers[0][idx, :T_v, :T_v]
            im     = ax.imshow(A, aspect="auto", cmap="Blues", vmin=0)
            ax.set_title(f"{lbl} attn\np={probs[idx]:.3f}", fontsize=8)
            ax.set_xlabel("Key frame", fontsize=7);  ax.set_ylabel("Query frame", fontsize=7)
            ax.tick_params(labelsize=6)
        for col in range(n_col):
            if col not in shown:
                axes[2, col].set_visible(False)

    fig.suptitle(
        "Per-sample temporal attention β — high-confidence predictions\n"
        "(flat β near 1/T → all frames equally weighted; peaked → selective)",
        fontsize=10,
    )
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {Path(out_path).name}")
    else:
        plt.show()


def plot_intra_window_spread(femb_p, tmask_p, probs, ys, out_path):
    N = femb_p.shape[0]
    spreads = np.zeros(N)
    for i in range(N):
        valid = ~tmask_p[i]
        emb   = femb_p[i][valid]
        if len(emb) < 2:
            continue
        norms    = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        emb_n    = emb / norms
        cos_sim  = emb_n @ emb_n.T
        n        = len(emb)
        off_diag = cos_sim[~np.eye(n, dtype=bool)]
        spreads[i] = (1.0 - off_diag).mean()

    groups = {
        "High-conf multi  (p>0.8)":     (probs > 0.8) & (ys == 1),
        "High-conf single (p<0.2)":     (probs < 0.2) & (ys == 0),
        "Low-conf  (0.4<p<0.6)":        (probs > 0.4) & (probs < 0.6),
    }
    colors = ["#a23b72", "#2e86ab", "gray"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    data_v   = [spreads[mask] for mask in groups.values() if mask.sum() > 1]
    labels_v = [f"{lbl}\n(n={mask.sum()})"
                for lbl, mask in groups.items() if mask.sum() > 1]
    parts = axes[0].violinplot(data_v, positions=range(len(data_v)), showmedians=True)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col);  pc.set_alpha(0.7)
    axes[0].set_xticks(range(len(labels_v)));  axes[0].set_xticklabels(labels_v, fontsize=8)
    axes[0].set_ylabel("Mean pairwise cosine distance")
    axes[0].set_title("Intra-window structural spread")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].scatter(probs[ys == 0], spreads[ys == 0],
                    c="#2e86ab", alpha=0.5, s=20, label="Single-state")
    axes[1].scatter(probs[ys == 1], spreads[ys == 1],
                    c="#a23b72", alpha=0.5, s=20, label="Multi-state")
    axes[1].set_xlabel("Predicted probability (multi-state)")
    axes[1].set_ylabel("Intra-window spread")
    axes[1].set_title("Structural spread vs predicted confidence")
    axes[1].legend(fontsize=8);  axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {Path(out_path).name}")


def plot_half_mask_ablation(orig_m, first_m, second_m, out_path):
    metrics = ["auc", "f1", "sR", "mR"]
    labels  = ["AUC", "F1", "Single recall", "Multi recall"]
    x       = np.arange(len(metrics))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, [orig_m[k]   for k in metrics], width, label="Full window",     color="steelblue")
    ax.bar(x,         [first_m[k]  for k in metrics], width, label="First ½ masked",  color="tomato")
    ax.bar(x + width, [second_m[k] for k in metrics], width, label="Second ½ masked", color="orange")

    ax.set_xticks(x);  ax.set_xticklabels(labels)
    ax.set_ylabel("Score");  ax.set_ylim(0, 1.05)
    ax.set_title("Half-window masking ablation")
    ax.legend(fontsize=9);  ax.grid(axis="y", alpha=0.3)

    d1 = orig_m["auc"] - first_m["auc"]
    d2 = orig_m["auc"] - second_m["auc"]
    ax.text(0.98, 0.02,
            f"ΔAUC: first½={d1:+.4f}  second½={d2:+.4f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.6))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {Path(out_path).name}")


def plot_no_pos_ablation(orig_m, nopos_m, out_path):
    metrics = ["auc", "f1", "sR", "mR"]
    labels  = ["AUC", "F1", "Single recall", "Multi recall"]
    x       = np.arange(len(metrics))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, [orig_m[k]  for k in metrics], width, label="With pos. enc.",    color="steelblue")
    ax.bar(x + width/2, [nopos_m[k] for k in metrics], width, label="No pos. enc. (zeroed)", color="tomato")

    ax.set_xticks(x);  ax.set_xticklabels(labels)
    ax.set_ylabel("Score");  ax.set_ylim(0, 1.05)
    ax.set_title("Positional encoding ablation")
    ax.legend(fontsize=9);  ax.grid(axis="y", alpha=0.3)

    delta = orig_m["auc"] - nopos_m["auc"]
    verdict = ("Position contributes." if abs(delta) > 0.02
               else "Bag-of-frames: position not needed.")
    ax.text(0.98, 0.02, f"ΔAUC = {delta:+.4f}  →  {verdict}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.6))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {Path(out_path).name}")


def plot_umap_frame_paths(femb_p, tmask_p, probs, ys, out_path,
                          n_samples=8, random_state=42):
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D

    multi_idx  = np.where((probs > 0.8) & (ys == 1))[0]
    single_idx = np.where((probs < 0.2) & (ys == 0))[0]
    multi_idx  = multi_idx[np.argsort(-probs[multi_idx])][:n_samples]
    single_idx = single_idx[np.argsort( probs[single_idx])][:n_samples]
    sel_idx    = np.concatenate([multi_idx, single_idx])
    sel_cls    = np.array([1] * len(multi_idx) + [0] * len(single_idx))

    all_embs, sample_ids, frame_cls = [], [], []
    for si, (idx, cls) in enumerate(zip(sel_idx, sel_cls)):
        valid = ~tmask_p[idx]
        emb   = femb_p[idx][valid]           # (T_i, D)
        all_embs.append(emb)
        sample_ids.extend([si] * len(emb))
        frame_cls.extend([cls] * len(emb))

    all_embs_np = np.concatenate(all_embs, axis=0)
    sample_ids  = np.array(sample_ids)

    try:
        import umap as umap_lib
        reducer = umap_lib.UMAP(n_components=2,
                                n_neighbors=min(15, len(all_embs_np) - 1),
                                min_dist=0.1, random_state=random_state)
        Z = reducer.fit_transform(all_embs_np)
        method = "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=random_state)
        Z   = pca.fit_transform(all_embs_np)
        method = "PCA"
        print(f"  umap-learn not found — falling back to PCA")

    multi_cmap  = cm.get_cmap("Reds",  len(multi_idx)  + 3)
    single_cmap = cm.get_cmap("Blues", len(single_idx) + 3)

    fig, ax = plt.subplots(figsize=(10, 8))

    for si, (idx, cls) in enumerate(zip(sel_idx, sel_cls)):
        mask = sample_ids == si
        Z_s = Z[mask]
        T_v = len(Z_s)
        col = (multi_cmap(si + 2) if cls == 1
               else single_cmap(si - len(multi_idx) + 2))

        ax.plot(Z_s[:, 0], Z_s[:, 1], '-', color=col, alpha=0.55, lw=1.5)
        for t in range(T_v):
            a = 0.25 + 0.75 * t / max(T_v - 1, 1)
            ax.scatter(Z_s[t, 0], Z_s[t, 1], c=[col], s=18, alpha=a, linewidths=0)
        # Start / end markers: circle = start, filled square = end
        ax.scatter(Z_s[0,  0], Z_s[0,  1], marker='o', s=55, c=[col],
                   edgecolors='black', linewidths=0.7, zorder=5)
        ax.scatter(Z_s[-1, 0], Z_s[-1, 1], marker='s', s=55, c=[col],
                   edgecolors='black', linewidths=0.7, zorder=5)

    legend_elems = [
        Line2D([0], [0], color="tomato",    lw=2,
               label=f"Multi-state  (n={len(multi_idx)}, p>0.8)"),
        Line2D([0], [0], color="steelblue", lw=2,
               label=f"Single-state (n={len(single_idx)}, p<0.2)"),
        Line2D([0], [0], marker='o', color='gray', lw=0,
               markersize=6, label="Start of window"),
        Line2D([0], [0], marker='s', color='gray', lw=0,
               markersize=6, label="End of window"),
    ]
    ax.legend(handles=legend_elems, fontsize=9)
    ax.set_xlabel("Component 1");  ax.set_ylabel("Component 2")
    ax.set_title(
        f"{method} of per-frame residue-pooled embeddings\n"
        f"Each path = one bootstrapped window  •  ○ start  ▪ end",
        fontsize=10,
    )
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {Path(out_path).name}")


def plot_roc_curves(curves, out_path):
    from sklearn.metrics import roc_curve
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(curves), 1)))
    fig, ax = plt.subplots(figsize=(6, 5))
    for (label, ys_c, probs_c), col in zip(curves, colors):
        fpr, tpr, _ = roc_curve(ys_c, probs_c)
        auc = roc_auc_score(ys_c, probs_c)
        ax.plot(fpr, tpr, lw=2, color=col, label=f"{label}  (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {Path(out_path).name}")


def plot_attn_vs_distance(attn_layers, tmask_p, ys, out_path):
    n_layers = len(attn_layers)
    T_max     = attn_layers[0].shape[1]
    N         = tmask_p.shape[0]
    dist_sums = np.zeros((n_layers, 2, T_max))
    dist_cnts = np.zeros((n_layers, 2, T_max))

    for i in range(N):
        cls = int(ys[i])
        T_v = int((~tmask_p[i]).sum())
        if T_v < 2:
            continue
        for l in range(n_layers):
            A = attn_layers[l][i, :T_v, :T_v]
            for d in range(T_v):
                if d == 0:
                    vals = np.diag(A, 0)
                else:
                    vals = np.concatenate([np.diag(A, d), np.diag(A, -d)])
                dist_sums[l, cls, d] += vals.sum()
                dist_cnts[l, cls, d] += len(vals)

    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4), squeeze=False)
    colors = {0: "#2e86ab", 1: "#a23b72"}
    labels = {0: "Single-state", 1: "Multi-state"}

    for l in range(n_layers):
        ax = axes[0, l]
        for cls in [0, 1]:
            cnts = dist_cnts[l, cls]
            mask = cnts > 0
            d_vals    = np.where(mask)[0]
            mean_attn = dist_sums[l, cls][mask] / cnts[mask]
            ax.plot(d_vals, mean_attn, lw=2, color=colors[cls], label=labels[cls])
        ax.set_xlabel("Frame distance  |t₁ − t₂|")
        ax.set_ylabel("Mean attention weight")
        ax.set_title(f"Layer {l + 1}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Attention weight vs. frame distance", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {Path(out_path).name}")


def plot_shuffling_auroc(orig_m, shuf_m, shuf_std, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))

    orig_auc  = orig_m["auc"]
    shuf_auc  = shuf_m["auc"]
    shuf_err  = shuf_std["auc"]

    bars = ax.bar(
        ["Normal order", "Shuffled frames"],
        [orig_auc, shuf_auc],
        yerr=[0, shuf_err],
        color=["steelblue", "tomato"],
        width=0.45,
        capsize=6,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.axhline(0.5, color="gray", lw=1.2, ls="--", label="Chance (0.50)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC")
    ax.set_title("Frame-shuffling ablation")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    delta = orig_auc - shuf_auc
    ax.text(
        0.98, 0.02, f"ΔAUC = {delta:+.4f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round", fc="wheat", alpha=0.7),
    )
    for bar, val in zip(bars, [orig_auc, shuf_auc]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {Path(out_path).name}")


def plot_attn_entropy(attn_layers, tmask_p, ys, out_path, n_bins=20):
    n_layers = len(attn_layers)
    N        = tmask_p.shape[0]
    bins     = np.linspace(0, 1, n_bins + 1)
    bin_ctrs = (bins[:-1] + bins[1:]) / 2

    # ent_vals[layer][bin][class] = list of entropy values
    ent_vals = [[{0: [], 1: []} for _ in range(n_bins)] for _ in range(n_layers)]

    for i in range(N):
        cls = int(ys[i])
        T_v = int((~tmask_p[i]).sum())
        if T_v < 2:
            continue
        rel_pos = np.arange(T_v) / max(T_v - 1, 1)
        bin_idx = np.clip(np.digitize(rel_pos, bins) - 1, 0, n_bins - 1)
        for l in range(n_layers):
            A = attn_layers[l][i, :T_v, :T_v]
            H = -(A * np.log(A + 1e-10)).sum(axis=1)   # (T_v,)
            for t, bi in enumerate(bin_idx):
                ent_vals[l][bi][cls].append(H[t])

    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4), squeeze=False)
    colors = {0: "#2e86ab", 1: "#a23b72"}
    labels = {0: "Single-state", 1: "Multi-state"}

    for l in range(n_layers):
        ax = axes[0, l]
        for cls in [0, 1]:
            means = np.array([np.mean(ent_vals[l][bi][cls])
                               if ent_vals[l][bi][cls] else np.nan
                               for bi in range(n_bins)])
            stds  = np.array([np.std(ent_vals[l][bi][cls])
                               if len(ent_vals[l][bi][cls]) > 1 else 0.0
                               for bi in range(n_bins)])
            ok = ~np.isnan(means)
            ax.plot(bin_ctrs[ok], means[ok], lw=2, color=colors[cls], label=labels[cls])
            ax.fill_between(bin_ctrs[ok],
                            (means - stds)[ok], (means + stds)[ok],
                            color=colors[cls], alpha=0.15)
        ax.set_xlabel("Relative frame position  (0 = start,  1 = end)")
        ax.set_ylabel("Attention entropy  H_t")
        ax.set_title(f"Layer {l + 1}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Attention entropy over frame position", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {Path(out_path).name}")


def _load_model(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg   = ckpt["args"]
    model = ResAttnTransformer(
        d_inner      = cfg.get("d_inner",      128),
        d_model      = cfg.get("d_model",      128),
        n_heads      = cfg.get("n_heads",        4),
        n_layers     = cfg.get("n_layers",       2),
        dim_ff       = cfg.get("dim_ff",       256),
        score_hidden = cfg.get("score_hidden",  32),
        layer_drop   = 0.0,
        causal       = cfg.get("causal",      False),
        dropout      = 0.0,
    ).to(device)
    state = ckpt["model"]
    if any(k.startswith("transformer.layers.") for k in state):
        state = {k.replace("transformer.layers.", "layers.", 1): v for k, v in state.items()}
    model_keys = set(model.state_dict().keys())
    state = {k: v for k, v in state.items() if k in model_keys}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint",        required=True)
    p.add_argument("--extra-checkpoints", nargs="*", default=[], metavar="CKPT",
                   help="Additional checkpoints to overlay on the ROC plot")
    p.add_argument("--extra-labels",      nargs="*", default=[], metavar="LABEL",
                   help="Display names for --extra-checkpoints (same order)")
    p.add_argument("--test-csv",    default=None)
    p.add_argument("--local-cache", default=None)
    p.add_argument("--output-dir",  default=None)
    p.add_argument("--batch-size",  type=int, default=16)
    p.add_argument("--n-shuffles",  type=int, default=5)
    args = p.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = _load_model(args.checkpoint, device)

    test_csv    = args.test_csv    or cfg["test_csv"]
    local_cache = args.local_cache or cfg.get("local_cache")
    max_frames  = cfg.get("max_frames", 32)
    max_res     = cfg.get("max_res", None)

    ckpt_path = Path(args.checkpoint)
    out_dir   = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "analysis"
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    log_path  = ckpt_path.parent / "train.log"

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {device}")
    print(f"max_frames : {max_frames}  |  max_res : {max_res}\n")

    import pandas as pd
    test_meta = pd.read_csv(test_csv)

    if "esmif_emb_file" in test_meta.columns:
        print("  Mode: on-the-fly loading (esmif_emb_file detected)")
        from live_dataset import LiveChunkDataset
        ds = LiveChunkDataset(test_meta, max_frames=max_frames, max_res=max_res)
    else:
        ds = ChunkDataset(test_meta, local_cache, max_frames=max_frames, max_res=max_res)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=2, pin_memory=True)

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params\n")

    print("Running inference on test set ...")
    probs, ys, beta, attn_layers, alpha, femb_p, tmask_p, pooled_p = collect_all(model, loader, device)
    m = _metrics(probs, ys)
    print(f"  Test metrics: {_fmt(m)}\n")

    print("Plotting ROC curves ...")
    primary_label = Path(args.checkpoint).parent.name
    roc_curves = [(primary_label, ys, probs)]
    extra_labels = list(args.extra_labels)
    for i, extra_ckpt in enumerate(args.extra_checkpoints):
        label = extra_labels[i] if i < len(extra_labels) else Path(extra_ckpt).parent.name
        print(f"  Loading extra checkpoint: {extra_ckpt}  ({label})")
        extra_model, extra_cfg = _load_model(extra_ckpt, device)
        # Build a loader using this checkpoint's own max_frames / max_res / test_csv
        e_test_csv    = args.test_csv    or extra_cfg["test_csv"]
        e_local_cache = args.local_cache or extra_cfg.get("local_cache")
        e_max_frames  = extra_cfg.get("max_frames", 32)
        e_max_res     = extra_cfg.get("max_res", None)
        print(f"    max_frames={e_max_frames}  max_res={e_max_res}  test_csv={e_test_csv}")
        e_meta = pd.read_csv(e_test_csv)
        if "esmif_emb_file" in e_meta.columns:
            from live_dataset import LiveChunkDataset
            e_ds = LiveChunkDataset(e_meta, max_frames=e_max_frames, max_res=e_max_res)
        else:
            e_ds = ChunkDataset(e_meta, e_local_cache, max_frames=e_max_frames, max_res=e_max_res)
        e_loader = DataLoader(e_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
        extra_probs, extra_ys, *_ = collect_all(extra_model, e_loader, device)
        e_m = _metrics(extra_probs, extra_ys)
        print(f"    {label} metrics: {_fmt(e_m)}")
        roc_curves.append((label, extra_ys, extra_probs))
    plot_roc_curves(roc_curves, out_dir / "plots" / "00_roc_curves.png")

    np.savez(out_dir / "roc_data.npz", y_true=np.array(ys), probs=np.array(probs))

    print("\nRunning frame order ablation (shuffle + reverse) ...")
    rev_probs, *_ = collect_all(model, loader, device, reverse=True)
    rev_m = _metrics(rev_probs, ys)
    print(f"  [Original order]  {_fmt(m)}")
    print(f"  [Reversed order]  {_fmt(rev_m)}")

    shuf_runs = []
    for trial in range(args.n_shuffles):
        sp, *_ = collect_all(model, loader, device, shuffle=True, seed=trial)
        shuf_runs.append(_metrics(sp, ys))
    shuf_m   = {k: np.mean([r[k] for r in shuf_runs]) for k in shuf_runs[0]}
    shuf_std = {k: np.std( [r[k] for r in shuf_runs]) for k in shuf_runs[0]}
    print(f"  [Shuffled (n={args.n_shuffles})]  {_fmt(shuf_m)}")
    print(f"  ± std  AUC={shuf_std['auc']:.4f}")
    delta = m["auc"] - shuf_m["auc"]
    print(f"  ΔAUC (orig − shuffled) = {delta:+.4f}")

    print("\nPlotting attention vs. frame distance ...")
    plot_attn_vs_distance(attn_layers, tmask_p, ys,
                          out_dir / "plots" / "01_attn_vs_distance.png")

    print("\nPlotting UMAP frame paths ...")
    plot_umap_frame_paths(femb_p, tmask_p, probs, ys,
                          out_dir / "plots" / "02_umap_frame_paths.png",
                          n_samples=40)

    print("Plotting shuffling AUROC bar chart ...")
    plot_shuffling_auroc(m, shuf_m, shuf_std,
                         out_dir / "plots" / "03_shuffling_auroc.png")

    print("Plotting attention entropy over frame position ...")
    plot_attn_entropy(attn_layers, tmask_p, ys,
                      out_dir / "plots" / "04_attn_entropy.png")

    print("Plotting training curves ...")
    plot_training_curves(log_path, out_dir / "plots" / "05_training_curves.png")

    print("Plotting self-attention maps ...")
    plot_self_attention(attn_layers, ys, max_frames,
                        out_dir / "plots" / "06_self_attention.png")

    print("Plotting temporal pooling attention (β) ...")
    plot_beta(beta, ys, max_frames, out_dir / "plots" / "07_beta_attention.png")

    print("Plotting ALiBi bias ...")
    plot_alibi_bias(model, max_frames, out_dir / "plots" / "08_alibi_bias.png")

    print("Plotting residue attention ...")
    plot_residue_attention(alpha, ys, out_dir / "plots" / "09_residue_attention.png")

    print("Plotting probability calibration ...")
    plot_calibration(probs, ys, out_dir / "plots" / "10_prob_calibration.png")

    print("Plotting error analysis ...")
    plot_error_analysis(probs, ys, out_dir / "plots" / "11_error_analysis.png")

    plot_frame_order_ablation(m, rev_m, shuf_m, shuf_std,
                              out_dir / "plots" / "12_frame_order_ablation.png")

    print("\nPlotting per-sample attention for high-confidence predictions ...")
    plot_individual_attention_samples(beta, attn_layers, tmask_p, probs, ys,
                                      n_samples=5,
                                      out_path=out_dir / "plots" / "13_individual_attention.png")

    print("Computing intra-window structural spread ...")
    plot_intra_window_spread(femb_p, tmask_p, probs, ys,
                             out_dir / "plots" / "14_intra_window_spread.png")

    print("\nRunning half-window masking ablation ...")
    fp, *_  = collect_all(model, loader, device, mask_half="first")
    sp2, *_ = collect_all(model, loader, device, mask_half="second")
    first_m  = _metrics(fp,  ys)
    second_m = _metrics(sp2, ys)
    print(f"  [Full window]      {_fmt(m)}")
    print(f"  [First ½ masked]   {_fmt(first_m)}")
    print(f"  [Second ½ masked]  {_fmt(second_m)}")
    plot_half_mask_ablation(m, first_m, second_m,
                            out_dir / "plots" / "15_half_mask_ablation.png")

    print("\nRunning no-positional-encoding ablation ...")
    np_probs, *_ = collect_all(model, loader, device, zero_pos=True)
    nopos_m = _metrics(np_probs, ys)
    print(f"  [With pos. enc.]   {_fmt(m)}")
    print(f"  [No  pos. enc.]    {_fmt(nopos_m)}")
    delta_pos = m["auc"] - nopos_m["auc"]
    print(f"  ΔAUC (pos − nopos) = {delta_pos:+.4f}")
    plot_no_pos_ablation(m, nopos_m,
                         out_dir / "plots" / "16_no_pos_ablation.png")

    print(f"\nAll plots saved to: {out_dir}/plots/")


if __name__ == "__main__":
    main()
