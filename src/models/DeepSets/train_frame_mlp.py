#!/usr/bin/env python3

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, log_loss, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from collect_data_gcs import BASE_DATASET_CSV_URI, DEFAULT_GCS_PREFIX
from lr_data import load_lr_data

FEATURES_PER_FRAME = 3
INPUT_DIM = FEATURES_PER_FRAME * 2

POOL_MODES = ("mean", "max", "std", "mean+max", "mean+std", "mean+std+max")


def _pool_dim(mode: str, hidden: int) -> int:
    return hidden * (mode.count("+") + 1)


class DeepSetsClassifier(nn.Module):
    def __init__(self, in_dim: int = 6, hidden: int = 16,
                 dropout: float = 0.1, pool: str = "mean+std+max"):
        super().__init__()
        self.pool = pool
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        agg_dim = _pool_dim(pool, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(agg_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb     = self.encoder(x)
        mask_f  = mask.float().unsqueeze(-1)
        n_valid = mask_f.sum(dim=1).clamp(min=1)

        mean_pool = (emb * mask_f).sum(dim=1) / n_valid

        parts = []
        if "mean" in self.pool:
            parts.append(mean_pool)
        if "std" in self.pool:
            sq_diff  = ((emb - mean_pool.unsqueeze(1)) ** 2) * mask_f
            std_pool = (sq_diff.sum(dim=1) / n_valid + 1e-8).sqrt()
            parts.append(std_pool)
        if "max" in self.pool:
            emb_max  = emb.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            max_pool = emb_max.amax(dim=1)
            max_pool = torch.nan_to_num(max_pool, nan=0.0, neginf=0.0)
            parts.append(max_pool)

        agg = torch.cat(parts, dim=-1)
        return self.classifier(agg).squeeze(-1)


def prepare_frame_data(X_flat: np.ndarray):
    n, total = X_flat.shape
    T = total // FEATURES_PER_FRAME
    frames = X_flat.reshape(n, T, FEATURES_PER_FRAME)
    mask = ~(frames == 0).all(axis=-1)
    mask[mask.sum(axis=1) == 0, 0] = True
    return frames, mask


def fit_frame_scaler(frames_raw: np.ndarray, mask: np.ndarray) -> StandardScaler:
    valid_frames = frames_raw[mask]
    return StandardScaler().fit(valid_frames)


def scale_frames(frames_raw: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n, T, d = frames_raw.shape
    scaled = scaler.transform(frames_raw.reshape(-1, d)).reshape(n, T, d)
    np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return scaled


def add_delta_features(frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(frames)
    delta[:, 1:, :] = frames[:, 1:, :] - frames[:, :-1, :]
    delta *= mask[:, :, None]
    return np.concatenate([frames, delta], axis=-1)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.nan_to_num(z, nan=0.0, posinf=500.0, neginf=-500.0)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _class_sample_weights(y: np.ndarray) -> np.ndarray:
    n = len(y)
    n_pos = int(y.sum()); n_neg = n - n_pos
    return np.where(y == 1, n / (2 * max(n_pos, 1)), n / (2 * max(n_neg, 1)))


@torch.no_grad()
def _eval_split(model, X_t, mask_t, y_np, sw_np, device):
    model.eval()
    logits = model(X_t, mask_t).cpu().numpy()
    prob   = _sigmoid(logits)
    loss   = log_loss(y_np, np.column_stack([1 - prob, prob]), sample_weight=sw_np)
    auc    = roc_auc_score(y_np, prob) if len(np.unique(y_np)) > 1 else float("nan")
    return loss, auc, logits


def run_training(
    frames_tr, mask_tr, y_train,
    frames_te, mask_te, y_test,
    hidden, dropout, lr, weight_decay, n_epochs, patience, random_state, device,
    pool: str = "mean+std+max",
):
    torch.manual_seed(random_state)
    in_dim = frames_tr.shape[-1]
    model = DeepSetsClassifier(in_dim=in_dim, hidden=hidden, dropout=dropout,
                               pool=pool).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}  pool={pool}")

    n_pos = int(y_train.sum()); n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    optimizer  = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_tr_t    = torch.tensor(frames_tr, dtype=torch.float32, device=device)
    mask_tr_t = torch.tensor(mask_tr,   dtype=torch.bool,    device=device)
    y_tr_t    = torch.tensor(y_train,   dtype=torch.float32, device=device)

    has_test = frames_te is not None and y_test is not None
    if has_test:
        X_te_t    = torch.tensor(frames_te, dtype=torch.float32, device=device)
        mask_te_t = torch.tensor(mask_te,   dtype=torch.bool,    device=device)

    sw_tr = _class_sample_weights(y_train)
    sw_te = _class_sample_weights(y_test) if has_test else None

    tr_losses, te_losses, tr_aucs, te_aucs = [], [], [], []
    log_every = 20

    best_auc       = -1.0
    best_state     = None
    epochs_no_imp  = 0
    best_epoch     = 0

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tr_t, mask_tr_t)
        loss   = nn.functional.binary_cross_entropy_with_logits(logits, y_tr_t, pos_weight=pos_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        l_tr, auc_tr, _ = _eval_split(model, X_tr_t, mask_tr_t, y_train, sw_tr, device)
        tr_losses.append(l_tr); tr_aucs.append(auc_tr)

        if has_test:
            l_te, auc_te, _ = _eval_split(model, X_te_t, mask_te_t, y_test, sw_te, device)
            te_losses.append(l_te); te_aucs.append(auc_te)
            if auc_te > best_auc:
                best_auc      = auc_te
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch    = epoch + 1
                epochs_no_imp = 0
            else:
                epochs_no_imp += 1

        if (epoch + 1) % log_every == 0:
            te_str = (f"  val_loss={te_losses[-1]:.4f}  val_AUC={te_aucs[-1]:.4f}"
                      if has_test else "")
            print(f"  epoch {epoch+1:4d}  tr_loss={tr_losses[-1]:.4f}  "
                  f"tr_AUC={tr_aucs[-1]:.4f}{te_str}")

        if has_test and patience > 0 and epochs_no_imp >= patience:
            print(f"\n  Early stopping at epoch {epoch+1} "
                  f"(no improvement for {patience} epochs, best epoch={best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"  Restored best model from epoch {best_epoch} (val AUC={best_auc:.4f})")

    return model, tr_losses, te_losses, tr_aucs, te_aucs


def _get_prob(model, frames, mask, device):
    X_t = torch.tensor(frames, dtype=torch.float32, device=device)
    m_t = torch.tensor(mask,   dtype=torch.bool,    device=device)
    with torch.no_grad():
        model.eval()
        return _sigmoid(model(X_t, m_t).cpu().numpy())


def evaluate_and_print(
    model, frames_tr, mask_tr, y_train, frames_te, mask_te, y_test,
    plot_roc, plot_roc_output, base, device,
):
    def _section(label, frames, mask, y):
        prob = _get_prob(model, frames, mask, device)
        pred = (prob >= 0.5).astype(int)
        auc  = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan")
        print(f"\n{'='*50}")
        print(label)
        print(f"{'='*50}")
        print(f"Accuracy:          {accuracy_score(y, pred):.4f}")
        print(f"Balanced accuracy: {balanced_accuracy_score(y, pred):.4f}")
        print(f"AUROC:             {auc:.4f}")
        print("\nConfusion matrix (rows=true, cols=predicted):")
        print(confusion_matrix(y, pred))
        print("\nClassification report:")
        print(classification_report(y, pred, target_names=["single_state", "multi_state"]))
        return prob, auc

    prob_tr, _ = _section("Train set", frames_tr, mask_tr, y_train)
    has_test = frames_te is not None and y_test is not None
    if has_test:
        prob_te, auc_te = _section("Test set", frames_te, mask_te, y_test)
        if plot_roc and HAS_MATPLOTLIB:
            fpr, tpr, _ = roc_curve(y_test, prob_te)
            fpr_tr, tpr_tr, _ = roc_curve(y_train, prob_tr)
            auc_tr = roc_auc_score(y_train, prob_tr)
            fig, ax = plt.subplots()
            ax.plot(fpr,    tpr,    color="#a23b72", lw=2, label=f"Test  (AUC={auc_te:.3f})")
            ax.plot(fpr_tr, tpr_tr, color="#2e86ab", lw=2, linestyle="--",
                    label=f"Train (AUC={auc_tr:.3f})")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve"); ax.legend(); ax.grid(True, alpha=0.3)
            out = Path(plot_roc_output) if plot_roc_output else Path("frame_mlp_roc.png")
            if base is not None and not out.is_absolute():
                out = base / out
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150); plt.close(fig)
            print(f"Saved ROC curve to {out}")


def plot_training_curves(
    tr_losses, te_losses, tr_aucs, te_aucs,
    frames_tr, mask_tr, y_train,
    frames_te, mask_te, y_test,
    model, device, output, base,
):
    if not HAS_MATPLOTLIB:
        return
    has_test = frames_te is not None and y_test is not None
    ep_ax = np.arange(1, len(tr_losses) + 1)
    fig, (ax_l, ax_a, ax_r) = plt.subplots(1, 3, figsize=(18, 4))

    ax_l.plot(ep_ax, tr_losses, color="#2e86ab", lw=2, label="Train")
    if has_test:
        ax_l.plot(ep_ax, te_losses, color="#a23b72", lw=2, linestyle="--", label="Test")
    ax_l.set_xlabel("Epoch"); ax_l.set_ylabel("Weighted log-loss")
    ax_l.set_title("Log-loss over epochs"); ax_l.legend(); ax_l.grid(True, alpha=0.3)

    ax_a.plot(ep_ax, tr_aucs, color="#2e86ab", lw=2, label="Train")
    if has_test:
        ax_a.plot(ep_ax, te_aucs, color="#a23b72", lw=2, linestyle="--", label="Test")
    ax_a.set_xlabel("Epoch"); ax_a.set_ylabel("AUROC")
    ax_a.set_title("AUROC over epochs"); ax_a.legend(); ax_a.grid(True, alpha=0.3)

    prob_tr = _get_prob(model, frames_tr, mask_tr, device)
    fpr_tr, tpr_tr, _ = roc_curve(y_train, prob_tr)
    auc_tr = roc_auc_score(y_train, prob_tr)
    if has_test:
        prob_te = _get_prob(model, frames_te, mask_te, device)
        fpr, tpr, _ = roc_curve(y_test, prob_te)
        auc_te = roc_auc_score(y_test, prob_te)
        ax_r.plot(fpr, tpr, color="#a23b72", lw=2, label=f"Test  (AUC={auc_te:.3f})")
    ax_r.plot(fpr_tr, tpr_tr, color="#2e86ab", lw=2, linestyle="--",
              label=f"Train (AUC={auc_tr:.3f})")
    ax_r.plot([0, 1], [0, 1], "k--", lw=1)
    ax_r.set_xlabel("False Positive Rate"); ax_r.set_ylabel("True Positive Rate")
    ax_r.set_title("ROC Curve"); ax_r.legend(); ax_r.grid(True, alpha=0.3)

    fig.suptitle("Frame DeepSets — training curves")
    fig.tight_layout()
    out = Path(output)
    if base is not None and not out.is_absolute():
        out = base / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Saved training curves to {out}")


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train-receptors", type=Path, default=None)
    p.add_argument("--test-receptors",  type=Path, default=None)
    p.add_argument("--gcs-prefix",   type=str,  default=DEFAULT_GCS_PREFIX)
    p.add_argument("--base-dataset", type=str,  default=BASE_DATASET_CSV_URI)
    p.add_argument("--gcs-cache",    type=Path, default=Path(".gcs_cache"))
    p.add_argument("--gcs-workers",  type=int,  default=16)
    p.add_argument("--output-csv",   type=Path, default=None)
    p.add_argument("--data-dir",     type=Path, default=None)
    p.add_argument("--random-state", type=int,  default=42)
    p.add_argument("--val-frac",     type=float, default=0.2)
    p.add_argument("--hidden",   type=int,   default=16)
    p.add_argument("--dropout",  type=float, default=0.1)
    p.add_argument("--pool",     type=str,   default="mean+std+max",
                   choices=list(POOL_MODES))
    p.add_argument("--sweep-pool", action="store_true")
    p.add_argument("--epochs",   type=int,   default=1000)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--wd",       type=float, default=1e-3)
    p.add_argument("--patience", type=int,   default=100)
    p.add_argument("--plot-curves",        action="store_true")
    p.add_argument("--plot-curves-output", type=Path,
                   default=Path("frame_mlp_training_curves.png"))
    p.add_argument("--plot-roc",           action="store_true")
    p.add_argument("--plot-roc-output",    type=Path,
                   default=Path("frame_mlp_roc.png"))
    return p.parse_args()


def main():
    args = parse_args()
    base   = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data    = load_lr_data(args, base)
    y_train = data.y_train.copy()
    y_test  = data.y_test.copy() if data.y_test is not None else None
    has_test = (y_test is not None
                and data.X_test is not None
                and len(data.X_test) > 0)

    frames_tr_raw, mask_tr = prepare_frame_data(data.X_train)
    frames_te_raw, mask_te = (prepare_frame_data(data.X_test)
                               if has_test else (None, None))

    scaler    = fit_frame_scaler(frames_tr_raw, mask_tr)
    frames_tr = add_delta_features(scale_frames(frames_tr_raw, scaler), mask_tr)
    frames_te = (add_delta_features(scale_frames(frames_te_raw, scaler), mask_te)
                 if has_test else None)

    if has_test:
        idx_te = np.arange(len(y_test))
        idx_val, idx_eval = train_test_split(
            idx_te, test_size=(1 - args.val_frac), stratify=y_test,
            random_state=args.random_state,
        )
        frames_val  = frames_te[idx_val];   mask_val  = mask_te[idx_val];   y_val  = y_test[idx_val]
        frames_eval = frames_te[idx_eval];  mask_eval = mask_te[idx_eval];  y_eval = y_test[idx_eval]
    else:
        frames_val = mask_val = y_val = None
        frames_eval = mask_eval = y_eval = None

    n_tr   = len(y_train)
    n_pos  = int(y_train.sum())
    n_val  = len(y_val)  if y_val  is not None else 0
    n_eval = len(y_eval) if y_eval is not None else 0
    T      = frames_tr.shape[1]
    print(f"\n=== Frame DeepSets dataset ===")
    print(f"  {n_tr} train  {n_val} val  {n_eval} test  |  {T} frames × {FEATURES_PER_FRAME} features")
    print(f"  Train: multi_state={n_pos} ({n_pos/n_tr:.1%})  "
          f"single_state={n_tr-n_pos} ({(n_tr-n_pos)/n_tr:.1%})")
    if has_test:
        n_pos_val  = int(y_val.sum())
        n_pos_eval = int(y_eval.sum())
        print(f"  Val:   multi_state={n_pos_val}  ({n_pos_val/n_val:.1%})")
        print(f"  Test:  multi_state={n_pos_eval} ({n_pos_eval/n_eval:.1%})")
    print(f"\nModel: DeepSets  hidden={args.hidden}  dropout={args.dropout}  "
          f"lr={args.lr}  wd={args.wd}  epochs={args.epochs}  "
          f"input_dim={frames_tr.shape[-1]} (raw+delta)\n")

    HIDDEN_SWEEP = [5, 10, 15, 20, 25, 30]

    def _train_one(pool_mode, hidden_size):
        return run_training(
            frames_tr, mask_tr, y_train,
            frames_val, mask_val, y_val,
            hidden=hidden_size,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.wd,
            n_epochs=args.epochs,
            patience=args.patience,
            random_state=args.random_state,
            device=device,
            pool=pool_mode,
        )

    if args.sweep_pool:
        combos = [(pm, h) for pm in POOL_MODES for h in HIDDEN_SWEEP]
        print(f"\n{'='*78}")
        print(f"  Sweep: {len(POOL_MODES)} pool modes × {len(HIDDEN_SWEEP)} hidden sizes "
              f"= {len(combos)} configs")
        print(f"{'='*78}")
        sweep_results = {}
        for pm, h in combos:
            print(f"\n--- pool={pm}  hidden={h} ---")
            m, trl, tel, tra, tea = _train_one(pm, h)
            best_val = max(tea) if tea else float("nan")
            if has_test:
                prob_te  = _get_prob(m, frames_eval, mask_eval, device)
                test_auc = roc_auc_score(y_eval, prob_te) if len(np.unique(y_eval)) > 1 else float("nan")
            else:
                test_auc = float("nan")
            sweep_results[(pm, h)] = (best_val, test_auc, m, trl, tel, tra, tea)

        print(f"\n{'='*78}")
        print(f"  {'Pool mode':<20}  {'hidden':>6}  {'Val AUC':>10}  {'Test AUC':>10}")
        print(f"{'='*78}")
        best_key = max(sweep_results, key=lambda k: sweep_results[k][0])
        for pm, h in combos:
            val_auc, tst_auc = sweep_results[(pm, h)][:2]
            marker = "  <- best (val)" if (pm, h) == best_key else ""
            print(f"  {pm:<20}  {h:>6}  {val_auc:>10.4f}  {tst_auc:>10.4f}{marker}")
        print(f"{'='*78}\n")

        _, _, model, tr_losses, te_losses, tr_aucs, te_aucs = sweep_results[best_key]
        bv, bt = sweep_results[best_key][:2]
        print(f"Best config: pool={best_key[0]}  hidden={best_key[1]}  "
              f"val AUC={bv:.4f}  test AUC={bt:.4f}")
    else:
        model, tr_losses, te_losses, tr_aucs, te_aucs = _train_one(args.pool, args.hidden)

    if args.plot_curves:
        plot_training_curves(
            tr_losses, te_losses, tr_aucs, te_aucs,
            frames_tr, mask_tr, y_train,
            frames_val, mask_val, y_val,
            model, device,
            output=args.plot_curves_output,
            base=base,
        )

    evaluate_and_print(
        model,
        frames_tr, mask_tr, y_train,
        frames_eval, mask_eval, y_eval,
        plot_roc=args.plot_roc,
        plot_roc_output=args.plot_roc_output if args.plot_roc else None,
        base=base,
        device=device,
    )


if __name__ == "__main__":
    main()
