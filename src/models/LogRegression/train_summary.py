#!/usr/bin/env python3

import argparse
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from collect_data_gcs import BASE_DATASET_CSV_URI, DEFAULT_GCS_PREFIX
from lr_data import load_lr_data
from summary_features import drop_nan_rows, extract_summary_features


def evaluate_and_print(
    clf,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: "np.ndarray | None",
    y_test: "np.ndarray | None",
    model_name: str = "Model",
    plot_roc: bool = False,
    plot_roc_output: "Path | None" = None,
    base: "Path | None" = None,
) -> None:
    def _section(label, X, y):
        y_pred = np.asarray(clf.predict(X), dtype=np.int64)
        if hasattr(clf, "predict_proba"):
            prob = clf.predict_proba(X)[:, 1]
        else:
            prob = clf.decision_function(X)
        auc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan")
        print(f"\n{'='*50}")
        print(label)
        print(f"{'='*50}")
        print(f"Accuracy:          {accuracy_score(y, y_pred):.4f}")
        print(f"Balanced accuracy: {balanced_accuracy_score(y, y_pred):.4f}")
        print(f"AUROC:             {auc:.4f}")
        print("\nConfusion matrix (rows=true, cols=predicted):")
        print(confusion_matrix(y, y_pred))
        print("\nClassification report:")
        print(classification_report(y, y_pred, target_names=["single_state", "multi_state"]))
        return prob, auc

    prob_tr, _ = _section("Train set", X_train, y_train)

    if X_test is not None and y_test is not None:
        prob_te, auc_te = _section("Test set", X_test, y_test)

        if plot_roc and HAS_MATPLOTLIB:
            fpr, tpr, _ = roc_curve(y_test, prob_te)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color="#2e86ab", lw=2, label=f"Test ROC (AUC = {auc_te:.3f})")
            if len(np.unique(y_train)) > 1:
                fpr_tr, tpr_tr, _ = roc_curve(y_train, prob_tr)
                auc_tr = roc_auc_score(y_train, prob_tr)
                ax.plot(fpr_tr, tpr_tr, color="#a23b72", lw=2, linestyle="--",
                        label=f"Train ROC (AUC = {auc_tr:.3f})")
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
            ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve — {model_name} (summary features)")
            ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
            out = Path(plot_roc_output) if plot_roc_output else Path("summary_roc_curve.png")
            if base is not None and not out.is_absolute():
                out = base / out
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150); plt.close(fig)
            print(f"Saved ROC curve to {out}")
        elif plot_roc and not HAS_MATPLOTLIB:
            print("matplotlib not available; skipping ROC curve.")
    else:
        print("(No test set — skipping test evaluation)")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class _LogisticModel:
    def __init__(self, w: np.ndarray, b: float):
        self.w_ = w
        self.b_ = b
        self.coef_ = w.reshape(1, -1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = _sigmoid(X @ self.w_ + self.b_)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _run_gd(X_train, y_train, X_test, y_test,
            penalty, lam, n_epochs, lr, batch_size, rng):
    n, d = X_train.shape
    n_pos = int(y_train.sum()); n_neg = n - n_pos
    sw = np.where(y_train == 1,
                  n / (2.0 * max(n_pos, 1)),
                  n / (2.0 * max(n_neg, 1)))
    has_test = X_test is not None and y_test is not None
    if has_test:
        n_pos_te = int(y_test.sum()); n_neg_te = len(y_test) - n_pos_te
        sw_te = np.where(y_test == 1,
                         len(y_test) / (2.0 * max(n_pos_te, 1)),
                         len(y_test) / (2.0 * max(n_neg_te, 1)))
    use_minibatch = 0 < batch_size < n
    w = np.zeros(d); b = 0.0
    tr_losses, te_losses, tr_aucs, te_aucs = [], [], [], []
    for epoch in range(n_epochs):
        if use_minibatch:
            idx = rng.permutation(n)
            for start in range(0, n, batch_size):
                batch = idx[start: start + batch_size]
                nb = len(batch)
                Xb, yb, swb = X_train[batch], y_train[batch], sw[batch]
                p_b = _sigmoid(Xb @ w + b)
                err = (p_b - yb) * swb
                grad_w = Xb.T @ err / nb
                grad_b = err.mean()
                if penalty == "l2":   grad_w += lam * w
                elif penalty == "l1": grad_w += lam * np.sign(w)
                w -= lr * grad_w; b -= lr * grad_b
        else:
            p = _sigmoid(X_train @ w + b)
            err = (p - y_train) * sw
            grad_w = X_train.T @ err / n
            grad_b = err.mean()
            if penalty == "l2":   grad_w += lam * w
            elif penalty == "l1": grad_w += lam * np.sign(w)
            w -= lr * grad_w; b -= lr * grad_b
        p_tr = _sigmoid(X_train @ w + b)
        tr_losses.append(log_loss(y_train, np.column_stack([1 - p_tr, p_tr]), sample_weight=sw))
        tr_aucs.append(roc_auc_score(y_train, p_tr) if len(np.unique(y_train)) > 1 else float("nan"))
        if has_test:
            p_te = _sigmoid(X_test @ w + b)
            te_losses.append(log_loss(y_test, np.column_stack([1 - p_te, p_te]), sample_weight=sw_te))
            te_aucs.append(roc_auc_score(y_test, p_te) if len(np.unique(y_test)) > 1 else float("nan"))
    return w, b, tr_losses, te_losses, tr_aucs, te_aucs


def train_lr_gd_lam_sweep(
    X_train, y_train, X_test, y_test,
    penalty, lam_values, n_epochs, lr, batch_size, random_state,
    plot_output, base,
    val_frac: float = 0.2,
):
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=random_state)
    tr_idx, val_idx = next(sss.split(X_train, y_train))
    X_tr, y_tr   = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    n_pos_val = int(y_val.sum()); n_neg_val = len(y_val) - n_pos_val
    print(f"\nλ sweep — val split: {len(y_tr)} train  {len(y_val)} val "
          f"(multi_state: tr={int(y_tr.sum())}/{len(y_tr)}, "
          f"val={n_pos_val}/{len(y_val)})")

    has_test = X_test is not None and y_test is not None
    palette = ["#2e86ab", "#a23b72", "#f18f01", "#44bba4", "#e84855",
               "#6c4675", "#c85250", "#8ecefd"]
    ep_ax = np.arange(1, n_epochs + 1)
    all_results = []

    for lam in lam_values:
        rng = np.random.RandomState(random_state)
        mode = "mini-batch" if 0 < batch_size < len(X_tr) else "full-batch"
        print(f"\n  λ={lam}  penalty={penalty}  lr={lr}  epochs={n_epochs}  ({mode})")
        w, b, tr_losses, val_losses, tr_aucs, val_aucs = _run_gd(
            X_tr, y_tr, X_val, y_val,
            penalty, lam, n_epochs, lr, batch_size, rng,
        )
        final_val_auc  = val_aucs[-1]  if val_aucs  else float("nan")
        final_val_loss = val_losses[-1] if val_losses else float("nan")
        all_results.append((lam, final_val_auc, final_val_loss, tr_aucs, val_aucs, tr_losses, val_losses))
        print(f"    Final → tr_loss={tr_losses[-1]:.4f}  tr_AUC={tr_aucs[-1]:.4f}"
              f"  val_loss={final_val_loss:.4f}  val_AUC={final_val_auc:.4f}")
        tmp_model = _LogisticModel(w, b)
        print("    Train classification report:")
        print(classification_report(y_tr, tmp_model.predict(X_tr),
                                    target_names=["single_state", "multi_state"], digits=3))
        print("    Val classification report:")
        print(classification_report(y_val, tmp_model.predict(X_val),
                                    target_names=["single_state", "multi_state"], digits=3))

    best_entry = max(all_results, key=lambda r: r[1])
    lam_best   = best_entry[0]
    print(f"\nBest λ={lam_best}  (val_AUC={best_entry[1]:.4f}  val_loss={best_entry[2]:.4f})")

    print(f"Retraining on full training set with λ={lam_best}...")
    rng_final = np.random.RandomState(random_state)
    w_final, b_final, tr_losses_best, te_losses_best, tr_aucs_best, te_aucs_best = _run_gd(
        X_train, y_train, X_test, y_test,
        penalty, lam_best, n_epochs, lr, batch_size, rng_final,
    )

    if HAS_MATPLOTLIB and plot_output is not None:
        ep_ax_best = np.arange(1, n_epochs + 1)
        has_test   = X_test is not None and y_test is not None
        fig_b, (ax_l, ax_a, ax_r) = plt.subplots(1, 3, figsize=(18, 4))
        ax_l.plot(ep_ax_best, tr_losses_best, color="#2e86ab", lw=2, label="Train")
        if has_test and te_losses_best:
            ax_l.plot(ep_ax_best, te_losses_best, color="#a23b72", lw=2,
                      linestyle="--", label="Test")
        ax_l.set_xlabel("Epoch"); ax_l.set_ylabel("Weighted log-loss")
        ax_l.set_title("Log-loss over epochs"); ax_l.legend(); ax_l.grid(True, alpha=0.3)
        ax_a.plot(ep_ax_best, tr_aucs_best, color="#2e86ab", lw=2, label="Train")
        if has_test and te_aucs_best:
            ax_a.plot(ep_ax_best, te_aucs_best, color="#a23b72", lw=2,
                      linestyle="--", label="Test")
        ax_a.set_xlabel("Epoch"); ax_a.set_ylabel("AUROC")
        ax_a.set_title("AUROC over epochs"); ax_a.legend(); ax_a.grid(True, alpha=0.3)
        best_model = _LogisticModel(w_final, b_final)
        if has_test:
            prob_te = best_model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, prob_te)
            auc_te = roc_auc_score(y_test, prob_te)
            ax_r.plot(fpr, tpr, color="#a23b72", lw=2, label=f"Test (AUC={auc_te:.3f})")
        prob_tr = best_model.predict_proba(X_train)[:, 1]
        fpr_tr, tpr_tr, _ = roc_curve(y_train, prob_tr)
        auc_tr = roc_auc_score(y_train, prob_tr)
        ax_r.plot(fpr_tr, tpr_tr, color="#2e86ab", lw=2, linestyle="--",
                  label=f"Train (AUC={auc_tr:.3f})")
        ax_r.plot([0, 1], [0, 1], "k--", lw=1)
        ax_r.set_xlabel("False Positive Rate"); ax_r.set_ylabel("True Positive Rate")
        ax_r.set_title("ROC Curve"); ax_r.legend(); ax_r.grid(True, alpha=0.3)
        fig_b.suptitle(f"Best λ={lam_best} — train/test curves  (penalty={penalty}, lr={lr})")
        fig_b.tight_layout()
        best_out = Path(plot_output)
        best_out = best_out.parent / (best_out.stem + "_best_lam" + best_out.suffix)
        if base is not None and not best_out.is_absolute():
            best_out = base / best_out
        best_out.parent.mkdir(parents=True, exist_ok=True)
        fig_b.savefig(best_out, dpi=150); plt.close(fig_b)
        print(f"Saved best-λ curves to {best_out}")

    if HAS_MATPLOTLIB and plot_output is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for i, (lam, fva, fvl, tr_aucs, val_aucs, tr_losses, val_losses) in enumerate(all_results):
            color = palette[i % len(palette)]
            is_best = (lam == lam_best)
            lw = 2.5 if is_best else 1.2
            alpha = 1.0 if is_best else 0.55
            suffix = " ★" if is_best else ""
            ax1.plot(ep_ax, val_losses, color=color, lw=lw, alpha=alpha,
                     label=f"Val λ={lam}{suffix}")
            ax1.plot(ep_ax, tr_losses,  color=color, lw=lw * 0.6, alpha=alpha * 0.5,
                     linestyle=":", label=f"Train λ={lam}{suffix}")
            ax2.plot(ep_ax, val_aucs,  color=color, lw=lw, alpha=alpha,
                     label=f"Val λ={lam}{suffix}")
            ax2.plot(ep_ax, tr_aucs,   color=color, lw=lw * 0.6, alpha=alpha * 0.5,
                     linestyle=":", label=f"Train λ={lam}{suffix}")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Weighted log-loss")
        ax1.set_title("Log-loss vs Epoch (λ sweep, solid=Val, dot=Train)")
        ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("AUROC")
        ax2.set_title("AUROC vs Epoch (λ sweep, solid=Val, dot=Train)")
        ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)
        fig.suptitle(
            f"LR GD λ sweep  (penalty={penalty}, lr={lr})  "
            f"Best λ={lam_best} (val AUC={best_entry[1]:.4f}) ★"
        )
        fig.tight_layout()
        out = Path(plot_output)
        if base is not None and not out.is_absolute():
            out = base / out
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"Saved λ-sweep curves to {out}")

    return _LogisticModel(w_final, b_final)


def train_lr_gd_curves(
    X_train, y_train, X_test, y_test,
    penalty, lam, n_epochs, lr, batch_size, random_state,
    plot_output, base,
):
    n = len(y_train)
    rng = np.random.RandomState(random_state)
    has_test = X_test is not None and y_test is not None

    use_minibatch = 0 < batch_size < n
    mode = f"mini-batch (bs={batch_size})" if use_minibatch else "full-batch"
    print(f"\nTraining LR via GD  "
          f"(penalty={penalty}, λ={lam}, lr={lr}, epochs={n_epochs}, {mode})")

    w, b, tr_losses, te_losses, tr_aucs, te_aucs = _run_gd(
        X_train, y_train, X_test, y_test,
        penalty, lam, n_epochs, lr, batch_size, rng,
    )

    for epoch in range(n_epochs):
        if (epoch + 1) % max(1, n_epochs // 10) == 0:
            te_str = (f"  te_loss={te_losses[epoch]:.4f}  te_AUC={te_aucs[epoch]:.4f}"
                      if has_test else "")
            print(f"  epoch {epoch+1:4d}  tr_loss={tr_losses[epoch]:.4f}  "
                  f"tr_AUC={tr_aucs[epoch]:.4f}{te_str}")

    if HAS_MATPLOTLIB and plot_output is not None:
        ep_ax = np.arange(1, n_epochs + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(ep_ax, tr_losses, color="#2e86ab", lw=2, label="Train")
        if has_test:
            ax1.plot(ep_ax, te_losses, color="#a23b72", lw=2, linestyle="--", label="Test")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Weighted log-loss")
        ax1.set_title("Weighted log-loss vs Epoch"); ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.plot(ep_ax, tr_aucs, color="#2e86ab", lw=2, label="Train")
        if has_test:
            ax2.plot(ep_ax, te_aucs, color="#a23b72", lw=2, linestyle="--", label="Test")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("AUROC")
        ax2.set_title("AUROC vs Epoch"); ax2.legend(); ax2.grid(True, alpha=0.3)
        fig.suptitle(f"LR training curves  (penalty={penalty}, λ={lam}, lr={lr})")
        fig.tight_layout()
        out = Path(plot_output)
        if base is not None and not out.is_absolute():
            out = base / out
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"Saved training curves to {out}")

    return _LogisticModel(w, b)


def tune_lr(
    X_train, y_train, X_test, y_test,
    penalty_values, max_iter, random_state, penalty,
    plot_output, base,
) -> float:
    use_lambda = (penalty == "l2")
    x_label    = "λ (log scale)" if use_lambda else "C (log scale)"
    col_header = "λ" if use_lambda else "C"

    solver = "saga" if penalty == "l1" else "lbfgs"
    print("\n" + "=" * 80)
    print(f"  LR sweep  penalty={penalty}  solver={solver}  "
          + ("(x-axis = λ, C = 1/λ)" if use_lambda else "(x-axis = C)"))
    print(f"  {col_header:>10}  {'tr_Loss':>8}  {'te_Loss':>8}  {'tr_AUC':>7}  {'te_AUC':>7}  {'te_Acc':>7}")
    print("=" * 80)
    results = []
    for pv in penalty_values:
        C = 1.0 / pv if use_lambda else pv
        clf = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter,
                                 class_weight="balanced", random_state=random_state)
        clf.fit(X_train, y_train)
        p_tr = clf.predict_proba(X_train)
        p_te = clf.predict_proba(X_test)
        tr_loss = log_loss(y_train, p_tr)
        te_loss = log_loss(y_test,  p_te)
        tr_auc  = roc_auc_score(y_train, p_tr[:, 1]) if len(np.unique(y_train)) > 1 else float("nan")
        te_auc  = roc_auc_score(y_test,  p_te[:, 1]) if len(np.unique(y_test))  > 1 else float("nan")
        te_acc  = accuracy_score(y_test, clf.predict(X_test))
        results.append(dict(pv=pv, C=C, tr_loss=tr_loss, te_loss=te_loss,
                            tr_auc=tr_auc, te_auc=te_auc, te_acc=te_acc))
        print(f"  {col_header}={pv:8g}  {tr_loss:8.4f}  {te_loss:8.4f}  "
              f"{tr_auc:7.4f}  {te_auc:7.4f}  {te_acc:7.4f}")
    print("=" * 80)
    best = max(results, key=lambda r: r["te_auc"])
    print(f"\nBest {col_header}={best['pv']}  "
          f"(te_AUC={best['te_auc']:.4f}  te_Loss={best['te_loss']:.4f})")
    if HAS_MATPLOTLIB and plot_output is not None:
        xs      = [r["pv"]      for r in results]
        tr_loss = [r["tr_loss"] for r in results]
        te_loss = [r["te_loss"] for r in results]
        tr_auc  = [r["tr_auc"]  for r in results]
        te_auc  = [r["te_auc"]  for r in results]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.semilogx(xs, tr_loss, "o-",  color="#2e86ab", label="Train")
        ax1.semilogx(xs, te_loss, "o--", color="#a23b72", label="Test")
        ax1.axvline(best["pv"], color="gray", linestyle=":", lw=1,
                    label=f"Best {col_header}={best['pv']}")
        ax1.set_xlabel(x_label); ax1.set_ylabel("Log-loss")
        ax1.set_title(f"Log-loss vs {col_header}"); ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.semilogx(xs, tr_auc, "o-",  color="#2e86ab", label="Train")
        ax2.semilogx(xs, te_auc, "o--", color="#a23b72", label="Test")
        ax2.axvline(best["pv"], color="gray", linestyle=":", lw=1,
                    label=f"Best {col_header}={best['pv']}")
        ax2.set_xlabel(x_label); ax2.set_ylabel("AUROC")
        ax2.set_title(f"AUROC vs {col_header}"); ax2.legend(); ax2.grid(True, alpha=0.3)
        fig.suptitle(f"LR sweep — summary features  (penalty={penalty})")
        fig.tight_layout()
        out = Path(plot_output)
        if base is not None and not out.is_absolute():
            out = base / out
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"Saved sweep plot to {out}")
    return best["pv"]


def tune_svm(
    X_train, y_train, X_test, y_test,
    c_values, gamma_values, random_state,
    plot_output, base,
) -> "tuple[float, float]":
    from sklearn.svm import SVC
    print("\n" + "=" * 90)
    print("  SVM sweep  (RBF kernel, class_weight=balanced)")
    print(f"{'C':>10}  {'gamma':>10}  {'tr_Loss':>8}  {'te_Loss':>8}  {'tr_AUC':>7}  {'te_AUC':>7}  {'te_Acc':>7}")
    print("=" * 90)
    results = []
    for C in c_values:
        for gamma in gamma_values:
            clf = SVC(C=C, kernel="rbf", gamma=gamma, probability=True,
                      class_weight="balanced", random_state=random_state)
            clf.fit(X_train, y_train)
            p_tr = clf.predict_proba(X_train)
            p_te = clf.predict_proba(X_test)
            tr_loss = log_loss(y_train, p_tr)
            te_loss = log_loss(y_test,  p_te)
            tr_auc  = roc_auc_score(y_train, p_tr[:, 1]) if len(np.unique(y_train)) > 1 else float("nan")
            te_auc  = roc_auc_score(y_test,  p_te[:, 1]) if len(np.unique(y_test))  > 1 else float("nan")
            te_acc  = accuracy_score(y_test, clf.predict(X_test))
            results.append(dict(C=C, gamma=gamma, tr_loss=tr_loss, te_loss=te_loss,
                                tr_auc=tr_auc, te_auc=te_auc, te_acc=te_acc))
            print(f"  C={C:8g}  gamma={gamma:8g}  {tr_loss:8.4f}  {te_loss:8.4f}  "
                  f"{tr_auc:7.4f}  {te_auc:7.4f}  {te_acc:7.4f}")
    print("=" * 90)
    best = max(results, key=lambda r: r["te_auc"])
    print(f"\nBest C={best['C']}  gamma={best['gamma']}  "
          f"(te_AUC={best['te_auc']:.4f}  te_Loss={best['te_loss']:.4f})")

    if HAS_MATPLOTLIB and plot_output is not None:
        unique_gammas = sorted(set(r["gamma"] for r in results))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
        palette = ["#2e86ab", "#a23b72", "#f18f01", "#44bba4", "#e84855"]
        for gi, g in enumerate(unique_gammas):
            color = palette[gi % len(palette)]
            sub = [r for r in results if r["gamma"] == g]
            cs      = [r["C"]       for r in sub]
            tr_loss = [r["tr_loss"] for r in sub]
            te_loss = [r["te_loss"] for r in sub]
            tr_auc  = [r["tr_auc"]  for r in sub]
            te_auc  = [r["te_auc"]  for r in sub]
            ax1.semilogx(cs, tr_loss, "o-",  color=color, alpha=0.6, label=f"Train γ={g}")
            ax1.semilogx(cs, te_loss, "o--", color=color, alpha=1.0, label=f"Test  γ={g}")
            ax2.semilogx(cs, tr_auc,  "o-",  color=color, alpha=0.6, label=f"Train γ={g}")
            ax2.semilogx(cs, te_auc,  "o--", color=color, alpha=1.0, label=f"Test  γ={g}")
        ax1.axvline(best["C"], color="gray", linestyle=":", lw=1)
        ax1.set_xlabel("C (log scale)"); ax1.set_ylabel("Log-loss")
        ax1.set_title("Log-loss vs C"); ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)
        ax2.axvline(best["C"], color="gray", linestyle=":", lw=1)
        ax2.set_xlabel("C (log scale)"); ax2.set_ylabel("AUROC")
        ax2.set_title("AUROC vs C"); ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)
        fig.suptitle("SVM sweep — summary features  (RBF kernel)")
        fig.tight_layout()
        out = Path(plot_output)
        if base is not None and not out.is_absolute():
            out = base / out
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"Saved sweep plot to {out}")

    return best["C"], best["gamma"]


def plot_feature_importance(
    clf,
    feature_names: list,
    model_name: str,
    output_path: "Path | None" = None,
    base: "Path | None" = None,
) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not available; skipping feature importance plot.")
        return

    if hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    elif hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        print("Model has no coef_ or feature_importances_; skipping.")
        return

    idx = np.argsort(importances)[::-1]
    names_sorted = [feature_names[i] for i in idx]
    vals_sorted  = importances[idx]

    colors_map = {"RMSD": "#2e86ab", "Rg": "#a23b72", "TM3": "#f18f01",
                  "corr": "#44bba4"}
    bar_colors = [
        colors_map.get(n.split("_")[0], "#888888")
        for n in names_sorted
    ]

    fig, ax = plt.subplots(figsize=(max(10, len(names_sorted) * 0.35), 4))
    ax.bar(range(len(names_sorted)), vals_sorted, color=bar_colors, alpha=0.85)
    ax.set_xticks(range(len(names_sorted)))
    ax.set_xticklabels(names_sorted, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Coefficient Norm")
    ax.set_title("Highest-weighted descriptors")
    ax.grid(True, axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=v, label=k) for k, v in colors_map.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=8)

    fig.tight_layout()
    out = Path(output_path) if output_path else Path("summary_feature_importance.png")
    if base is not None and not out.is_absolute():
        out = base / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Saved feature importance plot to {out}")


def plot_feature_distributions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
    output_path: "Path | None" = None,
    base: "Path | None" = None,
    n_cols: int = 9,
) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib not available; skipping feature distribution plot.")
        return
    from scipy.stats import gaussian_kde

    n_feat = len(feature_names)
    n_cols = (n_feat + 1) // 2
    n_rows = 2
    colors = {0: "#2e86ab", 1: "#a23b72"}
    labels = {0: "single_state", 1: "multi_state"}

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    axes = axes.ravel()

    for fi, (ax, fname) in enumerate(zip(axes, feature_names)):
        for c in [0, 1]:
            vals = X_train[y_train == c, fi]
            vals = vals[~np.isnan(vals)]
            if len(vals) < 5:
                continue
            xs = np.linspace(vals.min(), vals.max(), 200)
            try:
                kde = gaussian_kde(vals)
                ax.plot(xs, kde(xs), color=colors[c], label=labels[c], lw=1.5)
                ax.fill_between(xs, kde(xs), alpha=0.15, color=colors[c])
            except Exception:
                ax.hist(vals, bins=20, color=colors[c], alpha=0.4, density=True)
        ax.set_title(fname, fontsize=7)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=6)
        ax.grid(True, alpha=0.2)

    for ax in axes[n_feat:]:
        ax.set_visible(False)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors[c], alpha=0.7, label=labels[c]) for c in [0, 1]]
    fig.legend(handles=handles, loc="lower right", ncol=2)
    fig.suptitle("Class distributions for highest-weighted descriptors", y=1.01)
    fig.tight_layout()

    out = Path(output_path) if output_path else Path("summary_feature_distributions.png")
    if base is not None and not out.is_absolute():
        out = base / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved feature distribution plot to {out}")


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-receptors", type=Path, default=None)
    parser.add_argument("--test-receptors",  type=Path, default=None)
    parser.add_argument("--gcs-prefix",   type=str,  default=DEFAULT_GCS_PREFIX)
    parser.add_argument("--base-dataset", type=str,  default=BASE_DATASET_CSV_URI)
    parser.add_argument("--gcs-cache",    type=Path, default=Path(".gcs_cache"))
    parser.add_argument("--gcs-workers",  type=int,  default=16)
    parser.add_argument("--output-csv",   type=Path, default=None)
    parser.add_argument("--data-dir",     type=Path, default=None)
    parser.add_argument("--random-state", type=int,  default=42)
    parser.add_argument("--model", type=str, default="lr", choices=["lr", "svm"])
    parser.add_argument("--penalty", type=str, default="l2", choices=["l1", "l2"])
    parser.add_argument("--lam",     type=float, default=1.0)
    parser.add_argument("--C",       type=float, default=1.0)
    parser.add_argument("--max-iter",type=int,   default=2000)
    parser.add_argument("--svm-C",     type=float, default=1.0)
    parser.add_argument("--svm-gamma", type=str,   default="scale")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--tune-lam-values", type=float, nargs="+",
                        default=[0.0, 0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--tune-c-values", type=float, nargs="+",
                        default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--tune-gamma-values", type=float, nargs="+",
                        default=[0.001, 0.01, 0.1, 1.0])
    parser.add_argument("--tune-plot-output", type=Path,
                        default=Path("summary_hyperparam_sweep.png"))
    parser.add_argument("--plot-curves", action="store_true")
    parser.add_argument("--plot-curves-output", type=Path,
                        default=Path("summary_training_curves.png"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--plot-roc",     action="store_true")
    parser.add_argument("--plot-roc-output", type=Path, default=Path("summary_roc_curve.png"))
    parser.add_argument("--plot-features", action="store_true")
    parser.add_argument("--plot-features-output", type=Path,
                        default=Path("summary_feature_distributions.png"))
    parser.add_argument("--plot-features-top-n", type=int, default=4)
    parser.add_argument("--plot-importance", action="store_true")
    parser.add_argument("--plot-importance-output", type=Path,
                        default=Path("summary_feature_importance.png"))
    return parser.parse_args()


def main():
    args = parse_args()
    base = Path(__file__).resolve().parent

    data = load_lr_data(args, base)

    X_train_raw = data.X_train
    X_test_raw  = data.X_test
    y_train = data.y_train.copy()
    y_test  = data.y_test.copy() if data.y_test is not None else None

    X_train_sum, feat_names = extract_summary_features(X_train_raw)
    X_test_sum = extract_summary_features(X_test_raw)[0] \
                 if X_test_raw is not None and len(X_test_raw) > 0 else None

    X_train_sum, y_train = drop_nan_rows(X_train_sum, y_train, "train")
    if X_test_sum is not None and y_test is not None:
        X_test_sum, y_test = drop_nan_rows(X_test_sum, y_test, "test")

    scaler = StandardScaler().fit(X_train_sum)
    X_train_s = scaler.transform(X_train_sum)
    X_test_s  = scaler.transform(X_test_sum) if X_test_sum is not None else None

    n_tr  = len(y_train)
    n_pos = int(y_train.sum())
    n_te  = len(y_test) if y_test is not None else 0
    print(f"\n=== Summary feature dataset ===")
    print(f"  {n_tr} train samples  {n_te} test samples  |  {len(feat_names)} features")
    print(f"  Features: {feat_names}")
    print(f"  Train class balance: multi_state={n_pos} ({n_pos/n_tr:.1%})  "
          f"single_state={n_tr-n_pos} ({(n_tr-n_pos)/n_tr:.1%})")
    if y_test is not None:
        n_pos_te = int(y_test.sum())
        print(f"  Test:  multi_state={n_pos_te} ({n_pos_te/n_te:.1%})")
    print()

    if args.model == "lr":
        use_lambda = (args.penalty == "l2")
        solver = "saga" if args.penalty == "l1" else "lbfgs"

        if args.plot_curves:
            if args.tune and use_lambda:
                lam_values = args.tune_lam_values
                if args.lam == 0.0 and 0.0 not in lam_values:
                    lam_values = [0.0] + list(lam_values)
                print(f"\nλ sweep with --plot-curves: {lam_values}")
                clf = train_lr_gd_lam_sweep(
                    X_train_s, y_train, X_test_s, y_test,
                    penalty=args.penalty,
                    lam_values=lam_values,
                    n_epochs=args.epochs,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    random_state=args.random_state,
                    plot_output=args.plot_curves_output,
                    base=base,
                )
                model_name = f"LR GD ({args.penalty}, λ sweep)"
            else:
                lam = args.lam if use_lambda else 1.0 / args.C
                clf = train_lr_gd_curves(
                    X_train_s, y_train, X_test_s, y_test,
                    penalty=args.penalty,
                    lam=lam,
                    n_epochs=args.epochs,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    random_state=args.random_state,
                    plot_output=args.plot_curves_output,
                    base=base,
                )
                model_name = f"LR GD ({args.penalty}, λ={lam})"
        else:
            best_pv = args.lam if use_lambda else args.C
            sweep_values = args.tune_lam_values if use_lambda else args.tune_c_values
            if args.tune and X_test_s is not None:
                best_pv = tune_lr(
                    X_train_s, y_train, X_test_s, y_test,
                    penalty_values=sweep_values,
                    max_iter=args.max_iter,
                    random_state=args.random_state,
                    penalty=args.penalty,
                    plot_output=args.tune_plot_output,
                    base=base,
                )
            best_C = 1.0 / best_pv if use_lambda else best_pv
            if use_lambda:
                print(f"\nTraining LogisticRegression (λ={best_pv}, C=1/λ={best_C:.6g}, l2, balanced)...")
            else:
                print(f"\nTraining LogisticRegression (C={best_C}, l1, balanced)...")
            clf = LogisticRegression(
                C=best_C, penalty=args.penalty, solver=solver, max_iter=args.max_iter,
                class_weight="balanced", random_state=args.random_state,
            )
            clf.fit(X_train_s, y_train)
            model_name = f"LR ({args.penalty})"

    else:
        gamma_val = args.svm_gamma
        try:
            gamma_val = float(gamma_val)
        except ValueError:
            pass

        best_C, best_gamma = args.svm_C, gamma_val
        if args.tune and X_test_s is not None:
            tune_gammas = args.tune_gamma_values
            swept_C, swept_gamma = tune_svm(
                X_train_s, y_train, X_test_s, y_test,
                c_values=args.tune_c_values,
                gamma_values=tune_gammas,
                random_state=args.random_state,
                plot_output=args.tune_plot_output,
                base=base,
            )
            if args.tune:
                best_C, best_gamma = swept_C, swept_gamma
        print(f"\nTraining SVM (C={best_C}, gamma={best_gamma}, RBF, balanced)...")
        clf = SVC(
            C=best_C, kernel="rbf", gamma=best_gamma,
            probability=True, class_weight="balanced",
            random_state=args.random_state,
        )
        clf.fit(X_train_s, y_train)
        model_name = f"SVM (RBF, C={best_C}, gamma={best_gamma})"

    evaluate_and_print(
        clf, X_train_s, y_train, X_test_s, y_test,
        model_name=model_name,
        plot_roc=args.plot_roc,
        plot_roc_output=args.plot_roc_output if args.plot_roc else None,
        base=base,
    )

    top_feat_indices = None
    if args.model == "lr" and hasattr(clf, "coef_"):
        coefs = np.abs(clf.coef_[0])
        top_idx = np.argsort(coefs)[::-1]
        n_top = args.plot_features_top_n if args.plot_features_top_n > 0 else len(feat_names)
        top_feat_indices = top_idx[:n_top]

    if args.plot_importance and args.model == "lr":
        plot_feature_importance(
            clf, feat_names, model_name,
            output_path=args.plot_importance_output,
            base=base,
        )
        print(f"\nTop 10 features by |coefficient|:")
        for i in top_feat_indices[:10]:
            print(f"  {feat_names[i]:30s}  {clf.coef_[0][i]:+.4f}")

    if args.plot_features:
        if top_feat_indices is not None:
            plot_feat_names = [feat_names[i] for i in top_feat_indices]
            plot_X = X_train_s[:, top_feat_indices]
        else:
            plot_feat_names = feat_names
            plot_X = X_train_s
        plot_feature_distributions(
            plot_X, y_train, plot_feat_names,
            output_path=args.plot_features_output,
            base=base,
        )


if __name__ == "__main__":
    main()
