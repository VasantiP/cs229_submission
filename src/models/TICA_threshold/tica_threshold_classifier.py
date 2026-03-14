import argparse
import warnings
import numpy as np
import pandas as pd
import fsspec
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.linalg import eigh
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

warnings.filterwarnings("ignore")

ROOT           = Path(__file__).resolve().parents[1]
BOOTSTRAP_META = ROOT / "notebooks/bootstrap_metadata.csv"
TRAIN_CSV      = ROOT / "data/processed/protein_level_train.csv"
TEST_CSV       = ROOT / "data/processed/protein_level_test.csv"
GCS_PROCESSED  = "gs://cs229-central/data/processed/v3"

_gcs_cache = {}


def load_npy_gcs(path):
    if path not in _gcs_cache:
        with fsspec.open(path, "rb") as f:
            _gcs_cache[path] = np.load(f)
    return _gcs_cache[path]


def fit_tica_eigenvalues(X, lag, n_components):
    """Solve the TICA generalized eigenvalue problem on X (n_frames, n_features)."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_frames, n_feat = X.shape
    n_components = min(n_components, n_feat - 1)
    if n_frames <= lag + 2 or n_components < 1:
        return np.full(n_components, np.nan)

    X0 = X[:-lag].astype(np.float64)
    Xt = X[lag:].astype(np.float64)
    mu = X0.mean(axis=0)
    X0 -= mu
    Xt -= mu

    T  = len(X0)
    C0 = (X0.T @ X0) / T
    Ct = (X0.T @ Xt) / T
    Ct = (Ct + Ct.T) / 2
    C0 += np.eye(n_feat) * 1e-6

    try:
        vals, _ = eigh(Ct, C0, subset_by_index=[n_feat - n_components, n_feat - 1])
        ev = vals[::-1]
        if len(ev) < n_components:
            ev = np.concatenate([ev, np.full(n_components - len(ev), np.nan)])
        return ev[:n_components]
    except Exception as e:
        warnings.warn(f"TICA eigh failed: {e}")
        return np.full(n_components, np.nan)


def compute_chunk_eigenvalues(bootstrap_meta, base_df, lag, n_ev):
    frames_lookup = {
        (row.receptor, int(row.rep), int(row.simID)): row.early_ts_path
        for row in base_df.itertuples(index=False)
    }

    rows = []
    for (receptor, rep, simID), grp in tqdm(
        bootstrap_meta.groupby(["receptor", "rep", "simID"]), desc="trajectories"
    ):
        key = (receptor, int(rep), int(simID))
        if key not in frames_lookup:
            warnings.warn(f"No frames path for {key}, skipping")
            continue

        gcs_uri = f"{GCS_PROCESSED}/{frames_lookup[key]}"
        try:
            frames = load_npy_gcs(gcs_uri)
        except Exception as e:
            warnings.warn(f"Could not load {gcs_uri}: {e}")
            continue

        if frames.ndim == 3:
            frames = frames.reshape(frames.shape[0], -1).astype(np.float32)
        elif frames.ndim == 1:
            frames = frames.reshape(-1, 1).astype(np.float32)
        else:
            frames = frames.astype(np.float32)

        tqdm.write(f"  [{receptor}] frames shape: {frames.shape}")

        for _, row in grp.iterrows():
            start  = int(row["chunk_start"])
            length = int(row["chunk_length"])
            chunk  = frames[start : start + length]

            ev = fit_tica_eigenvalues(chunk, lag, n_ev)
            ev = np.concatenate([ev, np.full(max(0, n_ev - len(ev)), np.nan)])[:n_ev]

            rows.append({
                "traj_id":      row["traj_id"],
                "receptor":     receptor,
                "rep":          int(rep),
                "simID":        int(simID),
                "chunk_start":  start,
                "chunk_length": length,
                "y":            int(row["label"]),
                **{f"ev_{i}": ev[i] for i in range(n_ev)},
            })

    return pd.DataFrame(rows)


def load_eigenvalues(args):
    if args.load_ev and Path(args.load_ev).exists():
        print(f"Loading cached eigenvalues from {args.load_ev}")
        ev_df = pd.read_csv(args.load_ev)
        ev_df = ev_df.rename(columns={
            c: c.replace("tica_eigenvalue_", "ev_")
            for c in ev_df.columns if c.startswith("tica_eigenvalue_")
        })
        if "traj_id" not in ev_df.columns:
            ev_df["traj_id"] = ev_df.apply(
                lambda r: f"{r['receptor']}_rep{int(r['rep'])}_{int(r['simID'])}", axis=1
            )
        if "y" not in ev_df.columns:
            ev_df["y"] = ev_df["label"]
        return ev_df

    print(f"Loading bootstrap metadata from {BOOTSTRAP_META}")
    bootstrap_meta = pd.read_csv(BOOTSTRAP_META)

    print("Loading base dataset from GCS")
    with fsspec.open(f"{GCS_PROCESSED}/base_dataset.csv", "r") as f:
        base_df = pd.read_csv(f)

    print(f"\nComputing TICA eigenvalues (lag={args.lag}, n_ev={args.n_ev})")
    print(f"  {len(bootstrap_meta)} chunks across {bootstrap_meta['traj_id'].nunique()} trajectories\n")

    ev_df = compute_chunk_eigenvalues(bootstrap_meta, base_df, args.lag, args.n_ev)
    print(f"\nComputed eigenvalues for {len(ev_df)} chunks ({ev_df['traj_id'].nunique()} trajectories)")

    if args.save_ev:
        ev_df.to_csv(args.save_ev, index=False)
        print(f"Saved eigenvalues -> {args.save_ev}")

    return ev_df


def train_test_split(ev_df):
    train_ids = set(pd.read_csv(TRAIN_CSV)["traj_id"].str.replace("~", "_"))
    test_ids  = set(pd.read_csv(TEST_CSV) ["traj_id"].str.replace("~", "_"))
    ev_df = ev_df.copy()
    ev_df["_tid_norm"] = ev_df["traj_id"].str.replace("~", "_")
    train_ev = ev_df[ev_df["_tid_norm"].isin(train_ids)]
    test_ev  = ev_df[ev_df["_tid_norm"].isin(test_ids)]
    return train_ev, test_ev


def topk_mean(ev_mat, k):
    return np.sort(ev_mat, axis=1)[:, ::-1][:, :k].mean(axis=1)


def aggregate_to_traj(chunk_df, score_fn, ev_cols):
    df = chunk_df.copy()
    df["score"] = score_fn(df[ev_cols].values)
    return df.groupby("traj_id").agg(score=("score", "mean"), y=("y", "first")).reset_index()


def find_best_threshold(scores, y, n_steps=300):
    best_t, best_f1 = scores.min(), -1.0
    for t in np.linspace(scores.min(), scores.max(), n_steps):
        f1 = f1_score(y, (scores >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def chunk_metrics(scores, y, threshold):
    pred = (scores >= threshold).astype(int)
    auc  = roc_auc_score(y, scores) if len(np.unique(y)) > 1 else float("nan")
    return dict(
        auc    = auc,
        acc    = (pred == y).mean(),
        prec_s = precision_score(y, pred, pos_label=0, zero_division=0),
        rec_s  = recall_score   (y, pred, pos_label=0, zero_division=0),
        prec_m = precision_score(y, pred, pos_label=1, zero_division=0),
        rec_m  = recall_score   (y, pred, pos_label=1, zero_division=0),
    )


def sweep_metrics(train_ev, test_ev, ev_cols, n_ev_list):
    train_rows, test_rows = [], []
    for n in n_ev_list:
        cols = ev_cols[:n]
        fn   = lambda ev, n=n: topk_mean(ev, n)
        tr   = aggregate_to_traj(train_ev, fn, cols)
        te   = aggregate_to_traj(test_ev,  fn, cols)
        t, _ = find_best_threshold(tr["score"].values, tr["y"].values.astype(int))
        train_rows.append({"n_ev": n, **chunk_metrics(tr["score"].values, tr["y"].values.astype(int), t)})
        test_rows .append({"n_ev": n, **chunk_metrics(te["score"].values, te["y"].values.astype(int), t)})
    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


def print_results_table(train_df, test_df):
    hdr = (f"{'Split':<6}  {'# Eig':>6}  {'AUC':>6}  {'Accuracy':>8}  "
           f"{'P(S)':>6}  {'R(S)':>6}  {'P(M)':>6}  {'R(M)':>6}")
    sep = "-" * len(hdr)
    print(f"\n{hdr}\n{sep}")
    for label, df in [("Train", train_df), ("Test", test_df)]:
        for _, r in df.iterrows():
            print(f"{label:<6}  {int(r.n_ev):>6}  {r.auc:>6.4f}  {r.acc:>8.4f}  "
                  f"{r.prec_s:>6.2f}  {r.rec_s:>6.2f}  {r.prec_m:>6.2f}  {r.rec_m:>6.2f}")
        print(sep)


def print_latex_table(train_df, test_df):
    print(r"""
\begin{table}[h]
\centering
\caption{TICA Eigenvalue Thresholding Performance (threshold fit on train set)}
\label{tab:tica_combined}
\begin{tabular}{l c c c c c c c}
\toprule
\textbf{Split} & \textbf{\# Eig.} & \textbf{AUC} & \textbf{Accuracy}
  & \textbf{Prec. (S)} & \textbf{Rec. (S)}
  & \textbf{Prec. (M)} & \textbf{Rec. (M)} \\
\midrule""")
    for label, df in [("Train", train_df), ("Test", test_df)]:
        for i, (_, r) in enumerate(df.iterrows()):
            split_cell = f"\\multirow{{{len(df)}}}{{*}}{{{label}}}" if i == 0 else ""
            print(f"{split_cell} & {int(r.n_ev)} & {r.auc:.4f} & {r.acc:.4f} & "
                  f"{r.prec_s:.2f} & {r.rec_s:.2f} & "
                  f"{r.prec_m:.2f} & {r.rec_m:.2f} \\\\")
        print(r"\midrule")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")


def make_plot(ev_df, ev_cols, all_scores, all_y, best_t, auc, f1, args):
    ev_mat = ev_df[ev_cols].values
    mask0, mask1 = ev_df["y"].values == 0, ev_df["y"].values == 1
    colors = {"single": "#4878d0", "multi": "#ee854a"}
    ncols  = min(ev_mat.shape[1], 4)

    _, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8))
    if ncols == 1:
        axes = axes.reshape(2, 1)

    for i in range(ncols):
        ax = axes[0, i]
        for lbl, mask, c in [("single", mask0, colors["single"]), ("multi", mask1, colors["multi"])]:
            vals = ev_mat[mask, i]
            ax.hist(vals[np.isfinite(vals)], bins=30, alpha=0.6, color=c, label=lbl, density=True)
        ax.set_title(rf"$\lambda_{{{i+1}}}$")
        ax.set_xlabel("eigenvalue")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)

    bins = np.linspace(all_scores.min(), all_scores.max(), 30)
    ax = axes[1, 0]
    ax.hist(all_scores[all_y == 0], bins=bins, alpha=0.6, color=colors["single"], label="single", density=True)
    ax.hist(all_scores[all_y == 1], bins=bins, alpha=0.6, color=colors["multi"],  label="multi",  density=True)
    ax.axvline(best_t, color="black", linestyle="--", linewidth=1.5, label=f"threshold={best_t:.3f}")
    ax.set_title("traj-level score: mean of top-k eigenvalues")
    ax.set_xlabel("score"); ax.set_ylabel("density"); ax.legend(fontsize=8)

    if ncols > 1:
        ax = axes[1, 1]
        ts  = np.linspace(all_scores.min(), all_scores.max(), 300)
        f1s = [f1_score(all_y, (all_scores >= t).astype(int), average="macro", zero_division=0) for t in ts]
        ax.plot(ts, f1s, color="#6acc65")
        ax.axvline(best_t, color="black", linestyle="--", linewidth=1.5, label=f"best={best_t:.3f}")
        ax.set_title("macro-F1 vs threshold")
        ax.set_xlabel("threshold"); ax.set_ylabel("macro-F1"); ax.legend(fontsize=8)

    if ncols > 2:
        ax  = axes[1, 2]
        rng = np.random.default_rng(0)
        jitter = rng.uniform(-0.15, 0.15, len(all_y))
        for cls, lbl, c in [(0, "single", colors["single"]), (1, "multi", colors["multi"])]:
            idx = all_y == cls
            ax.scatter(all_scores[idx], jitter[idx] + cls, alpha=0.5, s=25, color=c, label=lbl)
        ax.axvline(best_t, color="black", linestyle="--", linewidth=1.5)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["single", "multi"])
        ax.set_xlabel("score"); ax.set_title("per-trajectory scores"); ax.legend(fontsize=8)

    for col in range(3 if ncols > 3 else ncols, ncols):
        axes[1, col].set_visible(False)

    plt.suptitle(
        f"TICA eigenvalue classifier  |  lag={args.lag}  n_ev={args.n_ev}  AUC={auc:.3f}  F1={f1:.3f}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(args.plot, dpi=150)
    print(f"\nPlot saved -> {args.plot}")
    try:
        plt.show()
    except Exception:
        pass


def run(args):
    ev_df   = load_eigenvalues(args)
    ev_cols = [c for c in ev_df.columns if c.startswith("ev_")]
    ev_df   = ev_df.dropna(subset=ev_cols)

    print(f"\nChunks loaded: {len(ev_df)} from {ev_df['traj_id'].nunique()} trajectories")

    train_ev, test_ev = train_test_split(ev_df)
    print(
        f"Train: {train_ev['traj_id'].nunique()} trajs  "
        f"({(train_ev.groupby('traj_id')['y'].first()==0).sum()} single, "
        f"{(train_ev.groupby('traj_id')['y'].first()==1).sum()} multi)"
    )
    print(
        f"Test:  {test_ev['traj_id'].nunique()} trajs  "
        f"({(test_ev.groupby('traj_id')['y'].first()==0).sum()} single, "
        f"{(test_ev.groupby('traj_id')['y'].first()==1).sum()} multi)"
    )

    n_ev_list = [n for n in args.n_ev_list if n <= len(ev_cols)]
    train_df, test_df = sweep_metrics(train_ev, test_ev, ev_cols, n_ev_list)
    print_results_table(train_df, test_df)

    if args.latex:
        print_latex_table(train_df, test_df)

    if args.plot:
        sel_n  = args.n_ev if args.n_ev in n_ev_list else n_ev_list[-1]
        cols   = ev_cols[:sel_n]
        tr_sel = aggregate_to_traj(train_ev, lambda ev, n=sel_n: topk_mean(ev, n), cols)
        best_t, _ = find_best_threshold(tr_sel["score"].values, tr_sel["y"].values.astype(int))
        scores = tr_sel["score"].values
        y      = tr_sel["y"].values.astype(int)
        auc    = float(train_df[train_df["n_ev"] == sel_n]["auc"].values[0])
        f1     = f1_score((scores >= best_t).astype(int), y, average="macro", zero_division=0)
        make_plot(ev_df, ev_cols, scores, y, best_t, auc, f1, args)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--lag",       type=int, default=10,             help="TICA lag time in frames")
    p.add_argument("--n-ev",      type=int, default=2,              help="Max eigenvalues to compute per chunk")
    p.add_argument("--n-ev-list", type=int, nargs="+", default=[1, 2, 3, 5, 10], help="n_ev values to sweep")
    p.add_argument("--save-ev",   default=None,                     help="Save computed eigenvalues to CSV")
    p.add_argument("--load-ev",   default=None,                     help="Load precomputed eigenvalues from CSV")
    p.add_argument("--latex",     action="store_true",              help="Print LaTeX table")
    p.add_argument("--plot",      default=None, metavar="OUT.png",  help="Save diagnostic plot to this path")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
