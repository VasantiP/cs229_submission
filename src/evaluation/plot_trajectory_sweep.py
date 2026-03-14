"""
Plot trajectory length sweep results for cross-protein TCN experiments.

Reads results from train_tcn.py output directories and generates:
  1. AUROC vs trajectory % (main figure)
  2. Summary table (CSV)

Supports MULTIPLE RUNS per (pct, feature_type) — plots the mean with
individual data points shown for reference.

Usage:
    # Point to individual result directories (can have duplicates per pct)
    python plot_trajectory_sweep.py \
        --results \
            "50,sanity_check,results/tcn_sanity_check_50pct/tcn_20260303_143046" \
            "50,sanity_check,results/tcn_sanity_check_50pct/tcn_20260303_143512" \
            "50,sanity_check,results/tcn_sanity_check_50pct/tcn_20260303_144808" \
            "60,sanity_check,results/tcn_sanity_check_60pct/tcn_20260304_233042" \
            "70,sanity_check,results/tcn_sanity_check_70pct/tcn_20260304_233807" \
            "80,sanity_check,results/tcn_sanity_check_80pct/tcn_20260304_234248" \
            "90,sanity_check,results/tcn_sanity_check_90pct/tcn_20260303_152556" \
            "100,sanity_check,results/tcn_sanity_check_100pct/tcn_20260304_235700"

    # Or use a CSV file
    python plot_trajectory_sweep.py --results_csv sweep_results.csv
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score


def load_run_metrics(result_dir):
    """Extract metrics from a train_tcn.py output directory."""
    result_dir = Path(result_dir)
    metrics = {}

    # Try config.json for settings
    config_path = result_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        metrics['config'] = config

    # Try test_predictions.csv for final metrics (most reliable)
    pred_path = result_dir / "test_predictions.csv"
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        labels = pred_df['true_label'].values
        probs = pred_df['pred_prob'].values
        preds_binary = (probs > 0.5).astype(int)

        if len(np.unique(labels)) >= 2:
            metrics['auroc'] = roc_auc_score(labels, probs)
            metrics['ap'] = average_precision_score(labels, probs)
        else:
            metrics['auroc'] = float('nan')
            metrics['ap'] = float('nan')
        metrics['bal_acc'] = balanced_accuracy_score(labels, preds_binary)
        metrics['n_test'] = len(pred_df)
        metrics['n_multi'] = int(labels.sum())
        metrics['n_single'] = int((labels == 0).sum())
        metrics['tp'] = int(((preds_binary == 1) & (labels == 1)).sum())
        metrics['fp'] = int(((preds_binary == 1) & (labels == 0)).sum())
        metrics['fn'] = int(((preds_binary == 0) & (labels == 1)).sum())
        metrics['tn'] = int(((preds_binary == 0) & (labels == 0)).sum())

    # Try training_history.csv for best epoch info
    hist_path = result_dir / "training_history.csv"
    if hist_path.exists():
        hist_df = pd.read_csv(hist_path)
        if 'test_auroc' in hist_df.columns:
            best_idx = hist_df['test_auroc'].idxmax()
            metrics['best_epoch'] = int(hist_df.loc[best_idx, 'epoch'])
            metrics['best_train_auroc'] = float(hist_df.loc[best_idx, 'train_auroc'])
            metrics['total_epochs'] = len(hist_df)

    return metrics


def plot_sweep_with_bal_acc(summary_df, output_path, title=None):
    """
    Generate side-by-side AUROC, Avg Precision, and Balanced Accuracy plots.

    When multiple runs exist for the same (pct, feature_type), plots:
      - Mean value as the connected line with larger markers
      - Individual run values as smaller semi-transparent scatter points
      - Shaded region showing ±1 std (if 2+ runs)
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    colors = {
        'scalar': '#2196F3',
        'tica': '#FF9800',
        'sanity_check': '#4CAF50',
        'combined': '#9C27B0',
    }
    markers = {
        'scalar': 'o',
        'tica': 's',
        'sanity_check': '^',
        'combined': 'D',
    }
    labels_map = {
        'scalar': 'Scalar (RMSD, Rg, TM3-TM6)',
        'tica': 'TICA (5 components)',
        'sanity_check': 'Label Features (TICA + clustering)',
        'combined': 'Combined (scalar + graph + structural)',
    }

    metrics_axes = [
        (ax1, 'auroc', 'AUROC'),
        (ax2, 'ap', 'Average Precision'),
        (ax3, 'bal_acc', 'Balanced Accuracy'),
    ]

    for feat_type in summary_df['feature_type'].unique():
        subset = summary_df[summary_df['feature_type'] == feat_type]
        color = colors.get(feat_type, '#666666')
        marker = markers.get(feat_type, 'D')
        label = labels_map.get(feat_type, feat_type)

        # Group by pct to compute mean/std/individual values
        grouped = subset.groupby('pct')

        for ax, metric_col, metric_name in metrics_axes:
            if metric_col not in subset.columns:
                continue

            pcts = sorted(subset['pct'].unique())
            means = []
            stds = []

            for pct in pcts:
                group = grouped.get_group(pct)
                vals = group[metric_col].dropna().values

                means.append(np.mean(vals))
                stds.append(np.std(vals) if len(vals) > 1 else 0)

                # Plot individual points if multiple runs
                if len(vals) > 1:
                    ax.scatter([pct] * len(vals), vals,
                              color=color, marker=marker, s=25,
                              alpha=0.3, zorder=2, linewidths=0.5,
                              edgecolors=color)

            means = np.array(means)
            stds = np.array(stds)

            # Plot mean line
            ax.plot(pcts, means,
                    color=color, marker=marker, markersize=8,
                    linewidth=2, label=label, zorder=4)

            # Shaded std region if any pct has multiple runs
            if any(s > 0 for s in stds):
                ax.fill_between(pcts, means - stds, means + stds,
                                color=color, alpha=0.12, zorder=1)

    # Chance-level AP = class prevalence
    if 'n_multi' in summary_df.columns and 'n_test' in summary_df.columns:
        avg_prevalence = (summary_df['n_multi'] / summary_df['n_test']).mean()
    else:
        avg_prevalence = 0.29

    all_pcts = sorted(summary_df['pct'].unique())

    chance_levels = {
        'auroc': 0.5,
        'ap': avg_prevalence,
        'bal_acc': 0.5,
    }

    for ax, metric_col, metric_name in metrics_axes:
        chance = chance_levels.get(metric_col, 0.5)
        ax.axhline(y=chance, color='#999999', linestyle='--',
                   linewidth=1, label=f'Chance ({chance:.2f})', zorder=0)
        ax.set_xlabel('Trajectory Fraction (%)', fontsize=12)
        ax.set_ylabel(f'Test {metric_name}', fontsize=12)
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(all_pcts)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(title or 'Cross-Protein TCN: Trajectory Length Sweep',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close(fig)


def main(args):
    rows = []

    # ── Load results ──────────────────────────────────────────────────
    if args.results_csv:
        input_df = pd.read_csv(args.results_csv)

        if 'result_dir' in input_df.columns:
            for _, row in input_df.iterrows():
                pct = int(row['pct'])
                feat_type = row['feature_type']
                result_dir = row['result_dir']
                metrics = load_run_metrics(result_dir)
                metrics['pct'] = pct
                metrics['feature_type'] = feat_type
                metrics['result_dir'] = result_dir
                rows.append(metrics)
        else:
            for _, row in input_df.iterrows():
                rows.append(row.to_dict())

    elif args.results:
        for entry in args.results:
            parts = entry.split(',', 2)
            if len(parts) != 3:
                print(f"WARNING: Skipping '{entry}' — expected 'pct,feature_type,dir'")
                continue
            pct, feat_type, result_dir = int(parts[0]), parts[1].strip(), parts[2].strip()
            metrics = load_run_metrics(result_dir)
            metrics['pct'] = pct
            metrics['feature_type'] = feat_type
            metrics['result_dir'] = result_dir
            rows.append(metrics)

    else:
        print("ERROR: Provide --results or --results_csv")
        return

    if not rows:
        print("No results loaded.")
        return

    # ── Build summary ─────────────────────────────────────────────────
    summary_rows = []
    for r in rows:
        summary_rows.append({
            'pct': r.get('pct'),
            'feature_type': r.get('feature_type'),
            'auroc': r.get('auroc'),
            'ap': r.get('ap'),
            'bal_acc': r.get('bal_acc'),
            'n_test': r.get('n_test'),
            'n_multi': r.get('n_multi'),
            'n_single': r.get('n_single'),
            'tp': r.get('tp'),
            'fp': r.get('fp'),
            'fn': r.get('fn'),
            'tn': r.get('tn'),
            'best_epoch': r.get('best_epoch'),
            'best_train_auroc': r.get('best_train_auroc'),
            'total_epochs': r.get('total_epochs'),
            'result_dir': r.get('result_dir'),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ['feature_type', 'pct']).reset_index(drop=True)

    # ── Print individual run results ──────────────────────────────────
    print(f"\n{'='*80}")
    print("INDIVIDUAL RUN RESULTS")
    print(f"{'='*80}")
    display_cols = ['pct', 'feature_type', 'auroc', 'ap', 'bal_acc',
                    'n_test', 'tp', 'fp', 'fn', 'tn', 'best_epoch']
    available = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[available].to_string(index=False))

    # ── Print aggregated summary (mean ± std per pct/feature_type) ────
    print(f"\n{'='*80}")
    print("AGGREGATED RESULTS (mean ± std)")
    print(f"{'='*80}")
    agg_metrics = ['auroc', 'ap', 'bal_acc']
    agg_df = summary_df.groupby(['feature_type', 'pct'])[agg_metrics].agg(
        ['mean', 'std', 'count']).reset_index()

    # Flatten multi-level columns
    flat_cols = []
    for col in agg_df.columns:
        if isinstance(col, tuple):
            flat_cols.append('_'.join(col).rstrip('_'))
        else:
            flat_cols.append(col)
    agg_df.columns = flat_cols

    print(f"  {'Type':<15s} {'Pct':>4s} {'N':>3s} "
          f"{'AUROC':>14s} {'AP':>14s} {'Bal Acc':>14s}")
    print(f"  {'-'*15} {'-'*4} {'-'*3} {'-'*14} {'-'*14} {'-'*14}")

    for _, r in agg_df.iterrows():
        ft = r['feature_type']
        pct = int(r['pct'])
        n = int(r['auroc_count'])
        auroc_str = f"{r['auroc_mean']:.3f}"
        ap_str = f"{r['ap_mean']:.3f}"
        ba_str = f"{r['bal_acc_mean']:.3f}"
        if n > 1:
            auroc_str += f"±{r['auroc_std']:.3f}"
            ap_str += f"±{r['ap_std']:.3f}"
            ba_str += f"±{r['bal_acc_std']:.3f}"
        print(f"  {ft:<15s} {pct:>4d} {n:>3d} "
              f"{auroc_str:>14s} {ap_str:>14s} {ba_str:>14s}")

    # ── Save outputs ──────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV (all individual runs)
    csv_path = output_dir / "trajectory_sweep_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Aggregated CSV
    agg_csv_path = output_dir / "trajectory_sweep_aggregated.csv"
    agg_df.to_csv(agg_csv_path, index=False)
    print(f"Saved: {agg_csv_path}")

    # Plot
    plot_sweep_with_bal_acc(summary_df,
                           output_dir / "trajectory_sweep_auroc_balacc.png",
                           title=args.title)

    print(f"\n✓ All outputs in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot trajectory length sweep results")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--results", nargs='+',
        help="Space-separated entries: 'pct,feature_type,result_dir'")
    input_group.add_argument(
        "--results_csv", type=str,
        help="CSV with columns: pct, feature_type, result_dir")

    parser.add_argument("--output_dir", type=str,
                        default="results/trajectory_sweep")
    parser.add_argument("--title", type=str, default=None,
                        help="Custom plot title")

    args = parser.parse_args()
    main(args)