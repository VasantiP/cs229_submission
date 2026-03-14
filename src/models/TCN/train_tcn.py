"""
Train TCN for multistate prediction (cross-protein generalization).

Usage:
    python train_tcn.py --batch_size 32 --epochs 100 --lr 0.001

    # With architectural improvements:
    python train_tcn.py --use_batch_norm --pooling attention --tcn_channels 32 32 32 32
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from tcn_model import TCN, print_model_summary


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class TemporalTrajectoryDataset(Dataset):
    """Load bootstrapped trajectory chunks from NPY files."""

    def __init__(self, metadata_csv, npy_dir, feature_cols=None, frac=1.0):
        """
        Args:
            metadata_csv: Path to train_chunks.csv or test_chunks.csv
            npy_dir: Directory containing NPY files
            feature_cols: Indices of features to use (None = use all)
            frac: Fraction of trajectory to use
        """
        self.metadata = pd.read_csv(metadata_csv)
        self.npy_dir = Path(npy_dir)
        self.feature_cols = feature_cols
        self.frac = frac

        print(f"Loaded {len(self.metadata)} samples")
        print(f"  Multi: {self.metadata['y'].sum()} ({self.metadata['y'].mean()*100:.1f}%)")
        print(f"  Single: {(self.metadata['y']==0).sum()} ({(1-self.metadata['y'].mean())*100:.1f}%)")
        print(f"  Using frac={self.frac} of each trajectory")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        # Load NPY File
        npy_path = self.npy_dir / row['chunk_file']

        # Handle missing files gracefully
        if not npy_path.is_file():
            return None

        data = np.load(npy_path)

        n_use = max(1, int(len(data) * self.frac))
        data = data[:n_use]

        # Select features if specified
        if self.feature_cols is not None:
            data = data[:, self.feature_cols]

        # Handle NaNs
        data = np.nan_to_num(data, nan=0.0)
        # Transpose to (n_features, n_frames) for Conv1d
        features = torch.tensor(data.T, dtype=torch.float32)

        # Label
        label = torch.tensor(row['y'], dtype=torch.float32)
        return features, label

# ══════════════════════════════════════════════════════════════════════════════
# Collate (repeat-last-frame padding)
# ══════════════════════════════════════════════════════════════════════════════

def pad_trunc_collate(batch):
    """
    Pad variable-length trajectories by repeating the last frame.
    Produces tensor of shape (B, F, T_max_in_batch).
    """
    # Filter out None samples (missing files) before collating
    # getting error: ValueError: not enough values to unpack (expected 2, got 0) 
    # Handle case when all samples in batch are None
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None 
    features, labels = zip(*batch)

    max_len = max(f.shape[1] for f in features)
    n_features = features[0].shape[0]
    B = len(features)

    X = torch.zeros(B, n_features, max_len)
    y = torch.stack(labels)

    for i, f in enumerate(features):
        T = f.shape[1]

        # Copy real frames
        X[i, :, :T] = f

        # Repeat last frame if shorter than max_len
        if T < max_len:
            last_frame = f[:, -1:].repeat(1, max_len - T)
            X[i, :, T:] = last_frame

    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# Training & Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, train_loader, criterion, optimizer, device, max_grad_norm=1.0):
    """Train for one epoch with gradient clipping."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    skip_batches = 0

    # Handle case when train_loader returns None, None due to all samples in batch being None
    if train_loader is None:
        skip_batches += 1
#        print("  -- WARNING: No valid samples in train_loader for this epoch")
        return 0, 0, 0, 0
    # Now getting error: features = features.to(device) AttributeError: 'NoneType' object has no attribute 'to'
    # but i thought we handled that with the check above? maybe need to handle it inside the loop as well? answer:

    for features, labels in train_loader:
        if features is None or labels is None:
            skip_batches += 1
#            print("  -- WARNING: Skipping batch with no valid samples")
            continue

        features = features.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        # Backward pass
        loss.backward()

        # Gradient clipping for training stability
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    avg_loss = total_loss / len(train_loader)
    preds_binary = (np.array(all_preds) > 0.5).astype(int)
    bal_acc = balanced_accuracy_score(all_labels, preds_binary)
    auroc = roc_auc_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_preds)

    return avg_loss, bal_acc, auroc, ap, skip_batches


def evaluate(model, test_loader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    avg_loss = total_loss / len(test_loader)
    preds_binary = (np.array(all_preds) > 0.5).astype(int)
    bal_acc = balanced_accuracy_score(all_labels, preds_binary)
    auroc = roc_auc_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, preds_binary)

    return avg_loss, bal_acc, auroc, ap, cm, all_preds, all_labels


# ══════════════════════════════════════════════════════════════════════════════
# Main Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    """Main training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"tcn_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save config
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = TemporalTrajectoryDataset(
        args.train_csv, args.npy_dir, feature_cols=args.feature_cols, frac=args.frac
    )
    test_dataset = TemporalTrajectoryDataset(
        args.test_csv, args.npy_dir, feature_cols=args.feature_cols, frac=args.frac
    )

    cuda_available = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=cuda_available,
        collate_fn=pad_trunc_collate
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=cuda_available,
        collate_fn=pad_trunc_collate
    )

    # Initialize model
    if args.feature_cols:
        n_features = len(args.feature_cols)
    else:
        sample_file = Path(args.npy_dir) / train_dataset.metadata.iloc[0]['chunk_file']
        n_features = np.load(sample_file).shape[1]
        print(f"  Auto-detected {n_features} features from data")

    print(f"\nInitializing TCN with {n_features} input features...")
    model = TCN(
        num_inputs=n_features,
        num_channels=args.tcn_channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        use_batch_norm=args.use_batch_norm,
        use_weight_norm=args.use_weight_norm,
        pooling=args.pooling,
    ).to(device)

    print_model_summary(model)

    if args.pos_weight > 0:
        pos_weight = torch.tensor(args.pos_weight).to(device)
    else:
        n_pos = train_dataset.metadata['y'].sum()
        n_neg = len(train_dataset.metadata) - n_pos
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
      
    print(f"\nUsing pos_weight={pos_weight} for imbalanced data")
    criterion = lambda pred, target: nn.functional.binary_cross_entropy(pred, target, weight=torch.where(target == 1, pos_weight, torch.ones_like(pos_weight)))
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    # Training loop
    print("\nStarting training...")
    best_auroc = 0
    patience_counter = 0
    history = []

    for epoch in range(args.epochs):
        # Train
        train_loss, train_bal_acc, train_auroc, train_ap, skipped_batches = train_epoch(
            model, train_loader, criterion, optimizer, device,
            max_grad_norm=args.max_grad_norm
        )
        # Evaluate
        test_loss, test_bal_acc, test_auroc, test_ap, cm, _, _ = evaluate(
            model, test_loader, criterion, device
        )

        scheduler.step(test_auroc)

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train: Loss={train_loss:.4f}, Bal Acc={train_bal_acc:.3f}, AUROC={train_auroc:.3f}, AP={train_ap:.3f}")
        print(f"  Test:  Loss={test_loss:.4f}, Bal Acc={test_bal_acc:.3f}, AUROC={test_auroc:.3f}, AP={test_ap:.3f}")
        if skipped_batches > 0:
            print(f"  Skipped batches due to no valid samples: {skipped_batches}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss, 'train_bal_acc': train_bal_acc, 'train_auroc': train_auroc, 'train_ap': train_ap,
            'test_loss': test_loss, 'test_bal_acc': test_bal_acc, 'test_auroc': test_auroc, 'test_ap': test_ap,
        })

        # Save best model
        if test_auroc > best_auroc:
            best_auroc = test_auroc
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"   New best AUROC: {best_auroc:.3f}")
        else:
            patience_counter += 1

        # Early stopping (patience-based)
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
            break

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)

    # Final evaluation on best model
    print("\n" + "=" * 80)
    print("FINAL EVALUATION (best model)")
    print("=" * 80)

    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))
    test_loss, test_bal_acc, test_auroc, test_ap, cm, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nTest Metrics:")
    print(f"  Balanced Accuracy: {test_bal_acc:.3f}")
    print(f"  AUROC: {test_auroc:.3f}")
    print(f"  Avg Precision: {test_ap:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    print("\nClassification Report:")
    print(classification_report(
        test_labels,
        (np.array(test_preds) > 0.5).astype(int),
        target_names=['single_state', 'multi_state']
    ))

    # Save predictions
    pred_df = pd.DataFrame({
        'true_label': test_labels,
        'pred_prob': test_preds,
        'pred_label': (np.array(test_preds) > 0.5).astype(int)
    })
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

    print(f"\n Results saved to {output_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TCN for multistate prediction")

    # Data
    parser.add_argument("--train_csv", type=str, default="data/temporal/train_chunks.csv")
    parser.add_argument("--test_csv", type=str, default="data/temporal/test_chunks.csv")
    parser.add_argument("--npy_dir", type=str, default="data/bootstrapped/npy_files/")
    parser.add_argument("--feature_cols", type=int, nargs='+', default=None,
                        help="Feature columns to use (0=RMSD, 1=Rg, 2=TM3-TM6)")
    parser.add_argument("--frac", type=float, default=1.0,
                        help="Fraction of trajectory to use (e.g., 0.5 for first half)")

    # Model architecture
    parser.add_argument("--tcn_channels", type=int, nargs='+', default=[25, 25, 25, 25])
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--use_batch_norm", action='store_true', default=False,
                        help="Add BatchNorm after each conv layer")
    parser.add_argument("--use_weight_norm", action='store_true', default=False,
                        help="Apply weight normalization to convolutions")
    parser.add_argument("--pooling", type=str, default='mean', choices=['mean', 'max', 'attention'],
                        help="Temporal pooling strategy")

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (0 to disable)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pos_weight", type=float, default=1.0,
                        help="Positive class weight for imbalanced data (0 to auto-calculate)")

    # Output
    parser.add_argument("--output_dir", type=str, default="results/")

    args = parser.parse_args()
    main(args)

