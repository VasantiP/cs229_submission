# TCN — Temporal Convolutional Network

Temporal Convolutional Network for predicting multi-state vs. single-state MD trajectory behavior from handcrafted features.

## Overview

The TCN uses dilated causal convolutions to capture multi-scale temporal patterns in trajectory features. This model serves as our primary baseline for evaluating whether handcrafted structural features contain predictive signal.

**Key findings:**
- Handcrafted features (scalar, TICA, combined) achieve modest AUROC (0.56–0.78) under protein-level holdout
- All handcrafted features collapse to chance under family-level holdout
- Oracle features achieve 0.86+ AUROC, confirming the architecture can learn when signal exists

## Files

| File | Description |
|------|-------------|
| `tcn_model.py` | TCN architecture (residual blocks, attention pooling) |
| `train_tcn.py` | Training script with early stopping, metrics logging |
| `tcn_receptor_holdout_experiments.ipynb` | Main experiments notebook (protein-level holdout) |

## Quick Start

### Train on Scalar Features
```bash
python train_tcn.py \
    --train_csv ../../../data/metadata/splits/protein_level_train.csv \
    --test_csv ../../../data/metadata/splits/protein_level_test.csv \
    --npy_dir ../../../data/gcs_mount/data/features_50pct/scalar \
    --batch_size 1 \
    --epochs 200 \
    --patience 50 \
    --output_dir ../../../results/tcn_scalar
```

### Train on TICA Projections
```bash
python train_tcn.py \
    --train_csv ../../../data/metadata/splits/protein_level_train.csv \
    --test_csv ../../../data/metadata/splits/protein_level_test.csv \
    --npy_dir ../../../data/gcs_mount/data/features_50pct/tica/projections \
    --output_dir ../../../results/tcn_tica
```

### Train on Oracle Features (Upper Bound)
```bash
python train_tcn.py \
    --train_csv ../../../data/metadata/splits/protein_level_train.csv \
    --test_csv ../../../data/metadata/splits/protein_level_test.csv \
    --npy_dir ../../../data/gcs_mount/data/features_50pct/oracle_features \
    --output_dir ../../../results/tcn_oracle
```

## Model Architecture

```
TCN Residual Block:
    Input → Conv1d (dilated) → BatchNorm → ReLU → Dropout
          → Conv1d (dilated) → BatchNorm → ReLU → Dropout
          + Skip Connection (1x1 conv if channels change)
          → Output

Full Model:
    Input (B, F, T) → [Residual Block × 4] → Pooling → Linear → Sigmoid
```

**Architecture details:**
- Layers: 4 residual blocks with dilation factors [1, 2, 4, 8]
- Channels: 25 per layer (default)
- Kernel size: 3
- Receptive field: 61 frames (with k=3, L=4)
- Pooling: mean (default), max, or attention
- Parameters: ~14k

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--tcn_channels` | `[25,25,25,25]` | Channels per layer |
| `--kernel_size` | 3 | Convolution kernel size |
| `--dropout` | 0.2 | Dropout rate |
| `--pooling` | `mean` | Temporal pooling (`mean`, `max`, `attention`) |
| `--use_batch_norm` | False | Add BatchNorm after convolutions |
| `--patience` | 20 | Early stopping patience |
| `--frac` | 1.0 | Fraction of trajectory to use |

## Expected Results (Protein-Level Holdout)

| Feature | Dim | AUROC | Balanced Acc | Behavior |
|---------|-----|-------|--------------|----------|
| Scalar | 3 | 0.682 | 0.50 | Degenerate |
| TICA projections | 5 | 0.782 | 0.58 | Modest signal |
| Combined | 58 | 0.564 | 0.50 | Degenerate |
| Oracle (50%) | 7 | 0.862 | 0.71 | Real signal |
| Oracle (90%) | 7 | 0.974 | 0.86 | Real signal |

**"Degenerate"** = model predicts all samples as one class (balanced accuracy ≈ 0.50)

## Output Structure

```
results/tcn_<experiment>/tcn_<timestamp>/
├── best_model.pt          # Best validation AUROC checkpoint
├── config.json            # Hyperparameters
├── training_history.csv   # Per-epoch metrics
└── test_predictions.csv   # Final predictions with probabilities
```

## Reproducing Paper Results

The `tcn_receptor_holdout_experiments.ipynb` notebook contains all experiments from Table 5 of the paper. Run all cells to reproduce:

1. Scalar features (3 seeds)
2. TICA projections (3 seeds)
3. Combined features (3 seeds)
4. Oracle features at 10%, 50%, 90% trajectory fractions

Results are aggregated with mean ± std across seeds.