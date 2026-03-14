# CS229: Early Prediction of Multi-State Behavior in MD Simulations
### Stanford CS229 Final Project | Winter 2026

This repository contains the code for predicting exhibition of protein conformational transitions in GPCR molecular dynamics (MD) trajectories. We evaluate handcrafted features versus pretrained structural embeddings (ESM-IF) using temporal architectures including TCN, Mamba, and Transformers.

## Team Members
- **Ellie Lin** (ellielin@stanford.edu)
- **Ion Martinis** (ion.martinis@stanford.edu)  
- **Vasanti Wall-Persad** (persav@stanford.edu)

## Project Overview

Our pipeline processes raw MD trajectories to identify early structural signals that predict whether a system will explore multiple metastable states. The most promising result focuses on using attention-pooled ESM-IF embeddings to capture family-specific structural signatures.

---

## Quick Start

### 1. Environment Setup 
```bash
# Clone the repository
git clone https://github.com/VasantiP/cs229_submission.git
cd cs229_submission

# Set up environment
conda env create -f environment.yml
conda activate cs229_md_project

# GPU setup: pip install
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU-only: pip install
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. Data Access (GCS Fuse)

All processed features and metadata are stored in Google Cloud Storage. Mount the bucket locally:

**On Google Cloud VM (Vertex AI Workbench):**
gcsfuse is pre-installed. Just authenticate and mount:
```bash
gcloud auth application-default login  # if not already authenticated
mkdir ./data/gcs_mount # if does not exist
gcsfuse --implicit-dirs cs229_public ./data/gcs_mount
```

**On Linux (local development):**
```bash
# Install gcsfuse (requires sudo - uses your local user password)
export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install gcsfuse

# Authenticate and mount
gcloud auth application-default login # if not already authenticated
mkdir ./data/gcs_mount  # if does not exist
gcsfuse --implicit-dirs cs229_public ./data/gcs_mount
```

**Verify mount:**
```bash
ls ./data/gcs_mount  # directory should not be empty
```

---

## Project Structure

```
cs229_submission/
├── data/                      # Data directory
│   ├── metadata/              # Test/train split CSVs and file mappings
│   ├── raw/                   # Raw GPCRmd trajectories (gitignored)
│   └── processed/             # Local processed features and labels (gitignored)
├── src/                       # Source code
│   ├── data/                  # Data loading and preprocessing
│   ├── features/              # Feature extraction (TICA, structural, ESMDance, ESM-IF)
│   ├── models/                # Model implementations (see below)
│   ├── evaluation/            # Evaluation metrics and utilities
│   └── utils/                 # Shared utilities
├── notebooks/                 # Jupyter notebooks for EDA, visualizations
├── scripts/                   # Standalone scripts
├── results/                   # Experimental results (gitignored)
├── environment.yml            # Conda environment specification
└── README.md                  # This file
```

---

## Reproducing Results

### Model Documentation

Each model has its own README with detailed instructions:

| Model | Documentation | Description |
|-------|---------------|-------------|
| **TCN** | [`src/models/TCN/README.md`](./src/models/TCN/README.md) | Temporal Convolutional Network on handcrafted features |
| **Mamba** | [`src/models/README.md`](./src/models/README.md) | State-space model on ESM-IF embeddings |
| **Transformer** | [`src/models/README.md`](./src/models/README.md) | Attention-based model with ALiBi (best results) |
| **DeepSets** | [`src/models/README.md`](./src/models/README.md) | Permutation-invariant frame pooling baseline |
| **LogRegression** | [`src/models/README.md`](./src/models/README.md) | Logistic regression on summary features |

### Summary of Results

| Model | Features | Protein-Level AUROC | Family-Level AUROC |
|-------|----------|---------------------|-------------------|
| TCN | Scalar (3-dim) | 0.682 | 0.500 |
| TCN | TICA (5-dim) | 0.782 | 0.500 |
| TCN | Oracle (50%) | 0.862 | 0.811 |
| Mamba | ESM-IF (attention) | 0.751 | 0.612 |
| **Transformer** | **ESM-IF (ALiBi)** | **0.823** | 0.571 |

---

## Data Details

### Feature Sets

| Feature | Dim | Description | GCS Path |
|---------|-----|-------------|----------|
| Scalar | 3 | RMSD, Rg, TM3-TM6 distance | `data/features_50pct/scalar/` |
| TICA | 5 | Top-5 slow mode projections | `data/features_50pct/tica/projections/` |
| Combined | 58 | All handcrafted features | `data/features_50pct/combined_features/` |
| Oracle | 7 | TICA + cluster features (upper bound) | `data/features_50pct/oracle_features/` |
| ESM-IF | 512 | Per-residue structural embeddings | `data/raw_esmif_chunks/` |

### Train/Test Splits

**Protein-level holdout** (primary evaluation):
- 80% train / 20% test by protein identity
- Same family can appear in both splits
- Tests within-family generalization

**Family-level holdout** (secondary evaluation):
- Peptide, Adenosine, Serotonin families held out (~23% of data)
- Tests cross-family generalization
- All methods drop to chance under this setting

---

## External Resources
- GPCRmd: https://www.gpcrmd.org/
- MDAnalysis: https://www.mdanalysis.org/
- DeepTime: https://deeptime-ml.github.io/