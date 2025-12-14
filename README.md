cat << 'EOF' > README.md
# Uncovering Neural Networks using Mapper Graphs

> **Visualizing latent structure in time-series representation learning using Topological Data Analysis**

---

## Overview

This repository presents an **end-to-end framework** for:
1. generating **synthetic multivariate time-series with controlled anomalies**,  
2. learning **latent representations** using neural autoencoders, and  
3. analyzing and comparing these representations via **Kepler Mapper graphs**.

The goal is to **interpret and compare neural representations**—not via scalar metrics alone, but by studying their **topological organization**.

---

## Pipeline at a glance

Synthetic Time Series
↓
Sliding Window Segmentation
↓
Representation Learning
(LSTM-AE / DC-VAE / Transformer-AE)
↓
Latent Embeddings per Window
↓
Kepler Mapper Graph Construction
↓
Interactive Graph Visualization (D3)


---

## Models implemented

| Model | File | Representation |
|------|------|---------------|
| Synthetic baseline | `synthetic_mapper_pipeline.py` | Raw window features |
| LSTM Autoencoder | `lstm_mapper_pipeline.py` | LSTM latent vectors |
| DC-VAE | `dcvae_mapper_pipeline.py` | Convolutional VAE latent space |
| Transformer Autoencoder | `trans_mapper_pipeline.py` | Transformer latent embeddings |

Each model produces a **window-level embedding**, which is then used as input to the Mapper algorithm.

---

## Synthetic dataset

### Dataset
- File: `synthetic_multidim_ts.csv`
- Channels: `ch0 … ch{C−1}`
- Time-indexed multivariate sequence

### Labels
| Field | Description |
|-----|------------|
| `label` | Binary anomaly indicator (0 = normal, 1 = anomaly) |
| `type` | Anomaly class (0–5) |

### Anomaly types
| Type | Description |
|----|------------|
| 0 | Normal |
| 1 | Mean shift |
| 2 | Spike |
| 3 | Variance burst |
| 4 | Channel dropout |
| 5 | Regime change |

---

## Windowing strategy

- Fixed-length sliding windows
- Configurable window size and stride
- Each window is annotated with:
  - anomaly fraction
  - dominant anomaly type
  - dominant channel
  - reconstruction error (for learned models)

These window-level summaries are propagated to Mapper nodes.

---

## Mapper construction

- **Lens**: PCA (primary), UMAP (alternative layout)
- **Cover**: overlapping hypercubes
- **Clustering**: K-Means
- **Node attributes**:
  - node size
  - anomaly fraction
  - anomaly type distribution
  - dominant channel
  - reconstruction error statistics

---

## Interactive visualization

The D3-based viewer allows:
- switching between **model representations**
- toggling **PCA vs UMAP layouts**
- coloring nodes by:
  - anomaly fraction
  - dominant anomaly type
  - dominant channel
  - reconstruction error
- inspecting node composition interactively

### Launch locally
```bash
python -m http.server 8000

