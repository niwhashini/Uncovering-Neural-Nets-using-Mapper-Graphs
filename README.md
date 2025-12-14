
# Uncovering Neural Networks using Mapper Graphs

> Visualizing latent structure across **neural layers and architectures** using Topological Data Analysis

---

## Overview

This repository presents a research framework for **interpreting neural networks using Mapper graphs**.  
The project is organized into **two complementary parts**, both aimed at uncovering structure in learned representations beyond scalar performance metrics.

The central idea is to study **how information is organized topologically**—across *layers* and *architectures*—using Kepler Mapper.

This repository includes an additional exploratory study on the Mapper algorithm, applying Kepler Mapper to **COVID-19 data**, along with a custom **D3-based visualization** pipeline.

---

## Project structure

### Part 0 — Covid-19 Dataset Visualization
To study the behavior of the Mapper algorithm in practice, we apply Kepler Mapper to COVID-19 case data from 9 U.S. states over a six-month period.  
The resulting Mapper graph is visualized using D3 to enable interactive inspection and trend analysis.

#### D3-based Viewer Features:
- Toggle node coloring by:
  - dominant state
  - average number of confirmed cases
- Interactive pie chart showing node composition
- Side panel displaying detailed node statistics

---

### Part I — QFormer: Mapper graphs through neural network layers

This part focuses on **intra-model analysis**.

Given a single neural network we:
- extract activations from **intermediate layers**,
- construct Mapper graphs at each layer, and
- study how representation topology evolves **depth-wise**.

**Goal:**  
Understand how neural networks progressively reshape information as it flows through layers.


**Note:**  
QFormer query tensors from all layers are excluded from this repository due to their large size (~2.07 GB).

---

### Part II — Time Series: Mapper graphs across architectures

This part focuses on **inter-model comparison** for time-series representation learning.

We:
1. generate **synthetic multivariate time series with controlled anomalies**,  
2. learn latent representations using different architectures, and  
3. compare their Mapper graphs to reveal structural differences.

Implemented architectures:
- LSTM Autoencoder
- DC-VAE
- Transformer Autoencoder

**Goal:**  
Compare how different model families organize temporal patterns and anomalies in latent space.

## Pipeline at a glance

Synthetic Time Series -> 
Sliding Window Segmentation ->
Representation Learning
(LSTM-AE / DC-VAE / Transformer-AE) ->
Latent Embeddings per Window ->
Kepler Mapper Graph Construction ->
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

