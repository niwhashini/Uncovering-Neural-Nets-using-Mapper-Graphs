# Uncovering-Neural-Nets-using-Mapper-Graphs
=======
Synthetic Time-Series → Representation Learning → Mapper Graph Viewer

This repo generates a synthetic multichannel time series with labeled anomaly segments, trains multiple unsupervised representation models (LSTM-AE, DC-VAE, Transformer-AE), and visualizes the resulting KeplerMapper graphs in an interactive D3 viewer.

The viewer supports switching between:

Synthetic (raw-window) features

LSTM latent features

DC-VAE latent features

Transformer latent features

…and toggling between PCA vs UMAP layout lenses, plus multiple node color modes.

What you get

A synthetic dataset: synthetic_multidim_ts.csv

Channels: ch0..ch{C-1}

Labels:

label ∈ {0,1} (anomaly vs normal)

type ∈ {0..5} (0=normal; 1..5 anomaly class)

Mapper JSON outputs (consumed by the viewer):

synthetic_mapper_pca_umap.json

mapper_lstm.json

mapper_vae.json

mapper_transf.json

A standalone interactive viewer:

viewer.html + viewer.js 

viewer

 

viewer

1) Synthetic dataset generation
Anomaly types (label type)

0: Normal

1: Mean shift (persistent offset)

2: Spike (single-point impulse)

3: Variance burst (high-noise interval)

4: Dropout (flatline / zeros on one channel)

5: Regime switch (pattern change on one channel)

Generate dataset
python synthetic_ts_plot.py


This writes:

synthetic_multidim_ts.csv


If the CSV already exists, the model pipelines will reuse it.

2) Windowing + feature construction (shared idea across pipelines)

All pipelines segment the multichannel series into overlapping windows:

window_size (default: 200)

stride (default: 20)

Each window has:

Ground-truth summary labels (for visualization only):

anomaly_fraction: fraction of anomalous timesteps within the window

dominant_type: most frequent anomaly type in the window (ignoring type=0)

dominant_channel: channel with highest standard deviation in that window

A learned representation (depends on model):

Synthetic baseline: flattened raw window

LSTM-AE / DC-VAE / Transformer-AE: latent embedding per window

Optional: reconstruction error score per window (used as a color mode)

3) AI models used
A) LSTM Autoencoder (lstm_mapper_pipeline.py)

Encoder: LSTM → last hidden state → linear projection to latent

Decoder: unfolds a sequence from latent state → linear output to channels

Output per window:

latent vector z

reconstruction MSE (used as “mean_score” in Mapper nodes)

Run:

python lstm_mapper_pipeline.py --out mapper_lstm.json

B) DC-VAE (1D Convolutional VAE) (dcvae_mapper_pipeline.py)

Encoder: Conv1D stack → flattened → μ, logσ²

Latent: reparameterization trick

Decoder: ConvTranspose1D stack back to sequence

Output per window:

latent μ (embedding)

reconstruction error (MSE)

Run:

python dcvae_mapper_pipeline.py --out mapper_vae.json

C) Transformer Autoencoder (trans_mapper_pipeline.py)

Input projection to d_model + sinusoidal positional encoding

TransformerEncoder stacks

Reconstruction from encoder outputs

Latent: mean pooling over time → linear projection

Output per window:

latent embedding

reconstruction MSE

Run:

python trans_mapper_pipeline.py --out mapper_transf.json

4) Mapper graph construction

All pipelines build a KeplerMapper graph from window-level features:

Lens: PCA (2D) for Mapper construction

Additional lens: UMAP (2D) for alternative layout in viewer

Cover: n_cubes and overlap

Clusterer: KMeans (k=3)

Each Mapper node stores:

members: which windows belong to the node

size: member count

pca / umap: mean coordinates of member windows

anomaly_fraction, dominant_type, type_hist

dominant_channel, channel_std

birth_step: earliest window start index

mean_score (for model pipelines): mean reconstruction error across members

5) Run end-to-end
Install
pip install -r requirements.txt

Generate dataset (optional)
python synthetic_ts_plot.py

Generate all Mapper JSONs
python synthetic_mapper_pipeline.py --out synthetic_mapper_pca_umap.json
python lstm_mapper_pipeline.py --out mapper_lstm.json
python dcvae_mapper_pipeline.py --out mapper_vae.json
python trans_mapper_pipeline.py --out mapper_transf.json

View locally

Because the viewer fetches JSON files, run a local server:

python -m http.server 8000


Open:

http://localhost:8000/viewer.html

