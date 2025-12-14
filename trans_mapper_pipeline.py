#!/usr/bin/env python
# transformer_mapper_pipeline.py
#
# Transformer autoencoder + KeplerMapper pipeline for the synthetic
# multichannel time series. 
# writes mapper_transf.json 

import json
from pathlib import Path
import math

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import kmapper as km
import umap

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from synthetic_ts_plot import generate_synthetic_dataset


# ---------------------------------------------------------------------
# Data loading + windowing  
# ---------------------------------------------------------------------
def load_or_generate(path: str = "synthetic_multidim_ts.csv") -> pd.DataFrame:
    """Load the synthetic CSV if it exists, otherwise generate + save it."""
    p = Path(path)
    if p.exists():
        print(f"[data] Loaded {path}")
        return pd.read_csv(path)

    print("[data] CSV not found, generating synthetic dataset...")
    df = generate_synthetic_dataset()
    df.to_csv(path, index=False)
    print(f"[data] Saved new synthetic dataset to {path}")
    return df


def build_windows_for_transformer(
    df: pd.DataFrame, window_size: int = 200, stride: int = 20
):
    """
    Slice the multichannel series into overlapping windows for the transformer
    and compute per-window labels/features used in the Mapper JSON.

    Assumes:
      - channel columns named ch0, ch1, ...
      - df["label"] : 0/1 global anomaly flag per timestep
      - df["type"]  : 0..5 anomaly type per timestep (0 = normal)
    """
    channels = [c for c in df.columns if c.startswith("ch")]
    data = df[channels].values              # (T, C)
    labels_global = df["label"].values      # 0/1
    labels_type = df["type"].values.astype(int)  # 0..5
    T, C = data.shape

    seq_windows = []
    win_frac = []
    win_type = []
    win_ch_std = []
    win_start_idx = []

    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        seg = data[start:end, :]               # (W, C)
        seg_label_glob = labels_global[start:end]
        seg_label_type = labels_type[start:end]

        seq_windows.append(seg)

        # anomaly fraction in this window
        frac = seg_label_glob.mean()
        win_frac.append(frac)

        # dominant anomaly type (1..5), ignoring 0 = normal
        unique, counts = np.unique(seg_label_type, return_counts=True)
        if len(unique) == 1 and unique[0] == 0:
            dom_type = 0
        else:
            mask = unique != 0
            if mask.any():
                uniq_nz = unique[mask]
                counts_nz = counts[mask]
                dom_type = int(uniq_nz[np.argmax(counts_nz)])
            else:
                dom_type = 0
        win_type.append(dom_type)

        # channel-wise std in this window
        win_ch_std.append(seg.std(axis=0))
        win_start_idx.append(start)

    seq_windows = np.stack(seq_windows, axis=0)   # (N, W, C)
    win_frac = np.array(win_frac)
    win_type = np.array(win_type, dtype=int)
    win_ch_std = np.stack(win_ch_std, axis=0)     # (N, C)
    win_start_idx = np.array(win_start_idx, dtype=int)

    print(
        f"[data] Windows: {seq_windows.shape[0]} x "
        f"{seq_windows.shape[1]} steps, {C} channels"
    )

    return (
        seq_windows,
        win_frac,
        win_type,
        win_ch_std,
        win_start_idx,
        channels,
    )

# ---------------------------------------------------------------------
# Transformer autoencoder
# ---------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Standard sine/cosine positional encoding (batch_first=True)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        """
        T = x.size(1)
        return x + self.pe[:, :T]


class TransformerAE(nn.Module):
    """
    Simple Transformer encoder used as a sequence autoencoder.
    Input:  (B, T, C)  (time-major windows, C channels)
    Output: recon: (B, T, C), latent: (B, latent_dim)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        latent_dim: int = 16,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.latent_dim = latent_dim

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # project encoder outputs back to input space
        self.output_proj = nn.Linear(d_model, input_dim)

        # compact latent from pooled encoder outputs
        self.latent_fc = nn.Linear(d_model, latent_dim)

    def encode(self, x: torch.Tensor):
        """
        x: (B, T, C)
        Returns:
            recon_seq: (B, T, C)
            latent:    (B, latent_dim)
        """
        h = self.input_proj(x)         # (B, T, d_model)
        h = self.pos_encoding(h)
        z_seq = self.encoder(h)        # (B, T, d_model)
        recon = self.output_proj(z_seq)

        # mean-pool over time for latent representation
        pooled = z_seq.mean(dim=1)     # (B, d_model)
        latent = self.latent_fc(pooled)
        return recon, latent

    def forward(self, x: torch.Tensor):
        recon, latent = self.encode(x)
        return recon, latent


def train_transformer_ae(
    seq_windows: np.ndarray,
    n_epochs: int = 10,
    batch_size: int = 64,
    d_model: int = 64,
    n_heads: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    latent_dim: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
):
    """
    Train a Transformer autoencoder on the windowed sequences.

    seq_windows: (N, W, C)
    Returns:
        latents : (N, latent_dim)
        scores  : (N,) reconstruction error per window (MSE)
    """
    N, W, C = seq_windows.shape

    # Standardise per-channel across all windows
    flat = seq_windows.reshape(-1, C)
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True) + 1e-8
    seq_norm = (seq_windows - mean) / std

    x_tensor = torch.tensor(seq_norm, dtype=torch.float32)  # (N, W, C)
    dataset = TensorDataset(x_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerAE(
        input_dim=C,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        latent_dim=latent_dim,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print(f"[transf] Training on {N} windows, device={device}")
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            opt.zero_grad()
            recon, _ = model(batch_x)
            loss = loss_fn(recon, batch_x)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch_x.size(0)

        avg_loss = total_loss / N
        print(f"[transf] Epoch {epoch:02d}/{n_epochs}, loss={avg_loss:.6f}")

    # Extract latent and reconstruction error per window
    model.eval()
    all_latents = []
    all_scores = []
    loader_eval = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch_x,) in loader_eval:
            batch_x = batch_x.to(device)
            recon, latent = model(batch_x)
            # per-sample MSE across (T, C)
            recon_err = ((recon - batch_x) ** 2).mean(dim=(1, 2))
            all_latents.append(latent.cpu().numpy())
            all_scores.append(recon_err.cpu().numpy())

    latents = np.concatenate(all_latents, axis=0)   # (N, latent_dim)
    scores = np.concatenate(all_scores, axis=0)     # (N,)

    print("[transf] Latent feature shape:", latents.shape)
    print(
        "[transf] Score stats: min={:.6f}, max={:.6f}".format(
            scores.min(), scores.max()
        )
    )

    return latents, scores


# ---------------------------------------------------------------------
# Mapper pipeline on Transformer features
# ---------------------------------------------------------------------
def run_transformer_mapper(
    df: pd.DataFrame,
    window_size: int = 200,
    stride: int = 20,
    json_out: str = "mapper_transf.json",
    n_cubes: int = 30,
    overlap: float = 0.4,
):
    (
        seq_windows,
        win_frac,
        win_type,
        win_ch_std,
        win_starts,
        channels,
    ) = build_windows_for_transformer(df, window_size=window_size, stride=stride)

    N, W, C = seq_windows.shape

    # Train Transformer AE and get latent features + scores
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latents, scores = train_transformer_ae(
        seq_windows,
        n_epochs=15,
        batch_size=64,
        d_model=64,
        n_heads=4,
        num_layers=2,
        dim_feedforward=128,
        latent_dim=16,
        lr=1e-3,
        device=device,
    )

    # Standardise latent features for Mapper
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(latents)

    # PCA lens for Mapper
    mapper = km.KeplerMapper(verbose=1)
    lens_pca = mapper.fit_transform(
        X_scaled,
        projection=PCA(n_components=2),
        scaler=None,
    )

    # UMAP embedding for viewer lens toggle
    print("[mapper] Computing UMAP embedding on Transformer latents...")
    lens_umap = umap.UMAP(
        n_components=2,
        random_state=42,
    ).fit_transform(X_scaled)

    # Cover + clustering
    cover = km.Cover(n_cubes=n_cubes, perc_overlap=overlap)
    clusterer = KMeans(n_clusters=3, random_state=42)

    print("[mapper] Building Mapper graph from Transformer features...")
    graph = mapper.map(
        lens_pca,
        X_scaled,
        cover=cover,
        clusterer=clusterer,
    )

    print(
        f"[mapper] Graph has {len(graph['nodes'])} raw nodes and "
        f"{len(graph['links'])} raw links"
    )

    # Map node keys like "cube0_cluster0" -> integer ids
    node_keys = list(graph["nodes"].keys())
    key_to_int = {k: i for i, k in enumerate(node_keys)}

    nodes_out = []
    members_by_id = []

    for key in node_keys:
        node_id = key_to_int[key]
        members = np.array(graph["nodes"][key], dtype=int)
        members_by_id.append(set(members.tolist()))
        size = len(members)

        # Node coordinates = mean of member windows in each lens
        pca_coords = lens_pca[members].mean(axis=0).tolist()
        umap_coords = lens_umap[members].mean(axis=0).tolist()

        # average anomaly fraction across windows (ground truth)
        anomaly_fraction = float(win_frac[members].mean())

        # type histogram across windows (0..5)
        types_node = win_type[members].astype(int)
        type_counts = np.bincount(types_node, minlength=6)
        type_hist = type_counts.tolist()

        # dominant non-normal type (else 0)
        total = type_counts.sum()
        anomalous = total - type_counts[0]
        if anomalous > 0:
            dominant_type = int(1 + np.argmax(type_counts[1:]))
        else:
            dominant_type = 0

        # per-node channel std + dominant channel (ground truth)
        ch_std_node = win_ch_std[members].mean(axis=0)
        dominant_channel = int(np.argmax(ch_std_node))
        channel_std_list = ch_std_node.tolist()

        # birth_step = earliest window start index in this node
        birth_step = int(win_starts[members].min())

        # model-based mean reconstruction error for this node
        node_score = float(scores[members].mean())

        nodes_out.append(
            {
                "id": node_id,
                "members": members.tolist(),
                "size": size,
                "pca": pca_coords,
                "umap": umap_coords,
                "anomaly_fraction": anomaly_fraction,
                "dominant_type": dominant_type,
                "type_hist": type_hist,
                "dominant_channel": dominant_channel,
                "channel_std": channel_std_list,
                "birth_step": birth_step,
                "mean_score": node_score,
            }
        )

    # Rebuild links from membership overlaps so they are simple
    links_out = []
    n_nodes = len(nodes_out)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if members_by_id[i] & members_by_id[j]:
                links_out.append({"source": i, "target": j})

    print(f"[mapper] Rebuilt {len(links_out)} links from membership overlaps")

    meta = {
        "model": "Transformer-AE",
        "window_size": window_size,
        "stride": stride,
        "n_channels": C,
        "channels": channels,
        "lens": ["PCA", "UMAP"],
        "color_modes": [
            "anomaly_fraction",
            "anomaly_type",
            "dominant_channel",
            "reconstruction_error",
        ],
        "n_nodes": len(nodes_out),
        "n_links": len(links_out),
    }

    mapper_json = {
        "nodes": nodes_out,
        "links": links_out,
        "meta": meta,
    }

    with open(json_out, "w") as f:
        json.dump(mapper_json, f, indent=2)

    print(f"[mapper] Saved Transformer Mapper JSON to {json_out}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    df = load_or_generate("synthetic_multidim_ts.csv")
    run_transformer_mapper(df)

