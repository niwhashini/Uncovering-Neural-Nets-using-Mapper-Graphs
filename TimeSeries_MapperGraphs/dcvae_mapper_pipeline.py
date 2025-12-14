#!/usr/bin/env python
# dcvae_mapper_pipeline.py
#
# DC-VAE Mapper pipeline for the synthetic multichannel time series.
# Mirrors run_lstm_mapper() and produces mapper_vae.json which plugs
# directly into viewer.html / viewer.js.

import json
from pathlib import Path

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

# If your synthetic generator lives elsewhere, adjust this import
# or comment it out and just load the CSV directly.
from synthetic_ts_plot import generate_synthetic_dataset


# ---------------------------------------------------------------------
# Data loading + windowing (same as LSTM pipeline)
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


def build_windows_for_dcvae(
    df: pd.DataFrame, window_size: int = 200, stride: int = 20
):
    """
    Same semantics as build_windows_for_lstm in lstm_mapper_pipeline.py:
      - channel columns named ch0, ch1, ...
      - df['label'] : 0/1 global anomaly flag per timestep
      - df['type']  : 0..5 anomaly type per timestep (0 = normal)

    Returns:
        seq_windows : (N, W, C)
        win_frac    : (N,) anomaly fraction per window
        win_type    : (N,) dominant anomaly type per window
        win_ch_std  : (N, C) channel-wise std per window
        win_starts  : (N,) window start index in the global series
        channels    : list[str] channel names
    """
    channels = [c for c in df.columns if c.startswith("ch")]
    data = df[channels].values             # (T, C)
    labels_global = df["label"].values     # 0/1
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
# DC-VAE model
# ---------------------------------------------------------------------

class DCVAE(nn.Module):
    """
    Simple 1D convolutional VAE for multichannel time series.
    Expects input of shape (B, C, T) where:
      C = number of channels
      T = window_size
    """

    def __init__(
        self,
        input_channels: int,
        latent_dim: int = 16,
        hidden_channels: int = 32,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim

        # Encoder: Conv1d stack → global pooling → latent
        self.enc_conv = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # We'll infer the flattened size at runtime with a dummy forward
        self.enc_mu = None
        self.enc_logvar = None

        # Decoder: linear → ConvTranspose1d stack back to (C, T)
        self.dec_fc = None
        self.dec_conv = None

    def build_latent_layers(self, enc_flat_dim: int):
        self.enc_mu = nn.Linear(enc_flat_dim, self.latent_dim)
        self.enc_logvar = nn.Linear(enc_flat_dim, self.latent_dim)

        self.dec_fc = nn.Linear(self.latent_dim, enc_flat_dim)

        # We don't know exact temporal length here, but we mirror enc_conv
        # channels and then upsample with ConvTranspose1d.
        hidden_channels = self.enc_conv[-2].out_channels  # last Conv1d out_channels

        self.dec_conv = nn.Sequential(
            nn.ConvTranspose1d(
                hidden_channels, hidden_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(
                hidden_channels, hidden_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(
                hidden_channels, self.input_channels,
                kernel_size=4, stride=2, padding=1
            ),
        )

    def encode(self, x):
        """
        x: (B, C, T)
        Returns: mu, logvar  (B, latent_dim)
        """
        h = self.enc_conv(x)  # (B, H, T')
        B = h.size(0)
        h_flat = torch.flatten(h, start_dim=1)  # (B, H*T')

        if self.enc_mu is None:
            # First call: build latent + decoder layers lazily
            enc_flat_dim = h_flat.size(1)
            self.build_latent_layers(enc_flat_dim)
            # Move to correct device
            self.enc_mu.to(h.device)
            self.enc_logvar.to(h.device)
            self.dec_fc.to(h.device)
            self.dec_conv.to(h.device)

        mu = self.enc_mu(h_flat)
        logvar = self.enc_logvar(h_flat)
        return mu, logvar, h.shape

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, enc_shape):
        """
        z: (B, latent_dim)
        enc_shape: shape of encoder feature map (B, H, T')
        Returns: x_hat (B, C, T)
        """
        B, H, T_down = enc_shape
        h_flat = self.dec_fc(z)            # (B, H*T')
        h = h_flat.view(B, H, T_down)      # (B, H, T')
        x_hat = self.dec_conv(h)           # (B, C, T_dec)

        return x_hat

    def forward(self, x):
        mu, logvar, enc_shape = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, enc_shape)
        return x_hat, mu, logvar, z


def vae_loss(x, x_hat, mu, logvar):
    """
    Standard VAE loss = reconstruction MSE + KL divergence.
    x, x_hat: (B, C, T)
    mu, logvar: (B, latent_dim)
    """
    recon = ((x_hat - x) ** 2).mean(dim=(1, 2))  # per-sample MSE
    # KL divergence between q(z|x) and N(0, I)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    loss = recon + 1e-3 * kl      # KL weight is small; tune if needed
    return loss.mean(), recon.detach()


def train_dc_vae(
    seq_windows: np.ndarray,
    n_epochs: int = 15,
    batch_size: int = 64,
    latent_dim: int = 16,
    hidden_channels: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
):
    """
    Train DC-VAE on windowed sequences.

    seq_windows: (N, W, C)
    Returns:
        latents : (N, latent_dim)  (mu of the posterior)
        scores  : (N,) reconstruction error per window (MSE)
    """
    N, W, C = seq_windows.shape

    # Standardise per-channel across all windows
    flat = seq_windows.reshape(-1, C)
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True) + 1e-8
    seq_norm = (seq_windows - mean) / std

    # DC-VAE expects (B, C, T)
    x_tensor = torch.tensor(seq_norm.transpose(0, 2, 1), dtype=torch.float32)  # (N, C, W)
    dataset = TensorDataset(x_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DCVAE(
        input_channels=C,
        latent_dim=latent_dim,
        hidden_channels=hidden_channels,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"[dcvae] Training on {N} windows, device={device}")
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            opt.zero_grad()
            x_hat, mu, logvar, _ = model(batch_x)
            loss, recon = vae_loss(batch_x, x_hat, mu, logvar)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch_x.size(0)
        avg_loss = total_loss / N
        print(f"[dcvae] Epoch {epoch:02d}/{n_epochs}, loss={avg_loss:.6f}")

    # Extract latent (mu) + reconstruction error for each window
    model.eval()
    all_latents = []
    all_scores = []
    loader_eval = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch_x,) in loader_eval:
            batch_x = batch_x.to(device)
            x_hat, mu, logvar, _ = model(batch_x)
            _, recon = vae_loss(batch_x, x_hat, mu, logvar)  # recon: (B,)
            all_latents.append(mu.cpu().numpy())
            all_scores.append(recon.cpu().numpy())

    latents = np.concatenate(all_latents, axis=0)  # (N, latent_dim)
    scores = np.concatenate(all_scores, axis=0)    # (N,)

    print("[dcvae] Latent feature shape:", latents.shape)
    print(
        "[dcvae] Score stats: min={:.6f}, max={:.6f}".format(
            scores.min(), scores.max()
        )
    )

    return latents, scores


# ---------------------------------------------------------------------
# Mapper pipeline on DC-VAE features (mirrors run_lstm_mapper)
# ---------------------------------------------------------------------

def run_dcvae_mapper(
    df: pd.DataFrame,
    window_size: int = 200,
    stride: int = 20,
    json_out: str = "mapper_vae.json",
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
    ) = build_windows_for_dcvae(df, window_size=window_size, stride=stride)

    N, W, C = seq_windows.shape

    # Train DC-VAE and get latent features + scores
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latents, scores = train_dc_vae(
        seq_windows,
        n_epochs=15,
        batch_size=64,
        latent_dim=16,
        hidden_channels=32,
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
    print("[mapper] Computing UMAP embedding on DC-VAE latents...")
    lens_umap = umap.UMAP(
        n_components=2,
        random_state=42,
    ).fit_transform(X_scaled)

    # Cover + clustering
    cover = km.Cover(n_cubes=n_cubes, perc_overlap=overlap)
    clusterer = KMeans(n_clusters=3, random_state=42)

    print("[mapper] Building Mapper graph from DC-VAE features...")
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
        "model": "DC-VAE",
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

    print(f"[mapper] Saved DC-VAE Mapper JSON to {json_out}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    df = load_or_generate("synthetic_multidim_ts.csv")
    run_dcvae_mapper(df)

