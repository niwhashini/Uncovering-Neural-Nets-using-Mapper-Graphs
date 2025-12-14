# synthetic_mapper_pipeline.py
import json
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import kmapper as km
import umap

from synthetic_ts_plot import generate_synthetic_dataset  # or comment out & just load CSV

# ---------------------------------------------------------------------
# Data loading + windowing
# ---------------------------------------------------------------------
def load_or_generate(path: str = "synthetic_multidim_ts.csv") -> pd.DataFrame:
    """Load the synthetic CSV if it exists, otherwise generate + save it."""
    try:
        df = pd.read_csv(path)
        print(f"Loaded {path}")
    except FileNotFoundError:
        print("CSV not found, generating synthetic dataset...")
        df = generate_synthetic_dataset()
        df.to_csv(path, index=False)
        print(f"Saved new synthetic dataset to {path}")
    return df


def build_windows(df: pd.DataFrame, window_size: int = 200, stride: int = 20):
    """
    Slice the multichannel series into overlapping windows and compute
    per-window features + labels.

    Assumes:
      - channel columns named ch0, ch1, ...
      - df["label"] : 0/1 global anomaly flag per timestep
      - df["type"]  : 0..5 anomaly type per timestep (0 = normal)
    """
    channels = [c for c in df.columns if c.startswith("ch")]
    data = df[channels].values
    labels_global = df["label"].values          # 0/1
    labels_type = df["type"].values.astype(int) # 0..5
    T = len(df)

    windows = []
    win_labels_glob = []
    win_labels_type = []
    win_channel_std = []
    win_start_idx = []

    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        seg = data[start:end, :]
        seg_label_glob = labels_global[start:end]
        seg_label_type = labels_type[start:end]

        # flattened window feature for Mapper
        windows.append(seg.reshape(-1))

        # anomaly fraction in this window (share of timesteps flagged anomalous)
        frac = seg_label_glob.mean()
        win_labels_glob.append(frac)

        # dominant anomaly *type* inside window, ignoring 0 = normal
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
        win_labels_type.append(dom_type)

        # channel-wise std in this window
        win_channel_std.append(seg.std(axis=0))
        win_start_idx.append(start)

    X = np.vstack(windows)
    win_labels_glob = np.array(win_labels_glob)
    win_labels_type = np.array(win_labels_type, dtype=int)
    win_channel_std = np.vstack(win_channel_std)
    win_start_idx = np.array(win_start_idx, dtype=int)

    return X, win_labels_glob, win_labels_type, win_channel_std, win_start_idx


# ---------------------------------------------------------------------
# Mapper pipeline 
# ---------------------------------------------------------------------
def run_mapper_pipeline(
    df: pd.DataFrame,
    window_size: int = 200,
    stride: int = 20,
    n_cubes: int = 30,
    overlap: float = 0.4,
    json_out: str = "synthetic_mapper_pca_umap.json",
) -> None:
    channels = [c for c in df.columns if c.startswith("ch")]
    n_channels = len(channels)

    # Window features + labels
    X, win_frac, win_type, win_ch_std, win_starts = build_windows(
        df, window_size=window_size, stride=stride
    )
    print("Window feature matrix:", X.shape)

    # Standardise features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA lens for Mapper
    mapper = km.KeplerMapper(verbose=1)
    lens_pca = mapper.fit_transform(
        X_scaled,
        projection=PCA(n_components=2),
        scaler=None,
    )

    # UMAP embedding for viewer (alternative lens)
    print("Computing UMAP embedding for viewer...")
    lens_umap = umap.UMAP(
        n_components=2,
        random_state=42,
    ).fit_transform(X_scaled)

    # Cover + clustering
    cover = km.Cover(n_cubes=n_cubes, perc_overlap=overlap)
    clusterer = KMeans(n_clusters=3, random_state=42)

    print("Building Mapper graph...")
    graph = mapper.map(
        lens_pca,
        X_scaled,
        cover=cover,
        clusterer=clusterer,
    )

    print(
        f"Graph has {len(graph['nodes'])} raw nodes and "
        f"{len(graph['links'])} raw links"
    )

    # Map node keys like "cube0_cluster0" -> 0,1,2,...
    node_keys = list(graph["nodes"].keys())
    key_to_int = {k: i for i, k in enumerate(node_keys)}

    # Aggregate node stats
    nodes_out = []
    members_by_id = []  # for rebuilding links

    for key in node_keys:
        node_id = key_to_int[key]
        members = np.array(graph["nodes"][key], dtype=int)
        members_by_id.append(set(members.tolist()))
        size = len(members)

        # node positions = mean of member windows in each lens
        pca_coords = lens_pca[members].mean(axis=0).tolist()
        umap_coords = lens_umap[members].mean(axis=0).tolist()

        # average anomaly fraction across windows in node
        anomaly_fraction = float(win_frac[members].mean())

        # window-level anomaly types for this node (0..5)
        types_node = win_type[members].astype(int)
        type_counts = np.bincount(types_node, minlength=6)

        # dominant *non-normal* anomaly type, else 0
        total = type_counts.sum()
        anomalous = total - type_counts[0]
        if anomalous > 0:
            dominant_type = int(1 + np.argmax(type_counts[1:]))
        else:
            dominant_type = 0

        # histogram for pies: [n0, n1, n2, n3, n4, n5]
        type_hist = type_counts.tolist()

        # per-node channel std + dominant channel
        ch_std_node = win_ch_std[members].mean(axis=0)
        dominant_channel = int(np.argmax(ch_std_node))
        channel_std_list = ch_std_node.tolist()

        # birth_step = earliest window start index in this node
        birth_step = int(win_starts[members].min())

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
            }
        )

    # Rebuild links from overlaps of member sets
    links_out = []
    n_nodes = len(nodes_out)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if members_by_id[i] & members_by_id[j]:  # non-empty intersection
                links_out.append({"source": i, "target": j})

    print(f"Rebuilt {len(links_out)} links from membership overlaps")

    meta = {
        "window_size": window_size,
        "stride": stride,
        "n_channels": n_channels,
        "channels": channels,
        "lens": ["PCA", "UMAP"],
        "color_modes": ["anomaly_fraction", "anomaly_type", "dominant_channel"],
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

    print(f"Saved Mapper JSON to {json_out}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    df = load_or_generate("synthetic_multidim_ts.csv")
    run_mapper_pipeline(df)

