import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the saved QFormer query output file
print(" loading data")
data = torch.load("./qformer_embeddings_allLayers_konvid.pt")  

queries = data["embeddings"]          # [N, 13, 47, 768]
mos = np.array(data["mos"])        # [N]
print(" completed data loading.")

# Extract every layers' output
for l in range(queries.shape[1]):

    last_layer = queries[:, l, :, :]  # shape = [N, 47, 768]
    N, Q, D = last_layer.shape

    # flatten
    X = last_layer.reshape(N, Q * D)
    X = X.numpy().astype(np.float32)
    print(f" Layer {l} flattened shape: {X.shape}")

    # normalize
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # PCA
    print(f" applying PCA for layer {l}")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_norm)

    # Plot
    print(f" visualizing PCA for layer {l}")

    plt.figure(figsize=(8,8))

    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=mos,
        cmap="viridis",
        s=25,
        alpha=0.85
    )

    plt.colorbar(scatter, label="MOS Score")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA of layer {l} query outputs")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save figure
    out_name = f"./pca_out/pca_layer_{l}.png"
    plt.savefig(out_name, dpi=300)
    plt.close()

    print(f" saved PCA plot for layer {l} → {out_name}")
