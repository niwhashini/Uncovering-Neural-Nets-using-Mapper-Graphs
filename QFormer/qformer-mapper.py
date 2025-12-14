import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import kmapper as km
from kmapper import KeplerMapper
# import umap
import umap.umap_ as umap


EMBEDDING_FILE = "./qformer_embeddings_allLayers_konvid.pt"
N_COMPONENTS_PCA = 200          
N_CUBES = 20
OVERLAP = 0.3
USE_UMAP = True                 # If False → PCA lens instead
OUTPUT_HTML = "mapper_qformer_layer1_lensUMAP.html"


# Load embeddings
print(f" Loading file: {EMBEDDING_FILE}")
data = torch.load(EMBEDDING_FILE)

emb = data["embeddings"]      # [N, 13, 47, 768] 
emb = emb[:, 1, :, :]         # Use layer 1 embeddings only → [N, 47, 768]
emb.squeeze_()                # [N, 47, 768]
# print(emb.shape);exit()
mos = data["mos"].numpy()     # [N]
names = data["video_names"]   # list of strings

N, Q, D = emb.shape
print(f" Embeddings shape: {emb.shape}")


# Flatten to [N, Q*D]
emb_flat = emb.reshape(N, Q * D)
emb_flat = emb_flat.numpy().astype(np.float32)
print(f" Flattened shape: {emb_flat.shape}")   # [N, 36096]


#  Standardize + PCA to 200 dims
print(" Running PCA reduction...")
scaler = StandardScaler()
emb_scaled = scaler.fit_transform(emb_flat)

pca = PCA(n_components=N_COMPONENTS_PCA)
emb_pca = pca.fit_transform(emb_scaled)
print(f" PCA output shape: {emb_pca.shape}")   # [N, 200]


# Lens function (UMAP or PCA)
if USE_UMAP:
    print(" Computing UMAP lens...")
    lens = umap.UMAP(n_neighbors=50, min_dist=0.5).fit_transform(emb_pca)
    
else:
    print(" Using PCA[0] as 1D lens...")
    # lens = emb_pca[:, 0].reshape(-1, 1)
    lens = PCA(n_components=2).fit_transform(emb_pca)


#  Build Mapper Graph
print(" Building Kepler Mapper graph...")

mapper = KeplerMapper(verbose=1)

graph = mapper.map(
    lens,
    emb_pca,
    cover=km.Cover(n_cubes=5, perc_overlap=0.5),
    clusterer=km.cluster.KMeans(n_clusters=5),
)

# Visualize
print(f" Creating visualization → {OUTPUT_HTML}")

mapper.visualize(
    graph,
    path_html=OUTPUT_HTML,
    title="QFormer Embeddings — Kepler Mapper",
    color_values=mos,                    # continuous MOS coloring
    color_function_name="MOS score",
    # node_color_function=None,
)

print(f" Open {OUTPUT_HTML}.")
