import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import kmapper as km
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
import json
import re
import matplotlib.pyplot as plt
import os
from pathlib import Path


csv_path = "covid_clean_data.csv"   # path to your data file
# out_html = "covid_mapper.html"
# out_html = "covid_mapper_9states.html"

# n_cubes = 35
# overlap_pct = 0.5
# db_eps = 0.1
# db_min_samples = 5

n_cubes = 20
overlap_pct = 0.5
db_eps = 0.15
db_min_samples = 5


feature_cols = ['Confirmed','Deaths','Active','People_Tested','Testing_Rate','Mortality_Rate','Incident_Rate']
lens_col = "Days_Since_Start"  # filter

# Load data 
df = pd.read_csv(csv_path)
df = df.dropna(subset=feature_cols + [lens_col])
print(len(df))
df = df[df['Province_State'].isin(['Arizona', 'California', 'Florida', 'Georgia', 'Illinois', 'North Carolina', 'New Jersey', 'New York', 'Texas'])]
print(len(df))
# exit()

X_raw = df[feature_cols].values.astype(float)
days = df[lens_col].values.astype(float)

# Normalize features 
scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X_raw)

# # Compute pairwise distance matrix 
# pairwise_dist = squareform(pdist(X, metric="euclidean"))
# np.save("pairwise_dist.npy", pairwise_dist)

# Lens 
lens = days.reshape(-1, 1)  # using Days_Since_Start 

# Mapper setup 
mapper = km.KeplerMapper(verbose=1)
cover = km.Cover(n_cubes=n_cubes, perc_overlap=overlap_pct)
clusterer = DBSCAN(eps=db_eps, min_samples=db_min_samples, metric="euclidean")

graph = mapper.map(
    X = X,
    lens = lens,
    cover=cover,
    clusterer=clusterer,
)


# Node statistics (state counts)
state_labels = df['Province_State'].astype('category').cat.codes.values
state_names = dict(enumerate(df['Province_State'].astype('category').cat.categories))
out_dict = {"nodes": graph["nodes"], "links": graph["links"], "state_names": state_names}
# print(out_dict);exit()

node_stats = {}
for nid, members in graph["nodes"].items():
    ys = state_labels[np.array(members, dtype=int)]
    vals, counts = np.unique(ys, return_counts=True)
    dominant = int(vals[np.argmax(counts)])
    purity = float(np.max(counts)) / float(len(ys))
    node_stats[str(nid)] = dict(
        size=len(ys),
        counts={int(k): int(v) for k, v in zip(vals, counts)},
        dominant_class=dominant,
        purity=purity,
        misrate=None
    )

# Export JSON for D3 viewer
nodes = []
for nid_str, members in graph["nodes"].items():
    s = node_stats[str(nid_str)]
    nodes.append({
        "id": str(nid_str),
        "members": members,
        "size": s["size"],
        "counts": s["counts"],
        "dominant_class": s["dominant_class"],
        "purity": s["purity"],
        "misrate": s["misrate"],
    })

seen = set()
links = []
for s, nbrs in graph["links"].items():
    for t in nbrs:
        if s == t: continue
        key = tuple(sorted((str(s), str(t))))
        if key in seen: continue
        seen.add(key)
        links.append({"source": str(s), "target": str(t)})

out = {"nodes": nodes, "links": links, "state_names": state_names}
out_json = os.path.join('./', f"covid_9states.json")
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(out, f, indent=2)
print(f"[saved] {out_json}")