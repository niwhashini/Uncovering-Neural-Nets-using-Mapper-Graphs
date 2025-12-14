all: json

data:
	python synthetic_ts_plot.py

json:
	python synthetic_mapper_pipeline.py --out synthetic_mapper_pca_umap.json
	python lstm_mapper_pipeline.py --out mapper_lstm.json
	python dcvae_mapper_pipeline.py --out mapper_vae.json
	python trans_mapper_pipeline.py --out mapper_transf.json

view:
	python -m http.server 8000

