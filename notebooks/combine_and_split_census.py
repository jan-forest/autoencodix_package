import scanpy
import anndata
import glob 
import numpy as np


def load_rename_adata(file_path: str) -> anndata.AnnData:
	adata = scanpy.read_h5ad(file_path)
	adata.obs_names = adata.obs.soma_joinid.astype(str)
	adata.var_names = adata.var.feature_id.astype(str)
	return adata


file_paths = glob.glob("./notebooks/census_chunks/*.h5ad")

adata_list = [load_rename_adata(fp) for fp in file_paths[:100]]  # Limit to first 1000 files for testing
adata = anndata.concat(adata_list, merge="same")
adata.obs.drop("soma_joinid", axis=1, inplace=True)
adata.var.drop("feature_id", axis=1, inplace=True)

scanpy.pp.log1p(adata, copy=False) ## Log-normalize the data inplace

## Split samples in 5% for tuning, 5% as holdout, 90% for training

holdout_names, holdout_idx = scanpy.pp.sample(np.array(adata.obs_names), fraction=0.05, copy=True, rng=42)
tune_names, tune_idx = scanpy.pp.sample(np.array(adata.obs_names.difference(holdout_names)), fraction=0.05, copy=True, rng=42)  
# combine numpy arrays using numpy's union1d (works with numpy.ndarray)
combined = np.union1d(holdout_names, tune_names)
train_names = adata.obs_names.difference(combined)

## Save splits to h5ad files
scanpy.write("./large_sc_data/census_train_split.h5ad", adata[train_names])
scanpy.write("./large_sc_data/census_tune_split.h5ad", adata[tune_names])
scanpy.write("./large_sc_data/census_holdout_split.h5ad", adata[holdout_names])