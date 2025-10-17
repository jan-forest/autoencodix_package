import scanpy as sc
import pandas as pd


## Pre-handling of Anndata-files
# anndata_file = "./large_sc_data/gtex_all-tissue_v9.h5ad"
anndata_file = "./large_sc_data/census_tune_split.h5ad"
# anndata_file = "./large_sc_data/pan_immune_blood.h5ad"


adata = sc.read_h5ad(anndata_file)
print(adata)

### Reduce adata to relevant genes only
import glob

# Find all files ending with "_level2.tsv" in the folder
tsv_files = glob.glob("llm_ontologies/*_level2.tsv")

# Read only the first column (gene names) from each file and collect unique gene names
gene_names = set()
for file in tsv_files:
	df = pd.read_csv(file, sep='\t', usecols=[0])
	gene_names.update(df.iloc[:, 0].dropna().unique())

print(f"Total unique gene names: {len(gene_names)}")

# Filter adata to keep only the relevant genes
print(list(gene_names)[:5])
print(adata.var_names[:5])

adata = adata[:, adata.var_names.isin(gene_names)]
print(adata)

## Already done
# # Filter cells based on the number of genes expressed
# print("Filter and log-normalize the data")
# perc_genes = 0.05  # Minimum percentage of genes that must be expressed in a cell
# sc.pp.filter_cells(adata, min_genes=int(perc_genes * len(adata.var_names)), inplace=True)
# sc.pp.log1p(adata, copy=False) ## Log-normalize the data inplace
# print(adata)

from sklearn.model_selection import train_test_split  # type: ignore
import numpy as np
n_samples = adata.n_obs
random_seed = 42  # For reproducibility
valid_proportion = 0.9
first_test_size = 0.4 

train_idx, temp_idx = train_test_split(
	np.arange(n_samples),
	test_size=first_test_size,
	random_state=random_seed,
)
val_idx, test_idx = train_test_split(
	temp_idx,
	test_size=(1 - valid_proportion),  # Remaining proportion goes to test
	random_state=random_seed,
)

# Create split indicators as numpy arrays
train_split = np.zeros(n_samples, dtype=bool)
train_split[train_idx] = True

val_split = np.zeros(n_samples, dtype=bool)
val_split[val_idx] = True

test_split = np.zeros(n_samples, dtype=bool)
test_split[test_idx] = True


## Build own DataSetContainer with preprocessed data
import torch
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.data._datasetcontainer import DatasetContainer

from autoencodix.configs.ontix_config import OntixConfig
import yaml
from pathlib import Path

scconfig = OntixConfig.model_validate(
    yaml.safe_load(Path("large-ontix.yaml").read_text())
)


train_dataset = NumericDataset(
	# data=torch.from_numpy(adata.X[train_idx].toarray()),
	data=adata.X[train_idx],
	config=scconfig,
	sample_ids=adata.obs.index[train_idx],
	metadata=adata.obs.loc[adata.obs.index[train_idx],:],
	split_indices=train_split,
	feature_ids=adata.var.index,
)
print("train ready")
test_dataset = NumericDataset(
	# data=torch.from_numpy(adata.X[test_idx].toarray()),
	data=adata.X[test_idx],
	config=scconfig,
	sample_ids=adata.obs.index[test_idx],
	metadata=adata.obs.loc[adata.obs.index[test_idx],:],
	split_indices=test_split,
	feature_ids=adata.var.index,
)
print("test ready")

val_dataset = NumericDataset(
	# data=torch.from_numpy(adata.X[val_idx].toarray()),
	data=adata.X[val_idx],
	config=scconfig,
	sample_ids=adata.obs.index[val_idx],
	metadata=adata.obs.loc[adata.obs.index[val_idx],:],
	split_indices=val_split,
	feature_ids=adata.var.index,
)
print("val ready")
del adata  # Free up memory
print("adata deleted")
processed_data = DatasetContainer(train=train_dataset, valid=val_dataset, test=test_dataset)
print("DatasetContainer ready")
del train_dataset, val_dataset, test_dataset  # Free up memory
print("individual datasets deleted")
## Save the processed data as pickle file
import pickle
with open("./large_sc_data/census-processed-chatgpt.pkl", "wb") as f:
	pickle.dump(processed_data, f)

print("processed data saved")


# ontology_name = "chatgpt_ontology__"
# ont_files = [
# 	# Order from Latent Dim -> Hidden Dim -> Input Dim
# 	f"./llm_ontologies/{ontology_name}ensembl_level1.tsv",
# 	f"./llm_ontologies/{ontology_name}ensembl_level2.tsv",
# 	]

# import autoencodix as acx

# ontix = acx.Ontix(data=processed_data,ontologies=ont_files, config=scconfig)
# del processed_data  # Free up memory
# ontix.run()
# ontix._visualizer.show_latent_space(result=ontix.result, plot_type='Ridgeline', param=["cell_type","tissue"], split='test')
# ontix._visualizer.save_plots(path='./large_ontix_save/', which='all', format='png')
# ontix.save(file_path="./large_ontix_save/large-ontix.pkl")