#### STEP 0 - Definitions #####
import os
import sys
data_folder = "./census_chunks/"
data_final_folder = "./notebooks/large_sc_data/"
llm_ontology_folder = "./notebooks/llm_ontologies/"

step_from_cli = sys.argv[1]  # "step1", "step2", "..."
fraction_for_tuning = 0.05
fraction_for_holdout = 0.05
seed = 42

def load_rename_adata(file_path: str) -> anndata.AnnData:
	adata = scanpy.read_h5ad(file_path)
	adata.obs_names = adata.obs.soma_joinid.astype(str)
	adata.var_names = adata.var.feature_id.astype(str)
	return adata

#### STEP 1 - Combine and split census chunks #####
print("STEP 1 - Combine and split census chunks")
if step_from_cli == "step1":
	import scanpy
	import anndata
	import glob 
	import numpy as np

	## Combine all census chunks
	print("Combining census chunks")
	file_paths = glob.glob(os.path.join(data_folder, "*.h5ad"))

	adata_list = [load_rename_adata(fp) for fp in file_paths[:]]  # Limit to first 1000 files for testing
	adata = anndata.concat(adata_list, merge="same")
	adata.obs.drop("soma_joinid", axis=1, inplace=True)
	adata.var.drop("feature_id", axis=1, inplace=True)

	 ## Log-normalize the data inplace
	print("Log-normalizing the data")
	scanpy.pp.log1p(adata, copy=False)

	## Split samples in 5% for tuning, 5% as holdout, 90% for training
	print("Splitting data into train, tune, and holdout sets")
	holdout_names, holdout_idx = scanpy.pp.sample(np.array(adata.obs_names), fraction=fraction_for_holdout, copy=True, rng=seed)
	tune_names, tune_idx = scanpy.pp.sample(np.array(adata.obs_names.difference(holdout_names)), fraction=fraction_for_tuning, copy=True, rng=seed)  
	# combine numpy arrays using numpy's union1d (works with numpy.ndarray)
	combined = np.union1d(holdout_names, tune_names)
	train_names = adata.obs_names.difference(combined)

	## Save splits to h5ad files
	print("Saving train, tune, and holdout splits to files")
	scanpy.write(os.path.join(data_final_folder, "census_train_split.h5ad"), adata[train_names])
	scanpy.write(os.path.join(data_final_folder, "census_tune_split.h5ad"), adata[tune_names])
	scanpy.write(os.path.join(data_final_folder, "census_holdout_split.h5ad"), adata[holdout_names])	

#### STEP 2 - Prepare each split for large Ontix training #####
print("STEP 2 - Prepare each split for large Ontix training")
if step_from_cli == "step2":
	split_from_cli = sys.argv[2]  # "train", "tune", "holdout"
	import scanpy as sc
	# import pandas as pd
	import numpy as np
	from sklearn.model_selection import train_test_split
	from autoencodix.data._numeric_dataset import NumericDataset
	from autoencodix.data._datasetcontainer import DatasetContainer

	from autoencodix.configs.ontix_config import OntixConfig
	import yaml
	from pathlib import Path

	anndata_file = os.path.join(data_final_folder, f"census_{split_from_cli}_split.h5ad")
	adata = sc.read_h5ad(anndata_file)
	n_samples = adata.n_obs
	if split_from_cli in ["train", "tune"]:
		if split_from_cli == "train":
			first_test_size = 0.01 # Must be non-zero since acx requires all splits to be non-empty
			valid_proportion = 0.5 # final train: 99%, valid : 0.5%, test: 0.5%
		if split_from_cli == "tune":
			first_test_size = 0.4
			valid_proportion = 0.9 # in tune: 60%, valid: 36%, test: 4%
		

		train_idx, temp_idx = train_test_split(
			np.arange(n_samples),
			test_size=first_test_size,
			random_state=seed,
		)
		val_idx, test_idx = train_test_split(
			temp_idx,
			test_size=(1 - valid_proportion),  # Remaining proportion goes to test
			random_state=seed,
		)

		# Create split indicators as numpy arrays
		train_split = np.zeros(n_samples, dtype=bool)
		train_split[train_idx] = True

		val_split = np.zeros(n_samples, dtype=bool)
		val_split[val_idx] = True

		test_split = np.zeros(n_samples, dtype=bool)
		test_split[test_idx] = True 

		scconfig = OntixConfig.model_validate(
			yaml.safe_load(Path(os.path.join(data_final_folder, "large-ontix.yaml")).read_text())
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
		with open(os.path.join(data_final_folder, f"census-acxcontainer_{split_from_cli}.pkl"), "wb") as f:
			pickle.dump(processed_data, f)

		print("processed data saved")
	elif split_from_cli == "holdout":
		# For holdout, we only need a test split
		test_split = np.ones(n_samples, dtype=bool)  # All samples in holdout are for testing

		scconfig = OntixConfig.model_validate(
			yaml.safe_load(Path(os.path.join(data_final_folder, "large-ontix.yaml")).read_text())
		)

		test_dataset = NumericDataset(
			data=adata.X,
			config=scconfig,
			sample_ids=adata.obs.index,
			metadata=adata.obs.loc[adata.obs.index,:],
			split_indices=test_split,
			feature_ids=adata.var.index,
		)
		print("holdout test ready")
		del adata  # Free up memory
		print("adata deleted")
		processed_data = DatasetContainer(train=None, valid=None, test=test_dataset)
		print("Holdout DatasetContainer ready")
		del test_dataset  # Free up memory
		print("individual datasets deleted")
		## Save the processed data as pickle file
		import pickle
		with open(os.path.join(data_final_folder, f"census-acxcontainer_{split_from_cli}.pkl"), "wb") as f:
			pickle.dump(processed_data, f)

		print("holdout processed data saved")
	else:
		print("Invalid split_from_cli argument. Use 'train', 'tune', or 'holdout'.")
	

