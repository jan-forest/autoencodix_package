#### Step 0 - Definitions #####
import os
import sys
import pickle
import autoencodix as acx
from autoencodix.configs.ontix_config import OntixConfig

data_final_folder = "./data/large_sc_data/"
llm_ontology_folder = "./data/llm_ontologies/"
results_folder = "./results/large_ontix_save/"

ont_from_cli = sys.argv[1]  # "chatgpt_ontology__", "custom_ontology__"
tuning_experiment_file = sys.argv[2]  # Name of tuning experiment pickle file

file_processed = os.path.join(data_final_folder, "census-acxcontainer_train.pkl")

ont_files = [
	# Order from Latent Dim -> Hidden Dim -> Input Dim
	os.path.join(llm_ontology_folder, f"{ont_from_cli}ensembl_level1.tsv"),
	os.path.join(llm_ontology_folder, f"{ont_from_cli}ensembl_level2.tsv"),
	]

#### Step 1 - Load best hyperparameters from tuning experiment #####
with open(os.path.join(results_folder, tuning_experiment_file), "rb") as f:
	tuning_experiment = pickle.load(f)
best_hyperparams = tuning_experiment.best_config()

#### Step 2 - Load data and define config #####
with open(file_processed, "rb") as f:
	acx_container = pickle.load(f)

scconfig = OntixConfig(
	## Fixed params
	epochs=best_hyperparams['config_epochs'],
	# epochs=5, # Reduce for testing
	checkpoint_interval= best_hyperparams['config_checkpoint_interval'],
	loss_reduction= best_hyperparams['config_loss_reduction'],
	## Tunable params
	batch_size= best_hyperparams['config_batch_size'],
	drop_p= best_hyperparams['config_drop_p'],
	enc_factor= best_hyperparams['config_enc_factor'],
	weight_decay= best_hyperparams['config_weight_decay'],
	beta= best_hyperparams['config_beta'],
	learning_rate= best_hyperparams['config_learning_rate'],
	n_layers= best_hyperparams['config_n_layers'],
	save_memory=True,
)

#### Step 3 - Final training #####
ontix = acx.Ontix(
	data=acx_container,
	ontologies=ont_files,
	config=scconfig
	)

print("Starting preprocessing ...")
ontix.preprocess()
print("Starting training ...")
ontix.fit()
print("Training finished, saving model ...")
ontix.save(os.path.join(results_folder, f"large_ontix_final_model_{ont_from_cli}.pkl"), save_all=False)