#### Step 0 - Definitions #####

import os
import sys

data_final_folder = "./data/large_sc_data/"
# llm_ontology_folder = "./data/llm_ontologies/"
results_folder = "./results/large_ontix_save/"

ont_from_cli = sys.argv[1]  # "chatgpt_ontology__", "custom_ontology__"

file_holdout = os.path.join(data_final_folder, f"census-acxcontainer_holdout.pkl")
final_ontix_model_file = os.path.join(results_folder, f"large_ontix_final_model_{ont_from_cli}.pkl")


#### Step 1 - Load and predict on holdout set #####
import pickle
import autoencodix as acx
# from autoencodix.configs.ontix_config import OntixConfig

print("Loading holdout data ...")
with open(file_holdout, "rb") as f:
	acx_container = pickle.load(f)

print("Loading trained model ...")
loaded_ontix = acx.Ontix.load(file_path=final_ontix_model_file)

print("Predicting on holdout data ...")
result = loaded_ontix.predict(
	data= acx_container
)

#### Step 2 - create plots ####

# UMAP representations of latent space
params_umap = ["sex","healthy"]
loaded_ontix.visualizer.show_latent_space(
	result=loaded_ontix.result,
	plot_type='2D-scatter',
	param=params_umap,
	split='test',
	n_downsample=10000)
# Ridgeline plots of latent space
params_ridge = ["sex","healthy"]
loaded_ontix.visualizer.show_latent_space(
	result=loaded_ontix.result,
	plot_type='Ridgeline',
	param=params_ridge,
	split='test',
	n_downsample=10000)
# Heatmap representations of latent space
params_heatmap = ["cell_type","tissue_general", "development_stage", "sex", "disease"]
loaded_ontix.visualizer.show_latent_space(
	result=loaded_ontix.result,
	plot_type='Clustermap',
	param=params_heatmap,
	split='test',
	n_downsample=10000)


#### Step 3 - Evaluate embeddings ####
import sklearn
from sklearn import linear_model
tasks = ["cell_type", "tissue", "development_stage", "sex", "disease"] 

sklearn.set_config(enable_metadata_routing=True)

sklearn_ml_class = linear_model.LogisticRegression(
							solver="sag",
							n_jobs=-1,
							class_weight="balanced",
							max_iter=200,
) 
sklearn_ml_regression = linear_model.LinearRegression() ## Unused, only classification tasks
own_metric_class = 'roc_auc_ovo'  
own_metric_regression = 'r2' 

loaded_ontix.evaluate(
	ml_model_class=sklearn_ml_class, 
	ml_model_regression=sklearn_ml_regression, 
	params= tasks,	
	metric_class = own_metric_class, 
	metric_regression = own_metric_regression, 
	reference_methods = ["PCA"], # No reference methods for tuning
	split_type = "CV-5",
	n_downsample = int(acx_container.test.data.shape[0]*0.25), # Use a subset of the data for faster evaluation
)

#### Step 4 - Save ####

# Save plots
path_plots = os.path.join(results_folder, f"large_ontix_holdout_{ont_from_cli}_plots/")
# Create directory if it does not exist
os.makedirs(path_plots, exist_ok=True)
loaded_ontix.visualizer.save_plots(
	path=path_plots, which ='all', format ='png'
)

# Save evaluation results
path_eval = os.path.join(results_folder, f"large_ontix_holdout_{ont_from_cli}_evaluation.csv")
loaded_ontix.result.embedding_evaluation.to_csv(path_eval, index=False)