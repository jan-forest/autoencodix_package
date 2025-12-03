#!/bin/bash


### Set up environment ###
# gh repo clone jan-forest/autoencodix_package -b large-ontix-stuff
# cd autoencodix_package
# uv venv --python 3.10
# source .venv/bin/activate
# uv sync
# uv pip install syne-tune[extra] # Getting first all extra dependencies
# uv pip install syne-tune==0.14.2 # Specific version for compatibility
# uv pip install cellxgene-census # For census data retrieval
############################



source .venv/bin/activate
### Retrieve census chunks and prepare data ### -> no GPU required
# Define chunks
python large_ontix_code/01_get_census_data.py "step1" 

# Retrieve chunks in parallel 
for i in {0..12};do # Parallelize on HPC for speed up (takes around 24h)
	python large_ontix_code/01_get_census_data.py "step2" $i "test" # Test mode with only 24 chunks for quick testing
done

# Combine chunks and split data, save as h5ad files
python large_ontix_code/02_census_preparation.py "step1"
# Prepare acx containers for each data split
python large_ontix_code/02_census_preparation.py "step2" "train"
python large_ontix_code/02_census_preparation.py "step2" "tune"
python large_ontix_code/02_census_preparation.py "step2" "holdout"
###############################################################



### Tuning of large Ontix model ### -> multiple GPU required
# Define list of ontologies
ont_list = ("chatgpt_ontology__", "gemini_ontology__", "claude_ontology__")
# ont_list=("claude_ontology__")

n_workers=1
time_for_tuning=0.1  # in hours
# Loop over ontologies for tuning -> should be parallelized
for ont in ${ont_list[@]}; do
	python large_ontix_code/03_large_ontix_tuning.py $ont $time_for_tuning $n_workers
done
###############################################



### Final training of large Ontix model ### -> one GPU required
# Loop over ontologies for final training -> should be parallelized
for ont in ${ont_list[@]}; do
	tuning_file=large_ontix_${ont}tuning.pkl
	python large_ontix_code/04_large_ontix_finaltraining.py $ont $tuning_file
	# Visualize final model results
	python large_ontix_code/05_pred_vis_eval.py $ont
done
###############################################

