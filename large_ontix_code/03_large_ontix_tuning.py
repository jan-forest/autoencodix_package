#### Step 0 - Definitions #####
import sys
import os
from datetime import datetime

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import PythonBackend
from syne_tune.config_space import randint, uniform, loguniform
from syne_tune.optimizer.baselines import CQR
from syne_tune.experiments import load_experiment

data_final_folder = "./data/large_sc_data/"
results_folder = "./results/large_ontix_save/"
# Create results_folder if it doesn't exist
if not os.path.exists(results_folder):
	os.makedirs(results_folder)

ont_from_cli = sys.argv[1]  # "chatgpt_ontology__", "custom_ontology__"
max_wallclock_time = int(float(sys.argv[2])*60*60)  # given in hours translated to seconds
n_workers = int(sys.argv[3])  # number of parallel workers for tuning

metric = "ml_performance" # "ml_performance" or "recon_loss"

tasks = ["cell_type", "tissue", "development_stage", "sex", "disease"] 

# file_processed = os.path.join(data_final_folder, "census-acxcontainer_tune.pkl")
file_processed = os.path.join(data_final_folder, "census-acxcontainer_train.pkl") ## Only for testing


#### Step 1 - Definition of syne_trainer function #####
def syne_trainer(
	## Fixed params
	epochs: int,
	checkpoint_interval: int,
	loss_reduction: str,
	data_path: str,
	ontology_name: str,
	tasks: str,
	## Tunable params
	batch_size: int,
	drop_p: float,
	enc_factor: int,
	weight_decay: float,
	beta: float,
	learning_rate: float,
	n_layers: int,
):
	## Load Dataset Container from pickle with already preprocessed data
	import os
	import pickle
	import sklearn
	from sklearn import linear_model
	import autoencodix as acx
	from autoencodix.configs.ontix_config import OntixConfig
	from syne_tune import Reporter

	llm_ontology_folder = "./data/llm_ontologies/"
	
	file_pkl = data_path

	with open(file_pkl, "rb") as f:
		acx_container = pickle.load(f)

	scconfig = OntixConfig(
		## Fixed params
		epochs=epochs,
		checkpoint_interval= checkpoint_interval,
		loss_reduction= loss_reduction,
		## Tunable params
		batch_size= batch_size,
		drop_p= drop_p,
		enc_factor= enc_factor,
		weight_decay= weight_decay,
		beta= beta,
		learning_rate= learning_rate,
		n_layers= n_layers,
	)


	ont_files = [
		# Order from Latent Dim -> Hidden Dim -> Input Dim
		os.path.join(llm_ontology_folder, f"{ontology_name}ensembl_level1.tsv"),
		os.path.join(llm_ontology_folder, f"{ontology_name}ensembl_level2.tsv"),
		]

	ontix = acx.Ontix(
		data=acx_container,
		ontologies=ont_files,
		config=scconfig
		)

	ontix.run()

	## Embedding Evaluation
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

	tasks_list = tasks.split("$")
	ontix.evaluate(
		ml_model_class=sklearn_ml_class, 
		ml_model_regression=sklearn_ml_regression, 
		params= tasks_list,	
		metric_class = own_metric_class, 
		metric_regression = own_metric_regression, 
		reference_methods = [], # No reference methods for tuning
		split_type = "use-split",
		n_downsample = int(acx_container.train.data.shape[0]*0.1), # Use a subset of the data for faster evaluation
	)

	avg_mltask_performance = ontix.result.embedding_evaluation.loc[
		ontix.result.embedding_evaluation.score_split == "valid",
		"value"
	].mean()
	valid_recon_loss = float(ontix.result.sub_losses.get("recon_loss").get(epoch=-1, split="valid"))

	report = Reporter()
	report(ml_performance=avg_mltask_performance, recon_loss=valid_recon_loss)

### Step 2 - Tuning #####


# Hyperparameter configuration 
epoch = 5  # For testing, reduce number of epochs
config_space = {
	## Fixed params
	"epochs": epoch,
	"checkpoint_interval": epoch,
	"loss_reduction": "sum",
	"data_path": file_processed,
	"ontology_name": ont_from_cli,
	"tasks": "$".join(tasks),
	## Tunable params
	"batch_size": randint(128, 4096),
	"drop_p": uniform(0.0, 0.9),
	"enc_factor": randint(1, 5),
	"weight_decay": loguniform(1e-5, 1e-1),
	"beta": loguniform(1e-5, 10),
	"learning_rate": loguniform(1e-5, 1e-1),
	"n_layers": randint(2, 5),
}

# Define whether to minimize or maximize the metric
if metric == "ml_performance":
	do_minimize = False
elif metric == "recon_loss":
	do_minimize = True
else:
	do_minimize = False

# Scheduler (i.e., HPO algorithm)
scheduler = CQR(
    config_space,
    metric=metric,
    do_minimize=do_minimize
)

# Tuning
print("Starting tuning ...")
tuner = Tuner(
    trial_backend=PythonBackend(tune_function=syne_trainer, config_space=config_space),
    scheduler=scheduler,
    stop_criterion=StoppingCriterion(
		max_wallclock_time=max_wallclock_time,  # in seconds
		# max_num_trials_completed=2, # Alternatively, limit number of trials
		),
    n_workers=n_workers,  # how many trials are evaluated in parallel (GPUs)
)

tuner.run()
tuning_experiment = load_experiment(tuner.name)
print(f"best result found: {tuning_experiment.best_config()}")

#### Step 3 - Save tuning experiment results #####
import pickle

# current date for filename
current_date = datetime.now().strftime("%Y-%m-%d")

# with open(f"./notebooks/large_ontix_save/large_ontix_chatgpt_experiment_{current_date}.pkl", "wb") as f:
with open(os.path.join(results_folder, f"large_ontix_{ont_from_cli}tuning.pkl"), "wb") as f:
	pickle.dump(tuning_experiment, f)