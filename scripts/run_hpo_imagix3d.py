from syne_tune.config_space import choice, loguniform
from syne_tune.optimizer.baselines import CQR
from syne_tune import Tuner, StoppingCriterion
from syne_tune.experiments import load_experiment
from syne_tune.backend import PythonBackend
from pathlib import Path
from typing import Literal, Union


def run_synetune_hpo(
	data_path: Path,
	tasks: str | list[str] = "group",
	metric: Literal["reconstruction_loss", "downstream_performance"] = "reconstruction_loss", 
	folder: str = "train", 
	anno: str = "train_metadata.csv"):

	volume_root = data_path / folder
	annotation_file = data_path / anno

	### Step 1: Defining the Configuration Space ###

	# Categorization label column in annotation file for downstream prediction task
	#tasks = ["group"]

	# Hyperparameter configuration space
	config_space = {
		## Fixed params
		"epochs": 100,
		"checkpoint_interval": 10,
		"loss_reduction": "mean",
		## Tunable params
		"batch_size": choice([16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 256]),
 		"learning_rate": loguniform(1e-5, 1e-1),
  		"weight_decay": loguniform(1e-5, 1e-1),
		"beta": loguniform(1e-5, 5e-2),
  		"latent_dim": choice([16, 32, 48, 64, 128, 256]),
  		"hidden_dim": choice([8, 16, 32, 48, 64]),
  		"train_normalization": choice(["group", "instance", "batch"]),
  		"anneal_function": choice([
    		"5phase-constant",
    		"3phase-linear",
    		"3phase-log",
        	"logistic-mid",
        	"logistic-early",
        	"logistic-late",
  		]),
        "keep_mu_positive": choice([False, True]),
        "logvar_range": choice(
            [(-10.0, 20.0),
            (0.1, 20.0)]
        ),
  		# Downstream tasks to evaluate on
    	"tasks": tasks,
  		# "tasks": "$".join(tasks)
	}

	# Provides a baseline by comparing the manually chosen hp set with the combinations of hp ranges from the config_space
	# This specific parameter combination was chosen based on the last hpo procedure
	points_to_evaluate = [
    	{
        	"batch_size": 256,
        	"learning_rate": 1e-3,
        	"weight_decay": 5e-3,
        	"beta": 0.02,
        	"latent_dim": 32,
        	"hidden_dim": 8,
        	"anneal_function": "logistic-late",
        	"train_normalization": "instance",
            "keep_mu_positive": False,
            "logvar_range": (-10.0, 20.0),
    	}
	]


	### Step 2: Choosing objective and algorithm ###

	if metric == "downstream_performance":
		do_minimize = False
	else:
		do_minimize = True

	# Scheduler (i.e. Optimizer)
	scheduler = CQR(
    	config_space=config_space,
    	metric=metric,
    	do_minimize=do_minimize,
    	points_to_evaluate=points_to_evaluate,
	)

	### Step 3: Defining the Objective Function ###

	def objective_function(
     
		## Fixed params
		epochs: int,
		checkpoint_interval: int,
		loss_reduction: str,
  
		## Tunable params
		batch_size: int,
    	learning_rate: float,
    	weight_decay: float,
		beta: float,
    	latent_dim: int,
    	hidden_dim: int,
    	anneal_function: str,
    	train_normalization: str,
		keep_mu_positive: bool,
		logvar_range: tuple,
     
    	# List of tasks for downstream evaluation
    	tasks: str,
    ):

    	# Imports
		from autoencodix.configs.imagix3d_config import Imagix3DConfig
		from autoencodix.configs.default_config import (
        	DataConfig,
        	DataCase,
        	DataInfo
    	)
		import autoencodix as acx
    
		import sklearn
		import numpy as np
		from sklearn import linear_model
		from syne_tune import Reporter

    	## Step 3.1 instantiating our model with a given configuration
    	# Define path to data and annotation file
		VOLROOT = volume_root
		VOLANNO = annotation_file
    
		volconfig = Imagix3DConfig(     
        	## Tunable params
        	beta=beta,
        	batch_size= batch_size,
        	latent_dim=latent_dim,
        	hidden_dim=hidden_dim,
			weight_decay=weight_decay,
			learning_rate=learning_rate,
   			anneal_function=anneal_function,
        	train_normalization=train_normalization,
			keep_mu_positive=keep_mu_positive,
			logvar_range=tuple(logvar_range),
         
        	## Fixed params
        	data_case=DataCase.IMG_TO_IMG,
        	img_path_col="filepath",
        	spatial_shape_policy="crop_or_pad_to_shape",
        	target_shape_3d=(64,64,64),
        	checkpoint_interval=checkpoint_interval,
        	epochs=epochs,
        	reconstruction_loss="mse",
        	loss_reduction=loss_reduction,
        	scaling="MINMAX",
        	normalize_nonzero_only=False,
        	clamp_logvar=True,
        	train_norm_groupsize=8,
        	data_config=DataConfig(
            	data_info={
                	"IMG": DataInfo(
                    	file_path=VOLROOT,
                    	scaling="MINMAX",
                    	data_type="IMG",
                	),
                	"ANNO": DataInfo(
                    	file_path=VOLANNO,
                    	data_type="ANNOTATION",
                	),
            	},
        	),
    	)
    
		imagix3d = acx.Imagix3D(config=volconfig)
		imagix3d.run()

		# Step 3.2 Evaluating our learned embedding
		valid_recon_loss = float(np.asarray(imagix3d.result.sub_losses.get("recon_loss").get(epoch=-1, split="valid")).item())
		train_recon_loss = float(np.asarray(imagix3d.result.sub_losses.get("recon_loss").get(epoch=-1, split="train")).item())
		valid_total_loss = float(np.asarray(imagix3d.result.losses.get(epoch=-1, split="valid")).item())
		train_total_loss = float(np.asarray(imagix3d.result.losses.get(epoch=-1, split="train")).item())
		valid_var_loss = float(np.asarray(imagix3d.result.sub_losses.get("var_loss").get(epoch=-1, split="valid")).item())

    	# Compute the downstream performance
    	# We define a list of tasks which are either regression or classification tasks from our annotation file.
    	# A linear model is used on the learned embedding to compute the performance for each task.
		sklearn.set_config(enable_metadata_routing=True)

		# Define Classifier
		sklearn_ml_class = linear_model.LogisticRegression(
			solver="sag",
			n_jobs=-1,
			class_weight="balanced",
			max_iter=200,
		)
    	# Define Regressor
		sklearn_ml_regression = linear_model.LinearRegression() # Unused, only needed if downstream task is regression variable

    	# Downstream performance metrics
		own_metric_class = 'roc_auc_ovo'
		own_metric_regression = 'r2'

		# make sure the task list has the proper type for autoencodix evaluate function
		tasks_list = [t for s in (tasks.split("$") if isinstance(tasks,str) else tasks) for t in (s.split("$") if isinstance(s,str) else [s])]

		imagix3d.evaluate(
			ml_model_class=sklearn_ml_class,
			ml_model_regression=sklearn_ml_regression,
			params= tasks_list,
			metric_class = own_metric_class,
			metric_regression = own_metric_regression,
			reference_methods = [], # No reference methods for tuning
			split_type = "use-split",
			n_downsample = None, # Use a subset of the data for faster evaluation
		)

		# here we take the average over all downstream tasks
		downstream_performance = imagix3d.result.embedding_evaluation.loc[
			imagix3d.result.embedding_evaluation.score_split == "valid",
			"value"
		].mean()

    	# We instantiate the Syne Tune Reporter and pass the model performance back to our Tuner
		report = Reporter()
		report(
        	downstream_performance=downstream_performance, 
        	reconstruction_loss=valid_recon_loss,
        	train_reconstruction_loss=train_recon_loss,
        	valid_total_loss=valid_total_loss,
        	train_total_loss=train_total_loss,
        	valid_var_loss=valid_var_loss,
        	recon_generalization_gap=valid_recon_loss - train_recon_loss,
    	)


	### Step 4: Running the Optimization ###

	# Define the Tuner
	tuner = Tuner(
    	trial_backend=PythonBackend(tune_function=objective_function, config_space=config_space),
    	scheduler=scheduler,
    	stop_criterion=StoppingCriterion(
			max_num_trials_completed=100, # Number of different hyperparameter configurations which are evaluated
			),
    	n_workers=4,  # how many trials are evaluated in parallel
	)

	# Start tuning
	tuner.run()
	return load_experiment(tuner.name)


def run_optuna_hpo():
    raise NotImplementedError("Optuna HPO will be added later.")