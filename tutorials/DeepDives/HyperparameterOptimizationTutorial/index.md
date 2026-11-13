> **Note:** This environment variable is required for fully deterministic CuBLAS ops on CUDA >= 10.2 when `reproducible=True` is set below. Without it, PyTorch raises a `RuntimeError` instead of training deterministically. It must be set before `torch` is imported. See the [README FAQ](https://jan-forest.github.io/autoencodix_package/tutorials/README.md#reproducibility-and-cublas_workspace_config) for details.

In \[1\]:

Copied!

```
%env CUBLAS_WORKSPACE_CONFIG=:16:8
```

%env CUBLAS_WORKSPACE_CONFIG=:16:8

```
env: CUBLAS_WORKSPACE_CONFIG=:16:8
```

### Disclaimer[¶](#disclaimer)

This tutorial assumes that you have already familiarized yourself with the Autoencodix pipeline and thus only briefly touches on topics like data preparation and training your Autoencodix models. If you are not yet familiar with these topics, the other tutorial notebooks in this repo are a good reference.

______________________________________________________________________

## Overview[¶](#overview)

Hyperparameter optimization allows you to systematically search over model parameters such as batch size, learning rate, network depth, and regularization strength. Instead of choosing these values manually, a hyperparameter optimization (HPO) framework like Syne Tune can be used to sample different configurations, run the model, evaluate it, and use the results to guide future trials.

Each trial follows this workflow:

1. Build a model configuration from sampled hyperparameters.
1. Train the model on the provided dataset.
1. Evaluate the learned embedding on downstream machine-learning tasks or other metrics like reconstruction loss.
1. Report evaluation metrics back to the optimizer.

The optimizer then uses an optimization algorithm to efficiently decide which hyperparameters to try next.

______________________________________________________________________

# Hyperparameter Optimization with Autoencodix using Syne Tune[¶](#hyperparameter-optimization-with-autoencodix-using-syne-tune)

This tutorial explains how to perform hyperparameter optimization (HPO) for Autoencodix models using Syne Tune. More information and documentation can be found here: <https://github.com/syne-tune/syne-tune>. The goal is to automatically find good training settings for a chosen architecture by evaluating many different configurations and selecting the one with the best (downstream) performance.

Prerequisites:

- training data
- an Autoencodix model, e.g. Vanillix, Ontix, etc.

This tutorial covers all the necessary steps:

- specifying which parameters should be optimized
- defining a training function
- choosing an optimization algorithm
- running the optimizer
- and finally retrieving the best configuration

______________________________________________________________________

## Step 0: Install Syne Tune[¶](#step-0-install-syne-tune)

To install Syne Tune from pip:

```
pip install 'syne-tune'
```

or to install the latest version from source:

```
git clone https://github.com/syne-tune/syne-tune.git
cd syne-tune
pip install -e .
```

## Step 1: Defining the Configuration Space[¶](#step-1-defining-the-configuration-space)

### Hyperparameter Configuration Space[¶](#hyperparameter-configuration-space)

First, we need to decide which parameters of our model we want to optimize and what values they can take. This is called the configuration space. We define the configuration space as a dictionary which is passed to the Syne Tune scheduler in order to sample and evaluate a model configuration.

In this case, some parameters are fixed and stay the same for all trials, such as the number of training epochs or the loss aggregation strategy. Other parameters are defined as ranges or choices, which can be sampled from, such as:

- Batch size
- Dropout probability
- Encoder width factor
- Weight decay
- Learning rate
- Number of layers
- Regularization parameters

The full list of configurable parameters can be found in DefaultConfig.py

______________________________________________________________________

In \[2\]:

Copied!

```
from syne_tune.config_space import randint, uniform, loguniform

tasks = ["condition"]

# Hyperparameter configuration space
config_space = {
	## Fixed params
	"epochs": 100,
	"checkpoint_interval": 25,
	"loss_reduction": "sum",
	## Tunable params
	"batch_size": randint(128, 4096),
	"drop_p": uniform(0.0, 0.9),
	"enc_factor": randint(1, 5),
	"weight_decay": loguniform(1e-5, 1e-1),
	"beta": loguniform(1e-5, 10),
	"learning_rate": loguniform(1e-5, 1e-1),
	"n_layers": randint(2, 5),
    # Downstream tasks to evaluate on
    "tasks": "$".join(["condition"])
}
```

from syne_tune.config_space import randint, uniform, loguniform tasks = ["condition"]

# Hyperparameter configuration space

config_space = {

## Fixed params

"epochs": 100, "checkpoint_interval": 25, "loss_reduction": "sum",

## Tunable params

"batch_size": randint(128, 4096), "drop_p": uniform(0.0, 0.9), "enc_factor": randint(1, 5), "weight_decay": loguniform(1e-5, 1e-1), "beta": loguniform(1e-5, 10), "learning_rate": loguniform(1e-5, 1e-1), "n_layers": randint(2, 5),

# Downstream tasks to evaluate on

"tasks": "$".join(["condition"]) }

## Step 2: Choosing objective and algorithm[¶](#step-2-choosing-objective-and-algorithm)

Now that we have defined a search space, we need to define our optimization goal, i.e. the metric we want to improve. In our case, we want to asses our model on two dimensions. First, how well our learned embeddings performs on downstream tasks, like classifying cancer types. Secondly, how much information our embedding retains, i.e. the reconstruction loss.

Once we have decided on a metric, we need to choose an optimization algorithm. Syne Tune provides several optimizers to efficiently find good hyperparameter configurations. In this case we choose CQR (Conformalized Quantile Regression).

To properly work, we need to provide the optimizer with several parameters:

- The search space, to sample configurations.
- The performance metric to optimize.
- The optimization direction, so whether we want to minimize or maximize our metric value.

In \[3\]:

Copied!

```
from syne_tune.optimizer.baselines import CQR

#metric = "downstream_performance"
metric = "reconstruction_loss"
if metric == "downstream_performance":
	do_minimize = False
else:
	do_minimize = True

# Scheduler (i.e. Optimizer)
scheduler = CQR(
    config_space=config_space,
    metric=metric,
    do_minimize=do_minimize
)
```

from syne_tune.optimizer.baselines import CQR #metric = "downstream_performance" metric = "reconstruction_loss" if metric == "downstream_performance": do_minimize = False else: do_minimize = True

# Scheduler (i.e. Optimizer)

scheduler = CQR( config_space=config_space, metric=metric, do_minimize=do_minimize )

## Step 3: Defining the Objective Function[¶](#step-3-defining-the-objective-function)

### Objective Function[¶](#objective-function)

The core of the optimization is the training or objective function. This function defines what happens in a single trial.

For each hyperparameter configuration proposed by the optimizer, the training function:

1. Instantiates and trains the model with a sampled configuration on the dataset.
1. Evaluates the learned embedding by computing performance metrics such as downstream performance and reconstruction loss.
1. Reports these metrics back to Syne Tune.

Syne Tune treats this function as a black box: it only cares about the input hyperparameters and the reported metrics.

______________________________________________________________________

In \[4\]:

Copied!

```
def objective_function(
	## Fixed params
	epochs: int,
	checkpoint_interval: int,
	loss_reduction: str,
	## Tunable params
	batch_size: int,
	drop_p: float,
	enc_factor: int,
	weight_decay: float,
	beta: float,
	learning_rate: float,
	n_layers: int,
    # List of tasks for downstream evaluation
    tasks: str,

):
	from autoencodix.data.datapackage import DataPackage
	from autoencodix.configs.vanillix_config import VanillixConfig
	import sklearn
	from sklearn import linear_model
	import autoencodix as acx
	from syne_tune import Reporter
	from autoencodix.utils.example_data import raw_protein, raw_rna, annotation
	from autoencodix.configs.default_config import DataCase

	# Step 3.1 instantiating our model with a given configuration
    # Our dataset
	my_datapackage = DataPackage(
    	multi_bulk={"rna": raw_rna, "protein": raw_protein},
    	annotation={"paired": annotation},
	)

    # The sampled configuration
	myconfig = VanillixConfig(
			## Fixed params
			data_case=DataCase.MULTI_BULK,
			epochs=epochs,
			checkpoint_interval= checkpoint_interval,
			loss_reduction= loss_reduction,
        	# device="cpu",  # default is "auto", which uses a GPU when available; uncomment to force CPU
			## Tunable params
			batch_size= batch_size,
			drop_p= drop_p,
			enc_factor= enc_factor,
			weight_decay= weight_decay,
			beta= beta,
			learning_rate= learning_rate,
			n_layers= n_layers,
			)

	vanillix = acx.Vanillix(data=my_datapackage, config=myconfig)
	vanillix.run()

	# Step 3.2 Evaluating our learned embedding
    # The reconstruction loss is provided directly by autoencodix
	reconstruction_loss = float(vanillix.result.sub_losses.get("recon_loss").get(epoch=-1, split="valid"))

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

	vanillix.evaluate(
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
	downstream_performance = vanillix.result.embedding_evaluation.loc[
		vanillix.result.embedding_evaluation.score_split == "valid",
		"value"
	].mean()

    # We instantiate the Syne Tune Reporter and pass the model performance back to our Tuner
	report = Reporter()
	report(downstream_performance=downstream_performance, reconstruction_loss=reconstruction_loss)
```

def objective_function(

## Fixed params

epochs: int, checkpoint_interval: int, loss_reduction: str,

## Tunable params

batch_size: int, drop_p: float, enc_factor: int, weight_decay: float, beta: float, learning_rate: float, n_layers: int,

# List of tasks for downstream evaluation

tasks: str, ): from autoencodix.data.datapackage import DataPackage from autoencodix.configs.vanillix_config import VanillixConfig import sklearn from sklearn import linear_model import autoencodix as acx from syne_tune import Reporter from autoencodix.utils.example_data import raw_protein, raw_rna, annotation from autoencodix.configs.default_config import DataCase

# Step 3.1 instantiating our model with a given configuration

# Our dataset

my_datapackage = DataPackage( multi_bulk={"rna": raw_rna, "protein": raw_protein}, annotation={"paired": annotation}, )

# The sampled configuration

myconfig = VanillixConfig(

## Fixed params

data_case=DataCase.MULTI_BULK, epochs=epochs, checkpoint_interval= checkpoint_interval, loss_reduction= loss_reduction,

# device="cpu", # default is "auto", which uses a GPU when available; uncomment to force CPU

## Tunable params

batch_size= batch_size, drop_p= drop_p, enc_factor= enc_factor, weight_decay= weight_decay, beta= beta, learning_rate= learning_rate, n_layers= n_layers, ) vanillix = acx.Vanillix(data=my_datapackage, config=myconfig) vanillix.run()

# Step 3.2 Evaluating our learned embedding

# The reconstruction loss is provided directly by autoencodix

reconstruction_loss = float(vanillix.result.sub_losses.get("recon_loss").get(epoch=-1, split="valid"))

# Compute the downstream performance

# We define a list of tasks which are either regression or classification tasks from our annotation file.

# A linear model is used on the learned embedding to compute the performance for each task.

sklearn.set_config(enable_metadata_routing=True)

# Define Classifier

sklearn_ml_class = linear_model.LogisticRegression( solver="sag", n_jobs=-1, class_weight="balanced", max_iter=200, )

# Define Regressor

sklearn_ml_regression = linear_model.LinearRegression() # Unused, only needed if downstream task is regression variable

# Downstream performance metrics

own_metric_class = 'roc_auc_ovo' own_metric_regression = 'r2'

# make sure the task list has the proper type for autoencodix evaluate function

tasks_list = \[t for s in (tasks.split("$") if isinstance(tasks,str) else tasks) for t in (s.split("$") if isinstance(s,str) else [s])\] vanillix.evaluate( ml_model_class=sklearn_ml_class, ml_model_regression=sklearn_ml_regression, params= tasks_list, metric_class = own_metric_class, metric_regression = own_metric_regression, reference_methods = [], # No reference methods for tuning split_type = "use-split", n_downsample = None, # Use a subset of the data for faster evaluation )

# here we take the average over all downstream tasks

downstream_performance = vanillix.result.embedding_evaluation.loc[ vanillix.result.embedding_evaluation.score_split == "valid", "value" ].mean()

# We instantiate the Syne Tune Reporter and pass the model performance back to our Tuner

report = Reporter() report(downstream_performance=downstream_performance, reconstruction_loss=reconstruction_loss)

## Step 4: Running the Optimization[¶](#step-4-running-the-optimization)

Once the objective function and search space are defined, we create the tuner. The tuner coordinates the optimization process by:

- Sampling hyperparameter configurations using the optimizer.
- Launching trials using the training function.
- Collecting reported metrics.

The optimizer runs until a stopping criterion is met, such as a maximum number of completed trials or a certain amount of time.

______________________________________________________________________

In \[5\]:

Copied!

```
from syne_tune import Tuner, StoppingCriterion
from syne_tune.experiments import load_experiment
from syne_tune.backend import PythonBackend
from syne_tune.optimizer.baselines import CQR

# Define the Tuner
tuner = Tuner(
    trial_backend=PythonBackend(tune_function=objective_function, config_space=config_space),
    scheduler=scheduler,
    stop_criterion=StoppingCriterion(
		#max_wallclock_time=300,  # How long the optimization process may run in seconds
		max_num_trials_completed=10, # Number of different hyperparameter configurations which are evaluated
		),
    n_workers=1,  # how many trials are evaluated in parallel
)

# Start tuning
tuner.run()

# We load relevant information stored by Syne Tune.
tuning_experiment = load_experiment(tuner.name)
```

from syne_tune import Tuner, StoppingCriterion from syne_tune.experiments import load_experiment from syne_tune.backend import PythonBackend from syne_tune.optimizer.baselines import CQR

# Define the Tuner

tuner = Tuner( trial_backend=PythonBackend(tune_function=objective_function, config_space=config_space), scheduler=scheduler, stop_criterion=StoppingCriterion( #max_wallclock_time=300, # How long the optimization process may run in seconds max_num_trials_completed=10, # Number of different hyperparameter configurations which are evaluated ), n_workers=1, # how many trials are evaluated in parallel )

# Start tuning

tuner.run()

# We load relevant information stored by Syne Tune.

tuning_experiment = load_experiment(tuner.name)

```
🚀 Syne Tune - Hyperparameter Optimization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 Experiment Configuration
├─ Name: python-entrypoint-2026-11-13-14-05-18-725
├─ Backend: PythonBackend
├─ Workers: 1
├─ Scheduler: CQR
├─ Results Path: /home/janew/syne-tune/python-entrypoint-2026-11-13-14-05-18-725
└─ Log Path: /home/janew/syne-tune/python-entrypoint-2026-11-13-14-05-18-725

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏁 Starting hyperparameter optimization...
[15:05:18] 🚀 Trial 0 started - config: epochs=100, checkpoint_interval=25, loss_reduction=sum, batch_size=1240, drop_p=0.5747568679582536, enc_factor=4, weight_decay=0.005350551637553704, beta=0.011103838225001075, learning_rate=2.196641844192667e-05, n_l
... [output truncated]
```

```
[15:05:39] ✅ Trial 0 completed!
[15:05:39] 🚀 Trial 1 started - config: epochs=100, checkpoint_interval=25, loss_reduction=sum, batch_size=3786, drop_p=0.3247605317064866, enc_factor=2, weight_decay=0.00024235043429375263, beta=0.01681450209847977, learning_rate=0.029046772340186336, n_layers=2, tasks=condition
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Tuning Status (last metric is reported)
 trial_id     status  iter  epochs  checkpoint_interval loss_reduction  batch_size   drop_p  enc_factor  weight_decay     beta  learning_rate  n_layers     tasks  downstream_performance  reconstruction_loss  worker-time
        0  Completed     1     100                   25            sum        1240 0.574757           4      0.005351 0.011104       0.000022         5 condition                0.526772           155.967031     0.000008
        1 InProgress     0     100                   25            sum        3786 0.324761           2      0.000242 0.016815       0.029047         2 condition                       -                    -            -
1 trials runnin
... [output truncated]
```

```
[15:06:19] ✅ Trial 2 completed!
[15:06:19] 🚀 Trial 3 started - config: epochs=100, checkpoint_interval=25, loss_reduction=sum, batch_size=3246, drop_p=0.004179913989218709, enc_factor=5, weight_decay=0.0006284349564596045, beta=0.052941758544331016, learning_rate=1.1022866113739377e-05, n_layers=5, tasks=condition
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Tuning Status (last metric is reported)
 trial_id     status  iter  epochs  checkpoint_interval loss_reduction  batch_size   drop_p  enc_factor  weight_decay     beta  learning_rate  n_layers     tasks  downstream_performance  reconstruction_loss  worker-time
        0  Completed     1     100                   25            sum        1240 0.574757           4      0.005351 0.011104       0.000022         5 condition                0.526772           155.967031     0.000008
        1  Completed     1     100                   25            sum        3786 0.324761           2      0.000242 0.016815       0.029047         2 condition                0.889065            63.816455     0.000008
        2  Comp
... [output truncated]
```

```
[15:06:40] ✅ Trial 3 completed!
[15:06:40] 🚀 Trial 4 started - config: epochs=100, checkpoint_interval=25, loss_reduction=sum, batch_size=3518, drop_p=0.1269391288856593, enc_factor=2, weight_decay=0.001414133015620124, beta=0.12267351975687857, learning_rate=0.000487182457793855, n_layers=5, tasks=condition
```

```
[15:07:00] ✅ Trial 4 completed!
```

```
[15:07:00] 🚀 Trial 5 started - config: epochs=100, checkpoint_interval=25, loss_reduction=sum, batch_size=2711, drop_p=0.5656734364212888, enc_factor=1, weight_decay=1.0510498995736113e-05, beta=8.386935396298032, learning_rate=0.00048700893048800466, n_layers=2, tasks=condition
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Tuning Status (last metric is reported)
 trial_id     status  iter  epochs  checkpoint_interval loss_reduction  batch_size   drop_p  enc_factor  weight_decay     beta  learning_rate  n_layers     tasks  downstream_performance  reconstruction_loss  worker-time
        0  Completed     1     100                   25            sum        1240 0.574757           4      0.005351 0.011104       0.000022         5 condition                0.526772           155.967031     0.000008
        1  Completed     1     100                   25            sum        3786 0.324761           2      0.000242 0.016815       0.029047         2 condition                0.889065            63.816455     0.000008
        2  Comp
... [output truncated]
```

```
[15:07:21] ✅ Trial 5 completed!
```

```
[15:07:21] 🚀 Trial 6 started - config: epochs=100, checkpoint_interval=25, loss_reduction=sum, batch_size=3706, drop_p=0.49371205736798757, enc_factor=1, weight_decay=0.018035789439704544, beta=5.6702969205585275e-05, learning_rate=0.024104395594215886, n_layers=3, tasks=condition
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Tuning Status (last metric is reported)
 trial_id     status  iter  epochs  checkpoint_interval loss_reduction  batch_size   drop_p  enc_factor  weight_decay     beta  learning_rate  n_layers     tasks  downstream_performance  reconstruction_loss  worker-time
        0  Completed     1     100                   25            sum        1240 0.574757           4      0.005351 0.011104       0.000022         5 condition                0.526772           155.967031     0.000008
        1  Completed     1     100                   25            sum        3786 0.324761           2      0.000242 0.016815       0.029047         2 condition                0.889065            63.816455     0.000008
        2  Comp
... [output truncated]
```

```
[15:07:42] ✅ Trial 6 completed!
```

```
[15:07:42] 🚀 Trial 7 started - config: epochs=100, checkpoint_interval=25, loss_reduction=sum, batch_size=3911, drop_p=0.40399205840381613, enc_factor=1, weight_decay=0.0028441291840514567, beta=0.0064200768285000214, learning_rate=0.06646590699020742, n_layers=2, tasks=condition
```

```
[15:07:57] ✅ Trial 7 completed!
```

```
[15:07:58] 🚀 Trial 8 started - config: epochs=100, checkpoint_interval=25, loss_reduction=sum, batch_size=3794, drop_p=0.13160172244036167, enc_factor=5, weight_decay=0.01851016049118515, beta=0.0014701692625455611, learning_rate=0.03887832772615192, n_layers=5, tasks=condition
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Tuning Status (last metric is reported)
 trial_id     status  iter  epochs  checkpoint_interval loss_reduction  batch_size   drop_p  enc_factor  weight_decay     beta  learning_rate  n_layers     tasks  downstream_performance  reconstruction_loss  worker-time
        0  Completed     1     100                   25            sum        1240 0.574757           4      0.005351 0.011104       0.000022         5 condition                0.526772           155.967031     0.000008
        1  Completed     1     100                   25            sum        3786 0.324761           2      0.000242 0.016815       0.029047         2 condition                0.889065            63.816455     0.000008
        2  Comp
... [output truncated]
```

```
[15:08:24] ✅ Trial 8 completed!
```

```
[15:08:24] 🚀 Trial 9 started - config: epochs=100, checkpoint_interval=25, loss_reduction=sum, batch_size=3865, drop_p=0.48512150451951247, enc_factor=1, weight_decay=0.007080645705697765, beta=0.0006179422392261735, learning_rate=0.07806784205123984, n_layers=2, tasks=condition
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Tuning Status (last metric is reported)
 trial_id     status  iter  epochs  checkpoint_interval loss_reduction  batch_size   drop_p  enc_factor  weight_decay     beta  learning_rate  n_layers     tasks  downstream_performance  reconstruction_loss  worker-time
        0  Completed     1     100                   25            sum        1240 0.574757           4      0.005351 0.011104       0.000022         5 condition                0.526772           155.967031     0.000008
        1  Completed     1     100                   25            sum        3786 0.324761           2      0.000242 0.016815       0.029047         2 condition                0.889065            63.816455     0.000008
        2  Comp
... [output truncated]
```

```
[15:08:45] ✅ Trial 9 completed!
```

```
[15:08:46] 🚀 Trial 10 started - config: epochs=100, checkpoint_interval=25, loss_reduction=sum, batch_size=3810, drop_p=0.33888069983035984, enc_factor=4, weight_decay=6.198384625530537e-05, beta=0.010645874058964128, learning_rate=0.03816015621637652, n_layers=2, tasks=condition
```

```
[15:09:06] ✅ Trial 10 completed!
```

```
[15:09:06] 🚀 Trial 11 started - config: epochs=100, checkpoint_interval=25, loss_reduction=sum, batch_size=4009, drop_p=0.05708373102710941, enc_factor=2, weight_decay=0.00100654704069504, beta=3.087305254755213, learning_rate=0.02913068140562028, n_layers=4, tasks=condition

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎉🎉🎉 HYPERPARAMETER OPTIMIZATION COMPLETE!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

--------------------
Resource summary (last result is reported):
 trial_id     status  iter  epochs  checkpoint_interval loss_reduction  batch_size   drop_p  enc_factor  weight_decay     beta  learning_rate  n_layers     tasks  downstream_performance  reconstruction_loss  worker-time
        0  Completed     1     1
... [output truncated]
```

```
🛑 Stopping trials that may still be running.
✅ Tuning finished, results of trials can be found at /home/janew/syne-tune/python-entrypoint-2026-11-13-14-05-18-725

💾 Results saved to: /home/janew/syne-tune/python-entrypoint-2026-11-13-14-05-18-725
✨ Happy training with your optimized hyperparameters!
```

## Visualizing the Optimization process[¶](#visualizing-the-optimization-process)

### Best Performance over Time[¶](#best-performance-over-time)

In \[6\]:

Copied!

```
tuning_experiment.plot()
```

tuning_experiment.plot()

```
.venv/lib/python3.12/site-packages/syne_tune/experiments/experiment_result.py:139: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  fig.show()
```

### Scatterplot of all trials[¶](#scatterplot-of-all-trials)

In \[7\]:

Copied!

```
tuning_experiment.plot_trials_over_time()
```

tuning_experiment.plot_trials_over_time()

```
.venv/lib/python3.12/site-packages/syne_tune/experiments/experiment_result.py:178: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  fig.show()
```

### Comparing tuning on Reconstruction Loss vs. Downstream Performance[¶](#comparing-tuning-on-reconstruction-loss-vs-downstream-performance)

In \[8\]:

Copied!

```
import numpy as np

import matplotlib.pyplot as plt

y = tuning_experiment.results.reconstruction_loss
x = tuning_experiment.results.downstream_performance

corr = np.corrcoef(x, y)[0, 1]

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.6)
plt.xlabel("Downstream Performance")
plt.ylabel("Reconstruction Loss")
plt.title("Downstream Performance vs Reconstruction Loss")
# plt.xlim(0.8, 1.0) ## Limit to good and very good runs
# plt.ylim(500, 600) ##

plt.text(
	0.05, 0.95,
	f"Correlation: {corr:.3f}",
	transform=plt.gca().transAxes,
	fontsize=12,
	verticalalignment='top',
	bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
)

plt.show()
```

import numpy as np import matplotlib.pyplot as plt y = tuning_experiment.results.reconstruction_loss x = tuning_experiment.results.downstream_performance corr = np.corrcoef(x, y)[0, 1] plt.figure(figsize=(8, 6)) plt.scatter(x, y, alpha=0.6) plt.xlabel("Downstream Performance") plt.ylabel("Reconstruction Loss") plt.title("Downstream Performance vs Reconstruction Loss")

# plt.xlim(0.8, 1.0) ## Limit to good and very good runs

# plt.ylim(500, 600)

plt.text( 0.05, 0.95, f"Correlation: {corr:.3f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray') ) plt.show()

## Selecting and Saving the Best Configuration[¶](#selecting-and-saving-the-best-configuration)

After the optimization finishes, the experiment results can be loaded and queried for the best configuration. The “best” configuration is defined as the one that achieved the highest (or lowest, depending on the metric) reported performance on the validation data.

This configuration can then be reused for final training or further experiments.

______________________________________________________________________

In \[9\]:

Copied!

```
print(f"best result found: {tuning_experiment.best_config()}")

import pickle

with open("tuning_experiment.pkl", "wb") as f:
    pickle.dump(tuning_experiment, f)

print("Saved tuning_experiment.pkl")

with open("tuning_experiment.pkl", "rb") as f:
# with open("./large_ontix_save/large_ontix_syne_tune_experiment_recon.pkl", "rb") as f:
	tuning_experiment = pickle.load(f)
print("loaded tuning_experiment.pkl")
```

print(f"best result found: {tuning_experiment.best_config()}") import pickle with open("tuning_experiment.pkl", "wb") as f: pickle.dump(tuning_experiment, f) print("Saved tuning_experiment.pkl") with open("tuning_experiment.pkl", "rb") as f:

# with open("./large_ontix_save/large_ontix_syne_tune_experiment_recon.pkl", "rb") as f:

tuning_experiment = pickle.load(f) print("loaded tuning_experiment.pkl")

```
best result found: {'downstream_performance': np.float64(0.8890648567119156), 'reconstruction_loss': np.float64(63.816455078125), 'trial_id': np.int64(1), 'config_epochs': np.int64(100), 'config_checkpoint_interval': np.int64(25), 'config_loss_reduction': 'sum', 'config_batch_size': np.int64(3786), 'config_drop_p': np.float64(0.3247605317064866), 'config_enc_factor': np.int64(2), 'config_weight_decay': np.float64(0.0002423504342937), 'config_beta': np.float64(0.0168145020984797), 'config_learning_rate': np.float64(0.0290467723401863), 'config_n_layers': np.int64(2), 'config_tasks': 'condition'}
Saved tuning_experiment.pkl
loaded tuning_experiment.pkl
```
