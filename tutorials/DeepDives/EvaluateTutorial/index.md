> **Note:** This environment variable is required for fully deterministic CuBLAS ops on CUDA >= 10.2 when `reproducible=True` is set below. Without it, PyTorch raises a `RuntimeError` instead of training deterministically. It must be set before `torch` is imported. See the [README FAQ](https://jan-forest.github.io/autoencodix_package/tutorials/README.md#reproducibility-and-cublas_workspace_config) for details.

In \[ \]:

Copied!

```
%env CUBLAS_WORKSPACE_CONFIG=:16:8
```

%env CUBLAS_WORKSPACE_CONFIG=:16:8

# Evaluate Overview[¶](#evaluate-overview)

This tutorial provides an overview of the in-built functionalities to evaluate the learned embeddings for downstream machine learning tasks leveraging metadata information.

Evaluation is designed as additional pipeline step after `.run()` (which includes preprocessing, training, predict on test data and visualization).

When you work with a pre-trained model, make sure that you perform the `.predict()` step such that embeddings are calculated for newly provided test-samples.

## Stackix example[¶](#stackix-example)

We will use a Stackix as an example to show options in embedding evaluation.

In \[1\]:

Copied!

```
# Train the Stackix similar to pipeline tutorial
import autoencodix as acx
from autoencodix.configs.default_config import DataCase
from autoencodix.configs.stackix_config import StackixConfig
from autoencodix.utils.example_data import EXAMPLE_MULTI_BULK

# Stackix has its own config class
# instead of passing a pandas DataFrame, we use a pre-defined DataPackage object directly.
# this time with single cell data
print("Input data:")
print(EXAMPLE_MULTI_BULK)
print("-" * 50)
my_config = StackixConfig(
    epochs=100,
    checkpoint_interval=5,
    default_vae_loss="kl",  
    data_case=DataCase.MULTI_BULK,
)
print("\n")
print("Starting Pipeline")
print("-" * 50)
print("-" * 50)
stackix = acx.Stackix(data=EXAMPLE_MULTI_BULK, config=my_config)
result = stackix.run()
```

# Train the Stackix similar to pipeline tutorial

import autoencodix as acx from autoencodix.configs.default_config import DataCase from autoencodix.configs.stackix_config import StackixConfig from autoencodix.utils.example_data import EXAMPLE_MULTI_BULK

# Stackix has its own config class

# instead of passing a pandas DataFrame, we use a pre-defined DataPackage object directly.

# this time with single cell data

print("Input data:") print(EXAMPLE_MULTI_BULK) print("-" * 50) my_config = StackixConfig( epochs=100, checkpoint_interval=5, default_vae_loss="kl", data_case=DataCase.MULTI_BULK, ) print("\\n") print("Starting Pipeline") print("-" * 50) print("-" * 50) stackix = acx.Stackix(data=EXAMPLE_MULTI_BULK, config=my_config) result = stackix.run()

```
Input data:
multi_bulk:
  transcriptomics: 500 samples × 100 features
  proteomics: 500 samples × 80 features
annotation:
  transcriptomics: 500 samples × 3 features
  proteomics: 500 samples × 3 features
--------------------------------------------------


Starting Pipeline
--------------------------------------------------
--------------------------------------------------
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
anno key: transcriptomics
anno key: proteomics
Training each modality model...
Training modality: transcriptomics
Training modality: transcriptomics
Epoch 1 - Train Loss: 119.1194
Sub-losses: recon_loss: 119.1194, var_loss: 0.0000, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 1 - Valid Loss: 105.4678
Sub-losses: reco
... [output truncated]
```

Before we perform embedding evaluation on a metadata parameter. Let's check visually if latent space shows discrimination regarding this parameter.

In \[2\]:

Copied!

```
my_target_param = "condition"  

stackix.visualizer.show_latent_space(
	result= result,
	plot_type="Ridgeline",
	param= [my_target_param]
)
```

my_target_param = "condition" stackix.visualizer.show_latent_space( result= result, plot_type="Ridgeline", param= [my_target_param] )

We can see a separation of conditions along several dimension. Hence, we can expect that embeddings are useful to classify.

TODO make example stackix better

Now we can call the pipeline step `stackix.evaluate()`

In \[3\]:

Copied!

```
result = stackix.evaluate(params=[my_target_param])

print("Evaluation results:")
print(result.embedding_evaluation)
```

result = stackix.evaluate(params=[my_target_param]) print("Evaluation results:") print(result.embedding_evaluation)

```
Perform ML task with feature df: Latent
Latent
Perform ML task for target parameter: condition
Evaluation results:
  score_split CLINIC_PARAM       metric     value                ML_ALG  \
0       train    condition  roc_auc_ovo  0.718757  LogisticRegression()   
1       valid    condition  roc_auc_ovo  0.752485  LogisticRegression()   
2        test    condition  roc_auc_ovo  0.673119  LogisticRegression()   

          ML_TYPE ML_TASK ML_SUBTASK  
0  classification  Latent     Latent  
1  classification  Latent     Latent  
2  classification  Latent     Latent
```

From the dataframe containing the results, we can see that default configuration of `.evaluate()` uses linear machine learning models for regression and classification tasks.

But evaluate offers many options for a detailed inspection:

- With `ml_model_class` and `ml_model_regression` user can provide any machine learning method if they are using a sklearn-like interface.
- `params` - Is the list of metadata parameter on which downstream tasks will be performed. If not a list, but the string "all" is provided, all parameters will be evaluated. Type of task classification vs. regression will be auto-detected. Make sure to avoid providing class information as integers or floats.
- With `metric_class` and `metric_regression` the metric for evaluation is specified for classification (Default: `roc_auc_ovo`) or regression (Default: `r2`). Must be one as described in sklearn: <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names>
- `split_type` - Specifies how data should be splitted for training and testing downstream ML models. Default is `use-split` which will use established split and will train on the train split. Alternatively and in case there is no train split, cross-validation can be performed by using a string like "LOOCV", "CV-5", ... "CV-10", where the number specifies the number of folds.
- `n_downsample` - To speed up evaluation on large data sets, evaluation can be performed on a random subset. Default is 10000. You may need to increase in case of extreme class-imbalance.
- `reference_methods` - For cross-refence of embedding performance you can check (provided as list):
  - "RandomFeature" will pick randomly pick input features in the same number as `latent_dim` and provides a lower performance baseline. Will be repeated five times.
  - "PCA", "UMAP" or "TSNE" will perform dimension reduction to `latent_dim` to compare autoencoder embeddings to other techniques.

Let's check this with an advanced example:

In \[4\]:

Copied!

```
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)  # Use a custom classifier from sklearn including hyperparameters
metric_class = "accuracy"  # Choose another metric to evaluate classification performance
split_type = "CV-5" # Use 5-fold cross-validation instead of the predefined train/test split
reference_methods = ["RandomFeature", "PCA"]  # Compare results to additional reference methods

result = stackix.evaluate(
	params=[my_target_param],
    ml_model_class=rf,
    metric_class=metric_class,
    split_type=split_type,
    reference_methods=reference_methods
)
```

from sklearn.ensemble import RandomForestClassifier rf = RandomForestClassifier(n_estimators=100) # Use a custom classifier from sklearn including hyperparameters metric_class = "accuracy" # Choose another metric to evaluate classification performance split_type = "CV-5" # Use 5-fold cross-validation instead of the predefined train/test split reference_methods = ["RandomFeature", "PCA"] # Compare results to additional reference methods result = stackix.evaluate( params=[my_target_param], ml_model_class=rf, metric_class=metric_class, split_type=split_type, reference_methods=reference_methods )

```
Perform ML task with feature df: RandomFeature
RandomFeature_R1
RandomFeature_R2
RandomFeature_R3
RandomFeature_R4
RandomFeature_R5
Perform ML task with feature df: PCA
PCA
Perform ML task with feature df: Latent
Latent
Perform ML task for target parameter: condition
```

Note, that evaluation results will be updated with additional evaluate calls:

In \[5\]:

Copied!

```
print("Evaluation results:")
print(result.embedding_evaluation)
```

print("Evaluation results:") print(result.embedding_evaluation)

```
Evaluation results:
   score_split CLINIC_PARAM       metric     value                    ML_ALG  \
0        train    condition  roc_auc_ovo  0.718757      LogisticRegression()   
1        valid    condition  roc_auc_ovo  0.752485      LogisticRegression()   
2         test    condition  roc_auc_ovo  0.673119      LogisticRegression()   
0         test    condition     accuracy  0.700000  RandomForestClassifier()   
1         test    condition     accuracy  0.790000  RandomForestClassifier()   
..         ...          ...          ...       ...                       ...   
5        train    condition     accuracy  1.000000  RandomForestClassifier()   
6        train    condition     accuracy  1.000000  RandomForestClassifier()   
7        train    condition     accuracy  1.000000  RandomF
... [output truncated]
```

Results can be visualized as bar plots:

In \[6\]:

Copied!

```
fig_condition_eval_linear = stackix.visualizer.show_evaluation(
	param=my_target_param, # The parameter to visualize
	metric="accuracy", # The metric we specified
	ml_alg=str(rf) # The string representation of the classifier used
)
```

fig_condition_eval_linear = stackix.visualizer.show_evaluation( param=my_target_param, # The parameter to visualize metric="accuracy", # The metric we specified ml_alg=str(rf) # The string representation of the classifier used )

## Using evaluate with non-sklearn models[¶](#using-evaluate-with-non-sklearn-models)

We can use other machine learning models which are not from sklearn, but have the same interface. An recent and powerful example is the foundation model TabPFN <https://github.com/PriorLabs/TabPFN>

In \[7\]:

Copied!

```
## Variable needs to set otherwise an error occurs when importing scipy
%env SCIPY_ARRAY_API=1
```

## Variable needs to set otherwise an error occurs when importing scipy

%env SCIPY_ARRAY_API=1

```
env: SCIPY_ARRAY_API=1
```

In \[8\]:

Copied!

```
import torch 
if torch.cuda.is_available():
	device = torch.cuda.current_device()
	torch.cuda.get_device_properties(device).total_memory / 1024**3

if torch.cuda.is_available() and torch.cuda.get_device_properties(device).total_memory > 10*1024**4:
	from tabpfn import TabPFNClassifier, TabPFNRegressor

	tabpfn_class = TabPFNClassifier()

	result = stackix.evaluate(
		params=[my_target_param],
		ml_model_class=tabpfn_class,
		metric_class=metric_class,
		split_type=split_type,
		reference_methods=reference_methods
	)
else:
	print("Get a GPU with >4GB VRAM to run TabPFN models, you poor soul.")
```

import torch if torch.cuda.is_available(): device = torch.cuda.current_device() torch.cuda.get_device_properties(device).total_memory / 1024\*\*3 if torch.cuda.is_available() and torch.cuda.get_device_properties(device).total_memory > 10\*1024\*\*4: from tabpfn import TabPFNClassifier, TabPFNRegressor tabpfn_class = TabPFNClassifier() result = stackix.evaluate( params=[my_target_param], ml_model_class=tabpfn_class, metric_class=metric_class, split_type=split_type, reference_methods=reference_methods ) else: print("Get a GPU with >4GB VRAM to run TabPFN models, you poor soul.")

```
Get a GPU with >4GB VRAM to run TabPFN models, you poor soul.
```

In \[9\]:

Copied!

```
if torch.cuda.is_available():
	fig_condition_eval_tabpfn = stackix.visualizer.show_evaluation(
		param=my_target_param, # The parameter to visualize
		metric="accuracy", # The metric we specified
		ml_alg=str(tabpfn_class) # The string representation of the classifier used
	)
```

if torch.cuda.is_available(): fig_condition_eval_tabpfn = stackix.visualizer.show_evaluation( param=my_target_param, # The parameter to visualize metric="accuracy", # The metric we specified ml_alg=str(tabpfn_class) # The string representation of the classifier used )

# XModalix specifics[¶](#xmodalix-specifics)

## Embedding evaluation[¶](#embedding-evaluation)

Since XModalix is composed of multiple VAE for each modality, evaluate behaves slightly different. In the case embedding evaluation it will perform evaluation for each VAE (modality) and test its performance for the specified downstream task.

## Pure VAE comparison[¶](#pure-vae-comparison)

The function `xmodalix.evaluator.pure_vae_comparison()` is designed to evaluate the reconstruction capability of a XModalix (reference and translated) against a normal Imagix on the image target modality. This evaluation provides insights whether the enforced latent space alignment and coupled training in the XModalix decreases or improves the reconstruction capability in comparison to image-only autoencoder.

The prerequisite for this comparison is the prior training of an imagix. An example is provided in the [XModalix tutorial](https://jan-forest.github.io/autoencodix_package/tutorials/DeepDives/PipelineTutorials/XModalix.ipynb).
