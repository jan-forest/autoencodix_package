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

# How To Use Varix[¶](#how-to-use-varix)

Varix is our implementation of a variational autoencoder.\
This tutorial follows the structure of our `Getting Started - Vanillix`, but is much less extensive, because\
our pipeline works similarly for different architectures, so here we focus only on Varix specifics.

**AUTOENCODIX** supports far more functionality than shown here, so we’ll also point to advanced tutorials where relevant.

**IMPORTANT**

> This tutorial only shows the specifics of the Varix pipeline. If you're unfamiliar with general concepts,\
> we recommend following the [`Getting Started - Vanillix`](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/Varix/Vanillix.ipynb) tutorial first.

## What You'll Learn[¶](#what-youll-learn)

You’ll learn how to:

1. **Initialize** the pipeline and run the pipeline.
1. Understand the Varix-specific **pipeline steps**.
1. Access the Varix-specific **results** (mus, sigma, KL/MMD losses).
1. **Visualize** outputs effectively.
1. Apply **custom parameters**.
1. **Save, load, and reuse** a trained pipeline.

Let’s get started! 🚀

## Requirements 1: Be in the correct directory (execute below)[¶](#requirements-1-be-in-the-correct-directory-execute-below)

In \[2\]:

Copied!

```
import os

p = os.getcwd()
d = "autoencodix_package"
if d not in p:
    raise FileNotFoundError(f"'{d}' not found in path: {p}")
os.chdir(os.sep.join(p.split(os.sep)[: p.split(os.sep).index(d) + 1]))
print(f"Changed to: {os.getcwd()}")
```

import os p = os.getcwd() d = "autoencodix_package" if d not in p: raise FileNotFoundError(f"'{d}' not found in path: {p}") os.chdir(os.sep.join(p.split(os.sep)[: p.split(os.sep).index(d) + 1])) print(f"Changed to: {os.getcwd()}")

```
Changed to: /home/alicia/dev/biomarker_autoencoder/autoencodix_package
```

## Requirements 2: Obtain tutorial data or use own data[¶](#requirements-2-obtain-tutorial-data-or-use-own-data)

We use real human lung single-cell RNA-seq data assembled from the [CZ CELLxGENE Census](https://cellxgene.cziscience.com/). It is hosted on Hugging Face Hub ([autoencodix/census-lung](https://huggingface.co/datasets/autoencodix/census-lung)) and is downloaded automatically in the cells below on first run — no manual download or placement into `data/raw` needed. The full dataset contains ~89,000 cells spanning four clearly distinguishable lung cell types (malignant cells, pulmonary alveolar type 2 cells, respiratory basal cells, alveolar macrophages). To keep this tutorial fast to run, we draw a balanced subsample of 2,000 cells per cell type below. Alternatively you can use your own single-cell dataset (Varix works with other datatypes, but in this example we expect single-cell data: `h5ad`).

In \[3\]:

Copied!

```
import numpy as np
import anndata as ad
import mudata

from huggingface_hub import hf_hub_download
from autoencodix.data.datapackage import DataPackage

sc_path = hf_hub_download(
    repo_id="autoencodix/census-lung", repo_type="dataset", filename="census_lung.h5ad"
)
full_adata = ad.read_h5ad(sc_path)

# Balanced subsample: 2,000 cells per cell type keeps the tutorial fast to run
# while keeping all four (well-separated) cell types equally represented.
N_PER_TYPE = 2000
rng = np.random.RandomState(42)
subsample_idx = []
for _, group in full_adata.obs.groupby("cell_type", observed=True):
    n = min(N_PER_TYPE, len(group))
    subsample_idx.extend(rng.choice(group.index.values, size=n, replace=False))

adata = full_adata[subsample_idx].copy()
print(adata.obs["cell_type"].value_counts())

# Instead of passing a pandas DataFrame, we wrap the AnnData in a MuData object
# inside our custom DataPackage structure, this time with single-cell data.
sc_data = DataPackage()
sc_data.multi_sc = {"multi_sc": mudata.MuData({"rna": adata})}
```

import numpy as np import anndata as ad import mudata from huggingface_hub import hf_hub_download from autoencodix.data.datapackage import DataPackage sc_path = hf_hub_download( repo_id="autoencodix/census-lung", repo_type="dataset", filename="census_lung.h5ad" ) full_adata = ad.read_h5ad(sc_path)

# Balanced subsample: 2,000 cells per cell type keeps the tutorial fast to run

# while keeping all four (well-separated) cell types equally represented.

N_PER_TYPE = 2000 rng = np.random.RandomState(42) subsample_idx = [] for \_, group in full_adata.obs.groupby("cell_type", observed=True): n = min(N_PER_TYPE, len(group)) subsample_idx.extend(rng.choice(group.index.values, size=n, replace=False)) adata = full_adata[subsample_idx].copy() print(adata.obs["cell_type"].value_counts())

# Instead of passing a pandas DataFrame, we wrap the AnnData in a MuData object

# inside our custom DataPackage structure, this time with single-cell data.

sc_data = DataPackage() sc_data.multi_sc = {"multi_sc": mudata.MuData({"rna": adata})}

```
cell_type
alveolar macrophage               2000
malignant cell                    2000
pulmonary alveolar type 2 cell    2000
respiratory basal cell            2000
Name: count, dtype: int64
```

## 1) Initialize and Run Varix[¶](#1-initialize-and-run-varix)

We set a few custom parameters of the config file. For a deep dive into the config object, see:

[`Tutorials/DeepDives/ConfigTutorial.ipynb`](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/DeepDives/ConfigTutorial.ipynb)

#### 1.1 The Dataset[¶](#11-the-dataset)

We use the `sc_data` `DataPackage` prepared above (the census-lung dataset wrapped in an `AnnData`/`MuData` object) and color the downstream plots by the `cell_type` metadata column.

In \[4\]:

Copied!

```
from autoencodix.configs.varix_config import VarixConfig
from autoencodix.configs.default_config import DataCase, DataConfig, DataInfo
import autoencodix as acx

# Varix has its own config class
# with an additional loss term: either Kullback-Leibler (KL) or Maximum Mean Discrepancy (MMD)
my_config = VarixConfig(
    learning_rate=0.001,
    epochs=30,
    checkpoint_interval=5,
    default_vae_loss="kl",  # 'kl' or 'mmd' possible
    anneal_function="logistic-mid",
    k_filter=2000,
    data_case=DataCase.MULTI_SINGLE_CELL,
    annotation_columns=["cell_type"],
    data_config=DataConfig(
        data_info={"multi_sc": DataInfo(is_single_cell=True, data_type="NUMERIC")},
    ),
)

print("\nStarting Pipeline")
print("-" * 50)
varix = acx.Varix(data=sc_data, config=my_config)
result = varix.run()
```

from autoencodix.configs.varix_config import VarixConfig from autoencodix.configs.default_config import DataCase, DataConfig, DataInfo import autoencodix as acx

# Varix has its own config class

# with an additional loss term: either Kullback-Leibler (KL) or Maximum Mean Discrepancy (MMD)

my_config = VarixConfig( learning_rate=0.001, epochs=30, checkpoint_interval=5, default_vae_loss="kl", # 'kl' or 'mmd' possible anneal_function="logistic-mid", k_filter=2000, data_case=DataCase.MULTI_SINGLE_CELL, annotation_columns=["cell_type"], data_config=DataConfig( data_info={"multi_sc": DataInfo(is_single_cell=True, data_type="NUMERIC")}, ), ) print("\\nStarting Pipeline") print("-" * 50) varix = acx.Varix(data=sc_data, config=my_config) result = varix.run()

```
Starting Pipeline
--------------------------------------------------
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
mudata: View of MuData object with n_obs × n_vars = 8000 × 8189
  var:	'soma_joinid', 'feature_id', 'feature_name', 'feature_type', 'feature_length', 'nnz', 'n_measured_obs'
  1 modality
    rna:	8000 x 8189
      obs:	'soma_joinid', 'dataset_id', 'assay', 'assay_ontology_term_id', 'cell_type', 'cell_type_ontology_term_id', 'development_stage', 'development_stage_ontology_term_id', 'disease', 'disease_ontology_term_id', 'donor_id', 'is_primary_data', 'observation_joinid', 'self_reported_ethnicity', 'self_reported_ethnicity_ontology_term_id', 'sex', 'sex_ontology_term_id', 'suspension_type', 'tissue', 'tissue_ontology_term_id', 'tiss
... [output truncated]
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:233: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 13.2772 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 5 - Train Loss: 1772.1442
Sub-losses: recon_loss: 1772.1319, var_loss: 0.0123, anneal_factor: 0.0007, effective_beta_factor: 0.0001
Epoch 5 - Valid Loss: 1632.8667
Sub-losses: recon_loss: 1632.8576, var_loss: 0.0092, anneal_factor: 0.0007, effective_beta_factor: 0.0001
Epoch 10 - Train Loss: 1720.7331
Sub-losses: recon_loss: 1720.1487, var_loss: 0.5844, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Epoch 10 - Valid Loss: 1593.5738
Sub-losses: recon_loss: 1593.0346, var_loss: 0.5393, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Epoch 15 - Train Loss: 1710.1009
Sub-losses: recon_loss: 1701.1942, var_loss: 8.9067, anneal_factor: 0.3392, effective_beta_factor: 0.0339
Epoch 15 - Valid Loss: 1585.0462
Sub-losses: recon_loss: 1577.8735, var_loss: 7.1727, anneal_factor: 0.339
... [output truncated]
```

## 2) Specific Varix Steps (with Beta Annealing Explanation)[¶](#2-specific-varix-steps-with-beta-annealing-explanation)

The pipeline (`run`) does not have Varix-specific steps.\
It calls:

- preprocess
- fit
- predict
- visualize

However, the fit step works slightly differently because we have a second loss term for the distribution loss.\
This loss term can be weighted with a hyperparameter `beta` (set via config), and the weighting can be changed during training depending on the epoch via **beta annealing**.

**Beta annealing** gradually increases the weight of the VAE distribution loss during training.\
For example, you might start with `beta=0` to let the reconstruction loss dominate early, and then gradually increase `beta` to 1 as training progresses.\
Different annealing strategies are supported, including logistic schedules, multi-phase linear or logarithmic schedules, and constant or no annealing.\
This allows more stable training and better latent space learning, especially for early epochs.

Since we're working with a variational autoencoder, we can call an additional pipeline step, `sample_latent_space`.\
The latent space inside the pipeline is sampled once from the fitted normal distribution, with one set of `mu` and `sigma` per latent dimension.

By default, we sample from the trained model and the test data, but you can specify the split and epoch.

**ATTENTION**

> If you want to sample from a different split (uses data from that split as input) or epoch (uses the model at this epoch), you need to ensure you select an epoch that was checkpointed (see `checkpoint_interval` config parameter).

In \[5\]:

Copied!

```
print(f"Annealing Strategy for training was: {my_config.anneal_function}")
print(f"beta hyperparam was set to: {my_config.beta}")
print("\n")
print("Starting sampling different latent spaces")
print("-"*50)
latent_spaces_sampled = varix.sample_latent_space(n_samples=100)
print(f"sampled latent space shape: {latent_spaces_sampled.shape}")

# or
sampled = varix.sample_latent_space(split="train", epoch=4,n_samples=50)
sampled
```

print(f"Annealing Strategy for training was: {my_config.anneal_function}") print(f"beta hyperparam was set to: {my_config.beta}") print("\\n") print("Starting sampling different latent spaces") print("-"\*50) latent_spaces_sampled = varix.sample_latent_space(n_samples=100) print(f"sampled latent space shape: {latent_spaces_sampled.shape}")

# or

sampled = varix.sample_latent_space(split="train", epoch=4,n_samples=50) sampled

```
Annealing Strategy for training was: logistic-mid
beta hyperparam was set to: 0.1


Starting sampling different latent spaces
--------------------------------------------------
sampled latent space shape: torch.Size([100, 16])
```

Out\[5\]:

```
tensor([[ 3.9897,  4.1779,  1.8247,  2.3360,  1.1621,  6.5704,  4.2918,  0.8106,
          1.2946, -1.0189,  3.0731,  3.7557,  1.0169,  5.1850,  3.3394,  2.4349],
        [ 5.4976,  3.6496,  2.7947,  1.3314,  2.6984,  5.9357,  3.2230, -1.7498,
          0.4365,  0.9875, -0.4079,  5.1207,  3.0713,  3.0493,  3.3609,  2.9824],
        [ 4.2046,  2.2885,  4.5450,  0.0513,  1.7615,  7.2449,  4.5604,  0.9969,
          2.8329,  3.4808,  2.3766,  3.0640,  1.7307,  6.3812,  3.9349,  1.3993],
        [ 4.6709,  3.0498,  1.0624,  1.7734,  0.0657,  5.5635,  3.4210, -2.4210,
         -0.4838,  1.5256, -0.5709,  5.5177,  3.0401,  4.6961,  2.1426,  2.6823],
        [ 4.4480,  2.9986,  3.1832,  1.2259,  3.6005,  5.0852,  3.9627, -0.7497,
          1.2437,  2.3503,  0.5664,  6.6973,  3.5103,  4.7849,  2.
... [output truncated]
```

## 3) Inspect Varix-Specific Results[¶](#3-inspect-varix-specific-results)

In addition to the results that the Vanillix pipeline provided, we can access:

- Fitted distribution parameters `mu` and `sigma`
- `total`, `reconstruction`, and `vae` losses
- `anneal_factor`

A note on the different loss types:\
For our variational autoencoder, the total loss consists of a reconstruction loss and a distribution loss (i.e., KL-divergence).\
To investigate these losses, the `result` object has the attribute `sub_losses`.\
This is a `LossRegistry` with the name of the loss as the key, and the value is a `TrainingDynamics` object, which can be accessed in the same way as for the Vanillix results.

For more details, check [`Tutorials/DeepDives/PipelineOutputTutorial.ipynb`](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/DeepDives/PipelineOutputTutorial.ipynb).

In \[6\]:

Copied!

```
sub_losses = result.sub_losses
print("Sub Losses:")
print(f"keys: {sub_losses.keys()}")
print("\n")
recon_dyn = sub_losses.get(key="recon_loss")
print("Value of reconstruction loss in epoch 4 for train split")
print(recon_dyn.get(split="train", epoch=4))
```

sub_losses = result.sub_losses print("Sub Losses:") print(f"keys: {sub_losses.keys()}") print("\\n") recon_dyn = sub_losses.get(key="recon_loss") print("Value of reconstruction loss in epoch 4 for train split") print(recon_dyn.get(split="train", epoch=4))

```
Sub Losses:
keys: dict_keys(['recon_loss', 'var_loss', 'anneal_factor', 'effective_beta_factor'])


Value of reconstruction loss in epoch 4 for train split
1772.131879185268
```

## 4) Show Visualizations[¶](#4-show-visualizations)

This follows the standard pipeline process and can be done by calling `show_result()`.\
We can also add the keyword argument `params` to `show_result()`, which colors the plots according to a metadata column.

In \[7\]:

Copied!

```
varix.result.datasets.test.metadata.head()

varix.show_result(params=["cell_type"])
```

varix.result.datasets.test.metadata.head() varix.show_result(params=["cell_type"])

```
Creating plots ...
```

In \[8\]:

Copied!

```
varix.visualizer.show_loss(plot_type="relative")
```

varix.visualizer.show_loss(plot_type="relative")

In \[9\]:

Copied!

```
# A quick look at the real metadata available for coloring plots:
adata.obs[["cell_type", "disease", "sex"]].value_counts()
```

# A quick look at the real metadata available for coloring plots:

adata.obs\[["cell_type", "disease", "sex"]\].value_counts()

Out\[9\]:

```
cell_type                       disease              sex    
malignant cell                  lung adenocarcinoma  male       1196
alveolar macrophage             normal               male       1012
respiratory basal cell          normal               unknown     894
pulmonary alveolar type 2 cell  normal               male        792
                                                     female      765
malignant cell                  lung adenocarcinoma  female      730
respiratory basal cell          cystic fibrosis      unknown     542
alveolar macrophage             normal               female      539
                                                     unknown     449
pulmonary alveolar type 2 cell  normal               unknown     443
respiratory basal cell          normal
... [output truncated]
```

## 5) Customize Varix[¶](#5-customize-varix)

To customize the behavior of our pipeline, you adjust the configuration.\
There are two ways to work with the config:

1. Create a customized instance of the config class.
1. Provide a `yaml` file and use the config class to read it.

We will focus on option 1 and show a few examples. For a deeper dive into configurations, please refer to [`Tutorials/DeepDives/ConfigTutorial.ipynb`](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/DeepDives/ConfigTutorial.ipynb)

In this section, we demonstrate some Varix-specific parameters:

- Loss term
- Beta
- Annealing strategy
- Retrieve information about all config parameters

In \[10\]:

Copied!

```
from autoencodix.configs import VarixConfig

custom_config = VarixConfig(
    default_vae_loss="mmd",
    beta=1.5,
    anneal_function="3phase-linear",
    data_case=DataCase.MULTI_SINGLE_CELL,
    epochs=50,
    k_filter=2000,
    annotation_columns=["cell_type"],
    data_config=DataConfig(
        data_info={"multi_sc": DataInfo(is_single_cell=True, data_type="NUMERIC")},
    ),
)

custom_varix = acx.Varix(config=custom_config, data=sc_data)
result = custom_varix.run()
```

from autoencodix.configs import VarixConfig custom_config = VarixConfig( default_vae_loss="mmd", beta=1.5, anneal_function="3phase-linear", data_case=DataCase.MULTI_SINGLE_CELL, epochs=50, k_filter=2000, annotation_columns=["cell_type"], data_config=DataConfig( data_info={"multi_sc": DataInfo(is_single_cell=True, data_type="NUMERIC")}, ), ) custom_varix = acx.Varix(config=custom_config, data=sc_data) result = custom_varix.run()

```
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
mudata: View of MuData object with n_obs × n_vars = 8000 × 8189
  var:	'soma_joinid', 'feature_id', 'feature_name', 'feature_type', 'feature_length', 'nnz', 'n_measured_obs'
  1 modality
    rna:	8000 x 8189
      obs:	'soma_joinid', 'dataset_id', 'assay', 'assay_ontology_term_id', 'cell_type', 'cell_type_ontology_term_id', 'development_stage', 'development_stage_ontology_term_id', 'disease', 'disease_ontology_term_id', 'donor_id', 'is_primary_data', 'observation_joinid', 'self_reported_ethnicity', 'self_reported_ethnicity_ontology_term_id', 'sex', 'sex_ontology_term_id', 'suspension_type', 'tissue', 'tissue_ontology_term_id', 'tissue_type', 'tissue_general', 'tissue_general_ontology_term_id', 'raw_su
... [output truncated]
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:233: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 11.8769 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 10 - Train Loss: 1716.3193
Sub-losses: recon_loss: 1716.3193, var_loss: 0.0000, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 10 - Valid Loss: 1702.9074
Sub-losses: recon_loss: 1702.9074, var_loss: 0.0000, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 20 - Train Loss: 1682.3635
Sub-losses: recon_loss: 1677.4623, var_loss: 4.9012, anneal_factor: 0.1400, effective_beta_factor: 0.2100
Epoch 20 - Valid Loss: 1681.6059
Sub-losses: recon_loss: 1676.8131, var_loss: 4.7928, anneal_factor: 0.1400, effective_beta_factor: 0.2100
Epoch 30 - Train Loss: 1678.1424
Sub-losses: recon_loss: 1661.8913, var_loss: 16.2512, anneal_factor: 0.7400, effective_beta_factor: 1.1100
Epoch 30 - Valid Loss: 1680.7923
Sub-losses: recon_loss: 1666.1517, var_loss: 14.6406, anneal_factor: 0
... [output truncated]
```

In \[11\]:

Copied!

```
custom_varix.visualizer.show_loss(plot_type="absolute")
```

custom_varix.visualizer.show_loss(plot_type="absolute")

In \[12\]:

Copied!

```
custom_varix.visualizer.show_loss(plot_type="relative")
```

custom_varix.visualizer.show_loss(plot_type="relative")

## 6) Re-use, Save, Load[¶](#6-re-use-save-load)

There are not `Varix` specific steps here. See the `Getting Started - Vanillix` for details. Below is a basic save/load usecase:

In \[13\]:

Copied!

```
import os
import glob
# use a filename without extension, we handle this internally
outpath = os.path.join("tutorial_res", "varix")
varix.save(file_path=outpath, save_all=False)

folder = os.path.dirname(outpath)
pkl_files = glob.glob(os.path.join(folder, "*.pkl"))
model_files = glob.glob(os.path.join(folder, "*.pth"))

print("PKL files:", pkl_files)
print("Model files:", model_files)

# the load functionality automatically will build the pipeline object out of the three saved files
varix_loaded = acx.Varix.load(outpath)
varix_loaded.predict(data=sc_data)
varix_loaded.visualize()
varix_loaded.show_result(params=["cell_type"])
```

import os import glob

# use a filename without extension, we handle this internally

outpath = os.path.join("tutorial_res", "varix") varix.save(file_path=outpath, save_all=False) folder = os.path.dirname(outpath) pkl_files = glob.glob(os.path.join(folder, "\*.pkl")) model_files = glob.glob(os.path.join(folder, "\*.pth")) print("PKL files:", pkl_files) print("Model files:", model_files)

# the load functionality automatically will build the pipeline object out of the three saved files

varix_loaded = acx.Varix.load(outpath) varix_loaded.predict(data=sc_data) varix_loaded.visualize() varix_loaded.show_result(params=["cell_type"])

```
Preprocessor saved successfully.
saving memory efficient
Pipeline object saved successfully.
PKL files: ['tutorial_res/varix_preprocessor.pkl', 'tutorial_res/ontix_reactome_preprocessor.pkl', 'tutorial_res/ontix.pkl', 'tutorial_res/maskix_preprocessor.pkl', 'tutorial_res/imagix.pkl', 'tutorial_res/ontix_preprocessor.pkl', 'tutorial_res/disent.pkl', 'tutorial_res/imagix_preprocessor.pkl', 'tutorial_res/ontix_reactome.pkl', 'tutorial_res/disent_preprocessor.pkl']
Model files: ['tutorial_res/varix_model.pth', 'tutorial_res/ontix_reactome_model.pth', 'tutorial_res/disent_model.pth', 'tutorial_res/imagix_model.pth', 'tutorial_res/ontix_model.pth', 'tutorial_res/maskix_model.pth']
Attempting to load a pipeline from tutorial_res/varix...
Pipeline object loaded successfully. Actual type: Varix
Pr
... [output truncated]
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/utils/_result.py:348: UserWarning: Could not retrieve latent representations for epoch 29 and split 'train'. Returning empty DataFrame. This may be due to missing data in the Result object or incorrect keys.


  warnings.warn(
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/utils/_result.py:348: UserWarning: Could not retrieve latent representations for epoch 29 and split 'valid'. Returning empty DataFrame. This may be due to missing data in the Result object or incorrect keys.


  warnings.warn(
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/utils/_result.py:348: UserWarning: Could not retrieve latent representations for epoch 29 and split 'train'. Returning empty DataFrame. This may be due to missing data in the Result object or incorrect keys.


  warnings.warn(
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/utils/_result.py:348: UserWarning: Could not retrieve latent representations for epoch 29 and split 'valid'. Returning empty DataFrame. This may be due to missing data in the Result object or incorrect keys.


  warnings.warn(
```

In \[14\]:

Copied!

```
varix_loaded.evaluate(
	params=["cell_type"],
	split_type="CV-5",
	)
```

varix_loaded.evaluate( params=["cell_type"], split_type="CV-5", )

```
Perform ML task with feature df: Latent
Latent
Perform ML task for target parameter: cell_type
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/utils/_result.py:348: UserWarning: Could not retrieve latent representations for epoch 29 and split 'train'. Returning empty DataFrame. This may be due to missing data in the Result object or incorrect keys.


  warnings.warn(
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/utils/_result.py:348: UserWarning: Could not retrieve latent representations for epoch 29 and split 'valid'. Returning empty DataFrame. This may be due to missing data in the Result object or incorrect keys.


  warnings.warn(
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/.venv/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP:
... [output truncated]
```

Out\[14\]:

```
Result Object Public Attributes:
------------------------------
latentspaces: TrainingDynamics object
sample_ids: TrainingDynamics object
reconstructions: TrainingDynamics object
mus: TrainingDynamics object
sigmas: TrainingDynamics object
losses: TrainingDynamics object
sub_losses: LossRegistry(_losses={'recon_loss': TrainingDynamics(), 'var_loss': TrainingDynamics(), 'anneal_factor': TrainingDynamics(), 'effective_beta_factor': TrainingDynamics()})
preprocessed_data: Tensor of shape (0,)
model: VarixArchitecture
model_checkpoints: TrainingDynamics object
datasets: DatasetContainer(train=None, valid=None, test=None)
new_datasets: DatasetContainer(train=<autoencodix.data._numeric_dataset.NumericDataset object at 0x725f6510a5c0>, valid=None, test=<autoencodix.data._numeric_dataset.NumericD
... [output truncated]
```

In \[15\]:

Copied!

```
varix_loaded.visualizer.show_evaluation(
	param="cell_type",
	metric="roc_auc_ovo",
)
```

varix_loaded.visualizer.show_evaluation( param="cell_type", metric="roc_auc_ovo", )

```
Showing plot for ML algorithm: LogisticRegression
```
