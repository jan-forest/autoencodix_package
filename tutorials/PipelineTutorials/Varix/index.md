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
Changed to: /home/janew/ACX/autoencodix_package
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
Processing 1 MuData objects: ['multi_sc']
Processing train modality: multi_sc
```

```
Processing valid split
Processing valid modality: multi_sc
Processing test split
Processing test modality: multi_sc
```

```
You are using a CUDA device ('NVIDIA GeForce RTX 5060 Ti') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 13.9460 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 5 - Train Loss: 1774.9840
Sub-losses: recon_loss: 1774.9734, var_loss: 0.0106, anneal_factor: 0.0007, effective_beta_factor: 0.0001
Epoch 5 - Valid Loss: 1586.1633
Sub-losses: recon_loss: 1586.1513, var_loss: 0.0120, anneal_factor: 0.0007, effective_beta_factor: 0.0001
```

```
Epoch 10 - Train Loss: 1731.5486
Sub-losses: recon_loss: 1730.9184, var_loss: 0.6302, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Epoch 10 - Valid Loss: 1550.8277
Sub-losses: recon_loss: 1550.1988, var_loss: 0.6289, anneal_factor: 0.0180, effective_beta_factor: 0.0018
```

```
Epoch 15 - Train Loss: 1721.2885
Sub-losses: recon_loss: 1711.8086, var_loss: 9.4799, anneal_factor: 0.3392, effective_beta_factor: 0.0339
Epoch 15 - Valid Loss: 1546.6144
Sub-losses: recon_loss: 1538.1241, var_loss: 8.4904, anneal_factor: 0.3392, effective_beta_factor: 0.0339
```

```
Epoch 20 - Train Loss: 1713.1763
Sub-losses: recon_loss: 1702.2103, var_loss: 10.9660, anneal_factor: 0.9350, effective_beta_factor: 0.0935
Epoch 20 - Valid Loss: 1542.9735
Sub-losses: recon_loss: 1533.1573, var_loss: 9.8161, anneal_factor: 0.9350, effective_beta_factor: 0.0935
```

```
Epoch 25 - Train Loss: 1703.6316
Sub-losses: recon_loss: 1693.3572, var_loss: 10.2744, anneal_factor: 0.9975, effective_beta_factor: 0.0998
```

```
Epoch 25 - Valid Loss: 1534.5916
Sub-losses: recon_loss: 1526.1381, var_loss: 8.4535, anneal_factor: 0.9975, effective_beta_factor: 0.0998
```

```
Epoch 30 - Train Loss: 1691.5348
Sub-losses: recon_loss: 1681.4464, var_loss: 10.0884, anneal_factor: 0.9999, effective_beta_factor: 0.1000
```

```
Epoch 30 - Valid Loss: 1526.7488
Sub-losses: recon_loss: 1518.0554, var_loss: 8.6934, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Processed 32 / 1600 samples
Processed 64 / 1600 samples
Processed 96 / 1600 samples
Processed 128 / 1600 samples
Processed 160 / 1600 samples
Processed 192 / 1600 samples
Processed 224 / 1600 samples
Processed 256 / 1600 samples
Processed 288 / 1600 samples
Processed 320 / 1600 samples
Processed 352 / 1600 samples
Processed 384 / 1600 samples
Processed 416 / 1600 samples
Processed 448 / 1600 samples
Processed 480 / 1600 samples
Processed 512 / 1600 samples
Processed 544 / 1600 samples
Processed 576 / 1600 samples
Processed 608 / 1600 samples
Processed 640 / 1600 samples
Processed 672 / 1600 samples
Processed 704 / 1600 samples
Processed 736 / 1600 samp
... [output truncated]
```

```
Processed 864 / 1600 samples
Processed 896 / 1600 samples
Processed 928 / 1600 samples
Processed 960 / 1600 samples
Processed 992 / 1600 samples
Processed 1024 / 1600 samples
Processed 1056 / 1600 samples
Processed 1088 / 1600 samples
Processed 1120 / 1600 samples
Processed 1152 / 1600 samples
Processed 1184 / 1600 samples
Processed 1216 / 1600 samples
Processed 1248 / 1600 samples
Processed 1280 / 1600 samples
Processed 1312 / 1600 samples
Processed 1344 / 1600 samples
Processed 1376 / 1600 samples
Processed 1408 / 1600 samples
Processed 1440 / 1600 samples
Processed 1472 / 1600 samples
Processed 1504 / 1600 samples
Processed 1536 / 1600 samples
Processed 1568 / 1600 samples
Processed 1600 / 1600 samples
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
tensor([[-9.0151e-01,  4.1121e+00,  2.0569e+00,  4.6674e+00, -3.5426e-01,
          4.4271e+00,  2.9221e+00,  3.7027e+00,  5.5142e+00,  2.4096e+00,
          3.4524e+00,  3.3114e+00,  1.6839e+00,  3.1980e+00,  5.6445e+00,
          7.5028e-01],
        [-6.2570e-01,  4.3095e+00,  2.0333e+00,  5.1628e+00,  1.2702e+00,
          5.7894e+00,  2.5599e+00,  1.0660e+00,  5.2623e+00,  2.0217e+00,
          5.3762e+00,  1.7210e+00,  2.3712e+00,  2.3283e+00,  1.2552e+00,
          1.7572e+00],
        [ 1.9471e-01,  4.2462e+00, -1.8693e+00,  4.1160e+00,  9.0699e-01,
          4.0397e+00,  2.3787e+00,  3.0838e+00,  3.5594e+00,  4.9432e+00,
          4.8438e+00,  2.5668e+00,  2.2467e+00,  2.7558e+00,  2.5294e+00,
          7.5489e-01],
        [ 2.4275e-01,  2.5891e+00,  1.5906e+00,  5.0505e+00,  3.
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
1774.9734127371653
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
Processing 1 MuData objects: ['multi_sc']
Processing train modality: multi_sc
```

```
Processing valid split
Processing valid modality: multi_sc
Processing test split
Processing test modality: multi_sc
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 11.8250 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 10 - Train Loss: 1722.1053
Sub-losses: recon_loss: 1722.1053, var_loss: 0.0000, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 10 - Valid Loss: 1911.5156
Sub-losses: recon_loss: 1911.5156, var_loss: 0.0000, anneal_factor: 0.0000, effective_beta_factor: 0.0000
```

```
Epoch 20 - Train Loss: 1693.3474
Sub-losses: recon_loss: 1688.5897, var_loss: 4.7578, anneal_factor: 0.1400, effective_beta_factor: 0.2100
Epoch 20 - Valid Loss: 1890.6848
Sub-losses: recon_loss: 1885.8566, var_loss: 4.8282, anneal_factor: 0.1400, effective_beta_factor: 0.2100
```

```
Epoch 30 - Train Loss: 1686.8980
Sub-losses: recon_loss: 1670.2706, var_loss: 16.6274, anneal_factor: 0.7400, effective_beta_factor: 1.1100
Epoch 30 - Valid Loss: 1886.2877
Sub-losses: recon_loss: 1870.9648, var_loss: 15.3229, anneal_factor: 0.7400, effective_beta_factor: 1.1100
```

```
Epoch 40 - Train Loss: 1673.4968
Sub-losses: recon_loss: 1656.1596, var_loss: 17.3372, anneal_factor: 1.0000, effective_beta_factor: 1.5000
Epoch 40 - Valid Loss: 1875.6428
Sub-losses: recon_loss: 1858.4759, var_loss: 17.1669, anneal_factor: 1.0000, effective_beta_factor: 1.5000
```

```
Epoch 50 - Train Loss: 1661.7413
Sub-losses: recon_loss: 1644.8980, var_loss: 16.8434, anneal_factor: 1.0000, effective_beta_factor: 1.5000
```

```
Epoch 50 - Valid Loss: 1870.6242
Sub-losses: recon_loss: 1855.3265, var_loss: 15.2977, anneal_factor: 1.0000, effective_beta_factor: 1.5000
Processed 32 / 1600 samples
Processed 64 / 1600 samples
Processed 96 / 1600 samples
Processed 128 / 1600 samples
Processed 160 / 1600 samples
Processed 192 / 1600 samples
Processed 224 / 1600 samples
Processed 256 / 1600 samples
Processed 288 / 1600 samples
Processed 320 / 1600 samples
Processed 352 / 1600 samples
```

```
Processed 384 / 1600 samples
Processed 416 / 1600 samples
Processed 448 / 1600 samples
Processed 480 / 1600 samples
Processed 512 / 1600 samples
Processed 544 / 1600 samples
Processed 576 / 1600 samples
Processed 608 / 1600 samples
Processed 640 / 1600 samples
Processed 672 / 1600 samples
Processed 704 / 1600 samples
Processed 736 / 1600 samples
Processed 768 / 1600 samples
Processed 800 / 1600 samples
Processed 832 / 1600 samples
Processed 864 / 1600 samples
Processed 896 / 1600 samples
Processed 928 / 1600 samples
Processed 960 / 1600 samples
Processed 992 / 1600 samples
Processed 1024 / 1600 samples
Processed 1056 / 1600 samples
Processed 1088 / 1600 samples
Processed 1120 / 1600 samples
```

```
Processed 1152 / 1600 samples
Processed 1184 / 1600 samples
Processed 1216 / 1600 samples
Processed 1248 / 1600 samples
Processed 1280 / 1600 samples
Processed 1312 / 1600 samples
Processed 1344 / 1600 samples
Processed 1376 / 1600 samples
Processed 1408 / 1600 samples
Processed 1440 / 1600 samples
Processed 1472 / 1600 samples
Processed 1504 / 1600 samples
Processed 1536 / 1600 samples
Processed 1568 / 1600 samples
Processed 1600 / 1600 samples
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
```

```
saving memory efficient
```

```
Pipeline object saved successfully.
PKL files: ['tutorial_res/ontix_reactome_preprocessor.pkl', 'tutorial_res/ontix.pkl', 'tutorial_res/imagix_preprocessor.pkl', 'tutorial_res/varix_preprocessor.pkl', 'tutorial_res/imagix.pkl', 'tutorial_res/ontix_reactome.pkl', 'tutorial_res/supervisix_preprocessor.pkl', 'tutorial_res/maskix_preprocessor.pkl', 'tutorial_res/imagix3d_preprocessor.pkl', 'tutorial_res/xmodalix_preprocessor.pkl', 'tutorial_res/xmodalix.pkl', 'tutorial_res/imagix3d.pkl', 'tutorial_res/disent.pkl', 'tutorial_res/disent_preprocessor.pkl', 'tutorial_res/ontix_preprocessor.pkl']
Model files: ['tutorial_res/imagix3d_model.pth', 'tutorial_res/imagix_model.pth', 'tutorial_res/disent_model.pth', 'tutorial_res/supervisix_model.pth', 'tutorial_res/img.img_xmodalix_model.pth', 'tutorial
... [output truncated]
```

```
Pipeline object loaded successfully. Actual type: Varix
```

```
Preprocessor loaded successfully.
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
```

```
mudata: View of MuData object with n_obs × n_vars = 8000 × 8189
  var:	'soma_joinid', 'feature_id', 'feature_name', 'feature_type', 'feature_length', 'nnz', 'n_measured_obs'
  1 modality
    rna:	8000 x 8189
      obs:	'soma_joinid', 'dataset_id', 'assay', 'assay_ontology_term_id', 'cell_type', 'cell_type_ontology_term_id', 'development_stage', 'development_stage_ontology_term_id', 'disease', 'disease_ontology_term_id', 'donor_id', 'is_primary_data', 'observation_joinid', 'self_reported_ethnicity', 'self_reported_ethnicity_ontology_term_id', 'sex', 'sex_ontology_term_id', 'suspension_type', 'tissue', 'tissue_ontology_term_id', 'tissue_type', 'tissue_general', 'tissue_general_ontology_term_id', 'raw_sum', 'nnz', 'raw_mean_nnz', 'raw_variance_nnz', 'n_measured_vars', 'n_genes'
      var:	's
... [output truncated]
```

```
Processed 32 / 8000 samples
Processed 64 / 8000 samples
Processed 96 / 8000 samples
Processed 128 / 8000 samples
Processed 160 / 8000 samples
Processed 192 / 8000 samples
Processed 224 / 8000 samples
Processed 256 / 8000 samples
Processed 288 / 8000 samples
Processed 320 / 8000 samples
Processed 352 / 8000 samples
Processed 384 / 8000 samples
Processed 416 / 8000 samples
Processed 448 / 8000 samples
Processed 480 / 8000 samples
Processed 512 / 8000 samples
Processed 544 / 8000 samples
Processed 576 / 8000 samples
Processed 608 / 8000 samples
Processed 640 / 8000 samples
Processed 672 / 8000 samples
Processed 704 / 8000 samples
```

```
Processed 736 / 8000 samples
Processed 768 / 8000 samples
Processed 800 / 8000 samples
Processed 832 / 8000 samples
Processed 864 / 8000 samples
Processed 896 / 8000 samples
Processed 928 / 8000 samples
Processed 960 / 8000 samples
Processed 992 / 8000 samples
Processed 1024 / 8000 samples
Processed 1056 / 8000 samples
Processed 1088 / 8000 samples
Processed 1120 / 8000 samples
Processed 1152 / 8000 samples
Processed 1184 / 8000 samples
Processed 1216 / 8000 samples
Processed 1248 / 8000 samples
Processed 1280 / 8000 samples
Processed 1312 / 8000 samples
Processed 1344 / 8000 samples
Processed 1376 / 8000 samples
Processed 1408 / 8000 samples
Processed 1440 / 8000 samples
```

```
Processed 1472 / 8000 samples
Processed 1504 / 8000 samples
Processed 1536 / 8000 samples
Processed 1568 / 8000 samples
Processed 1600 / 8000 samples
Processed 1632 / 8000 samples
Processed 1664 / 8000 samples
Processed 1696 / 8000 samples
Processed 1728 / 8000 samples
Processed 1760 / 8000 samples
Processed 1792 / 8000 samples
Processed 1824 / 8000 samples
Processed 1856 / 8000 samples
Processed 1888 / 8000 samples
Processed 1920 / 8000 samples
Processed 1952 / 8000 samples
Processed 1984 / 8000 samples
Processed 2016 / 8000 samples
Processed 2048 / 8000 samples
Processed 2080 / 8000 samples
Processed 2112 / 8000 samples
Processed 2144 / 8000 samples
Processed 2176 / 8000 samples
Processed 2208 / 8000 samples
Processed 2240 / 8000 samples
Processed 2272 / 8000 samples
```

```
Processed 2304 / 8000 samples
Processed 2336 / 8000 samples
Processed 2368 / 8000 samples
Processed 2400 / 8000 samples
Processed 2432 / 8000 samples
Processed 2464 / 8000 samples
Processed 2496 / 8000 samples
Processed 2528 / 8000 samples
Processed 2560 / 8000 samples
Processed 2592 / 8000 samples
Processed 2624 / 8000 samples
Processed 2656 / 8000 samples
Processed 2688 / 8000 samples
Processed 2720 / 8000 samples
Processed 2752 / 8000 samples
Processed 2784 / 8000 samples
Processed 2816 / 8000 samples
Processed 2848 / 8000 samples
Processed 2880 / 8000 samples
Processed 2912 / 8000 samples
Processed 2944 / 8000 samples
Processed 2976 / 8000 samples
Processed 3008 / 8000 samples
Processed 3040 / 8000 samples
Processed 3072 / 8000 samples
Processed 3104 / 8000 samples
Processed 3136 / 80
... [output truncated]
```

```
Processed 3296 / 8000 samples
Processed 3328 / 8000 samples
Processed 3360 / 8000 samples
Processed 3392 / 8000 samples
Processed 3424 / 8000 samples
Processed 3456 / 8000 samples
Processed 3488 / 8000 samples
Processed 3520 / 8000 samples
Processed 3552 / 8000 samples
Processed 3584 / 8000 samples
Processed 3616 / 8000 samples
Processed 3648 / 8000 samples
Processed 3680 / 8000 samples
Processed 3712 / 8000 samples
Processed 3744 / 8000 samples
Processed 3776 / 8000 samples
Processed 3808 / 8000 samples
Processed 3840 / 8000 samples
Processed 3872 / 8000 samples
Processed 3904 / 8000 samples
Processed 3936 / 8000 samples
Processed 3968 / 8000 samples
Processed 4000 / 8000 samples
```

```
Processed 4032 / 8000 samples
Processed 4064 / 8000 samples
Processed 4096 / 8000 samples
Processed 4128 / 8000 samples
Processed 4160 / 8000 samples
Processed 4192 / 8000 samples
Processed 4224 / 8000 samples
Processed 4256 / 8000 samples
Processed 4288 / 8000 samples
Processed 4320 / 8000 samples
Processed 4352 / 8000 samples
Processed 4384 / 8000 samples
Processed 4416 / 8000 samples
Processed 4448 / 8000 samples
Processed 4480 / 8000 samples
Processed 4512 / 8000 samples
Processed 4544 / 8000 samples
Processed 4576 / 8000 samples
Processed 4608 / 8000 samples
Processed 4640 / 8000 samples
Processed 4672 / 8000 samples
Processed 4704 / 8000 samples
Processed 4736 / 8000 samples
```

```
Processed 4768 / 8000 samples
Processed 4800 / 8000 samples
Processed 4832 / 8000 samples
Processed 4864 / 8000 samples
Processed 4896 / 8000 samples
Processed 4928 / 8000 samples
Processed 4960 / 8000 samples
Processed 4992 / 8000 samples
Processed 5024 / 8000 samples
Processed 5056 / 8000 samples
Processed 5088 / 8000 samples
Processed 5120 / 8000 samples
Processed 5152 / 8000 samples
Processed 5184 / 8000 samples
Processed 5216 / 8000 samples
Processed 5248 / 8000 samples
Processed 5280 / 8000 samples
Processed 5312 / 8000 samples
Processed 5344 / 8000 samples
Processed 5376 / 8000 samples
Processed 5408 / 8000 samples
Processed 5440 / 8000 samples
Processed 5472 / 8000 samples
```

```
Processed 5504 / 8000 samples
Processed 5536 / 8000 samples
Processed 5568 / 8000 samples
Processed 5600 / 8000 samples
Processed 5632 / 8000 samples
Processed 5664 / 8000 samples
Processed 5696 / 8000 samples
Processed 5728 / 8000 samples
Processed 5760 / 8000 samples
Processed 5792 / 8000 samples
Processed 5824 / 8000 samples
Processed 5856 / 8000 samples
Processed 5888 / 8000 samples
Processed 5920 / 8000 samples
Processed 5952 / 8000 samples
Processed 5984 / 8000 samples
Processed 6016 / 8000 samples
Processed 6048 / 8000 samples
Processed 6080 / 8000 samples
Processed 6112 / 8000 samples
Processed 6144 / 8000 samples
Processed 6176 / 8000 samples
Processed 6208 / 8000 samples
Processed 6240 / 8000 samples
Processed 6272 / 8000 samples
Processed 6304 / 8000 samples
Processed 6336 / 80
... [output truncated]
```

```
Processed 7072 / 8000 samples
Processed 7104 / 8000 samples
Processed 7136 / 8000 samples
Processed 7168 / 8000 samples
Processed 7200 / 8000 samples
Processed 7232 / 8000 samples
Processed 7264 / 8000 samples
Processed 7296 / 8000 samples
Processed 7328 / 8000 samples
Processed 7360 / 8000 samples
Processed 7392 / 8000 samples
Processed 7424 / 8000 samples
Processed 7456 / 8000 samples
Processed 7488 / 8000 samples
Processed 7520 / 8000 samples
Processed 7552 / 8000 samples
Processed 7584 / 8000 samples
Processed 7616 / 8000 samples
Processed 7648 / 8000 samples
Processed 7680 / 8000 samples
Processed 7712 / 8000 samples
Processed 7744 / 8000 samples
Processed 7776 / 8000 samples
```

```
Processed 7808 / 8000 samples
Processed 7840 / 8000 samples
Processed 7872 / 8000 samples
Processed 7904 / 8000 samples
Processed 7936 / 8000 samples
Processed 7968 / 8000 samples
Processed 8000 / 8000 samples
n_samples in format recon: 8000
train
n_samples from datatpackge: {'paired_count': 8000}
```

```
Creating plots ...
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
.venv/lib/python3.12/site-packages/sklearn/linear_model/_logistic.py:451: OptimizeWarning: Unknown solver options: iprint
  opt_res = optimize.minimize(
.venv/lib/python3.12/site-packages/sklearn/linear_model/_logistic.py:451: OptimizeWarning: Unknown solver options: iprint
  opt_res = optimize.minimize(
.venv/lib/python3.12/site-packages/sklearn/linear_model/_logistic.py:451: OptimizeWarning: Unknown solver options: iprint
  opt_res = optimize.minimize(
.venv/lib/python3.12/site-packages/sklearn/linear_model/_logistic.py:451: OptimizeWarning: Unknown solver options: iprint
  opt_res = optimize.minimize(
.venv/lib/python3.12/site-packages/sklearn/linear_model/_logistic.py:451: OptimizeWarning: Unknown solver options: iprint
  opt_res = optimize.minimize(
.venv/lib/python3.12/site-packages
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
class_means: TrainingDynamics object
datasets: DatasetContainer(train=<autoencodix.data._numeric_dataset.NumericDataset object at 0x77bd64494fb0>, valid=<autoencodix.data._numeric_dataset.NumericDataset object at 0x77bd653b03b0>, test=<
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
