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

# How to Train and Use Supervisix: A Semi-Supervised VAE[¶](#how-to-train-and-use-supervisix-a-semi-supervised-vae)

`Supervisix` is a variant of the `Varix` pipeline that implements a **variational autoencoder with semi-supervised learning**.\
Its key difference lies in the loss function, which adds two additional class-based loss terms to the standard vae loss.

**IMPORTANT**

> This tutorial only shows the specifics of the Supervisix pipeline. If you're unfamiliar with general concepts,\
> we recommend following the `Getting Started - Vanillix` tutorial first.

## What You'll Learn[¶](#what-youll-learn)

- How the new losses influence learning:
  - **Class cohesion loss** (`class_cohesion_loss`)
  - **Class separation loss** (`class_sep_loss`)
- How to set the parameter that determines which classes in the dataset will be used for the class-based loss calculation (`class_param`)
- How to initialize, run and visualize a semi-supervised VAE

## 1) The class-based Loss Terms[¶](#1-the-class-based-loss-terms)

The `SupervisixLoss` class computes the following terms for each batch:

1. **Reconstruction Loss** (`recon_loss`)\
   Measures how well the decoded output matches the input.
1. **Variational Loss** (`var_loss`)\
   Measures how well the vae distribution matches the prior distribution of the input data.
1. **Class Cohesion Loss** (`class_cohesion_loss`)\
   Minimizes the latent space distance of each sample to the mean latent space distance of all samples in the same class.
1. **Class separation Loss** (`class_sep_loss`)\
   Minimizes the negative latent space distance between each sample and the mean of any class in the data set except its own class.

The total loss is computed as a weighted sum:

```
total_loss = recon_loss 
             + beta * var_loss 
             + delta_class_cohesion * class_cohesion_loss 
             + gamma_class_separation * class_sep_loss
```

### Mathematical formulation[¶](#mathematical-formulation)

The class-based losses are inspired by the cluster loss (center loss) of the semi-supervised VAE in [Das et al.](https://arxiv.org/abs/2203.14865), which minimizes distances of samples to their class mean (intra-class) while maximizing the distances between classes (inter-class). Das et al. combine both parts into a single ratio (D\_{\\text{intra}}/D\_{\\text{inter}}) with one weight. In Supervisix, the two parts are separate loss terms with their own weights (`delta_class_cohesion` and `gamma_class_separation`), and the separation term uses the distances between samples and the means of the *other* classes. For readability the formulas show the loss computed once on the full data set. In practice it is computed per batch, using the class means from the previous epoch.

Let (\\mathcal{C}) be the set of classes with (K=|\\mathcal{C}|), and (S_c) the set of samples of class (c) with (n_c=|S_c|). Each sample (i) has a latent vector (z_i \\in \\mathbb{R}^d), where (d) is the number of latent dimensions, and (z\_{ij}) is its (j)-th entry. (\\mu_c \\in \\mathbb{R}^d) is the mean latent vector of class (c), with entries (\\mu\_{c,j}).

The **distance** between two vectors in the latend space is computed as:

[ D(S_a, \\mu_b) = \\frac{1}{d}\\sum\_{j=1}^{d}\\left|z\_{ij}-\\mu\_{b,j}\\right| ]

**Class cohesion** pulls samples towards the mean of their own class:

\[ \\mathcal{L}_{\\text{coh}} = \\frac{1}{K}\\sum_{c \\in \\mathcal{C}} \\frac{1}{n_c}\\sum\_{i \\in S_c} d(z_i, \\mu_c) \]

**Class separation** pushes samples away from the means of all other classes (note the minus sign):

\[ \\mathcal{L}_{\\text{sep}} = -\\frac{1}{K}\\sum_{a \\in \\mathcal{C}}\\frac{1}{K-1}\\sum\_{b \\neq a} \\frac{1}{n_a}\\sum\_{i \\in S_a} d(z_i, \\mu_b) \]

They are weighted by `delta_class_cohesion` ((\\delta)) and `gamma_class_separation` ((\\gamma)) in the total loss:

\[ \\mathcal{L} = \\mathcal{L}_{\\text{recon}} + \\beta,\\mathcal{L}_{\\text{KL}} + \\gamma,\\mathcal{L}_{\\text{sep}} + \\delta,\\mathcal{L}_{\\text{coh}} \]

## 2) Set classes to use for class-based loss calculation[¶](#2-set-classes-to-use-for-class-based-loss-calculation)

In order to calculate the class-based losses, Supervisix needs access to class label information for the input samples. For this the `class_param` parameter needs to be set when initializing the `SupervisixConfig`. It should be the name of a column in the data set metadata that connects the class labels with sampleIDs.

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

We use real human lung single-cell RNA-seq data assembled from the [CZ CELLxGENE Census](https://cellxgene.cziscience.com/). It is hosted on Hugging Face Hub ([autoencodix/census-lung](https://huggingface.co/datasets/autoencodix/census-lung)) and is downloaded automatically in the cells below on first run. The full dataset contains ~89,000 cells spanning four lung cell types (malignant cells, pulmonary alveolar type 2 cells, respiratory basal cells, alveolar macrophages), which also correlate closely with the `disease` metadata column (normal / lung adenocarcinoma / cystic fibrosis). We use `cell_type` and `disease` as the two factors to inspect for disentanglement below. To keep this tutorial fast to run, we draw a balanced subsample of 2,000 cells per cell type.

## 3) Initialize and Run Supervisix[¶](#3-initialize-and-run-supervisix)

This works like `Varix`, except we don't have only the `beta` parameter to weigh the distribution loss, we have one additional weight for each of the class-based loss terms, `delta_class_cohesion` and `gamma_class_separation`, that we can set in the config. We also have to make sure to set the `class_param` parameter in the config or an Error will be raised.

In \[3\]:

Copied!

```
import numpy as np
import anndata as ad
import mudata

from autoencodix.configs.supervisix_config import SupervisixConfig
from autoencodix.configs.default_config import DataCase, DataConfig, DataInfo
from autoencodix.data.datapackage import DataPackage
from huggingface_hub import hf_hub_download
import autoencodix as acx

sc_path = hf_hub_download(
    repo_id="autoencodix/census-lung", repo_type="dataset", filename="census_lung.h5ad"
)
full_adata = ad.read_h5ad(sc_path)

# Balanced subsample: 2,000 cells per cell type keeps the tutorial fast to run
# while keeping all four (well-separated) cell types equally represented.
N_PER_TYPE = 2000
rng = np.random.RandomState(43)
subsample_idx = []
for _, group in full_adata.obs.groupby("cell_type", observed=True):
    n = min(N_PER_TYPE, len(group))
    subsample_idx.extend(rng.choice(group.index.values, size=n, replace=False))

adata = full_adata[subsample_idx].copy()

sc_data = DataPackage()
sc_data.multi_sc = {"multi_sc": mudata.MuData({"rna": adata})}

my_cfg = SupervisixConfig(
	data_case=DataCase.MULTI_SINGLE_CELL,
	loss_reduction="sum",
	k_filter=2000,
	latent_dim=3,
	scaling="STANDARD",
	epochs=50,
	learning_rate=0.01,
	batch_size = 128,
	beta = 0.1,
	gamma_class_separation=20,
    delta_class_cohesion=70,
	use_mss = True,
	drop_p =0.1,
	global_seed=7,
	reproducible=True,
	checkpoint_interval=10,
	n_layers=1,
	annotation_columns=["cell_type", "disease"],
    class_param="disease",
	data_config=DataConfig(
		data_info={"multi_sc": DataInfo(is_single_cell=True, data_type="NUMERIC")},
	),
)

supervis = acx.Supervisix(data=sc_data, config=my_cfg)
```

import numpy as np import anndata as ad import mudata from autoencodix.configs.supervisix_config import SupervisixConfig from autoencodix.configs.default_config import DataCase, DataConfig, DataInfo from autoencodix.data.datapackage import DataPackage from huggingface_hub import hf_hub_download import autoencodix as acx sc_path = hf_hub_download( repo_id="autoencodix/census-lung", repo_type="dataset", filename="census_lung.h5ad" ) full_adata = ad.read_h5ad(sc_path)

# Balanced subsample: 2,000 cells per cell type keeps the tutorial fast to run

# while keeping all four (well-separated) cell types equally represented.

N_PER_TYPE = 2000 rng = np.random.RandomState(43) subsample_idx = [] for \_, group in full_adata.obs.groupby("cell_type", observed=True): n = min(N_PER_TYPE, len(group)) subsample_idx.extend(rng.choice(group.index.values, size=n, replace=False)) adata = full_adata[subsample_idx].copy() sc_data = DataPackage() sc_data.multi_sc = {"multi_sc": mudata.MuData({"rna": adata})} my_cfg = SupervisixConfig( data_case=DataCase.MULTI_SINGLE_CELL, loss_reduction="sum", k_filter=2000, latent_dim=3, scaling="STANDARD", epochs=50, learning_rate=0.01, batch_size = 128, beta = 0.1, gamma_class_separation=20, delta_class_cohesion=70, use_mss = True, drop_p =0.1, global_seed=7, reproducible=True, checkpoint_interval=10, n_layers=1, annotation_columns=["cell_type", "disease"], class_param="disease", data_config=DataConfig( data_info={"multi_sc": DataInfo(is_single_cell=True, data_type="NUMERIC")}, ), ) supervis = acx.Supervisix(data=sc_data, config=my_cfg)

```
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
```

In \[4\]:

Copied!

```
result = supervis.run()
```

result = supervis.run()

```
mudata: View of MuData object with n_obs × n_vars = 8000 × 8189
  var:	'soma_joinid', 'feature_id', 'feature_name', 'feature_type', 'feature_length', 'nnz', 'n_measured_obs'
  1 modality
    rna:	8000 x 8189
      obs:	'soma_joinid', 'dataset_id', 'assay', 'assay_ontology_term_id', 'cell_type', 'cell_type_ontology_term_id', 'development_stage', 'development_stage_ontology_term_id', 'disease', 'disease_ontology_term_id', 'donor_id', 'is_primary_data', 'observation_joinid', 'self_reported_ethnicity', 'self_reported_ethnicity_ontology_term_id', 'sex', 'sex_ontology_term_id', 'suspension_type', 'tissue', 'tissue_ontology_term_id', 'tissue_type', 'tissue_general', 'tissue_general_ontology_term_id', 'raw_sum', 'nnz', 'raw_mean_nnz', 'raw_variance_nnz', 'n_measured_vars'
      var:	'soma_joinid'
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
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 31.1268 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 10 - Train Loss: 1852.7631
Sub-losses: recon_loss: 1815.3291, var_loss: 0.0494, class_separation_loss: -15.3328, class_cohesion_loss: 52.7174, anneal_factor: 0.0017, effective_beta_factor: 0.0002
Epoch 10 - Valid Loss: 2110.0043
Sub-losses: recon_loss: 2071.3368, var_loss: 0.0511, class_separation_loss: -15.9767, class_cohesion_loss: 54.5931, anneal_factor: 0.0017, effective_beta_factor: 0.0002
```

```
Epoch 20 - Train Loss: 1854.3458
Sub-losses: recon_loss: 1812.9887, var_loss: 4.1599, class_separation_loss: -15.4794, class_cohesion_loss: 52.6766, anneal_factor: 0.0832, effective_beta_factor: 0.0083
Epoch 20 - Valid Loss: 2115.5669
Sub-losses: recon_loss: 2068.0816, var_loss: 4.3873, class_separation_loss: -17.4118, class_cohesion_loss: 60.5099, anneal_factor: 0.0832, effective_beta_factor: 0.0083
```

```
Epoch 30 - Train Loss: 1865.3655
Sub-losses: recon_loss: 1804.5089, var_loss: 22.7064, class_separation_loss: -16.1989, class_cohesion_loss: 54.3492, anneal_factor: 0.8320, effective_beta_factor: 0.0832
Epoch 30 - Valid Loss: 2118.5156
Sub-losses: recon_loss: 2054.1389, var_loss: 23.1051, class_separation_loss: -17.2160, class_cohesion_loss: 58.4875, anneal_factor: 0.8320, effective_beta_factor: 0.0832
```

```
Epoch 40 - Train Loss: 1847.0812
Sub-losses: recon_loss: 1802.9000, var_loss: 9.1868, class_separation_loss: -14.9180, class_cohesion_loss: 49.9124, anneal_factor: 0.9963, effective_beta_factor: 0.0996
Epoch 40 - Valid Loss: 2100.8650
Sub-losses: recon_loss: 2055.3828, var_loss: 9.3858, class_separation_loss: -15.6300, class_cohesion_loss: 51.7265, anneal_factor: 0.9963, effective_beta_factor: 0.0996
```

```
Epoch 50 - Train Loss: 1845.0989
Sub-losses: recon_loss: 1804.4329, var_loss: 7.3433, class_separation_loss: -14.1877, class_cohesion_loss: 47.5104, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Epoch 50 - Valid Loss: 2099.0162
Sub-losses: recon_loss: 2055.9918, var_loss: 7.6010, class_separation_loss: -15.1744, class_cohesion_loss: 50.5976, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Processed 128 / 1600 samples
Processed 256 / 1600 samples
Processed 384 / 1600 samples
Processed 512 / 1600 samples
Processed 640 / 1600 samples
```

```
Processed 768 / 1600 samples
Processed 896 / 1600 samples
Processed 1024 / 1600 samples
Processed 1152 / 1600 samples
Processed 1280 / 1600 samples
Processed 1408 / 1600 samples
Processed 1536 / 1600 samples
Processed 1600 / 1600 samples
```

#### 3.1) Obtain Results[¶](#31-obtain-results)

This works as for other pipeline, but we have the two additional class-based sub-losses. See below:

In \[5\]:

Copied!

```
result.sub_losses.keys()
```

result.sub_losses.keys()

Out\[5\]:

```
dict_keys(['recon_loss', 'var_loss', 'class_separation_loss', 'class_cohesion_loss', 'anneal_factor', 'effective_beta_factor'])
```

#### 3.2) Visualize Results[¶](#32-visualize-results)

In principle this also works the same as for `Varix`. However, we might be interested in a relative loss plot since we have more different loss types.

In \[6\]:

Copied!

```
supervis.visualizer.show_loss(plot_type="relative")
```

supervis.visualizer.show_loss(plot_type="relative")

We can visualize our standard plots with `show_result`:

In \[7\]:

Copied!

```
supervis.show_result(params=["disease"])
```

supervis.show_result(params=["disease"])

```
Creating plots ...
```

### Comparing to a standard VAE run without the class-based losses[¶](#comparing-to-a-standard-vae-run-without-the-class-based-losses)

To better showcase the effect of the class-based losses, we run the plain Varix on the same data with the same set of general parameters and compare the separation classes to the Supervisix results.

In \[8\]:

Copied!

```
from autoencodix.configs.varix_config import VarixConfig

my_cfg2 = VarixConfig(
	data_case=DataCase.MULTI_SINGLE_CELL,
	loss_reduction="sum",
	k_filter=2000,
	latent_dim=3,
	scaling="STANDARD",
	epochs=50,
	learning_rate=0.01,
	batch_size = 128,
	beta = 0.1,
	use_mss = True,
	drop_p =0.1,
	global_seed=7,
	reproducible=True,
	checkpoint_interval=10,
	n_layers=1,
	annotation_columns=["cell_type", "disease"],
	data_config=DataConfig(
		data_info={"multi_sc": DataInfo(is_single_cell=True, data_type="NUMERIC")},
	),
)

varix = acx.Varix(data=sc_data, config=my_cfg2)
```

from autoencodix.configs.varix_config import VarixConfig my_cfg2 = VarixConfig( data_case=DataCase.MULTI_SINGLE_CELL, loss_reduction="sum", k_filter=2000, latent_dim=3, scaling="STANDARD", epochs=50, learning_rate=0.01, batch_size = 128, beta = 0.1, use_mss = True, drop_p =0.1, global_seed=7, reproducible=True, checkpoint_interval=10, n_layers=1, annotation_columns=["cell_type", "disease"], data_config=DataConfig( data_info={"multi_sc": DataInfo(is_single_cell=True, data_type="NUMERIC")}, ), ) varix = acx.Varix(data=sc_data, config=my_cfg2)

```
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
```

In \[9\]:

Copied!

```
result2 = varix.run()
```

result2 = varix.run()

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
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 35.1597 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 10 - Train Loss: 1731.6490
Sub-losses: recon_loss: 1730.3192, var_loss: 1.3298, anneal_factor: 0.0017, effective_beta_factor: 0.0002
Epoch 10 - Valid Loss: 1699.4385
Sub-losses: recon_loss: 1698.2469, var_loss: 1.1916, anneal_factor: 0.0017, effective_beta_factor: 0.0002
```

```
Epoch 20 - Train Loss: 1730.3793
Sub-losses: recon_loss: 1723.9799, var_loss: 6.3995, anneal_factor: 0.0832, effective_beta_factor: 0.0083
Epoch 20 - Valid Loss: 1699.3371
Sub-losses: recon_loss: 1693.8070, var_loss: 5.5301, anneal_factor: 0.0832, effective_beta_factor: 0.0083
```

```
Epoch 30 - Train Loss: 1743.2866
Sub-losses: recon_loss: 1729.7835, var_loss: 13.5031, anneal_factor: 0.8320, effective_beta_factor: 0.0832
Epoch 30 - Valid Loss: 1714.3307
Sub-losses: recon_loss: 1700.9803, var_loss: 13.3504, anneal_factor: 0.8320, effective_beta_factor: 0.0832
```

```
Epoch 40 - Train Loss: 1745.2877
Sub-losses: recon_loss: 1731.1592, var_loss: 14.1285, anneal_factor: 0.9963, effective_beta_factor: 0.0996
Epoch 40 - Valid Loss: 1713.1936
Sub-losses: recon_loss: 1700.3286, var_loss: 12.8650, anneal_factor: 0.9963, effective_beta_factor: 0.0996
```

```
Epoch 50 - Train Loss: 1745.6215
Sub-losses: recon_loss: 1731.5583, var_loss: 14.0632, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Epoch 50 - Valid Loss: 1717.2863
Sub-losses: recon_loss: 1703.7823, var_loss: 13.5039, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Processed 128 / 1600 samples
Processed 256 / 1600 samples
Processed 384 / 1600 samples
Processed 512 / 1600 samples
Processed 640 / 1600 samples
Processed 768 / 1600 samples
Processed 896 / 1600 samples
Processed 1024 / 1600 samples
Processed 1152 / 1600 samples
Processed 1280 / 1600 samples
Processed 1408 / 1600 samples
Processed 1536 / 1600 samples
Processed 1600 / 1600 samples
```

In \[10\]:

Copied!

```
varix.show_result(params=["disease"])
```

varix.show_result(params=["disease"])

```
Creating plots ...
```

Compare these plots to the first run's plots above: without the class-based losses the classes of the diseases in the data set appear much less cleanly separated and samples within a class are more spread out in the latents space.

#### 3.3) Save and Load Supervisix[¶](#33-save-and-load-supervisix)

There are no `Supervisix` specific steps here. See the `Getting Started - Vanillix` for details. Below is a basic save/load usecase:

In \[11\]:

Copied!

```
import glob
import os
outpath = os.path.join("tutorial_res", "supervisix")
supervis.save(file_path=outpath, save_all=True)

folder = os.path.dirname(outpath)
pkl_files = glob.glob(os.path.join(folder, "*.pkl"))
model_files = glob.glob(os.path.join(folder, "*.pth"))

print("PKL files:", pkl_files)
print("Model files:", model_files)

# the load functionality automatically will build the pipeline object out of the three saved files
supervis_loaded = acx.Supervisix.load(outpath)
```

import glob import os outpath = os.path.join("tutorial_res", "supervisix") supervis.save(file_path=outpath, save_all=True) folder = os.path.dirname(outpath) pkl_files = glob.glob(os.path.join(folder, "\*.pkl")) model_files = glob.glob(os.path.join(folder, "\*.pth")) print("PKL files:", pkl_files) print("Model files:", model_files)

# the load functionality automatically will build the pipeline object out of the three saved files

supervis_loaded = acx.Supervisix.load(outpath)

```
Preprocessor saved successfully.
```

```
Pipeline object saved successfully.
PKL files: ['tutorial_res/ontix_reactome_preprocessor.pkl', 'tutorial_res/ontix.pkl', 'tutorial_res/imagix_preprocessor.pkl', 'tutorial_res/varix_preprocessor.pkl', 'tutorial_res/imagix.pkl', 'tutorial_res/ontix_reactome.pkl', 'tutorial_res/supervisix_preprocessor.pkl', 'tutorial_res/maskix_preprocessor.pkl', 'tutorial_res/imagix3d_preprocessor.pkl', 'tutorial_res/xmodalix_preprocessor.pkl', 'tutorial_res/xmodalix.pkl', 'tutorial_res/imagix3d.pkl', 'tutorial_res/disent.pkl', 'tutorial_res/disent_preprocessor.pkl', 'tutorial_res/ontix_preprocessor.pkl']
Model files: ['tutorial_res/imagix3d_model.pth', 'tutorial_res/imagix_model.pth', 'tutorial_res/disent_model.pth', 'tutorial_res/supervisix_model.pth', 'tutorial_res/img.img_xmodalix_model.pth', 'tutorial
... [output truncated]
```

```
Pipeline object loaded successfully. Actual type: Supervisix
```

```
Preprocessor loaded successfully.
```

In \[12\]:

Copied!

```
supervis_loaded.predict(data=supervis.result.datasets)
```

supervis_loaded.predict(data=supervis.result.datasets)

```
Processed 128 / 1600 samples
Processed 256 / 1600 samples
Processed 384 / 1600 samples
Processed 512 / 1600 samples
Processed 640 / 1600 samples
Processed 768 / 1600 samples
Processed 896 / 1600 samples
Processed 1024 / 1600 samples
Processed 1152 / 1600 samples
Processed 1280 / 1600 samples
Processed 1408 / 1600 samples
Processed 1536 / 1600 samples
Processed 1600 / 1600 samples
```

Out\[12\]:

```
Result Object Public Attributes:
------------------------------
latentspaces: TrainingDynamics object
sample_ids: TrainingDynamics object
reconstructions: TrainingDynamics object
mus: TrainingDynamics object
sigmas: TrainingDynamics object
losses: TrainingDynamics object
sub_losses: LossRegistry(_losses={'recon_loss': TrainingDynamics(), 'var_loss': TrainingDynamics(), 'class_separation_loss': TrainingDynamics(), 'class_cohesion_loss': TrainingDynamics(), 'anneal_factor': TrainingDynamics(), 'effective_beta_factor': TrainingDynamics()})
preprocessed_data: Tensor of shape (0,)
model: VarixArchitecture
model_checkpoints: TrainingDynamics object
class_means: TrainingDynamics object
datasets: DatasetContainer(train=<autoencodix.data._numeric_dataset.NumericDataset object at 0x74cb889e4da0>, v
... [output truncated]
```

In \[13\]:

Copied!

```
supervis_loaded.show_result(params=["disease"])
```

supervis_loaded.show_result(params=["disease"])

```
Creating plots ...
```

#### Generate New Data[¶](#generate-new-data)

For a variational autoencoder, the generate or sample_latent_space step draws new latent vectors from the model’s learned latent distribution.

**Latent Sampling:** The model first aggregates the posterior over all encoded latent vectors in the chosen split and epoch by computing the mean (global_mu) and log-variance (global_logvar). It then samples new latent points from a diagonal Gaussian defined by these aggregate statistics, using the reparameterization trick to inject Gaussian noise.

**Number of Samples (n_samples):** Users can specify how many latent points to generate. The method expands the aggregated mean and log-variance to match the requested number of samples before sampling.

**Custom Latent Prior:** Optionally, a custom latent_prior can be provided (a tensor or NumPy array with shape (n_samples, latent_dim)), which will be used directly instead of the aggregated posterior. This is basically the `decode` step.

In \[14\]:

Copied!

```
reconstructions_generated = supervis_loaded.generate(n_samples=5)
print(reconstructions_generated.shape)
```

reconstructions_generated = supervis_loaded.generate(n_samples=5) print(reconstructions_generated.shape)

```
torch.Size([5, 2000])
```
