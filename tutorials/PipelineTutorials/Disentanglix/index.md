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

# How to Train and Use Disentanglix: A Disentangled VAE[¶](#how-to-train-and-use-disentanglix-a-disentangled-vae)

`Disentanglix` is a variant of the `Varix` pipeline that implements a **disentangled variational autoencoder**.\
Its key difference lies in the loss function, which decomposes the standard VAE loss into subcomponents to encourage disentanglement in the latent space with independent latent dimensions.

**IMPORTANT**

> This tutorial only shows the specifics of the Disentanglix pipeline. If you're unfamiliar with general concepts,\
> we recommend following the `Getting Started - Vanillix` tutorial first.

## What You'll Learn[¶](#what-youll-learn)

- How the loss decomposes into:
  - **Reconstruction loss** (`recon_loss`)
  - **Mutual information** (`mut_info_loss`)
  - **Total correlation** (`tot_corr_loss`)
  - **Dimension-wise KL divergence** (`dimwise_kl_loss`)
- How annealing can be applied to disentanglement weights (`beta_mi`, `beta_tc`, `beta_dimKL`)
- How to initialize, run and visualize a disentangled VAE

## 1) Loss Decomposition[¶](#1-loss-decomposition)

The `DisentanglixLoss` class computes the following terms for each batch:

1. **Reconstruction Loss** (`recon_loss`)\
   Measures how well the decoded output matches the input.
1. **Mutual Information Loss** (`mut_info_loss`)\
   Encourages each latent dimension to encode information about the inputs.
1. **Total Correlation Loss** (`tot_corr_loss`)\
   Penalizes correlations between latent dimensions to encourage independence.
1. **Dimension-wise KL Loss** (`dimwise_kl_loss`)\
   Regularizes each latent dimension individually against the prior.

The total loss is computed as a weighted sum:

```
total_loss = recon_loss 
             + beta_mi * mut_info_loss 
             + beta_tc * tot_corr_loss 
             + beta_dimKL * dimwise_kl_loss
```

## 2) Annealing[¶](#2-annealing)

Disentanglix supports flexible annealing strategies:

- no-annealing: uses constant weights throughout training
- Custom annealing functions: scales the disentanglement weights according to the current epoch

Annealing is applied independently to each loss term, allowing smooth transitions between reconstruction-focused and disentanglement-focused training.

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

## 3) Initialize and Run Disentanglix[¶](#3-initialize-and-run-disentanglix)

This works like `Varix`, except we don't have one `beta` parameter to weight the distribution loss, we have three `betas` one for each sub loss term which we can set a value for in the config.

In \[3\]:

Copied!

```
import numpy as np
import anndata as ad
import mudata

from autoencodix.configs.disentanglix_config import DisentanglixConfig
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

my_cfg = DisentanglixConfig(
	data_case=DataCase.MULTI_SINGLE_CELL,
	loss_reduction="sum",
	k_filter=2000,
	latent_dim=2,
	scaling="STANDARD",
	epochs=50,
	learning_rate=0.01,
	batch_size = 128,
	beta_mi = 0.1,
	beta_tc = 0.1,
	beta_dimKL= 0.1,
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

disent = acx.Disentanglix(data=sc_data, config=my_cfg)
```

import numpy as np import anndata as ad import mudata from autoencodix.configs.disentanglix_config import DisentanglixConfig from autoencodix.configs.default_config import DataCase, DataConfig, DataInfo from autoencodix.data.datapackage import DataPackage from huggingface_hub import hf_hub_download import autoencodix as acx sc_path = hf_hub_download( repo_id="autoencodix/census-lung", repo_type="dataset", filename="census_lung.h5ad" ) full_adata = ad.read_h5ad(sc_path)

# Balanced subsample: 2,000 cells per cell type keeps the tutorial fast to run

# while keeping all four (well-separated) cell types equally represented.

N_PER_TYPE = 2000 rng = np.random.RandomState(43) subsample_idx = [] for \_, group in full_adata.obs.groupby("cell_type", observed=True): n = min(N_PER_TYPE, len(group)) subsample_idx.extend(rng.choice(group.index.values, size=n, replace=False)) adata = full_adata[subsample_idx].copy() sc_data = DataPackage() sc_data.multi_sc = {"multi_sc": mudata.MuData({"rna": adata})} my_cfg = DisentanglixConfig( data_case=DataCase.MULTI_SINGLE_CELL, loss_reduction="sum", k_filter=2000, latent_dim=2, scaling="STANDARD", epochs=50, learning_rate=0.01, batch_size = 128, beta_mi = 0.1, beta_tc = 0.1, beta_dimKL= 0.1, use_mss = True, drop_p =0.1, global_seed=7, reproducible=True, checkpoint_interval=10, n_layers=1, annotation_columns=["cell_type", "disease"], data_config=DataConfig( data_info={"multi_sc": DataInfo(is_single_cell=True, data_type="NUMERIC")}, ), ) disent = acx.Disentanglix(data=sc_data, config=my_cfg)

```
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
```

In \[4\]:

Copied!

```
result = disent.run()
```

result = disent.run()

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
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 50.3036 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 10 - Train Loss: 1733.1841
Sub-losses: recon_loss: 1732.1143, mut_info_loss: 0.0008, tot_corr_loss: 0.0003, dimwise_kl_loss: 1.0688, anneal_factor: 0.0017, effective_beta_mi_factor: 0.0002, effective_beta_tc_factor: 0.0002, effective_beta_dimKL_factor: 0.0002
Epoch 10 - Valid Loss: 1978.9073
Sub-losses: recon_loss: 1977.7334, mut_info_loss: 0.0007, tot_corr_loss: 0.0003, dimwise_kl_loss: 1.1729, anneal_factor: 0.0017, effective_beta_mi_factor: 0.0002, effective_beta_tc_factor: 0.0002, effective_beta_dimKL_factor: 0.0002
```

```
Epoch 20 - Train Loss: 1731.1827
Sub-losses: recon_loss: 1724.9158, mut_info_loss: 0.0321, tot_corr_loss: 0.0082, dimwise_kl_loss: 6.2267, anneal_factor: 0.0832, effective_beta_mi_factor: 0.0083, effective_beta_tc_factor: 0.0083, effective_beta_dimKL_factor: 0.0083
Epoch 20 - Valid Loss: 1979.4648
Sub-losses: recon_loss: 1973.1344, mut_info_loss: 0.0319, tot_corr_loss: 0.0081, dimwise_kl_loss: 6.2905, anneal_factor: 0.0832, effective_beta_mi_factor: 0.0083, effective_beta_tc_factor: 0.0083, effective_beta_dimKL_factor: 0.0083
```

```
Epoch 30 - Train Loss: 1742.1375
Sub-losses: recon_loss: 1729.7685, mut_info_loss: 0.2609, tot_corr_loss: 0.0384, dimwise_kl_loss: 12.0697, anneal_factor: 0.8320, effective_beta_mi_factor: 0.0832, effective_beta_tc_factor: 0.0832, effective_beta_dimKL_factor: 0.0832
Epoch 30 - Valid Loss: 1988.7086
Sub-losses: recon_loss: 1975.8932, mut_info_loss: 0.2590, tot_corr_loss: 0.0409, dimwise_kl_loss: 12.5155, anneal_factor: 0.8320, effective_beta_mi_factor: 0.0832, effective_beta_tc_factor: 0.0832, effective_beta_dimKL_factor: 0.0832
```

```
Epoch 40 - Train Loss: 1739.4153
Sub-losses: recon_loss: 1726.6906, mut_info_loss: 0.3056, tot_corr_loss: 0.0413, dimwise_kl_loss: 12.3778, anneal_factor: 0.9963, effective_beta_mi_factor: 0.0996, effective_beta_tc_factor: 0.0996, effective_beta_dimKL_factor: 0.0996
Epoch 40 - Valid Loss: 1986.8330
Sub-losses: recon_loss: 1973.7732, mut_info_loss: 0.3032, tot_corr_loss: 0.0379, dimwise_kl_loss: 12.7188, anneal_factor: 0.9963, effective_beta_mi_factor: 0.0996, effective_beta_tc_factor: 0.0996, effective_beta_dimKL_factor: 0.0996
```

```
Epoch 50 - Train Loss: 1739.3977
Sub-losses: recon_loss: 1726.5200, mut_info_loss: 0.3076, tot_corr_loss: 0.0401, dimwise_kl_loss: 12.5301, anneal_factor: 0.9999, effective_beta_mi_factor: 0.1000, effective_beta_tc_factor: 0.1000, effective_beta_dimKL_factor: 0.1000
Epoch 50 - Valid Loss: 1987.1413
Sub-losses: recon_loss: 1973.8067, mut_info_loss: 0.3062, tot_corr_loss: 0.0376, dimwise_kl_loss: 12.9907, anneal_factor: 0.9999, effective_beta_mi_factor: 0.1000, effective_beta_tc_factor: 0.1000, effective_beta_dimKL_factor: 0.1000
Processed 128 / 1600 samples
Processed 256 / 1600 samples
Processed 384 / 1600 samples
Processed 512 / 1600 samples
Processed 640 / 1600 samples
Processed 768 / 1600 samples
Processed 896 / 1600 samples
Processed 1024 / 1600 samples
Processed 1152 / 1600 samples
Pr
... [output truncated]
```

#### 3.1) Obtain Results[¶](#31-obtain-results)

This works as for other pipeline, but we have more sub-losses. See below:

In \[5\]:

Copied!

```
result.sub_losses.keys()
```

result.sub_losses.keys()

Out\[5\]:

```
dict_keys(['recon_loss', 'mut_info_loss', 'tot_corr_loss', 'dimwise_kl_loss', 'anneal_factor', 'effective_beta_mi_factor', 'effective_beta_tc_factor', 'effective_beta_dimKL_factor'])
```

#### 3.2) Visualize Results[¶](#32-visualize-results)

In principle this also works the same as for `Varix`. However, we might be interested in a relative loss plot since we have more different loss types.

In \[6\]:

Copied!

```
disent.visualizer.show_loss(plot_type="relative")
```

disent.visualizer.show_loss(plot_type="relative")

We can visualize our standard plots with `show_result`:

In \[7\]:

Copied!

```
disent.show_result(params=["cell_type", "disease"])
```

disent.show_result(params=["cell_type", "disease"])

```
Creating plots ...
```

### Comparing a Second `beta_tc` Setting[¶](#comparing-a-second-beta_tc-setting)

To show what disentanglement regularization actually does, we run the pipeline a second time with `beta_tc` raised from `0.1` to `100` (`beta_dimKL` is lowered from `0.1` to `0.01` to compensate, keeping the total loss balanced). `beta_tc` weights the total-correlation term, which directly penalizes correlation between latent dimensions. A higher value therefore results in stronger disentanglement.

In \[8\]:

Copied!

```
my_cfg2 = DisentanglixConfig(
	data_case=DataCase.MULTI_SINGLE_CELL,
	loss_reduction="sum",
	k_filter=2000,
	latent_dim=2,
	scaling="STANDARD",
	epochs=50,
	learning_rate=0.01,
	batch_size = 128,
	beta_mi = 0.1,
	beta_tc = 100,
	beta_dimKL= 0.01,
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

disent2 = acx.Disentanglix(data=sc_data, config=my_cfg2)
```

my_cfg2 = DisentanglixConfig( data_case=DataCase.MULTI_SINGLE_CELL, loss_reduction="sum", k_filter=2000, latent_dim=2, scaling="STANDARD", epochs=50, learning_rate=0.01, batch_size = 128, beta_mi = 0.1, beta_tc = 100, beta_dimKL= 0.01, use_mss = True, drop_p =0.1, global_seed=7, reproducible=True, checkpoint_interval=10, n_layers=1, annotation_columns=["cell_type", "disease"], data_config=DataConfig( data_info={"multi_sc": DataInfo(is_single_cell=True, data_type="NUMERIC")}, ), ) disent2 = acx.Disentanglix(data=sc_data, config=my_cfg2)

```
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
```

In \[9\]:

Copied!

```
result2 = disent2.run()
```

result2 = disent2.run()

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
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 55.7621 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 10 - Train Loss: 1822.2642
Sub-losses: recon_loss: 1822.1652, mut_info_loss: 0.0006, tot_corr_loss: 0.0000, dimwise_kl_loss: 0.0984, anneal_factor: 0.0017, effective_beta_mi_factor: 0.0002, effective_beta_tc_factor: 0.1659, effective_beta_dimKL_factor: 0.0000
Epoch 10 - Valid Loss: 1772.0760
Sub-losses: recon_loss: 1771.9657, mut_info_loss: 0.0006, tot_corr_loss: 0.0000, dimwise_kl_loss: 0.1097, anneal_factor: 0.0017, effective_beta_mi_factor: 0.0002, effective_beta_tc_factor: 0.1659, effective_beta_dimKL_factor: 0.0000
```

```
Epoch 20 - Train Loss: 1812.0741
Sub-losses: recon_loss: 1809.0657, mut_info_loss: 0.0266, tot_corr_loss: 0.0000, dimwise_kl_loss: 2.9819, anneal_factor: 0.0832, effective_beta_mi_factor: 0.0083, effective_beta_tc_factor: 8.3173, effective_beta_dimKL_factor: 0.0008
Epoch 20 - Valid Loss: 1773.9799
Sub-losses: recon_loss: 1771.1935, mut_info_loss: 0.0265, tot_corr_loss: 0.0000, dimwise_kl_loss: 2.7598, anneal_factor: 0.0832, effective_beta_mi_factor: 0.0083, effective_beta_tc_factor: 8.3173, effective_beta_dimKL_factor: 0.0008
```

```
Epoch 30 - Train Loss: 1813.0499
Sub-losses: recon_loss: 1807.7945, mut_info_loss: 0.2067, tot_corr_loss: 0.0000, dimwise_kl_loss: 5.0487, anneal_factor: 0.8320, effective_beta_mi_factor: 0.0832, effective_beta_tc_factor: 83.2018, effective_beta_dimKL_factor: 0.0083
Epoch 30 - Valid Loss: 1783.4342
Sub-losses: recon_loss: 1778.6270, mut_info_loss: 0.2004, tot_corr_loss: 0.0000, dimwise_kl_loss: 4.6069, anneal_factor: 0.8320, effective_beta_mi_factor: 0.0832, effective_beta_tc_factor: 83.2018, effective_beta_dimKL_factor: 0.0083
```

```
Epoch 40 - Train Loss: 1810.8446
Sub-losses: recon_loss: 1805.6993, mut_info_loss: 0.2365, tot_corr_loss: 0.0000, dimwise_kl_loss: 4.9089, anneal_factor: 0.9963, effective_beta_mi_factor: 0.0996, effective_beta_tc_factor: 99.6316, effective_beta_dimKL_factor: 0.0100
Epoch 40 - Valid Loss: 1777.4261
Sub-losses: recon_loss: 1772.2850, mut_info_loss: 0.2393, tot_corr_loss: 0.0000, dimwise_kl_loss: 4.9018, anneal_factor: 0.9963, effective_beta_mi_factor: 0.0996, effective_beta_tc_factor: 99.6316, effective_beta_dimKL_factor: 0.0100
```

```
Epoch 50 - Train Loss: 1805.4972
Sub-losses: recon_loss: 1800.3266, mut_info_loss: 0.2389, tot_corr_loss: 0.0000, dimwise_kl_loss: 4.9317, anneal_factor: 0.9999, effective_beta_mi_factor: 0.1000, effective_beta_tc_factor: 99.9932, effective_beta_dimKL_factor: 0.0100
Epoch 50 - Valid Loss: 1778.3342
Sub-losses: recon_loss: 1772.8539, mut_info_loss: 0.2432, tot_corr_loss: 0.0000, dimwise_kl_loss: 5.2372, anneal_factor: 0.9999, effective_beta_mi_factor: 0.1000, effective_beta_tc_factor: 99.9932, effective_beta_dimKL_factor: 0.0100
Processed 128 / 1600 samples
Processed 256 / 1600 samples
Processed 384 / 1600 samples
Processed 512 / 1600 samples
Processed 640 / 1600 samples
Processed 768 / 1600 samples
Processed 896 / 1600 samples
Processed 1024 / 1600 samples
Processed 1152 / 1600 samples
Pr
... [output truncated]
```

In \[10\]:

Copied!

```
disent2.visualizer.show_loss(plot_type="absolute")
```

disent2.visualizer.show_loss(plot_type="absolute")

In \[11\]:

Copied!

```
disent2.visualizer.show_latent_space(result=result2, param=["cell_type", "disease"], plot_type="2D-scatter")
```

disent2.visualizer.show_latent_space(result=result2, param=["cell_type", "disease"], plot_type="2D-scatter")

Compare these plots to the first run's plots above: with the higher `beta_tc`, the latent dimensions tend to separate more along the axes rather than mixing diagonally, which is a sign of stronger disentanglement.

#### 3.3) Save and Load Disentanglix[¶](#33-save-and-load-disentanglix)

There are not `Disentanglix` specific steps here. See the `Getting Started - Vanillix` for details. Below is a basic save/load usecase:

In \[12\]:

Copied!

```
import glob
import os
outpath = os.path.join("tutorial_res", "disent.pkl")
disent.save(file_path=outpath, save_all=True)

folder = os.path.dirname(outpath)
pkl_files = glob.glob(os.path.join(folder, "*.pkl"))
model_files = glob.glob(os.path.join(folder, "*.pth"))

print("PKL files:", pkl_files)
print("Model files:", model_files)

# the load functionality automatically will build the pipeline object out of the three saved files
disent_loaded = acx.Disentanglix.load(outpath)
```

import glob import os outpath = os.path.join("tutorial_res", "disent.pkl") disent.save(file_path=outpath, save_all=True) folder = os.path.dirname(outpath) pkl_files = glob.glob(os.path.join(folder, "\*.pkl")) model_files = glob.glob(os.path.join(folder, "\*.pth")) print("PKL files:", pkl_files) print("Model files:", model_files)

# the load functionality automatically will build the pipeline object out of the three saved files

disent_loaded = acx.Disentanglix.load(outpath)

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
Pipeline object loaded successfully. Actual type: Disentanglix
```

```
Preprocessor loaded successfully.
```

In \[13\]:

Copied!

```
disent_loaded.predict(data=disent.result.datasets)
```

disent_loaded.predict(data=disent.result.datasets)

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

Out\[13\]:

```
Result Object Public Attributes:
------------------------------
latentspaces: TrainingDynamics object
sample_ids: TrainingDynamics object
reconstructions: TrainingDynamics object
mus: TrainingDynamics object
sigmas: TrainingDynamics object
losses: TrainingDynamics object
sub_losses: LossRegistry(_losses={'recon_loss': TrainingDynamics(), 'mut_info_loss': TrainingDynamics(), 'tot_corr_loss': TrainingDynamics(), 'dimwise_kl_loss': TrainingDynamics(), 'anneal_factor': TrainingDynamics(), 'effective_beta_mi_factor': TrainingDynamics(), 'effective_beta_tc_factor': TrainingDynamics(), 'effective_beta_dimKL_factor': TrainingDynamics()})
preprocessed_data: Tensor of shape (0,)
model: VarixArchitecture
model_checkpoints: TrainingDynamics object
class_means: TrainingDynamics object
datasets: Datase
... [output truncated]
```

In \[14\]:

Copied!

```
disent_loaded.show_result(params=["cell_type", "disease"])
```

disent_loaded.show_result(params=["cell_type", "disease"])

```
Creating plots ...
```

#### Generate New Data[¶](#generate-new-data)

For a variational autoencoder, the generate or sample_latent_space step draws new latent vectors from the model’s learned latent distribution.

**Latent Sampling:** The model first aggregates the posterior over all encoded latent vectors in the chosen split and epoch by computing the mean (global_mu) and log-variance (global_logvar). It then samples new latent points from a diagonal Gaussian defined by these aggregate statistics, using the reparameterization trick to inject Gaussian noise.

**Number of Samples (n_samples):** Users can specify how many latent points to generate. The method expands the aggregated mean and log-variance to match the requested number of samples before sampling.

**Custom Latent Prior:** Optionally, a custom latent_prior can be provided (a tensor or NumPy array with shape (n_samples, latent_dim)), which will be used directly instead of the aggregated posterior. This is basically the `decode` step.

In \[15\]:

Copied!

```
reconstructions_generated = disent_loaded.generate(n_samples=5)
print(reconstructions_generated.shape)
```

reconstructions_generated = disent_loaded.generate(n_samples=5) print(reconstructions_generated.shape)

```
torch.Size([5, 2000])
```
