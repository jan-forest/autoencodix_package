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

# How To Use Stackix[¶](#how-to-use-stackix)

Stackix is our implementation of a stacked/hierarchical autoencoder.\
This tutorial follows the structure of our `Getting Started - Vanillix`, but is less extensive because\
our pipeline works similarly for different architectures. Here, we focus only on Stackix specifics.

**AUTOENCODIX** supports far more functionality than shown here, so we’ll also point to advanced tutorials where relevant.

**IMPORTANT**

> This tutorial only shows the specifics of the Stackix pipeline. If you're unfamiliar with general concepts,\
> we recommend following the [`Getting Started - Vanillix`](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/Stackix/Vanillix.ipynb) tutorial first.

## Stackix Theory[¶](#stackix-theory)

In our implementation, we train one variational autoencoder (VAE) per data modality end-to-end.\
The latent spaces of these *outer* autoencoders are then concatenated and used as input to another *inner* autoencoder.\
Downstream visualization and evaluation are performed on this **meta-latent space**, which represents a joint embedding of all modalities.

For unpaired data, each outer autoencoder is trained on all available samples of its modality, even if the corresponding sample is missing in other modalities.\
When constructing the input for the inner autoencoder, non-overlapping samples are dropped to ensure consistency across modalities.

This approach has two main goals:

1. To produce more informative latent spaces for each modality.
1. To provide a richer, joint representation for the inner autoencoder.

We also store the dropped samples and their indices. This allows us to reconstruct the full dataset: the output of the inner autoencoder can be passed back through the outer autoencoders, where decoding recovers the original modalities.

## What You'll Learn[¶](#what-youll-learn)

You’ll learn how to:

1. **Initialize** the pipeline and run it.
1. Understand the Stackix-specific **pipeline steps** (paired vs. unpaired data).
1. Access the Stackix-specific **results** (sub-results for "outer" autoencoders).
1. **Visualize** outputs.
1. Apply **custom parameters**.
1. **Save, load, and reuse** a trained pipeline.

Let’s get started! 🚀

## 1) Initialize and Run Stackix[¶](#1-initialize-and-run-stackix)

We set a few custom parameters of the config file. For a deep dive into the config object, see:

[`Tutorials/DeepDives/ConfigTutorial.ipynb`](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/DeepDives/ConfigTutorial.ipynb)

#### 1.1 The Dataset[¶](#11-the-dataset)

We use a real multi-omics dataset: a balanced subset of the TCGA pan-cancer cohort ([`autoencodix/tcga`](https://huggingface.co/datasets/autoencodix/tcga) on Hugging Face), combining RNA-seq and DNA methylation. The dataset is accompanied by clinical annotation data containing metadata such as cancer type.

In \[2\]:

Copied!

```
import pandas as pd
from huggingface_hub import hf_hub_download

import autoencodix as acx
from autoencodix.configs.default_config import DataCase
from autoencodix.configs.stackix_config import StackixConfig
from autoencodix.data.datapackage import DataPackage

# Data is hosted on Hugging Face Hub (autoencodix/tcga) and downloaded (and locally cached) automatically
HF_REPO_ID = "autoencodix/tcga"
rna_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="rna.parquet")
meth_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="methylation.parquet")
clin_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="clinical.parquet")

rna_full = pd.read_parquet(rna_path)
meth_full = pd.read_parquet(meth_path)
clin_full = pd.read_parquet(clin_path)

# Keep only samples present in all three modalities
shared_ids = rna_full.index.intersection(meth_full.index).intersection(clin_full.index)

# Balanced subsample across the 5 broad TCGA cancer types
n_per_type = 120
sampled_ids = (
    clin_full.loc[shared_ids]
    .groupby("CANCER_TYPE", group_keys=False)
    .apply(lambda g: g.sample(n=min(n_per_type, len(g)), random_state=42), include_groups=False)
    .index
)

raw_rna = rna_full.loc[sampled_ids]
raw_meth = meth_full.loc[sampled_ids]
annotation = clin_full.loc[sampled_ids]

tcga_multi_bulk = DataPackage(
    multi_bulk={"rna": raw_rna, "meth": raw_meth},
    annotation={"paired": annotation},
)

my_config = StackixConfig(
    epochs=100,
    beta=0.01,
    checkpoint_interval=20,
    default_vae_loss="kl",  # kl or mmd possible
    data_case=DataCase.MULTI_BULK,
    k_filter=2000,
    reproducible=True,
    global_seed=5,
)

print("\n")
print("Starting Pipeline")
print("-" * 50)
print("-" * 50)
stackix = acx.Stackix(data=tcga_multi_bulk, config=my_config)
result = stackix.run()
```

import pandas as pd from huggingface_hub import hf_hub_download import autoencodix as acx from autoencodix.configs.default_config import DataCase from autoencodix.configs.stackix_config import StackixConfig from autoencodix.data.datapackage import DataPackage

# Data is hosted on Hugging Face Hub (autoencodix/tcga) and downloaded (and locally cached) automatically

HF_REPO_ID = "autoencodix/tcga" rna_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="rna.parquet") meth_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="methylation.parquet") clin_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="clinical.parquet") rna_full = pd.read_parquet(rna_path) meth_full = pd.read_parquet(meth_path) clin_full = pd.read_parquet(clin_path)

# Keep only samples present in all three modalities

shared_ids = rna_full.index.intersection(meth_full.index).intersection(clin_full.index)

# Balanced subsample across the 5 broad TCGA cancer types

n_per_type = 120 sampled_ids = ( clin_full.loc[shared_ids] .groupby("CANCER_TYPE", group_keys=False) .apply(lambda g: g.sample(n=min(n_per_type, len(g)), random_state=42), include_groups=False) .index ) raw_rna = rna_full.loc[sampled_ids] raw_meth = meth_full.loc[sampled_ids] annotation = clin_full.loc[sampled_ids] tcga_multi_bulk = DataPackage( multi_bulk={"rna": raw_rna, "meth": raw_meth}, annotation={"paired": annotation}, ) my_config = StackixConfig( epochs=100, beta=0.01, checkpoint_interval=20, default_vae_loss="kl", # kl or mmd possible data_case=DataCase.MULTI_BULK, k_filter=2000, reproducible=True, global_seed=5, ) print("\\n") print("Starting Pipeline") print("-" * 50) print("-" * 50) stackix = acx.Stackix(data=tcga_multi_bulk, config=my_config) result = stackix.run()

```
Starting Pipeline
--------------------------------------------------
--------------------------------------------------
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
anno key: paired
Reproducibility settings for device auto are not implemented or necessary i.e. for cpu.
Training each modality model...
Training modality: rna
Training modality: rna
Reproducibility settings for device auto are not implemented or necessary i.e. for cpu.
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:233: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 9.6730 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 20 - Train Loss: 862.4087
Sub-losses: recon_loss: 862.4078, var_loss: 0.0009, anneal_factor: 0.0020, effective_beta_factor: 0.0000
Epoch 20 - Valid Loss: 802.0719
Sub-losses: recon_loss: 802.0713, var_loss: 0.0006, anneal_factor: 0.0020, effective_beta_factor: 0.0000
Epoch 40 - Train Loss: 806.1211
Sub-losses: recon_loss: 806.0600, var_loss: 0.0611, anneal_factor: 0.0998, effective_beta_factor: 0.0010
Epoch 40 - Valid Loss: 740.7936
Sub-losses: recon_loss: 740.7423, var_loss: 0.0513, anneal_factor: 0.0998, effective_beta_factor: 0.0010
Epoch 60 - Train Loss: 763.1939
Sub-losses: recon_loss: 762.3805, var_loss: 0.8135, anneal_factor: 0.8581, effective_beta_factor: 0.0086
Epoch 60 - Valid Loss: 727.0296
Sub-losses: recon_loss: 726.3649, var_loss: 0.6646, anneal_factor: 0.8581, effecti
... [output truncated]
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:233: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 11.1517 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 20 - Train Loss: 661.1864
Sub-losses: recon_loss: 661.1857, var_loss: 0.0007, anneal_factor: 0.0020, effective_beta_factor: 0.0000
Epoch 20 - Valid Loss: 655.0351
Sub-losses: recon_loss: 655.0345, var_loss: 0.0007, anneal_factor: 0.0020, effective_beta_factor: 0.0000
Epoch 40 - Train Loss: 612.0515
Sub-losses: recon_loss: 611.9937, var_loss: 0.0579, anneal_factor: 0.0998, effective_beta_factor: 0.0010
Epoch 40 - Valid Loss: 603.8224
Sub-losses: recon_loss: 603.7703, var_loss: 0.0520, anneal_factor: 0.0998, effective_beta_factor: 0.0010
Epoch 60 - Train Loss: 561.3298
Sub-losses: recon_loss: 560.5643, var_loss: 0.7655, anneal_factor: 0.8581, effective_beta_factor: 0.0086
Epoch 60 - Valid Loss: 562.8977
Sub-losses: recon_loss: 562.1656, var_loss: 0.7321, anneal_factor: 0.8581, effecti
... [output truncated]
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:233: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 73.4787 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 20 - Train Loss: 363.1186
Sub-losses: recon_loss: 363.1184, var_loss: 0.0002, anneal_factor: 0.0020, effective_beta_factor: 0.0000
Epoch 20 - Valid Loss: 321.1639
Sub-losses: recon_loss: 321.1637, var_loss: 0.0002, anneal_factor: 0.0020, effective_beta_factor: 0.0000
Epoch 40 - Train Loss: 247.1598
Sub-losses: recon_loss: 247.1190, var_loss: 0.0408, anneal_factor: 0.0998, effective_beta_factor: 0.0010
Epoch 40 - Valid Loss: 209.5626
Sub-losses: recon_loss: 209.5314, var_loss: 0.0312, anneal_factor: 0.0998, effective_beta_factor: 0.0010
Epoch 60 - Train Loss: 214.3203
Sub-losses: recon_loss: 213.7860, var_loss: 0.5344, anneal_factor: 0.8581, effective_beta_factor: 0.0086
Epoch 60 - Valid Loss: 176.2616
Sub-losses: recon_loss: 175.7611, var_loss: 0.5005, anneal_factor: 0.8581, effecti
... [output truncated]
```

## 2) Stackix-Specific Steps[¶](#2-stackix-specific-steps)

Stackix does not introduce additional steps to the overall pipeline. However, its preprocessing and training differ slightly from Varix and Vanillix. Stackix processes each data modality individually and thus supports unpaired data input.

This is enabled by setting the config parameter `requires_paired=False`. In the following example, we build unpaired data by dropping some samples from our raw_rna data. The outer autoencoder for the methylation modality still trains on all methylation samples, including those without corresponding RNA. These RNA samples are only excluded when concatenating the latent spaces for the inner autoencoder. Later, we can add these samples back to the inner AE output and feed them into the decoder of the outer methylation autoencoder, allowing full reconstruction.

In \[3\]:

Copied!

```
import pandas as pd

from autoencodix.data.datapackage import DataPackage
from autoencodix.configs.stackix_config import StackixConfig
from autoencodix.configs.default_config import DataCase
import autoencodix as acx

dropped_ids = raw_rna.sample(n=20, random_state=42).index

rna_anno: pd.DataFrame = annotation.drop(index=dropped_ids)
print(rna_anno.shape)
rna_unpaired: pd.DataFrame = raw_rna.drop(index=dropped_ids)
print(rna_unpaired.shape)
unpaired_dp: DataPackage = DataPackage(
    multi_bulk={"rna": rna_unpaired, "meth": raw_meth},
    annotation={"rna": rna_anno, "meth": annotation},
)
unpaired_config: StackixConfig = StackixConfig(
    data_case=DataCase.MULTI_BULK, requires_paired=False
)
unpaired_stackix: acx.Stackix = acx.Stackix(data=unpaired_dp, config=unpaired_config)
unpaired_result = unpaired_stackix.run()
```

import pandas as pd from autoencodix.data.datapackage import DataPackage from autoencodix.configs.stackix_config import StackixConfig from autoencodix.configs.default_config import DataCase import autoencodix as acx dropped_ids = raw_rna.sample(n=20, random_state=42).index rna_anno: pd.DataFrame = annotation.drop(index=dropped_ids) print(rna_anno.shape) rna_unpaired: pd.DataFrame = raw_rna.drop(index=dropped_ids) print(rna_unpaired.shape) unpaired_dp: DataPackage = DataPackage( multi_bulk={"rna": rna_unpaired, "meth": raw_meth}, annotation={"rna": rna_anno, "meth": annotation}, ) unpaired_config: StackixConfig = StackixConfig( data_case=DataCase.MULTI_BULK, requires_paired=False ) unpaired_stackix: acx.Stackix = acx.Stackix(data=unpaired_dp, config=unpaired_config) unpaired_result = unpaired_stackix.run()

```
(580, 56)
(580, 17448)
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
anno key: rna
anno key: meth
Training each modality model...
Training modality: rna
Training modality: rna
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:233: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 10.8362 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 3 - Train Loss: 53474.2982
Sub-losses: recon_loss: 18050.3304, var_loss: 35423.9675, anneal_factor: 0.9656, effective_beta_factor: 0.0966
Epoch 3 - Valid Loss: 2383693.7091
Sub-losses: recon_loss: 2383298.5129, var_loss: 395.0898, anneal_factor: 0.9656, effective_beta_factor: 0.0966
Training modality: meth
Training modality: meth
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:233: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 11.6226 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 3 - Train Loss: 7836.0943
Sub-losses: recon_loss: 7752.6758, var_loss: 83.4184, anneal_factor: 0.9656, effective_beta_factor: 0.0966
Epoch 3 - Valid Loss: 11965.9589
Sub-losses: recon_loss: 10710.3828, var_loss: 1255.5765, anneal_factor: 0.9656, effective_beta_factor: 0.0966
Found 406 common samples for the stacked autoencoder.
Found 58 common samples for the stacked autoencoder.
finished training each modality model
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:233: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 3753.9534 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 3 - Train Loss: 1097675.9266
Sub-losses: recon_loss: 1097672.9758, var_loss: 2.9551, anneal_factor: 0.9656, effective_beta_factor: 0.0966
Epoch 3 - Valid Loss: 14828.0579
Sub-losses: recon_loss: 14827.8384, var_loss: 0.2195, anneal_factor: 0.9656, effective_beta_factor: 0.0966
Found 116 common samples for the stacked autoencoder.
<autoencodix.data._numeric_dataset.NumericDataset object at 0x7891a6988ca0>
Successfully created annotated latent space object (adata_latent).
```

## 3) Access Stackix-Specific Outputs[¶](#3-access-stackix-specific-outputs)

As explained above, we train *outer* autoencoders for each data modality and one shared *inner* autoencoder.\
The results (losses, reconstructions, latent spaces, etc.) for the *inner* autoencoder can be accessed in the same way as for the Vanillix autoencoder (see [1] and [2]).\
The results for the *outer* autoencoders are stored in a special `sub_results` structure, as illustrated in the Python code below.

The `sub_results` can be accessed as an attribute of the `result` object.\
On the first level, this is a `Dict` with keys corresponding to the data modalities (as defined in Step 1).\
The value for each key is a `result` object analogous to the general result object.\
These can be accessed via our standard API (see also [2]).

[1] [`Tutorials/PipelineTutorials/Vanillix.ipynb`](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/Stackix/Vanillix.ipynb)\
[2] [`Tutorials/DeepDives/PipelineOutputTutorial.ipynb`](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/DeepDives/PipelineOutputTutorial.ipynb)

In \[4\]:

Copied!

```
outer_result = result.sub_results
print(outer_result.keys())
```

outer_result = result.sub_results print(outer_result.keys())

```
dict_keys(['rna', 'meth'])
```

In \[5\]:

Copied!

```
print(outer_result["meth"])
# access latent space of outer methylation autoencoder
outer_result["meth"].latentspaces.get(split="test", epoch=-1)
```

print(outer_result["meth"])

# access latent space of outer methylation autoencoder

outer_result["meth"].latentspaces.get(split="test", epoch=-1)

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
new_datasets: DatasetContainer(train=None, valid=None, test=None)
adata_latent: AnnData object with n_obs × n_vars = 0 × 0
final_reconstruction: None
sub_results: None
sub_re
... [output truncated]
```

Out\[5\]:

```
array([[ 0.881716  ,  1.2293729 ,  0.72128266, ...,  2.4728608 ,
         6.5299053 ,  1.1532463 ],
       [ 1.9333293 ,  1.5813494 ,  1.2028021 , ...,  5.103081  ,
         4.5182433 ,  0.36976436],
       [ 2.459379  ,  5.0468388 ,  5.430481  , ..., -1.0237892 ,
         7.628327  ,  2.1867306 ],
       ...,
       [ 7.0717545 ,  6.199747  ,  1.9782643 , ...,  5.257984  ,
        11.63677   ,  2.5935605 ],
       [ 0.21196252,  0.7826589 ,  0.6282544 , ...,  3.063758  ,
         0.6773172 ,  1.109165  ],
       [ 1.9420035 ,  8.156875  , 11.350476  , ..., -0.08163258,
        -0.7722243 ,  7.3420835 ]], shape=(120, 16), dtype=float32)
```

In \[6\]:

Copied!

```
# the inner result can be accessed directly
result.latentspaces.get(split="test", epoch=-1)
# or as dataframe with ids
result.get_latent_df(split="test", epoch=-1)
```

# the inner result can be accessed directly

result.latentspaces.get(split="test", epoch=-1)

# or as dataframe with ids

result.get_latent_df(split="test", epoch=-1)

Out\[6\]:

```
LatDim_0
LatDim_1
LatDim_2
LatDim_3
LatDim_4
LatDim_5
LatDim_6
LatDim_7
LatDim_8
LatDim_9
LatDim_10
LatDim_11
LatDim_12
LatDim_13
LatDim_14
LatDim_15




TCGA-05-4250-01
-0.178447
2.736038
-0.746352
-0.764565
-0.667412
1.717845
-1.600573
-2.225891
-0.023309
-0.783873
3.320147
4.199130
2.715884
-1.023837
4.936493
4.073131


TCGA-05-4410-01
1.164364
2.972247
0.209474
1.088743
-1.119766
0.989842
-0.003885
0.724461
-0.071547
0.963581
0.872269
0.817888
1.490780
0.051327
4.229981
1.636360


TCGA-05-4420-01
1.361521
1.197402
-0.829542
0.702733
-0.512372
1.131414
-1.352620
-0.470067
-0.032610
1.409997
1.676570
0.123716
2.481240
-0.988570
2.873714
2.274129


TCGA-09-1669-01
2.612077
-1.159732
7.229315
3.524749
6.000990
1.690343
4.885739
3.818074
-0.478044
3.715651
0.476996
0.767875
1.055437
... [output truncated]
```

## 4) Visualize Outputs[¶](#4-visualize-outputs)

This works exactly as for our other pipelines, by calling the `show_result` method.\
For more information, see our [Visualization Deep Dive](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/DeepDives/VisualizeTutorial.ipynb).

We can pass a column from the annotation file using the keyword argument `params` to color the plots according to this column.

[3] [`Tutorials/DeepDives/VisualizeTutorial.ipynb`](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/DeepDives/VisualizeTutorial.ipynb)

In \[7\]:

Copied!

```
stackix.show_result(params=["CANCER_TYPE"])
```

stackix.show_result(params=["CANCER_TYPE"])

```
Creating plots ...
```

## 5) Customize Stackix Parameters[¶](#5-customize-stackix-parameters)

Since Stackix is composed of variational autoencoders, there are no Stackix-specific customizations that were not already shown in the `Varix` tutorial.

However, it is possible to change the architecture to a vanilla autoencoder via the `model_type` attribute when initializing Stackix.\
Please note that this attribute expects a **class type**, not an instance of the class.\
Therefore, you need to import the appropriate `Architecture` (see [API reference](https://maxjoas.github.io/autoencodix_package/api/modeling)) and the corresponding loss type (e.g., `VanillixLoss` if using a vanilla autoencoder).

See the code below for a clearer example.

In \[8\]:

Copied!

```
import autoencodix as acx
from autoencodix.configs.default_config import DataCase
from autoencodix.configs.stackix_config import StackixConfig
from autoencodix.modeling import VanillixArchitecture
from autoencodix.utils import VanillixLoss

my_config = StackixConfig(
    epochs=27,
    checkpoint_interval=5,
    data_case=DataCase.MULTI_BULK,
    k_filter=2000,
)
stackix_custom = acx.Stackix(
    data=tcga_multi_bulk,
    config=my_config,
    model_type=VanillixArchitecture,
    loss_type=VanillixLoss,
)
result_custom = stackix_custom.run()
```

import autoencodix as acx from autoencodix.configs.default_config import DataCase from autoencodix.configs.stackix_config import StackixConfig from autoencodix.modeling import VanillixArchitecture from autoencodix.utils import VanillixLoss my_config = StackixConfig( epochs=27, checkpoint_interval=5, data_case=DataCase.MULTI_BULK, k_filter=2000, ) stackix_custom = acx.Stackix( data=tcga_multi_bulk, config=my_config, model_type=VanillixArchitecture, loss_type=VanillixLoss, ) result_custom = stackix_custom.run()

```
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
anno key: paired
Training each modality model...
Training modality: rna
Training modality: rna
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:233: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 23.0555 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 5 - Train Loss: 856.5413
Sub-losses: recon_loss: 856.5413
Epoch 5 - Valid Loss: 1227.5358
Sub-losses: recon_loss: 1227.5358
Epoch 10 - Train Loss: 796.2182
Sub-losses: recon_loss: 796.2182
Epoch 10 - Valid Loss: 1179.8255
Sub-losses: recon_loss: 1179.8255
Epoch 15 - Train Loss: 761.0203
Sub-losses: recon_loss: 761.0203
Epoch 15 - Valid Loss: 1156.3602
Sub-losses: recon_loss: 1156.3602
Epoch 20 - Train Loss: 740.2954
Sub-losses: recon_loss: 740.2954
Epoch 20 - Valid Loss: 1145.7398
Sub-losses: recon_loss: 1145.7398
Epoch 25 - Train Loss: 738.6436
Sub-losses: recon_loss: 738.6436
Epoch 25 - Valid Loss: 1138.0193
Sub-losses: recon_loss: 1138.0193
Epoch 27 - Train Loss: 722.9525
Sub-losses: recon_loss: 722.9525
Epoch 27 - Valid Loss: 1129.5311
Sub-losses: recon_loss: 1129.5311
Training
... [output truncated]
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:233: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 19.4628 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 5 - Train Loss: 659.6040
Sub-losses: recon_loss: 659.6040
Epoch 5 - Valid Loss: 531.5841
Sub-losses: recon_loss: 531.5841
Epoch 10 - Train Loss: 602.5172
Sub-losses: recon_loss: 602.5172
Epoch 10 - Valid Loss: 484.7523
Sub-losses: recon_loss: 484.7523
Epoch 15 - Train Loss: 577.7273
Sub-losses: recon_loss: 577.7273
Epoch 15 - Valid Loss: 466.4911
Sub-losses: recon_loss: 466.4911
Epoch 20 - Train Loss: 573.4865
Sub-losses: recon_loss: 573.4865
Epoch 20 - Valid Loss: 463.3812
Sub-losses: recon_loss: 463.3812
Epoch 25 - Train Loss: 558.4631
Sub-losses: recon_loss: 558.4631
Epoch 25 - Valid Loss: 447.9033
Sub-losses: recon_loss: 447.9033
Epoch 27 - Train Loss: 538.6525
Sub-losses: recon_loss: 538.6525
Epoch 27 - Valid Loss: 448.4826
Sub-losses: recon_loss: 448.4826
Found 420 common samp
... [output truncated]
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:233: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 35.7969 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 5 - Train Loss: 25.4678
Sub-losses: recon_loss: 25.4678
Epoch 5 - Valid Loss: 20.9136
Sub-losses: recon_loss: 20.9136
Epoch 10 - Train Loss: 20.5687
Sub-losses: recon_loss: 20.5687
Epoch 10 - Valid Loss: 16.4208
Sub-losses: recon_loss: 16.4208
Epoch 15 - Train Loss: 17.1140
Sub-losses: recon_loss: 17.1140
Epoch 15 - Valid Loss: 12.7210
Sub-losses: recon_loss: 12.7210
Epoch 20 - Train Loss: 15.5596
Sub-losses: recon_loss: 15.5596
Epoch 20 - Valid Loss: 9.8392
Sub-losses: recon_loss: 9.8392
Epoch 25 - Train Loss: 14.3893
Sub-losses: recon_loss: 14.3893
Epoch 25 - Valid Loss: 8.7181
Sub-losses: recon_loss: 8.7181
Epoch 27 - Train Loss: 14.2813
Sub-losses: recon_loss: 14.2813
Epoch 27 - Valid Loss: 9.0706
Sub-losses: recon_loss: 9.0706
Found 120 common samples for the stacked autoencode
... [output truncated]
```

## 6) Saving, Loading and Reusing[¶](#6-saving-loading-and-reusing)

This works for `Stackix` as for any other models, by using the `save` and `load` methods.

In \[9\]:

Copied!

```
import os
import glob

outpath = os.path.join("tutorial_res", "stackix")
stackix.save(file_path=outpath, save_all=False)

folder = os.path.dirname(outpath)
pkl_files = glob.glob(os.path.join(folder, "*.pkl"))
model_files = glob.glob(os.path.join(folder, "*.pth"))

print("PKL files:", pkl_files)
print("Model files:", model_files)

# The load functionality will automatically rebuild the pipeline object
# from the saved files.
stackix_loaded = acx.Stackix.load(outpath)
```

import os import glob outpath = os.path.join("tutorial_res", "stackix") stackix.save(file_path=outpath, save_all=False) folder = os.path.dirname(outpath) pkl_files = glob.glob(os.path.join(folder, "\*.pkl")) model_files = glob.glob(os.path.join(folder, "\*.pth")) print("PKL files:", pkl_files) print("Model files:", model_files)

# The load functionality will automatically rebuild the pipeline object

# from the saved files.

stackix_loaded = acx.Stackix.load(outpath)

```
Preprocessor saved successfully.
saving memory efficient
Pipeline object saved successfully.
PKL files: ['tutorial_res/van_preprocessor.pkl', 'tutorial_res/stackix_preprocessor.pkl']
Model files: ['tutorial_res/stackix_model.pth', 'tutorial_res/van_model.pth']
Attempting to load a pipeline from tutorial_res/stackix...
Pipeline object loaded successfully. Actual type: Stackix
Preprocessor loaded successfully.
```

In \[10\]:

Copied!

```
stackix_loaded.predict(data=tcga_multi_bulk)
```

stackix_loaded.predict(data=tcga_multi_bulk)

```
Reproducibility settings for device auto are not implemented or necessary i.e. for cpu.
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
Found 600 common samples for the stacked autoencoder.
<autoencodix.data._numeric_dataset.NumericDataset object at 0x7890f077c040>
Successfully created annotated latent space object (adata_latent).
```

Out\[10\]:

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
new_datasets: DatasetContainer(train=<autoencodix.data._multimodal_dataset.MultiModalDataset object at 0x7890f077c550>, valid=None, test=<autoencodix.data._multimodal_dataset
... [output truncated]
```

In \[11\]:

Copied!

```
stackix_loaded.show_result(params=["CANCER_TYPE"])
```

stackix_loaded.show_result(params=["CANCER_TYPE"])

```
Creating plots ...
Absolute loss plot not found in the plots dictionary
This happens, when you did not run visualize() or if you saved and loaded the model with `save_all=False`
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/utils/_result.py:348: UserWarning: Could not retrieve latent representations for epoch 99 and split 'train'. Returning empty DataFrame. This may be due to missing data in the Result object or incorrect keys.


  warnings.warn(
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/utils/_result.py:348: UserWarning: Could not retrieve latent representations for epoch 99 and split 'valid'. Returning empty DataFrame. This may be due to missing data in the Result object or incorrect keys.


  warnings.warn(
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/utils/_result.py:348: UserWarning: Could not retrieve latent representations for epoch 99 and split 'train'. Returning empty DataFrame. This may be due to missing data in the Result object or incorrect keys.


  warnings.warn(
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/utils/_result.py:348: UserWarning: Could not retrieve latent representations for epoch 99 and split 'valid'. Returning empty DataFrame. This may be due to missing data in the Result object or incorrect keys.


  warnings.warn(
```

#### Generate New Data[¶](#generate-new-data)

For a variational autoencoder, the generate or sample_latent_space step draws new latent vectors from the model’s learned latent distribution.

**Latent Sampling:** The model first aggregates the posterior over all encoded latent vectors in the chosen split and epoch by computing the mean (global_mu) and log-variance (global_logvar). It then samples new latent points from a diagonal Gaussian defined by these aggregate statistics, using the reparameterization trick to inject Gaussian noise.

**Number of Samples (n_samples):** Users can specify how many latent points to generate. The method expands the aggregated mean and log-variance to match the requested number of samples before sampling.

**Custom Latent Prior:** Optionally, a custom latent_prior can be provided (a tensor or NumPy array with shape (n_samples, latent_dim)), which will be used directly instead of the aggregated posterior. This is basically the `decode` step.

In \[12\]:

Copied!

```
generated_reconstructions = stackix_loaded.generate(n_samples=5)
print(generated_reconstructions)
```

generated_reconstructions = stackix_loaded.generate(n_samples=5) print(generated_reconstructions)

```
Reproducibility settings for device auto are not implemented or necessary i.e. for cpu.
tensor([[3.4581, 0.9484, 4.1967, 3.9275, 1.9968, 2.9378, 4.8404, 0.7652, 1.7989,
         3.4892, 4.3203, 2.6945, 4.6708, 2.5322, 2.8759, 2.5423, 2.4730, 1.9494,
         2.3085, 3.2754, 1.2611, 1.5800, 1.2624, 0.8034, 2.3158, 1.8371, 1.3834,
         1.5676, 4.7108, 3.6217, 3.6610, 1.9044],
        [2.1054, 0.3580, 2.1148, 2.7985, 2.3127, 3.2155, 4.1482, 0.2411, 1.6331,
         3.5144, 2.3068, 1.7943, 4.4628, 2.9603, 1.4355, 3.1935, 4.0196, 0.9700,
         1.3288, 4.3767, 1.0587, 2.1889, 1.4707, 1.4002, 2.4744, 1.8154, 0.7167,
         1.8016, 6.1686, 3.6641, 4.3081, 1.0681],
        [2.6230, 0.8783, 2.3425, 2.3219, 0.7897, 1.7055, 2.0328, 0.6214, 1.6282,
         1.3181, 2.5319, 1.6533, 3.8746, 2.2
... [output truncated]
```
