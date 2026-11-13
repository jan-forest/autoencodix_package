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
```

```
You are using a CUDA device ('NVIDIA GeForce RTX 5060 Ti') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
```

```
Training each modality model...
Training modality: rna
Training modality: rna
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 11.0494 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 20 - Train Loss: 861.5075
Sub-losses: recon_loss: 861.5067, var_loss: 0.0008, anneal_factor: 0.0020, effective_beta_factor: 0.0000
Epoch 20 - Valid Loss: 809.1370
Sub-losses: recon_loss: 809.1366, var_loss: 0.0003, anneal_factor: 0.0020, effective_beta_factor: 0.0000
```

```
Epoch 40 - Train Loss: 798.5528
Sub-losses: recon_loss: 798.4950, var_loss: 0.0578, anneal_factor: 0.0998, effective_beta_factor: 0.0010
Epoch 40 - Valid Loss: 741.4023
Sub-losses: recon_loss: 741.3617, var_loss: 0.0406, anneal_factor: 0.0998, effective_beta_factor: 0.0010
```

```
Epoch 60 - Train Loss: 766.8894
Sub-losses: recon_loss: 766.2031, var_loss: 0.6862, anneal_factor: 0.8581, effective_beta_factor: 0.0086
Epoch 60 - Valid Loss: 719.2603
Sub-losses: recon_loss: 718.7603, var_loss: 0.5000, anneal_factor: 0.8581, effective_beta_factor: 0.0086
```

```
Epoch 80 - Train Loss: 744.9275
Sub-losses: recon_loss: 743.8341, var_loss: 1.0934, anneal_factor: 0.9970, effective_beta_factor: 0.0100
Epoch 80 - Valid Loss: 701.3495
Sub-losses: recon_loss: 700.5113, var_loss: 0.8382, anneal_factor: 0.9970, effective_beta_factor: 0.0100
```

```
Epoch 100 - Train Loss: 737.5424
Sub-losses: recon_loss: 736.1616, var_loss: 1.3807, anneal_factor: 0.9999, effective_beta_factor: 0.0100
Epoch 100 - Valid Loss: 701.1273
Sub-losses: recon_loss: 700.1542, var_loss: 0.9730, anneal_factor: 0.9999, effective_beta_factor: 0.0100
Training modality: meth
Training modality: meth
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 9.7115 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 20 - Train Loss: 708.7072
Sub-losses: recon_loss: 708.7065, var_loss: 0.0008, anneal_factor: 0.0020, effective_beta_factor: 0.0000
Epoch 20 - Valid Loss: 827.8635
Sub-losses: recon_loss: 827.8621, var_loss: 0.0014, anneal_factor: 0.0020, effective_beta_factor: 0.0000
```

```
Epoch 40 - Train Loss: 615.2633
Sub-losses: recon_loss: 615.2055, var_loss: 0.0578, anneal_factor: 0.0998, effective_beta_factor: 0.0010
Epoch 40 - Valid Loss: 612.6787
Sub-losses: recon_loss: 612.6109, var_loss: 0.0677, anneal_factor: 0.0998, effective_beta_factor: 0.0010
```

```
Epoch 60 - Train Loss: 588.3950
Sub-losses: recon_loss: 587.6555, var_loss: 0.7395, anneal_factor: 0.8581, effective_beta_factor: 0.0086
Epoch 60 - Valid Loss: 587.5274
Sub-losses: recon_loss: 586.7058, var_loss: 0.8216, anneal_factor: 0.8581, effective_beta_factor: 0.0086
```

```
Epoch 80 - Train Loss: 577.5370
Sub-losses: recon_loss: 576.3876, var_loss: 1.1494, anneal_factor: 0.9970, effective_beta_factor: 0.0100
Epoch 80 - Valid Loss: 572.7204
Sub-losses: recon_loss: 571.5363, var_loss: 1.1841, anneal_factor: 0.9970, effective_beta_factor: 0.0100
```

```
Epoch 100 - Train Loss: 557.4491
Sub-losses: recon_loss: 555.9831, var_loss: 1.4660, anneal_factor: 0.9999, effective_beta_factor: 0.0100
Epoch 100 - Valid Loss: 543.0278
Sub-losses: recon_loss: 541.2497, var_loss: 1.7781, anneal_factor: 0.9999, effective_beta_factor: 0.0100
Found 420 common samples for the stacked autoencoder.
Found 60 common samples for the stacked autoencoder.
finished training each modality model
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 86.0775 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 20 - Train Loss: 366.4243
Sub-losses: recon_loss: 366.4241, var_loss: 0.0002, anneal_factor: 0.0020, effective_beta_factor: 0.0000
Epoch 20 - Valid Loss: 337.7049
Sub-losses: recon_loss: 337.7048, var_loss: 0.0001, anneal_factor: 0.0020, effective_beta_factor: 0.0000
```

```
Epoch 40 - Train Loss: 251.6875
Sub-losses: recon_loss: 251.6583, var_loss: 0.0292, anneal_factor: 0.0998, effective_beta_factor: 0.0010
Epoch 40 - Valid Loss: 218.8828
Sub-losses: recon_loss: 218.8616, var_loss: 0.0212, anneal_factor: 0.0998, effective_beta_factor: 0.0010
```

```
Epoch 60 - Train Loss: 217.7289
Sub-losses: recon_loss: 217.2342, var_loss: 0.4947, anneal_factor: 0.8581, effective_beta_factor: 0.0086
Epoch 60 - Valid Loss: 182.7816
Sub-losses: recon_loss: 182.4353, var_loss: 0.3463, anneal_factor: 0.8581, effective_beta_factor: 0.0086
```

```
Epoch 80 - Train Loss: 192.9095
Sub-losses: recon_loss: 192.0899, var_loss: 0.8196, anneal_factor: 0.9970, effective_beta_factor: 0.0100
Epoch 80 - Valid Loss: 167.0142
Sub-losses: recon_loss: 166.4467, var_loss: 0.5675, anneal_factor: 0.9970, effective_beta_factor: 0.0100
```

```
Epoch 100 - Train Loss: 195.3627
Sub-losses: recon_loss: 194.3243, var_loss: 1.0384, anneal_factor: 0.9999, effective_beta_factor: 0.0100
Epoch 100 - Valid Loss: 152.8711
Sub-losses: recon_loss: 152.0966, var_loss: 0.7745, anneal_factor: 0.9999, effective_beta_factor: 0.0100
Processed 32 / 120 samples
Processed 64 / 120 samples
Processed 96 / 120 samples
Processed 120 / 120 samples
Processed 32 / 120 samples
Processed 64 / 120 samples
Processed 96 / 120 samples
Processed 120 / 120 samples
Found 120 common samples for the stacked autoencoder.
<autoencodix.data._numeric_dataset.NumericDataset object at 0x73fd70183da0>
Processed 32 / 120 samples
Processed 64 / 120 samples
Processed 96 / 120 samples
Processed 120 / 120 samples
Successfully created annotated latent space object (adata_latent).
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
```

```
Training each modality model...
Training modality: rna
Training modality: rna
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 11.0010 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 3 - Train Loss: 43112.5306
Sub-losses: recon_loss: 18463.2616, var_loss: 24649.2694, anneal_factor: 0.9656, effective_beta_factor: 0.0966
Epoch 3 - Valid Loss: 156541.3513
Sub-losses: recon_loss: 156189.7737, var_loss: 351.5800, anneal_factor: 0.9656, effective_beta_factor: 0.0966
Training modality: meth
Training modality: meth
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 10.8091 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 3 - Train Loss: 7730.2220
Sub-losses: recon_loss: 7599.5141, var_loss: 130.7078, anneal_factor: 0.9656, effective_beta_factor: 0.0966
Epoch 3 - Valid Loss: 9269.0419
Sub-losses: recon_loss: 9249.8117, var_loss: 19.2304, anneal_factor: 0.9656, effective_beta_factor: 0.0966
Found 406 common samples for the stacked autoencoder.
Found 58 common samples for the stacked autoencoder.
finished training each modality model
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 264.4442 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 3 - Train Loss: 382916.6880
Sub-losses: recon_loss: 382916.2968, var_loss: 0.3747, anneal_factor: 0.9656, effective_beta_factor: 0.0966
Epoch 3 - Valid Loss: 7758.7018
Sub-losses: recon_loss: 7758.6870, var_loss: 0.0150, anneal_factor: 0.9656, effective_beta_factor: 0.0966
Processed 32 / 116 samples
Processed 64 / 116 samples
Processed 96 / 116 samples
Processed 116 / 116 samples
Processed 32 / 120 samples
Processed 64 / 120 samples
Processed 96 / 120 samples
Processed 120 / 120 samples
Found 116 common samples for the stacked autoencoder.
<autoencodix.data._numeric_dataset.NumericDataset object at 0x73fd70448380>
Processed 32 / 116 samples
Processed 64 / 116 samples
Processed 96 / 116 samples
Processed 116 / 116 samples
Successfully created annotated latent space object (adata_late
... [output truncated]
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
class_means: TrainingDynamics object
datasets: DatasetContainer(train=None, valid=None, test=None)
new_datasets: DatasetContainer(train=None, valid=None, test=None)
adata_latent: AnnData object with n_obs × n_vars = 0 × 0
final_reconstr
... [output truncated]
```

Out\[5\]:

```
array([[ 0.60902447,  1.8953304 ,  1.6210244 , ..., -0.77807117,
         0.13796484,  0.55225635],
       [ 0.7116955 ,  2.1084826 ,  4.525612  , ...,  0.42529175,
         1.7997122 ,  2.3922684 ],
       [-1.4662203 , -0.16053778,  1.4779546 , ...,  1.6357734 ,
         0.8535643 , -0.2062298 ],
       ...,
       [13.1728945 ,  9.191634  ,  9.9418    , ..., -0.15876473,
         5.6592045 ,  0.9905605 ],
       [ 1.1091481 ,  2.0996447 , -0.5123983 , ...,  0.11502871,
         5.8076015 , -0.87619036],
       [ 4.4038086 ,  3.2189286 ,  8.247406  , ...,  0.96144074,
         9.293939  ,  2.3476388 ]], shape=(120, 16), dtype=float32)
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




TCGA-09-1670-01
1.678541
-0.756772
1.766413
5.494479
5.885232
2.213073
-0.659003
4.496250
-1.154735
2.784639
1.886887
0.316816
2.120650
-1.987382
0.855010
2.937171


TCGA-09-1674-01
1.173382
0.408939
1.389013
4.767611
5.374843
2.044036
0.479978
2.096513
0.776345
3.812770
-0.617784
0.124010
1.663672
0.685736
-0.884433
2.594606


TCGA-24-0975-01
-0.875569
1.486634
2.251036
1.607407
4.139259
1.688630
0.304271
2.961607
0.838249
4.214644
1.642579
0.805008
3.197985
2.320084
0.302546
0.986147


TCGA-24-1417-01
-0.575101
1.011320
1.628110
3.975448
5.451651
3.162087
1.878302
3.828600
1.160349
4.455536
0.647374
-0.609331
2.483908
0.160349
-2
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
```

```
Training each modality model...
Training modality: rna
Training modality: rna
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 19.9810 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 5 - Train Loss: 849.0952
Sub-losses: recon_loss: 849.0952
Epoch 5 - Valid Loss: 805.0664
Sub-losses: recon_loss: 805.0664
```

```
Epoch 10 - Train Loss: 796.1807
Sub-losses: recon_loss: 796.1807
Epoch 10 - Valid Loss: 770.1940
Sub-losses: recon_loss: 770.1940
```

```
Epoch 15 - Train Loss: 772.3496
Sub-losses: recon_loss: 772.3496
Epoch 15 - Valid Loss: 747.3774
Sub-losses: recon_loss: 747.3774
```

```
Epoch 20 - Train Loss: 750.7159
Sub-losses: recon_loss: 750.7159
Epoch 20 - Valid Loss: 744.3476
Sub-losses: recon_loss: 744.3476
```

```
Epoch 25 - Train Loss: 730.3083
Sub-losses: recon_loss: 730.3083
Epoch 25 - Valid Loss: 737.4077
Sub-losses: recon_loss: 737.4077
```

```
Epoch 27 - Train Loss: 733.9101
Sub-losses: recon_loss: 733.9101
Epoch 27 - Valid Loss: 735.1336
Sub-losses: recon_loss: 735.1336
Training modality: meth
Training modality: meth
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 16.7733 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 5 - Train Loss: 667.1786
Sub-losses: recon_loss: 667.1786
Epoch 5 - Valid Loss: 561.2035
Sub-losses: recon_loss: 561.2035
```

```
Epoch 10 - Train Loss: 592.9923
Sub-losses: recon_loss: 592.9923
Epoch 10 - Valid Loss: 519.3777
Sub-losses: recon_loss: 519.3777
```

```
Epoch 15 - Train Loss: 581.2438
Sub-losses: recon_loss: 581.2438
Epoch 15 - Valid Loss: 504.0353
Sub-losses: recon_loss: 504.0353
```

```
Epoch 20 - Train Loss: 571.8997
Sub-losses: recon_loss: 571.8997
Epoch 20 - Valid Loss: 500.9520
Sub-losses: recon_loss: 500.9520
```

```
Epoch 25 - Train Loss: 543.3780
Sub-losses: recon_loss: 543.3780
Epoch 25 - Valid Loss: 493.9917
Sub-losses: recon_loss: 493.9917
```

```
Epoch 27 - Train Loss: 547.9194
Sub-losses: recon_loss: 547.9194
Epoch 27 - Valid Loss: 487.6528
Sub-losses: recon_loss: 487.6528
Found 420 common samples for the stacked autoencoder.
Found 60 common samples for the stacked autoencoder.
finished training each modality model
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 42.4750 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 5 - Train Loss: 28.9800
Sub-losses: recon_loss: 28.9800
Epoch 5 - Valid Loss: 26.2177
Sub-losses: recon_loss: 26.2177
```

```
Epoch 10 - Train Loss: 23.7372
Sub-losses: recon_loss: 23.7372
Epoch 10 - Valid Loss: 21.4609
Sub-losses: recon_loss: 21.4609
```

```
Epoch 15 - Train Loss: 20.6745
Sub-losses: recon_loss: 20.6745
Epoch 15 - Valid Loss: 18.5181
Sub-losses: recon_loss: 18.5181
```

```
Epoch 20 - Train Loss: 19.5151
Sub-losses: recon_loss: 19.5151
Epoch 20 - Valid Loss: 16.0991
Sub-losses: recon_loss: 16.0991
```

```
Epoch 25 - Train Loss: 17.8456
Sub-losses: recon_loss: 17.8456
Epoch 25 - Valid Loss: 13.6030
Sub-losses: recon_loss: 13.6030
```

```
Epoch 27 - Train Loss: 17.1849
Sub-losses: recon_loss: 17.1849
Epoch 27 - Valid Loss: 12.8338
Sub-losses: recon_loss: 12.8338
Processed 32 / 120 samples
Processed 64 / 120 samples
Processed 96 / 120 samples
Processed 120 / 120 samples
Processed 32 / 120 samples
Processed 64 / 120 samples
Processed 96 / 120 samples
Processed 120 / 120 samples
Found 120 common samples for the stacked autoencoder.
<autoencodix.data._numeric_dataset.NumericDataset object at 0x73fad037f950>
Processed 32 / 120 samples
```

```
Processed 64 / 120 samples
Processed 96 / 120 samples
Processed 120 / 120 samples
Successfully created annotated latent space object (adata_latent).
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
```

```
saving memory efficient
```

```
Pipeline object saved successfully.
PKL files: ['tutorial_res/supervisix_preprocessor.pkl', 'tutorial_res/van_preprocessor.pkl', 'tutorial_res/stackix_preprocessor.pkl']
Model files: ['tutorial_res/stackix_model.pth', 'tutorial_res/supervisix_model.pth', 'tutorial_res/van_model.pth']
Attempting to load a pipeline from tutorial_res/stackix...
```

```
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
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
```

```
Processed 32 / 600 samples
Processed 64 / 600 samples
Processed 96 / 600 samples
Processed 128 / 600 samples
Processed 160 / 600 samples
Processed 192 / 600 samples
Processed 224 / 600 samples
Processed 256 / 600 samples
Processed 288 / 600 samples
Processed 320 / 600 samples
Processed 352 / 600 samples
Processed 384 / 600 samples
Processed 416 / 600 samples
Processed 448 / 600 samples
Processed 480 / 600 samples
Processed 512 / 600 samples
Processed 544 / 600 samples
Processed 576 / 600 samples
Processed 600 / 600 samples
Processed 32 / 600 samples
Processed 64 / 600 samples
Processed 96 / 600 samples
Processed 128 / 600 samples
Processed 160 / 600 samples
Processed 192 / 600 samples
Processed 224 / 600 samples
Processed 256 / 600 samples
Processed 288 / 600 samples
Processed 320 / 600 s
... [output truncated]
```

```
Processed 128 / 600 samples
Processed 160 / 600 samples
Processed 192 / 600 samples
Processed 224 / 600 samples
Processed 256 / 600 samples
Processed 288 / 600 samples
Processed 320 / 600 samples
Processed 352 / 600 samples
Processed 384 / 600 samples
Processed 416 / 600 samples
Processed 448 / 600 samples
Processed 480 / 600 samples
Processed 512 / 600 samples
Processed 544 / 600 samples
Processed 576 / 600 samples
Processed 600 / 600 samples
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
class_means: TrainingDynamics object
datasets: DatasetContainer(train=<autoencodix.data._multimodal_dataset.MultiModalDataset object at 0x73fab8636540>, valid=<autoencodix.data._multimodal_dataset.MultiModalDataset object at 0x73fab8b39
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
tensor([[2.0793, 1.7237, 1.4916, 2.8190, 1.3942, 1.6960, 1.5467, 1.5444, 1.7116,
         1.1534, 1.2416, 1.0904, 4.1049, 2.8662, 2.8259, 1.3614, 4.8762, 4.0383,
         4.5663, 3.9517, 2.0408, 1.4265, 2.7195, 2.2117, 3.4810, 1.9435, 2.3055,
         0.6842, 4.9059, 2.4917, 3.9826, 3.0114],
        [2.7321, 2.1541, 1.9380, 3.8838, 1.9447, 1.6806, 2.1549, 2.1850, 0.8857,
         1.6568, 1.8522, 1.6868, 3.6741, 3.3703, 1.8000, 1.4678, 6.1978, 4.8756,
         4.6193, 5.1619, 1.7323, 1.9495, 2.5986, 2.9031, 2.1829, 1.3280, 1.4037,
         0.1897, 6.4398, 3.2108, 5.1410, 2.8926],
        [2.0714, 2.4733, 2.6246, 3.1415, 1.2934, 1.8166, 2.0684, 1.5501, 0.8833,
         2.0830, 2.2447, 1.6188, 2.7827, 2.2695, 1.6414, 2.1501, 5.3980, 4.6438,
         3.9895, 4.9139, 2.2990, 1.6801, 2.5718, 2.
... [output truncated]
```
