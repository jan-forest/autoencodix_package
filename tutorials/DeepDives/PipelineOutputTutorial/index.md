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

# How to Access Pipeline Results - Deep Dive[¶](#how-to-access-pipeline-results-deep-dive)

Each step in the pipeline writes its results in the `Result` object of the pipeline instance (Vanillix, Varix, etc.).\
In this tutorial, we explore how to access and interpret the results.

The attributes of the `Result` object are mostly instances of a `TrainingDynamics` class.\
This class provides a standardized interface for accessing results from different splits and epochs.

**IMPORTANT**

> Epoch-specific `TrainingDynamics`—such as losses or intermediate latent spaces—are not stored every epoch by default.\
> You need to set the `checkpoint_interval` parameter in the config according to your needs.

## What You Will Learn[¶](#what-you-will-learn)

We go in depth into:

- The `TrainingDynamics` API

  - latent spaces
  - losses
  - reconstructions
  - sample_ids

- Nested `TrainingDynamics` like `sub_losses`

- Non-`TrainingDynamics` result attributes, such as:

  - datasets
  - new_datasets
  - model
  - adata_latent
  - final_reconstruction
  - embedding_evaluation

- Special methods to obtain pandas DataFrames

  - get latent space as a DataFrame with `sample_ids`
  - get reconstruction as a DataFrame with `sample_ids`

## 1) Filling the Result Object[¶](#1-filling-the-result-object)

Before we can investigate the result object, we first need to create results.\
Therefore, we run two pipelines: `XModalix` and `Varix`.

### 1.1 The Datasets[¶](#11-the-datasets)

For the `Varix` example, we use a mock single-cell dataset as a `MuData` object inside our custom `DataPackage` class.\
For our `XModalix` example, we use the same dataset as in the [`XModalix.ipynb`](https://jan-forest.github.io/autoencodix_package/tutorials/DeepDives/PipelineTutorials/XModalix.ipynb) tutorial.

As a showcase for data modality translation with `XModalix`, we use cancer gene expression from TCGA in combination with handwritten digits from the [MNIST dataset](https://keras.io/api/datasets/mnist/).\
Our goal is to translate the gene expression signature of five selected cancer subtypes to images of digits, where each cancer subtype class is assigned a digit between 0-4.

In practice, these images could be histopathological images or any other data modality.\
Before showing data preparation and `XModalix` training, here is some background on the basic idea of a cross-modal VAE as proposed by [Yang & Uhler](https://arxiv.org/abs/1902.03515).

### ❗❗ Requirements: Getting Tutorial Data ❗❗[¶](#requirements-getting-tutorial-data)

The data for this tutorial is hosted on Hugging Face Hub ([autoencodix/tcga](https://huggingface.co/datasets/autoencodix/tcga)) and is downloaded automatically in the cell below on first run.

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

In \[3\]:

Copied!

```
# %%capture
from autoencodix.utils.example_data import EXAMPLE_MULTI_SC
from autoencodix.configs.varix_config import VarixConfig
from autoencodix.configs.default_config import DataCase, DataInfo, DataConfig
from autoencodix.configs.xmodalix_config import XModalixConfig
import autoencodix as acx

varix_config = VarixConfig(
    learning_rate=0.001,
    epochs=33,
    checkpoint_interval=1,
    default_vae_loss="kl",  # kl or mmd possible
    data_case=DataCase.MULTI_SINGLE_CELL,
)
varix = acx.Varix(data=EXAMPLE_MULTI_SC, config=varix_config)
result = varix.run()

# XModalix
# ---------------------------------------------------------------------
# Data is hosted on Hugging Face Hub and downloaded (and locally cached)
# automatically, then placed under the paths used below
# ---------------------------------------------------------------------
import shutil
import zipfile
import pandas as pd
from huggingface_hub import hf_hub_download

HF_REPO_ID = "autoencodix/tcga"
xmod_data_root = "data/XModalix-Tut-data"
os.makedirs(xmod_data_root, exist_ok=True)

rna_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="rna.parquet")
shutil.copyfile(rna_path, os.path.join(xmod_data_root, "combined_rnaseq_formatted.parquet"))

clin_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="clinical.parquet")
mapping_path = hf_hub_download(
    repo_id=HF_REPO_ID, repo_type="dataset", filename="xmodalix/tcga_image_mappings.txt"
)
# The base clinical file has no image column; the XModalix image mapping is a separate,
# per-architecture extension that we join in here (left join: not every sample has an image).
clin_df = pd.read_parquet(clin_path)
mapping_df = pd.read_csv(mapping_path, sep="\t", index_col="sample_ids")
clin_df["img_paths"] = mapping_df["img_paths"]
clin_df.to_parquet(os.path.join(xmod_data_root, "combined_clin_formatted.parquet"))

images_zip = hf_hub_download(
    repo_id=HF_REPO_ID, repo_type="dataset", filename="xmodalix/tcga_fake_images.zip"
)
images_extract_dir = os.path.join(xmod_data_root, "images")
if not os.path.isdir(os.path.join(images_extract_dir, "tcga_fake")):
    with zipfile.ZipFile(images_zip) as zf:
        zf.extractall(images_extract_dir)

clin_file = os.path.join("data/XModalix-Tut-data/combined_clin_formatted.parquet")
rna_file = os.path.join("data/XModalix-Tut-data/combined_rnaseq_formatted.parquet")
img_root = os.path.join("data/XModalix-Tut-data/images/tcga_fake")

xmodalix_config = XModalixConfig(
    checkpoint_interval=5,
    class_param="CANCER_TYPE_ACRONYM",
    epochs=10,
    data_case=DataCase.IMG_TO_BULK,
    data_config=DataConfig(
        data_info={
            "img": DataInfo(
                file_path=img_root,
                data_type="IMG",
                scaling="MINMAX",
                translate_direction="to",
            ),
            "rna": DataInfo(
                file_path=rna_file,
                data_type="NUMERIC",
                scaling="MINMAX",
                translate_direction="from",
            ),
            "anno": DataInfo(file_path=clin_file, data_type="ANNOTATION", sep="\t"),
        },
    ),
)

xmodalix = acx.XModalix(config=xmodalix_config)
xmodalix_result = xmodalix.run()
```

# %%capture

from autoencodix.utils.example_data import EXAMPLE_MULTI_SC from autoencodix.configs.varix_config import VarixConfig from autoencodix.configs.default_config import DataCase, DataInfo, DataConfig from autoencodix.configs.xmodalix_config import XModalixConfig import autoencodix as acx varix_config = VarixConfig( learning_rate=0.001, epochs=33, checkpoint_interval=1, default_vae_loss="kl", # kl or mmd possible data_case=DataCase.MULTI_SINGLE_CELL, ) varix = acx.Varix(data=EXAMPLE_MULTI_SC, config=varix_config) result = varix.run()

# XModalix

# ---------------------------------------------------------------------

# Data is hosted on Hugging Face Hub and downloaded (and locally cached)

# automatically, then placed under the paths used below

# ---------------------------------------------------------------------

import shutil import zipfile import pandas as pd from huggingface_hub import hf_hub_download HF_REPO_ID = "autoencodix/tcga" xmod_data_root = "data/XModalix-Tut-data" os.makedirs(xmod_data_root, exist_ok=True) rna_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="rna.parquet") shutil.copyfile(rna_path, os.path.join(xmod_data_root, "combined_rnaseq_formatted.parquet")) clin_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="clinical.parquet") mapping_path = hf_hub_download( repo_id=HF_REPO_ID, repo_type="dataset", filename="xmodalix/tcga_image_mappings.txt" )

# The base clinical file has no image column; the XModalix image mapping is a separate,

# per-architecture extension that we join in here (left join: not every sample has an image).

clin_df = pd.read_parquet(clin_path) mapping_df = pd.read_csv(mapping_path, sep="\\t", index_col="sample_ids") clin_df["img_paths"] = mapping_df["img_paths"] clin_df.to_parquet(os.path.join(xmod_data_root, "combined_clin_formatted.parquet")) images_zip = hf_hub_download( repo_id=HF_REPO_ID, repo_type="dataset", filename="xmodalix/tcga_fake_images.zip" ) images_extract_dir = os.path.join(xmod_data_root, "images") if not os.path.isdir(os.path.join(images_extract_dir, "tcga_fake")): with zipfile.ZipFile(images_zip) as zf: zf.extractall(images_extract_dir) clin_file = os.path.join("data/XModalix-Tut-data/combined_clin_formatted.parquet") rna_file = os.path.join("data/XModalix-Tut-data/combined_rnaseq_formatted.parquet") img_root = os.path.join("data/XModalix-Tut-data/images/tcga_fake") xmodalix_config = XModalixConfig( checkpoint_interval=5, class_param="CANCER_TYPE_ACRONYM", epochs=10, data_case=DataCase.IMG_TO_BULK, data_config=DataConfig( data_info={ "img": DataInfo( file_path=img_root, data_type="IMG", scaling="MINMAX", translate_direction="to", ), "rna": DataInfo( file_path=rna_file, data_type="NUMERIC", scaling="MINMAX", translate_direction="from", ), "anno": DataInfo(file_path=clin_file, data_type="ANNOTATION", sep="\\t"), }, ), ) xmodalix = acx.XModalix(config=xmodalix_config) xmodalix_result = xmodalix.run()

```
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
mudata: View of MuData object with n_obs × n_vars = 1000 × 700
  2 modalities
    rna:	1000 x 500
      obs:	'cell_type', 'batch', 'donor', 'cell_cycle'
    protein:	1000 x 200
      obs:	'cell_type', 'batch', 'donor', 'cell_cycle'
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
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 12.0150 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 1 - Train Loss: 789.5454
Sub-losses: recon_loss: 789.5454, var_loss: 0.0000, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 1 - Valid Loss: 861.7886
Sub-losses: recon_loss: 861.7884, var_loss: 0.0003, anneal_factor: 0.0000, effective_beta_factor: 0.0000
```

```
Epoch 2 - Train Loss: 714.1906
Sub-losses: recon_loss: 714.1905, var_loss: 0.0001, anneal_factor: 0.0001, effective_beta_factor: 0.0000
Epoch 2 - Valid Loss: 726.2190
Sub-losses: recon_loss: 726.2188, var_loss: 0.0002, anneal_factor: 0.0001, effective_beta_factor: 0.0000
```

```
Epoch 3 - Train Loss: 654.3003
Sub-losses: recon_loss: 654.3000, var_loss: 0.0003, anneal_factor: 0.0002, effective_beta_factor: 0.0000
Epoch 3 - Valid Loss: 661.9915
Sub-losses: recon_loss: 661.9913, var_loss: 0.0002, anneal_factor: 0.0002, effective_beta_factor: 0.0000
```

```
Epoch 4 - Train Loss: 603.1925
Sub-losses: recon_loss: 603.1918, var_loss: 0.0007, anneal_factor: 0.0003, effective_beta_factor: 0.0000
Epoch 4 - Valid Loss: 628.7880
Sub-losses: recon_loss: 628.7874, var_loss: 0.0006, anneal_factor: 0.0003, effective_beta_factor: 0.0000
```

```
Epoch 5 - Train Loss: 565.8273
Sub-losses: recon_loss: 565.8261, var_loss: 0.0013, anneal_factor: 0.0005, effective_beta_factor: 0.0001
Epoch 5 - Valid Loss: 580.7370
Sub-losses: recon_loss: 580.7361, var_loss: 0.0009, anneal_factor: 0.0005, effective_beta_factor: 0.0001
```

```
Epoch 6 - Train Loss: 535.0951
Sub-losses: recon_loss: 535.0925, var_loss: 0.0026, anneal_factor: 0.0009, effective_beta_factor: 0.0001
Epoch 6 - Valid Loss: 580.8360
Sub-losses: recon_loss: 580.8344, var_loss: 0.0016, anneal_factor: 0.0009, effective_beta_factor: 0.0001
```

```
Epoch 7 - Train Loss: 515.1641
Sub-losses: recon_loss: 515.1593, var_loss: 0.0047, anneal_factor: 0.0017, effective_beta_factor: 0.0002
Epoch 7 - Valid Loss: 544.3053
Sub-losses: recon_loss: 544.3018, var_loss: 0.0035, anneal_factor: 0.0017, effective_beta_factor: 0.0002
```

```
Epoch 8 - Train Loss: 493.2187
Sub-losses: recon_loss: 493.1982, var_loss: 0.0205, anneal_factor: 0.0031, effective_beta_factor: 0.0003
Epoch 8 - Valid Loss: 555.5275
Sub-losses: recon_loss: 555.5203, var_loss: 0.0072, anneal_factor: 0.0031, effective_beta_factor: 0.0003
```

```
Epoch 9 - Train Loss: 487.6101
Sub-losses: recon_loss: 487.5851, var_loss: 0.0251, anneal_factor: 0.0058, effective_beta_factor: 0.0006
Epoch 9 - Valid Loss: 535.1194
Sub-losses: recon_loss: 535.1047, var_loss: 0.0148, anneal_factor: 0.0058, effective_beta_factor: 0.0006
```

```
Epoch 10 - Train Loss: 475.9777
Sub-losses: recon_loss: 475.9373, var_loss: 0.0404, anneal_factor: 0.0105, effective_beta_factor: 0.0011
Epoch 10 - Valid Loss: 540.3726
Sub-losses: recon_loss: 540.3461, var_loss: 0.0264, anneal_factor: 0.0105, effective_beta_factor: 0.0011
```

```
Epoch 11 - Train Loss: 465.1706
Sub-losses: recon_loss: 465.0587, var_loss: 0.1119, anneal_factor: 0.0191, effective_beta_factor: 0.0019
Epoch 11 - Valid Loss: 535.0998
Sub-losses: recon_loss: 535.0498, var_loss: 0.0500, anneal_factor: 0.0191, effective_beta_factor: 0.0019
```

```
Epoch 12 - Train Loss: 461.6776
Sub-losses: recon_loss: 461.5501, var_loss: 0.1275, anneal_factor: 0.0344, effective_beta_factor: 0.0034
Epoch 12 - Valid Loss: 527.3585
Sub-losses: recon_loss: 527.2690, var_loss: 0.0895, anneal_factor: 0.0344, effective_beta_factor: 0.0034
```

```
Epoch 13 - Train Loss: 462.7907
Sub-losses: recon_loss: 462.5608, var_loss: 0.2298, anneal_factor: 0.0614, effective_beta_factor: 0.0061
Epoch 13 - Valid Loss: 509.7819
Sub-losses: recon_loss: 509.6216, var_loss: 0.1603, anneal_factor: 0.0614, effective_beta_factor: 0.0061
```

```
Epoch 14 - Train Loss: 453.6761
Sub-losses: recon_loss: 453.2983, var_loss: 0.3779, anneal_factor: 0.1071, effective_beta_factor: 0.0107
Epoch 14 - Valid Loss: 514.1650
Sub-losses: recon_loss: 513.8674, var_loss: 0.2976, anneal_factor: 0.1071, effective_beta_factor: 0.0107
```

```
Epoch 15 - Train Loss: 448.2364
Sub-losses: recon_loss: 447.5243, var_loss: 0.7121, anneal_factor: 0.1802, effective_beta_factor: 0.0180
Epoch 15 - Valid Loss: 506.4760
Sub-losses: recon_loss: 505.9687, var_loss: 0.5072, anneal_factor: 0.1802, effective_beta_factor: 0.0180
```

```
Epoch 16 - Train Loss: 450.8643
Sub-losses: recon_loss: 449.7722, var_loss: 1.0922, anneal_factor: 0.2872, effective_beta_factor: 0.0287
Epoch 16 - Valid Loss: 507.3997
Sub-losses: recon_loss: 506.4213, var_loss: 0.9784, anneal_factor: 0.2872, effective_beta_factor: 0.0287
```

```
Epoch 17 - Train Loss: 446.2062
Sub-losses: recon_loss: 444.6207, var_loss: 1.5854, anneal_factor: 0.4248, effective_beta_factor: 0.0425
Epoch 17 - Valid Loss: 505.6747
Sub-losses: recon_loss: 504.2951, var_loss: 1.3796, anneal_factor: 0.4248, effective_beta_factor: 0.0425
```

```
Epoch 18 - Train Loss: 445.6188
Sub-losses: recon_loss: 443.2989, var_loss: 2.3199, anneal_factor: 0.5752, effective_beta_factor: 0.0575
Epoch 18 - Valid Loss: 509.4240
Sub-losses: recon_loss: 507.6395, var_loss: 1.7845, anneal_factor: 0.5752, effective_beta_factor: 0.0575
```

```
Epoch 19 - Train Loss: 440.7485
Sub-losses: recon_loss: 438.1607, var_loss: 2.5878, anneal_factor: 0.7128, effective_beta_factor: 0.0713
Epoch 19 - Valid Loss: 513.0084
Sub-losses: recon_loss: 510.5956, var_loss: 2.4128, anneal_factor: 0.7128, effective_beta_factor: 0.0713
```

```
Epoch 20 - Train Loss: 444.3923
Sub-losses: recon_loss: 441.1604, var_loss: 3.2319, anneal_factor: 0.8198, effective_beta_factor: 0.0820
Epoch 20 - Valid Loss: 505.8253
Sub-losses: recon_loss: 503.4233, var_loss: 2.4019, anneal_factor: 0.8198, effective_beta_factor: 0.0820
```

```
Epoch 21 - Train Loss: 438.8480
Sub-losses: recon_loss: 435.6881, var_loss: 3.1599, anneal_factor: 0.8929, effective_beta_factor: 0.0893
Epoch 21 - Valid Loss: 510.4828
Sub-losses: recon_loss: 507.7820, var_loss: 2.7008, anneal_factor: 0.8929, effective_beta_factor: 0.0893
```

```
Epoch 22 - Train Loss: 440.5572
Sub-losses: recon_loss: 437.1135, var_loss: 3.4436, anneal_factor: 0.9386, effective_beta_factor: 0.0939
Epoch 22 - Valid Loss: 512.5383
Sub-losses: recon_loss: 509.4187, var_loss: 3.1197, anneal_factor: 0.9386, effective_beta_factor: 0.0939
```

```
Epoch 23 - Train Loss: 436.6567
Sub-losses: recon_loss: 433.0894, var_loss: 3.5673, anneal_factor: 0.9656, effective_beta_factor: 0.0966
Epoch 23 - Valid Loss: 507.8697
Sub-losses: recon_loss: 505.1926, var_loss: 2.6771, anneal_factor: 0.9656, effective_beta_factor: 0.0966
```

```
Epoch 24 - Train Loss: 440.1171
Sub-losses: recon_loss: 436.5504, var_loss: 3.5667, anneal_factor: 0.9809, effective_beta_factor: 0.0981
Epoch 24 - Valid Loss: 507.4889
Sub-losses: recon_loss: 504.7422, var_loss: 2.7468, anneal_factor: 0.9809, effective_beta_factor: 0.0981
```

```
Epoch 25 - Train Loss: 429.4857
Sub-losses: recon_loss: 425.7818, var_loss: 3.7039, anneal_factor: 0.9895, effective_beta_factor: 0.0989
Epoch 25 - Valid Loss: 504.5467
Sub-losses: recon_loss: 501.6651, var_loss: 2.8816, anneal_factor: 0.9895, effective_beta_factor: 0.0989
```

```
Epoch 26 - Train Loss: 435.2628
Sub-losses: recon_loss: 431.6024, var_loss: 3.6604, anneal_factor: 0.9942, effective_beta_factor: 0.0994
Epoch 26 - Valid Loss: 505.1390
Sub-losses: recon_loss: 502.1949, var_loss: 2.9441, anneal_factor: 0.9942, effective_beta_factor: 0.0994
```

```
Epoch 27 - Train Loss: 435.7699
Sub-losses: recon_loss: 432.1537, var_loss: 3.6161, anneal_factor: 0.9969, effective_beta_factor: 0.0997
Epoch 27 - Valid Loss: 504.8547
Sub-losses: recon_loss: 501.9875, var_loss: 2.8672, anneal_factor: 0.9969, effective_beta_factor: 0.0997
```

```
Epoch 28 - Train Loss: 433.8389
Sub-losses: recon_loss: 430.2849, var_loss: 3.5539, anneal_factor: 0.9983, effective_beta_factor: 0.0998
Epoch 28 - Valid Loss: 505.3432
Sub-losses: recon_loss: 502.3962, var_loss: 2.9471, anneal_factor: 0.9983, effective_beta_factor: 0.0998
```

```
Epoch 29 - Train Loss: 433.8051
Sub-losses: recon_loss: 430.1422, var_loss: 3.6630, anneal_factor: 0.9991, effective_beta_factor: 0.0999
Epoch 29 - Valid Loss: 504.1737
Sub-losses: recon_loss: 501.2206, var_loss: 2.9530, anneal_factor: 0.9991, effective_beta_factor: 0.0999
```

```
Epoch 30 - Train Loss: 443.3228
Sub-losses: recon_loss: 439.7624, var_loss: 3.5604, anneal_factor: 0.9995, effective_beta_factor: 0.0999
Epoch 30 - Valid Loss: 503.1838
Sub-losses: recon_loss: 500.1693, var_loss: 3.0145, anneal_factor: 0.9995, effective_beta_factor: 0.0999
```

```
Epoch 31 - Train Loss: 431.9200
Sub-losses: recon_loss: 428.1943, var_loss: 3.7258, anneal_factor: 0.9997, effective_beta_factor: 0.1000
Epoch 31 - Valid Loss: 504.0523
Sub-losses: recon_loss: 501.0170, var_loss: 3.0353, anneal_factor: 0.9997, effective_beta_factor: 0.1000
```

```
Epoch 32 - Train Loss: 432.1389
Sub-losses: recon_loss: 428.5857, var_loss: 3.5532, anneal_factor: 0.9998, effective_beta_factor: 0.1000
Epoch 32 - Valid Loss: 503.3811
Sub-losses: recon_loss: 500.4593, var_loss: 2.9218, anneal_factor: 0.9998, effective_beta_factor: 0.1000
```

```
Epoch 33 - Train Loss: 435.9395
Sub-losses: recon_loss: 432.3635, var_loss: 3.5760, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Epoch 33 - Valid Loss: 505.8085
Sub-losses: recon_loss: 502.9571, var_loss: 2.8514, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Processed 32 / 200 samples
Processed 64 / 200 samples
Processed 96 / 200 samples
Processed 128 / 200 samples
Processed 160 / 200 samples
Processed 192 / 200 samples
Processed 200 / 200 samples
```

```
reading parquet: data/XModalix-Tut-data/combined_rnaseq_formatted.parquet
```

```
reading parquet: data/XModalix-Tut-data/combined_clin_formatted.parquet
Given image size is possible, rescaling images to: 64x64
```

```
Successfully loaded 3230 images for img
```

```
anno key: rna
anno key: img
```

```
Converting 2261 images to torch.float32 tensors...
Converting 646 images to torch.float32 tensors...
Converting 323 images to torch.float32 tensors...
key: train, type: <class 'dict'>
key: valid, type: <class 'dict'>
key: test, type: <class 'dict'>
Dataset has UNPAIRED samples → using CoverageEnsuringSampler (2261 paired + 225 unpaired)
Dataset has UNPAIRED samples → using CoverageEnsuringSampler (323 paired + 32 unpaired)
```

```
Check if we need to pretrain: multi_bulk.rna
pretrain epochs : None
No pretraining for multi_bulk.rna
Check if we need to pretrain: img.img
pretrain epochs : None
No pretraining for img.img
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:469: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 7.3381 exceeded max norm of 5.
  warnings.warn(
```

```
--- Epoch 5/10 ---
```

```
Epoch 5/10 - Train Loss: 253.0092
Sub-losses - adver_loss: 17.9186, aggregated_sub_losses: 192.4159, paired_loss: 7.7542, class_loss: 34.9205, multi_bulk.rna.recon_loss: 115.5716, multi_bulk.rna.var_loss: 1.8430, multi_bulk.rna.loss: 117.4146, img.img.recon_loss: 73.5219, img.img.var_loss: 1.4794, img.img.loss: 75.0013, clf_loss: 1.0618
```

```
Epoch 5/10 - Valid Loss: 227.1937
Sub-losses - adver_loss: 17.8239, aggregated_sub_losses: 166.8495, paired_loss: 7.7761, class_loss: 34.7441, multi_bulk.rna.recon_loss: 90.2641, multi_bulk.rna.var_loss: 1.6118, multi_bulk.rna.loss: 91.8760, img.img.recon_loss: 73.6118, img.img.var_loss: 1.3618, img.img.loss: 74.9735, clf_loss: 0.9859
Storing checkpoint for epoch 4...
```

```
--- Epoch 10/10 ---
```

```
Epoch 10/10 - Train Loss: 240.8600
Sub-losses - adver_loss: 19.2354, aggregated_sub_losses: 182.4831, paired_loss: 7.0897, class_loss: 32.0518, multi_bulk.rna.recon_loss: 97.8461, multi_bulk.rna.var_loss: 6.0918, multi_bulk.rna.loss: 103.9379, img.img.recon_loss: 72.7175, img.img.var_loss: 5.8277, img.img.loss: 78.5452, clf_loss: 1.0012
```

```
Epoch 10/10 - Valid Loss: 229.5862
Sub-losses - adver_loss: 18.2204, aggregated_sub_losses: 172.5583, paired_loss: 7.1107, class_loss: 31.6967, multi_bulk.rna.recon_loss: 86.9691, multi_bulk.rna.var_loss: 5.6081, multi_bulk.rna.loss: 92.5772, img.img.recon_loss: 74.3049, img.img.var_loss: 5.6762, img.img.loss: 79.9811, clf_loss: 0.9996
Storing checkpoint for epoch 9...
```

```
Prediction complete.
Processing latent space results into a single AnnData object...
Identified source modality for latent space: 'multi_bulk.rna'
  - Added 17448 source feature IDs to .uns
Finished processing latent results.
```

## 2) TrainingDynamics Interface Deep Dive[¶](#2-trainingdynamics-interface-deep-dive)

Before accessing the actual results, we provide a theory section on our interface:

The `TrainingDynamics` object has the following form:\
`<epoch><split><data>`

So, if you want to access the train loss for the 5th epoch, you would use:

```
result.loss.get(epoch=5, split="train")
```

##### The `.get()` Method Explained[¶](#the-get-method-explained)

Let's say, we're interessted in thre reconstructions of our autoencoder.\
The `reconstructions.get()` method provides flexible access to reconstruction data stored during training. It can retrieve data for specific epochs, specific splits, or any combination of these parameters.

##### Parameters[¶](#parameters)

- **`epoch`** (Optional[int]):

  - Positive integer (e.g., `2`): Get reconstructions from that specific epoch
  - Negative integer (e.g., `-1`): Get the latest epoch (-1), second-to-last (-2), etc.
  - `None`: Return data for all epochs

- **`split`** (Optional[str]):

  - Valid values: `"train"`, `"valid"`, `"test"`
  - `None`: Return data for all splits

##### Return Value Behavior[¶](#return-value-behavior)

The method returns different types depending on the parameters:

1. **Both `epoch` and `split` specified**:

   - Returns a NumPy array for that specific epoch and split
   - Example: `get(epoch=2, split="train")` → `array([...])`

1. **Only `epoch` specified**:

   - Returns a dictionary of all splits for that epoch
   - Example: `get(epoch=2)` → `{"train": array([...]), "valid": array([...]), ...}`

1. **Only `split` specified**:

   - Returns a NumPy array containing data for that split across all epochs
   - Example: `get(split="train")` → `array([[...], [...], ...])` (first dimension represents epochs)

1. **Neither specified**:

   - Returns the complete nested dictionary structure
   - Example: `get()` → `{0: {"train": array([...])}, 1: {...}, ...}`

##### Special Handling[¶](#special-handling)

- If an invalid split is provided, a `KeyError` is raised
- Negative epoch indices work like Python list indexing (-1 is the last epoch)
- If an epoch doesn't exist, an empty array or dictionary is returned

##### Code Example[¶](#code-example)

```
# Access train reconstructions for the 5th epoch
train_epoch_5 = result.reconstructions.get(epoch=5, split="train")

# Access all splits for the latest epoch
latest_epoch_all_splits = result.reconstructions.get(epoch=-1)

# Access data for all epochs for the "valid" split
all_epochs_valid = result.reconstructions.get(split="valid")

# Access the full nested dictionary
full_data = result.reconstructions.get()
```

## 3) Working with Actual Results[¶](#3-working-with-actual-results)

### 3.1) Varix[¶](#31-varix)

Exemplary, we show how to get the `latentspaces` of the `result` attribute and to access different loss types.

In \[4\]:

Copied!

```
all_ls = result.latentspaces.get()
print(f"Keys of all latentspaces: {all_ls.keys()}")
```

all_ls = result.latentspaces.get() print(f"Keys of all latentspaces: {all_ls.keys()}")

```
Keys of all latentspaces: dict_keys([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])
```

We see that we have latent spaces for each epoch because we set `checkpoint_interval=1` in our configs in [Step 1](#1-filling-the-result-object).

For each epoch, we have the latent space for the `train` and `valid` splits. The `-1` epoch is a special key for the `test` split. For the other splits, negative indexing works as in Python lists: `-1` gives the last epoch, `-2` the second-to-last, and so on.

**Special Case**

> You cannot get the last epoch for all splits at once. You can either get the last epoch for `train` and `valid`, or only for `test`. See code below.

In \[5\]:

Copied!

```
print(f"Splits in 2. epoch: {result.latentspaces.get(epoch=2).keys()}")

# this will only give you the data for train and valid, since 'test' is a special case
print(f"Splits in epoch =-1: {result.latentspaces.get(epoch=-1).keys()}")
# Get test by adding the 'split' argument.
test_ls = result.latentspaces.get(split="test")
print(test_ls[0][0])
# get a specific epoch and specific split:

print("\n")
print("-"*80)

print(f"latentspace of one sample in train split at epoch 4: {result.latentspaces.get(split='train', epoch=4)[0]}")
```

print(f"Splits in 2. epoch: {result.latentspaces.get(epoch=2).keys()}")

# this will only give you the data for train and valid, since 'test' is a special case

print(f"Splits in epoch =-1: {result.latentspaces.get(epoch=-1).keys()}")

# Get test by adding the 'split' argument.

test_ls = result.latentspaces.get(split="test") print(test_ls[0][0])

# get a specific epoch and specific split:

print("\\n") print("-"\*80) print(f"latentspace of one sample in train split at epoch 4: {result.latentspaces.get(split='train', epoch=4)[0]}")

```
Splits in 2. epoch: dict_keys(['train', 'valid'])
Splits in epoch =-1: dict_keys(['train', 'valid'])
[ 3.6223373   0.09076981  0.9311101   0.4686319   0.5648282   4.362979
  7.462656    4.2597566  -3.0950072   5.624643   -0.7086104  -1.9337438
 -0.5916452  -0.30150637  6.15409     7.991743  ]


--------------------------------------------------------------------------------
latentspace of one sample in train split at epoch 4: [-0.7468408   2.9747071   0.17302047  0.04815996 -2.0036192   0.17140199
 -1.1278459   4.9971147   2.4795966  -0.19799425  2.1384008   2.5776906
 -0.8080328  -0.23301628 -0.6225459   2.2121964 ]
```

### 3.2 XModalix[¶](#32-xmodalix)

Accessing the results works slightly differently for `XModalix`, because we have:

1. **Multiple latent spaces** (one for each data modality).

1. **Translations and reference translations**, not just reconstructions.

   - These are reconstructions within one data modality on the same split as the translation. For example, if we translate from `rna` to `img`, the reference translation would be the reconstruction from `img` to `img`.
   - You can access these `translations` via the `reconstruction` attribute of the `result` object. First, apply the usual `TrainingDynamics` API via `get()`, then you get a dictionary for each data modality with translations and reference translations.

1. **Multiple losses**.

   - These are accessed via the `sub_losses` attribute of the `result` object, which is a dictionary of `TrainingDynamics`. First, select the loss type you're interested in, then work with the usual `TrainingDynamics` API.

   **Note on naming sub-losses:**

   - Global losses are simply named after the loss type, e.g., `class_loss`.

   - Losses per data modality are named with the following convention:

     ```
     <global_data_modality_key>.<specific_data_modality_name>.<loss_name>
     ```

     For example: `multi_bulk.rna.class_loss` or `multi_sc.celltype.reconstruction_loss`. See print below.

#### 3.2.1 Access Modality Latent Spaces[¶](#321-access-modality-latent-spaces)

As described in our `XModalix Deep Dive` [1], we fit one latent space per data modality.\
You can access this by first selecting the epoch and split you're interested in (standard `TrainingDynamics` API).\
The result will be a `Dict` with the name of each data modality as the key.

In \[6\]:

Copied!

```
print(f" Keys of data modalities for latent space dynamic: {xmodalix_result.latentspaces.get(epoch=-1, split='test').keys()}")
```

print(f" Keys of data modalities for latent space dynamic: {xmodalix_result.latentspaces.get(epoch=-1, split='test').keys()}")

```
 Keys of data modalities for latent space dynamic: dict_keys(['multi_bulk.rna', 'img.img'])
```

Now you can access the latent space of the `image` modality with the key `img.img`

In \[7\]:

Copied!

```
xmodalix_result.latentspaces.get(epoch=-1, split="test").get("img.img")
```

xmodalix_result.latentspaces.get(epoch=-1, split="test").get("img.img")

Out\[7\]:

```
array([[ 0.8092547 ,  4.121351  ,  1.4299108 , ...,  5.3962545 ,
         0.6538164 ,  0.9702057 ],
       [ 2.1221623 ,  5.256636  ,  0.04541756, ...,  5.646598  ,
         0.04192398,  1.9187347 ],
       [ 5.382866  ,  6.639792  , -0.14126651, ...,  6.236802  ,
        -1.1598693 ,  2.861813  ],
       ...,
       [ 1.267178  , -2.6413867 ,  0.61661077, ..., -0.13882025,
        -1.8977132 , 12.516397  ],
       [ 7.283764  , -1.0312403 , -1.7102883 , ...,  3.6146328 ,
        -0.6609394 ,  4.1691995 ],
       [ 1.247888  ,  4.62085   ,  1.3288131 , ...,  4.4530516 ,
        -0.98057437,  0.6013475 ]], shape=(646, 16), dtype=float32)
```

#### 3.2.2 Access Translation[¶](#322-access-translation)

In \[8\]:

Copied!

```
print("Get reconstruction keys")
# Frist define split and epoch you're interested in
# usually test split (there are no epochs, so by default this is always epoch=-1)
recons = xmodalix_result.reconstructions.get(split="test", epoch=-1)
print(recons.keys())
print("Getting Translation")
trans = recons.get("translation")
print(f"shape of translation: {trans.shape}")
```

print("Get reconstruction keys")

# Frist define split and epoch you're interested in

# usually test split (there are no epochs, so by default this is always epoch=-1)

recons = xmodalix_result.reconstructions.get(split="test", epoch=-1) print(recons.keys()) print("Getting Translation") trans = recons.get("translation") print(f"shape of translation: {trans.shape}")

```
Get reconstruction keys
dict_keys(['multi_bulk.rna', 'img.img', 'translation', 'reference_img.img_to_img.img'])
Getting Translation
shape of translation: (711, 1, 64, 64)
```

#### 3.2.3 Access Sub-Losses[¶](#323-access-sub-losses)

In \[9\]:

Copied!

```
sub_losses = xmodalix_result.sub_losses
print("Sub Losses:")
print(f"keys: {sub_losses.keys()}")
print("\n")
recon_dyn = sub_losses.get(key="paired_loss")
print("Value of paired loss in epoch 4 for train split")
print(f"{recon_dyn.get(split='train', epoch=4):.2f}")
```

sub_losses = xmodalix_result.sub_losses print("Sub Losses:") print(f"keys: {sub_losses.keys()}") print("\\n") recon_dyn = sub_losses.get(key="paired_loss") print("Value of paired loss in epoch 4 for train split") print(f"{recon_dyn.get(split='train', epoch=4):.2f}")

```
Sub Losses:
keys: dict_keys(['adver_loss', 'aggregated_sub_losses', 'paired_loss', 'class_loss', 'multi_bulk.rna.recon_loss', 'multi_bulk.rna.var_loss', 'multi_bulk.rna.anneal_factor', 'multi_bulk.rna.effective_beta_factor', 'multi_bulk.rna.loss', 'img.img.recon_loss', 'img.img.var_loss', 'img.img.anneal_factor', 'img.img.effective_beta_factor', 'img.img.loss', 'clf_loss'])


Value of paired loss in epoch 4 for train split
7.75
```

## 4) Non-TrainingDynamics Result Attributes[¶](#4-non-trainingdynamics-result-attributes)

There are other (intermediate) results that are not created during training but might still be interesting.\
These results do not follow a uniform interface like `TrainingDynamics`, but are often more straightforward. We go over each attribute quickly.

#### 4.1 Datasets[¶](#41-datasets)

The `datasets` attribute stores the preprocessed data in a `DatasetContainer`.\
This is basically a dict with `train`, `valid`, and `test` as keys, and each value is a child class of a PyTorch dataset.\
Whenever you need to re-access your preprocessed data, you can do so using the `datasets` attribute, as shown below:

In \[10\]:

Copied!

```
print(result.datasets)
print(result.datasets.train)
print(type(result.datasets.train.data))
result.datasets.train.metadata.head()
```

print(result.datasets) print(result.datasets.train) print(type(result.datasets.train.data)) result.datasets.train.metadata.head()

```
DatasetContainer(train=<autoencodix.data._numeric_dataset.NumericDataset object at 0x7c930fb41b80>, valid=<autoencodix.data._numeric_dataset.NumericDataset object at 0x7c930fc8e0f0>, test=<autoencodix.data._numeric_dataset.NumericDataset object at 0x7c933c175cd0>)
<autoencodix.data._numeric_dataset.NumericDataset object at 0x7c930fb41b80>
<class 'torch.Tensor'>
```

Out\[10\]:

|          | cell_type | batch  | donor  | cell_cycle | n_genes |
| -------- | --------- | ------ | ------ | ---------- | ------- |
| cell_0   | type_0    | batch2 | donor2 | S          | 346     |
| cell_1   | type_0    | batch3 | donor4 | G1         | 357     |
| cell_19  | type_3    | batch1 | donor1 | G1         | 345     |
| cell_190 | type_4    | batch1 | donor2 | S          | 338     |
| cell_191 | type_0    | batch3 | donor2 | S          | 349     |

#### 4.2 New Datasets[¶](#42-new-datasets)

Whenever you run the `predict` step of the pipeline and pass new, unseen data to it, we preprocess this data (if necessary).\
To avoid overwriting the original `datasets`, we store this in `new_datasets`.

If you run `predict` again with other data, `new_datasets` will be overridden.\
Otherwise, `new_datasets` works the same way as `datasets`.

First we create new data and then we run predict.

In \[11\]:

Copied!

```
import copy

from autoencodix.utils.example_data import EXAMPLE_MULTI_SC
new_data = copy.copy(EXAMPLE_MULTI_SC)
new_multi_sc = new_data.multi_sc["multi_sc"]
for modname, mod in new_multi_sc.mod.items():
    new_names = mod.obs_names.str.replace('cell', 'new_cell')
    mod.index = new_names

    mod.obs_names = new_names
    print(mod.index)
    new_multi_sc.mod[modname] = mod

new_data.multi_sc["multi_sc"] = new_multi_sc
new_data.multi_sc["multi_sc"].update()
```

import copy from autoencodix.utils.example_data import EXAMPLE_MULTI_SC new_data = copy.copy(EXAMPLE_MULTI_SC) new_multi_sc = new_data.multi_sc["multi_sc"] for modname, mod in new_multi_sc.mod.items(): new_names = mod.obs_names.str.replace('cell', 'new_cell') mod.index = new_names mod.obs_names = new_names print(mod.index) new_multi_sc.mod[modname] = mod new_data.multi_sc["multi_sc"] = new_multi_sc new_data.multi_sc["multi_sc"].update()

```
Index(['new_cell_0', 'new_cell_1', 'new_cell_10', 'new_cell_100',
       'new_cell_101', 'new_cell_102', 'new_cell_103', 'new_cell_104',
       'new_cell_105', 'new_cell_106',
       ...
       'new_cell_990', 'new_cell_991', 'new_cell_992', 'new_cell_993',
       'new_cell_994', 'new_cell_995', 'new_cell_996', 'new_cell_997',
       'new_cell_998', 'new_cell_999'],
      dtype='object', length=1000)
Index(['new_cell_0', 'new_cell_1', 'new_cell_10', 'new_cell_100',
       'new_cell_101', 'new_cell_102', 'new_cell_103', 'new_cell_104',
       'new_cell_105', 'new_cell_106',
       ...
       'new_cell_990', 'new_cell_991', 'new_cell_992', 'new_cell_993',
       'new_cell_994', 'new_cell_995', 'new_cell_996', 'new_cell_997',
       'new_cell_998', 'new_cell_999'],
      dtype='object', leng
... [output truncated]
```

```
.venv/lib/python3.12/site-packages/mudata/_core/mudata.py:758: ImplicitModificationWarning: Setting element `.varm['rna']` of view, initializing view as actual.
  attrm[mod] = mapping > 0
.venv/lib/python3.12/site-packages/mudata/_core/mudata.py:758: ImplicitModificationWarning: Setting element `.varm['protein']` of view, initializing view as actual.
  attrm[mod] = mapping > 0
```

Run the predict step:

In \[12\]:

Copied!

```
%%capture
varix.predict(data=new_data)
```

%%capture varix.predict(data=new_data)

Examine `datasets` and `new_datasets`:\
We see that `datasets` still is kept and only new_datasets is updated.

In \[13\]:

Copied!

```
print(f"Sample of original dataset: {result.datasets.train.sample_ids[0]}")

print(f"Sample of new dataset: {result.new_datasets.test.sample_ids[0]}")
```

print(f"Sample of original dataset: {result.datasets.train.sample_ids[0]}") print(f"Sample of new dataset: {result.new_datasets.test.sample_ids[0]}")

```
Sample of original dataset: cell_0
Sample of new dataset: new_cell_0
```

#### 4.3 Model Attribute[¶](#43-model-attribute)

This is straightforward the trained model as PytTorch Module.

In \[14\]:

Copied!

```
result.model
```

result.model

Out\[14\]:

```
VarixArchitecture(
  (_encoder): Sequential(
    (0): Linear(in_features=700, out_features=175, bias=True)
    (1): BatchNorm1d(175, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): Dropout(p=0.1, inplace=False)
    (3): ReLU()
    (4): Linear(in_features=175, out_features=43, bias=True)
    (5): BatchNorm1d(43, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): Dropout(p=0.1, inplace=False)
    (7): ReLU()
    (8): Linear(in_features=43, out_features=16, bias=True)
    (9): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): Dropout(p=0.1, inplace=False)
    (11): ReLU()
  )
  (_mu): Linear(in_features=16, out_features=16, bias=True)
  (_logvar): Linear(in_features=16, out_features=16, bias=True)
  (_deco
... [output truncated]
```

#### 4.4 Adata Latent[¶](#44-adata-latent)

We save the latent space of the `test` split from the final trained model as an `AnnData` object for the single-cell community.

This is also useful for non-single-cell cases, because you can still obtain the sample IDs via `.obs`.

In \[15\]:

Copied!

```
print(result.adata_latent)
print(result.adata_latent.obs)
```

print(result.adata_latent) print(result.adata_latent.obs)

```
AnnData object with n_obs × n_vars = 1000 × 16
    uns: 'var_names'
Empty DataFrame
Columns: []
Index: [new_cell_0, new_cell_1, new_cell_10, new_cell_100, new_cell_101, new_cell_102, new_cell_103, new_cell_104, new_cell_105, new_cell_106, new_cell_107, new_cell_108, new_cell_109, new_cell_11, new_cell_110, new_cell_111, new_cell_112, new_cell_113, new_cell_114, new_cell_115, new_cell_116, new_cell_117, new_cell_118, new_cell_119, new_cell_12, new_cell_120, new_cell_121, new_cell_122, new_cell_123, new_cell_124, new_cell_125, new_cell_126, new_cell_127, new_cell_128, new_cell_129, new_cell_13, new_cell_130, new_cell_131, new_cell_132, new_cell_133, new_cell_134, new_cell_135, new_cell_136, new_cell_137, new_cell_138, new_cell_139, new_cell_14, new_cell_140, new_cell_141, new_cell_142, new_
... [output truncated]
```

#### 4.5 Final Reconstruction[¶](#45-final-reconstruction)

This attribute gives you the exact data structure as you used for input i.e. `MuData` in our case, with the reconstructed values.

In \[16\]:

Copied!

```
result.final_reconstruction
```

result.final_reconstruction

Out\[16\]:

```
MuData object with n_obs × n_vars = 1000 × 700
  2 modalities
    rna:	1000 x 500
      obs:	'cell_type', 'batch', 'donor', 'cell_cycle', 'n_genes'
    protein:	1000 x 200
      obs:	'cell_type', 'batch', 'donor', 'cell_cycle', 'n_genes'
```

#### 4.6 Evaluation Embeddings[¶](#46-evaluation-embeddings)

Before we can access this attribute, we first need to run the `evaluate` step. This will use the latent space and train a downstream machine learning task. In our case, we want to classify the cancer type.

The results of this evaluate step will be stored in `embedding_evaluation`.

In \[17\]:

Copied!

```
%%capture
xmodalix.evaluate(params=["CANCER_TYPE"])
```

%%capture xmodalix.evaluate(params=["CANCER_TYPE"])

In \[18\]:

Copied!

```
xmodalix_result.embedding_evaluation
```

xmodalix_result.embedding_evaluation

Out\[18\]:

|     | score_split | CLINIC_PARAM | metric      | value    | ML_ALG               | ML_TYPE        | MODALITY       | ML_TASK | ML_SUBTASK                |
| --- | ----------- | ------------ | ----------- | -------- | -------------------- | -------------- | -------------- | ------- | ------------------------- |
| 0   | train       | CANCER_TYPE  | roc_auc_ovo | 0.995432 | LogisticRegression() | classification | multi_bulk.rna | Latent  | Latent\_$\_multi_bulk.rna |
| 1   | valid       | CANCER_TYPE  | roc_auc_ovo | 0.998581 | LogisticRegression() | classification | multi_bulk.rna | Latent  | Latent\_$\_multi_bulk.rna |
| 2   | test        | CANCER_TYPE  | roc_auc_ovo | 0.995974 | LogisticRegression() | classification | multi_bulk.rna | Latent  | Latent\_$\_multi_bulk.rna |
| 0   | train       | CANCER_TYPE  | roc_auc_ovo | 0.993840 | LogisticRegression() | classification | img.img        | Latent  | Latent\_$\_img.img        |
| 1   | valid       | CANCER_TYPE  | roc_auc_ovo | 0.986195 | LogisticRegression() | classification | img.img        | Latent  | Latent\_$\_img.img        |
| 2   | test        | CANCER_TYPE  | roc_auc_ovo | 0.986864 | LogisticRegression() | classification | img.img        | Latent  | Latent\_$\_img.img        |

In \[19\]:

Copied!

```
varix.result.datasets.test.metadata.head()
varix.evaluate(params=["batch"])
```

varix.result.datasets.test.metadata.head() varix.evaluate(params=["batch"])

```
Perform ML task with feature df: Latent
Latent
Perform ML task for target parameter: batch
Perform ML task with feature df: Latent
Latent
Perform ML task for target parameter: batch
```

```
.venv/lib/python3.12/site-packages/sklearn/linear_model/_logistic.py:451: OptimizeWarning: Unknown solver options: iprint
  opt_res = optimize.minimize(
.venv/lib/python3.12/site-packages/sklearn/linear_model/_logistic.py:451: OptimizeWarning: Unknown solver options: iprint
  opt_res = optimize.minimize(
```

Out\[19\]:

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
datasets: DatasetContainer(train=<autoencodix.data._numeric_dataset.NumericDataset object at 0x7c933c70fbf0>, valid=<autoencodix.data._numeric_dataset.NumericDataset object at 0x7c930fc8e0f0>, test=<
... [output truncated]
```

## 5 Special Methods to Obtain DataFrames[¶](#5-special-methods-to-obtain-dataframes)

As you've seen in the [Training Dynamics Section](#2-trainingdynamics-interface-deep-dive), we only get plain values of reconstructions and latetnspaces. Often it is more useful to have sample ids, too. We could obtain the sample ids in the same order via the `sample_ids` TrainingDynamic. To make this more accessible, we added the methods `get_latent_df` and `get_reconstructions_df`. Here you pass `epoch` and `split` as seen before and you get a pandas DataFrame for the specific split and epoch for the latent space or the reconstruction.

In \[20\]:

Copied!

```
result.get_latent_df(epoch=-1, split="test").head()
```

result.get_latent_df(epoch=-1, split="test").head()

Out\[20\]:

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




new_cell_0
1.169099
6.428045
-0.909395
0.374270
0.522860
-0.824939
1.320667
0.916621
3.864682
0.547017
8.015424
4.431547
-0.753605
-1.017359
0.735275
4.925269


new_cell_1
3.025050
6.690741
0.612838
-0.001472
0.647654
-1.513513
1.393777
-1.178969
1.668167
-1.575379
9.334797
3.258978
1.502755
0.931644
-0.203693
5.011363


new_cell_10
-0.952251
0.109878
0.152486
0.666301
-0.460282
0.398765
-0.068070
1.201444
-0.069014
-1.078704
1.034621
1.796616
0.391577
0.868087
-0.355039
0.085928


new_cell_100
-0.601422
-2.526164
-0.972690
0.702619
2.239834
0.029732
0.817120
0.423556
-0.162558
0.135305
0.155343
0.305283
-0.373023
0.563850
-0.91012
... [output truncated]
```

In \[21\]:

Copied!

```
result.get_reconstructions_df(epoch=-1, split="test").head()
```

result.get_reconstructions_df(epoch=-1, split="test").head()

Out\[21\]:

```
gene_0
gene_1
gene_2
gene_3
gene_4
gene_5
gene_6
gene_7
gene_8
gene_9
...
protein_190
protein_191
protein_192
protein_193
protein_194
protein_195
protein_196
protein_197
protein_198
protein_199




new_cell_0
1.182295
-0.796981
0.030274
1.458777
-0.325000
0.197361
-0.000707
-0.019382
-0.742352
-0.733279
...
-1.050608
1.612186
0.162170
-0.430544
-0.057345
-1.153803
-0.433501
0.075737
-1.191396
1.251562


new_cell_1
1.172235
-0.790350
0.034178
1.442763
-0.313463
0.197347
-0.000446
-0.024263
-0.737805
-0.726801
...
-1.042170
1.587032
0.162941
-0.421808
-0.042026
-1.147030
-0.442250
0.061307
-1.191653
1.268421


new_cell_10
-0.613804
0.630371
-0.016076
-0.494771
-0.308604
-0.622271
-0.010671
-0.022798
0.682072
0.599002
...
0.906552
-0.558953
-0.075986
-0.452418
-0.095015
0.129426
-0.444
... [output truncated]
```
