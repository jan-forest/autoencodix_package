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

# How to Save the Pipeline Memory Efficiently[¶](#how-to-save-the-pipeline-memory-efficiently)

Our pipeline offers extensive insights into the dynamics of training and other intermediate results.\
When re-using the pipeline (with `.save` and `.load` functionality), this can create very large binary files, especially when working with large datasets.

Therefore, the default way of saving the pipeline is by discarding most of the training dynamics and intermediate results.\
In this tutorial, we show:

- Which data is kept and which is discarded.
- The trade-offs in visualizing and evaluating a saved pipeline.
- How to save the complete pipeline including all training dynamics and intermediate results.

## Theory: What Data is Kept?[¶](#theory-what-data-is-kept)

### Result Object[¶](#result-object)

The main driver of memory is the `result` attribute and the `_datasets` (depending on the data size) in the pipeline and the trainer.

In the `result` attribute, we save the following:

- model

- adata_latent

  All other attributes are set to empty default values.

### Pipeline[¶](#pipeline)

Here we keep all attributes excpept:

- `.visualizer`
- `_datasets` We remove the `.visualizer`, because this stores plots which can be misleading when rerunning the pipeline and seeing old plots that do not fit the new data.\
  We remove the `_datasets` attribute, because it is often uses the most memory. All other attributes are kept. This is especially relevant for the `config` and the `_preprocessor`.\
  Whenever you perform any pipeline step, it will use the initial config you passed when first creating the pipeline.\
  The preprocessor is necessary because when running `predict` with new data, this should be preprocessed the same way as initially (e.g., which genes to keep, how the scalers were fitted, etc.).

### Trainer[¶](#trainer)

We remove most attribute of the `_trainer` via the method `purge`.

**Note**

> Calling `.save` does not change the pipeline object in memory: after saving, you can keep working with it as before.\
> The slimming described above only applies to the saved files, that is, to the pipeline you get back from `.load`.\
> If you want to keep everything in the saved files as well, please run `.save(save_all=True)`.

## Practical Examples[¶](#practical-examples)

### First we run a varix pipeline[¶](#first-we-run-a-varix-pipeline)

In \[2\]:

Copied!

```
from autoencodix.utils.example_data import EXAMPLE_MULTI_SC
from autoencodix.configs.varix_config import VarixConfig
from autoencodix.configs.default_config import DataCase, DataConfig, DataInfo
import autoencodix as acx

my_config = VarixConfig(
    learning_rate=0.001,
    epochs=30,
    checkpoint_interval=5,
    default_vae_loss="kl",  # kl or mmd possible
    data_case=DataCase.MULTI_SINGLE_CELL,
)
print("\n")
print("Starting Pipeline")
print("-" * 50)
print("-" * 50)
varix = acx.Varix(
    data=EXAMPLE_MULTI_SC,
    config=my_config,
)
result = varix.run()
```

from autoencodix.utils.example_data import EXAMPLE_MULTI_SC from autoencodix.configs.varix_config import VarixConfig from autoencodix.configs.default_config import DataCase, DataConfig, DataInfo import autoencodix as acx my_config = VarixConfig( learning_rate=0.001, epochs=30, checkpoint_interval=5, default_vae_loss="kl", # kl or mmd possible data_case=DataCase.MULTI_SINGLE_CELL, ) print("\\n") print("Starting Pipeline") print("-" * 50) print("-" * 50) varix = acx.Varix( data=EXAMPLE_MULTI_SC, config=my_config, ) result = varix.run()

```
Starting Pipeline
--------------------------------------------------
--------------------------------------------------
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
.venv/lib/python3.12/site-packages/torch/nn/modules/linear.py:134: UserWarning: Requested unified CUBLASLT workspace size of 1048576 bytes exceeds CUBLAS workspace size of 131072 bytes. Please increase CUBLAS workspace size via CUBLAS_WORKSPACE_CONFIG or decrease requested CUBLASLT_WORKSPACE_SIZE. Otherwise CUBLASLT workspace size will be limited to the CUBLAS workspace size. (Triggered internally at /__w/pytorch/pytorch/aten/src/ATen/cuda/CublasHandlePool.cpp:294.)
  return F.linear(input, self.weight, self.bias)
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:305: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 12.0150 exceeded max norm of 1.
  warnings.warn(
```

```
Epoch 5 - Train Loss: 573.3867
Sub-losses: recon_loss: 573.3854, var_loss: 0.0014, anneal_factor: 0.0007, effective_beta_factor: 0.0001
Epoch 5 - Valid Loss: 593.4502
Sub-losses: recon_loss: 593.4493, var_loss: 0.0009, anneal_factor: 0.0007, effective_beta_factor: 0.0001
```

```
Epoch 10 - Train Loss: 480.1706
Sub-losses: recon_loss: 480.0405, var_loss: 0.1300, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Epoch 10 - Valid Loss: 558.9977
Sub-losses: recon_loss: 558.9171, var_loss: 0.0807, anneal_factor: 0.0180, effective_beta_factor: 0.0018
```

```
Epoch 15 - Train Loss: 456.1446
Sub-losses: recon_loss: 454.7632, var_loss: 1.3814, anneal_factor: 0.3392, effective_beta_factor: 0.0339
Epoch 15 - Valid Loss: 515.6952
Sub-losses: recon_loss: 514.7790, var_loss: 0.9161, anneal_factor: 0.3392, effective_beta_factor: 0.0339
```

```
Epoch 20 - Train Loss: 446.9890
Sub-losses: recon_loss: 443.6115, var_loss: 3.3775, anneal_factor: 0.9350, effective_beta_factor: 0.0935
Epoch 20 - Valid Loss: 508.1452
Sub-losses: recon_loss: 505.6702, var_loss: 2.4750, anneal_factor: 0.9350, effective_beta_factor: 0.0935
```

```
Epoch 25 - Train Loss: 432.6413
Sub-losses: recon_loss: 429.0416, var_loss: 3.5996, anneal_factor: 0.9975, effective_beta_factor: 0.0998
Epoch 25 - Valid Loss: 505.4625
Sub-losses: recon_loss: 502.6706, var_loss: 2.7919, anneal_factor: 0.9975, effective_beta_factor: 0.0998
```

```
Epoch 30 - Train Loss: 442.5652
Sub-losses: recon_loss: 439.1634, var_loss: 3.4018, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Epoch 30 - Valid Loss: 505.6029
Sub-losses: recon_loss: 502.7022, var_loss: 2.9007, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Processed 32 / 200 samples
Processed 64 / 200 samples
Processed 96 / 200 samples
Processed 128 / 200 samples
Processed 160 / 200 samples
Processed 192 / 200 samples
Processed 200 / 200 samples
```

#### Saving and Loading Explained[¶](#saving-and-loading-explained)

Here we first check the latent space (could be any other training dynamic) before saving.\
Saving leaves the pipeline in memory untouched, so the latent space is still there afterwards.\
Only the saved files are slimmed down: in the pipeline we load in the next step, the `result` object keeps only `model`, `adata_latent`, and `embedding_evaluation`.

In \[3\]:

Copied!

```
import os

ls_before_save = result.latentspaces.get(epoch=-1, split="test")
print("Length of latentspace before saving")
print(len(ls_before_save))
varix.save(os.path.join("tutorial_res", "varix_backup.pkl"))
ls_after_save = result.latentspaces.get(epoch=-1, split="test")
print("Length of latentspace after saving (the pipeline in memory is unchanged)")
print(len(ls_after_save))
```

import os ls_before_save = result.latentspaces.get(epoch=-1, split="test") print("Length of latentspace before saving") print(len(ls_before_save)) varix.save(os.path.join("tutorial_res", "varix_backup.pkl")) ls_after_save = result.latentspaces.get(epoch=-1, split="test") print("Length of latentspace after saving (the pipeline in memory is unchanged)") print(len(ls_after_save))

```
Length of latentspace before saving
200
Preprocessor saved successfully.
```

```
saving memory efficient
```

```
Pipeline object saved successfully.
Length of latentspace after saving (the pipeline in memory is unchanged)
200
```

#### Implications for Visualization[¶](#implications-for-visualization)

We load the model and see the implications of this in `visualize`.

In \[4\]:

Copied!

```
varix_loaded = acx.Varix().load(os.path.join("tutorial_res", "varix_backup.pkl"))
print("Number of epochs with a stored latent space in the loaded pipeline")
print(len(varix_loaded.result.latentspaces.get()))
```

varix_loaded = acx.Varix().load(os.path.join("tutorial_res", "varix_backup.pkl")) print("Number of epochs with a stored latent space in the loaded pipeline") print(len(varix_loaded.result.latentspaces.get()))

```
Attempting to load a pipeline from tutorial_res/varix_backup.pkl...
Pipeline object loaded successfully. Actual type: Varix
Preprocessor loaded successfully.
Number of epochs with a stored latent space in the loaded pipeline
0
```

In \[5\]:

Copied!

```
varix_loaded.visualize()
```

varix_loaded.visualize()

> Note that `visualize` shows nothing here. This is expected: the loss curves and latent spaces from training are not part of the memory-efficient save, so there is nothing to plot.\
> However, we can run a predict step with the trained model to get latent space visualizations again.

In \[6\]:

Copied!

```
res_loaded = varix_loaded.predict(data=EXAMPLE_MULTI_SC)
```

res_loaded = varix_loaded.predict(data=EXAMPLE_MULTI_SC)

```
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
mudata: View of MuData object with n_obs × n_vars = 1000 × 700
  2 modalities
    rna:	1000 x 500
      obs:	'cell_type', 'batch', 'donor', 'cell_cycle', 'n_genes'
    protein:	1000 x 200
      obs:	'cell_type', 'batch', 'donor', 'cell_cycle', 'n_genes'
Processing 1 MuData objects: ['multi_sc']
Processing test split
Processing test modality: multi_sc
Processed 32 / 1000 samples
Processed 64 / 1000 samples
Processed 96 / 1000 samples
Processed 128 / 1000 samples
Processed 160 / 1000 samples
Processed 192 / 1000 samples
Processed 224 / 1000 samples
Processed 256 / 1000 samples
Processed 288 / 1000 samples
Processed 320 / 1000 samples
Processed 352 / 1000 samples
Processed 384 / 1000 samples
Processed 41
... [output truncated]
```

```
Processed 608 / 1000 samples
Processed 640 / 1000 samples
Processed 672 / 1000 samples
Processed 704 / 1000 samples
Processed 736 / 1000 samples
Processed 768 / 1000 samples
Processed 800 / 1000 samples
Processed 832 / 1000 samples
Processed 864 / 1000 samples
Processed 896 / 1000 samples
Processed 928 / 1000 samples
Processed 960 / 1000 samples
Processed 992 / 1000 samples
Processed 1000 / 1000 samples
n_samples in format recon: 1000
train
n_samples from datatpackge: {'paired_count': 1000}
```

In \[7\]:

Copied!

```
varix_loaded.visualize()
# we use test split here, because predict with new data is equal to test split
varix_loaded.show_result(split="test")
```

varix_loaded.visualize()

# we use test split here, because predict with new data is equal to test split

varix_loaded.show_result(split="test")

```
Creating plots ...
```

#### Implications for Evaluate[¶](#implications-for-evaluate)

The evaluate step needs latent spaces. Directly after loading a memory-efficient save there are none, and `evaluate` raises an error that asks you to run `predict` first.\
After the predict step above it works, using the data we passed to `predict`:

In \[8\]:

Copied!

```
res_eval = varix_loaded.evaluate(params=["cell_type"])
```

res_eval = varix_loaded.evaluate(params=["cell_type"])

```
Perform ML task with feature df: Latent
Latent
Perform ML task for target parameter: cell_type
```

```
.venv/lib/python3.12/site-packages/sklearn/linear_model/_logistic.py:451: OptimizeWarning: Unknown solver options: iprint
  opt_res = optimize.minimize(
```

## How to Keep All Data[¶](#how-to-keep-all-data)

When you really want to investigate all training dynamics and results, we recommend setting `save_all=True`. See below:

In \[9\]:

Copied!

```
from autoencodix.utils.example_data import EXAMPLE_MULTI_BULK

my_config = VarixConfig(
    learning_rate=0.001,
    epochs=30,
    checkpoint_interval=5,
    default_vae_loss="kl",  # kl or mmd possible
    data_case=DataCase.MULTI_BULK,
)
n_varix = acx.Varix(data=EXAMPLE_MULTI_BULK, config=my_config)
result_new = n_varix.run()
n_varix.save(file_path=os.path.join("tutorial_res", "new_varix.pkl"), save_all=True)
```

from autoencodix.utils.example_data import EXAMPLE_MULTI_BULK my_config = VarixConfig( learning_rate=0.001, epochs=30, checkpoint_interval=5, default_vae_loss="kl", # kl or mmd possible data_case=DataCase.MULTI_BULK, ) n_varix = acx.Varix(data=EXAMPLE_MULTI_BULK, config=my_config) result_new = n_varix.run() n_varix.save(file_path=os.path.join("tutorial_res", "new_varix.pkl"), save_all=True)

```
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
anno key: transcriptomics
anno key: proteomics
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:305: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 11.2745 exceeded max norm of 1.
  warnings.warn(
```

```
Epoch 5 - Train Loss: 191.2635
Sub-losses: recon_loss: 191.2631, var_loss: 0.0004, anneal_factor: 0.0007, effective_beta_factor: 0.0001
Epoch 5 - Valid Loss: 157.2021
Sub-losses: recon_loss: 157.2018, var_loss: 0.0002, anneal_factor: 0.0007, effective_beta_factor: 0.0001
```

```
Epoch 10 - Train Loss: 180.9708
Sub-losses: recon_loss: 180.9558, var_loss: 0.0149, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Epoch 10 - Valid Loss: 148.6413
Sub-losses: recon_loss: 148.6300, var_loss: 0.0113, anneal_factor: 0.0180, effective_beta_factor: 0.0018
```

```
Epoch 15 - Train Loss: 171.6496
Sub-losses: recon_loss: 171.2838, var_loss: 0.3658, anneal_factor: 0.3392, effective_beta_factor: 0.0339
Epoch 15 - Valid Loss: 142.7222
Sub-losses: recon_loss: 142.4554, var_loss: 0.2668, anneal_factor: 0.3392, effective_beta_factor: 0.0339
```

```
Epoch 20 - Train Loss: 164.6376
Sub-losses: recon_loss: 163.4578, var_loss: 1.1798, anneal_factor: 0.9350, effective_beta_factor: 0.0935
Epoch 20 - Valid Loss: 142.1403
Sub-losses: recon_loss: 141.3613, var_loss: 0.7790, anneal_factor: 0.9350, effective_beta_factor: 0.0935
```

```
Epoch 25 - Train Loss: 160.7286
Sub-losses: recon_loss: 159.2397, var_loss: 1.4890, anneal_factor: 0.9975, effective_beta_factor: 0.0998
Epoch 25 - Valid Loss: 134.0311
Sub-losses: recon_loss: 133.0022, var_loss: 1.0289, anneal_factor: 0.9975, effective_beta_factor: 0.0998
```

```
Epoch 30 - Train Loss: 156.4330
Sub-losses: recon_loss: 154.7298, var_loss: 1.7032, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Epoch 30 - Valid Loss: 131.7347
Sub-losses: recon_loss: 130.5052, var_loss: 1.2295, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Processed 32 / 100 samples
Processed 64 / 100 samples
Processed 96 / 100 samples
Processed 100 / 100 samples
```

```
Preprocessor saved successfully.
```

```
Pipeline object saved successfully.
```

Now we can load the pipeline and all intermediate data is kept.

In \[10\]:

Copied!

```
loaded_all_varix = acx.Varix().load(file_path=os.path.join("tutorial_res", "new_varix.pkl"))
```

loaded_all_varix = acx.Varix().load(file_path=os.path.join("tutorial_res", "new_varix.pkl"))

```
Attempting to load a pipeline from tutorial_res/new_varix.pkl...
```

```
Pipeline object loaded successfully. Actual type: Varix
```

```
Preprocessor loaded successfully.
```

In \[11\]:

Copied!

```
loaded_all_varix.visualize()
```

loaded_all_varix.visualize()

In \[12\]:

Copied!

```
loaded_all_varix.show_result()
```

loaded_all_varix.show_result()

```
Creating plots ...
```

And we see that the loss plots from the initial training are still present.

## Implications for Ontix and XModalix[¶](#implications-for-ontix-and-xmodalix)

These two pipelines offer more/other visualizations.

### ❗❗ Requirements: Getting Tutorial Data for XModalixj ❗❗[¶](#requirements-getting-tutorial-data-for-xmodalixj)

The data for this tutorial is hosted on Hugging Face Hub ([autoencodix/tcga](https://huggingface.co/datasets/autoencodix/tcga)) and is downloaded automatically in the cell below on first run.

### XModalix[¶](#xmodalix)

In \[13\]:

Copied!

```
import os
import autoencodix as acx

from autoencodix.configs.xmodalix_config import XModalixConfig
from autoencodix.configs.default_config import DataCase

p = os.getcwd()
d = "autoencodix_package"
if d not in p:
    raise FileNotFoundError(f"'{d}' not found in path: {p}")
os.chdir(os.sep.join(p.split(os.sep)[: p.split(os.sep).index(d) + 1]))
print(f"Changed to: {os.getcwd()}")

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
    epochs=30,
    latent_dim=8,
    requires_paired=False,
    pretrain_epochs=2,
    data_case=DataCase.IMG_TO_BULK,
    annotation_columns=["CANCER_TYPE_ACRONYM"],
    data_config=DataConfig(
        data_info={
            "img": DataInfo(
                file_path=img_root,
                data_type="IMG",
                scaling="MINMAX",
                translate_direction="to",
                pretrain_epochs=2,
                # extra_anno_file=imganno_file,
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
result = xmodalix.run()
outpath = os.path.join("tutorial_res", "xmodalix.pkl")
xmodalix.save(file_path=outpath)
```

import os import autoencodix as acx from autoencodix.configs.xmodalix_config import XModalixConfig from autoencodix.configs.default_config import DataCase p = os.getcwd() d = "autoencodix_package" if d not in p: raise FileNotFoundError(f"'{d}' not found in path: {p}") os.chdir(os.sep.join(p.split(os.sep)[: p.split(os.sep).index(d) + 1])) print(f"Changed to: {os.getcwd()}")

# ---------------------------------------------------------------------

# Data is hosted on Hugging Face Hub and downloaded (and locally cached)

# automatically, then placed under the paths used below

# ---------------------------------------------------------------------

import shutil import zipfile import pandas as pd from huggingface_hub import hf_hub_download HF_REPO_ID = "autoencodix/tcga" xmod_data_root = "data/XModalix-Tut-data" os.makedirs(xmod_data_root, exist_ok=True) rna_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="rna.parquet") shutil.copyfile(rna_path, os.path.join(xmod_data_root, "combined_rnaseq_formatted.parquet")) clin_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="clinical.parquet") mapping_path = hf_hub_download( repo_id=HF_REPO_ID, repo_type="dataset", filename="xmodalix/tcga_image_mappings.txt" )

# The base clinical file has no image column; the XModalix image mapping is a separate,

# per-architecture extension that we join in here (left join: not every sample has an image).

clin_df = pd.read_parquet(clin_path) mapping_df = pd.read_csv(mapping_path, sep="\\t", index_col="sample_ids") clin_df["img_paths"] = mapping_df["img_paths"] clin_df.to_parquet(os.path.join(xmod_data_root, "combined_clin_formatted.parquet")) images_zip = hf_hub_download( repo_id=HF_REPO_ID, repo_type="dataset", filename="xmodalix/tcga_fake_images.zip" ) images_extract_dir = os.path.join(xmod_data_root, "images") if not os.path.isdir(os.path.join(images_extract_dir, "tcga_fake")): with zipfile.ZipFile(images_zip) as zf: zf.extractall(images_extract_dir) clin_file = os.path.join("data/XModalix-Tut-data/combined_clin_formatted.parquet") rna_file = os.path.join("data/XModalix-Tut-data/combined_rnaseq_formatted.parquet") img_root = os.path.join("data/XModalix-Tut-data/images/tcga_fake") xmodalix_config = XModalixConfig( checkpoint_interval=5, class_param="CANCER_TYPE_ACRONYM", epochs=30, latent_dim=8, requires_paired=False, pretrain_epochs=2, data_case=DataCase.IMG_TO_BULK, annotation_columns=["CANCER_TYPE_ACRONYM"], data_config=DataConfig( data_info={ "img": DataInfo( file_path=img_root, data_type="IMG", scaling="MINMAX", translate_direction="to", pretrain_epochs=2,

# extra_anno_file=imganno_file,

), "rna": DataInfo( file_path=rna_file, data_type="NUMERIC", scaling="MINMAX", translate_direction="from", ), "anno": DataInfo(file_path=clin_file, data_type="ANNOTATION", sep="\\t"), }, ), ) xmodalix = acx.XModalix(config=xmodalix_config) result = xmodalix.run() outpath = os.path.join("tutorial_res", "xmodalix.pkl") xmodalix.save(file_path=outpath)

```
Changed to: /home/janew/ACX/autoencodix_package
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
pretrain epochs : 2
starting pretraining for: multi_bulk.rna with <class 'autoencodix.trainers._general_trainer.GeneralTrainer'>
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:305: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 8.7383 exceeded max norm of 5.
  warnings.warn(
```

```
Check if we need to pretrain: img.img
pretrain epochs : 2
starting pretraining for: img.img with <class 'autoencodix.trainers._general_trainer.GeneralTrainer'>
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:305: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 87.7819 exceeded max norm of 5.
  warnings.warn(
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:469: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 5.2390 exceeded max norm of 5.
  warnings.warn(
```

```
--- Epoch 5/30 ---
```

```
Epoch 5/30 - Train Loss: 973.1145
Sub-losses - adver_loss: 148.6768, aggregated_sub_losses: 497.6880, paired_loss: 34.4381, class_loss: 292.3116, multi_bulk.rna.recon_loss: 303.5127, multi_bulk.rna.var_loss: 15.0653, multi_bulk.rna.loss: 318.5780, img.img.recon_loss: 179.1060, img.img.var_loss: 0.0040, img.img.loss: 179.1099, clf_loss: 0.5747
```

```
Epoch 5/30 - Valid Loss: 708.2265
Sub-losses - adver_loss: 142.1280, aggregated_sub_losses: 366.1297, paired_loss: 16.3558, class_loss: 183.6130, multi_bulk.rna.recon_loss: 175.1527, multi_bulk.rna.var_loss: 16.4612, multi_bulk.rna.loss: 191.6139, img.img.recon_loss: 174.5117, img.img.var_loss: 0.0041, img.img.loss: 174.5158, clf_loss: 0.5569
Storing checkpoint for epoch 4...
```

```
--- Epoch 10/30 ---
```

```
Epoch 10/30 - Train Loss: 1033.3141
Sub-losses - adver_loss: 175.5369, aggregated_sub_losses: 641.3895, paired_loss: 26.5306, class_loss: 189.8571, multi_bulk.rna.recon_loss: 279.9206, multi_bulk.rna.var_loss: 182.0211, multi_bulk.rna.loss: 461.9417, img.img.recon_loss: 179.3397, img.img.var_loss: 0.1082, img.img.loss: 179.4478, clf_loss: 0.5574
```

```
Epoch 10/30 - Valid Loss: 3970.3277
Sub-losses - adver_loss: 295.0521, aggregated_sub_losses: 3041.7827, paired_loss: 73.1101, class_loss: 560.3826, multi_bulk.rna.recon_loss: 1562.5207, multi_bulk.rna.var_loss: 1304.1635, multi_bulk.rna.loss: 2866.6843, img.img.recon_loss: 174.9920, img.img.var_loss: 0.1064, img.img.loss: 175.0984, clf_loss: 0.5622
Storing checkpoint for epoch 9...
```

```
--- Epoch 15/30 ---
```

```
Epoch 15/30 - Train Loss: 5561.0575
Sub-losses - adver_loss: 168.6471, aggregated_sub_losses: 5180.0343, paired_loss: 23.0429, class_loss: 189.3332, multi_bulk.rna.recon_loss: 294.3180, multi_bulk.rna.var_loss: 4705.1097, multi_bulk.rna.loss: 4999.4278, img.img.recon_loss: 178.5465, img.img.var_loss: 2.0600, img.img.loss: 180.6065, clf_loss: 0.5540
```

```
Epoch 15/30 - Valid Loss: 10979.0627
Sub-losses - adver_loss: 173.6257, aggregated_sub_losses: 10563.2290, paired_loss: 20.6856, class_loss: 221.5224, multi_bulk.rna.recon_loss: 223.8582, multi_bulk.rna.var_loss: 10163.1214, multi_bulk.rna.loss: 10386.9796, img.img.recon_loss: 174.2959, img.img.var_loss: 1.9535, img.img.loss: 176.2494, clf_loss: 0.5458
Storing checkpoint for epoch 14...
```

```
--- Epoch 20/30 ---
```

```
Epoch 20/30 - Train Loss: 21712.9816
Sub-losses - adver_loss: 156.1817, aggregated_sub_losses: 21257.8955, paired_loss: 36.9777, class_loss: 261.9265, multi_bulk.rna.recon_loss: 312.3943, multi_bulk.rna.var_loss: 20760.3995, multi_bulk.rna.loss: 21072.7937, img.img.recon_loss: 179.3999, img.img.var_loss: 5.7020, img.img.loss: 185.1019, clf_loss: 0.5507
```

```
Epoch 20/30 - Valid Loss: 64190.9724
Sub-losses - adver_loss: 436.0306, aggregated_sub_losses: 63348.0559, paired_loss: 39.8823, class_loss: 367.0028, multi_bulk.rna.recon_loss: 522.6638, multi_bulk.rna.var_loss: 62642.8544, multi_bulk.rna.loss: 63165.5182, img.img.recon_loss: 177.5929, img.img.var_loss: 4.9443, img.img.loss: 182.5372, clf_loss: 0.5701
Storing checkpoint for epoch 19...
```

```
--- Epoch 25/30 ---
```

```
Epoch 25/30 - Train Loss: 11477.7417
Sub-losses - adver_loss: 167.0133, aggregated_sub_losses: 11054.9451, paired_loss: 25.7645, class_loss: 230.0189, multi_bulk.rna.recon_loss: 291.4800, multi_bulk.rna.var_loss: 10577.9699, multi_bulk.rna.loss: 10869.4500, img.img.recon_loss: 179.3980, img.img.var_loss: 6.0971, img.img.loss: 185.4951, clf_loss: 0.5616
```

```
Epoch 25/30 - Valid Loss: 10475.6903
Sub-losses - adver_loss: 162.8632, aggregated_sub_losses: 10122.3749, paired_loss: 21.0298, class_loss: 169.4224, multi_bulk.rna.recon_loss: 240.1526, multi_bulk.rna.var_loss: 9701.4298, multi_bulk.rna.loss: 9941.5824, img.img.recon_loss: 175.7371, img.img.var_loss: 5.0552, img.img.loss: 180.7923, clf_loss: 0.5701
Storing checkpoint for epoch 24...
```

```
--- Epoch 30/30 ---
```

```
Epoch 30/30 - Train Loss: 13417.7246
Sub-losses - adver_loss: 187.3442, aggregated_sub_losses: 13003.1438, paired_loss: 28.1446, class_loss: 199.0920, multi_bulk.rna.recon_loss: 270.4404, multi_bulk.rna.var_loss: 12546.8633, multi_bulk.rna.loss: 12817.3036, img.img.recon_loss: 179.8017, img.img.var_loss: 6.0385, img.img.loss: 185.8402, clf_loss: 0.5627
```

```
Epoch 30/30 - Valid Loss: 15275.7903
Sub-losses - adver_loss: 147.8860, aggregated_sub_losses: 14873.3880, paired_loss: 19.8689, class_loss: 234.6479, multi_bulk.rna.recon_loss: 198.6000, multi_bulk.rna.var_loss: 14493.7134, multi_bulk.rna.loss: 14692.3131, img.img.recon_loss: 174.8119, img.img.var_loss: 6.2627, img.img.loss: 181.0746, clf_loss: 0.5552
Storing checkpoint for epoch 29...
```

```
Prediction complete.
Processing latent space results into a single AnnData object...
Identified source modality for latent space: 'multi_bulk.rna'
  - Added 17448 source feature IDs to .uns
Finished processing latent results.
```

```
Preprocessor saved successfully.
```

```
saving memory efficient
```

```
Pipeline object saved successfully.
```

Now we can load the pipeline again

In \[14\]:

Copied!

```
xmodalix_loaded = acx.XModalix.load(outpath)
```

xmodalix_loaded = acx.XModalix.load(outpath)

```
Attempting to load a pipeline from tutorial_res/xmodalix.pkl...
```

```
Pipeline object loaded successfully. Actual type: XModalix
```

```
Preprocessor loaded successfully.
```

Since we used memory efficient saving, we did not save the data, thus we need to run preprocessing again, this will read and preprocess the files defined in the config.

In \[15\]:

Copied!

```
xmodalix_loaded.preprocess()
```

xmodalix_loaded.preprocess()

```
reading parquet: data/XModalix-Tut-data/combined_rnaseq_formatted.parquet
```

```
reading parquet: data/XModalix-Tut-data/combined_clin_formatted.parquet
Given image size is possible, rescaling images to: 64x64
```

```
Successfully loaded 3230 images for img
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
```

No we can use the trained model to predict again:

In \[16\]:

Copied!

```
# now you can use the model to predict with a different pair again:

r = xmodalix_loaded.predict(from_key="rna", to_key="img")
```

# now you can use the model to predict with a different pair again:

r = xmodalix_loaded.predict(from_key="rna", to_key="img")

```
Prediction complete.
Processing latent space results into a single AnnData object...
Identified source modality for latent space: 'multi_bulk.rna'
  - Added 17448 source feature IDs to .uns
Finished processing latent results.
```

In \[17\]:

Copied!

```
xmodalix_loaded.visualize()
```

xmodalix_loaded.visualize()

In \[18\]:

Copied!

```
xmodalix_loaded.show_result()
```

xmodalix_loaded.show_result()

```
Creating plots ...
```

### Ontix[¶](#ontix)

In \[13\]:

Copied!

```
import os
import autoencodix as acx
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.configs.ontix_config import OntixConfig

from autoencodix.utils.example_data import (
    EXAMPLE_PROCESSED_DATA,
)

# EXAMPLE_DATA hold PyTorch Datasets (child with extra info) with metdata for train, test and valid splits
processed_data = EXAMPLE_PROCESSED_DATA


ont_lvl1 = dict()
ont_lvl2 = dict()

ont_lvl1["pwy-1"] = ["sub-pwy-1", "sub-pwy-2"]
ont_lvl1["pwy-2"] = ["sub-pwy-2"]
ont_lvl1["pwy-3"] = ["sub-pwy-1", "sub-pwy-3"]
# first third of feature ids in processed_data.train.feature_ids
ont_lvl2["sub-pwy-1"] = processed_data.train.feature_ids[
    : int(len(processed_data.train.feature_ids) / 3)
]
# second third of feature ids in processed_data.train.feature_ids
ont_lvl2["sub-pwy-2"] = processed_data.train.feature_ids[
    int(len(processed_data.train.feature_ids) / 3) : int(
        2 * len(processed_data.train.feature_ids) / 3
    )
]
# last third of feature ids in processed_data.train.feature_ids
ont_lvl2["sub-pwy-3"] = processed_data.train.feature_ids[
    int(2 * len(processed_data.train.feature_ids) / 3) : int(
        len(processed_data.train.feature_ids)
    )
]

# ont_lvl2["sub-pwy-1"] = ["gene-1", "gene-2"]
# ont_lvl2["sub-pwy-2"] = ["gene-3", "gene-4"]
# ont_lvl2["sub-pwy-3"] = ["gene-2", "gene-6"]

ontology_tuple = (ont_lvl1, ont_lvl2)

# Write each dictionary in ontology_tuple to a separate text file
os.makedirs("tutorial_res", exist_ok=True)
for i, ont_dict in enumerate(ontology_tuple):
    file_name = os.path.join("tutorial_res", f"ontology_level_{i + 1}.txt")
    with open(file_name, "w") as f:
        for key, values in ont_dict.items():
            for value in values:
                f.write(f"{value}\t{key}\n")
print("Ontology dictionaries written to tutorial_res/ontology_level_1.txt and tutorial_res/ontology_level_2.txt")

ont_files = [
    os.path.join("tutorial_res", "ontology_level_1.txt"),
    os.path.join("tutorial_res", "ontology_level_2.txt"),
]
```

import os import autoencodix as acx from autoencodix.configs.default_config import DefaultConfig from autoencodix.configs.ontix_config import OntixConfig from autoencodix.utils.example_data import ( EXAMPLE_PROCESSED_DATA, )

# EXAMPLE_DATA hold PyTorch Datasets (child with extra info) with metdata for train, test and valid splits

processed_data = EXAMPLE_PROCESSED_DATA ont_lvl1 = dict() ont_lvl2 = dict() ont_lvl1["pwy-1"] = ["sub-pwy-1", "sub-pwy-2"] ont_lvl1["pwy-2"] = ["sub-pwy-2"] ont_lvl1["pwy-3"] = ["sub-pwy-1", "sub-pwy-3"]

# first third of feature ids in processed_data.train.feature_ids

ont_lvl2["sub-pwy-1"] = processed_data.train.feature_ids[ : int(len(processed_data.train.feature_ids) / 3) ]

# second third of feature ids in processed_data.train.feature_ids

ont_lvl2["sub-pwy-2"] = processed_data.train.feature_ids[ int(len(processed_data.train.feature_ids) / 3) : int( 2 * len(processed_data.train.feature_ids) / 3 ) ]

# last third of feature ids in processed_data.train.feature_ids

ont_lvl2["sub-pwy-3"] = processed_data.train.feature_ids[ int(2 * len(processed_data.train.feature_ids) / 3) : int( len(processed_data.train.feature_ids) ) ]

# ont_lvl2["sub-pwy-1"] = ["gene-1", "gene-2"]

# ont_lvl2["sub-pwy-2"] = ["gene-3", "gene-4"]

# ont_lvl2["sub-pwy-3"] = ["gene-2", "gene-6"]

ontology_tuple = (ont_lvl1, ont_lvl2)

# Write each dictionary in ontology_tuple to a separate text file

os.makedirs("tutorial_res", exist_ok=True) for i, ont_dict in enumerate(ontology_tuple): file_name = os.path.join("tutorial_res", f"ontology_level\_{i + 1}.txt") with open(file_name, "w") as f: for key, values in ont_dict.items(): for value in values: f.write(f"{value}\\t{key}\\n") print("Ontology dictionaries written to tutorial_res/ontology_level_1.txt and tutorial_res/ontology_level_2.txt") ont_files = [ os.path.join("tutorial_res", "ontology_level_1.txt"), os.path.join("tutorial_res", "ontology_level_2.txt"), ]

```
Ontology dictionaries written to tutorial_res/ontology_level_1.txt and tutorial_res/ontology_level_2.txt
```

In \[14\]:

Copied!

```
ontix = acx.Ontix(
    ontologies=ont_files,
    sep="\t",
    config=OntixConfig(epochs=30, learning_rate=0.005, n_layers=1),
    data=processed_data,
)
result_onitx = ontix.run()
```

ontix = acx.Ontix( ontologies=ont_files, sep="\\t", config=OntixConfig(epochs=30, learning_rate=0.005, n_layers=1), data=processed_data, ) result_onitx = ontix.run()

```
Ontix checks:
All possible feature names length: 30
Feature order length: 30
Feature names without filtering: 30
Mask layer 0 with shape torch.Size([3, 3]) and 5.0 connections
Mask layer 1 with shape torch.Size([30, 3]) and 30.0 connections
Latent Dim: 3
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:305: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 38.5091 exceeded max norm of 1.
  warnings.warn(
```

```
Epoch 10 - Train Loss: 1089.6765
Sub-losses: recon_loss: 1089.1903, var_loss: 0.4861, anneal_factor: 0.0180, effective_beta_factor: 0.0180
Epoch 10 - Valid Loss: 1072.8932
Sub-losses: recon_loss: 1072.4681, var_loss: 0.4251, anneal_factor: 0.0180, effective_beta_factor: 0.0180
```

```
Epoch 20 - Train Loss: 953.5512
Sub-losses: recon_loss: 931.8715, var_loss: 21.6797, anneal_factor: 0.9350, effective_beta_factor: 0.9350
Epoch 20 - Valid Loss: 911.9373
Sub-losses: recon_loss: 891.3343, var_loss: 20.6030, anneal_factor: 0.9350, effective_beta_factor: 0.9350
```

```
Epoch 30 - Train Loss: 893.6976
Sub-losses: recon_loss: 874.0712, var_loss: 19.6264, anneal_factor: 0.9999, effective_beta_factor: 0.9999
Epoch 30 - Valid Loss: 858.3214
Sub-losses: recon_loss: 838.5320, var_loss: 19.7894, anneal_factor: 0.9999, effective_beta_factor: 0.9999
Processed 32 / 201 samples
Processed 64 / 201 samples
Processed 96 / 201 samples
Processed 128 / 201 samples
Processed 160 / 201 samples
Processed 192 / 201 samples
Processed 201 / 201 samples
```

In \[15\]:

Copied!

```
ontix.save(file_path=os.path.join("tutorial_res", "ontix_backup.pkl"))
```

ontix.save(file_path=os.path.join("tutorial_res", "ontix_backup.pkl"))

```
Preprocessor saved successfully.
```

```
saving memory efficient
Ontix checks:
All possible feature names length: 30
Feature order length: 30
Feature names without filtering: 30
Mask layer 0 with shape torch.Size([3, 3]) and 5.0 connections
Mask layer 1 with shape torch.Size([30, 3]) and 30.0 connections
Latent Dim: 3
Pipeline object saved successfully.
```

In \[16\]:

Copied!

```
ontix_loaded = acx.Ontix(ontologies=ont_files).load(file_path=os.path.join("tutorial_res", "ontix_backup.pkl"))
```

ontix_loaded = acx.Ontix(ontologies=ont_files).load(file_path=os.path.join("tutorial_res", "ontix_backup.pkl"))

```
Attempting to load a pipeline from tutorial_res/ontix_backup.pkl...
Pipeline object loaded successfully. Actual type: Ontix
Preprocessor loaded successfully.
```

In \[17\]:

Copied!

```
pred_res = ontix_loaded.predict(EXAMPLE_PROCESSED_DATA)
```

pred_res = ontix_loaded.predict(EXAMPLE_PROCESSED_DATA)

```
Processed 32 / 201 samples
Processed 64 / 201 samples
Processed 96 / 201 samples
Processed 128 / 201 samples
Processed 160 / 201 samples
Processed 192 / 201 samples
Processed 201 / 201 samples
```

In \[18\]:

Copied!

```
ontix_loaded.visualize()

ontix_loaded.evaluate(params=["cluster"])
```

ontix_loaded.visualize() ontix_loaded.evaluate(params=["cluster"])

```
Perform ML task with feature df: Latent
Latent
Perform ML task for target parameter: cluster
Perform ML task with feature df: Latent
Latent
Perform ML task for target parameter: cluster
```

```
.venv/lib/python3.12/site-packages/sklearn/linear_model/_logistic.py:451: OptimizeWarning: Unknown solver options: iprint
  opt_res = optimize.minimize(
.venv/lib/python3.12/site-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
.venv/lib/python3.12/site-packages/sklearn/linear_model/_logistic.py:451: OptimizeWarning: Unknown solver options: iprint
  opt_res = optimi
... [output truncated]
```

Out\[18\]:

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
model: OntixArchitecture
model_checkpoints: TrainingDynamics object
class_means: TrainingDynamics object
datasets: DatasetContainer(train=<autoencodix.data._numeric_dataset.NumericDataset object at 0x7959dd11b890>, valid=<autoencodix.data._numeric_dataset.NumericDataset object at 0x795797b7e630>, test=<
... [output truncated]
```

In \[19\]:

Copied!

```
ontix_loaded.show_result()
```

ontix_loaded.show_result()

```
Creating plots ...
```
