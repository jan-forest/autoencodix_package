> **Note:** This environment variable is required for fully deterministic CuBLAS ops on CUDA >= 10.2 when `reproducible=True` is set below. Without it, PyTorch raises a `RuntimeError` instead of training deterministically. It must be set before `torch` is imported. See the [README FAQ](https://jan-forest.github.io/autoencodix_package/tutorials/README.md#reproducibility-and-cublas_workspace_config) for details.

In \[ \]:

Copied!

```
%env CUBLAS_WORKSPACE_CONFIG=:16:8
```

%env CUBLAS_WORKSPACE_CONFIG=:16:8

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

**IMPORTANT**

> When calling `.save`, the `result` attribute is modified in-place (to avoid copies).\
> This means after calling `.save`, your current pipeline object does not contain information from the other result objects.\
> This applies to the pipeline object in current memory, not only to the pipeline object after loading.\
> If you want to avoid this, please run `.save(save_all=True)`.

## Practical Examples[¶](#practical-examples)

### First we run a varix pipeline[¶](#first-we-run-a-varix-pipeline)

In \[1\]:

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
Processing valid split
Processing valid modality: multi_sc
Processing test split
Processing test modality: multi_sc
```

```
OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
```

```
Epoch 1 - Train Loss: 783.8270
Sub-losses: recon_loss: 783.8269, var_loss: 0.0000, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 1 - Valid Loss: 674.2708
Sub-losses: recon_loss: 674.2707, var_loss: 0.0001, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 2 - Train Loss: 699.6010
Sub-losses: recon_loss: 699.6008, var_loss: 0.0002, anneal_factor: 0.0001, effective_beta_factor: 0.0000
Epoch 2 - Valid Loss: 652.1881
Sub-losses: recon_loss: 652.1878, var_loss: 0.0003, anneal_factor: 0.0001, effective_beta_factor: 0.0000
Epoch 3 - Train Loss: 656.4180
Sub-losses: recon_loss: 656.4168, var_loss: 0.0012, anneal_factor: 0.0002, effective_beta_factor: 0.0000
Epoch 3 - Valid Loss: 625.0739
Sub-losses: recon_loss: 625.0734, var_loss: 0.0006, anneal_factor: 0.0002, effective_bet
... [output truncated]
```

#### Saving and Loading Explained[¶](#saving-and-loading-explained)

Here we first check the latent space (could be any other training dynamic) before saving.\
After saving, we see that the `result` object got cleaned and only `model`, `adata_latent`, and `embedding_evaluation` are kept.

In \[2\]:

Copied!

```
ls_before_save = result.latentspaces.get(epoch=-1, split="test")
print("Length of latentspace before saving")
print(len(ls_before_save))
varix.save("varix_backup.pkl")
ls_after_save = result.latentspaces.get()
print("Length of latentspace after saving")
print(len(ls_after_save))
```

ls_before_save = result.latentspaces.get(epoch=-1, split="test") print("Length of latentspace before saving") print(len(ls_before_save)) varix.save("varix_backup.pkl") ls_after_save = result.latentspaces.get() print("Length of latentspace after saving") print(len(ls_after_save))

```
Length of latentspace before saving
200
Preprocessor saved successfully.
saving memory efficient
Pipeline object saved successfully.
Length of latentspace after saving
0
```

#### Implications for Visualization[¶](#implications-for-visualization)

We load the model and see the implications of this in `visualize`.

In \[3\]:

Copied!

```
varix_loaded = acx.Varix().load("varix_backup.pkl")
```

varix_loaded = acx.Varix().load("varix_backup.pkl")

```
Attempting to load a pipeline from ./varix_backup.pkl...
Pipeline object loaded successfully. Actual type: Varix
Preprocessor loaded successfully.
```

In \[4\]:

Copied!

```
varix_loaded.visualize()
```

varix_loaded.visualize()

```
<Figure size 0x500 with 0 Axes>
```

> Note that we code a `UserWarning`. This is expected, since it tells us that we cannot visualize loss plots anymore because the data for this is no longer in the result object.\
> However, we could run a predict step on the trained model to get some visualizations.

In \[5\]:

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
n_samples in format recon: 1000
train
n_samples from datatpackge: {'paired_count': 1000}
```

In \[6\]:

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

```
<Figure size 0x500 with 0 Axes>
```

#### Implications for Evaluate[¶](#implications-for-evaluate)

The evaluate step is not supported for this memory-efficient saving as of now, and you'll get the following warning and error:

In \[7\]:

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

## How to Keep All Data[¶](#how-to-keep-all-data)

When you really want to investigate all training dynamics and results, we recommend setting `save_all=True`. See below:

In \[8\]:

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
n_varix.save(file_path="new_varix.pkl", save_all=True)
```

from autoencodix.utils.example_data import EXAMPLE_MULTI_BULK my_config = VarixConfig( learning_rate=0.001, epochs=30, checkpoint_interval=5, default_vae_loss="kl", # kl or mmd possible data_case=DataCase.MULTI_BULK, ) n_varix = acx.Varix(data=EXAMPLE_MULTI_BULK, config=my_config) result_new = n_varix.run() n_varix.save(file_path="new_varix.pkl", save_all=True)

```
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
anno key: transcriptomics
anno key: proteomics
Epoch 1 - Train Loss: 214.2306
Sub-losses: recon_loss: 214.2306, var_loss: 0.0000, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 1 - Valid Loss: 185.5984
Sub-losses: recon_loss: 185.5983, var_loss: 0.0000, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 2 - Train Loss: 203.4312
Sub-losses: recon_loss: 203.4311, var_loss: 0.0001, anneal_factor: 0.0001, effective_beta_factor: 0.0000
Epoch 2 - Valid Loss: 187.5715
Sub-losses: recon_loss: 187.5715, var_loss: 0.0000, anneal_factor: 0.0001, effective_beta_factor: 0.0000
Epoch 3 - Train Loss: 197.6002
Sub-losses: recon_loss: 197.6001, var_loss: 0.0001, anneal_factor: 0.0002, effective_bet
... [output truncated]
```

Now we can load the pipeline and all intermediate data is kept.

In \[9\]:

Copied!

```
loaded_all_varix = acx.Varix().load(file_path="new_varix.pkl")
```

loaded_all_varix = acx.Varix().load(file_path="new_varix.pkl")

```
Attempting to load a pipeline from ./new_varix.pkl...
Pipeline object loaded successfully. Actual type: Varix
Preprocessor loaded successfully.
```

In \[10\]:

Copied!

```
loaded_all_varix.visualize()
```

loaded_all_varix.visualize()

In \[11\]:

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

In \[ \]:

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
    data_config=DataConfig(
        annotation_columns=["CANCER_TYPE_ACRONYM"],
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

clin_df = pd.read_parquet(clin_path) mapping_df = pd.read_csv(mapping_path, sep="\\t", index_col="sample_ids") clin_df["img_paths"] = mapping_df["img_paths"] clin_df.to_parquet(os.path.join(xmod_data_root, "combined_clin_formatted.parquet")) images_zip = hf_hub_download( repo_id=HF_REPO_ID, repo_type="dataset", filename="xmodalix/tcga_fake_images.zip" ) images_extract_dir = os.path.join(xmod_data_root, "images") if not os.path.isdir(os.path.join(images_extract_dir, "tcga_fake")): with zipfile.ZipFile(images_zip) as zf: zf.extractall(images_extract_dir) clin_file = os.path.join("data/XModalix-Tut-data/combined_clin_formatted.parquet") rna_file = os.path.join("data/XModalix-Tut-data/combined_rnaseq_formatted.parquet") img_root = os.path.join("data/XModalix-Tut-data/images/tcga_fake") xmodalix_config = XModalixConfig( checkpoint_interval=5, class_param="CANCER_TYPE_ACRONYM", epochs=30, latent_dim=8, requires_paired=False, pretrain_epochs=2, data_case=DataCase.IMG_TO_BULK, data_config=DataConfig( annotation_columns=["CANCER_TYPE_ACRONYM"], data_info={ "img": DataInfo( file_path=img_root, data_type="IMG", scaling="MINMAX", translate_direction="to", pretrain_epochs=2,

# extra_anno_file=imganno_file,

), "rna": DataInfo( file_path=rna_file, data_type="NUMERIC", scaling="MINMAX", translate_direction="from", ), "anno": DataInfo(file_path=clin_file, data_type="ANNOTATION", sep="\\t"), }, ), ) xmodalix = acx.XModalix(config=xmodalix_config) result = xmodalix.run() outpath = os.path.join("tutorial_res", "xmodalix.pkl") xmodalix.save(file_path=outpath)

Now we can load the pipeline again

In \[13\]:

Copied!

```
xmodalix_loaded = acx.XModalix.load(outpath)
```

xmodalix_loaded = acx.XModalix.load(outpath)

```
Attempting to load a pipeline from tutorial_res/xmodalix.pkl...
Pipeline object loaded successfully. Actual type: XModalix
Preprocessor loaded successfully.
```

Since we used memory efficient saving, we did not save the data, thus we need to run preprocessing again, this will read and preprocess the files defined in the config.

In \[14\]:

Copied!

```
xmodalix_loaded.preprocess()
```

xmodalix_loaded.preprocess()

```
reading parquet: data/XModalix-Tut-data/combined_rnaseq_formatted.parquet
reading parquet: data/XModalix-Tut-data/combined_clin_formatted.parquet
Given image size is possible, rescaling images to: 64x64
Successfully loaded 3230 images for img
anno key: rna
anno key: img
Converting 2261 images to torch.float32 tensors...
Converting 646 images to torch.float32 tensors...
Converting 323 images to torch.float32 tensors...
key: train, type: <class 'dict'>
key: valid, type: <class 'dict'>
key: test, type: <class 'dict'>
```

No we can use the trained model to predict again:

In \[15\]:

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
  - Added 17450 source feature IDs to .uns
Finished processing latent results.
```

In \[16\]:

Copied!

```
xmodalix_loaded.visualize()
```

xmodalix_loaded.visualize()

In \[17\]:

Copied!

```
xmodalix_loaded.show_result()
```

xmodalix_loaded.show_result()

```
Creating plots ...
Absolute loss plot not found in the plots dictionary
This happens, when you did not run visualize() or if you saved and loaded the model with `save_all=False`
```

### Ontix[¶](#ontix)

In \[18\]:

Copied!

```
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
for i, ont_dict in enumerate(ontology_tuple):
    file_name = f"ontology_level_{i + 1}.txt"
    with open(file_name, "w") as f:
        for key, values in ont_dict.items():
            for value in values:
                f.write(f"{value}\t{key}\n")
print("Ontology dictionaries written to ontology_level_1.txt and ontology_level_2.txt")

ont_files = ["ontology_level_1.txt", "ontology_level_2.txt"]
```

import autoencodix as acx from autoencodix.configs.default_config import DefaultConfig from autoencodix.configs.ontix_config import OntixConfig from autoencodix.utils.example_data import ( EXAMPLE_PROCESSED_DATA, )

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

for i, ont_dict in enumerate(ontology_tuple): file_name = f"ontology_level\_{i + 1}.txt" with open(file_name, "w") as f: for key, values in ont_dict.items(): for value in values: f.write(f"{value}\\t{key}\\n") print("Ontology dictionaries written to ontology_level_1.txt and ontology_level_2.txt") ont_files = ["ontology_level_1.txt", "ontology_level_2.txt"]

```
Ontology dictionaries written to ontology_level_1.txt and ontology_level_2.txt
```

In \[19\]:

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
Epoch 1 - Train Loss: 1221.0144
Sub-losses: recon_loss: 1221.0143, var_loss: 0.0001, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 1 - Valid Loss: 1201.5706
Sub-losses: recon_loss: 1201.5705, var_loss: 0.0002, anneal_factor: 0.0000, effective_beta_factor: 0.0000
Epoch 2 - Train Loss: 1197.5059
Sub-losses: recon_loss: 1197.5055, var_loss: 0.0004, anneal_factor: 0.0001, effective_beta_factor: 0.0001
Epoch 2 - Valid Loss: 1176.2030
Sub-losses: recon_loss: 1176.2024, var_loss: 0.0006, anneal_factor: 0.0001, effective_beta_factor:
... [output truncated]
```

In \[20\]:

Copied!

```
ontix.save(file_path="ontix_backup.pkl")
```

ontix.save(file_path="ontix_backup.pkl")

```
Preprocessor saved successfully.
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

In \[21\]:

Copied!

```
ontix_loaded = acx.Ontix(ontologies=ont_files).load(file_path="ontix_backup.pkl")
```

ontix_loaded = acx.Ontix(ontologies=ont_files).load(file_path="ontix_backup.pkl")

```
Attempting to load a pipeline from ./ontix_backup.pkl...
Pipeline object loaded successfully. Actual type: Ontix
Preprocessor loaded successfully.
```

In \[22\]:

Copied!

```
pred_res = ontix_loaded.predict(EXAMPLE_PROCESSED_DATA)
```

pred_res = ontix_loaded.predict(EXAMPLE_PROCESSED_DATA)

In \[23\]:

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

Out\[23\]:

```
Result Object Public Attributes:
------------------------------
latentspaces: TrainingDynamics object
sample_ids: TrainingDynamics object
reconstructions: TrainingDynamics object
mus: TrainingDynamics object
sigmas: TrainingDynamics object
losses: TrainingDynamics object
sub_losses: LossRegistry(_losses={})
preprocessed_data: Tensor of shape (0,)
model: OntixArchitecture
model_checkpoints: TrainingDynamics object
datasets: DatasetContainer(train=None, valid=None, test=None)
new_datasets: DatasetContainer(train=<autoencodix.data._numeric_dataset.NumericDataset object at 0x30eb85ab0>, valid=<autoencodix.data._numeric_dataset.NumericDataset object at 0x30e8bf8b0>, test=<autoencodix.data._numeric_dataset.NumericDataset object at 0x30e8bf670>)
adata_latent: AnnData object with n_obs × n_vars =
... [output truncated]
```

In \[24\]:

Copied!

```
ontix_loaded.show_result()
```

ontix_loaded.show_result()

```
Creating plots ...
```

```
<Figure size 0x500 with 0 Axes>
```
