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

# How to Prepare Your Data for AUTOENCODIX[¶](#how-to-prepare-your-data-for-autoencodix)

In this tutorial we'll show how `AUTOENCODIX` deals with different types of data and the different ways to pass data into the pipeline.

**IMPORTANT**

> This tutorial mainly explains the types of data we allow and how to format the data so that AUTOENCODIX can work with it. If you're unfamiliar with general concepts,\
> we recommend following the `Getting Started - Vanillix` Tutorial first.

## What You'll Learn[¶](#what-youll-learn)

We'll cover:

- Our general data logic
- How to pass data from files for different data modalities
- How to pass data directly into our pipeline
- Differences between paired and unpaired data
- How our preprocessing works and how to skip it

## Data Model Theory[¶](#data-model-theory)

There are `two main ways` to pass data to `AUTOENCODIX` and we support `three main data modalities`. For these main modalities, we offer `seven main use cases`. Besides that, you can choose whether we should preprocess the data or you want to start with training the models directly.

The `two main ways` to pass data are:

- Providing a file path in the config
- Passing the data directly to our pipeline

The `three main data modalities` are:

- Tabular numeric data
- Sparse numeric data (Single Cell)
- Image data

The `seven main use cases` are:

- Combining multi-omics data from single cell sequencing
- Combining multi-omics data from bulk sequencing (e.g., mRNA and methylation)
- "Translating" between bulk sequencing data
- "Translating" bulk and image data, and vice versa
- "Translating" between single cell data
- Training an image autoencoder

We encode these cases in our `DataCase` class, and you need to specify in the config which `DataCase` you have. This is shown in the code in the next section.

The general way to feed data to our pipeline is independent of the specific model (Varix, XModalix, etc.). There might be minor pipeline-specific requirements; for example, for `Ontix` we won't cover these here—please refer to the tutorials of the specific pipeline. Here we use `XModalix` and `Varix` as examples.

## 1) Passing Data from File[¶](#1-passing-data-from-file)

Depending on which pipeline you use, you will be interested in one of our `seven main use cases` as described above.

Regardless of the use case, the principle of passing data is always the same; only your file types will differ, and you need to specify the use case in the config via the `DataCase`:

```
from enum import Enum

class DataCase(str, Enum):
    MULTI_SINGLE_CELL = "Multi Single Cell"
    MULTI_BULK = "Multi Bulk"
    BULK_TO_BULK = "Bulk<->Bulk"
    IMG_TO_BULK = "IMG<->Bulk"
    SINGLE_CELL_TO_SINGLE_CELL = "Single Cell<->Single Cell"
    SINGLE_CELL_TO_IMG = "Single Cell<->IMG"
    IMG_TO_IMG = "IMG<->IMG"  # standard image autoencoder
```

#### 1.1 Combining Multi-Omics Data from Bulk-Sequencing[¶](#11-combining-multi-omics-data-from-bulk-sequencing)

First, we need to prepare our config object. We can either (a) directly provide a Python object, or (b) provide a YAML file. Here, we show option (a). For option (b), refer to [1].

**ATTENTION:**\
If you use `.txt` or `.csv` files, it is best practice to add the `sep` parameter. If none is given, the reader will try to auto-detect the separator, which is error-prone. This would look like:

**IMPORTANT:**

> For all your bulk data files, we expect the first column to be some kind of unique sample ID. Please prepare the data accordingly.

[1] [`Tutorials/DeepDives/ConfigTutorial.ipynb`](https://jan-forest.github.io/autoencodix_package/tutorials/DeepDives/InputDataTutorials/ConfigTutorial.ipynb)

**First make sure we are in the root of our package:**

**Now we can prepare the config:**

> Besides the actual data files, we expect one `annotation` file with metadata. We add the parameter `data_type` and set it to `ANNOTATION`, as shown in the code below.\
> If you have columns inside the `ANNOTATION` data that you want to use later in downstream visualization or evaluation tasks, please specify them with the `annotation_columns` parameter of the `DataConfig`. This will prevent errors later, as NaNs will be handled automatically during preprocessing for these columns.

#### ❗❗ Requirements: Getting Tutorial Data ❗❗[¶](#requirements-getting-tutorial-data)

The data for this example is hosted on Hugging Face Hub ([autoencodix/tcga](https://huggingface.co/datasets/autoencodix/tcga)) and is downloaded automatically in the cell below.

In \[4\]:

Copied!

```
import autoencodix as acx
from autoencodix.configs.default_config import DataConfig, DataInfo, DataCase
from autoencodix.configs import VarixConfig
from huggingface_hub import hf_hub_download

# Data is hosted on Hugging Face Hub and downloaded (and locally cached) automatically
HF_REPO_ID = "autoencodix/tcga"
mrna_file = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="rna.parquet")
meth_file = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="methylation.parquet")
clin_file = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="clinical.parquet")

"""
Our config class has the attribute `data_config`.
Here, we fill the `data_info` Dict, by defining names for our data modalities ("RNA" for example).
The value of this Dict is a instance of the `DataInfo` class. All we need to pass here is a `file_path` argument.
"""
data_config = DataConfig(
    annotation_columns=["CANCER_TYPE"],
    data_info={
        "RNA": DataInfo(file_path=mrna_file),
        "METHYLATION": DataInfo(file_path=meth_file),
        "CLINICAL": DataInfo(
            file_path=clin_file, data_type="ANNOTATION"
        ),
    },
)

"""
Lastly, we pass our data_config to the VarixConfig. Remember to pass the appropriate `DataCase`.
"""
bulk_config = VarixConfig(data_config=data_config, data_case=DataCase.MULTI_BULK, epochs=30, checkpoint_interval=10, k_filter=100)
varix = acx.Varix(config=bulk_config)

result_varix = varix.run()
varix.show_result()
```

import autoencodix as acx from autoencodix.configs.default_config import DataConfig, DataInfo, DataCase from autoencodix.configs import VarixConfig from huggingface_hub import hf_hub_download

# Data is hosted on Hugging Face Hub and downloaded (and locally cached) automatically

HF_REPO_ID = "autoencodix/tcga" mrna_file = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="rna.parquet") meth_file = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="methylation.parquet") clin_file = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="clinical.parquet") """ Our config class has the attribute `data_config`. Here, we fill the `data_info` Dict, by defining names for our data modalities ("RNA" for example). The value of this Dict is a instance of the `DataInfo` class. All we need to pass here is a `file_path` argument. """ data_config = DataConfig( annotation_columns=["CANCER_TYPE"], data_info={ "RNA": DataInfo(file_path=mrna_file), "METHYLATION": DataInfo(file_path=meth_file), "CLINICAL": DataInfo( file_path=clin_file, data_type="ANNOTATION" ), }, ) """ Lastly, we pass our data_config to the VarixConfig. Remember to pass the appropriate `DataCase`. """ bulk_config = VarixConfig(data_config=data_config, data_case=DataCase.MULTI_BULK, epochs=30, checkpoint_interval=10, k_filter=100) varix = acx.Varix(config=bulk_config) result_varix = varix.run() varix.show_result()

```
/home/ewald/Github/autoencodix_package/src/autoencodix/configs/default_config.py:439: UserWarning: annotation_columns in DataConfig is deprecated. Please set it directly in DefaultConfig instead.
  warnings.warn(
```

```
reading parquet: /home/ewald/.cache/huggingface/hub/datasets--autoencodix--tcga/snapshots/c59ee679d11ea17c0fd615248a382ef1c38ef456/rna.parquet
reading parquet: /home/ewald/.cache/huggingface/hub/datasets--autoencodix--tcga/snapshots/c59ee679d11ea17c0fd615248a382ef1c38ef456/methylation.parquet
reading parquet: /home/ewald/.cache/huggingface/hub/datasets--autoencodix--tcga/snapshots/c59ee679d11ea17c0fd615248a382ef1c38ef456/clinical.parquet
anno key: paired
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:250: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 21.2159 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 10 - Train Loss: 68.6068
Sub-losses: recon_loss: 68.4236, var_loss: 0.1833, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Epoch 10 - Valid Loss: 60.8824
Sub-losses: recon_loss: 60.7143, var_loss: 0.1681, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Epoch 20 - Train Loss: 68.2652
Sub-losses: recon_loss: 65.2557, var_loss: 3.0095, anneal_factor: 0.9350, effective_beta_factor: 0.0935
Epoch 20 - Valid Loss: 60.4574
Sub-losses: recon_loss: 57.6298, var_loss: 2.8276, anneal_factor: 0.9350, effective_beta_factor: 0.0935
Epoch 30 - Train Loss: 66.6436
Sub-losses: recon_loss: 63.7218, var_loss: 2.9218, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Epoch 30 - Valid Loss: 59.4852
Sub-losses: recon_loss: 57.0109, var_loss: 2.4743, anneal_factor: 0.9999, effective_beta_fact
... [output truncated]
```

#### 1.1.1 All Options for Data Specifc Parameters[¶](#111-all-options-for-data-specifc-parameters)

Inside the `DataInfo` object we set the file_path, but there are way more custom data parameters that we can set. Here's a complete list of all parameters you can set in the `DataInfo` class:

##### **DataInfo General**[¶](#datainfo-general)

| **Parameter**     | **Type**                                                              | **Description**                                                 |
| ----------------- | --------------------------------------------------------------------- | --------------------------------------------------------------- |
| `file_path`       | `str`                                                                 | Path to the raw data file                                       |
| `data_type`       | `Literal['NUMERIC', 'CATEGORICAL', 'IMG', 'ANNOTATION']`              | Type of data modality                                           |
| `scaling`         | `Literal['STANDARD', 'MINMAX', 'ROBUST', 'MAXABS', 'NONE', 'NOTSET']` | Overrides the globally set scaling method for this modality     |
| `filtering`       | `Literal['VAR', 'MAD', 'CORR', 'VARCORR', 'NOFILT', 'NONZEROVAR']`    | Feature filtering method                                        |
| `sep`             | `Optional[str]`                                                       | Delimiter for CSV/TSV input files (passed to `pandas.read_csv`) |
| `extra_anno_file` | `Optional[str]`                                                       | Path to an additional annotation file                           |

______________________________________________________________________

##### **Single-Cell Specific Parameters**[¶](#single-cell-specific-parameters)

| **Parameter**      | **Type**        | **Description**                                                                             |
| ------------------ | --------------- | ------------------------------------------------------------------------------------------- |
| `is_single_cell`   | `bool`          | Whether the dataset represents single-cell data                                             |
| `min_cells`        | `float`         | Minimum fraction of cells in which a gene must be expressed to be kept (filters rare genes) |
| `min_genes`        | `float`         | Minimum fraction of genes a cell must express to be kept (filters low-quality cells)        |
| `selected_layers`  | `List[str]`     | Layers to include from the single-cell dataset; must always include `"X"`                   |
| `is_X`             | `bool`          | Whether the data originates from the `"X"` matrix only                                      |
| `normalize_counts` | `bool`          | Whether to normalize single-cell counts by total expression per cell                        |
| `log_transform`    | `bool`          | Whether to apply `log1p` transformation after normalization                                 |
| `k_filter`         | `Optional[int]` | Automatically set based on global config; do not override manually                          |

______________________________________________________________________

###### **Image-Specific Parameters**[¶](#image-specific-parameters)

| **Parameter**       | **Type**        | **Description**                                     |
| ------------------- | --------------- | --------------------------------------------------- |
| `img_width_resize`  | `Optional[int]` | Target width for image resizing (must equal height) |
| `img_height_resize` | `Optional[int]` | Target height for image resizing (must equal width) |

______________________________________________________________________

###### **XModalix & Translation Parameters**[¶](#xmodalix-translation-parameters)

| **Parameter**         | **Type**                          | **Description**                                                                               |
| --------------------- | --------------------------------- | --------------------------------------------------------------------------------------------- |
| `translate_direction` | `Optional[Literal['from', 'to']]` | Defines translation direction in cross-modal (XModalix) training                              |
| `pretrain_epochs`     | `int`                             | Number of pretraining epochs specific to this modality (overrides global pretraining setting) |

#### 1.2 Use Single Cell Data[¶](#12-use-single-cell-data)

This works analogously to bulk/tabular data. You will most likely pass `h5ad` files instead of parquet or CSV files.\
For single cell data, you typically use only one data modality, so the term `MULTI_SC` in our `DataCase` may be a bit misleading; think of it as supporting 1–n modalities. Also, you most likely won't have a separate `annotation` file, because this information is already encoded in the `h5ad` file.

#### ❗❗ Requirements: Getting Tutorial Data ❗❗[¶](#requirements-getting-tutorial-data)

The data for this example is hosted on Hugging Face Hub ([autoencodix/pancreas-fang](https://huggingface.co/datasets/autoencodix/pancreas-fang)) and is downloaded automatically in the cell below.

**Note:**

> It is best practice to add the `is_single_cell` parameter to the `DataInfo` of the modality.

In \[1\]:

Copied!

```
import autoencodix as acx
from autoencodix.configs.default_config import DataConfig, DataInfo, DataCase
from autoencodix.configs import VarixConfig
from huggingface_hub import hf_hub_download

filepath = hf_hub_download(repo_id="autoencodix/pancreas-fang", repo_type="dataset", filename="pancreas.h5ad")
data_config = DataConfig(
    annotation_columns=["assigned_cluster"],
    data_info={"SC": DataInfo(file_path=filepath, is_single_cell=True)},
)
sc_config = VarixConfig(data_config=data_config, data_case=DataCase.MULTI_SINGLE_CELL, k_filter=100, epochs=30, checkpoint_interval=10)
varix_sc = acx.Varix(config=sc_config)
result_sc = varix_sc.run()
```

import autoencodix as acx from autoencodix.configs.default_config import DataConfig, DataInfo, DataCase from autoencodix.configs import VarixConfig from huggingface_hub import hf_hub_download filepath = hf_hub_download(repo_id="autoencodix/pancreas-fang", repo_type="dataset", filename="pancreas.h5ad") data_config = DataConfig( annotation_columns=["assigned_cluster"], data_info={"SC": DataInfo(file_path=filepath, is_single_cell=True)}, ) sc_config = VarixConfig(data_config=data_config, data_case=DataCase.MULTI_SINGLE_CELL, k_filter=100, epochs=30, checkpoint_interval=10) varix_sc = acx.Varix(config=sc_config) result_sc = varix_sc.run()

```
/home/ewald/Github/autoencodix_package/src/autoencodix/configs/default_config.py:439: UserWarning: annotation_columns in DataConfig is deprecated. Please set it directly in DefaultConfig instead.
  warnings.warn(
```

```
Number of common cells: 8569
mudata: View of MuData object with n_obs × n_vars = 8569 × 20125
  obs:	'barcode', 'assigned_cluster', 'sample_id'
  1 modality
    SC:	8569 x 20125
      obs:	'barcode', 'assigned_cluster', 'sample_id'
Processing 1 MuData objects: ['multi_sc']
Processing train modality: multi_sc
Processing valid split
Processing valid modality: multi_sc
Processing test split
Processing test modality: multi_sc
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:250: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 17.0524 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 10 - Train Loss: 60.9338
Sub-losses: recon_loss: 60.4786, var_loss: 0.4553, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Epoch 10 - Valid Loss: 51.1281
Sub-losses: recon_loss: 50.6877, var_loss: 0.4404, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Epoch 20 - Train Loss: 62.9274
Sub-losses: recon_loss: 59.8972, var_loss: 3.0302, anneal_factor: 0.9350, effective_beta_factor: 0.0935
Epoch 20 - Valid Loss: 52.7536
Sub-losses: recon_loss: 49.8155, var_loss: 2.9380, anneal_factor: 0.9350, effective_beta_factor: 0.0935
Epoch 30 - Train Loss: 61.4364
Sub-losses: recon_loss: 58.4847, var_loss: 2.9518, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Epoch 30 - Valid Loss: 51.3027
Sub-losses: recon_loss: 48.4929, var_loss: 2.8098, anneal_factor: 0.9999, effective_beta_fact
... [output truncated]
```

**Special Case for Visualization**\
As explained before, we can define one or more columns to use for visualization. In our example, we chose the `assigned_cluster` column. Internally, the column gets prefixed with the name of the data modality defined in the `DataInfo` dict. Therefore, to visualize this column, we need to add the prefix to the column name, e.g., `SC:assigned_cluster`.

> This should be fixed now, allowing you to pass the column name without the prefix.

In \[2\]:

Copied!

```
varix_sc.show_result(params=["assigned_cluster"])
```

varix_sc.show_result(params=["assigned_cluster"])

```
Creating plots ...
```

#### 1.3 Working with Images[¶](#13-working-with-images)

Working with images is similar to the previous cases. The main differences are:

- You need to provide a **directory path** where the image files are located, instead of a path to a single file.
- You need to **match each image path to a `sample_id`**, so that we know which metadata belongs to which image.
  - This is done via the annotation file. The annotation file should have one column containing only the image file name (e.g., `image.jpg`, NOT `mydir/image.jpg`). The name of this column must be passed to the `img_path_col` parameter in the config.

We support the following image file extensions (case-insensitive):\
`".jpg", ".jpeg", ".png", ".tif", ".tiff"`\
Ensure the directory you specify contains files with these supported extensions.

Optionally, you can specify an **image size** (only square dimensions are allowed).

#### ❗❗ Requirements: Getting Tutorial Data 2❗❗[¶](#requirements-getting-tutorial-data-2)

The data for this example is hosted on Hugging Face Hub ([autoencodix/tcga](https://huggingface.co/datasets/autoencodix/tcga)) and is downloaded automatically in the cell below.

In \[1\]:

Copied!

```
from autoencodix.configs import XModalixConfig
import os
import autoencodix as acx
from autoencodix.configs.default_config import DataConfig, DataInfo, DataCase


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
    img_path_col="img_paths", # col in your annnotation file
    checkpoint_interval=5,
    class_param="CANCER_TYPE_ACRONYM",
    k_filter=100,
    epochs=30,
    requires_paired=False,
    pretrain_epochs=10,
    data_case=DataCase.IMG_TO_BULK,
    data_config=DataConfig(
        data_info={
            "img": DataInfo(
                file_path=img_root,
                data_type="IMG",
                scaling="MINMAX",
                translate_direction="to",  # For XModalix you need to specify "to" or "from"
                img_width_resize=32,
                img_height_resize=32,
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
```

from autoencodix.configs import XModalixConfig import os import autoencodix as acx from autoencodix.configs.default_config import DataConfig, DataInfo, DataCase

# ---------------------------------------------------------------------

# Data is hosted on Hugging Face Hub and downloaded (and locally cached)

# automatically, then placed under the paths used below

# ---------------------------------------------------------------------

import shutil import zipfile import pandas as pd from huggingface_hub import hf_hub_download HF_REPO_ID = "autoencodix/tcga" xmod_data_root = "data/XModalix-Tut-data" os.makedirs(xmod_data_root, exist_ok=True) rna_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="rna.parquet") shutil.copyfile(rna_path, os.path.join(xmod_data_root, "combined_rnaseq_formatted.parquet")) clin_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="clinical.parquet") mapping_path = hf_hub_download( repo_id=HF_REPO_ID, repo_type="dataset", filename="xmodalix/tcga_image_mappings.txt" )

# The base clinical file has no image column; the XModalix image mapping is a separate,

# per-architecture extension that we join in here (left join: not every sample has an image).

clin_df = pd.read_parquet(clin_path) mapping_df = pd.read_csv(mapping_path, sep="\\t", index_col="sample_ids") clin_df["img_paths"] = mapping_df["img_paths"] clin_df.to_parquet(os.path.join(xmod_data_root, "combined_clin_formatted.parquet")) images_zip = hf_hub_download( repo_id=HF_REPO_ID, repo_type="dataset", filename="xmodalix/tcga_fake_images.zip" ) images_extract_dir = os.path.join(xmod_data_root, "images") if not os.path.isdir(os.path.join(images_extract_dir, "tcga_fake")): with zipfile.ZipFile(images_zip) as zf: zf.extractall(images_extract_dir) clin_file = os.path.join("data/XModalix-Tut-data/combined_clin_formatted.parquet") rna_file = os.path.join("data/XModalix-Tut-data/combined_rnaseq_formatted.parquet") img_root = os.path.join("data/XModalix-Tut-data/images/tcga_fake") xmodalix_config = XModalixConfig( img_path_col="img_paths", # col in your annnotation file checkpoint_interval=5, class_param="CANCER_TYPE_ACRONYM", k_filter=100, epochs=30, requires_paired=False, pretrain_epochs=10, data_case=DataCase.IMG_TO_BULK, data_config=DataConfig( data_info={ "img": DataInfo( file_path=img_root, data_type="IMG", scaling="MINMAX", translate_direction="to", # For XModalix you need to specify "to" or "from" img_width_resize=32, img_height_resize=32,

# extra_anno_file=imganno_file,

), "rna": DataInfo( file_path=rna_file, data_type="NUMERIC", scaling="MINMAX", translate_direction="from", ), "anno": DataInfo(file_path=clin_file, data_type="ANNOTATION", sep="\\t"), }, ), ) xmodalix = acx.XModalix(config=xmodalix_config) result = xmodalix.run()

```
reading parquet: data/XModalix-Tut-data/combined_rnaseq_formatted.parquet
reading parquet: data/XModalix-Tut-data/combined_clin_formatted.parquet
Given image size is possible, rescaling images to: 32x32
Successfully loaded 3230 images for img
anno key: rna
anno key: img
Converting 2261 images to torch.float32 tensors...
Converting 646 images to torch.float32 tensors...
Converting 323 images to torch.float32 tensors...
key: train, type: <class 'dict'>
key: valid, type: <class 'dict'>
key: test, type: <class 'dict'>
Dataset has UNPAIRED samples → using CoverageEnsuringSampler (2261 paired + 225 unpaired)
Dataset has UNPAIRED samples → using CoverageEnsuringSampler (323 paired + 32 unpaired)
Check if we need to pretrain: multi_bulk.rna
pretrain epochs : 10
starting pretraining for: multi_bul
... [output truncated]
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:250: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 12.5916 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 5 - Train Loss: 0.5879
Sub-losses: recon_loss: 0.5771, var_loss: 0.0108, anneal_factor: 0.0007, effective_beta_factor: 0.0001
Epoch 5 - Valid Loss: 0.7038
Sub-losses: recon_loss: 0.5518, var_loss: 0.1520, anneal_factor: 0.0007, effective_beta_factor: 0.0001
Epoch 10 - Train Loss: 0.4741
Sub-losses: recon_loss: 0.4458, var_loss: 0.0282, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Epoch 10 - Valid Loss: 0.4253
Sub-losses: recon_loss: 0.3947, var_loss: 0.0306, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Check if we need to pretrain: img.img
pretrain epochs : 10
starting pretraining for: img.img with <class 'autoencodix.trainers._general_trainer.GeneralTrainer'>
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:250: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 82.4634 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 5 - Train Loss: 27.1822
Sub-losses: recon_loss: 27.1701, var_loss: 0.0121, anneal_factor: 0.0007, effective_beta_factor: 0.0001
Epoch 5 - Valid Loss: 25.2441
Sub-losses: recon_loss: 25.2298, var_loss: 0.0143, anneal_factor: 0.0007, effective_beta_factor: 0.0001
Epoch 10 - Train Loss: 20.2610
Sub-losses: recon_loss: 19.5585, var_loss: 0.7025, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Epoch 10 - Valid Loss: 19.1020
Sub-losses: recon_loss: 18.3307, var_loss: 0.7713, anneal_factor: 0.0180, effective_beta_factor: 0.0018
--- Epoch 1/30 ---
split: train, n_samples: 5765.0
Epoch 1/30 - Train Loss: 160.6849
Sub-losses - adver_loss: 55.9999, aggregated_sub_losses: 15.5742, paired_loss: 20.3146, class_loss: 68.7962, multi_bulk.rna.recon_loss: 0.5393, multi_bulk.rna.var_loss: 0.0001,
... [output truncated]
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 1. Total norm of gradients (adjusted for number of features): 8.3429 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 2/30 - Train Loss: 205.1911
Sub-losses - adver_loss: 100.3661, aggregated_sub_losses: 15.5742, paired_loss: 20.3194, class_loss: 68.9313, multi_bulk.rna.recon_loss: 0.5341, multi_bulk.rna.var_loss: 0.0001, multi_bulk.rna.loss: 0.5343, img.img.recon_loss: 15.0372, img.img.var_loss: 0.0027, img.img.loss: 15.0400, clf_loss: 0.0513
split: valid, n_samples: 323
Epoch 2/30 - Valid Loss: 218.6657
Sub-losses - adver_loss: 114.5413, aggregated_sub_losses: 14.7325, paired_loss: 20.4167, class_loss: 68.9751, multi_bulk.rna.recon_loss: 0.4792, multi_bulk.rna.var_loss: 0.0002, multi_bulk.rna.loss: 0.4794, img.img.recon_loss: 14.2504, img.img.var_loss: 0.0027, img.img.loss: 14.2532, clf_loss: 0.0183
--- Epoch 3/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 2. Total norm of gradients (adjusted for number of features): 6.1006 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 3/30 - Train Loss: 221.7280
Sub-losses - adver_loss: 117.0082, aggregated_sub_losses: 15.6420, paired_loss: 20.3398, class_loss: 68.7379, multi_bulk.rna.recon_loss: 0.5383, multi_bulk.rna.var_loss: 0.0003, multi_bulk.rna.loss: 0.5386, img.img.recon_loss: 15.0981, img.img.var_loss: 0.0053, img.img.loss: 15.1034, clf_loss: 0.0291
split: valid, n_samples: 323
Epoch 3/30 - Valid Loss: 229.9468
Sub-losses - adver_loss: 126.1234, aggregated_sub_losses: 14.7124, paired_loss: 20.4732, class_loss: 68.6377, multi_bulk.rna.recon_loss: 0.4801, multi_bulk.rna.var_loss: 0.0003, multi_bulk.rna.loss: 0.4804, img.img.recon_loss: 14.2266, img.img.var_loss: 0.0054, img.img.loss: 14.2320, clf_loss: 0.0209
--- Epoch 4/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 3. Total norm of gradients (adjusted for number of features): 7.5513 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 4/30 - Train Loss: 239.3957
Sub-losses - adver_loss: 134.7512, aggregated_sub_losses: 15.6616, paired_loss: 20.2884, class_loss: 68.6946, multi_bulk.rna.recon_loss: 0.5415, multi_bulk.rna.var_loss: 0.0005, multi_bulk.rna.loss: 0.5420, img.img.recon_loss: 15.1092, img.img.var_loss: 0.0104, img.img.loss: 15.1196, clf_loss: 0.0141
split: valid, n_samples: 323
Epoch 4/30 - Valid Loss: 251.5439
Sub-losses - adver_loss: 148.7990, aggregated_sub_losses: 14.8734, paired_loss: 20.0834, class_loss: 67.7882, multi_bulk.rna.recon_loss: 0.4893, multi_bulk.rna.var_loss: 0.0006, multi_bulk.rna.loss: 0.4899, img.img.recon_loss: 14.3733, img.img.var_loss: 0.0102, img.img.loss: 14.3835, clf_loss: 0.0107
--- Epoch 5/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 4. Total norm of gradients (adjusted for number of features): 7.7933 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 5/30 - Train Loss: 248.5907
Sub-losses - adver_loss: 143.6507, aggregated_sub_losses: 15.7524, paired_loss: 20.3334, class_loss: 68.8543, multi_bulk.rna.recon_loss: 0.5359, multi_bulk.rna.var_loss: 0.0010, multi_bulk.rna.loss: 0.5369, img.img.recon_loss: 15.1954, img.img.var_loss: 0.0201, img.img.loss: 15.2155, clf_loss: 0.0162
split: valid, n_samples: 323
Epoch 5/30 - Valid Loss: 255.8977
Sub-losses - adver_loss: 153.1569, aggregated_sub_losses: 14.6911, paired_loss: 20.1511, class_loss: 67.8986, multi_bulk.rna.recon_loss: 0.4865, multi_bulk.rna.var_loss: 0.0011, multi_bulk.rna.loss: 0.4876, img.img.recon_loss: 14.1837, img.img.var_loss: 0.0198, img.img.loss: 14.2035, clf_loss: 0.0027
Storing checkpoint for epoch 4...
--- Epoch 6/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 5. Total norm of gradients (adjusted for number of features): 7.1650 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 6/30 - Train Loss: 252.4860
Sub-losses - adver_loss: 147.6023, aggregated_sub_losses: 15.7572, paired_loss: 20.3229, class_loss: 68.8036, multi_bulk.rna.recon_loss: 0.5333, multi_bulk.rna.var_loss: 0.0020, multi_bulk.rna.loss: 0.5353, img.img.recon_loss: 15.1827, img.img.var_loss: 0.0392, img.img.loss: 15.2219, clf_loss: 0.0109
split: valid, n_samples: 323
Epoch 6/30 - Valid Loss: 255.7790
Sub-losses - adver_loss: 151.7353, aggregated_sub_losses: 14.6970, paired_loss: 20.4212, class_loss: 68.9255, multi_bulk.rna.recon_loss: 0.4973, multi_bulk.rna.var_loss: 0.0020, multi_bulk.rna.loss: 0.4993, img.img.recon_loss: 14.1580, img.img.var_loss: 0.0397, img.img.loss: 14.1977, clf_loss: 0.0059
--- Epoch 7/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 6. Total norm of gradients (adjusted for number of features): 6.7283 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 7/30 - Train Loss: 259.3877
Sub-losses - adver_loss: 154.5182, aggregated_sub_losses: 15.7702, paired_loss: 20.2834, class_loss: 68.8159, multi_bulk.rna.recon_loss: 0.5395, multi_bulk.rna.var_loss: 0.0039, multi_bulk.rna.loss: 0.5434, img.img.recon_loss: 15.1506, img.img.var_loss: 0.0763, img.img.loss: 15.2269, clf_loss: 0.0068
split: valid, n_samples: 323
Epoch 7/30 - Valid Loss: 258.6879
Sub-losses - adver_loss: 155.0229, aggregated_sub_losses: 14.7683, paired_loss: 20.4279, class_loss: 68.4687, multi_bulk.rna.recon_loss: 0.4847, multi_bulk.rna.var_loss: 0.0037, multi_bulk.rna.loss: 0.4884, img.img.recon_loss: 14.2027, img.img.var_loss: 0.0772, img.img.loss: 14.2799, clf_loss: 0.0057
--- Epoch 8/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 7. Total norm of gradients (adjusted for number of features): 12.5114 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 8/30 - Train Loss: 267.3811
Sub-losses - adver_loss: 162.5574, aggregated_sub_losses: 15.7098, paired_loss: 20.3145, class_loss: 68.7994, multi_bulk.rna.recon_loss: 0.5307, multi_bulk.rna.var_loss: 0.0074, multi_bulk.rna.loss: 0.5381, img.img.recon_loss: 15.0234, img.img.var_loss: 0.1483, img.img.loss: 15.1717, clf_loss: 0.0068
split: valid, n_samples: 323
Epoch 8/30 - Valid Loss: 259.0375
Sub-losses - adver_loss: 155.9933, aggregated_sub_losses: 14.8562, paired_loss: 20.1929, class_loss: 67.9951, multi_bulk.rna.recon_loss: 0.4917, multi_bulk.rna.var_loss: 0.0074, multi_bulk.rna.loss: 0.4992, img.img.recon_loss: 14.2098, img.img.var_loss: 0.1473, img.img.loss: 14.3571, clf_loss: 0.0234
--- Epoch 9/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 8. Total norm of gradients (adjusted for number of features): 6.4650 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 9/30 - Train Loss: 268.4097
Sub-losses - adver_loss: 163.4658, aggregated_sub_losses: 15.9236, paired_loss: 20.3031, class_loss: 68.7172, multi_bulk.rna.recon_loss: 0.5397, multi_bulk.rna.var_loss: 0.0144, multi_bulk.rna.loss: 0.5541, img.img.recon_loss: 15.0821, img.img.var_loss: 0.2874, img.img.loss: 15.3695, clf_loss: 0.0056
split: valid, n_samples: 323
Epoch 9/30 - Valid Loss: 269.9840
Sub-losses - adver_loss: 167.7535, aggregated_sub_losses: 15.1728, paired_loss: 19.8707, class_loss: 67.1869, multi_bulk.rna.recon_loss: 0.4736, multi_bulk.rna.var_loss: 0.0156, multi_bulk.rna.loss: 0.4892, img.img.recon_loss: 14.4066, img.img.var_loss: 0.2771, img.img.loss: 14.6837, clf_loss: 0.0045
--- Epoch 10/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 9. Total norm of gradients (adjusted for number of features): 8.4092 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 10/30 - Train Loss: 277.5313
Sub-losses - adver_loss: 172.2902, aggregated_sub_losses: 16.2071, paired_loss: 20.3018, class_loss: 68.7323, multi_bulk.rna.recon_loss: 0.5460, multi_bulk.rna.var_loss: 0.0278, multi_bulk.rna.loss: 0.5739, img.img.recon_loss: 15.0779, img.img.var_loss: 0.5553, img.img.loss: 15.6332, clf_loss: 0.0055
split: valid, n_samples: 323
Epoch 10/30 - Valid Loss: 272.7187
Sub-losses - adver_loss: 170.1818, aggregated_sub_losses: 15.4158, paired_loss: 19.8617, class_loss: 67.2593, multi_bulk.rna.recon_loss: 0.4918, multi_bulk.rna.var_loss: 0.0294, multi_bulk.rna.loss: 0.5212, img.img.recon_loss: 14.3704, img.img.var_loss: 0.5243, img.img.loss: 14.8946, clf_loss: 0.0109
Storing checkpoint for epoch 9...
--- Epoch 11/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 10. Total norm of gradients (adjusted for number of features): 10.8791 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 11/30 - Train Loss: 281.4610
Sub-losses - adver_loss: 175.6005, aggregated_sub_losses: 16.8073, paired_loss: 20.3117, class_loss: 68.7415, multi_bulk.rna.recon_loss: 0.5372, multi_bulk.rna.var_loss: 0.0540, multi_bulk.rna.loss: 0.5913, img.img.recon_loss: 15.1534, img.img.var_loss: 1.0626, img.img.loss: 16.2160, clf_loss: 0.0050
split: valid, n_samples: 323
Epoch 11/30 - Valid Loss: 290.1814
Sub-losses - adver_loss: 183.8156, aggregated_sub_losses: 15.7639, paired_loss: 20.7781, class_loss: 69.8239, multi_bulk.rna.recon_loss: 0.4792, multi_bulk.rna.var_loss: 0.0619, multi_bulk.rna.loss: 0.5411, img.img.recon_loss: 14.1082, img.img.var_loss: 1.1146, img.img.loss: 15.2228, clf_loss: 0.0039
--- Epoch 12/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 11. Total norm of gradients (adjusted for number of features): 7.7414 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 12/30 - Train Loss: 282.1764
Sub-losses - adver_loss: 175.5065, aggregated_sub_losses: 17.6982, paired_loss: 20.3047, class_loss: 68.6670, multi_bulk.rna.recon_loss: 0.5381, multi_bulk.rna.var_loss: 0.1015, multi_bulk.rna.loss: 0.6396, img.img.recon_loss: 15.0541, img.img.var_loss: 2.0044, img.img.loss: 17.0585, clf_loss: 0.0058
split: valid, n_samples: 323
Epoch 12/30 - Valid Loss: 276.4003
Sub-losses - adver_loss: 169.6548, aggregated_sub_losses: 16.8063, paired_loss: 20.6306, class_loss: 69.3087, multi_bulk.rna.recon_loss: 0.4876, multi_bulk.rna.var_loss: 0.1303, multi_bulk.rna.loss: 0.6178, img.img.recon_loss: 14.1056, img.img.var_loss: 2.0828, img.img.loss: 16.1884, clf_loss: 0.0060
--- Epoch 13/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 12. Total norm of gradients (adjusted for number of features): 9.0457 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 13/30 - Train Loss: 278.0572
Sub-losses - adver_loss: 169.5071, aggregated_sub_losses: 19.4415, paired_loss: 20.3158, class_loss: 68.7928, multi_bulk.rna.recon_loss: 0.5327, multi_bulk.rna.var_loss: 0.1852, multi_bulk.rna.loss: 0.7179, img.img.recon_loss: 15.0470, img.img.var_loss: 3.6767, img.img.loss: 18.7237, clf_loss: 0.0037
split: valid, n_samples: 323
Epoch 13/30 - Valid Loss: 294.4367
Sub-losses - adver_loss: 185.5339, aggregated_sub_losses: 18.6540, paired_loss: 20.8176, class_loss: 69.4312, multi_bulk.rna.recon_loss: 0.4858, multi_bulk.rna.var_loss: 0.1777, multi_bulk.rna.loss: 0.6635, img.img.recon_loss: 14.1413, img.img.var_loss: 3.8492, img.img.loss: 17.9906, clf_loss: 0.0005
--- Epoch 14/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 13. Total norm of gradients (adjusted for number of features): 8.7713 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 14/30 - Train Loss: 291.3857
Sub-losses - adver_loss: 180.1248, aggregated_sub_losses: 22.4017, paired_loss: 20.2883, class_loss: 68.5709, multi_bulk.rna.recon_loss: 0.5451, multi_bulk.rna.var_loss: 0.3209, multi_bulk.rna.loss: 0.8660, img.img.recon_loss: 15.1003, img.img.var_loss: 6.4354, img.img.loss: 21.5357, clf_loss: 0.0041
split: valid, n_samples: 323
Epoch 14/30 - Valid Loss: 312.7153
Sub-losses - adver_loss: 203.3838, aggregated_sub_losses: 21.4690, paired_loss: 20.0427, class_loss: 67.8198, multi_bulk.rna.recon_loss: 0.4933, multi_bulk.rna.var_loss: 0.3544, multi_bulk.rna.loss: 0.8477, img.img.recon_loss: 14.3602, img.img.var_loss: 6.2611, img.img.loss: 20.6213, clf_loss: 0.0014
--- Epoch 15/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 14. Total norm of gradients (adjusted for number of features): 9.5465 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 15/30 - Train Loss: 297.3779
Sub-losses - adver_loss: 181.5914, aggregated_sub_losses: 26.6587, paired_loss: 20.3193, class_loss: 68.8085, multi_bulk.rna.recon_loss: 0.5389, multi_bulk.rna.var_loss: 0.5261, multi_bulk.rna.loss: 1.0651, img.img.recon_loss: 15.1259, img.img.var_loss: 10.4677, img.img.loss: 25.5937, clf_loss: 0.0072
split: valid, n_samples: 323
Epoch 15/30 - Valid Loss: 293.8940
Sub-losses - adver_loss: 179.0335, aggregated_sub_losses: 25.8470, paired_loss: 20.3407, class_loss: 68.6728, multi_bulk.rna.recon_loss: 0.5037, multi_bulk.rna.var_loss: 0.5958, multi_bulk.rna.loss: 1.0995, img.img.recon_loss: 14.1753, img.img.var_loss: 10.5722, img.img.loss: 24.7475, clf_loss: 0.0040
Storing checkpoint for epoch 14...
--- Epoch 16/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 15. Total norm of gradients (adjusted for number of features): 9.7712 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 16/30 - Train Loss: 295.5848
Sub-losses - adver_loss: 174.6323, aggregated_sub_losses: 31.8286, paired_loss: 20.3017, class_loss: 68.8223, multi_bulk.rna.recon_loss: 0.5380, multi_bulk.rna.var_loss: 0.7839, multi_bulk.rna.loss: 1.3219, img.img.recon_loss: 15.0793, img.img.var_loss: 15.4274, img.img.loss: 30.5067, clf_loss: 0.0021
split: valid, n_samples: 323
Epoch 16/30 - Valid Loss: 302.2223
Sub-losses - adver_loss: 183.1870, aggregated_sub_losses: 30.8501, paired_loss: 20.2629, class_loss: 67.9223, multi_bulk.rna.recon_loss: 0.4877, multi_bulk.rna.var_loss: 0.8153, multi_bulk.rna.loss: 1.3030, img.img.recon_loss: 14.2410, img.img.var_loss: 15.3061, img.img.loss: 29.5471, clf_loss: 0.0004
--- Epoch 17/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 16. Total norm of gradients (adjusted for number of features): 12.9886 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 17/30 - Train Loss: 311.8745
Sub-losses - adver_loss: 185.4926, aggregated_sub_losses: 37.0322, paired_loss: 20.3320, class_loss: 69.0177, multi_bulk.rna.recon_loss: 0.5391, multi_bulk.rna.var_loss: 1.0282, multi_bulk.rna.loss: 1.5673, img.img.recon_loss: 15.0722, img.img.var_loss: 20.3927, img.img.loss: 35.4649, clf_loss: 0.0061
split: valid, n_samples: 323
Epoch 17/30 - Valid Loss: 306.6333
Sub-losses - adver_loss: 181.2490, aggregated_sub_losses: 36.2831, paired_loss: 20.3458, class_loss: 68.7554, multi_bulk.rna.recon_loss: 0.4857, multi_bulk.rna.var_loss: 1.0644, multi_bulk.rna.loss: 1.5501, img.img.recon_loss: 14.2638, img.img.var_loss: 20.4692, img.img.loss: 34.7330, clf_loss: 0.0013
--- Epoch 18/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 17. Total norm of gradients (adjusted for number of features): 7.4515 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 18/30 - Train Loss: 316.9872
Sub-losses - adver_loss: 186.5819, aggregated_sub_losses: 41.2990, paired_loss: 20.3344, class_loss: 68.7719, multi_bulk.rna.recon_loss: 0.5444, multi_bulk.rna.var_loss: 1.2332, multi_bulk.rna.loss: 1.7776, img.img.recon_loss: 15.1158, img.img.var_loss: 24.4055, img.img.loss: 39.5213, clf_loss: 0.0031
split: valid, n_samples: 323
Epoch 18/30 - Valid Loss: 318.1626
Sub-losses - adver_loss: 188.0127, aggregated_sub_losses: 41.0329, paired_loss: 20.4496, class_loss: 68.6674, multi_bulk.rna.recon_loss: 0.4711, multi_bulk.rna.var_loss: 1.3841, multi_bulk.rna.loss: 1.8552, img.img.recon_loss: 14.2654, img.img.var_loss: 24.9123, img.img.loss: 39.1777, clf_loss: 0.0004
--- Epoch 19/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 18. Total norm of gradients (adjusted for number of features): 9.4652 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 19/30 - Train Loss: 324.3129
Sub-losses - adver_loss: 190.9484, aggregated_sub_losses: 44.1295, paired_loss: 20.3398, class_loss: 68.8954, multi_bulk.rna.recon_loss: 0.5407, multi_bulk.rna.var_loss: 1.3561, multi_bulk.rna.loss: 1.8968, img.img.recon_loss: 15.0426, img.img.var_loss: 27.1900, img.img.loss: 42.2326, clf_loss: 0.0016
split: valid, n_samples: 323
Epoch 19/30 - Valid Loss: 322.3584
Sub-losses - adver_loss: 190.8427, aggregated_sub_losses: 43.1444, paired_loss: 20.1505, class_loss: 68.2207, multi_bulk.rna.recon_loss: 0.4756, multi_bulk.rna.var_loss: 1.5958, multi_bulk.rna.loss: 2.0714, img.img.recon_loss: 14.2664, img.img.var_loss: 26.8066, img.img.loss: 41.0730, clf_loss: 0.0004
--- Epoch 20/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 19. Total norm of gradients (adjusted for number of features): 11.5217 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 20/30 - Train Loss: 330.2791
Sub-losses - adver_loss: 195.1311, aggregated_sub_losses: 45.9420, paired_loss: 20.3346, class_loss: 68.8714, multi_bulk.rna.recon_loss: 0.5358, multi_bulk.rna.var_loss: 1.4505, multi_bulk.rna.loss: 1.9862, img.img.recon_loss: 15.0967, img.img.var_loss: 28.8590, img.img.loss: 43.9558, clf_loss: 0.0029
split: valid, n_samples: 323
Epoch 20/30 - Valid Loss: 346.7808
Sub-losses - adver_loss: 211.9340, aggregated_sub_losses: 45.3938, paired_loss: 20.4719, class_loss: 68.9812, multi_bulk.rna.recon_loss: 0.4850, multi_bulk.rna.var_loss: 1.4234, multi_bulk.rna.loss: 1.9084, img.img.recon_loss: 14.3378, img.img.var_loss: 29.1476, img.img.loss: 43.4854, clf_loss: 0.0003
Storing checkpoint for epoch 19...
--- Epoch 21/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 20. Total norm of gradients (adjusted for number of features): 13.3257 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 21/30 - Train Loss: 331.8548
Sub-losses - adver_loss: 195.7193, aggregated_sub_losses: 46.9146, paired_loss: 20.3623, class_loss: 68.8586, multi_bulk.rna.recon_loss: 0.5425, multi_bulk.rna.var_loss: 1.5029, multi_bulk.rna.loss: 2.0454, img.img.recon_loss: 15.0658, img.img.var_loss: 29.8035, img.img.loss: 44.8693, clf_loss: 0.0034
split: valid, n_samples: 323
Epoch 21/30 - Valid Loss: 329.3173
Sub-losses - adver_loss: 193.4308, aggregated_sub_losses: 46.5868, paired_loss: 20.3911, class_loss: 68.9086, multi_bulk.rna.recon_loss: 0.4823, multi_bulk.rna.var_loss: 1.6546, multi_bulk.rna.loss: 2.1369, img.img.recon_loss: 14.2981, img.img.var_loss: 30.1519, img.img.loss: 44.4499, clf_loss: 0.0139
--- Epoch 22/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 21. Total norm of gradients (adjusted for number of features): 9.4405 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 22/30 - Train Loss: 335.7301
Sub-losses - adver_loss: 199.1504, aggregated_sub_losses: 47.4819, paired_loss: 20.2999, class_loss: 68.7979, multi_bulk.rna.recon_loss: 0.5423, multi_bulk.rna.var_loss: 1.5174, multi_bulk.rna.loss: 2.0596, img.img.recon_loss: 15.1296, img.img.var_loss: 30.2927, img.img.loss: 45.4223, clf_loss: 0.0018
split: valid, n_samples: 323
Epoch 22/30 - Valid Loss: 339.6571
Sub-losses - adver_loss: 205.4392, aggregated_sub_losses: 46.0752, paired_loss: 20.2046, class_loss: 67.9380, multi_bulk.rna.recon_loss: 0.4950, multi_bulk.rna.var_loss: 1.7062, multi_bulk.rna.loss: 2.2012, img.img.recon_loss: 14.2401, img.img.var_loss: 29.6339, img.img.loss: 43.8740, clf_loss: 0.0054
--- Epoch 23/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 22. Total norm of gradients (adjusted for number of features): 8.9085 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 23/30 - Train Loss: 337.3739
Sub-losses - adver_loss: 200.5690, aggregated_sub_losses: 47.6947, paired_loss: 20.3195, class_loss: 68.7907, multi_bulk.rna.recon_loss: 0.5308, multi_bulk.rna.var_loss: 1.5545, multi_bulk.rna.loss: 2.0853, img.img.recon_loss: 15.0332, img.img.var_loss: 30.5762, img.img.loss: 45.6093, clf_loss: 0.0041
split: valid, n_samples: 323
Epoch 23/30 - Valid Loss: 337.4436
Sub-losses - adver_loss: 200.8550, aggregated_sub_losses: 47.2664, paired_loss: 20.4041, class_loss: 68.9182, multi_bulk.rna.recon_loss: 0.4802, multi_bulk.rna.var_loss: 1.8243, multi_bulk.rna.loss: 2.3045, img.img.recon_loss: 14.1708, img.img.var_loss: 30.7911, img.img.loss: 44.9619, clf_loss: 0.0024
--- Epoch 24/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 23. Total norm of gradients (adjusted for number of features): 10.1026 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 24/30 - Train Loss: 333.7203
Sub-losses - adver_loss: 196.7116, aggregated_sub_losses: 47.9649, paired_loss: 20.2735, class_loss: 68.7703, multi_bulk.rna.recon_loss: 0.5406, multi_bulk.rna.var_loss: 1.5476, multi_bulk.rna.loss: 2.0882, img.img.recon_loss: 15.1677, img.img.var_loss: 30.7091, img.img.loss: 45.8768, clf_loss: 0.0030
split: valid, n_samples: 323
Epoch 24/30 - Valid Loss: 338.9710
Sub-losses - adver_loss: 201.0202, aggregated_sub_losses: 47.7696, paired_loss: 20.5201, class_loss: 69.6610, multi_bulk.rna.recon_loss: 0.5233, multi_bulk.rna.var_loss: 1.7429, multi_bulk.rna.loss: 2.2662, img.img.recon_loss: 14.2982, img.img.var_loss: 31.2052, img.img.loss: 45.5034, clf_loss: 0.0003
--- Epoch 25/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 24. Total norm of gradients (adjusted for number of features): 6.4946 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 25/30 - Train Loss: 340.9228
Sub-losses - adver_loss: 203.8271, aggregated_sub_losses: 48.0034, paired_loss: 20.2916, class_loss: 68.8007, multi_bulk.rna.recon_loss: 0.5381, multi_bulk.rna.var_loss: 1.5710, multi_bulk.rna.loss: 2.1091, img.img.recon_loss: 15.1269, img.img.var_loss: 30.7675, img.img.loss: 45.8943, clf_loss: 0.0017
split: valid, n_samples: 323
Epoch 25/30 - Valid Loss: 347.9312
Sub-losses - adver_loss: 206.9238, aggregated_sub_losses: 49.4616, paired_loss: 21.1116, class_loss: 70.4343, multi_bulk.rna.recon_loss: 0.4781, multi_bulk.rna.var_loss: 1.7041, multi_bulk.rna.loss: 2.1821, img.img.recon_loss: 14.1721, img.img.var_loss: 33.1074, img.img.loss: 47.2794, clf_loss: 0.0013
Storing checkpoint for epoch 24...
--- Epoch 26/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 25. Total norm of gradients (adjusted for number of features): 7.8311 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 26/30 - Train Loss: 351.5484
Sub-losses - adver_loss: 214.3203, aggregated_sub_losses: 48.0713, paired_loss: 20.2825, class_loss: 68.8743, multi_bulk.rna.recon_loss: 0.5353, multi_bulk.rna.var_loss: 1.5664, multi_bulk.rna.loss: 2.1018, img.img.recon_loss: 15.1753, img.img.var_loss: 30.7942, img.img.loss: 45.9695, clf_loss: 0.0028
split: valid, n_samples: 323
Epoch 26/30 - Valid Loss: 346.0363
Sub-losses - adver_loss: 212.2418, aggregated_sub_losses: 46.4096, paired_loss: 19.8235, class_loss: 67.5614, multi_bulk.rna.recon_loss: 0.4854, multi_bulk.rna.var_loss: 1.6902, multi_bulk.rna.loss: 2.1756, img.img.recon_loss: 14.3936, img.img.var_loss: 29.8404, img.img.loss: 44.2340, clf_loss: 0.0027
--- Epoch 27/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 26. Total norm of gradients (adjusted for number of features): 8.2142 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 27/30 - Train Loss: 348.0869
Sub-losses - adver_loss: 210.9238, aggregated_sub_losses: 48.0197, paired_loss: 20.3308, class_loss: 68.8126, multi_bulk.rna.recon_loss: 0.5364, multi_bulk.rna.var_loss: 1.5506, multi_bulk.rna.loss: 2.0870, img.img.recon_loss: 15.0766, img.img.var_loss: 30.8562, img.img.loss: 45.9327, clf_loss: 0.0011
split: valid, n_samples: 323
Epoch 27/30 - Valid Loss: 360.2113
Sub-losses - adver_loss: 221.8642, aggregated_sub_losses: 48.4344, paired_loss: 20.5780, class_loss: 69.3347, multi_bulk.rna.recon_loss: 0.5157, multi_bulk.rna.var_loss: 1.8154, multi_bulk.rna.loss: 2.3311, img.img.recon_loss: 14.1942, img.img.var_loss: 31.9091, img.img.loss: 46.1033, clf_loss: 0.0003
--- Epoch 28/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 27. Total norm of gradients (adjusted for number of features): 11.4655 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 28/30 - Train Loss: 349.0731
Sub-losses - adver_loss: 211.9900, aggregated_sub_losses: 48.0576, paired_loss: 20.3356, class_loss: 68.6899, multi_bulk.rna.recon_loss: 0.5320, multi_bulk.rna.var_loss: 1.5673, multi_bulk.rna.loss: 2.0993, img.img.recon_loss: 15.1350, img.img.var_loss: 30.8233, img.img.loss: 45.9583, clf_loss: 0.0031
split: valid, n_samples: 323
Epoch 28/30 - Valid Loss: 352.1399
Sub-losses - adver_loss: 215.2407, aggregated_sub_losses: 47.5142, paired_loss: 20.4821, class_loss: 68.9029, multi_bulk.rna.recon_loss: 0.4926, multi_bulk.rna.var_loss: 1.4499, multi_bulk.rna.loss: 1.9426, img.img.recon_loss: 14.3661, img.img.var_loss: 31.2055, img.img.loss: 45.5716, clf_loss: 0.0015
--- Epoch 29/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 28. Total norm of gradients (adjusted for number of features): 8.8409 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 29/30 - Train Loss: 338.3573
Sub-losses - adver_loss: 201.2539, aggregated_sub_losses: 48.0384, paired_loss: 20.3193, class_loss: 68.7456, multi_bulk.rna.recon_loss: 0.5428, multi_bulk.rna.var_loss: 1.5603, multi_bulk.rna.loss: 2.1031, img.img.recon_loss: 15.0984, img.img.var_loss: 30.8369, img.img.loss: 45.9353, clf_loss: 0.0031
split: valid, n_samples: 323
Epoch 29/30 - Valid Loss: 340.7493
Sub-losses - adver_loss: 203.9537, aggregated_sub_losses: 47.6893, paired_loss: 20.4101, class_loss: 68.6962, multi_bulk.rna.recon_loss: 0.4781, multi_bulk.rna.var_loss: 1.8940, multi_bulk.rna.loss: 2.3721, img.img.recon_loss: 14.0875, img.img.var_loss: 31.2298, img.img.loss: 45.3173, clf_loss: 0.0004
--- Epoch 30/30 ---
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:446: UserWarning: Gradient clipping was first applied in epoch 29. Total norm of gradients (adjusted for number of features): 10.0081 exceeded max norm of 5.
  warnings.warn(
```

```
split: train, n_samples: 5765.0
Epoch 30/30 - Train Loss: 339.5961
Sub-losses - adver_loss: 202.4635, aggregated_sub_losses: 48.0025, paired_loss: 20.2992, class_loss: 68.8309, multi_bulk.rna.recon_loss: 0.5385, multi_bulk.rna.var_loss: 1.5415, multi_bulk.rna.loss: 2.0800, img.img.recon_loss: 15.0539, img.img.var_loss: 30.8687, img.img.loss: 45.9225, clf_loss: 0.0023
split: valid, n_samples: 323
Epoch 30/30 - Valid Loss: 350.7390
Sub-losses - adver_loss: 211.1494, aggregated_sub_losses: 48.8886, paired_loss: 20.8458, class_loss: 69.8552, multi_bulk.rna.recon_loss: 0.4871, multi_bulk.rna.var_loss: 1.8902, multi_bulk.rna.loss: 2.3772, img.img.recon_loss: 14.3014, img.img.var_loss: 32.2100, img.img.loss: 46.5114, clf_loss: 0.0014
Storing checkpoint for epoch 29...
Prediction complete.
Proces
... [output truncated]
```

## 2) Passing Data Directly[¶](#2-passing-data-directly)

If you want to provide the data yourself, or the data comes from another script directly as a Python object, we show how to prepare it for `AUTOENCODIX`.\
We will go over our `three main data modalities` as in the previous section.

The main class for this is the `DataPackage` class. This serves as a container for `pandas` DataFrames and `AnnData`/`MuData`.\
Let's see in code how this works to make things clearer:

#### 2.1) Passing Numeric Tabular Data[¶](#21-passing-numeric-tabular-data)

Let's assume we have two data modalities as `pd.DataFrame`s and one annotation file with metadata, also as a `pd.DataFrame`.\
The only requirement for the data files is that they are numeric. For the annotation file, categorical variables are also allowed.

In \[2\]:

Copied!

```
from autoencodix.utils.example_data import raw_protein, raw_rna, annotation
from autoencodix.data import DataPackage

print(f"Type of rna: {type(raw_rna)}")
raw_protein.head()
```

from autoencodix.utils.example_data import raw_protein, raw_rna, annotation from autoencodix.data import DataPackage print(f"Type of rna: {type(raw_rna)}") raw_protein.head()

```
Type of rna: <class 'pandas.core.frame.DataFrame'>
```

Out\[2\]:

```
protein_0
protein_1
protein_2
protein_3
protein_4
protein_5
protein_6
protein_7
protein_8
protein_9
...
protein_70
protein_71
protein_72
protein_73
protein_74
protein_75
protein_76
protein_77
protein_78
protein_79




sample_0
-12.042401
0.087191
-2.569039
1.194324
3.232265
-0.849021
-2.456986
-4.714146
-0.330510
1.791241
...
1.298304
-0.888675
2.211052
0.704696
2.647467
3.053971
1.389405
-1.531050
-0.931632
-8.202654


sample_1
-5.774403
0.189626
-4.880973
0.945787
1.178465
-5.730645
-2.277935
-1.755148
-4.730100
0.394876
...
-4.046358
-3.408130
3.022748
-0.314424
2.378629
0.222959
-0.101802
4.570407
0.801779
-5.779545


sample_2
2.853659
-1.509067
-0.102335
-0.900886
-1.202590
1.023140
-0.556068
-0.431105
-3.262940
-0.705130
...
2.127268
2.228164
0.949771
0.055224
-5.031155
-2.850
... [output truncated]
```

In our `DataPackage` class we have four attributes:

- `multi_bulk`: any tabular numeric data goes here
- `multi_sc`: single-cell data
- `annotation`: metadata for samples
- `img`: image data

Each attribute holds a `Dict`, where you can pass your `DataFrame`s (or AnnData/MuData) and assign a data modality key, like:

In \[3\]:

Copied!

```
my_dp = DataPackage(
    multi_bulk={"rna": raw_rna, "protein": raw_protein}, annotation={"anno": annotation}
)
```

my_dp = DataPackage( multi_bulk={"rna": raw_rna, "protein": raw_protein}, annotation={"anno": annotation} )

The instance of `DataPackage` can be directly passed to our pipeline object via the data keyword argument.

In \[4\]:

Copied!

```
import autoencodix as acx
from autoencodix.configs.default_config import DataConfig, DataInfo, DataCase
from autoencodix.configs import VarixConfig

config = VarixConfig(data_case=DataCase.MULTI_BULK, epochs=30)
varix = acx.Varix(data=my_dp, config=config)
result = varix.run()
```

import autoencodix as acx from autoencodix.configs.default_config import DataConfig, DataInfo, DataCase from autoencodix.configs import VarixConfig config = VarixConfig(data_case=DataCase.MULTI_BULK, epochs=30) varix = acx.Varix(data=my_dp, config=config) result = varix.run()

```
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
anno key: anno
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:250: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 15.8682 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 10 - Train Loss: 152.2939
Sub-losses: recon_loss: 152.2562, var_loss: 0.0378, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Epoch 10 - Valid Loss: 152.1192
Sub-losses: recon_loss: 152.0843, var_loss: 0.0349, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Epoch 20 - Train Loss: 114.6131
Sub-losses: recon_loss: 109.9776, var_loss: 4.6355, anneal_factor: 0.9350, effective_beta_factor: 0.0935
Epoch 20 - Valid Loss: 120.5976
Sub-losses: recon_loss: 116.4424, var_loss: 4.1552, anneal_factor: 0.9350, effective_beta_factor: 0.0935
Epoch 30 - Train Loss: 96.5454
Sub-losses: recon_loss: 90.1262, var_loss: 6.4192, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Epoch 30 - Valid Loss: 103.8546
Sub-losses: recon_loss: 97.8245, var_loss: 6.0301, anneal_factor: 0.9999, effective_
... [output truncated]
```

You still can customize your config by providing a `DataConfig` with `DataInfo` as shown in [section 1](#11-combining-mulit-omics-data-from-bulk-sequencing).

In \[5\]:

Copied!

```
my_dp = DataPackage(
    multi_bulk={"rna": raw_rna, "protein": raw_protein}, annotation={"anno": annotation}
)
data_config = DataConfig(
    data_info={
        "rna": DataInfo(scaling="MINMAX"),
        "protein": DataInfo(scaling="MINMAX"),
    },
)

bulk_config = VarixConfig(data_config=data_config, data_case=DataCase.MULTI_BULK, epochs=30)
varix = acx.Varix(config=bulk_config, data=my_dp)
r = varix.run()
```

my_dp = DataPackage( multi_bulk={"rna": raw_rna, "protein": raw_protein}, annotation={"anno": annotation} ) data_config = DataConfig( data_info={ "rna": DataInfo(scaling="MINMAX"), "protein": DataInfo(scaling="MINMAX"), }, ) bulk_config = VarixConfig(data_config=data_config, data_case=DataCase.MULTI_BULK, epochs=30) varix = acx.Varix(config=bulk_config, data=my_dp) r = varix.run()

```
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
anno key: anno
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:250: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 15.7751 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 10 - Train Loss: 12.6953
Sub-losses: recon_loss: 12.6528, var_loss: 0.0425, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Epoch 10 - Valid Loss: 8.5690
Sub-losses: recon_loss: 8.5338, var_loss: 0.0352, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Epoch 20 - Train Loss: 7.9003
Sub-losses: recon_loss: 6.9505, var_loss: 0.9497, anneal_factor: 0.9350, effective_beta_factor: 0.0935
Epoch 20 - Valid Loss: 6.7541
Sub-losses: recon_loss: 5.9348, var_loss: 0.8193, anneal_factor: 0.9350, effective_beta_factor: 0.0935
Epoch 30 - Train Loss: 6.1444
Sub-losses: recon_loss: 5.5134, var_loss: 0.6310, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Epoch 30 - Valid Loss: 5.8938
Sub-losses: recon_loss: 5.3956, var_loss: 0.4981, anneal_factor: 0.9999, effective_beta_factor: 0.1000
... [output truncated]
```

#### 2.2) Passing Single Cell Data[¶](#22-passing-single-cell-data)

This works analogously to the numeric tabular case above. Instead of `pd.DataFrame`s, we pass `AnnData` or `MuData` objects (from `scverse`) to our `DataPackage`.

It is also possible to pass `AnnData` or `MuData` directly to the pipeline.

**Working with AnnData**

In \[6\]:

Copied!

```
from autoencodix.utils.example_data import sample_adata, sample_mudata

sample_adata
```

from autoencodix.utils.example_data import sample_adata, sample_mudata sample_adata

Out\[6\]:

```
AnnData object with n_obs × n_vars = 1000 × 500
    obs: 'cell_type', 'batch', 'donor', 'cell_cycle'
```

For `AnnData`, this is now actually super simple: you can just pass the `AnnData` object directly to the pipeline, without specifying the `DataCase` for `Varix`, because we can infer it from the pipeline type and the input data type.

In this example, we did not pass a config, which means the pipeline will use default parameters.

Of course, you can also define and pass a custom config if needed.

In \[7\]:

Copied!

```
varix = acx.Varix(data=sample_adata)
result = varix.run()
```

varix = acx.Varix(data=sample_adata) result = varix.run()

```
in handle_direct_user_data with data: <class 'anndata._core.anndata.AnnData'>
mudata: View of MuData object with n_obs × n_vars = 1000 × 500
  1 modality
    user-data:	1000 x 500
      obs:	'cell_type', 'batch', 'donor', 'cell_cycle'
Processing 1 MuData objects: ['multi_sc']
Processing train modality: multi_sc
Processing valid split
Processing valid modality: multi_sc
Processing test split
Processing test modality: multi_sc
```

```
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:250: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 16.4819 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 3 - Train Loss: 400.7616
Sub-losses: recon_loss: 396.2915, var_loss: 4.4701, anneal_factor: 0.9656, effective_beta_factor: 0.0966
Epoch 3 - Valid Loss: 327.8859
Sub-losses: recon_loss: 323.4257, var_loss: 4.4602, anneal_factor: 0.9656, effective_beta_factor: 0.0966
Processed 200 / 200 samples
```

**Working with MuData**\
In some cases, you may want to combine multiple single-cell modalities. Here, we expect a `MuData` object. This can be passed directly to the pipeline, as shown for `AnnData`, or via our `DataPackage`, as shown in the tabular numeric case.

In \[8\]:

Copied!

```
sample_mudata
```

sample_mudata

Out\[8\]:

```
MuData object with n_obs × n_vars = 1000 × 700
  2 modalities
    rna:	1000 x 500
      obs:	'cell_type', 'batch', 'donor', 'cell_cycle'
    protein:	1000 x 200
      obs:	'cell_type', 'batch', 'donor', 'cell_cycle'
```

In \[9\]:

Copied!

```
varix = acx.Varix(data=sample_mudata)
result = varix.run()
```

varix = acx.Varix(data=sample_mudata) result = varix.run()

```
in handle_direct_user_data with data: <class 'mudata._core.mudata.MuData'>
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
/home/ewald/Github/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:250: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 14.9491 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 3 - Train Loss: 464.2348
Sub-losses: recon_loss: 457.3495, var_loss: 6.8853, anneal_factor: 0.9656, effective_beta_factor: 0.0966
Epoch 3 - Valid Loss: 379.6347
Sub-losses: recon_loss: 372.9406, var_loss: 6.6941, anneal_factor: 0.9656, effective_beta_factor: 0.0966
Processed 200 / 200 samples
```

#### 2.3) Passing Image Data[¶](#23-passing-image-data)

We recommend passing image data via a folder path in the config, as shown in [Section 1.3](#13-working-with-images).\
For completeness, we show how one could pass images directly to the pipeline, but this is not recommended.

We store image data as a list, where each item represents one image and is of our custom class `ImgData`.

In \[10\]:

Copied!

```
from autoencodix.data import ImgData
from autoencodix.data.datapackage import DataPackage
import numpy as np
import pandas as pd

imgdata = [ImgData(img=np.array(1), sample_id="myImg", annotation=pd.DataFrame())]
dp = DataPackage(img={"IMG": imgdata})
# in real applications this will be a longer list of images
```

from autoencodix.data import ImgData from autoencodix.data.datapackage import DataPackage import numpy as np import pandas as pd imgdata = [ImgData(img=np.array(1), sample_id="myImg", annotation=pd.DataFrame())] dp = DataPackage(img={"IMG": imgdata})

# in real applications this will be a longer list of images

## 3) Note on Paired and Unpaired Data[¶](#3-note-on-paired-and-unpaired-data)

When working with mutli-modal data it is common that not all data modalities are present for all samples. For most the models `Vanillix`, `Varix`, and `Ontix`, we require paired data due to inherent architecture constraints and remove samples that are not present in all data modalities.

For the models `XModalix` and `Stackix` don't have the same arechitectural constraints. Thus, we can allow also unpaired data. This needs to be specified in the config via the param `requires_paired=False`.

## 4) Notes on Data Modality Specific Preprocessing and Skipping Preprocessing[¶](#4-notes-on-data-modality-specific-preprocessing-and-skipping-preprocessing)

Depending on the data modality, our pipeline applies different preprocessing steps. We provide an overview of applied preprocessing techniques for our `three main data modalities`.

#### 4.1 Preprocessing for Tabular Numeric Data[¶](#41-preprocessing-for-tabular-numeric-data)

The following steps are applied in order:

- Remove missing and NaN values from tabular numeric data.
- Consolidate missing and NaN values to one "missing" class for the annotation columns (as specified with the `annotation_columns` parameter in config) of the annotation dataframe.
- Split data into train/validation/test sets according to the ratios defined in the config.
- Filter features based on the method defined in the config. The statistical operations to filter the data are performed on the train set to avoid data leakage. The learned filtering from the train set is then applied to the validation and test sets.
- Fit a `sklearn` scaler on the train data, depending on the defined scaling algorithm (MinMax, Standard, etc.). The fitted scaler is then applied to the validation and test sets.
  - Filtering and scaling are also applied if you run the predict step with new unseen data.
- Transform the data into a PyTorch dataset to be ready for training with `nn.Module`.

#### 4.2 Preprocessing for Sparse Numeric Data (Single Cell)[¶](#42-preprocessing-for-sparse-numeric-data-single-cell)

The following steps are performed in order:

- Replace missing values with zeros.
- Apply two single-cell-specific processing steps if defined in the config:
  - Filter cells based on a gene count, defined in `min_genes` in the config via `sc.pp.filter_cells`.
  - Log-transform, if specified in config under `log_transform` via `sc.pp.log1p`.
- After this, follow the preprocessing for tabular numeric data (see above).
- Keep the data sparse in our PyTorch dataset until batch level. This prevents out-of-memory errors for large datasets when transforming from sparse to dense.

#### 4.3 Preprocessing for Image Data[¶](#43-preprocessing-for-image-data)

For image data, preprocessing is as follows:

- Split data into train/validation/test sets as defined in the config.
- Resize images to a quadratic image size that fits our encoder depth or to the specified width and height in the config.
- Scale pixels depending on the defined scaling algorithm (MinMax, Standard, etc.) specific to images, not using `sklearn`.
- Transform data into a PyTorch dataset specialized for image data.

#### 4.4 Skipping Preprocessing[¶](#44-skipping-preprocessing)

Preprocessing can be skipped via the config parameter `skip_preprocessing`. However, we cannot guarantee that there won't be `RuntimeErrors`.\
Errors should not occur if your data does not contain `NaNs` and your loss function matches the scaling. For example, if you set `reconstruction_loss` to `bce`, ensure your input data is between 0 and 1, because the last layer of our architecture is a Sigmoid.

## 5) Adding Custom Splits[¶](#5-adding-custom-splits)

Our standard preprocessing splits the data randomly into train/validation/test sets. Custom splits can be provided by via the pipeline option `custom_splits` providing a `Dict` containing `train`, `valid` and `test` keys with a corresponding list of sample IDs as strings.

Alternatively the `utils` function `custom_splits_from_anno()` can be used to read in splits from a CSV. The CSV should have two columns, `sample_id` with the sample IDs and `custom_split` with the split the sample belongs to (train/valid/test).

In the following is an example on the TCGA data where we exclude one cancer type from the training and validation set and define it as test set.

In \[1\]:

Copied!

```
import pandas as pd
import numpy as np
import autoencodix as acx
from autoencodix.configs.default_config import DataConfig, DataInfo, DataCase
from autoencodix.configs import VarixConfig
from huggingface_hub import hf_hub_download

# Data is hosted on Hugging Face Hub and downloaded (and locally cached) automatically
HF_REPO_ID = "autoencodix/tcga"
mrna_file = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="rna.parquet")
meth_file = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="methylation.parquet")
clin_file = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="clinical.parquet")

## Create custom splits for training, validation, and testing
clin_df = pd.read_parquet(clin_file)
test_indices = list(clin_df[clin_df['CANCER_TYPE'] == 'Breast Cancer'].index)
remaining_indices = list(clin_df[clin_df['CANCER_TYPE'] != 'Breast Cancer'].index)
# Split remaining indices randomly into train and validation sets (80% train, 20% validation)
train_indices = np.random.choice(remaining_indices, size=int(0.8 * len(remaining_indices)), replace=False)
val_indices = [i for i in remaining_indices if i not in train_indices]

custom_splits = {
	"train": train_indices,
	"valid": val_indices,
	"test": test_indices
}

clin_df.loc[custom_splits['train'],:]

data_config = DataConfig(
    annotation_columns=["CANCER_TYPE"],
    data_info={
        "RNA": DataInfo(file_path=mrna_file),
        "METHYLATION": DataInfo(file_path=meth_file),
        "CLINICAL": DataInfo(
            file_path=clin_file, data_type="ANNOTATION"
        ),
    },
)

bulk_config = VarixConfig(data_config=data_config, data_case=DataCase.MULTI_BULK, epochs=30, checkpoint_interval=10, k_filter=100)
varix = acx.Varix(config=bulk_config, custom_splits=custom_splits)

result_varix = varix.run()
varix.show_result()
```

import pandas as pd import numpy as np import autoencodix as acx from autoencodix.configs.default_config import DataConfig, DataInfo, DataCase from autoencodix.configs import VarixConfig from huggingface_hub import hf_hub_download

# Data is hosted on Hugging Face Hub and downloaded (and locally cached) automatically

HF_REPO_ID = "autoencodix/tcga" mrna_file = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="rna.parquet") meth_file = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="methylation.parquet") clin_file = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="clinical.parquet")

## Create custom splits for training, validation, and testing

clin_df = pd.read_parquet(clin_file) test_indices = list(clin_df\[clin_df['CANCER_TYPE'] == 'Breast Cancer'\].index) remaining_indices = list(clin_df\[clin_df['CANCER_TYPE'] != 'Breast Cancer'\].index)

# Split remaining indices randomly into train and validation sets (80% train, 20% validation)

train_indices = np.random.choice(remaining_indices, size=int(0.8 * len(remaining_indices)), replace=False) val_indices = [i for i in remaining_indices if i not in train_indices] custom_splits = { "train": train_indices, "valid": val_indices, "test": test_indices } clin_df.loc\[custom_splits['train'],:\] data_config = DataConfig( annotation_columns=["CANCER_TYPE"], data_info={ "RNA": DataInfo(file_path=mrna_file), "METHYLATION": DataInfo(file_path=meth_file), "CLINICAL": DataInfo( file_path=clin_file, data_type="ANNOTATION" ), }, ) bulk_config = VarixConfig(data_config=data_config, data_case=DataCase.MULTI_BULK, epochs=30, checkpoint_interval=10, k_filter=100) varix = acx.Varix(config=bulk_config, custom_splits=custom_splits) result_varix = varix.run() varix.show_result()

```
/home/kristina/Dokumente/autoencodix_package_custom_split/src/autoencodix/configs/default_config.py:559: UserWarning: annotation_columns in DataConfig is deprecated. Please set it directly in DefaultConfig instead.
  warnings.warn(
```

```
reading parquet: /home/kristina/.cache/huggingface/hub/datasets--autoencodix--tcga/snapshots/c59ee679d11ea17c0fd615248a382ef1c38ef456/rna.parquet
reading parquet: /home/kristina/.cache/huggingface/hub/datasets--autoencodix--tcga/snapshots/c59ee679d11ea17c0fd615248a382ef1c38ef456/methylation.parquet
reading parquet: /home/kristina/.cache/huggingface/hub/datasets--autoencodix--tcga/snapshots/c59ee679d11ea17c0fd615248a382ef1c38ef456/clinical.parquet
Using custom splits in PairedUnpairedSplitter.
anno key: paired
```

```
/home/kristina/Dokumente/autoencodix_package_custom_split/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 19.1786 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 10 - Train Loss: 72.7226
Sub-losses: recon_loss: 72.6276, var_loss: 0.0950, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Epoch 10 - Valid Loss: 62.9423
Sub-losses: recon_loss: 62.8509, var_loss: 0.0914, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Epoch 20 - Train Loss: 73.4160
Sub-losses: recon_loss: 71.1807, var_loss: 2.2353, anneal_factor: 0.9350, effective_beta_factor: 0.0935
Epoch 20 - Valid Loss: 63.2366
Sub-losses: recon_loss: 61.1673, var_loss: 2.0693, anneal_factor: 0.9350, effective_beta_factor: 0.0935
Epoch 30 - Train Loss: 71.8787
Sub-losses: recon_loss: 69.8613, var_loss: 2.0173, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Epoch 30 - Valid Loss: 62.3669
Sub-losses: recon_loss: 60.3995, var_loss: 1.9674, anneal_factor: 0.9999, effective_beta_fact
... [output truncated]
```

When we look at the three sets separately we can see that train and validation set contain all cancer types except breast cancer and the test set contains only breast cancer. Our custom split worked.

In \[8\]:

Copied!

```
varix.visualizer.show_latent_space(
    result=varix.result,
    plot_type="2D-scatter",
    split="train",
    param=["CANCER_TYPE"],  # optional: color points by this annotation
)
```

varix.visualizer.show_latent_space( result=varix.result, plot_type="2D-scatter", split="train", param=["CANCER_TYPE"], # optional: color points by this annotation )

In \[9\]:

Copied!

```
varix.visualizer.show_latent_space(
    result=varix.result,
    plot_type="2D-scatter",
    split="valid",
    param=["CANCER_TYPE"],  # optional: color points by this annotation
)
```

varix.visualizer.show_latent_space( result=varix.result, plot_type="2D-scatter", split="valid", param=["CANCER_TYPE"], # optional: color points by this annotation )

In \[10\]:

Copied!

```
varix.visualizer.show_latent_space(
    result=varix.result,
    plot_type="2D-scatter",
    split="test",
    param=["CANCER_TYPE"],  # optional: color points by this annotation
)
```

varix.visualizer.show_latent_space( result=varix.result, plot_type="2D-scatter", split="test", param=["CANCER_TYPE"], # optional: color points by this annotation )
