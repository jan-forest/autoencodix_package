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
> If you have columns inside the `ANNOTATION` data that you want to use later in downstream visualization or evaluation tasks, please specify them with the `annotation_columns` parameter of the pipeline config (here `VarixConfig`). This will prevent errors later, as NaNs will be handled automatically during preprocessing for these columns.

#### ❗❗ Requirements: Getting Tutorial Data ❗❗[¶](#requirements-getting-tutorial-data)

The data for this example is hosted on Hugging Face Hub ([autoencodix/tcga](https://huggingface.co/datasets/autoencodix/tcga)) and is downloaded automatically in the cell below.

In \[2\]:

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
bulk_config = VarixConfig(data_config=data_config, annotation_columns=["CANCER_TYPE"], data_case=DataCase.MULTI_BULK, epochs=30, checkpoint_interval=10, k_filter=100)
varix = acx.Varix(config=bulk_config)

result_varix = varix.run()
varix.show_result()
```

import autoencodix as acx from autoencodix.configs.default_config import DataConfig, DataInfo, DataCase from autoencodix.configs import VarixConfig from huggingface_hub import hf_hub_download

# Data is hosted on Hugging Face Hub and downloaded (and locally cached) automatically

HF_REPO_ID = "autoencodix/tcga" mrna_file = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="rna.parquet") meth_file = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="methylation.parquet") clin_file = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="clinical.parquet") """ Our config class has the attribute `data_config`. Here, we fill the `data_info` Dict, by defining names for our data modalities ("RNA" for example). The value of this Dict is a instance of the `DataInfo` class. All we need to pass here is a `file_path` argument. """ data_config = DataConfig( data_info={ "RNA": DataInfo(file_path=mrna_file), "METHYLATION": DataInfo(file_path=meth_file), "CLINICAL": DataInfo( file_path=clin_file, data_type="ANNOTATION" ), }, ) """ Lastly, we pass our data_config to the VarixConfig. Remember to pass the appropriate `DataCase`. """ bulk_config = VarixConfig(data_config=data_config, annotation_columns=["CANCER_TYPE"], data_case=DataCase.MULTI_BULK, epochs=30, checkpoint_interval=10, k_filter=100) varix = acx.Varix(config=bulk_config) result_varix = varix.run() varix.show_result()

```
reading parquet: /home/janew/.cache/huggingface/hub/datasets--autoencodix--tcga/snapshots/c59ee679d11ea17c0fd615248a382ef1c38ef456/rna.parquet
```

```
reading parquet: /home/janew/.cache/huggingface/hub/datasets--autoencodix--tcga/snapshots/c59ee679d11ea17c0fd615248a382ef1c38ef456/methylation.parquet
```

```
reading parquet: /home/janew/.cache/huggingface/hub/datasets--autoencodix--tcga/snapshots/c59ee679d11ea17c0fd615248a382ef1c38ef456/clinical.parquet
```

```
anno key: paired
```

```
You are using a CUDA device ('NVIDIA GeForce RTX 5060 Ti') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 15.0148 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 10 - Train Loss: 82.0013
Sub-losses: recon_loss: 81.9152, var_loss: 0.0861, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Epoch 10 - Valid Loss: 77.8209
Sub-losses: recon_loss: 77.7234, var_loss: 0.0975, anneal_factor: 0.0180, effective_beta_factor: 0.0018
```

```
Epoch 20 - Train Loss: 79.2955
Sub-losses: recon_loss: 77.3822, var_loss: 1.9132, anneal_factor: 0.9350, effective_beta_factor: 0.0935
Epoch 20 - Valid Loss: 74.8004
Sub-losses: recon_loss: 73.0722, var_loss: 1.7283, anneal_factor: 0.9350, effective_beta_factor: 0.0935
```

```
Epoch 30 - Train Loss: 76.5190
Sub-losses: recon_loss: 74.7004, var_loss: 1.8187, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Epoch 30 - Valid Loss: 72.3115
Sub-losses: recon_loss: 70.6669, var_loss: 1.6446, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Processed 32 / 707 samples
Processed 64 / 707 samples
Processed 96 / 707 samples
Processed 128 / 707 samples
Processed 160 / 707 samples
Processed 192 / 707 samples
Processed 224 / 707 samples
Processed 256 / 707 samples
Processed 288 / 707 samples
Processed 320 / 707 samples
Processed 352 / 707 samples
Processed 384 / 707 samples
Processed 416 / 707 samples
Processed 448 / 707 samples
Processed 480 / 707 samples
Processed 512 / 707 samples
Processed 544 / 707 samples
Processed 576 / 707 samples
Processed 608 / 707 samples
... [output truncated]
```

```
Creating plots ...
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

In \[3\]:

Copied!

```
import autoencodix as acx
from autoencodix.configs.default_config import DataConfig, DataInfo, DataCase
from autoencodix.configs import VarixConfig
from huggingface_hub import hf_hub_download

filepath = hf_hub_download(repo_id="autoencodix/pancreas-fang", repo_type="dataset", filename="pancreas.h5ad")
data_config = DataConfig(
    data_info={"SC": DataInfo(file_path=filepath, is_single_cell=True)},
)
sc_config = VarixConfig(data_config=data_config, annotation_columns=["assigned_cluster"], data_case=DataCase.MULTI_SINGLE_CELL, k_filter=100, epochs=30, checkpoint_interval=10)
varix_sc = acx.Varix(config=sc_config)
result_sc = varix_sc.run()
```

import autoencodix as acx from autoencodix.configs.default_config import DataConfig, DataInfo, DataCase from autoencodix.configs import VarixConfig from huggingface_hub import hf_hub_download filepath = hf_hub_download(repo_id="autoencodix/pancreas-fang", repo_type="dataset", filename="pancreas.h5ad") data_config = DataConfig( data_info={"SC": DataInfo(file_path=filepath, is_single_cell=True)}, ) sc_config = VarixConfig(data_config=data_config, annotation_columns=["assigned_cluster"], data_case=DataCase.MULTI_SINGLE_CELL, k_filter=100, epochs=30, checkpoint_interval=10) varix_sc = acx.Varix(config=sc_config) result_sc = varix_sc.run()

```
Number of common cells: 8569
```

```
mudata: View of MuData object with n_obs × n_vars = 8569 × 20125
  obs:	'barcode', 'assigned_cluster', 'sample_id'
  1 modality
    SC:	8569 x 20125
      obs:	'barcode', 'assigned_cluster', 'sample_id'
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
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 16.2997 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 10 - Train Loss: 73.1627
Sub-losses: recon_loss: 72.9744, var_loss: 0.1882, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Epoch 10 - Valid Loss: 66.8011
Sub-losses: recon_loss: 66.6294, var_loss: 0.1717, anneal_factor: 0.0180, effective_beta_factor: 0.0018
```

```
Epoch 20 - Train Loss: 73.1266
Sub-losses: recon_loss: 71.2037, var_loss: 1.9228, anneal_factor: 0.9350, effective_beta_factor: 0.0935
Epoch 20 - Valid Loss: 66.4729
Sub-losses: recon_loss: 64.5531, var_loss: 1.9198, anneal_factor: 0.9350, effective_beta_factor: 0.0935
```

```
Epoch 30 - Train Loss: 71.8575
Sub-losses: recon_loss: 69.9701, var_loss: 1.8874, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Epoch 30 - Valid Loss: 65.0221
Sub-losses: recon_loss: 63.2418, var_loss: 1.7803, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Processed 32 / 1715 samples
Processed 64 / 1715 samples
Processed 96 / 1715 samples
Processed 128 / 1715 samples
Processed 160 / 1715 samples
Processed 192 / 1715 samples
Processed 224 / 1715 samples
Processed 256 / 1715 samples
Processed 288 / 1715 samples
Processed 320 / 1715 samples
Processed 352 / 1715 samples
Processed 384 / 1715 samples
Processed 416 / 1715 samples
Processed 448 / 1715 samples
Processed 480 / 1715 samples
Processed 512 / 1715 samples
Processed 544 / 1715 samples
Processed 576 / 1715 samples
Processed
... [output truncated]
```

```
Processed 736 / 1715 samples
Processed 768 / 1715 samples
Processed 800 / 1715 samples
Processed 832 / 1715 samples
Processed 864 / 1715 samples
Processed 896 / 1715 samples
Processed 928 / 1715 samples
Processed 960 / 1715 samples
Processed 992 / 1715 samples
Processed 1024 / 1715 samples
Processed 1056 / 1715 samples
Processed 1088 / 1715 samples
Processed 1120 / 1715 samples
Processed 1152 / 1715 samples
Processed 1184 / 1715 samples
Processed 1216 / 1715 samples
Processed 1248 / 1715 samples
Processed 1280 / 1715 samples
Processed 1312 / 1715 samples
Processed 1344 / 1715 samples
Processed 1376 / 1715 samples
Processed 1408 / 1715 samples
Processed 1440 / 1715 samples
Processed 1472 / 1715 samples
Processed 1504 / 1715 samples
Processed 1536 / 1715 samples
Processed 1568 / 1715 sample
... [output truncated]
```

**Special Case for Visualization**\
As explained before, we can define one or more columns to use for visualization. In our example, we chose the `assigned_cluster` column. Internally, the column gets prefixed with the name of the data modality defined in the `DataInfo` dict. Therefore, to visualize this column, we need to add the prefix to the column name, e.g., `SC:assigned_cluster`.

> This should be fixed now, allowing you to pass the column name without the prefix.

In \[4\]:

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

In \[5\]:

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
```

```
reading parquet: data/XModalix-Tut-data/combined_clin_formatted.parquet
```

```
Given image size is possible, rescaling images to: 32x32
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
Check if we need to pretrain: multi_bulk.rna
pretrain epochs : 10
starting pretraining for: multi_bulk.rna with <class 'autoencodix.trainers._general_trainer.GeneralTrainer'>
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 9.1323 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 5 - Train Loss: 0.5863
Sub-losses: recon_loss: 0.5856, var_loss: 0.0007, anneal_factor: 0.0007, effective_beta_factor: 0.0001
Epoch 5 - Valid Loss: 0.4993
Sub-losses: recon_loss: 0.4987, var_loss: 0.0006, anneal_factor: 0.0007, effective_beta_factor: 0.0001
```

```
Epoch 10 - Train Loss: 0.4964
Sub-losses: recon_loss: 0.4826, var_loss: 0.0138, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Epoch 10 - Valid Loss: 0.4527
Sub-losses: recon_loss: 0.4402, var_loss: 0.0126, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Check if we need to pretrain: img.img
pretrain epochs : 10
starting pretraining for: img.img with <class 'autoencodix.trainers._general_trainer.GeneralTrainer'>
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 108.9137 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 5 - Train Loss: 33.6545
Sub-losses: recon_loss: 33.6446, var_loss: 0.0098, anneal_factor: 0.0007, effective_beta_factor: 0.0001
Epoch 5 - Valid Loss: 31.0203
Sub-losses: recon_loss: 31.0097, var_loss: 0.0107, anneal_factor: 0.0007, effective_beta_factor: 0.0001
```

```
Epoch 10 - Train Loss: 23.8403
Sub-losses: recon_loss: 23.2998, var_loss: 0.5405, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Epoch 10 - Valid Loss: 22.7215
Sub-losses: recon_loss: 22.1636, var_loss: 0.5579, anneal_factor: 0.0180, effective_beta_factor: 0.0018
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_xmodal_trainer.py:469: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 5.7760 exceeded max norm of 5.
  warnings.warn(
```

```
--- Epoch 5/30 ---
```

```
Epoch 5/30 - Train Loss: 237.9127
Sub-losses - adver_loss: 138.9067, aggregated_sub_losses: 19.5406, paired_loss: 17.4100, class_loss: 62.0554, multi_bulk.rna.recon_loss: 0.5762, multi_bulk.rna.var_loss: 0.0006, multi_bulk.rna.loss: 0.5768, img.img.recon_loss: 18.9482, img.img.var_loss: 0.0157, img.img.loss: 18.9638, clf_loss: 0.0162
```

```
Epoch 5/30 - Valid Loss: 229.4239
Sub-losses - adver_loss: 133.4085, aggregated_sub_losses: 18.1579, paired_loss: 16.9626, class_loss: 60.8949, multi_bulk.rna.recon_loss: 0.5501, multi_bulk.rna.var_loss: 0.0005, multi_bulk.rna.loss: 0.5506, img.img.recon_loss: 17.5924, img.img.var_loss: 0.0148, img.img.loss: 17.6073, clf_loss: 0.0067
Storing checkpoint for epoch 4...
```

```
--- Epoch 10/30 ---
```

```
Epoch 10/30 - Train Loss: 254.1130
Sub-losses - adver_loss: 154.8897, aggregated_sub_losses: 19.7585, paired_loss: 17.4046, class_loss: 62.0602, multi_bulk.rna.recon_loss: 0.5752, multi_bulk.rna.var_loss: 0.0161, multi_bulk.rna.loss: 0.5913, img.img.recon_loss: 18.7364, img.img.var_loss: 0.4309, img.img.loss: 19.1672, clf_loss: 0.0095
```

```
Epoch 10/30 - Valid Loss: 264.3958
Sub-losses - adver_loss: 169.0099, aggregated_sub_losses: 18.5332, paired_loss: 16.6654, class_loss: 60.1873, multi_bulk.rna.recon_loss: 0.5403, multi_bulk.rna.var_loss: 0.0194, multi_bulk.rna.loss: 0.5597, img.img.recon_loss: 17.5754, img.img.var_loss: 0.3981, img.img.loss: 17.9735, clf_loss: 0.0016
Storing checkpoint for epoch 9...
```

```
--- Epoch 15/30 ---
```

```
Epoch 15/30 - Train Loss: 266.7817
Sub-losses - adver_loss: 159.2618, aggregated_sub_losses: 27.8579, paired_loss: 17.4595, class_loss: 62.2024, multi_bulk.rna.recon_loss: 0.5761, multi_bulk.rna.var_loss: 0.3006, multi_bulk.rna.loss: 0.8767, img.img.recon_loss: 18.8535, img.img.var_loss: 8.1277, img.img.loss: 26.9813, clf_loss: 0.0084
```

```
Epoch 15/30 - Valid Loss: 256.3250
Sub-losses - adver_loss: 152.2392, aggregated_sub_losses: 26.2685, paired_loss: 17.0486, class_loss: 60.7688, multi_bulk.rna.recon_loss: 0.5385, multi_bulk.rna.var_loss: 0.2765, multi_bulk.rna.loss: 0.8150, img.img.recon_loss: 17.6880, img.img.var_loss: 7.7654, img.img.loss: 25.4535, clf_loss: 0.0006
Storing checkpoint for epoch 14...
```

```
--- Epoch 20/30 ---
```

```
Epoch 20/30 - Train Loss: 272.3434
Sub-losses - adver_loss: 149.9477, aggregated_sub_losses: 42.7666, paired_loss: 17.4099, class_loss: 62.2192, multi_bulk.rna.recon_loss: 0.5784, multi_bulk.rna.var_loss: 0.8358, multi_bulk.rna.loss: 1.4142, img.img.recon_loss: 18.9621, img.img.var_loss: 22.3903, img.img.loss: 41.3524, clf_loss: 0.0078
```

```
Epoch 20/30 - Valid Loss: 267.5110
Sub-losses - adver_loss: 149.7178, aggregated_sub_losses: 40.2521, paired_loss: 16.9398, class_loss: 60.6013, multi_bulk.rna.recon_loss: 0.5440, multi_bulk.rna.var_loss: 0.8285, multi_bulk.rna.loss: 1.3725, img.img.recon_loss: 17.8791, img.img.var_loss: 21.0006, img.img.loss: 38.8796, clf_loss: 0.0005
Storing checkpoint for epoch 19...
```

```
--- Epoch 25/30 ---
```

```
Epoch 25/30 - Train Loss: 291.5098
Sub-losses - adver_loss: 167.8221, aggregated_sub_losses: 44.1945, paired_loss: 17.3909, class_loss: 62.1022, multi_bulk.rna.recon_loss: 0.5667, multi_bulk.rna.var_loss: 0.8874, multi_bulk.rna.loss: 1.4542, img.img.recon_loss: 18.8370, img.img.var_loss: 23.9033, img.img.loss: 42.7404, clf_loss: 0.0059
```

```
Epoch 25/30 - Valid Loss: 287.5018
Sub-losses - adver_loss: 167.3556, aggregated_sub_losses: 41.9905, paired_loss: 16.9858, class_loss: 61.1699, multi_bulk.rna.recon_loss: 0.5215, multi_bulk.rna.var_loss: 0.9818, multi_bulk.rna.loss: 1.5033, img.img.recon_loss: 17.5631, img.img.var_loss: 22.9241, img.img.loss: 40.4872, clf_loss: 0.0004
Storing checkpoint for epoch 24...
```

```
--- Epoch 30/30 ---
```

```
Epoch 30/30 - Train Loss: 295.1008
Sub-losses - adver_loss: 171.4776, aggregated_sub_losses: 44.1573, paired_loss: 17.4004, class_loss: 62.0654, multi_bulk.rna.recon_loss: 0.5741, multi_bulk.rna.var_loss: 0.8797, multi_bulk.rna.loss: 1.4538, img.img.recon_loss: 18.7312, img.img.var_loss: 23.9723, img.img.loss: 42.7035, clf_loss: 0.0062
```

```
Epoch 30/30 - Valid Loss: 292.3552
Sub-losses - adver_loss: 170.3753, aggregated_sub_losses: 42.9188, paired_loss: 17.3933, class_loss: 61.6679, multi_bulk.rna.recon_loss: 0.5400, multi_bulk.rna.var_loss: 0.9466, multi_bulk.rna.loss: 1.4866, img.img.recon_loss: 17.8537, img.img.var_loss: 23.5784, img.img.loss: 41.4321, clf_loss: 0.0012
Storing checkpoint for epoch 29...
```

```
Prediction complete.
Processing latent space results into a single AnnData object...
Identified source modality for latent space: 'multi_bulk.rna'
  - Added 100 source feature IDs to .uns
Finished processing latent results.
```

## 2) Passing Data Directly[¶](#2-passing-data-directly)

If you want to provide the data yourself, or the data comes from another script directly as a Python object, we show how to prepare it for `AUTOENCODIX`.\
We will go over our `three main data modalities` as in the previous section.

The main class for this is the `DataPackage` class. This serves as a container for `pandas` DataFrames and `AnnData`/`MuData`.\
Let's see in code how this works to make things clearer:

#### 2.1) Passing Numeric Tabular Data[¶](#21-passing-numeric-tabular-data)

Let's assume we have two data modalities as `pd.DataFrame`s and one annotation file with metadata, also as a `pd.DataFrame`.\
The only requirement for the data files is that they are numeric. For the annotation file, categorical variables are also allowed.

In \[6\]:

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

Out\[6\]:

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

In \[7\]:

Copied!

```
my_dp = DataPackage(
    multi_bulk={"rna": raw_rna, "protein": raw_protein}, annotation={"anno": annotation}
)
```

my_dp = DataPackage( multi_bulk={"rna": raw_rna, "protein": raw_protein}, annotation={"anno": annotation} )

The instance of `DataPackage` can be directly passed to our pipeline object via the data keyword argument.

In \[8\]:

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
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 15.3995 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 10 - Train Loss: 183.8661
Sub-losses: recon_loss: 183.8494, var_loss: 0.0167, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Epoch 10 - Valid Loss: 153.6754
Sub-losses: recon_loss: 153.6688, var_loss: 0.0066, anneal_factor: 0.0180, effective_beta_factor: 0.0018
```

```
Epoch 20 - Train Loss: 170.8820
Sub-losses: recon_loss: 170.0307, var_loss: 0.8513, anneal_factor: 0.9350, effective_beta_factor: 0.0935
Epoch 20 - Valid Loss: 147.6164
Sub-losses: recon_loss: 147.2209, var_loss: 0.3955, anneal_factor: 0.9350, effective_beta_factor: 0.0935
```

```
Epoch 30 - Train Loss: 162.9200
Sub-losses: recon_loss: 161.7729, var_loss: 1.1471, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Epoch 30 - Valid Loss: 144.4241
Sub-losses: recon_loss: 143.8149, var_loss: 0.6091, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Processed 32 / 100 samples
Processed 64 / 100 samples
Processed 96 / 100 samples
Processed 100 / 100 samples
```

You still can customize your config by providing a `DataConfig` with `DataInfo` as shown in [section 1](#11-combining-mulit-omics-data-from-bulk-sequencing).

In \[9\]:

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
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 10.5066 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 10 - Train Loss: 11.1363
Sub-losses: recon_loss: 11.1196, var_loss: 0.0167, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Epoch 10 - Valid Loss: 8.2466
Sub-losses: recon_loss: 8.2371, var_loss: 0.0095, anneal_factor: 0.0180, effective_beta_factor: 0.0018
```

```
Epoch 20 - Train Loss: 8.4417
Sub-losses: recon_loss: 7.8645, var_loss: 0.5772, anneal_factor: 0.9350, effective_beta_factor: 0.0935
Epoch 20 - Valid Loss: 5.9198
Sub-losses: recon_loss: 5.6514, var_loss: 0.2684, anneal_factor: 0.9350, effective_beta_factor: 0.0935
```

```
Epoch 30 - Train Loss: 6.7288
Sub-losses: recon_loss: 6.4396, var_loss: 0.2892, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Epoch 30 - Valid Loss: 5.4979
Sub-losses: recon_loss: 5.3585, var_loss: 0.1394, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Processed 32 / 100 samples
Processed 64 / 100 samples
Processed 96 / 100 samples
Processed 100 / 100 samples
```

#### 2.2) Passing Single Cell Data[¶](#22-passing-single-cell-data)

This works analogously to the numeric tabular case above. Instead of `pd.DataFrame`s, we pass `AnnData` or `MuData` objects (from `scverse`) to our `DataPackage`.

It is also possible to pass `AnnData` or `MuData` directly to the pipeline.

**Working with AnnData**

In \[10\]:

Copied!

```
from autoencodix.utils.example_data import sample_adata, sample_mudata

sample_adata
```

from autoencodix.utils.example_data import sample_adata, sample_mudata sample_adata

Out\[10\]:

```
AnnData object with n_obs × n_vars = 1000 × 500
    obs: 'cell_type', 'batch', 'donor', 'cell_cycle'
```

For `AnnData`, this is now actually super simple: you can just pass the `AnnData` object directly to the pipeline, without specifying the `DataCase` for `Varix`, because we can infer it from the pipeline type and the input data type.

In this example, we did not pass a config, which means the pipeline will use default parameters.

Of course, you can also define and pass a custom config if needed.

In \[11\]:

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
```

```
Processing valid split
Processing valid modality: multi_sc
Processing test split
Processing test modality: multi_sc
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 12.6049 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 3 - Train Loss: 497.8862
Sub-losses: recon_loss: 496.7424, var_loss: 1.1438, anneal_factor: 0.9656, effective_beta_factor: 0.0966
Epoch 3 - Valid Loss: 575.4791
Sub-losses: recon_loss: 574.0580, var_loss: 1.4211, anneal_factor: 0.9656, effective_beta_factor: 0.0966
Processed 32 / 200 samples
Processed 64 / 200 samples
Processed 96 / 200 samples
Processed 128 / 200 samples
Processed 160 / 200 samples
Processed 192 / 200 samples
Processed 200 / 200 samples
```

**Working with MuData**\
In some cases, you may want to combine multiple single-cell modalities. Here, we expect a `MuData` object. This can be passed directly to the pipeline, as shown for `AnnData`, or via our `DataPackage`, as shown in the tabular numeric case.

In \[12\]:

Copied!

```
sample_mudata
```

sample_mudata

Out\[12\]:

```
MuData object with n_obs × n_vars = 1000 × 700
  2 modalities
    rna:	1000 x 500
      obs:	'cell_type', 'batch', 'donor', 'cell_cycle'
    protein:	1000 x 200
      obs:	'cell_type', 'batch', 'donor', 'cell_cycle'
```

In \[13\]:

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
```

```
Processing valid split
Processing valid modality: multi_sc
Processing test split
Processing test modality: multi_sc
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 9.5354 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 3 - Train Loss: 663.0028
Sub-losses: recon_loss: 657.0651, var_loss: 5.9377, anneal_factor: 0.9656, effective_beta_factor: 0.0966
Epoch 3 - Valid Loss: 709.9949
Sub-losses: recon_loss: 708.8445, var_loss: 1.1504, anneal_factor: 0.9656, effective_beta_factor: 0.0966
Processed 32 / 200 samples
Processed 64 / 200 samples
Processed 96 / 200 samples
Processed 128 / 200 samples
Processed 160 / 200 samples
Processed 192 / 200 samples
Processed 200 / 200 samples
```

#### 2.3) Passing Image Data[¶](#23-passing-image-data)

We recommend passing image data via a folder path in the config, as shown in [Section 1.3](#13-working-with-images).\
For completeness, we show how one could pass images directly to the pipeline, but this is not recommended.

We store image data as a list, where each item represents one image and is of our custom class `ImgData`.

In \[14\]:

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

In \[15\]:

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
    data_info={
        "RNA": DataInfo(file_path=mrna_file),
        "METHYLATION": DataInfo(file_path=meth_file),
        "CLINICAL": DataInfo(
            file_path=clin_file, data_type="ANNOTATION"
        ),
    },
)

bulk_config = VarixConfig(data_config=data_config, annotation_columns=["CANCER_TYPE"], data_case=DataCase.MULTI_BULK, epochs=30, checkpoint_interval=10, k_filter=100)
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

train_indices = np.random.choice(remaining_indices, size=int(0.8 * len(remaining_indices)), replace=False) val_indices = [i for i in remaining_indices if i not in train_indices] custom_splits = { "train": train_indices, "valid": val_indices, "test": test_indices } clin_df.loc\[custom_splits['train'],:\] data_config = DataConfig( data_info={ "RNA": DataInfo(file_path=mrna_file), "METHYLATION": DataInfo(file_path=meth_file), "CLINICAL": DataInfo( file_path=clin_file, data_type="ANNOTATION" ), }, ) bulk_config = VarixConfig(data_config=data_config, annotation_columns=["CANCER_TYPE"], data_case=DataCase.MULTI_BULK, epochs=30, checkpoint_interval=10, k_filter=100) varix = acx.Varix(config=bulk_config, custom_splits=custom_splits) result_varix = varix.run() varix.show_result()

```
reading parquet: /home/janew/.cache/huggingface/hub/datasets--autoencodix--tcga/snapshots/c59ee679d11ea17c0fd615248a382ef1c38ef456/rna.parquet
```

```
reading parquet: /home/janew/.cache/huggingface/hub/datasets--autoencodix--tcga/snapshots/c59ee679d11ea17c0fd615248a382ef1c38ef456/methylation.parquet
```

```
reading parquet: /home/janew/.cache/huggingface/hub/datasets--autoencodix--tcga/snapshots/c59ee679d11ea17c0fd615248a382ef1c38ef456/clinical.parquet
```

```
Using custom splits in PairedUnpairedSplitter.
anno key: paired
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 13.9052 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 10 - Train Loss: 79.4287
Sub-losses: recon_loss: 79.3370, var_loss: 0.0917, anneal_factor: 0.0180, effective_beta_factor: 0.0018
Epoch 10 - Valid Loss: 70.3153
Sub-losses: recon_loss: 70.2222, var_loss: 0.0931, anneal_factor: 0.0180, effective_beta_factor: 0.0018
```

```
Epoch 20 - Train Loss: 75.5938
Sub-losses: recon_loss: 73.4769, var_loss: 2.1169, anneal_factor: 0.9350, effective_beta_factor: 0.0935
Epoch 20 - Valid Loss: 65.9533
Sub-losses: recon_loss: 64.0218, var_loss: 1.9315, anneal_factor: 0.9350, effective_beta_factor: 0.0935
```

```
Epoch 30 - Train Loss: 71.9623
Sub-losses: recon_loss: 69.8754, var_loss: 2.0870, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Epoch 30 - Valid Loss: 62.6252
Sub-losses: recon_loss: 60.6122, var_loss: 2.0130, anneal_factor: 0.9999, effective_beta_factor: 0.1000
Processed 32 / 1065 samples
Processed 64 / 1065 samples
Processed 96 / 1065 samples
Processed 128 / 1065 samples
Processed 160 / 1065 samples
Processed 192 / 1065 samples
Processed 224 / 1065 samples
Processed 256 / 1065 samples
Processed 288 / 1065 samples
Processed 320 / 1065 samples
Processed 352 / 1065 samples
Processed 384 / 1065 samples
Processed 416 / 1065 samples
Processed 448 / 1065 samples
Processed 480 / 1065 samples
Processed 512 / 1065 samples
Processed 544 / 1065 samples
Processed 576 / 1065 samples
Processed
... [output truncated]
```

```
Processed 1024 / 1065 samples
Processed 1056 / 1065 samples
Processed 1065 / 1065 samples
```

```
Creating plots ...
```

When we look at the three sets separately we can see that train and validation set contain all cancer types except breast cancer and the test set contains only breast cancer. Our custom split worked.

In \[16\]:

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

In \[17\]:

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

In \[18\]:

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
