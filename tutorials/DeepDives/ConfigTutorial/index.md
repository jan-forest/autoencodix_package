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

# Config Deep Dive[¶](#config-deep-dive)

Our config class is the central point to customize our `AUTOENCODIX` pipelines. This notebook is more of a reference document than a tutorial, as we mainly list config parameters with explanations and default values and only include a few coding examples.

**IMPORTANT**

> This tutorial explains specific concepts of our config class. If you're unfamiliar with general concepts,\
> we recommend following the `Getting Started - Vanillix` Tutorial first.

## What You'll Learn[¶](#what-youll-learn)

We'll cover the following points:

- The two ways to provide a config:
  - as an instance of our config class
  - as a `YAML` file
- Which config parameters you should set
- The two parts of our config:
  - main config
  - data config
- Pipeline-specific config parameters
- Default config reference

## 1) How to Provide a Config[¶](#1-how-to-provide-a-config)

The main way is to pass an instance of our `DefaultConfig` class to the pipeline object, as you've seen many times in the pipeline tutorials.

#### 1.1 Provide Config as a Class Instance[¶](#11-provide-config-as-a-class-instance)

In \[2\]:

Copied!

```
import autoencodix as acx
from autoencodix.configs.default_config import DefaultConfig, DataCase
from autoencodix.configs import VarixConfig
from autoencodix.utils.example_data import EXAMPLE_MULTI_BULK
import yaml
from pathlib import Path

config = VarixConfig(
    latent_dim=8, scaling="MINMAX", data_case=DataCase.MULTI_BULK, epochs=10
)
varix = acx.Varix(config=config, data=EXAMPLE_MULTI_BULK)
```

import autoencodix as acx from autoencodix.configs.default_config import DefaultConfig, DataCase from autoencodix.configs import VarixConfig from autoencodix.utils.example_data import EXAMPLE_MULTI_BULK import yaml from pathlib import Path config = VarixConfig( latent_dim=8, scaling="MINMAX", data_case=DataCase.MULTI_BULK, epochs=10 ) varix = acx.Varix(config=config, data=EXAMPLE_MULTI_BULK)

```
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
```

#### 1.2 Provide the Config as a YAML file[¶](#12-provide-the-config-as-a-yaml-file)

In our GitHub repo, we prepared a directory called `configs` with sample YAML files.\
We can easily load the values into our config class with the `model_validate` method, as shown below:

#### ❗❗ Requirements: Getting Tutorial Data ❗❗[¶](#requirements-getting-tutorial-data)

The data for this tutorial is hosted on Hugging Face Hub ([autoencodix/tcga](https://huggingface.co/datasets/autoencodix/tcga)) and is downloaded automatically in the cell below on first run.

### Extra 2: Get correct path[¶](#extra-2-get-correct-path)

We assume you are in the root of the package. The following code ensures that the correct paths are used. [1] [Tutorials/DeepDives/ConfigTutorial.ipynb](https://jan-forest.github.io/autoencodix_package/tutorials/DeepDives/ConfigTutorial/ConfigTutorial.ipynb)

In \[3\]:

Copied!

```
# first be sure to be in root
import os

p = os.getcwd()
d = "autoencodix_package"
if d not in p:
    raise FileNotFoundError(f"'{d}' not found in path: {p}")
os.chdir(os.sep.join(p.split(os.sep)[: p.split(os.sep).index(d) + 1]))
print(f"Changed to: {os.getcwd()}")

# ---------------------------------------------------------------------
# Data is hosted on Hugging Face Hub and downloaded (and locally cached)
# automatically, then placed under the paths configs/multi_bulk.yaml expects
# ---------------------------------------------------------------------
import shutil
from huggingface_hub import hf_hub_download

HF_REPO_ID = "autoencodix/tcga"
os.makedirs("data/raw", exist_ok=True)
for hf_filename, local_name in [
    ("rna.parquet", "combined_rnaseq_formatted.parquet"),
    ("methylation.parquet", "combined_meth_formatted.parquet"),
    ("clinical.parquet", "combined_clin_formatted.parquet"),
]:
    downloaded_path = hf_hub_download(
        repo_id=HF_REPO_ID, repo_type="dataset", filename=hf_filename
    )
    shutil.copyfile(downloaded_path, os.path.join("data/raw", local_name))

# now we can load the config
custom_config = VarixConfig.model_validate(
    {
        **yaml.safe_load(Path("configs/multi_bulk.yaml").read_text()),
        "learning_rate": 0.77,
    }
)
# and pass to a pipeline
varix = acx.Varix(config=custom_config)
r = varix.run()
```

# first be sure to be in root

import os p = os.getcwd() d = "autoencodix_package" if d not in p: raise FileNotFoundError(f"'{d}' not found in path: {p}") os.chdir(os.sep.join(p.split(os.sep)[: p.split(os.sep).index(d) + 1])) print(f"Changed to: {os.getcwd()}")

# ---------------------------------------------------------------------

# Data is hosted on Hugging Face Hub and downloaded (and locally cached)

# automatically, then placed under the paths configs/multi_bulk.yaml expects

# ---------------------------------------------------------------------

import shutil from huggingface_hub import hf_hub_download HF_REPO_ID = "autoencodix/tcga" os.makedirs("data/raw", exist_ok=True) for hf_filename, local_name in \[ ("rna.parquet", "combined_rnaseq_formatted.parquet"), ("methylation.parquet", "combined_meth_formatted.parquet"), ("clinical.parquet", "combined_clin_formatted.parquet"), \]: downloaded_path = hf_hub_download( repo_id=HF_REPO_ID, repo_type="dataset", filename=hf_filename ) shutil.copyfile(downloaded_path, os.path.join("data/raw", local_name))

# now we can load the config

custom_config = VarixConfig.model_validate( { \*\*yaml.safe_load(Path("configs/multi_bulk.yaml").read_text()), "learning_rate": 0.77, } )

# and pass to a pipeline

varix = acx.Varix(config=custom_config) r = varix.run()

```
Changed to: /home/janew/ACX/autoencodix_package
```

```
reading parquet: data/raw/combined_rnaseq_formatted.parquet
```

```
reading parquet: data/raw/combined_meth_formatted.parquet
```

```
reading parquet: data/raw/combined_clin_formatted.parquet
```

```
anno key: paired
```

```
You are using a CUDA device ('NVIDIA GeForce RTX 5060 Ti') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:295: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 11.6709 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 3 - Train Loss: 244294676.3142
Sub-losses: recon_loss: 632267.9116, var_loss: 243662407.5660, anneal_factor: 0.9656, effective_beta_factor: 0.0966
```

```
Epoch 3 - Valid Loss: 330077893.8182
Sub-losses: recon_loss: 95851125.8182, var_loss: 234226768.0000, anneal_factor: 0.9656, effective_beta_factor: 0.0966
```

```
Processed 32 / 707 samples
Processed 64 / 707 samples
Processed 96 / 707 samples
Processed 128 / 707 samples
Processed 160 / 707 samples
Processed 192 / 707 samples
Processed 224 / 707 samples
Processed 256 / 707 samples
```

```
Processed 288 / 707 samples
Processed 320 / 707 samples
Processed 352 / 707 samples
Processed 384 / 707 samples
Processed 416 / 707 samples
Processed 448 / 707 samples
Processed 480 / 707 samples
Processed 512 / 707 samples
Processed 544 / 707 samples
```

```
Processed 576 / 707 samples
Processed 608 / 707 samples
Processed 640 / 707 samples
Processed 672 / 707 samples
Processed 704 / 707 samples
Processed 707 / 707 samples
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/visualize/_general_visualizer.py:64: UserWarning: Model weights plot is skipped since input dimension 27277 is larger than 3000 and heatmap would be too large.
  warnings.warn(
```

**Note**

> We can also go the reverse way and save our config object to a YAML file, as shown below:

In \[4\]:

Copied!

```
output_path = Path("tutorial_res") / "default_config.yaml"
output_path.parent.mkdir(exist_ok=True)

# Convert to plain Python dict first
config_dict = config.model_dump()
# Write YAML with nice formatting
with output_path.open("w") as f:
    yaml.dump(config_dict, f, sort_keys=False, default_flow_style=False)

print(f"✅ Saved config to {output_path.resolve()}")
```

output_path = Path("tutorial_res") / "default_config.yaml" output_path.parent.mkdir(exist_ok=True)

# Convert to plain Python dict first

config_dict = config.model_dump()

# Write YAML with nice formatting

with output_path.open("w") as f: yaml.dump(config_dict, f, sort_keys=False, default_flow_style=False) print(f"✅ Saved config to {output_path.resolve()}")

```
✅ Saved config to /home/janew/ACX/autoencodix_package/tutorial_res/default_config.yaml
```

## 2) Which Config Parameter to Set[¶](#2-which-config-parameter-to-set)

To make **AUTOENCODIX** easily usable, we provide sensible defaults per pipeline, so you don't need to think of and set 30+ config params before starting.\
However, there are a few parameters you should consider — and depending on the pipeline and data type, a few mandatory ones you’ll need to set.

### 2.1) Mandatory Config Parameters[¶](#21-mandatory-config-parameters)

*(These depend on your pipeline — see specific documentation for details.)*

### 2.2) Parameters to Consider[¶](#22-parameters-to-consider)

Although we set sensible defaults per pipeline for these parameters, it might make sense to adjust the following values:

| **Parameter**         | **Type**                                                           | **Description**                                                                        |
| --------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| `latent_dim`          | `int`                                                              | Dimension of the latent space                                                          |
| `epochs`              | `int`                                                              | Number of training epochs                                                              |
| `batch_size`          | `int`                                                              | Number of samples per batch (must be >1 due to BatchNorm)                              |
| `filtering`           | `Literal['VAR', 'MAD', 'CORR', 'VARCORR', 'NOFILT', 'NONZEROVAR']` | Feature filtering method                                                               |
| `scaling`             | `Literal['STANDARD', 'MINMAX', 'ROBUST', 'MAXABS', 'NONE']`        | How to scale your input data; should be compatible with your loss function             |
| `k_filter`            | `Optional[int]`                                                    | Number of features to keep                                                             |
| `learning_rate`       | `float`                                                            | Learning rate for optimization                                                         |
| `reconstruction_loss` | `Literal['mse', 'bce']`                                            | Type of reconstruction loss: `mse` = Mean Squared Error, `bce` = Binary Cross-Entropy  |
| `default_vae_loss`    | `Literal['kl', 'mmd']`                                             | Type of VAE loss: `kl` = Kullback-Leibler Divergence, `mmd` = Maximum Mean Discrepancy |
| `beta`                | `float`                                                            | β weighting factor for VAE loss                                                        |

## 3) General Config and Data Config[¶](#3-general-config-and-data-config)

Our config class consists of two parts: (a) a general part where we set global parameters, and (b) a `DataConfig` where we can set parameters for each data type individually.\
Inside the `DataConfig` we have a `DataInfo` object where we can set the parameters; see a minimal code example below.\
You can find a full list of parameters in [section 5.1](#51-dataconfig--configuration-parameters).

In \[5\]:

Copied!

```
import os
import autoencodix as acx
from autoencodix.configs.default_config import DataConfig, DataInfo, DataCase
from autoencodix.configs import VarixConfig
from autoencodix.utils.example_data import raw_protein, raw_rna, annotation
from autoencodix.data import DataPackage


my_dp = DataPackage(
    multi_bulk={"rna": raw_rna, "protein": raw_protein}, annotation={"anno": annotation}
)
data_config = DataConfig(
    data_info={
        "rna": DataInfo(scaling="MINMAX", data_type="NUMERIC"),
        "protein": DataInfo(scaling="MINMAX", data_type="NUMERIC"),
    },
)

bulk_config = VarixConfig(data_config=data_config, data_case=DataCase.MULTI_BULK, epochs=30)
varix = acx.Varix(config=bulk_config, data=my_dp)
```

import os import autoencodix as acx from autoencodix.configs.default_config import DataConfig, DataInfo, DataCase from autoencodix.configs import VarixConfig from autoencodix.utils.example_data import raw_protein, raw_rna, annotation from autoencodix.data import DataPackage my_dp = DataPackage( multi_bulk={"rna": raw_rna, "protein": raw_protein}, annotation={"anno": annotation} ) data_config = DataConfig( data_info={ "rna": DataInfo(scaling="MINMAX", data_type="NUMERIC"), "protein": DataInfo(scaling="MINMAX", data_type="NUMERIC"), }, ) bulk_config = VarixConfig(data_config=data_config, data_case=DataCase.MULTI_BULK, epochs=30) varix = acx.Varix(config=bulk_config, data=my_dp)

```
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
```

## 4) Pipeline Specific Configs[¶](#4-pipeline-specific-configs)

We have one `DefaultConfig` class that defines all configurable parameters and sets sensible defaults.\
However, not all pipelines (Varix, XModalix, etc.) should have the same sensible defaults.\
Thus, each pipeline has its own config class that inherits from `DefaultConfig` and overrides some default values.\
Be aware that when you build a config object, you use the appropriate config for the pipeline.\
The config can be imported from `autoencodix.configs`. The naming convention is `<pipeline-name>Config`, for example: `VarixConfig` or `OntixConfig`.

## 5) Configuration Parameters Reference[¶](#5-configuration-parameters-reference)

We list all configurable parameters sorted into functional sections below:

#### **Data Configuration**[¶](#data-configuration)

| **Parameter**        | **Type**                                                    | **Description**                                                                     |
| -------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `data_config`        | `autoencodix.configs.default_config.DataConfig`             | Contains detailed information about each data modality (see `DataConfig` section)   |
| `img_path_col`       | `str`                                                       | When working with images, defines the column name containing image paths per sample |
| `requires_paired`    | `Optional[bool]`                                            | Indicates whether samples for xmodalix are paired (based on sample ID)              |
| `data_case`          | `Optional[DataCase]`                                        | Data case for the model (auto-determined)                                           |
| `k_filter`           | `Optional[int]`                                             | Number of features to keep                                                          |
| `scaling`            | `Literal['STANDARD', 'MINMAX', 'ROBUST', 'MAXABS', 'NONE']` | Global scaling setting (can be overridden per modality)                             |
| `skip_preprocessing` | `bool`                                                      | Skip scaling, filtering, and cleaning                                               |
| `class_param`        | `Optional[str]`                                             | Optional column name for class labels                                               |

#### **Model Architecture**[¶](#model-architecture)

| **Parameter** | **Type** | **Description**                                           |
| ------------- | -------- | --------------------------------------------------------- |
| `latent_dim`  | `int`    | Dimension of the latent space                             |
| `n_layers`    | `int`    | Number of encoder/decoder layers (excluding latent layer) |
| `enc_factor`  | `int`    | Encoder dimension scaling factor                          |
| `input_dim`   | `int`    | Input feature dimension                                   |
| `drop_p`      | `float`  | Dropout probability                                       |
| `save_memory` | `bool`   | Skip storing `TrainingDynamics` to save memory            |

#### **Training Hyperparameters**[¶](#training-hyperparameters)

| **Parameter**         | **Type**                                                                                                                       | **Description**                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------- |
| `learning_rate`       | `float`                                                                                                                        | Learning rate for optimization                              |
| `batch_size`          | `int`                                                                                                                          | Samples per batch (>1 required due to BatchNorm)            |
| `epochs`              | `int`                                                                                                                          | Number of training epochs                                   |
| `weight_decay`        | `float`                                                                                                                        | L2 regularization factor                                    |
| `reconstruction_loss` | `Literal['mse', 'bce']`                                                                                                        | Type of reconstruction loss                                 |
| `default_vae_loss`    | `Literal['kl', 'mmd']`                                                                                                         | Type of VAE loss                                            |
| `loss_reduction`      | `Literal['sum', 'mean']`                                                                                                       | Loss reduction mode in PyTorch                              |
| `beta`                | `float`                                                                                                                        | β weight for VAE loss                                       |
| `beta_mi`             | `float`                                                                                                                        | β weight for mutual information term                        |
| `beta_tc`             | `float`                                                                                                                        | β weight for total correlation term                         |
| `beta_dimKL`          | `float`                                                                                                                        | β weight for dimension-wise KL                              |
| `use_mss`             | `bool`                                                                                                                         | Use minibatch stratified sampling for disentangled VAE loss |
| `gamma`               | `float`                                                                                                                        | γ weight for adversarial loss (XModalix classifier)         |
| `delta_pair`          | `float`                                                                                                                        | δ weight for paired loss (XModalix training)                |
| `delta_class`         | `float`                                                                                                                        | δ weight for class loss (XModalix training)                 |
| `anneal_function`     | `Literal['5phase-constant', '3phase-linear', '3phase-log', 'logistic-mid', 'logistic-early', 'logistic-late', 'no-annealing']` | Annealing function strategy for VAE loss                    |
| `pretrain_epochs`     | `int`                                                                                                                          | Number of pretraining epochs (can differ per modality)      |

______________________________________________________________________

#### **Device & Performance**[¶](#device-performance)

| **Parameter**         | **Type**                                                                                                                                                        | **Description**              |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| `device`              | `Literal['cpu', 'cuda', 'gpu', 'tpu', 'mps', 'auto']`                                                                                                           | Compute device               |
| `n_gpus`              | `int`                                                                                                                                                           | Number of GPUs to use        |
| `checkpoint_interval` | `int`                                                                                                                                                           | Checkpoint save interval     |
| `float_precision`     | `Literal['transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', '64', '32', '16', 'bf16']` | Floating-point precision     |
| `gpu_strategy`        | `Literal['auto', 'dp', 'ddp', 'ddp_spawn', 'ddp_find_unused_parameters_true', 'xla', 'deepspeed', 'fsdp']`                                                      | GPU parallelization strategy |

#### **Data Splits & Reproducibility**[¶](#data-splits-reproducibility)

| **Parameter**           | **Type** | **Description**           |
| ----------------------- | -------- | ------------------------- |
| `train_ratio`           | `float`  | Training split ratio      |
| `test_ratio`            | `float`  | Test split ratio          |
| `valid_ratio`           | `float`  | Validation split ratio    |
| `min_samples_per_split` | `int`    | Minimum samples per split |
| `reproducible`          | `bool`   | Ensure reproducibility    |
| `global_seed`           | `int`    | Global random seed        |

#### 5.1 DataConfig — Configuration Parameters[¶](#51-dataconfig-configuration-parameters)

______________________________________________________________________

##### **DataConfig**[¶](#dataconfig)

| **Parameter**          | **Type**              | **Description**                                                                             |
| ---------------------- | --------------------- | ------------------------------------------------------------------------------------------- |
| `data_info`            | `Dict[str, DataInfo]` | Dictionary mapping modality names (e.g. `"RNA"`, `"IMG"`) to their `DataInfo` configuration |
| `require_common_cells` | `Optional[bool]`      | Whether to require that all data modalities share a common set of cells/samples             |
| `annotation_columns`   | `Optional[List[str]]` | List of column names from the annotation file to include as metadata                        |

###### **DataInfo**[¶](#datainfo)

| **Parameter**     | **Type**                                                              | **Description**                                                 |
| ----------------- | --------------------------------------------------------------------- | --------------------------------------------------------------- |
| `file_path`       | `str`                                                                 | Path to the raw data file                                       |
| `data_type`       | `Literal['NUMERIC', 'CATEGORICAL', 'IMG', 'ANNOTATION']`              | Type of data modality                                           |
| `scaling`         | `Literal['STANDARD', 'MINMAX', 'ROBUST', 'MAXABS', 'NONE', 'NOTSET']` | Overrides the globally set scaling method for this modality     |
| `filtering`       | `Literal['VAR', 'MAD', 'CORR', 'VARCORR', 'NOFILT', 'NONZEROVAR']`    | Feature filtering method                                        |
| `sep`             | `Optional[str]`                                                       | Delimiter for CSV/TSV input files (passed to `pandas.read_csv`) |
| `extra_anno_file` | `Optional[str]`                                                       | Path to an additional annotation file                           |

**Single-Cell Specific Parameters**

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

**Image-Specific Parameters**

| **Parameter**       | **Type**        | **Description**                                     |
| ------------------- | --------------- | --------------------------------------------------- |
| `img_width_resize`  | `Optional[int]` | Target width for image resizing (must equal height) |
| `img_height_resize` | `Optional[int]` | Target height for image resizing (must equal width) |

**XModalix & Translation Parameters**

| **Parameter**         | **Type**                          | **Description**                                                                               |
| --------------------- | --------------------------------- | --------------------------------------------------------------------------------------------- |
| `translate_direction` | `Optional[Literal['from', 'to']]` | Defines translation direction in cross-modal (XModalix) training                              |
| `pretrain_epochs`     | `int`                             | Number of pretraining epochs specific to this modality (overrides global pretraining setting) |

**Validation Rules**

- `selected_layers` must always contain `"X"`.
- `img_width_resize` and `img_height_resize` must be **positive integers** and **equal** (enforces square resizing).

## 6) List Config Parameters Dynamically[¶](#6-list-config-parameters-dynamically)

If you want to see all available parameters directly via Python, you can call `<config-instance>.print_schema()` as shown below.

In \[6\]:

Copied!

```
config = DefaultConfig()
config.print_schema()
```

config = DefaultConfig() config.print_schema()

```
DefaultConfig Configuration Parameters:
--------------------------------------------------

data_config:
  Type: <class 'autoencodix.configs.default_config.DataConfig'>
  Default: data_info={} require_common_cells=False annotation_columns=None
  Description: Per-modality data settings; filled automatically from the DataPackage if not set

annotation_columns:
  Type: typing.Optional[typing.List[str]]
  Default: None
  Description: Metadata columns used to color plots and as default targets in evaluate()

img_path_col:
  Type: <class 'str'>
  Default: img_paths
  Description: When working with images, we except a column in your annotation file that specifies the path of the image for a particular sample. Here you can define the name of this column

requires_paired:
  Type: typing.Optional[
... [output truncated]
```
