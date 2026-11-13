> **Note:** This environment variable is required for fully deterministic CuBLAS ops on CUDA >= 10.2 when `reproducible=True` is set below. Without it, PyTorch raises a `RuntimeError` instead of training deterministically. It must be set before `torch` is imported. See the [README FAQ](https://jan-forest.github.io/autoencodix_package/tutorials/README.md#reproducibility-and-cublas_workspace_config) for details.

In \[ \]:

Copied!

```
%env CUBLAS_WORKSPACE_CONFIG=:16:8
```

%env CUBLAS_WORKSPACE_CONFIG=:16:8

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

In \[1\]:

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
Changed to: /Users/maximilianjoas/development/autoencodix_package
```

In \[ \]:

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
[ 6.30214     0.7521438   1.7172257  -0.9989515   5.5416946   6.0979557
 -1.0258414   1.5317484  -0.04637699  4.334748    0.31163976  0.3512562
  1.0720801  -0.54168016 -0.10310201  2.4067917 ]


--------------------------------------------------------------------------------
latentspace of one sample in train split at epoch 4: [ 2.1796706   0.60611284  0.18117622 -0.74476635  0.8137296   2.5618582
  2.2042491  -1.9576615  -1.4144119   0.5333763   0.9915411   0.44359267
 -0.09947515  0.13895065 -0.37461102  0.55542964]
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
array([[-0.78152806, -0.7762103 ,  4.448717  , ...,  1.779333  ,
        -1.209155  , -1.13602   ],
       [ 0.78200436, -0.6337442 ,  1.1372604 , ..., -0.47643963,
         1.0322978 , -0.09783195],
       [-0.22461136,  1.6985878 ,  3.380916  , ...,  2.5895224 ,
         1.4247885 ,  0.28271654],
       ...,
       [ 2.916834  , -0.281322  ,  1.6438428 , ..., -0.3873932 ,
         6.9723663 ,  0.6587008 ],
       [-0.80762166,  1.1656091 , -0.8296414 , ..., -1.0064985 ,
         7.8977246 , -0.944083  ],
       [ 6.3655725 ,  3.4816382 ,  3.0730734 , ...,  0.738921  ,
         3.650429  , -0.5597477 ]], shape=(646, 16), dtype=float32)
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
9.23
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
DatasetContainer(train=<autoencodix.data._numeric_dataset.NumericDataset object at 0x140b89070>, valid=<autoencodix.data._numeric_dataset.NumericDataset object at 0x140c243e0>, test=<autoencodix.data._numeric_dataset.NumericDataset object at 0x140c24c20>)
<autoencodix.data._numeric_dataset.NumericDataset object at 0x140b89070>
<class 'torch.Tensor'>
```

Out\[10\]:

|          | cell_type | batch  | donor  | cell_cycle | n_genes |
| -------- | --------- | ------ | ------ | ---------- | ------- |
| cell_0   | type_0    | batch2 | donor2 | S          | 346     |
| cell_1   | type_0    | batch3 | donor4 | G1         | 357     |
| cell_195 | type_4    | batch2 | donor2 | S          | 357     |
| cell_269 | type_4    | batch1 | donor1 | G2M        | 339     |
| cell_27  | type_4    | batch1 | donor4 | S          | 351     |

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
Index(['new_cell_0', 'new_cell_1', 'new_cell_107', 'new_cell_189',
       'new_cell_19', 'new_cell_190', 'new_cell_191', 'new_cell_192',
       'new_cell_193', 'new_cell_194',
       ...
       'new_cell_990', 'new_cell_991', 'new_cell_992', 'new_cell_993',
       'new_cell_994', 'new_cell_995', 'new_cell_996', 'new_cell_997',
       'new_cell_998', 'new_cell_999'],
      dtype='object', length=1000)
Index(['new_cell_0', 'new_cell_1', 'new_cell_107', 'new_cell_189',
       'new_cell_19', 'new_cell_190', 'new_cell_191', 'new_cell_192',
       'new_cell_193', 'new_cell_194',
       ...
       'new_cell_990', 'new_cell_991', 'new_cell_992', 'new_cell_993',
       'new_cell_994', 'new_cell_995', 'new_cell_996', 'new_cell_997',
       'new_cell_998', 'new_cell_999'],
      dtype='object', leng
... [output truncated]
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
| 0   | train       | CANCER_TYPE  | roc_auc_ovo | 0.930816 | LogisticRegression() | classification | multi_bulk.rna | Latent  | Latent\_$\_multi_bulk.rna |
| 1   | valid       | CANCER_TYPE  | roc_auc_ovo | 0.926246 | LogisticRegression() | classification | multi_bulk.rna | Latent  | Latent\_$\_multi_bulk.rna |
| 2   | test        | CANCER_TYPE  | roc_auc_ovo | 0.932935 | LogisticRegression() | classification | multi_bulk.rna | Latent  | Latent\_$\_multi_bulk.rna |
| 0   | train       | CANCER_TYPE  | roc_auc_ovo | 0.986654 | LogisticRegression() | classification | img.img        | Latent  | Latent\_$\_img.img        |
| 1   | valid       | CANCER_TYPE  | roc_auc_ovo | 0.976694 | LogisticRegression() | classification | img.img        | Latent  | Latent\_$\_img.img        |
| 2   | test        | CANCER_TYPE  | roc_auc_ovo | 0.980744 | LogisticRegression() | classification | img.img        | Latent  | Latent\_$\_img.img        |

In \[20\]:

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
Perform ML task with feature df: Latent
Latent
Perform ML task for target parameter: batch
```

Out\[20\]:

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
datasets: DatasetContainer(train=<autoencodix.data._numeric_dataset.NumericDataset object at 0x140b89070>, valid=<autoencodix.data._numeric_dataset.NumericDataset object at 0x140c243e0>, test=<autoencodix.data._numeric_dataset.NumericDa
... [output truncated]
```

## 5 Special Methods to Obtain DataFrames[¶](#5-special-methods-to-obtain-dataframes)

As you've seen in the [Training Dynamics Section](#2-trainingdynamics-interface-deep-dive), we only get plain values of reconstructions and latetnspaces. Often it is more useful to have sample ids, too. We could obtain the sample ids in the same order via the `sample_ids` TrainingDynamic. To make this more accessible, we added the methods `get_latent_df` and `get_reconstructions_df`. Here you pass `epoch` and `split` as seen before and you get a pandas DataFrame for the specific split and epoch for the latent space or the reconstruction.

In \[21\]:

Copied!

```
result.get_latent_df(epoch=-1, split="test").head()
```

result.get_latent_df(epoch=-1, split="test").head()

Out\[21\]:

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
7.326564
-1.073827
0.268320
2.035412
5.388091
4.979861
1.124839
0.911935
-0.284716
5.384183
-1.287263
0.413481
2.850581
-0.280163
-1.262128
2.862012


new_cell_1
8.377705
0.078919
-0.083633
0.767228
5.919516
5.684262
0.185831
3.151647
-0.802761
6.607008
0.175018
1.661315
1.336131
-1.485604
0.535221
1.720066


new_cell_10
2.120787
1.665626
0.899663
-0.595117
0.981167
0.226507
0.077835
-2.504569
1.947884
-0.049883
-0.647825
0.119985
-0.367380
0.747782
0.245587
-1.272236


new_cell_100
0.397119
-0.726314
-2.079093
1.323053
2.802404
0.744250
-0.432267
-1.008114
-0.103993
-0.014099
-0.150142
-0.122557
0.128536
-1.062114
1.936
... [output truncated]
```

In \[22\]:

Copied!

```
result.get_reconstructions_df(epoch=-1, split="test").head()
```

result.get_reconstructions_df(epoch=-1, split="test").head()

Out\[22\]:

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
1.320424
-0.772346
0.014947
1.532826
-0.318065
0.183227
0.027560
0.064022
-0.722645
-0.768205
...
-1.022267
1.740210
0.153457
-0.471486
0.059539
-1.215273
-0.381351
0.148173
-1.211444
1.394425


new_cell_1
1.285662
-0.763142
0.011849
1.498099
-0.305929
0.185441
0.015083
0.056283
-0.709083
-0.758961
...
-1.003222
1.696856
0.168587
-0.460411
0.066222
-1.185669
-0.360587
0.144142
-1.197535
1.368953


new_cell_10
-0.512781
0.637073
-0.010666
-0.435319
-0.297005
-0.545675
-0.012363
0.021127
0.567361
0.633929
...
0.852979
-0.450144
-0.094879
-0.448923
-0.007796
0.091232
-0.452214
-0.
... [output truncated]
```
