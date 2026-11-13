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

# How To Use Maskix[¶](#how-to-use-maskix)

Maskix is our implementation of a variational autoencoder.\
This tutorial follows the structure of our `Getting Started - Vanillix`, but is much less extensive, because\
our pipeline works similarly for different architectures, so here we focus only on Maskix specifics.

**AUTOENCODIX** supports far more functionality than shown here, so we’ll also point to advanced tutorials where relevant.

**IMPORTANT**

> This tutorial only shows the specifics of the Maskix pipeline. If you're unfamiliar with general concepts,\
> we recommend following the [`Getting Started - Vanillix`](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/Maskix/Vanillix.ipynb) tutorial first.

## What You'll Learn[¶](#what-youll-learn)

You’ll learn how to:

1. **Theory primer** of the Maskix architecture.
1. **Initialize** the pipeline and run the pipeline.
1. Understand Maskix-specific \**config parameters*.
1. **Access & Visualize** the results effectively.
1. Pass **custom masking fucntions**.
1. How to use Maskix for **data imputation**
1. **Save, load, and reuse** a trained pipeline.

## 1) Theory Primer[¶](#1-theory-primer)

Maskix adapts the scMAE (single-cell Masked Autoencoder) framework from Fang et al. (2024) for single-cell RNA-seq analysis. The model learns useful representations by corrupting the input expression matrix and training the network both to reconstruct the original data and to identify which entries were perturbed. This encourages the model to capture gene–gene (or feature–feature) relationships in high-dimensional, noisy datasets. Although originally designed for single-cell data, the same approach applies to other domains.

Each training iteration follows this corruption process:

1. Sample a Bernoulli distribution to determine which entries should be perturbed, producing a binary mask with the same shape as the input.
1. For each gene (or feature), generate a random permutation of sample indices.
1. For all positions marked as masked, replace the original value with the value from the permuted index for that feature.

The corrupted input is encoded into a low-dimensional latent representation. In parallel, a mask predictor takes the latent space as input and estimates which positions were masked. The predicted mask is then concatenated with the latent representation and passed into a decoder that attempts to reconstruct the original values.

Training uses two loss components. The mask predictor is optimized with binary cross-entropy against the true mask. Reconstruction uses a weighted mean-squared error where masked entries receive higher weight, controlled by the hyperparameter `delta_mask_corrupted`. The total loss is the weighted sum of these components, balanced by the hyperparameter `delta_mask_predictor`.

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
Changed to: /home/alicia/dev/biomarker_autoencoder/autoencodix_package
```

## Requirements 2: Obtain tutorial data or use own data[¶](#requirements-2-obtain-tutorial-data-or-use-own-data)

We use a dataset from [Fang et al. (2024)](https://doi.org/10.1093/bioinformatics/btae020), originally the human pancreas scRNA-seq data from Baron et al. (2016), GEO accession GSE84133. It is hosted on Hugging Face Hub ([autoencodix/pancreas-fang](https://huggingface.co/datasets/autoencodix/pancreas-fang)) and is downloaded automatically in the cells below on first run — no manual download or placement into `data/raw` needed. Alternatively you can use your own single-cell dataset (Maskix works with other datatypes, but in this example we expect single cell data: `h5ad`) file and replace the variable `sc_path`.

## 2) Initialize and Run Maskix[¶](#2-initialize-and-run-maskix)

As for every other pipline, we need to perform the following steps:

- import relevant classes
- define a config
- init the pipeline
- call the run step

In \[3\]:

Copied!

```
import autoencodix as acx
from autoencodix.configs import MaskixConfig
from autoencodix.configs.default_config import DataInfo, DataConfig, DataCase

from huggingface_hub import hf_hub_download

sc_path = hf_hub_download(
    repo_id="autoencodix/pancreas-fang", repo_type="dataset", filename="pancreas.h5ad"
)
config = MaskixConfig(
    epochs=30,
    checkpoint_interval=10,
    k_filter=1000,
    batch_size=128,
    annotation_columns=["assigned_cluster"],
    data_config=DataConfig(        
        data_info={
            "multi_sc": DataInfo(
                file_path=sc_path, is_single_cell=True, data_type="NUMERIC"
            )
        },
    ),
    data_case=DataCase.MULTI_SINGLE_CELL,
)
maskix= acx.Maskix(config=config)
```

import autoencodix as acx from autoencodix.configs import MaskixConfig from autoencodix.configs.default_config import DataInfo, DataConfig, DataCase from huggingface_hub import hf_hub_download sc_path = hf_hub_download( repo_id="autoencodix/pancreas-fang", repo_type="dataset", filename="pancreas.h5ad" ) config = MaskixConfig( epochs=30, checkpoint_interval=10, k_filter=1000, batch_size=128, annotation_columns=["assigned_cluster"], data_config=DataConfig( data_info={ "multi_sc": DataInfo( file_path=sc_path, is_single_cell=True, data_type="NUMERIC" ) }, ), data_case=DataCase.MULTI_SINGLE_CELL, ) maskix= acx.Maskix(config=config)

In \[4\]:

Copied!

```
maskix_result = maskix.run()
```

maskix_result = maskix.run()

```
Number of common cells: 8569
mudata: View of MuData object with n_obs × n_vars = 8569 × 20125
  obs:	'barcode', 'assigned_cluster', 'sample_id'
  1 modality
    multi_sc:	8569 x 20125
      obs:	'barcode', 'assigned_cluster', 'sample_id'
Processing 1 MuData objects: ['multi_sc']
Processing train modality: multi_sc
Processing valid split
Processing valid modality: multi_sc
Processing test split
Processing test modality: multi_sc
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/losses/maskix_loss.py:38: UserWarning: You chose loss reduction: sum, this deviates from the implementation in the literature for this architecture, the authors used 'mean'
  warnings.warn(
```

```
Epoch 10 - Train Loss: 351.5225
Sub-losses: recon_loss: 867.5084, recon_loss_weighted: 87.4652, mask_loss: 264.0573
Epoch 10 - Valid Loss: 351.4917
Sub-losses: recon_loss: 853.3143, recon_loss_weighted: 87.3680, mask_loss: 264.1237
Epoch 20 - Train Loss: 349.6205
Sub-losses: recon_loss: 846.8369, recon_loss_weighted: 86.0222, mask_loss: 263.5983
Epoch 20 - Valid Loss: 347.1490
Sub-losses: recon_loss: 830.9007, recon_loss_weighted: 83.8384, mask_loss: 263.3106
Epoch 30 - Train Loss: 348.0189
Sub-losses: recon_loss: 833.5078, recon_loss_weighted: 84.7925, mask_loss: 263.2265
Epoch 30 - Valid Loss: 347.1538
Sub-losses: recon_loss: 823.3418, recon_loss_weighted: 84.1736, mask_loss: 262.9802
Processed 1715 / 1715 samples
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/losses/maskix_loss.py:38: UserWarning: You chose loss reduction: sum, this deviates from the implementation in the literature for this architecture, the authors used 'mean'
  warnings.warn(
```

## 3) Understanding Maskix-specific Config Parameters[¶](#3-understanding-maskix-specific-config-parameters)

As described in the [theory section](#1-theory-primer), we implemented the architecture from [Fang et al. (2024)](https://doi.org/10.1093/bioinformatics/btae020), with weighted loss terms and a specific encoder–decoder design. To make this adaptable within our framework, we expose the following configuration parameters:

- **maskix_hidden_dim**: Hidden dimension used in the Maskix encoder and decoder, matching the scMAE reference architecture by default.
- **maskix_swap_prob**: Bernoulli probability controlling how often feature values are swapped during input corruption.
- **delta_mask_predictor**: Weighting factor for the mask prediction loss in the total training objective.
- **delta_mask_corrupted**: Weighting factor that increases the reconstruction penalty on corrupted entries.
- **maskix_architecture**: Selects between the default scMAE architecture or a custom architecture configured through `n_layers` and `enc_factor`.

You can find the default values by running the following:

In \[5\]:

Copied!

```
MaskixConfig.print_schema(
    filter_params=[
        "maskix_hidden_dim",
        "maskix_swap_prob",
        "delta_mask_predictor",
        "delta_mask_corrupted",
        "maskix_architecture",
    ]
)
```

MaskixConfig.print_schema( filter_params=[ "maskix_hidden_dim", "maskix_swap_prob", "delta_mask_predictor", "delta_mask_corrupted", "maskix_architecture", ] )

```
Valid Keyword Arguments:
--------------------------------------------------

maskix_hidden_dim:
  Type: <class 'int'>
  Default: 128
  Description: The Maskix implementation follows https://doi.org/10.1093/bioinformatics/btae020. The authors use a hidden dimension 0f 256 for their neural network, so we set this as default

maskix_swap_prob:
  Type: <class 'float'>
  Default: 0.2
  Description: For the Maskix input_data masinkg, we sample a probablity if samples within one gene should be swapt. This is done with a Bernoulli distribution, maskix_swap_prob is the probablity passed to the bernoulli distribution 

delta_mask_predictor:
  Type: <class 'float'>
  Default: 0.7
  Description: Delta weighting factor of the mask prediction loss term for the Maskix

delta_mask_corrupted:
  Type: <cla
... [output truncated]
```

## 4) Access & Visualize Results Effectively[¶](#4-access-visualize-results-effectively)

In addition to the results that the Vanillix pipeline provided, we can access:

- `total`, `reconstruction`, and `masked` losses

A note on the different loss types:\
For our maksed autoencoder, the total loss consists of a reconstruction loss and a mask predictor los To investigate these losses, the `result` object has the attribute `sub_losses`.\
This is a `LossRegistry` with the name of the loss as the key, and the value is a `TrainingDynamics` object, which can be accessed in the same way as for the Vanillix results.

For more details, check [`Tutorials/DeepDives/PipelineOutputTutorial.ipynb`](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/DeepDives/PipelineOutputTutorial.ipynb).

In \[6\]:

Copied!

```
sub_losses = maskix_result.sub_losses
print("Sub Losses:")
print(f"keys: {sub_losses.keys()}")
print("\n")
recon_dyn = sub_losses.get(key="recon_loss")
print("Value of reconstruction loss in epoch 4 for train split")
print(recon_dyn.get(split="train", epoch=4))
```

sub_losses = maskix_result.sub_losses print("Sub Losses:") print(f"keys: {sub_losses.keys()}") print("\\n") recon_dyn = sub_losses.get(key="recon_loss") print("Value of reconstruction loss in epoch 4 for train split") print(recon_dyn.get(split="train", epoch=4))

```
Sub Losses:
keys: dict_keys(['recon_loss', 'recon_loss_weighted', 'mask_loss'])


Value of reconstruction loss in epoch 4 for train split
886.0941694106368
```

As for our other pipelines we can visualize the loss with `.show_result()` For more infos on visualization, see: [`Tutorials/DeepDives/VisualizeTutorial.ipynb`](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/DeepDives/VisualizeTutorial.ipynb)

In \[7\]:

Copied!

```
maskix.show_result()
```

maskix.show_result()

```
Creating plots ...
```

## 5) Adding a Custom Masking Function to `MaskixTrainer`[¶](#5-adding-a-custom-masking-function-to-maskixtrainer)

`MaskixTrainer` supports replacing its default corruption mechanism with a user-defined masking function. This enables experimentation with alternative masking strategies while ensuring compatibility with the trainer’s data flow.

#### How to Add a Custom Masking Function[¶](#how-to-add-a-custom-masking-function)

Provide your masking function at initialization:

```
# We assume that config is defined and other imports are done (see above)

def my_masking_fn(x: torch.Tensor, strength: float = 0.2):
    noise = torch.randn_like(x) * strength
    return x + noise  # must return ONLY a single tensor in shape of input tensor

masking_fn_kwargs = {"strength": 0.1}
maskix = acx.Maskix(config=config, masking_fn, masking_fn_kwargs
)
```

#### Requirements for a Custom Masking Function[¶](#requirements-for-a-custom-masking-function)

A custom masking function must satisfy the following constraints:

1. **It must accept a `torch.Tensor` as the first positional argument.**\
   The trainer passes the input mini-batch `X` directly into the function.
1. **It must return exactly one value: a `torch.Tensor`.**\
   The trainer does not consume or propagate additional outputs.\
   Returning tuples or multiple values is not allowed.
1. **The returned tensor must have the same shape as the input tensor.**\
   Any shape mismatch will raise a validation error.
1. **The function must operate on the device of the input tensor.**\
   The function must not assume the tensor resides on the CPU; it must operate on the device of `x`.
1. **Any additional parameters must be passed via `masking_fn_kwargs`.**\
   These keyword arguments provide a clean separation between trainer configuration and masking logic.

##### Example[¶](#example)

This is our default masking method:

```
    def _maskix_hook(
        self, X: torch.Tensor
    ) -> torch.Tensor
        # expand probablities for bernoulli sampling to match input shape
        probs = self._mask_probas.expand(X.shape)

        # Create the Boolean Mask (1 = Swap, 0 = Keep)
        should_swap = torch.bernoulli(probs).bool()

        # COLUMN-WISE SHUFFLING
        # We generate a random float matrix and argsort it along dim=0.
        # This gives us independent random indices for every column.
        rand_indices = torch.rand(X.shape, device=X.device).argsort(dim=0)

        # Use gather to reorder X based on these random indices
        shuffled_X = torch.gather(X, 0, rand_indices)
        corrupted_X = torch.where(should_swap, shuffled_X, X)

        return corrupted_X
```

##### Code Example[¶](#code-example)

In \[8\]:

Copied!

```
import torch
import autoencodix as acx
from autoencodix.configs import MaskixConfig
from autoencodix.configs.default_config import DataInfo, DataConfig, DataCase


def my_masking_fn(x: torch.Tensor, strength: float = 0.2):
    # Noise is created with the same shape, dtype, and device as `x`
    # Because of randn_lie, if you use other function, take care of 
    # device and dtype casting.
    noise = torch.randn_like(x) * strength
    return x + noise


kwargs = {"strength": 0.5}


from huggingface_hub import hf_hub_download

sc_path = hf_hub_download(
    repo_id="autoencodix/pancreas-fang", repo_type="dataset", filename="pancreas.h5ad"
)
config = MaskixConfig(
    epochs=30,
    checkpoint_interval=10,
    k_filter=1000,
    batch_size=128,
    data_config=DataConfig(
        annotation_columns=["assigned_cluster"],
        data_info={
            "multi_sc": DataInfo(
                file_path=sc_path, is_single_cell=True, data_type="NUMERIC"
            )
        },
    ),
    data_case=DataCase.MULTI_SINGLE_CELL,
)
maskix = acx.Maskix(config=config, masking_fn=my_masking_fn, masking_fn_kwargs=kwargs)
```

import torch import autoencodix as acx from autoencodix.configs import MaskixConfig from autoencodix.configs.default_config import DataInfo, DataConfig, DataCase def my_masking_fn(x: torch.Tensor, strength: float = 0.2):

# Noise is created with the same shape, dtype, and device as `x`

# Because of randn_lie, if you use other function, take care of

# device and dtype casting.

noise = torch.randn_like(x) * strength return x + noise kwargs = {"strength": 0.5} from huggingface_hub import hf_hub_download sc_path = hf_hub_download( repo_id="autoencodix/pancreas-fang", repo_type="dataset", filename="pancreas.h5ad" ) config = MaskixConfig( epochs=30, checkpoint_interval=10, k_filter=1000, batch_size=128, data_config=DataConfig( annotation_columns=["assigned_cluster"], data_info={ "multi_sc": DataInfo( file_path=sc_path, is_single_cell=True, data_type="NUMERIC" ) }, ), data_case=DataCase.MULTI_SINGLE_CELL, ) maskix = acx.Maskix(config=config, masking_fn=my_masking_fn, masking_fn_kwargs=kwargs)

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/configs/default_config.py:439: UserWarning: annotation_columns in DataConfig is deprecated. Please set it directly in DefaultConfig instead.
  warnings.warn(
```

In \[9\]:

Copied!

```
result = maskix.run()
```

result = maskix.run()

```
Number of common cells: 8569
mudata: View of MuData object with n_obs × n_vars = 8569 × 20125
  obs:	'barcode', 'assigned_cluster', 'sample_id'
  1 modality
    multi_sc:	8569 x 20125
      obs:	'barcode', 'assigned_cluster', 'sample_id'
Processing 1 MuData objects: ['multi_sc']
Processing train modality: multi_sc
Processing valid split
Processing valid modality: multi_sc
Processing test split
Processing test modality: multi_sc
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/losses/maskix_loss.py:38: UserWarning: You chose loss reduction: sum, this deviates from the implementation in the literature for this architecture, the authors used 'mean'
  warnings.warn(
```

```
Epoch 10 - Train Loss: 214.1537
Sub-losses: recon_loss: 897.8995, recon_loss_weighted: 202.0274, mask_loss: 12.1264
Epoch 10 - Valid Loss: 210.4761
Sub-losses: recon_loss: 883.7049, recon_loss_weighted: 198.8336, mask_loss: 11.6424
Epoch 20 - Train Loss: 207.9269
Sub-losses: recon_loss: 882.6679, recon_loss_weighted: 198.6003, mask_loss: 9.3267
Epoch 20 - Valid Loss: 204.4001
Sub-losses: recon_loss: 866.8165, recon_loss_weighted: 195.0337, mask_loss: 9.3664
Epoch 30 - Train Loss: 203.9368
Sub-losses: recon_loss: 867.9988, recon_loss_weighted: 195.2997, mask_loss: 8.6370
Epoch 30 - Valid Loss: 201.3682
Sub-losses: recon_loss: 856.2013, recon_loss_weighted: 192.6453, mask_loss: 8.7229
Processed 1715 / 1715 samples
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/losses/maskix_loss.py:38: UserWarning: You chose loss reduction: sum, this deviates from the implementation in the literature for this architecture, the authors used 'mean'
  warnings.warn(
```

## 6) Use Maskix to Impute Data[¶](#6-use-maskix-to-impute-data)

You can also input corrupted/missing data and use `Maskix` to impute the data. Here we recommend using a custom masking function. For example, if you want to impute missing values, an imputer could randomly replace values with zeros.

Then you could use `Maskix` to clean your data and use the cleaned data to run your analysis, for example another autoencodix pipeline or anything else. In our mock example we will do the following:

- Create corrupted data with missing values
- Remove the "missing" data
- Train Maskix with clean data, but with a custom imputer that mimics missing data
- Feed corrupted data into trained Maskix and obtain imputed data
- Train Varix with:
  - Original data with missing values
  - Imputed data
- Compare results

#### Create Corrupted Data[¶](#create-corrupted-data)

We will use our single-cell example from before and use maskix to preprocess the data, which makes the artificall corruption process more robust, because we make sure to corrupt informative features/samples.

## [¶](#)

In \[10\]:

Copied!

```
import torch
import autoencodix as acx
from autoencodix.configs import MaskixConfig
from autoencodix.configs.default_config import DataInfo, DataConfig, DataCase


from huggingface_hub import hf_hub_download

sc_path = hf_hub_download(
    repo_id="autoencodix/pancreas-fang", repo_type="dataset", filename="pancreas.h5ad"
)
config = MaskixConfig(
    epochs=30,
    checkpoint_interval=10,
    k_filter=1000,
    batch_size=128,
    data_config=DataConfig(
        annotation_columns=["assigned_cluster"],
        data_info={
            "multi_sc": DataInfo(
                file_path=sc_path, is_single_cell=True, data_type="NUMERIC"
            )
        },
    ),
    data_case=DataCase.MULTI_SINGLE_CELL,
)
maskix_orig = acx.Maskix(config=config)
maskix_orig.preprocess()
data = maskix_orig.result.datasets
```

import torch import autoencodix as acx from autoencodix.configs import MaskixConfig from autoencodix.configs.default_config import DataInfo, DataConfig, DataCase from huggingface_hub import hf_hub_download sc_path = hf_hub_download( repo_id="autoencodix/pancreas-fang", repo_type="dataset", filename="pancreas.h5ad" ) config = MaskixConfig( epochs=30, checkpoint_interval=10, k_filter=1000, batch_size=128, data_config=DataConfig( annotation_columns=["assigned_cluster"], data_info={ "multi_sc": DataInfo( file_path=sc_path, is_single_cell=True, data_type="NUMERIC" ) }, ), data_case=DataCase.MULTI_SINGLE_CELL, ) maskix_orig = acx.Maskix(config=config) maskix_orig.preprocess() data = maskix_orig.result.datasets

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/configs/default_config.py:439: UserWarning: annotation_columns in DataConfig is deprecated. Please set it directly in DefaultConfig instead.
  warnings.warn(
```

```
Number of common cells: 8569
mudata: View of MuData object with n_obs × n_vars = 8569 × 20125
  obs:	'barcode', 'assigned_cluster', 'sample_id'
  1 modality
    multi_sc:	8569 x 20125
      obs:	'barcode', 'assigned_cluster', 'sample_id'
Processing 1 MuData objects: ['multi_sc']
Processing train modality: multi_sc
Processing valid split
Processing valid modality: multi_sc
Processing test split
Processing test modality: multi_sc
```

Now we will randomly set data do zero with.

In \[11\]:

Copied!

```
import torch
import copy
from autoencodix.data._numeric_dataset import NumericDataset
def drop_samples(ds: NumericDataset):
    """Randomly drops samples in .data and according .metadata (pd.DataFrame) and sample_ids"""
    data = ds.data
    n_samples = data.shape[0]
    drop_prob = 0.3
    keep_mask = torch.bernoulli((1 - drop_prob) * torch.ones(n_samples)).bool()
    # replace entries with zero
    imputed_data = data.clone()
    imputed_data[~keep_mask] = 0

    missing_data = data[keep_mask]
    missing_metadata = ds.metadata.iloc[keep_mask.cpu().numpy()].reset_index(drop=True)
    missing_sample_ids = [sid for i, sid in enumerate(ds.sample_ids) if keep_mask[i]]
    ds_with_missing = copy.deepcopy(ds)
    ds_with_missing.data = missing_data
    ds_with_missing.metadata = missing_metadata
    ds_with_missing.sample_ids = missing_sample_ids


    ds_with_zero = copy.deepcopy(ds)
    ds_with_zero.data = imputed_data
    ds_with_zero.metadata = ds.metadata
    ds_with_zero.sample_ids = ds.sample_ids
    return ds_with_missing, ds_with_zero

ds_with_missing = copy.deepcopy(data)
ds_with_zero = copy.deepcopy(data)
missing_train, ds_with_zero_train = drop_samples(ds_with_missing.train)

print(f"Original train data shape: {data.train.data.shape}, shape after corruption: {missing_train.data.shape}")
missing_test, ds_with_zero_test = drop_samples(data.test)
print(f"Original test data shape: {data.test.data.shape}, shape after corruption: {missing_test.data.shape}")
missing_valid, ds_with_zero_valid    = drop_samples(data.valid)
print(f"Original valid data shape: {data.valid.data.shape}, shape after corruption: {missing_valid.data.shape}")
ds_with_missing.train= missing_train
ds_with_missing.test = missing_test
ds_with_missing.valid = missing_valid


ds_with_zero.train= ds_with_zero_train
ds_with_zero.test = ds_with_zero_test
ds_with_zero.valid = ds_with_zero_valid
# clean corruped samples by removing zero values from dataset (drop samples) and also
```

import torch import copy from autoencodix.data.\_numeric_dataset import NumericDataset def drop_samples(ds: NumericDataset): """Randomly drops samples in .data and according .metadata (pd.DataFrame) and sample_ids""" data = ds.data n_samples = data.shape[0] drop_prob = 0.3 keep_mask = torch.bernoulli((1 - drop_prob) * torch.ones(n_samples)).bool()

# replace entries with zero

imputed_data = data.clone() imputed_data[~keep_mask] = 0 missing_data = data[keep_mask] missing_metadata = ds.metadata.iloc[keep_mask.cpu().numpy()].reset_index(drop=True) missing_sample_ids = \[sid for i, sid in enumerate(ds.sample_ids) if keep_mask[i]\] ds_with_missing = copy.deepcopy(ds) ds_with_missing.data = missing_data ds_with_missing.metadata = missing_metadata ds_with_missing.sample_ids = missing_sample_ids ds_with_zero = copy.deepcopy(ds) ds_with_zero.data = imputed_data ds_with_zero.metadata = ds.metadata ds_with_zero.sample_ids = ds.sample_ids return ds_with_missing, ds_with_zero ds_with_missing = copy.deepcopy(data) ds_with_zero = copy.deepcopy(data) missing_train, ds_with_zero_train = drop_samples(ds_with_missing.train) print(f"Original train data shape: {data.train.data.shape}, shape after corruption: {missing_train.data.shape}") missing_test, ds_with_zero_test = drop_samples(data.test) print(f"Original test data shape: {data.test.data.shape}, shape after corruption: {missing_test.data.shape}") missing_valid, ds_with_zero_valid = drop_samples(data.valid) print(f"Original valid data shape: {data.valid.data.shape}, shape after corruption: {missing_valid.data.shape}") ds_with_missing.train= missing_train ds_with_missing.test = missing_test ds_with_missing.valid = missing_valid ds_with_zero.train= ds_with_zero_train ds_with_zero.test = ds_with_zero_test ds_with_zero.valid = ds_with_zero_valid

# clean corruped samples by removing zero values from dataset (drop samples) and also

```
Original train data shape: torch.Size([5998, 1000]), shape after corruption: torch.Size([4251, 1000])
Original test data shape: torch.Size([1715, 1000]), shape after corruption: torch.Size([1189, 1000])
Original valid data shape: torch.Size([856, 1000]), shape after corruption: torch.Size([616, 1000])
```

Now we train our Maskix with this clean data, but we will pass a custom imputer that will simulate missing values

In \[12\]:

Copied!

```
import torch
import autoencodix as acx
from autoencodix.configs import MaskixConfig
from autoencodix.configs.default_config import DataInfo, DataConfig, DataCase


def my_imputer(x: torch.Tensor) -> torch.Tensor:
    "randomly replaces value with zero"
    rand_mask = torch.bernoulli(0.3 * torch.ones(x.shape, device=x.device)).bool()
    rand_mask.to(x.device)
    imputed_x = torch.where(rand_mask, torch.zeros_like(x, device=x.device), x)
    return imputed_x


from huggingface_hub import hf_hub_download

sc_path = hf_hub_download(
    repo_id="autoencodix/pancreas-fang", repo_type="dataset", filename="pancreas.h5ad"
)
config = MaskixConfig(
    epochs=30,
    checkpoint_interval=10,
    k_filter=3000,
    skip_preprocessing=True,
    batch_size=64,
    data_config=DataConfig(
        annotation_columns=["assigned_cluster"],
        data_info={
            "multi_sc": DataInfo(
                 is_single_cell=True, data_type="NUMERIC"
            )
        },
    ),
    data_case=DataCase.MULTI_SINGLE_CELL,
)
maskix = acx.Maskix(config=config, masking_fn=my_imputer, data=ds_with_missing)
result = maskix.run()
```

import torch import autoencodix as acx from autoencodix.configs import MaskixConfig from autoencodix.configs.default_config import DataInfo, DataConfig, DataCase def my_imputer(x: torch.Tensor) -> torch.Tensor: "randomly replaces value with zero" rand_mask = torch.bernoulli(0.3 * torch.ones(x.shape, device=x.device)).bool() rand_mask.to(x.device) imputed_x = torch.where(rand_mask, torch.zeros_like(x, device=x.device), x) return imputed_x from huggingface_hub import hf_hub_download sc_path = hf_hub_download( repo_id="autoencodix/pancreas-fang", repo_type="dataset", filename="pancreas.h5ad" ) config = MaskixConfig( epochs=30, checkpoint_interval=10, k_filter=3000, skip_preprocessing=True, batch_size=64, data_config=DataConfig( annotation_columns=["assigned_cluster"], data_info={ "multi_sc": DataInfo( is_single_cell=True, data_type="NUMERIC" ) }, ), data_case=DataCase.MULTI_SINGLE_CELL, ) maskix = acx.Maskix(config=config, masking_fn=my_imputer, data=ds_with_missing) result = maskix.run()

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/configs/default_config.py:439: UserWarning: annotation_columns in DataConfig is deprecated. Please set it directly in DefaultConfig instead.
  warnings.warn(
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/losses/maskix_loss.py:38: UserWarning: You chose loss reduction: sum, this deviates from the implementation in the literature for this architecture, the authors used 'mean'
  warnings.warn(
```

```
Epoch 10 - Train Loss: 525.6589
Sub-losses: recon_loss: 810.2212, recon_loss_weighted: 97.0191, mask_loss: 428.6398
Epoch 10 - Valid Loss: 523.1195
Sub-losses: recon_loss: 785.5748, recon_loss_weighted: 94.5569, mask_loss: 428.5626
Epoch 20 - Train Loss: 524.8635
Sub-losses: recon_loss: 799.0144, recon_loss_weighted: 96.2755, mask_loss: 428.5880
Epoch 20 - Valid Loss: 519.9710
Sub-losses: recon_loss: 771.4230, recon_loss_weighted: 91.7578, mask_loss: 428.2132
Epoch 30 - Train Loss: 523.7000
Sub-losses: recon_loss: 790.4772, recon_loss_weighted: 95.1403, mask_loss: 428.5597
Epoch 30 - Valid Loss: 519.5838
Sub-losses: recon_loss: 770.1543, recon_loss_weighted: 91.3486, mask_loss: 428.2352
Processed 1189 / 1189 samples
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/losses/maskix_loss.py:38: UserWarning: You chose loss reduction: sum, this deviates from the implementation in the literature for this architecture, the authors used 'mean'
  warnings.warn(
```

No we use the trained maskix to impute our missing data

Now, we can use the fitted model and use a corrupted input with missing to get a reconstruction without missing values.

In \[13\]:

Copied!

```
mo_train = maskix.impute(ds_with_zero.train.data)
recons_train = mo_train.reconstruction

mo_test = maskix.impute(ds_with_zero.test.data)
recons_test = mo_test.reconstruction

mo_valid = maskix.impute(ds_with_zero.valid.data)
recons_valid = mo_valid.reconstruction


ds_imputed = copy.deepcopy(ds_with_zero)
ds_imputed.train.data = recons_train
ds_imputed.test.data = recons_test
ds_imputed.valid.data = recons_valid
```

mo_train = maskix.impute(ds_with_zero.train.data) recons_train = mo_train.reconstruction mo_test = maskix.impute(ds_with_zero.test.data) recons_test = mo_test.reconstruction mo_valid = maskix.impute(ds_with_zero.valid.data) recons_valid = mo_valid.reconstruction ds_imputed = copy.deepcopy(ds_with_zero) ds_imputed.train.data = recons_train ds_imputed.test.data = recons_test ds_imputed.valid.data = recons_valid

We can compare the loss between the imputed data and the original and the reconstructed data and the original.

In \[14\]:

Copied!

```
%%capture
# ground truth without missing values
result_orig = maskix_orig.run()
```

%%capture

# ground truth without missing values

result_orig = maskix_orig.run()

In \[15\]:

Copied!

```
from torch.nn.functional import mse_loss
original_train = result_orig.datasets.train.data
original_recon = result_orig.reconstructions.get(split="train", epoch=-1)
original_recon_tensor = torch.from_numpy(original_recon)
loss = mse_loss(original_recon_tensor, original_train)
print(f"MSE Loss original reconstruction: {loss}")
loss_imputed = mse_loss(recons_train.to("cpu"), original_train)
print(f"MSE Loss imputed reconstruction: {loss_imputed}")
```

from torch.nn.functional import mse_loss original_train = result_orig.datasets.train.data original_recon = result_orig.reconstructions.get(split="train", epoch=-1) original_recon_tensor = torch.from_numpy(original_recon) loss = mse_loss(original_recon_tensor, original_train) print(f"MSE Loss original reconstruction: {loss}") loss_imputed = mse_loss(recons_train.to("cpu"), original_train) print(f"MSE Loss imputed reconstruction: {loss_imputed}")

```
MSE Loss original reconstruction: 0.8379753232002258
MSE Loss imputed reconstruction: 0.8577121496200562
```

Finally, you can use this data as input for other autoencoders like `varix` or `vanillix`, but also use it for maskix. We will train two Varix models, one with the missing data and one with the imputed data and compare the results

In \[16\]:

Copied!

```
original_train.shape
```

original_train.shape

Out\[16\]:

```
torch.Size([5998, 1000])
```

In \[17\]:

Copied!

```
recons_train.shape
```

recons_train.shape

Out\[17\]:

```
torch.Size([5998, 1000])
```

In \[18\]:

Copied!

```
import autoencodix as acx
from autoencodix.configs import VarixConfig
from autoencodix.configs.default_config import DataInfo, DataConfig, DataCase
from autoencodix.data._datasetcontainer import DatasetContainer
import copy

config = VarixConfig(
    epochs=50,
    checkpoint_interval=10,
    batch_size=64,
    skip_preprocessing=True,
    data_config=DataConfig(
        annotation_columns=["assigned_cluster"],
        data_info={"multi_sc": DataInfo(is_single_cell=True, data_type="NUMERIC")},
    ),
    data_case=DataCase.MULTI_SINGLE_CELL,
)

varix_imputed = acx.Varix(config=config, data=ds_imputed)
varix_missing = acx.Varix(config=config, data=ds_with_zero)
```

import autoencodix as acx from autoencodix.configs import VarixConfig from autoencodix.configs.default_config import DataInfo, DataConfig, DataCase from autoencodix.data.\_datasetcontainer import DatasetContainer import copy config = VarixConfig( epochs=50, checkpoint_interval=10, batch_size=64, skip_preprocessing=True, data_config=DataConfig( annotation_columns=["assigned_cluster"], data_info={"multi_sc": DataInfo(is_single_cell=True, data_type="NUMERIC")}, ), data_case=DataCase.MULTI_SINGLE_CELL, ) varix_imputed = acx.Varix(config=config, data=ds_imputed) varix_missing = acx.Varix(config=config, data=ds_with_zero)

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/configs/default_config.py:439: UserWarning: annotation_columns in DataConfig is deprecated. Please set it directly in DefaultConfig instead.
  warnings.warn(
```

In \[19\]:

Copied!

```
result_imputed = varix_imputed.run()
result_missing = varix_missing.run()
```

result_imputed = varix_imputed.run() result_missing = varix_missing.run()

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:233: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 22.9550 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 10 - Train Loss: 58.5504
Sub-losses: recon_loss: 58.5361, var_loss: 0.0143, anneal_factor: 0.0017, effective_beta_factor: 0.0002
Epoch 10 - Valid Loss: 33.6344
Sub-losses: recon_loss: 33.6179, var_loss: 0.0165, anneal_factor: 0.0017, effective_beta_factor: 0.0002
Epoch 20 - Train Loss: 45.7293
Sub-losses: recon_loss: 44.2866, var_loss: 1.4427, anneal_factor: 0.0832, effective_beta_factor: 0.0083
Epoch 20 - Valid Loss: 24.5973
Sub-losses: recon_loss: 23.1475, var_loss: 1.4498, anneal_factor: 0.0832, effective_beta_factor: 0.0083
Epoch 30 - Train Loss: 47.4960
Sub-losses: recon_loss: 41.4196, var_loss: 6.0765, anneal_factor: 0.8320, effective_beta_factor: 0.0832
Epoch 30 - Valid Loss: 28.1740
Sub-losses: recon_loss: 21.9422, var_loss: 6.2318, anneal_factor: 0.8320, effective_beta_fact
... [output truncated]
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:233: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 16.4512 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 10 - Train Loss: 598.3720
Sub-losses: recon_loss: 598.3560, var_loss: 0.0160, anneal_factor: 0.0017, effective_beta_factor: 0.0002
Epoch 10 - Valid Loss: 561.6330
Sub-losses: recon_loss: 561.6178, var_loss: 0.0152, anneal_factor: 0.0017, effective_beta_factor: 0.0002
Epoch 20 - Train Loss: 579.7749
Sub-losses: recon_loss: 578.2827, var_loss: 1.4922, anneal_factor: 0.0832, effective_beta_factor: 0.0083
Epoch 20 - Valid Loss: 545.6434
Sub-losses: recon_loss: 544.2280, var_loss: 1.4154, anneal_factor: 0.0832, effective_beta_factor: 0.0083
Epoch 30 - Train Loss: 581.2735
Sub-losses: recon_loss: 574.4224, var_loss: 6.8511, anneal_factor: 0.8320, effective_beta_factor: 0.0832
Epoch 30 - Valid Loss: 546.9111
Sub-losses: recon_loss: 540.5161, var_loss: 6.3950, anneal_factor: 0.8320, effecti
... [output truncated]
```

In \[20\]:

Copied!

```
varix_imputed._datasets.train.metadata
```

varix_imputed.\_datasets.train.metadata

Out\[20\]:

|                               | barcode             | assigned_cluster | sample_id | n_genes |
| ----------------------------- | ------------------- | ---------------- | --------- | ------- |
| human1_lib1.final_cell_0002_0 | GAGCGTTGCT-ACCTTCTT | acinar           | human1    | 4201    |
| human1_lib1.final_cell_0003_0 | CTTACGGG-CCATTACT   | acinar           | human1    | 2119    |
| human1_lib1.final_cell_0006_0 | AATCCCACG-ATTCGACG  | acinar           | human1    | 2477    |
| human1_lib1.final_cell_0009_0 | GAGAATTCGT-GTTTGTTT | acinar           | human1    | 3272    |
| human1_lib1.final_cell_0013_0 | AAAATCGTT-GGAAACAG  | delta            | human1    | 2357    |
| ...                           | ...                 | ...              | ...       | ...     |
| human4_lib3.final_cell_0693_3 | AGTTACCGC-CTACCGTT  | beta             | human4    | 1156    |
| human4_lib3.final_cell_0694_3 | GACTTACTCC-TAGAAATG | alpha            | human4    | 1285    |
| human4_lib3.final_cell_0695_3 | TAGCCTCG-GGCTACTA   | beta             | human4    | 983     |
| human4_lib3.final_cell_0696_3 | GACGGGCTTT-TATTGCCT | beta             | human4    | 1226    |
| human4_lib3.final_cell_0698_3 | GCTTACCT-ATGTTGGC   | alpha            | human4    | 1007    |

5998 rows × 4 columns

In \[21\]:

Copied!

```
varix_imputed.show_result()
```

varix_imputed.show_result()

```
Creating plots ...
```

In \[22\]:

Copied!

```
varix_imputed.show_result()
```

varix_imputed.show_result()

```
Creating plots ...
```

In \[23\]:

Copied!

```
varix_imputed.evaluate()
```

varix_imputed.evaluate()

```
Perform ML task with feature df: Latent
Latent
Perform ML task for target parameter: assigned_cluster
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/.venv/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
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
sub_losses: LossRegistry(_losses={'recon_loss': TrainingDynamics(), 'var_loss': TrainingDynamics(), 'anneal_factor': TrainingDynamics(), 'effective_beta_factor': TrainingDynamics()})
preprocessed_data: Tensor of shape (0,)
model: VarixArchitecture
model_checkpoints: TrainingDynamics object
datasets: DatasetContainer(train=<autoencodix.data._numeric_dataset.NumericDataset object at 0x721689746560>, valid=<autoencodix.data._numeric_dataset.NumericDataset object at 0x721689747280>, test=<autoencodix.data._numeric_dataset.Num
... [output truncated]
```

In \[24\]:

Copied!

```
varix_missing.evaluate()
```

varix_missing.evaluate()

```
Perform ML task with feature df: Latent
Latent
Perform ML task for target parameter: assigned_cluster
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/.venv/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
```

```
Perform ML task with feature df: Latent
Latent
Perform ML task for target parameter: assigned_cluster
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/.venv/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
```

Out\[24\]:

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
datasets: DatasetContainer(train=<autoencodix.data._numeric_dataset.NumericDataset object at 0x7216898a0700>, valid=<autoencodix.data._numeric_dataset.NumericDataset object at 0x7216898a0ca0>, test=<autoencodix.data._numeric_dataset.Num
... [output truncated]
```

## 7) Save, Load and Re-Use Maskix[¶](#7-save-load-and-re-use-maskix)

There are not `Maskix` specific steps here. See the [`Tutorials/PipelineTutorials/Vanillix.ipynb`](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/Maskix/Vanillix.ipynb) or [`Tutorials/DeepDives/MemoryEfficientSaving.ipynb`](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/DeepDives/MemoryEfficientSaving.ipynb) for details. Below is a basic save/load usecase:

In \[25\]:

Copied!

```
import os
import glob
# use a filename without extension, we handle this internally
outpath = os.path.join("tutorial_res", "maskix")
maskix.save(file_path=outpath, save_all=False)

folder = os.path.dirname(outpath)
pkl_files = glob.glob(os.path.join(folder, "*.pkl"))
model_files = glob.glob(os.path.join(folder, "*.pth"))

print("PKL files:", pkl_files)
print("Model files:", model_files)

# the load functionality automatically will build the pipeline object out of the three saved files
varix_loaded = acx.Maskix.load(outpath)
varix_loaded.predict(data=maskix_result.datasets)
varix_loaded.visualize()
varix_loaded.show_result()
```

import os import glob

# use a filename without extension, we handle this internally

outpath = os.path.join("tutorial_res", "maskix") maskix.save(file_path=outpath, save_all=False) folder = os.path.dirname(outpath) pkl_files = glob.glob(os.path.join(folder, "\*.pkl")) model_files = glob.glob(os.path.join(folder, "\*.pth")) print("PKL files:", pkl_files) print("Model files:", model_files)

# the load functionality automatically will build the pipeline object out of the three saved files

varix_loaded = acx.Maskix.load(outpath) varix_loaded.predict(data=maskix_result.datasets) varix_loaded.visualize() varix_loaded.show_result()

```
Preprocessor saved successfully.
saving memory efficient
Pipeline object saved successfully.
PKL files: ['tutorial_res/varix_preprocessor.pkl', 'tutorial_res/ontix.pkl', 'tutorial_res/maskix_preprocessor.pkl', 'tutorial_res/imagix.pkl', 'tutorial_res/ontix_preprocessor.pkl', 'tutorial_res/disent.pkl', 'tutorial_res/imagix_preprocessor.pkl', 'tutorial_res/disent_preprocessor.pkl']
Model files: ['tutorial_res/varix_model.pth', 'tutorial_res/disent_model.pth', 'tutorial_res/imagix_model.pth', 'tutorial_res/ontix_model.pth', 'tutorial_res/maskix_model.pth']
Attempting to load a pipeline from tutorial_res/maskix...
Pipeline object loaded successfully. Actual type: Maskix
Preprocessor loaded successfully.
Processed 1715 / 1715 samples
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/losses/maskix_loss.py:38: UserWarning: You chose loss reduction: sum, this deviates from the implementation in the literature for this architecture, the authors used 'mean'
  warnings.warn(
```

```
Creating plots ...
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/utils/_result.py:348: UserWarning: Could not retrieve latent representations for epoch 29 and split 'train'. Returning empty DataFrame. This may be due to missing data in the Result object or incorrect keys.


  warnings.warn(
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/utils/_result.py:348: UserWarning: Could not retrieve latent representations for epoch 29 and split 'valid'. Returning empty DataFrame. This may be due to missing data in the Result object or incorrect keys.


  warnings.warn(
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/utils/_result.py:348: UserWarning: Could not retrieve latent representations for epoch 29 and split 'train'. Returning empty DataFrame. This may be due to missing data in the Result object or incorrect keys.


  warnings.warn(
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/utils/_result.py:348: UserWarning: Could not retrieve latent representations for epoch 29 and split 'valid'. Returning empty DataFrame. This may be due to missing data in the Result object or incorrect keys.


  warnings.warn(
```
