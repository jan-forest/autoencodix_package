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

# Getting Started - Vanillix Tutorial[¶](#getting-started-vanillix-tutorial)

In this tutorial, we'll demonstrate how to use the basic functionalities of our **AUTOENCODIX** pipeline.\
We’ll cover the general **workflow**, show how to work with custom classes, and highlight the pipeline’s flexibility.\
**AUTOENCODIX** supports far more functionality than shown here, so we’ll also point to advanced tutorials where relevant.

**IMPORTANT**

> This tutorial serves two purposes: (a) introducing the framework in general, and (b) presenting the Vanillix pipeline.\
> For other architectures (e.g., Varix or Ontix), we’ll skip the general concepts and focus only on their specifics.\
> To get a solid overview of the framework, we recommend starting here. We’ll include this note in other tutorials as well, so you can always find the right entry point.

______________________________________________________________________

## What You'll Learn[¶](#what-youll-learn)

You’ll learn how to:

1. **Initialize** the pipeline with a pandas DataFrame.
1. Understand the **pipeline steps**.
1. Access the pipeline **results** (latent spaces, losses, etc.).
1. **Visualize** outputs effectively.
1. Apply **custom parameters**.
1. **Save, load, and reuse** a trained pipeline.

______________________________________________________________________

Let’s get started! 🚀

## 1) Initialization[¶](#1-initialization)

To get started, you need three things:

- A dataset
- A configuration object
- The Vanillix pipeline

#### 📊 Dataset[¶](#dataset)

We allow a variety of inputs (see [1]).\
Here, we demonstrate how to use the pipeline with pandas DataFrames.

We use a real multi-omics use case: a balanced subset of the TCGA pan-cancer cohort ([`autoencodix/tcga`](https://huggingface.co/datasets/autoencodix/tcga) on Hugging Face), downloaded and subsampled below to keep the tutorial fast:

- RNA-seq: `raw_rna`
- DNA methylation: `raw_meth`
- Clinical metadata: `annotation`

For simplicity, we assume paired metadata.\
For handling unpaired data, see [1] or [2].

#### ⚙️ Configuration Object[¶](#configuration-object)

We use the default parameters in `VanillixConfig`.

Additionally, we define a `DataCase`, which tells the pipeline what kind of data to expect.\
In this case: `MULTI_BULK`.

This can be any tabular data and does not have to be bulk sequencing data (we may rename this soon).\
For more details on `DataCase`, refer to [1].

#### 🪈 Vanillix Pipeline[¶](#vanillix-pipeline)

Finally, we import `Vanillix` from our package.

Each autoencoder architecture comes with a corresponding `pipeline`, which handles the complete analysis process—from preprocessing to evaluation.\
All autoencoders share this unified pipeline interface, which we’ll explore in the next sections.

[1] [Tutorials/DeepDives/InputDataTutorials.ipynb](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/DeepDives/InputDataTutorials.ipynb)

[2] [Tutorials/PipelineTutorials/XModalix.ipynb](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/Vanillix/XModalix.ipynb)

### 1.1) Inspect Data[¶](#11-inspect-data)

Before initializing the pipeline, we’ll show what the input data can look like.\
This should help when working with your own data later.

In \[2\]:

Copied!

```
import pandas as pd
from huggingface_hub import hf_hub_download

# Data is hosted on Hugging Face Hub (autoencodix/tcga) and downloaded (and locally cached) automatically.
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

print(f"Type of raw_rna input: {type(raw_rna)}")
```

import pandas as pd from huggingface_hub import hf_hub_download

# Data is hosted on Hugging Face Hub (autoencodix/tcga) and downloaded (and locally cached) automatically.

HF_REPO_ID = "autoencodix/tcga" rna_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="rna.parquet") meth_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="methylation.parquet") clin_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="clinical.parquet") rna_full = pd.read_parquet(rna_path) meth_full = pd.read_parquet(meth_path) clin_full = pd.read_parquet(clin_path)

# Keep only samples present in all three modalities

shared_ids = rna_full.index.intersection(meth_full.index).intersection(clin_full.index)

# Balanced subsample across the 5 broad TCGA cancer types

n_per_type = 120 sampled_ids = ( clin_full.loc[shared_ids] .groupby("CANCER_TYPE", group_keys=False) .apply(lambda g: g.sample(n=min(n_per_type, len(g)), random_state=42), include_groups=False) .index ) raw_rna = rna_full.loc[sampled_ids] raw_meth = meth_full.loc[sampled_ids] annotation = clin_full.loc[sampled_ids] print(f"Type of raw_rna input: {type(raw_rna)}")

```
Type of raw_rna input: <class 'pandas.core.frame.DataFrame'>
```

In \[3\]:

Copied!

```
raw_rna.head()
```

raw_rna.head()

Out\[3\]:

```
Entrez_Gene_Id
100133144
100134869
10357
10431
155060
388795
390284
57714
645851
653553
...
55055
11130
7789
158586
79364
440590
79699
7791
23140
26009




TCGA-A2-A0CK-01
0.9980
2.6384
113.6650
795.556
263.0300
0.4040
10.5051
132.121
9.6970
479.1920
...
344.493
732.929
43.2323
456.162
1349.490
94.9495
600.000
3474.34
738.182
785.859


TCGA-E2-A1BC-01
6.2212
7.3031
66.2609
1034.210
96.2609
0.0000
9.5465
818.616
10.3421
55.6881
...
337.852
247.415
103.4210
1278.440
1179.000
93.8743
1464.600
1774.86
1125.700
1349.240


TCGA-AR-A1AV-01
13.1083
8.2906
95.9250
718.020
190.5660
0.8675
9.8319
1058.090
17.6396
877.3550
...
550.805
422.195
105.8380
1049.700
1362.880
84.1498
910.610
2529.99
1488.670
871.861


TCGA-S3-A6ZH-01
5.1744
3.0139
113.4210
1157.700
170.3800
2.2046
6.2987
517.440
27.084
... [output truncated]
```

In \[4\]:

Copied!

```
annotation.head()
```

annotation.head()

Out\[4\]:

```
PATIENT_ID
ONCOTREE_CODE
CANCER_TYPE
CANCER_TYPE_DETAILED
TUMOR_TYPE
GRADE
TISSUE_PROSPECTIVE_COLLECTION_INDICATOR
TISSUE_RETROSPECTIVE_COLLECTION_INDICATOR
TISSUE_SOURCE_SITE_CODE
TUMOR_TISSUE_SITE
...
OS_STATUS
OS_MONTHS
DSS_STATUS
DSS_MONTHS
DFS_STATUS
DFS_MONTHS
PFS_STATUS
PFS_MONTHS
GENETIC_ANCESTRY_LABEL
AJCC_PATHOLOGIC_TUMOR_STAGE_SHORT




TCGA-A2-A0CK-01
TCGA-A2-A0CK
ILC
Breast Cancer
Breast Invasive Lobular Carcinoma
Infiltrating Lobular Carcinoma
unknown
No
Yes
A2
Breast
...
0:LIVING
136.732748
0:ALIVE OR DEAD TUMOR FREE
136.732748
1:Recurred/Progressed
44.843344
1:PROGRESSION
44.843344
EUR
STAGE III


TCGA-E2-A1BC-01
TCGA-E2-A1BC
IDC
Breast Cancer
Breast Invasive Ductal Carcinoma
Infiltrating Ductal Carcinoma
unknown
No
Yes
E2
Breast
...
0:LIVING
16.471052
0:ALIVE OR DEA
... [output truncated]
```

### 1.2) Initialize Pipeline[¶](#12-initialize-pipeline)

In this step, we import the pipeline, configuration, and data classes, and show how to feed pandas DataFrames into the pipeline.

We use the `DataPackage` class for this purpose. In our case, we fill the attribute `multi_bulk` with a dictionary of pandas DataFrames.\
*Note:* The term *multi* may be misleading — you can also provide just a single DataFrame.

Metadata is also passed as a dictionary to the `DataPackage`.\
Here, we have one annotation DataFrame for both data modalities (RNA and DNA methylation), so we use the identifier `"paired"` as the key.\
Otherwise, you would use the same keys as for the actual data, e.g.:

```
{"rna": rna_annotation, "meth": meth_annotation}
```

In \[5\]:

Copied!

```
import autoencodix as acx
from autoencodix.data.datapackage import DataPackage
from autoencodix.configs.vanillix_config import VanillixConfig
from autoencodix.configs.default_config import DataCase

# If your data is stored in pandas DataFrames, you can easily pass them to our custom DataPackage.
# For any tabular data that is not single-cell, provide it as a dictionary to the "multi_bulk" attribute of DataPackage.
# Note: "multi" might be misleading — it's valid to provide just one modality (1–n data modalities).
# Here, we assume paired metadata. If you have separate metadata for each modality, use the same dict keys as in multi_bulk, e.g.:
# annotation = {"rna": rna_annotation, "meth": meth_annotation}
my_datapackage: DataPackage = DataPackage(
    multi_bulk={"rna": raw_rna, "meth": raw_meth},
    annotation={"paired": annotation},
)

myconfig: VanillixConfig = VanillixConfig(
    data_case=DataCase.MULTI_BULK, epochs=30, device="cpu", k_filter=2000,
    reproducible=True, global_seed=1,
)
vanillix = acx.Vanillix(data=my_datapackage, config=myconfig)
```

import autoencodix as acx from autoencodix.data.datapackage import DataPackage from autoencodix.configs.vanillix_config import VanillixConfig from autoencodix.configs.default_config import DataCase

# If your data is stored in pandas DataFrames, you can easily pass them to our custom DataPackage.

# For any tabular data that is not single-cell, provide it as a dictionary to the "multi_bulk" attribute of DataPackage.

# Note: "multi" might be misleading — it's valid to provide just one modality (1–n data modalities).

# Here, we assume paired metadata. If you have separate metadata for each modality, use the same dict keys as in multi_bulk, e.g.:

# annotation = {"rna": rna_annotation, "meth": meth_annotation}

my_datapackage: DataPackage = DataPackage( multi_bulk={"rna": raw_rna, "meth": raw_meth}, annotation={"paired": annotation}, ) myconfig: VanillixConfig = VanillixConfig( data_case=DataCase.MULTI_BULK, epochs=30, device="cpu", k_filter=2000, reproducible=True, global_seed=1, ) vanillix = acx.Vanillix(data=my_datapackage, config=myconfig)

```
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
```

## 2) Pipeline Steps[¶](#2-pipeline-steps)

Next, we explain the four main steps of our pipeline:

1. Preprocess
1. Fit
1. Predict
1. Visualize

We can run all steps together with the `run` method.

### 2.1) Preprocess[¶](#21-preprocess)

If you're familiar with `AUTOENCODIX 1.0`, this is similar to `make data` in the old framework.\
In case you don’t want any preprocessing, you can set `skip_preprocessing=True` in the config object.\
**ATTENTION:** This could lead to `RuntimeErrors` if you violate assumptions like NaN-free data, which we normally handle during preprocessing.\
See Tutorial [1] for details on data requirements and assumptions.

#### 🔗 Data Pairing[¶](#data-pairing)

If you have multiple data modalities, e.g., mRNA and CNA data, and you work with a pipeline that requires paired data (Vanillix, Varix, Ontix, Disentanglix), we drop all samples that are not present in all data modalities.\
If you want to include unpaired data, please refer to [2] and [3].

#### 🧼 Data Cleaning[¶](#data-cleaning)

This step acts as a safety net to avoid runtime errors later in the pipeline. Here, we simply drop columns with NaN values.\
Ideally, you’ve already cleaned the data before putting it into our pipeline if you want more sophisticated handling of NaNs.\
**Note:** For metadata, we allow NaNs as long as they’re not specified in the config as relevant columns (see [4]).

#### ✂️ Splitting[¶](#splitting)

We designed the preprocessing pipeline to avoid data leakage, so almost all preprocessing transformations are applied **after** splitting the data.\
The only exception is filtering cells and log-transforming single-cell data (if not already done by you).\
The default behavior is to split your input data into three splits: train, validation, and test. You can define the ratio via the config (see [4]) or, for some pipelines, provide a custom split (see [1]).\
Additionally, you can always pass unseen data to the `predict` step of the pipeline.

#### 🗂️ Filtering[¶](#filtering)

We filter the columns of the train set and apply the learned filtering to the validation and test sets. The filtering method and the number of features to keep can be defined in the config.

The filtering method can be set per data modality, but we recommend keeping it consistent.\
If you work with multiple data modalities and provide a number for `k_filter` (e.g., 1000), this will be the total number of features across all data modalities. For example, you may have 500 features for modality A and 500 for modality B.

We provide the following options for filtering:

- `VAR` – Select features with the highest variance
- `MAD` – Select features with the highest median absolute deviation
- `CORR` – Select features based on correlation
- `VARCORR` – Combination of variance and correlation
- `NOFILT` – No filtering applied
- `NONZEROVAR` – Select features with non-zero variance

#### 📏 Scaling[¶](#scaling)

After the filtering step, we scale the data. We fit the scaler on the train set and apply the learned transformations to the validation and test sets.

Different scaling methods can be applied per data modality, but we strongly recommend setting this globally via the `scaling` parameter. The default is `StandardScaler`.

We provide the following options for scaling:

- `STANDARD` – Standard scaling (mean = 0, variance = 1)
- `MINMAX` – Min–max scaling to range [0, 1]
- `ROBUST` – Robust scaling using the median and interquartile range
- `MAXABS` – Scaling by the maximum absolute value
- `NONE` – No scaling applied

#### 🐍 Transforming to PyTorch Dataset[¶](#transforming-to-pytorch-dataset)

In the last step, we transform the data into a custom PyTorch `Dataset` class.\
For the Vanillix pipeline, we concatenate the different data modalities and store them in the `data` attribute of the `Dataset` class.\
Now the data is ready to serve as input for our PyTorch models.

[1] [Tutorials/DeepDives/InputDataTutorials.ipynb](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/DeepDives/InputDataTutorials.ipynb)\
[2] [Tutorials/PipelineTutorials/XModalix.ipynb](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/Vanillix/XModalix.ipynb)\
[3] [Tutorials/PipelineTutorials/Stackix.ipynb](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/Vanillix/Stackix.ipynb)\
[4] [Tutorials/DeepDives/ConfigTutorial.ipynb](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/DeepDives/ConfigTutorial.ipynb)

#### 🖥️ In Code[¶](#in-code)

All the above steps run with one command in our pipeline:

In \[6\]:

Copied!

```
# This does not return anything; the step updates internal attributes:
# It sets the private _datasets attribute.
# It populates the results.datasets attribute with the preprocessed and transformed PyTorch datasets (one for each split).
vanillix.preprocess()
```

# This does not return anything; the step updates internal attributes:

# It sets the private \_datasets attribute.

# It populates the results.datasets attribute with the preprocessed and transformed PyTorch datasets (one for each split).

vanillix.preprocess()

```
anno key: paired
```

### 2.2) The Fit Step[¶](#22-the-fit-step)

The fit step trains the autoencoder architecture and captures an extensive set of training dynamics.\
The training and model itself are highly customizable (`n_layers`, `latent_dim`, `epochs`, `learning_rate`, etc.). Refer to [Section 5](#5-customize) or Tutorial [4] for more details.\
During training, we capture losses, latent spaces, and reconstructions for each checkpoint epoch and store them in our results object.

[4] [Tutorials/DeepDives/ConfigTutorial.ipynb](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/DeepDives/ConfigTutorial.ipynb)

In \[7\]:

Copied!

```
# Job of the old make_model step.
# Calls self.Trainer class to initialize and train the model.
# Populates the self._model attribute with the trained model.
# Populates the self.results attribute with training results (model, losses, etc.).
vanillix.fit()
```

# Job of the old make_model step.

# Calls self.Trainer class to initialize and train the model.

# Populates the self.\_model attribute with the trained model.

# Populates the self.results attribute with training results (model, losses, etc.).

vanillix.fit()

```
Reproducibility settings for device cpu are not implemented or necessary i.e. for cpu.
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:233: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 21.2020 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 10 - Train Loss: 1343.8532
Sub-losses: recon_loss: 1343.8532
Epoch 10 - Valid Loss: 1444.9164
Sub-losses: recon_loss: 1444.9164
Epoch 20 - Train Loss: 1259.6486
Sub-losses: recon_loss: 1259.6486
Epoch 20 - Valid Loss: 1419.2240
Sub-losses: recon_loss: 1419.2240
Epoch 30 - Train Loss: 1219.7124
Sub-losses: recon_loss: 1219.7124
Epoch 30 - Valid Loss: 1403.2820
Sub-losses: recon_loss: 1403.2820
```

### 2.3) The Predict Step[¶](#23-the-predict-step)

In this step, we use the trained model to run a forward pass on our test dataset.\
This step can also take a `data` argument, allowing you to run predictions on held-out data other than the test split.\
If you provide new data, preprocessing will be applied automatically depending on the data format — see [1] for details.

[1] [Tutorials/DeepDives/InputDataTutorials.ipynb](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/DeepDives/InputDataTutorials.ipynb)

In \[8\]:

Copied!

```
# Job of the old make_predict step.
# If no data is passed, uses the test split from preprocessing.
# Otherwise, uses the provided data and preprocesses it.
# Updates the self.results attribute with predictions (latent space, reconstructions, etc.).
# Returns a results object.
result = vanillix.predict()
```

# Job of the old make_predict step.

# If no data is passed, uses the test split from preprocessing.

# Otherwise, uses the provided data and preprocesses it.

# Updates the self.results attribute with predictions (latent space, reconstructions, etc.).

# Returns a results object.

result = vanillix.predict()

```
Reproducibility settings for device cpu are not implemented or necessary i.e. for cpu.
Processed 120 / 120 samples
```

### 2.4) The Visualize Step[¶](#24-the-visualize-step)

This step takes the training results and creates visualizations,\
for example, plots of the loss curve or the latent spaces.\
The plots are stored internally and can be displayed with the `show_result` method.\
See [Section Visualize](#4-visualize) for more details on working with the generated plots.

In \[9\]:

Copied!

```
# Job of old make visualize
vanillix.visualize()
```

# Job of old make visualize

vanillix.visualize()

### 2.5) The Run Step[¶](#25-the-run-step)

This step combines all the previous steps and returns a results object.\
Furthermore, you can pass a `data` argument, which will then be used in the predict step instead of the test split.

In \[10\]:

Copied!

```
# Calls all the steps in one wrapper:
# vanillix.preprocess()
# vanillix.fit()
# vanillix.predict()
# vanillix.visualize
vanillix = acx.Vanillix(config=myconfig, data=my_datapackage)
result = vanillix.run()
# Also possilbe to pass data: vanillix.run(data=<your-data>)
```

# Calls all the steps in one wrapper:

# vanillix.preprocess()

# vanillix.fit()

# vanillix.predict()

# vanillix.visualize

vanillix = acx.Vanillix(config=myconfig, data=my_datapackage) result = vanillix.run()

# Also possilbe to pass data: vanillix.run(data=<your-data>)

```
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
anno key: paired
Reproducibility settings for device cpu are not implemented or necessary i.e. for cpu.
```

```
/home/alicia/dev/biomarker_autoencoder/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:233: UserWarning: Gradient clipping was first applied in epoch 0. Total norm of gradients (adjusted for number of features): 21.2020 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 10 - Train Loss: 1343.8532
Sub-losses: recon_loss: 1343.8532
Epoch 10 - Valid Loss: 1444.9164
Sub-losses: recon_loss: 1444.9164
Epoch 20 - Train Loss: 1259.6486
Sub-losses: recon_loss: 1259.6486
Epoch 20 - Valid Loss: 1419.2240
Sub-losses: recon_loss: 1419.2240
Epoch 30 - Train Loss: 1219.7124
Sub-losses: recon_loss: 1219.7124
Epoch 30 - Valid Loss: 1403.2820
Sub-losses: recon_loss: 1403.2820
Reproducibility settings for device cpu are not implemented or necessary i.e. for cpu.
Processed 120 / 120 samples
```

### 2.6) Extra: The Evaluate Step[¶](#26-extra-the-evaluate-step)

This step is not included when you call `.run()`. Here, we use the `latent space` and compare it to other dimensionality reduction techniques like PCA, UMAP, or a random baseline.\
In the basic case, you can specify a column in your annotation data to use for a machine learning task, e.g., training a classifier on disease state or cancer type. We then use the representations (latent space, UMAP, etc.) to train this classifier/regressor.\
You have many options to customize this step for your needs, such as passing a scikit-learn model and defining evaluation metrics. Please refer to [5] for a complete guide.

[5] [Tutorials/DeepDives/EvaluateTutorial.ipynb](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/DeepDives/EvaluateTutorial.ipynb)

In \[11\]:

Copied!

```
# Our clinical metadata has a column called "CANCER_TYPE"
# grouping samples into 5 broad cancer types.
vanillix.evaluate(params=["CANCER_TYPE"])  # TODO: Should also be possible via config using annotation_columns

# Next, we can visualize the evaluation output with:
# TODO: Build a method to avoid exposing the private _visualizer
vanillix.visualizer.show_evaluation(
    param="CANCER_TYPE",
    metric="roc_auc_ovo"
)
```

# Our clinical metadata has a column called "CANCER_TYPE"

# grouping samples into 5 broad cancer types.

vanillix.evaluate(params=["CANCER_TYPE"]) # TODO: Should also be possible via config using annotation_columns

# Next, we can visualize the evaluation output with:

# TODO: Build a method to avoid exposing the private \_visualizer

vanillix.visualizer.show_evaluation( param="CANCER_TYPE", metric="roc_auc_ovo" )

```
Perform ML task with feature df: Latent
Latent
Perform ML task for target parameter: CANCER_TYPE
Showing plot for ML algorithm: LogisticRegression
```

## 3) Inspect Results[¶](#3-inspect-results)

Each step in the pipeline writes its results to the results object of the Vanillix instance.\
In this section, we explore how to access and make sense of the results.\
The attributes of the results object are instances of a `TrainingDynamics` class.\
This class provides a standardized interface for accessing results from different splits and epochs.

##### TrainingDynamics Object in Results[¶](#trainingdynamics-object-in-results)

The `TrainingDynamics` object has the following form:\
`<epoch><split><data>`

For example, to access the train loss for the 5th epoch, you would use:

```
result.loss.get(epoch=5, split="train")
```

##### Special Handling[¶](#special-handling)

- If an invalid split is provided, a `KeyError` is raised
- Negative epoch indices work like Python list indexing (-1 is the last epoch)
- If an epoch doesn't exist, an empty array or dictionary is returned

______________________________________________________________________

For more details on how to work with TrainingDynamics and the Result object, refer to [5] [5] [Tutorials/DeepDives/PipelineOutputTutorial.ipynb](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/DeepDives/PipelineOutputTutorial.ipynb)

In \[12\]:

Copied!

```
# Get the latent spaces of the test split
rec_test = result.reconstructions.get(split="test")

# Get sample_ids for this latent space
sample_ids = result.sample_ids.get(split="test")

# A utility wrapper to get the latent space as a DataFrame
result.get_latent_df(epoch=-1, split="test")
```

# Get the latent spaces of the test split

rec_test = result.reconstructions.get(split="test")

# Get sample_ids for this latent space

sample_ids = result.sample_ids.get(split="test")

# A utility wrapper to get the latent space as a DataFrame

result.get_latent_df(epoch=-1, split="test")

Out\[12\]:

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




TCGA-13-1481-01
1.209185
0.531535
-0.527219
-0.491622
-0.290974
-0.311621
1.481660
-0.618345
-0.170473
-0.300490
0.870092
1.605035
1.067834
0.259719
-0.148721
-1.013062


TCGA-13-1505-01
0.944583
2.431056
1.349156
-1.322856
-0.449180
-0.266392
3.401706
0.173875
-0.479915
-0.800582
0.729345
3.361618
2.464790
-0.704554
-0.512607
-1.826883


TCGA-22-5491-01
-0.696727
1.686747
0.588093
-0.676969
1.154979
-2.120204
1.613690
-1.886487
-0.083105
1.387410
-2.589422
0.285199
1.382209
-1.589720
2.569001
-0.350600


TCGA-24-1103-01
1.857729
1.498289
-0.605264
-1.107758
-0.121941
-0.795754
2.683646
-1.515085
-1.146217
-0.780832
1.645314
2.2982
... [output truncated]
```

## 4) Visualize Results[¶](#4-visualize-results)

During the pipeline, the visualize step already prepared some plots.\
You can call the `show_result` method to display the actual plots. By passing the keyword argument `params` and providing a list of columns from the annotation data, you get visualizations based on these columns. In your example we use the `CANCER_TYPE` column again. This shows us how well the different cancer types are separated in the latent space.\
Refer to [5] for more details on how to work with plots.

[5] [Tutorials/DeepDives/VisualizeTutorial.ipynb](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/DeepDives/VisualizeTutorial.ipynb)

In \[13\]:

Copied!

```
vanillix.show_result(params=["CANCER_TYPE"])
```

vanillix.show_result(params=["CANCER_TYPE"])

```
Creating plots ...
```

## 5) Customize[¶](#5-customize)

To customize the behavior of our pipeline, you adjust the configuration.\
There are two ways to work with the config:

1. Create a customized instance of the config class.
1. Provide a `yaml` file and use the config class to read it.

We will focus on option 1 and show a few examples. For a deeper dive into configurations, please refer to [4].

In this section, we demonstrate how to change the following:

- Scaling method
- Number of layers in the autoencoder
- Number of epochs
- Learning rate
- Latent dimensions
- Retrieve information about all config parameters

[4] [Tutorials/DeepDives/ConfigTutorial.ipynb](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/DeepDives/ConfigTutorial.ipynb)

Each pipeline (Vanillix, Varix, Stackix, …) has a corresponding configuration class.\
All configuration classes inherit from a common `DefaultConfig` class, which defines all customizable parameters.\
Child classes override only those parameters that have been set as sensible defaults for their specific pipeline.

In \[14\]:

Copied!

```
from autoencodix.configs import VanillixConfig
# we simply import the specific config an init it with custom parameters
custom_config = VanillixConfig(
    scaling="MINMAX", n_layers=4, epochs=23, learning_rate=1e4, latent_dim=4, data_case="Multi Bulk"
)
# to see a list of all parameters and defaults use:
custom_config.print_schema()
```

from autoencodix.configs import VanillixConfig

# we simply import the specific config an init it with custom parameters

custom_config = VanillixConfig( scaling="MINMAX", n_layers=4, epochs=23, learning_rate=1e4, latent_dim=4, data_case="Multi Bulk" )

# to see a list of all parameters and defaults use:

custom_config.print_schema()

```
VanillixConfig Configuration Parameters:
--------------------------------------------------

data_config:
  Type: <class 'autoencodix.configs.default_config.DataConfig'>
  Default: data_info={} require_common_cells=False annotation_columns=None
  Description: No description available

annotation_columns:
  Type: typing.Optional[typing.List[str]]
  Default: None
  Description: No description available

img_path_col:
  Type: <class 'str'>
  Default: img_paths
  Description: When working with images, we except a column in your annotation file that specifies the path of the image for a particular sample. Here you can define the name of this column

requires_paired:
  Type: typing.Optional[bool]
  Default: PydanticUndefined
  Description: Indicator if the samples for the xmodalix are paired,
... [output truncated]
```

In \[15\]:

Copied!

```
# next we pass the config to our pipeline object
vanillix2 = acx.Vanillix(data=my_datapackage, config=custom_config)
```

# next we pass the config to our pipeline object

vanillix2 = acx.Vanillix(data=my_datapackage, config=custom_config)

```
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
```

## 6) Re-use, Save, Load[¶](#6-re-use-save-load)

Lastly, we show how to re-use a trained pipeline, as well as how to save and load it.\
You can do the following:

- Use the trained pipeline to run the `predict` step with new data.
- Obtain the latent space, apply transformations, and run the `decode` step of the trained pipeline.
- Generate new data by computing the mean and standard deviation of all encoded latent vectors in the chosen split and epoch, fitting a diagonal Gaussian to those statistics, and sampling new latent points by adding scaled Gaussian noise to that mean.
- Access the underlying PyTorch model and re-use it in any way you like.
- Store the pipeline on disk or in the cloud, then reload it and continue using it.

#### Run with New Data[¶](#run-with-new-data)

In \[16\]:

Copied!

```
# For demonstration purposes, we use the DataPackage from step 1 again.
# In real-world scenarios, you would use different data.
result_new_data = vanillix.predict(data=my_datapackage)
```

# For demonstration purposes, we use the DataPackage from step 1 again.

# In real-world scenarios, you would use different data.

result_new_data = vanillix.predict(data=my_datapackage)

```
Reproducibility settings for device cpu are not implemented or necessary i.e. for cpu.
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
n_samples in format recon: 600
train
n_samples from datatpackge: {'paired_count': 600}
```

#### Decode with Latent Space Shift[¶](#decode-with-latent-space-shift)

In \[17\]:

Copied!

```
import numpy as np
import torch
latent = result.latentspaces.get(split="test", epoch=-1)
latent_shift = np.random.normal(0, 0.1, latent.shape)
new_latent = (latent + latent_shift).astype(np.float32)
vanillix.decode(torch.from_numpy(new_latent))
```

import numpy as np import torch latent = result.latentspaces.get(split="test", epoch=-1) latent_shift = np.random.normal(0, 0.1, latent.shape) new_latent = (latent + latent_shift).astype(np.float32) vanillix.decode(torch.from_numpy(new_latent))

Out\[17\]:

```
tensor([[ 1.5274e-01, -3.6452e-01, -5.3712e-01,  ...,  2.6847e-01,
         -5.4154e-01, -6.7944e-01],
        [ 2.1247e-02, -2.8497e-01, -7.1899e-01,  ...,  3.8997e-01,
         -6.6211e-01, -5.5211e-01],
        [ 2.4753e-01,  6.3635e-01,  6.0138e-01,  ...,  5.4399e-01,
          6.5083e-01, -3.9466e-01],
        ...,
        [ 2.3143e+00, -3.6618e-02,  6.0653e-01,  ...,  2.0298e-01,
          5.3058e-01,  1.1579e-03],
        [-3.6200e-01,  2.3870e-01, -1.9901e-01,  ...,  6.1861e-01,
         -4.2072e-02,  6.5539e-01],
        [-2.3190e-01, -9.3487e-02, -1.8447e-01,  ...,  4.6633e-01,
         -4.2388e-01,  4.2086e-01]])
```

#### Generate New Data[¶](#generate-new-data)

While for a variational autoencoder this step is more intuitive, a vanilla autoencoder does not model a formal latent distribution. Instead, the generate method approximates one by computing the mean and standard deviation of all encoded latent vectors in the chosen split and epoch, fitting a diagonal Gaussian to these statistics, and sampling new latent points by adding scaled Gaussian noise to that mean. You can generate data by passing n_samples, or you can bypass this empirical distribution entirely by providing your own latent_prior. This latent_prior needs to have the form `n_samples, latend_dim`. This case is basically the same as calling `decode` directly.

In \[18\]:

Copied!

```
import torch

# Add this at the start of your cell
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
generated_reconstructions = vanillix.generate(n_samples=300)
print(f"Generated reconstructions shape: {generated_reconstructions.shape}")
```

import torch

# Add this at the start of your cell

if torch.backends.mps.is_available(): torch.mps.empty_cache() generated_reconstructions = vanillix.generate(n_samples=300) print(f"Generated reconstructions shape: {generated_reconstructions.shape}")

```
Reproducibility settings for device cpu are not implemented or necessary i.e. for cpu.
Generated reconstructions shape: torch.Size([300, 2000])
```

In \[19\]:

Copied!

```
generated_from_prior = vanillix.generate(latent_prior=new_latent)
print(f"Generated from prior shape: {generated_from_prior.shape}")
```

generated_from_prior = vanillix.generate(latent_prior=new_latent) print(f"Generated from prior shape: {generated_from_prior.shape}")

```
Reproducibility settings for device cpu are not implemented or necessary i.e. for cpu.
Generated from prior shape: torch.Size([600, 2000])
```

#### Obtain PyTorch Model[¶](#obtain-pytorch-model)

In \[20\]:

Copied!

```
model = result.model
# then do something with the model
print(model)
```

model = result.model

# then do something with the model

print(model)

```
VanillixArchitecture(
  (_encoder): Sequential(
    (0): Linear(in_features=2000, out_features=500, bias=True)
    (1): BatchNorm1d(500, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): Dropout(p=0.1, inplace=False)
    (3): ReLU()
    (4): Linear(in_features=500, out_features=125, bias=True)
    (5): BatchNorm1d(125, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): Dropout(p=0.1, inplace=False)
    (7): ReLU()
    (8): Linear(in_features=125, out_features=31, bias=True)
    (9): BatchNorm1d(31, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): Dropout(p=0.1, inplace=False)
    (11): ReLU()
    (12): Linear(in_features=31, out_features=16, bias=True)
  )
  (_decoder): Sequential(
    (0): Linear(in_features=16, out_fe
... [output truncated]
```

#### Save and Load[¶](#save-and-load)

Best practice is to provide a file path without extension, we set the extensions internally and handle all saving and loading.

In \[21\]:

Copied!

```
import os
import glob

outpath = os.path.join("tutorial_res", "van")
vanillix.save(file_path=outpath, save_all=True)

folder = os.path.dirname(outpath)
pkl_files = glob.glob(os.path.join(folder, "*.pkl"))
model_files = glob.glob(os.path.join(folder, "*.pth"))

print("PKL files:", pkl_files)
print("Model files:", model_files)

# The load functionality automatically rebuilds the pipeline object from the saved files
vanillix_loaded = acx.Vanillix.load(outpath)
```

import os import glob outpath = os.path.join("tutorial_res", "van") vanillix.save(file_path=outpath, save_all=True) folder = os.path.dirname(outpath) pkl_files = glob.glob(os.path.join(folder, "\*.pkl")) model_files = glob.glob(os.path.join(folder, "\*.pth")) print("PKL files:", pkl_files) print("Model files:", model_files)

# The load functionality automatically rebuilds the pipeline object from the saved files

vanillix_loaded = acx.Vanillix.load(outpath)

```
Preprocessor saved successfully.
Pipeline object saved successfully.
PKL files: ['tutorial_res/van_preprocessor.pkl', 'tutorial_res/stackix_preprocessor.pkl']
Model files: ['tutorial_res/stackix_model.pth', 'tutorial_res/van_model.pth']
Attempting to load a pipeline from tutorial_res/van...
Pipeline object loaded successfully. Actual type: Vanillix
Preprocessor loaded successfully.
```

In \[22\]:

Copied!

```
vanillix_loaded.predict(data=my_datapackage)
vanillix_loaded.visualize()
```

vanillix_loaded.predict(data=my_datapackage) vanillix_loaded.visualize()

```
Reproducibility settings for device cpu are not implemented or necessary i.e. for cpu.
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
n_samples in format recon: 600
train
n_samples from datatpackge: {'paired_count': 600}
```

In \[23\]:

Copied!

```
vanillix_loaded.visualize()
```

vanillix_loaded.visualize()

In \[24\]:

Copied!

```
vanillix_loaded.show_result(params=["CANCER_TYPE"])
```

vanillix_loaded.show_result(params=["CANCER_TYPE"])

```
Creating plots ...
```

In \[25\]:

Copied!

```
vanillix_loaded.predict(data=my_datapackage)
vanillix_loaded.show_result(params=["CANCER_TYPE"])
```

vanillix_loaded.predict(data=my_datapackage) vanillix_loaded.show_result(params=["CANCER_TYPE"])

```
Reproducibility settings for device cpu are not implemented or necessary i.e. for cpu.
in handle_direct_user_data with data: <class 'autoencodix.data.datapackage.DataPackage'>
n_samples in format recon: 600
train
n_samples from datatpackge: {'paired_count': 600}
Creating plots ...
```
