> **Note:** This environment variable is required for fully deterministic CuBLAS ops on CUDA >= 10.2 when `reproducible=True` is set below. Without it, PyTorch raises a `RuntimeError` instead of training deterministically. It must be set before `torch` is imported. See the [README FAQ](https://jan-forest.github.io/autoencodix_package/tutorials/README.md#reproducibility-and-cublas_workspace_config) for details.

In \[1\]:

Copied!

```
# Setting CUBLAS for reproducibility if you train on GPU
%env CUBLAS_WORKSPACE_CONFIG=:16:8
```

# Setting CUBLAS for reproducibility if you train on GPU

%env CUBLAS_WORKSPACE_CONFIG=:16:8

```
env: CUBLAS_WORKSPACE_CONFIG=:16:8
```

# Explain Step Tutorial[¶](#explain-step-tutorial)

## What you'll learn[¶](#what-youll-learn)

1. Theory Primer
1. How to run explain step
1. How to customize the explain step
1. Using LLMs to get biological insights of the latent dimensions

### 1) Theory Primer[¶](#1-theory-primer)

TODO

### 2) How To Perform the Explain Step[¶](#2-how-to-perform-the-explain-step)

First, we need to run an AUTOENCODIX pipeline, `Ontix` is a very good choice here.

#### ❗❗ Requirements: Getting Tutorial Data ❗❗[¶](#requirements-getting-tutorial-data)

The data for this tutorial is hosted on Hugging Face Hub ([autoencodix/tcga](https://huggingface.co/datasets/autoencodix/tcga)) and is downloaded automatically in the cell below on first run.

#### Extra 2: Get correct path[¶](#extra-2-get-correct-path)

We assume you are in the root of the package. The following code ensures that the correct paths are used. [1] [Tutorials/DeepDives/ConfigTutorial.ipynb](https://jan-forest.github.io/autoencodix_package/tutorials/DeepDives/ExplainStep/ConfigTutorial.ipynb)

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

# ---------------------------------------------------------------------
# Data is hosted on Hugging Face Hub and downloaded (and locally cached)
# automatically, then placed under the paths used below
# ---------------------------------------------------------------------
import shutil
from huggingface_hub import hf_hub_download

HF_REPO_ID = "autoencodix/tcga"
os.makedirs("data/raw", exist_ok=True)
for hf_filename, local_name in [
    ("rna.parquet", "combined_rnaseq_formatted.parquet"),
    ("methylation.parquet", "combined_meth_formatted.parquet"),
    ("clinical.parquet", "combined_clin_formatted.parquet"),
    ("ontology/chromosome_ont_lvl1_ncbi.txt", "chromosome_ont_genelevel_ncbi.txt"),
    ("ontology/chromosome_ont_lvl2.txt", "chromosome_ont_hiddenlevel.txt")
]:
    downloaded_path = hf_hub_download(
        repo_id=HF_REPO_ID, repo_type="dataset", filename=hf_filename
    )
    shutil.copyfile(downloaded_path, os.path.join("data/raw", local_name))
```

import os p = os.getcwd() d = "autoencodix_package" if d not in p: raise FileNotFoundError(f"'{d}' not found in path: {p}") os.chdir(os.sep.join(p.split(os.sep)[: p.split(os.sep).index(d) + 1])) print(f"Changed to: {os.getcwd()}")

# ---------------------------------------------------------------------

# Data is hosted on Hugging Face Hub and downloaded (and locally cached)

# automatically, then placed under the paths used below

# ---------------------------------------------------------------------

import shutil from huggingface_hub import hf_hub_download HF_REPO_ID = "autoencodix/tcga" os.makedirs("data/raw", exist_ok=True) for hf_filename, local_name in \[ ("rna.parquet", "combined_rnaseq_formatted.parquet"), ("methylation.parquet", "combined_meth_formatted.parquet"), ("clinical.parquet", "combined_clin_formatted.parquet"), ("ontology/chromosome_ont_lvl1_ncbi.txt", "chromosome_ont_genelevel_ncbi.txt"), ("ontology/chromosome_ont_lvl2.txt", "chromosome_ont_hiddenlevel.txt") \]: downloaded_path = hf_hub_download( repo_id=HF_REPO_ID, repo_type="dataset", filename=hf_filename ) shutil.copyfile(downloaded_path, os.path.join("data/raw", local_name))

```
Changed to: /home/janew/ACX/autoencodix_package
```

### 2a) Creating a synthetic ground truth[¶](#2a-creating-a-synthetic-ground-truth)

Feature importance by attribution scores is hard to quantify for its performance and reliability of outcome. Hence, we will introduce in this tutorial a little synthetic signal on selected genes of a specific chromosome only for a group of sample to check if they will be picked up by our xAI method. Hence, we will alter the gene expression and methylation intensities of some genes of the 21 Chromosome only for 'male' samples (compare also [Ontix tutorial](https://jan-forest.github.io/autoencodix_package/tutorials/DeepDives/PipelineTutorials/Ontix.ipynb)).

In \[3\]:

Copied!

```
import numpy as np
import pandas as pd
### Test with synthetic signal ###

def synth_signal(df, feature_list, sample_list):
	max_q90 = df.stack().quantile(0.9) # Our signal expression
	for sample in sample_list:
			df.loc[sample, feature_list] = max_q90*4 + np.random.normal(loc=0, scale=max_q90 * 0.25) # Add some noise to the signal
	return df

# Some genes on chromosome 21 which we will alter
# be aware that some will be filtered out during preprocessing by variance filtering
feature_synth = [
	"102724219",	# 21:p12
	"105379499",	# 21:p12
	"246312",	# 21:q21.1
	"105372749",	# 21:q21.2
	"266917",	# 21:q21.2
	"5651",	# 21:q21.1
	"105369292",	# 21:q21.1
	"102724951",	# 21:p11.2
	"1525",	# 21:q21.1
	"101927843",	# 21:q21.1
	"378828",	# 21:q21.1
	"118421",	# 21:q21.3
	"284825",	# 21:q21.3
	"101927869",	# 21:q21.2
	"9875",	# 21:q22.11
	"100874059",	# 13:q12.3
	"100131902",	# 21:q22.11
	"58494",	# 21:q21.3
	"105372746",	# 21:q21.1
]

df_rna = pd.read_parquet("data/raw/"+"combined_rnaseq_formatted.parquet")
df_meth = pd.read_parquet("data/raw/"+"combined_meth_formatted.parquet")
df_anno = pd.read_parquet("data/raw/"+"combined_clin_formatted.parquet")

sample__intersection = df_rna.index.intersection(df_meth.index).intersection(df_anno.index)
sample_group = "Male"
sample_synth = df_anno.loc[df_anno["SEX"] == sample_group].index.intersection(sample__intersection).to_list()

df_rna = synth_signal(df_rna, feature_list=feature_synth, sample_list=sample_synth)
df_meth = synth_signal(df_meth, feature_list=feature_synth, sample_list=sample_synth)

# Save the modified dataframes
df_rna.to_parquet("data/raw/"+"combined_rnaseq_formatted_synth.parquet")
df_meth.to_parquet("data/raw/"+"combined_meth_formatted_synth.parquet")
```

import numpy as np import pandas as pd

### Test with synthetic signal

def synth_signal(df, feature_list, sample_list): max_q90 = df.stack().quantile(0.9) # Our signal expression for sample in sample_list: df.loc[sample, feature_list] = max_q90\*4 + np.random.normal(loc=0, scale=max_q90 * 0.25) # Add some noise to the signal return df

# Some genes on chromosome 21 which we will alter

# be aware that some will be filtered out during preprocessing by variance filtering

feature_synth = [ "102724219", # 21:p12 "105379499", # 21:p12 "246312", # 21:q21.1 "105372749", # 21:q21.2 "266917", # 21:q21.2 "5651", # 21:q21.1 "105369292", # 21:q21.1 "102724951", # 21:p11.2 "1525", # 21:q21.1 "101927843", # 21:q21.1 "378828", # 21:q21.1 "118421", # 21:q21.3 "284825", # 21:q21.3 "101927869", # 21:q21.2 "9875", # 21:q22.11 "100874059", # 13:q12.3 "100131902", # 21:q22.11 "58494", # 21:q21.3 "105372746", # 21:q21.1 ] df_rna = pd.read_parquet("data/raw/"+"combined_rnaseq_formatted.parquet") df_meth = pd.read_parquet("data/raw/"+"combined_meth_formatted.parquet") df_anno = pd.read_parquet("data/raw/"+"combined_clin_formatted.parquet") sample\_\_intersection = df_rna.index.intersection(df_meth.index).intersection(df_anno.index) sample_group = "Male" sample_synth = df_anno.loc\[df_anno["SEX"] == sample_group\].index.intersection(sample\_\_intersection).to_list() df_rna = synth_signal(df_rna, feature_list=feature_synth, sample_list=sample_synth) df_meth = synth_signal(df_meth, feature_list=feature_synth, sample_list=sample_synth)

# Save the modified dataframes

df_rna.to_parquet("data/raw/"+"combined_rnaseq_formatted_synth.parquet") df_meth.to_parquet("data/raw/"+"combined_meth_formatted_synth.parquet")

We can check the result by plotting the gene expression distribution for the altered features (genes):

In \[4\]:

Copied!

```
# Plot the distribution of the modified features compared to all features
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(df_rna.loc[sample_synth,feature_synth].stack(), bins=100, alpha=0.5, label="Male", color="blue")
plt.hist(df_rna.loc[~df_rna.index.isin(sample_synth), feature_synth].stack(), bins=100, alpha=0.5, label="Female/unknown", color="orange")

plt.axvline(
	df_rna.stack().quantile(0.9),
	color="red",
	linestyle="dashed",
	linewidth=2,
	label="Injected Signal Threshold",
)
plt.title(f"Distribution of all injected features with Injected Signal")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()
```

# Plot the distribution of the modified features compared to all features

import matplotlib.pyplot as plt plt.figure(figsize=(10, 6)) plt.hist(df_rna.loc[sample_synth,feature_synth].stack(), bins=100, alpha=0.5, label="Male", color="blue") plt.hist(df_rna.loc[~df_rna.index.isin(sample_synth), feature_synth].stack(), bins=100, alpha=0.5, label="Female/unknown", color="orange") plt.axvline( df_rna.stack().quantile(0.9), color="red", linestyle="dashed", linewidth=2, label="Injected Signal Threshold", ) plt.title(f"Distribution of all injected features with Injected Signal") plt.xlabel("Value") plt.ylabel("Frequency") plt.legend() plt.show()

### 2b) Train an Ontix on the data set with synthetic signal.[¶](#2b-train-an-ontix-on-the-data-set-with-synthetic-signal)

In \[5\]:

Copied!

```
import os
import autoencodix as acx
from autoencodix.configs.default_config import DataConfig, DataInfo, DataCase
from autoencodix.configs import OntixConfig

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
data_root = "data/raw"
# rna_file = "combined_rnaseq_formatted.parquet"
# meth_file = "combined_meth_formatted.parquet"
# clin_file = "combined_clin_formatted.parquet"
rna_file = "combined_rnaseq_formatted_synth.parquet"
meth_file = "combined_meth_formatted_synth.parquet"
clin_file = "combined_clin_formatted.parquet"
ont_genelevel = "chromosome_ont_genelevel_ncbi.txt"
ont_hiddenlevel = "chromosome_ont_hiddenlevel.txt"
# ---------------------------------------------------------------------
# Define individual data modalities
# ---------------------------------------------------------------------
rna_info = DataInfo(
    file_path=os.path.join(data_root, rna_file),
    data_type="NUMERIC",
    filtering="VAR",
)

meth_info = DataInfo(
    file_path=os.path.join(data_root, meth_file),
    data_type="NUMERIC",
    filtering="VAR",
)

anno_info = DataInfo(
    file_path=os.path.join(data_root, clin_file),
    data_type="ANNOTATION",
)
# ---------------------------------------------------------------------
# Combine into DataConfig
# ---------------------------------------------------------------------
data_config = DataConfig(
    data_info={
        "RNA": rna_info,
        "METH": meth_info,
        "ANNO": anno_info,
    },
)

# ---------------------------------------------------------------------
# Define the full DefaultConfig (roughly equivalent to old cfg)
# ---------------------------------------------------------------------
ontix_config = OntixConfig(
    data_config=data_config,
    reproducible=True,
    global_seed=42,
    epochs=100,
    learning_rate=0.0005,
    batch_size=128,
    drop_p=0.3,
    k_filter=2000,
    latent_dim=6,
    # device="cpu",  # default is "auto", which uses a GPU when available; uncomment to force CPU
    reconstruction_loss="mse",
    default_vae_loss="kl",
    beta=0.001,
    save_memory=False,
    scaling="MINMAX",
    train_ratio=0.7,
    test_ratio=0.2,
    valid_ratio=0.1,
    annotation_columns=[
            # "CANCER_TYPE",
            "CANCER_TYPE_ACRONYM",
            # "TMB_NONSYNONYMOUS",
            # "AGE",
            # "OS_STATUS",
            # "GRADE",
            "SEX",
        ],
)

# ---------------------------------------------------------------------
# Now pass into your Ontix object
# ---------------------------------------------------------------------

ont_files = [ont_hiddenlevel, ont_genelevel]
ont_files = [os.path.join(data_root, f) for f in ont_files]
ontix = acx.Ontix(
    ontologies=ont_files,
    config=ontix_config,
)
```

import os import autoencodix as acx from autoencodix.configs.default_config import DataConfig, DataInfo, DataCase from autoencodix.configs import OntixConfig

# ---------------------------------------------------------------------

# Paths

# ---------------------------------------------------------------------

data_root = "data/raw"

# rna_file = "combined_rnaseq_formatted.parquet"

# meth_file = "combined_meth_formatted.parquet"

# clin_file = "combined_clin_formatted.parquet"

rna_file = "combined_rnaseq_formatted_synth.parquet" meth_file = "combined_meth_formatted_synth.parquet" clin_file = "combined_clin_formatted.parquet" ont_genelevel = "chromosome_ont_genelevel_ncbi.txt" ont_hiddenlevel = "chromosome_ont_hiddenlevel.txt"

# ---------------------------------------------------------------------

# Define individual data modalities

# ---------------------------------------------------------------------

rna_info = DataInfo( file_path=os.path.join(data_root, rna_file), data_type="NUMERIC", filtering="VAR", ) meth_info = DataInfo( file_path=os.path.join(data_root, meth_file), data_type="NUMERIC", filtering="VAR", ) anno_info = DataInfo( file_path=os.path.join(data_root, clin_file), data_type="ANNOTATION", )

# ---------------------------------------------------------------------

# Combine into DataConfig

# ---------------------------------------------------------------------

data_config = DataConfig( data_info={ "RNA": rna_info, "METH": meth_info, "ANNO": anno_info, }, )

# ---------------------------------------------------------------------

# Define the full DefaultConfig (roughly equivalent to old cfg)

# ---------------------------------------------------------------------

ontix_config = OntixConfig( data_config=data_config, reproducible=True, global_seed=42, epochs=100, learning_rate=0.0005, batch_size=128, drop_p=0.3, k_filter=2000, latent_dim=6,

# device="cpu", # default is "auto", which uses a GPU when available; uncomment to force CPU

reconstruction_loss="mse", default_vae_loss="kl", beta=0.001, save_memory=False, scaling="MINMAX", train_ratio=0.7, test_ratio=0.2, valid_ratio=0.1, annotation_columns=\[

# "CANCER_TYPE",

"CANCER_TYPE_ACRONYM",

# "TMB_NONSYNONYMOUS",

# "AGE",

# "OS_STATUS",

# "GRADE",

"SEX", \], )

# ---------------------------------------------------------------------

# Now pass into your Ontix object

# ---------------------------------------------------------------------

ont_files = [ont_hiddenlevel, ont_genelevel] ont_files = [os.path.join(data_root, f) for f in ont_files] ontix = acx.Ontix( ontologies=ont_files, config=ontix_config, )

In \[6\]:

Copied!

```
result = ontix.run()
```

result = ontix.run()

```
reading parquet: data/raw/combined_rnaseq_formatted_synth.parquet
```

```
reading parquet: data/raw/combined_meth_formatted_synth.parquet
```

```
reading parquet: data/raw/combined_clin_formatted.parquet
```

```
anno key: paired
```

```
Features in feature_order not found in all_feature_names, first 5: ['100133144', '10357', '10431', '155060', '390284']...
```

```
Features in feature_order not found in all_feature_names, first 5: ['100033413', '100130426', '100133144', '1038', '10431']...
```

```
You are using a CUDA device ('NVIDIA GeForce RTX 5060 Ti') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
```

```
Ontix checks:
All possible feature names length: 27248
Feature order length: 2000
```

```
Feature names without filtering: 2000
Mask layer 0 with shape torch.Size([817, 25]) and 817.0 connections
```

```
Mask layer 1 with shape torch.Size([2000, 817]) and 1950.0 connections
Latent Dim: 25
```

```
/home/janew/ACX/autoencodix_package/src/autoencodix/trainers/_general_trainer.py:305: UserWarning: Gradient clipping was first applied in epoch 3. Total norm of gradients (adjusted for number of features): 5.0206 exceeded max norm of 5.
  warnings.warn(
```

```
Epoch 10 - Train Loss: 203.7841
Sub-losses: recon_loss: 203.7841, var_loss: 0.0000, anneal_factor: 0.0003, effective_beta_factor: 0.0000
Epoch 10 - Valid Loss: 193.2462
Sub-losses: recon_loss: 193.2461, var_loss: 0.0000, anneal_factor: 0.0003, effective_beta_factor: 0.0000
```

```
Epoch 20 - Train Loss: 87.2142
Sub-losses: recon_loss: 87.2133, var_loss: 0.0010, anneal_factor: 0.0020, effective_beta_factor: 0.0000
Epoch 20 - Valid Loss: 81.9489
Sub-losses: recon_loss: 81.9480, var_loss: 0.0009, anneal_factor: 0.0020, effective_beta_factor: 0.0000
```

```
Epoch 30 - Train Loss: 73.3652
Sub-losses: recon_loss: 73.3569, var_loss: 0.0084, anneal_factor: 0.0148, effective_beta_factor: 0.0000
Epoch 30 - Valid Loss: 69.4877
Sub-losses: recon_loss: 69.4795, var_loss: 0.0082, anneal_factor: 0.0148, effective_beta_factor: 0.0000
```

```
Epoch 40 - Train Loss: 70.2031
Sub-losses: recon_loss: 70.1403, var_loss: 0.0628, anneal_factor: 0.0998, effective_beta_factor: 0.0001
Epoch 40 - Valid Loss: 67.1237
Sub-losses: recon_loss: 67.0624, var_loss: 0.0613, anneal_factor: 0.0998, effective_beta_factor: 0.0001
```

```
Epoch 50 - Train Loss: 68.3455
Sub-losses: recon_loss: 68.0337, var_loss: 0.3118, anneal_factor: 0.4502, effective_beta_factor: 0.0005
Epoch 50 - Valid Loss: 65.1846
Sub-losses: recon_loss: 64.8820, var_loss: 0.3026, anneal_factor: 0.4502, effective_beta_factor: 0.0005
```

```
Epoch 60 - Train Loss: 66.7748
Sub-losses: recon_loss: 66.1393, var_loss: 0.6355, anneal_factor: 0.8581, effective_beta_factor: 0.0009
Epoch 60 - Valid Loss: 63.7644
Sub-losses: recon_loss: 63.1313, var_loss: 0.6331, anneal_factor: 0.8581, effective_beta_factor: 0.0009
```

```
Epoch 70 - Train Loss: 65.1900
Sub-losses: recon_loss: 64.4215, var_loss: 0.7685, anneal_factor: 0.9781, effective_beta_factor: 0.0010
Epoch 70 - Valid Loss: 62.3462
Sub-losses: recon_loss: 61.5980, var_loss: 0.7483, anneal_factor: 0.9781, effective_beta_factor: 0.0010
```

```
Epoch 80 - Train Loss: 63.6874
Sub-losses: recon_loss: 62.8687, var_loss: 0.8186, anneal_factor: 0.9970, effective_beta_factor: 0.0010
Epoch 80 - Valid Loss: 61.1732
Sub-losses: recon_loss: 60.3729, var_loss: 0.8004, anneal_factor: 0.9970, effective_beta_factor: 0.0010
```

```
Epoch 90 - Train Loss: 62.6992
Sub-losses: recon_loss: 61.8668, var_loss: 0.8324, anneal_factor: 0.9996, effective_beta_factor: 0.0010
Epoch 90 - Valid Loss: 60.1497
Sub-losses: recon_loss: 59.3039, var_loss: 0.8459, anneal_factor: 0.9996, effective_beta_factor: 0.0010
```

```
Epoch 100 - Train Loss: 61.7623
Sub-losses: recon_loss: 60.9365, var_loss: 0.8257, anneal_factor: 0.9999, effective_beta_factor: 0.0010
Epoch 100 - Valid Loss: 59.2660
Sub-losses: recon_loss: 58.4456, var_loss: 0.8204, anneal_factor: 0.9999, effective_beta_factor: 0.0010
Processed 128 / 707 samples
Processed 256 / 707 samples
Processed 384 / 707 samples
Processed 512 / 707 samples
Processed 640 / 707 samples
Processed 707 / 707 samples
```

### Check if we have a little signal on Chromosome 21 (male/femal seperation)[¶](#check-if-we-have-a-little-signal-on-chromosome-21-malefemal-seperation)

In \[7\]:

Copied!

```
# ontix.show_result(params=["SEX"])
ontix.visualizer.show_latent_space(result=ontix.result, plot_type="Ridgeline", param=["SEX"])
```

# ontix.show_result(params=["SEX"])

ontix.visualizer.show_latent_space(result=ontix.result, plot_type="Ridgeline", param=["SEX"])

##### Now we can call ontix.explain()[¶](#now-we-can-call-ontixexplain)

This will calculate gene-by-latent-dimension attribution scores. The explain step returns these as pd.DataFrame and also saves this DataFrame to `result.embedding_attributions`

In \[8\]:

Copied!

```
# latent_contributions = ontix.explain()
latent_contributions = ontix.explain(
    method="DeepLiftShap", # "DeepLiftShap" or "IntegratedGradients"
    # sel_latent_dim=None, # To calculate contribution for all latent dimensions
    sel_latent_dim='21', # Or specify a single latent dimension to explain
	input_type="grouped", # We will test group 'Male' (input) vs. 'Female' (baseline)
	input_group="Male",
	baseline_type="grouped",
	baseline_group="Female",
	anno_col="SEX",
)
```

# latent_contributions = ontix.explain()

latent_contributions = ontix.explain( method="DeepLiftShap", # "DeepLiftShap" or "IntegratedGradients"

# sel_latent_dim=None, # To calculate contribution for all latent dimensions

sel_latent_dim='21', # Or specify a single latent dimension to explain input_type="grouped", # We will test group 'Male' (input) vs. 'Female' (baseline) input_group="Male", baseline_type="grouped", baseline_group="Female", anno_col="SEX", )

```
.venv/lib/python3.12/site-packages/anndata/_core/anndata.py:1820: UserWarning: Variable names are not unique. To make them unique, call `.var_names_make_unique`.
  utils.warn_names_duplicates("var")
.venv/lib/python3.12/site-packages/captum/attr/_core/deep_lift.py:304: UserWarning: Setting forward, backward hooks and attributes on non-linear
               activations. The hooks and attributes will be removed
            after the attribution is finished
  warnings.warn(
```

```
Start feature attribution calculation
Calculating attributions for latent dimension: 2
```

In \[9\]:

Copied!

```
print(latent_contributions)
```

print(latent_contributions)

```
              21
8755    0.000238
6439    0.000652
1277    0.000301
3481    0.000066
2335    0.001526
...          ...
3224    0.000680
4128    0.004705
143379  0.001751
2555    0.001041
6700    0.009494

[2000 rows x 1 columns]
```

In \[10\]:

Copied!

```
latent_contributions.loc[:, ["21"]].sort_values(by="21", ascending=False)
```

latent_contributions.loc\[:, ["21"]\].sort_values(by="21", ascending=False)

Out\[10\]:

|        | 21       |
| ------ | -------- |
| 337880 | 0.022189 |
| 83887  | 0.019646 |
| 84648  | 0.019497 |
| 254778 | 0.018882 |
| 79368  | 0.018222 |
| ...    | ...      |
| 251    | 0.000002 |
| 27121  | 0.000001 |
| 118430 | 0.000001 |
| 3848   | 0.000001 |
| 4246   | 0.000001 |

2000 rows × 1 columns

#### Now we can plot top 10 highest contribution scores of chromosome 21 and highlight features with synthetic signal[¶](#now-we-can-plot-top-10-highest-contribution-scores-of-chromosome-21-and-highlight-features-with-synthetic-signal)

We see indeed that our groundtruth signal genes show the highest attribution scores

In \[11\]:

Copied!

```
chrom_21_contributions = latent_contributions.loc[:, ["21"]].sort_values(by="21", ascending=False)[:10]

color_map = ["red" if feature in feature_synth else "blue" for feature in chrom_21_contributions.index]

plt.figure(figsize=(10, 6))
plt.bar(chrom_21_contributions.index, chrom_21_contributions.values.flatten(), color=color_map, label="Chromosome 21 Features")
plt.title("Top Chromosome 21 Feature Contributions to Latent Dimension 21")
plt.xlabel("Feature")
plt.ylabel("Contribution Score")
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()
```

chrom_21_contributions = latent_contributions.loc\[:, ["21"]\].sort_values(by="21", ascending=False)[:10] color_map = ["red" if feature in feature_synth else "blue" for feature in chrom_21_contributions.index] plt.figure(figsize=(10, 6)) plt.bar(chrom_21_contributions.index, chrom_21_contributions.values.flatten(), color=color_map, label="Chromosome 21 Features") plt.title("Top Chromosome 21 Feature Contributions to Latent Dimension 21") plt.xlabel("Feature") plt.ylabel("Contribution Score") plt.xticks(rotation=90) plt.legend() plt.tight_layout() plt.show()

```

```

### 3) How to Customize the Explain Step[¶](#3-how-to-customize-the-explain-step)

You have the follwong options:

```
        method: Literal["DeepLiftShap", "IntegratedGradients"] = "DeepLiftShap",
        sel_latent_dim: Union[list, int, str, None] = None,
        input_type: Literal["random", "grouped"] = "grouped",
        input_group: Optional[str] = None,
        baseline_type: Literal["mean", "random", "grouped"] = "mean",
        baseline_group: str = "all",
        anno_col: Optional[str] = None,
        n_subset: int = 100,
        seed_int: int = 12,
        split: Literal["train", "test", "valid"] = "train",
        llm_explain: bool = False,
        llm_client: Literal["ollama", "mistral", "openrouter", "scads-llm"] = "mistral",
        llm_model: str = "mistral-large-latest",
        top_n_genes: int = 40,
        prompt: str = PROMPT,
        save: Union[str, Path, None] = None,
```

**More Details on the Options:**

````
        """Runs the feature-importance explainer and returns gene-by-latent-dimension attribution scores.

        Args:
            method:  Specifies which attribution algorithm to use for explaining the model.
            sel_latent_dim: Specifies which latent dimension(s) to explain. If None, all dimensions will be explained.
            input_type: Specifies whether the feature-importance algorithm should use 'random' samples or 'grouped' samples (see input_group) as input for the attribution computation.
            input_group: If input_type is 'grouped', this specifies which subset in anno_col to filter for to use as input for the attribution computation.
            baseline_type: Specifies whether the feature-importance algorithm should use the 'mean' baseline, a 'random'-sample baseline, or samples from a 'grouped' baseline (see baseline_group).
            baseline_group: Specifies whether the baseline is computed using all data (default) or which subset in anno_col to filter for.
            anno_col: If baseline_group is not 'all', this specifies the annotation column used to filter the baseline data.
            n_subset: Specifies the number of cells to use when subsampling for the attribution computation.
            seed_int: Defines the random seed used for reproducible subsampling and attribution calculations.
            split: The split to use for feature importance calculation (train, valid, test), default is train.
            llm_explain: Whether to use LLM explainers for feature importance calculation.
            llm_client: The LLM client to use for feature importance calculation.
            llm_model: The LLM model to use for feature importance calculation.
            top_n_genes: How many top (contribution to embedding) genes to consider for LLM explanation.
            prompt: Prompt to pass to the LLM when explaining a latent dimension.
            save: Path to directory with or without filename to save the generated .md file to. If set to None the file will be saved to the current working directory as "latent explanations.md"

        Returns:
            pd.DataFrame: The generated samples in the input space.
                A DataFrame of attribution scores with genes as the index and latent dimensions as columns.
                Each entry represents the contribution of a given gene to a specific latent dimension.
        """
        ```
````

### 4) How To Use LLM to Get Insights of Gene Contributions[¶](#4-how-to-use-llm-to-get-insights-of-gene-contributions)

We can send the gene attribution score directly to an LLM and get biological insights for each latent dimension, i.e. latent dim 1 is associated with cell cylce .. The LLM will give you and explanation for each latent dim and will save this in an extra markdown file. Additionally, we get parsable JSON that is stored in `result.embedding_explanations`.

Before, we can obtain this, we need to set up our LLM provider (in the future this might be provided by use) See below how to do this:

#### 🧬 Gene Expression Explanation – README[¶](#gene-expression-explanation-readme)

You can get LLM explanations from our `.explain` step by setting `llm_explain=True`. Therefore you need to setup LLM Clients

You can use either:

- **Mistral API** (cloud-based)
- **Ollama** (local models running on your machine)
- **OpenRouter** (cloud-based)
- **ScaDS-LLM** (internal-LLM-server)

#### Requirements[¶](#requirements)

Depending on if you want to use Mistral or Ollama you need to have:

- a Mistral API key
- Ollam installed and at least one model served

#### Environment Setup[¶](#environment-setup)

##### 1. Using Mistral API[¶](#1-using-mistral-api)

Set an API key in your `.env` file in the root of the repository

```
MISTRAL_API_KEY=your_api_key_here
```

**Models**

You may use any model served by Mistral, for example:

- `mistral-small-latest`
- `mistral-medium-latest`
- `mistral-large-latest`

Make sure the `model_name` you pass to `.explain` matches an available Mistral model.

##### 2. Using Ollama[¶](#2-using-ollama)

Ollama runs models **locally**. You must first install Ollama and pull the model you want:

```
ollama pull <model-name>
```

Your `model_name` must match exactly the name of the model served by Ollama:

- `qwen2.5:0.5b`
- `deepseek-r1:8b`

##### 3. Using Openrouter[¶](#3-using-openrouter)

To use a widespread options of models you can use Openrouter. For this simply set a environment variable in your terminal with:

`export OPENROUTER_PREMIUM_API_KEY="sk-my-key"`

##### 4. Using ScaDS-LLM server[¶](#4-using-scads-llm-server)

If you are a ScaDS.AI member or have a ZIH TU Dresden account, you can use our self-hosted LLM server (<https://llm.scads.ai/docs/>).

To enable this, set an environment variable like this:

`export SCADS_LLM_API_KEY="sk-my-key"`

##### 🧬 Using `.explain()` for Gene Expression Interpretation[¶](#using-explain-for-gene-expression-interpretation)

The `.explain()` method can generate a short biological explanation and hypothesis about what is happening in disease vs healthy samples given a list of altered genes.

### Example[¶](#example)

In \[13\]:

Copied!

```
from pathlib import Path

ontix.explain(
	method="IntegratedGradients", # "DeepLiftShap" or "IntegratedGradients"
    # sel_latent_dim=None, # To calculate contribution for all latent dimensions
    sel_latent_dim=['21', 'X'], # Or specify a single latent dimension to explain
	input_type="grouped", # We will test group 'Male' (input) vs. 'Female' (baseline)
	input_group="Male",
	baseline_type="grouped",
	baseline_group="Female",
	anno_col="SEX",
    llm_explain=True,
    llm_client="scads-llm",
    llm_model="alias-code",
    top_n_genes=10,
    save=Path("data/"))
```

from pathlib import Path ontix.explain( method="IntegratedGradients", # "DeepLiftShap" or "IntegratedGradients"

# sel_latent_dim=None, # To calculate contribution for all latent dimensions

sel_latent_dim=['21', 'X'], # Or specify a single latent dimension to explain input_type="grouped", # We will test group 'Male' (input) vs. 'Female' (baseline) input_group="Male", baseline_type="grouped", baseline_group="Female", anno_col="SEX", llm_explain=True, llm_client="scads-llm", llm_model="alias-code", top_n_genes=10, save=Path("data/"))

```
/home/kristina/Dokumente/autoencodix_package/.venv/lib/python3.10/site-packages/anndata/_core/anndata.py:1820: UserWarning: Variable names are not unique. To make them unique, call `.var_names_make_unique`.
  utils.warn_names_duplicates("var")
```

```
Start feature attribution calculation
Calculating attributions for latent dimension: 2
Calculating attributions for latent dimension: 7
Start LLM explanation generation
Explaining latent dimension X with LLM...
Explaining latent dimension 21 with LLM...
Saved explanations to: data/latent_explanations.md
```

Out\[13\]:

|       | 21       | X        |
| ----- | -------- | -------- |
| 8755  | 0.000350 | 0.000780 |
| 6439  | 0.000424 | 0.000447 |
| 1277  | 0.000945 | 0.001406 |
| 3481  | 0.000052 | 0.000050 |
| 2335  | 0.000124 | 0.000245 |
| ...   | ...      | ...      |
| 6335  | 0.008317 | 0.006043 |
| 1378  | 0.000916 | 0.001517 |
| 2070  | 0.001297 | 0.001162 |
| 4640  | 0.009914 | 0.001994 |
| 80231 | 0.008780 | 0.001096 |

2000 rows × 2 columns

This creates a verbal descriptions which is stored in the file `latent_explanations.md` as markdown document with an output like this:

# Latent Dimension X[¶](#latent-dimension-x)

## Most influential genes[¶](#most-influential-genes)

9949, 6103, 3597, 9185, 11040, 56180, 10735, 6303, 10549, 537

## TLDR[¶](#tldr)

This latent dimension captures a T cell immune checkpoint and exhaustion program, integrating inhibitory receptors (PD-1, CTLA4, TIGIT), co-stimulatory signals (CD27, 4-1BB), immunosuppressive mechanisms (IDO1, IL10), and lymphoid tissue organization (CXCL13), likely reflecting the balance between T cell activation and immune suppression in a tumor or inflammatory microenvironment.

## Details[¶](#details)

### Dominant Biological Themes[¶](#dominant-biological-themes)

The gene set is dominated by immune checkpoint molecules (PDCD1/PD-1, CD274/PD-L1, CTLA4, TIGIT), T cell co-stimulatory receptors (CD27, TNFRSF9/4-1BB), immunosuppressive mediators (IDO1, IL10), a T cell growth factor (IL2), and a tertiary lymphoid structure-associated chemokine (CXCL13). Together these define a program of T cell engagement coupled with active immune counter-regulation.

### Mechanistic Hypotheses[¶](#mechanistic-hypotheses)

- T cell exhaustion in the tumor microenvironment: The co-occurrence of multiple inhibitory checkpoints (PD-1, CTLA4, TIGIT) alongside activation-associated molecules (CD27, 4-1BB, IL2) suggests a state where T cells have been chronically stimulated and are transitioning into an exhausted phenotype with progressively impaired effector function.
- Active immunosuppressive remodeling of the immune microenvironment: IDO1-mediated tryptophan catabolism and IL10 secretion represent coordinated enzymatic and cytokine-based suppression of T cell activation, potentially creating a tolerogenic niche that limits anti-tumor or anti-pathogen immunity.
- Immune microenvironment maturation and tertiary lymphoid structure (TLS) formation: CXCL13, a B cell zone-organizing chemokine, combined with checkpoint and co-stimulatory markers, may reflect the emergence of organized ectopic lymphoid structures that support adaptive immune responses and are associated with favorable response to immune checkpoint blockade.

### Summary of Pathways/Processes[¶](#summary-of-pathwaysprocesses)

Implicated pathways include the PD-1/PD-L1 checkpoint axis (KEGG: PD-L1 expression and PD-1 checkpoint pathway), CTLA4-mediated co-inhibition, T cell receptor signaling and downstream T cell activation, IDO1-mediated tryptophan biosynthesis and catabolism (KEGG: tryptophan metabolism), IL10 signaling and anti-inflammatory cytokine production, and CXCL13-CXCR5 chemotaxis involved in B cell zone organization (GO: immune checkpoint, T cell mediated cytotoxicity, cytokine-mediated signaling pathway regulation, positive regulation of T cell activation, and negative regulation of T cell proliferation).

______________________________________________________________________

# Latent Dimension 21[¶](#latent-dimension-21)

## Most influential genes[¶](#most-influential-genes)

58494, 58494, 1525, 9875, 9185, 56180, 10549, 1525, 2010, 537

## TLDR[¶](#tldr)

This latent dimension likely captures a proliferative-structural balance axis in which growth signaling (EGFR, MYC) and apoptosis control (TP53) interact with epithelial/structural identity (CDH1, FBN1, ACTB) and inflammatory crosstalk (MIF).

## Details[¶](#details)

### Dominant Biological Themes[¶](#dominant-biological-themes)

The gene set centers on three interconnected themes: (1) proliferative and survival signaling through EGFR, MYC, and TP53, which collectively govern the decision between cell growth, cycle progression, and apoptosis; (2) epithelial and structural identity maintained by CDH1 (cell-cell adhesion), FBN1 (extracellular matrix), and ACTB (cytoskeletal integrity); and (3) inflammatory and immune-stromal crosstalk mediated by MIF, which can modulate proliferation, anti-apoptosis, and immune cell recruitment in the tissue microenvironment.

### Mechanistic Hypotheses[¶](#mechanistic-hypotheses)

- Hypothesis 1 - Proliferative-Structural Tension: This dimension encodes the degree to which a cell or tissue compartment prioritizes proliferative signaling (EGFR, MYC) over maintenance of structural and epithelial identity (CDH1, FBN1, ACTB). High values may reflect tissues undergoing active remodeling or early neoplastic transformation where growth cues dominate but full dedifferentiation has not yet occurred.
- Hypothesis 2 - Inflammation-Driven Proliferation Axis: MIF acts as a bridge between the inflammatory microenvironment and intrinsic growth programs. This dimension may capture states where chronic inflammatory signaling (MIF) synergizes with EGFR/MYC-driven proliferation while TP53 activity determines whether cells undergo survival or apoptosis, and CDH1/FBN1 levels reflect the extent of tissue architecture disruption.
- Hypothesis 3 - Early EMT or Dedifferentiation Gradient: The co-occurrence of CDH1 and FBN1 (markers of epithelial/ECM integrity) with EGFR and MYC (drivers of motility and proliferation) and TP53 (gatekeeper of genomic stability) is consistent with a dimension tracking early epithelial-to-mesenchymal transition or stromal dedifferentiation, where structural markers are progressively lost as proliferative and survival programs are upregulated.

### Summary of Pathways/Processes[¶](#summary-of-pathwaysprocesses)

Implicated pathways include: (1) EGFR signaling and MAPK/ERK cascade (EGFR); (2) p53-mediated cell cycle arrest and apoptosis (TP53); (3) MYC-driven transcriptional programs for cell cycle entry and metabolic reprogramming (MYC); (4) cell adhesion and focal adhesion signaling via cadherin-catenin complexes (CDH1); (5) extracellular matrix organization and TGF-beta signaling (FBN1); (6) actin cytoskeleton organization and cell motility (ACTB); (7) inflammatory response and chemokine/cytokine regulation (MIF); and (8) Wnt/beta-catenin signaling, which intersects with CDH1 and can be influenced by MIF.

______________________________________________________________________

Summary generated with client scads-llm, model: alias-code
