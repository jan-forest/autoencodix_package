> **Note:** This environment variable is required for fully deterministic CuBLAS ops on CUDA >= 10.2 when `reproducible=True` is set below. Without it, PyTorch raises a `RuntimeError` instead of training deterministically. It must be set before `torch` is imported. See the [README FAQ](https://jan-forest.github.io/autoencodix_package/tutorials/README.md#reproducibility-and-cublas_workspace_config) for details.

In \[12\]:

Copied!

```
%env CUBLAS_WORKSPACE_CONFIG=:16:8
```

%env CUBLAS_WORKSPACE_CONFIG=:16:8

```
env: CUBLAS_WORKSPACE_CONFIG=:16:8
```

# How to train an ontology-based variational autoencoder (Ontix)[¶](#how-to-train-an-ontology-based-variational-autoencoder-ontix)

In normal autoencoders, latent dimensions are not explainable by design. To gain explainability and incorporate biological information, a popular approach is to restrict the decoder of the autoencoder to match feature connectivity, such as an ontology.

**IMPORTANT**

> This tutorial only shows the specifics of the Ontix pipeline. If you're unfamiliar with general concepts,\
> we recommend following the [`Getting Started - Vanillix`](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/Ontix/Vanillix.ipynb) Tutorial first.

## What You'll Learn[¶](#what-youll-learn)

In this notebook we will show two types of ontologies and how they can be used to train an explainable variational autoencoder, `Ontix`. The first is based on biological pathways from the Reactome database (left), and the second uses chromosomal location of genes (right) as a showcase.

You’ll learn how to:

1. **Initialize** the pipeline, ontologies, and run Ontix with **chromosomal ontologies**.
1. **Initialize** the pipeline, ontologies, and run Ontix with **Reactome pathways**.
1. **Evaluate** Ontix on downstream tasks.

By working with these examples, we will also cover on the go:

- Understand the Ontix-specific **pipeline steps**.
- Access the Ontix-specific **results** (mu, sigma, KL/MMD losses).
- **Visualize** outputs effectively.
- Apply **custom parameters**.
- **Save, load, and reuse** a trained pipeline.

### ❗❗ Requirements: Getting Tutorial Data ❗❗[¶](#requirements-getting-tutorial-data)

The data for this tutorial is hosted on Hugging Face Hub ([autoencodix/tcga](https://huggingface.co/datasets/autoencodix/tcga)) and is downloaded automatically in the cells below on first run.

### Extra 2: Get correct path[¶](#extra-2-get-correct-path)

We assume you are in the root of the package. The following code ensures that the correct paths are used. [1] [Tutorials/DeepDives/ConfigTutorial.ipynb](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/DeepDives/ConfigTutorial.ipynb)

In \[13\]:

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
Changed to: /home/ewald/Github/autoencodix_package
```

## 1) Set-up Ontology and Initialize Pipeline with Chromosomal Ontology[¶](#1-set-up-ontology-and-initialize-pipeline-with-chromosomal-ontology)

The only thing you need to do to train an `Ontix` model is to provide information about your ontologies. You can do this via `.txt` files or directly as Python dictionaries. In this example, we use text files. We plan to extend this tutorial in the future with an example using Python dictionaries (TODO JE).

You can provide up to `n` ontology files and pass them as a list to the `Ontix` pipeline object. The list should be ordered so that the last ontology level corresponds to the mapping of your features (e.g., Gene ID) to an ontology level like subpathways or cytobands of chromosomes. More levels are optional but recommended, as they map gene names to higher-level categories like top-level pathways or chromosome regions.

The mapping of the last level should have the format:\
Gene 1 `separator` Pathway1\
Gene 2 `separator` Pathway1\
Gene 3 `separator` Pathway2

#### 1.1 Prepare Ontology Data[¶](#11-prepare-ontology-data)

We quickly show how to prepare a downloaded ontology into a text file that serves as input for `Ontix`.

**Example 1: Set-up chromosomal ontology**:

From Ensembl via Biomart or any other sequence database, you can get cytoband (karyotype) and chromosomal information for human genes like this:

In \[14\]:

Copied!

```
import pandas as pd

# Download from https://github.com/jan-forest/autoencodix/blob/main/Tutorials/genes_chromosomes.txt
df_genes = pd.read_csv("genes_chromosomes.txt", sep="\t")
df_genes.head()
```

import pandas as pd

# Download from https://github.com/jan-forest/autoencodix/blob/main/Tutorials/genes_chromosomes.txt

df_genes = pd.read_csv("genes_chromosomes.txt", sep="\\t") df_genes.head()

Out\[14\]:

|     | Gene stable ID  | Gene stable ID version | Karyotype band | Chromosome/scaffold name | Gene start (bp) | Gene end (bp) | HGNC symbol | NCBI gene (formerly Entrezgene) ID |
| --- | --------------- | ---------------------- | -------------- | ------------------------ | --------------- | ------------- | ----------- | ---------------------------------- |
| 0   | ENSG00000198888 | ENSG00000198888.2      | NaN            | MT                       | 3307            | 4262          | MT-ND1      | 4535                               |
| 1   | ENSG00000198763 | ENSG00000198763.3      | NaN            | MT                       | 4470            | 5511          | MT-ND2      | 4536                               |
| 2   | ENSG00000198804 | ENSG00000198804.2      | NaN            | MT                       | 5904            | 7445          | MT-CO1      | 4512                               |
| 3   | ENSG00000210151 | ENSG00000210151.2      | NaN            | MT                       | 7446            | 7514          | MT-TS1      | 113219467                          |
| 4   | ENSG00000198712 | ENSG00000198712.1      | NaN            | MT                       | 7586            | 8269          | MT-CO2      | 4513                               |

We have to solve some issues before we can use this as ontology.\
(1) We only want chromosomes and not scaffolds\
(2) Karyotype/cytoband should have identification of the chromosome in their name as identifier

In \[15\]:

Copied!

```
df_genes = df_genes.loc[
    df_genes["Chromosome/scaffold name"].str.len() < 3
]  ## get rid of scaffolds and keep only chromosomes
df_genes.loc[df_genes["Chromosome/scaffold name"] == "MT", "Karyotype band"] = (
    "MT"  ## create missing karyotype for mito genes
)
print("This will be our chromosomes and latent dimensions in Ontix:")
print(df_genes["Chromosome/scaffold name"].unique())
print(f"Latent dimension: {len(df_genes['Chromosome/scaffold name'].unique())}")
```

df_genes = df_genes.loc\[ df_genes["Chromosome/scaffold name"].str.len() < 3 \] ## get rid of scaffolds and keep only chromosomes df_genes.loc\[df_genes["Chromosome/scaffold name"] == "MT", "Karyotype band"\] = ( "MT" ## create missing karyotype for mito genes ) print("This will be our chromosomes and latent dimensions in Ontix:") print(df_genes["Chromosome/scaffold name"].unique()) print(f"Latent dimension: {len(df_genes['Chromosome/scaffold name'].unique())}")

```
This will be our chromosomes and latent dimensions in Ontix:
['MT' 'Y' '21' '13' '18' '22' '20' 'X' '15' '14' '10' '9' '8' '16' '4' '5'
 '7' '6' '19' '12' '11' '3' '17' '2' '1']
Latent dimension: 25
```

In \[16\]:

Copied!

```
# Combine Chromosome name and cytoband
df_genes = df_genes.copy()
df_genes.loc[:, "Chr_and_karyotype"] = df_genes.loc[
    :, ["Chromosome/scaffold name", "Karyotype band"]
].apply(lambda x: ":".join(x.values.tolist()), axis=1)
print("This will be our hidden layer in the sparse decoder:")
print(df_genes["Chr_and_karyotype"].unique()[0:20])
print(f"Hidden layer dim: {len(df_genes['Chr_and_karyotype'].unique())}")
```

# Combine Chromosome name and cytoband

df_genes = df_genes.copy() df_genes.loc[:, "Chr_and_karyotype"] = df_genes.loc\[ :, ["Chromosome/scaffold name", "Karyotype band"] \].apply(lambda x: ":".join(x.values.tolist()), axis=1) print("This will be our hidden layer in the sparse decoder:") print(df_genes["Chr_and_karyotype"].unique()[0:20]) print(f"Hidden layer dim: {len(df_genes['Chr_and_karyotype'].unique())}")

```
This will be our hidden layer in the sparse decoder:
['MT:MT' 'Y:p11.2' 'Y:q11.223' 'Y:q11.221' 'Y:q11.222' 'Y:q11.23'
 'Y:p11.31' 'Y:p11.32' 'Y:q12' '21:p12' '21:q21.1' '21:q21.2' '21:p11.2'
 '13:q12.12' '21:q21.3' '21:q22.11' '13:q12.3' '13:q14.12' '13:q14.2'
 '21:q22.12']
Hidden layer dim: 817
```

Now we can save this as files in the correct format for the two levels:

In \[17\]:

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
# Paths
# ---------------------------------------------------------------------
data_root = "data/raw"
os.makedirs(data_root, exist_ok=True)
ont_genelevel = "chromosome_ont_genelevel_ncbi.txt"
ont_hiddenlevel = "chromosome_ont_hiddenlevel.txt"

# FEATURE TO TO ONTOLOGY LEVEL
df_genes[
    [
        "NCBI gene (formerly Entrezgene) ID",
        "Chr_and_karyotype",
    ]  # Level Feature: Feature (gene) to hidden layer (cytoband)
].drop_duplicates(  # Chromosomal ontology must be unique
).to_csv(
    os.path.join(data_root, ont_genelevel),
    sep="\t",
    header=False,
    index=False,
)

# ONTOLOGY TO LATENT DIMENSION LEVEL
df_genes[
    [
        "Chr_and_karyotype",
        "Chromosome/scaffold name",
    ]  # Level Hidden: hidden layer (cytoband) to latent dimension (chromosome)
].drop_duplicates(  # Chromosomal ontology must be unique
).to_csv(
    os.path.join(data_root, ont_hiddenlevel),
    sep="\t",
    header=False,
    index=False,
)
```

import os p = os.getcwd() d = "autoencodix_package" if d not in p: raise FileNotFoundError(f"'{d}' not found in path: {p}") os.chdir(os.sep.join(p.split(os.sep)[: p.split(os.sep).index(d) + 1])) print(f"Changed to: {os.getcwd()}")

# ---------------------------------------------------------------------

# Paths

# ---------------------------------------------------------------------

data_root = "data/raw" os.makedirs(data_root, exist_ok=True) ont_genelevel = "chromosome_ont_genelevel_ncbi.txt" ont_hiddenlevel = "chromosome_ont_hiddenlevel.txt"

# FEATURE TO TO ONTOLOGY LEVEL

df_genes\[ [ "NCBI gene (formerly Entrezgene) ID", "Chr_and_karyotype", ] # Level Feature: Feature (gene) to hidden layer (cytoband) \].drop_duplicates( # Chromosomal ontology must be unique ).to_csv( os.path.join(data_root, ont_genelevel), sep="\\t", header=False, index=False, )

# ONTOLOGY TO LATENT DIMENSION LEVEL

df_genes\[ [ "Chr_and_karyotype", "Chromosome/scaffold name", ] # Level Hidden: hidden layer (cytoband) to latent dimension (chromosome) \].drop_duplicates( # Chromosomal ontology must be unique ).to_csv( os.path.join(data_root, ont_hiddenlevel), sep="\\t", header=False, index=False, )

```
Changed to: /home/ewald/Github/autoencodix_package
```

#### 1.2 Create Your Config and Run Pipeline[¶](#12-create-your-config-and-run-pipeline)

The initialization of the config does not work differently as for other pipelines, but with the constraint that we only allow `MINMAX` scaling as scaling method. The `ontix` pipeline itself takes an additional `ontologies` keyword argument. Here you pass either the list of ontology files (with the pathway-to-gene-id mapping as last entry).

In \[18\]:

Copied!

```
import os
import autoencodix as acx
from autoencodix.configs.default_config import DataConfig, DataInfo, DataCase
from autoencodix.configs import OntixConfig

# ---------------------------------------------------------------------
# Data is hosted on Hugging Face Hub and downloaded (and locally cached) automatically
# ---------------------------------------------------------------------
from huggingface_hub import hf_hub_download

HF_REPO_ID = "autoencodix/tcga"
rna_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="rna.parquet")
meth_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="methylation.parquet")
clin_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="clinical.parquet")

# ---------------------------------------------------------------------
# Define individual data modalities
# ---------------------------------------------------------------------
rna_info = DataInfo(
    file_path=rna_path,
    data_type="NUMERIC",
    filtering="VAR",
)

meth_info = DataInfo(
    file_path=meth_path,
    data_type="NUMERIC",
    filtering="VAR",
)

anno_info = DataInfo(
    file_path=clin_path,
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
    checkpoint_interval=20,
    learning_rate=0.0005,
    batch_size=128,
    drop_p=0.3,
    k_filter=2000,
    latent_dim=6,
    device="cpu",
    reconstruction_loss="mse",
    default_vae_loss="kl",
    beta=0.001,
    save_memory=False,
    scaling="MINMAX",
	annotation_columns=[
        "CANCER_TYPE_ACRONYM",
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

# Data is hosted on Hugging Face Hub and downloaded (and locally cached) automatically

# ---------------------------------------------------------------------

from huggingface_hub import hf_hub_download HF_REPO_ID = "autoencodix/tcga" rna_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="rna.parquet") meth_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="methylation.parquet") clin_path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename="clinical.parquet")

# ---------------------------------------------------------------------

# Define individual data modalities

# ---------------------------------------------------------------------

rna_info = DataInfo( file_path=rna_path, data_type="NUMERIC", filtering="VAR", ) meth_info = DataInfo( file_path=meth_path, data_type="NUMERIC", filtering="VAR", ) anno_info = DataInfo( file_path=clin_path, data_type="ANNOTATION", )

# ---------------------------------------------------------------------

# Combine into DataConfig

# ---------------------------------------------------------------------

data_config = DataConfig( data_info={ "RNA": rna_info, "METH": meth_info, "ANNO": anno_info, }, )

# ---------------------------------------------------------------------

# Define the full DefaultConfig (roughly equivalent to old cfg)

# ---------------------------------------------------------------------

ontix_config = OntixConfig( data_config=data_config, reproducible=True, global_seed=42, epochs=100, checkpoint_interval=20, learning_rate=0.0005, batch_size=128, drop_p=0.3, k_filter=2000, latent_dim=6, device="cpu", reconstruction_loss="mse", default_vae_loss="kl", beta=0.001, save_memory=False, scaling="MINMAX", annotation_columns=[ "CANCER_TYPE_ACRONYM", "SEX", ], )

# ---------------------------------------------------------------------

# Now pass into your Ontix object

# ---------------------------------------------------------------------

ont_files = [ont_hiddenlevel, ont_genelevel] ont_files = [os.path.join(data_root, f) for f in ont_files] ontix = acx.Ontix( ontologies=ont_files, config=ontix_config, )

In \[19\]:

Copied!

```
r = ontix.run()
```

r = ontix.run()

```
reading parquet: /home/ewald/.cache/huggingface/hub/datasets--autoencodix--tcga/snapshots/c59ee679d11ea17c0fd615248a382ef1c38ef456/rna.parquet
reading parquet: /home/ewald/.cache/huggingface/hub/datasets--autoencodix--tcga/snapshots/c59ee679d11ea17c0fd615248a382ef1c38ef456/methylation.parquet
reading parquet: /home/ewald/.cache/huggingface/hub/datasets--autoencodix--tcga/snapshots/c59ee679d11ea17c0fd615248a382ef1c38ef456/clinical.parquet
anno key: paired
Features in feature_order not found in all_feature_names, first 5: ['100133144', '10357', '10431', '155060', '390284']...
Features in feature_order not found in all_feature_names, first 5: ['100033413', '100130426', '100133144', '1038', '10431']...
Reproducibility settings for device cpu are not implemented or necessary i.e. for cpu.
Onti
... [output truncated]
```

### Gaining insights by explainable latent dimensions and visualizations[¶](#gaining-insights-by-explainable-latent-dimensions-and-visualizations)

The idea behind using the chromosomal location of features (genes) as an ontology for `Ontix` is that we can expect male and female patients to be separated along the `Y` and `X` chromosomes.\
Let’s check if this is indeed the case:

In \[20\]:

Copied!

```
ontix.show_result(params=["SEX"])
# ontix.show_result()
```

ontix.show_result(params=["SEX"])

# ontix.show_result()

```
Creating plots ...
```

## 2) Set-up Ontology and Initialize Pipeline with Reactome Pathways[¶](#2-set-up-ontology-and-initialize-pipeline-with-reactome-pathways)

Chromosomal location as ontology is mostly a proof of concept as shown above how to gain explainability of latent dimensions.\
In practice, something like biological pathways or gene ontology is most commonly used to gain biological insights.

In \[21\]:

Copied!

```
from huggingface_hub import hf_hub_download

reactome_hiddenlevel = hf_hub_download(
    repo_id="autoencodix/tcga", repo_type="dataset", filename="ontology/reactome_ont_lvl2.txt"
)
reactome_genelevel = hf_hub_download(
    repo_id="autoencodix/tcga", repo_type="dataset", filename="ontology/reactome_ont_lvl1.txt"
)
ont_files = [reactome_hiddenlevel, reactome_genelevel]
ontix_react = acx.Ontix(ontologies=ont_files, config=ontix_config)
```

from huggingface_hub import hf_hub_download reactome_hiddenlevel = hf_hub_download( repo_id="autoencodix/tcga", repo_type="dataset", filename="ontology/reactome_ont_lvl2.txt" ) reactome_genelevel = hf_hub_download( repo_id="autoencodix/tcga", repo_type="dataset", filename="ontology/reactome_ont_lvl1.txt" ) ont_files = [reactome_hiddenlevel, reactome_genelevel] ontix_react = acx.Ontix(ontologies=ont_files, config=ontix_config)

In \[22\]:

Copied!

```
result = ontix_react.run()
```

result = ontix_react.run()

```
reading parquet: /home/ewald/.cache/huggingface/hub/datasets--autoencodix--tcga/snapshots/c59ee679d11ea17c0fd615248a382ef1c38ef456/rna.parquet
reading parquet: /home/ewald/.cache/huggingface/hub/datasets--autoencodix--tcga/snapshots/c59ee679d11ea17c0fd615248a382ef1c38ef456/methylation.parquet
reading parquet: /home/ewald/.cache/huggingface/hub/datasets--autoencodix--tcga/snapshots/c59ee679d11ea17c0fd615248a382ef1c38ef456/clinical.parquet
anno key: paired
Features in feature_order not found in all_feature_names, first 5: ['100133144', '100134869', '10357', '10431', '155060']...
Features in feature_order not found in all_feature_names, first 5: ['100033413', '10009', '100130426', '100133144', '10016']...
Reproducibility settings for device cpu are not implemented or necessary i.e. for cpu.
... [output truncated]
```

#### Gaining Insights by Visualizing Latent Dimensions[¶](#gaining-insights-by-visualizing-latent-dimensions)

In comparison to other pipelines, `Ontix` has interpretable latent dimensions as shown in the plots below:

In \[23\]:

Copied!

```
ontix_react.show_result(params=["SEX"])
```

ontix_react.show_result(params=["SEX"])

```
Creating plots ...
```

#### Obtain Results[¶](#obtain-results)

If we're interested in working with latent spaces, reconstructions, or losses, we can access these for `Ontix` as for any other pipeline. For more details see [`Tutorials/DeepDives/PipelineOutputTutorial.ipynb`](https://jan-forest.github.io/autoencodix_package/tutorials/PipelineTutorials/DeepDives/PipelineOutputTutorial.ipynb).

In \[24\]:

Copied!

```
print(f"The following results are saved: {list(r.__dict__.keys())}")
train_loss = r.losses.get(split="train", epoch=20)

print(f"Loss at epoch 20 for split train was: {train_loss}")
ls = r.latentspaces.get(split="test", epoch=-1)
ls
```

print(f"The following results are saved: {list(r.__dict__.keys())}") train_loss = r.losses.get(split="train", epoch=20) print(f"Loss at epoch 20 for split train was: {train_loss}") ls = r.latentspaces.get(split="test", epoch=-1) ls

```
The following results are saved: ['latentspaces', 'sample_ids', 'reconstructions', 'mus', 'sigmas', 'losses', 'sub_losses', 'preprocessed_data', 'model', 'model_checkpoints', 'datasets', 'new_datasets', 'adata_latent', 'final_reconstruction', 'sub_results', 'sub_reconstructions', 'embedding_evaluation', 'embedding_attributions', 'embedding_explanations']
Loss at epoch 20 for split train was: 83.3964628273659
```

Out\[24\]:

```
array([[ 0.9052321 , -2.1160748 ,  6.358244  , ...,  4.720868  ,
         5.377054  ,  6.5613422 ],
       [ 1.3414282 , -0.23278534,  7.4472494 , ...,  6.3337336 ,
         7.8087597 ,  8.431067  ],
       [-0.49043778,  3.119708  ,  9.700561  , ..., 10.339518  ,
        11.5424    , 12.886449  ],
       ...,
       [-0.3785962 ,  2.1159134 , 10.718863  , ..., 11.294542  ,
        10.780063  , 11.17475   ],
       [-1.407029  , -1.4642097 ,  9.627191  , ...,  7.1496205 ,
         8.618527  ,  6.83334   ],
       [ 0.481745  ,  0.07973982,  7.089814  , ...,  7.586668  ,
         7.938776  ,  5.4659786 ]], shape=(707, 25), dtype=float32)
```

#### Save Ontix[¶](#save-ontix)

Like all other pipelines we can save, load and re-use `Ontix` as shown below:

In \[25\]:

Copied!

```
import os
import glob

outpath = os.path.join("tutorial_res", "ontix.pkl")
ontix.save(file_path=outpath, save_all=True)

folder = os.path.dirname(outpath)
pkl_files = glob.glob(os.path.join(folder, "*.pkl"))
model_files = glob.glob(os.path.join(folder, "*.pth"))

print("PKL files:", pkl_files)
print("Model files:", model_files)

# the load functionality automatically will build the pipeline object out of the three saved files
ontix_loaded = acx.Ontix.load(outpath)
```

import os import glob outpath = os.path.join("tutorial_res", "ontix.pkl") ontix.save(file_path=outpath, save_all=True) folder = os.path.dirname(outpath) pkl_files = glob.glob(os.path.join(folder, "\*.pkl")) model_files = glob.glob(os.path.join(folder, "\*.pth")) print("PKL files:", pkl_files) print("Model files:", model_files)

# the load functionality automatically will build the pipeline object out of the three saved files

ontix_loaded = acx.Ontix.load(outpath)

```
Preprocessor saved successfully.
Pipeline object saved successfully.
PKL files: ['tutorial_res/imagix3d_preprocessor.pkl', 'tutorial_res/ontix_preprocessor.pkl', 'tutorial_res/ontix_reactome_preprocessor.pkl', 'tutorial_res/ontix_reactome.pkl', 'tutorial_res/imagix3d.pkl', 'tutorial_res/ontix.pkl']
Model files: ['tutorial_res/ontix_reactome_model.pth', 'tutorial_res/imagix3d_model.pth', 'tutorial_res/ontix_model.pth']
Attempting to load a pipeline from tutorial_res/ontix.pkl...
Pipeline object loaded successfully. Actual type: Ontix
Preprocessor loaded successfully.
```

In \[26\]:

Copied!

```
ontix_loaded.show_result()
```

ontix_loaded.show_result()

```
Creating plots ...
```

In \[27\]:

Copied!

```
test = ontix_react.result.datasets
ontix_react.save(file_path=os.path.join("tutorial_res", "ontix_reactome.pkl"))
ontix_react_loaded = acx.Ontix.load(os.path.join("tutorial_res", "ontix_reactome.pkl"))
ontix_react_loaded.predict(data=test)
```

test = ontix_react.result.datasets ontix_react.save(file_path=os.path.join("tutorial_res", "ontix_reactome.pkl")) ontix_react_loaded = acx.Ontix.load(os.path.join("tutorial_res", "ontix_reactome.pkl")) ontix_react_loaded.predict(data=test)

```
Preprocessor saved successfully.
saving memory efficient
Ontix checks:
All possible feature names length: 11308
Feature order length: 2000
Feature names without filtering: 2000
Mask layer 0 with shape torch.Size([638, 29]) and 641.0 connections
Mask layer 1 with shape torch.Size([2000, 638]) and 12130.0 connections
Latent Dim: 29
Pipeline object saved successfully.
Attempting to load a pipeline from tutorial_res/ontix_reactome.pkl...
Pipeline object loaded successfully. Actual type: Ontix
Preprocessor loaded successfully.
Reproducibility settings for device cpu are not implemented or necessary i.e. for cpu.
Processed 707 / 707 samples
```

Out\[27\]:

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
datasets: DatasetContainer(train=None, valid=None, test=None)
new_datasets: DatasetContainer(train=<autoencodix.data._numeric_dataset.NumericDataset object at 0x7b88bc877260>, valid=<autoencodix.data._numeric_dataset.NumericDataset obje
... [output truncated]
```

#### Generate New Data[¶](#generate-new-data)

For a variational autoencoder, the generate or sample_latent_space step draws new latent vectors from the model’s learned latent distribution.

**Latent Sampling:** The model first aggregates the posterior over all encoded latent vectors in the chosen split and epoch by computing the mean (global_mu) and log-variance (global_logvar). It then samples new latent points from a diagonal Gaussian defined by these aggregate statistics, using the reparameterization trick to inject Gaussian noise.

**Number of Samples (n_samples):** Users can specify how many latent points to generate. The method expands the aggregated mean and log-variance to match the requested number of samples before sampling.

**Custom Latent Prior:** Optionally, a custom latent_prior can be provided (a tensor or NumPy array with shape (n_samples, latent_dim)), which will be used directly instead of the aggregated posterior. This is basically the `decode` step.

In \[28\]:

Copied!

```
generated_reconstructions = ontix_react_loaded.generate(n_samples=5)
print(generated_reconstructions)
```

generated_reconstructions = ontix_react_loaded.generate(n_samples=5) print(generated_reconstructions)

```
Reproducibility settings for device cpu are not implemented or necessary i.e. for cpu.
tensor([[0.0243, 0.0570, 0.0089,  ..., 0.3891, 0.7217, 0.5899],
        [0.0243, 0.0508, 0.0089,  ..., 0.3879, 0.6934, 0.5781],
        [0.0245, 0.0520, 0.0089,  ..., 0.4005, 0.7085, 0.6128],
        [0.0245, 0.0383, 0.0089,  ..., 0.3409, 0.7096, 0.6172],
        [0.0245, 0.0513, 0.0089,  ..., 0.3951, 0.6990, 0.6244]])
```
