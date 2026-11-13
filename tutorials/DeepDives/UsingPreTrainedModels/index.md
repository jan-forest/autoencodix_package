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

# Tutorial showing usage of pretrained autoencoder models[¶](#tutorial-showing-usage-of-pretrained-autoencoder-models)

This tutorial focus on the usage of pretrained models for scRNASeq data and provided under: <https://huggingface.co/collections/autoencodix/acx-pretrained-models>

### Download the model files[¶](#download-the-model-files)

In this tutorial we will use the a Ontix model with explainable latent space based on ontologies generated with GPT5-6 TerraPro:

<https://huggingface.co/autoencodix/Ontix-Dim16-GPT5-6>

In \[2\]:

Copied!

```
from huggingface_hub import snapshot_download
import autoencodix as acx

model_name = "Ontix-Dim16-GPT5-6"
repo_id = f"autoencodix/{model_name}"

model_file = snapshot_download(
	repo_id=repo_id,
)
model_file = model_file + "/large_ontix_final_model_Dim16_GPT5.6_TerraPro.pkl"

print("Loading trained model ...")
loaded_ontix = acx.Ontix.load(file_path=model_file)
loaded_ontix._trainer._config.device = "cpu"  # Switch device if you want to run on CPU or GPU
# loaded_ontix._trainer._config.save_vram = True  # Enable memory saving for prediction
```

from huggingface_hub import snapshot_download import autoencodix as acx model_name = "Ontix-Dim16-GPT5-6" repo_id = f"autoencodix/{model_name}" model_file = snapshot_download( repo_id=repo_id, ) model_file = model_file + "/large_ontix_final_model_Dim16_GPT5.6_TerraPro.pkl" print("Loading trained model ...") loaded_ontix = acx.Ontix.load(file_path=model_file) loaded_ontix.\_trainer.\_config.device = "cpu" # Switch device if you want to run on CPU or GPU

# loaded_ontix.\_trainer.\_config.save_vram = True # Enable memory saving for prediction

```
Fetching 6 files:   0%|          | 0/6 [00:00<?, ?it/s]
```

```
Loading trained model ...
Attempting to load a pipeline from /home/ewald/.cache/huggingface/hub/models--autoencodix--Ontix-Dim16-GPT5-6/snapshots/192a8b01b94afc684c8fa25e6f1a226e6111cc14/large_ontix_final_model_Dim16_GPT5.6_TerraPro.pkl...
Pipeline object loaded successfully. Actual type: Ontix
Preprocessor loaded successfully.
```

## Download and prepare example scRNASeq data[¶](#download-and-prepare-example-scrnaseq-data)

To show how to use scRNASeq with pretrained models, we use an example lung tissue dataset originally extracted from the [CZ CELLxGENE Census](https://chanzuckerberg.github.io/cellxgene-census/) database. It is hosted on Hugging Face Hub ([autoencodix/census-lung](https://huggingface.co/datasets/autoencodix/census-lung)) and is downloaded automatically in the cell below on first run — no manual download needed.

In \[3\]:

Copied!

```
import anndata as ad
from huggingface_hub import hf_hub_download


census_path = hf_hub_download(
    repo_id="autoencodix/census-lung", repo_type="dataset", filename="census_lung.h5ad"
)
adata = ad.read_h5ad(census_path)

# Make sure the index contains Ensembl IDs
adata.var.index = adata.var.feature_id

# Downsample to 10k cells for speed
adata = adata[adata.obs.sample(n=10000, random_state=42).index].copy()
```

import anndata as ad from huggingface_hub import hf_hub_download census_path = hf_hub_download( repo_id="autoencodix/census-lung", repo_type="dataset", filename="census_lung.h5ad" ) adata = ad.read_h5ad(census_path)

# Make sure the index contains Ensembl IDs

adata.var.index = adata.var.feature_id

# Downsample to 10k cells for speed

adata = adata[adata.obs.sample(n=10000, random_state=42).index].copy()

All pretrained models have an individual gene input space which is mainly defined by the ontologies used to make the latent space explainable.

Hence, we need to make sure that our input has the same input space and order. For that we will create an AnnData template file to enforce zero padding of genes missing in the target AnnData file.

In \[4\]:

Copied!

```
# Gene space of the pretrained Ontix model
print(loaded_ontix.result.model.feature_order[:5])
print("Size of the input gene space:", len(loaded_ontix.result.model.feature_order))
print("Size of the dataset gene space:", adata.shape[1])
```

# Gene space of the pretrained Ontix model

print(loaded_ontix.result.model.feature_order[:5]) print("Size of the input gene space:", len(loaded_ontix.result.model.feature_order)) print("Size of the dataset gene space:", adata.shape[1])

```
['ENSG00000187608', 'ENSG00000186891', 'ENSG00000186827', 'ENSG00000176022', 'ENSG00000160087']
Size of the input gene space: 8357
Size of the dataset gene space: 8189
```

In \[5\]:

Copied!

```
import pandas as pd
import numpy as np
from scipy import sparse

# Create template
anndata_template = ad.AnnData(
    X=sparse.csr_matrix(np.zeros((1, len(loaded_ontix.result.model.feature_order)))),
	var=pd.DataFrame(index=loaded_ontix.result.model.feature_order)
)

# Then join with target adata
adata = ad.concat([anndata_template, adata], axis=0, join="outer").copy()

# Subset and reorder the adata object to match the feature space of the pretrained model
adata = adata[1:, loaded_ontix.result.model.feature_order].copy()

adata
```

import pandas as pd import numpy as np from scipy import sparse

# Create template

anndata_template = ad.AnnData( X=sparse.csr_matrix(np.zeros((1, len(loaded_ontix.result.model.feature_order)))), var=pd.DataFrame(index=loaded_ontix.result.model.feature_order) )

# Then join with target adata

adata = ad.concat([anndata_template, adata], axis=0, join="outer").copy()

# Subset and reorder the adata object to match the feature space of the pretrained model

adata = adata[1:, loaded_ontix.result.model.feature_order].copy() adata

Out\[5\]:

```
AnnData object with n_obs × n_vars = 10000 × 8357
    obs: 'soma_joinid', 'dataset_id', 'assay', 'assay_ontology_term_id', 'cell_type', 'cell_type_ontology_term_id', 'development_stage', 'development_stage_ontology_term_id', 'disease', 'disease_ontology_term_id', 'donor_id', 'is_primary_data', 'observation_joinid', 'self_reported_ethnicity', 'self_reported_ethnicity_ontology_term_id', 'sex', 'sex_ontology_term_id', 'suspension_type', 'tissue', 'tissue_ontology_term_id', 'tissue_type', 'tissue_general', 'tissue_general_ontology_term_id', 'raw_sum', 'nnz', 'raw_mean_nnz', 'raw_variance_nnz', 'n_measured_vars'
```

Importantly, models have been trained on log1p normalized counts. Then we simply create a DataPackage for autoencodix to provide only a test set for the prediction step to generate embeddings later.

In \[6\]:

Copied!

```
import scanpy 
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.configs.ontix_config import OntixConfig

scanpy.pp.log1p(adata, copy=False)

test_dataset = NumericDataset(
			data=adata.X, # the normalized expression data per cell
			config=OntixConfig(), # Empty placeholder
			sample_ids=adata.obs.index, # cell ids
			metadata=adata.obs.loc[adata.obs.index,:], # cell annotation data for plotting etc.
			split_indices=None, # Empty placeholder
			feature_ids=adata.var.index, # Gene ids
		)

acx_container = DatasetContainer(train=None, valid=None, test=test_dataset)
```

import scanpy from autoencodix.data.\_numeric_dataset import NumericDataset from autoencodix.data.\_datasetcontainer import DatasetContainer from autoencodix.configs.ontix_config import OntixConfig scanpy.pp.log1p(adata, copy=False) test_dataset = NumericDataset( data=adata.X, # the normalized expression data per cell config=OntixConfig(), # Empty placeholder sample_ids=adata.obs.index, # cell ids metadata=adata.obs.loc[adata.obs.index,:], # cell annotation data for plotting etc. split_indices=None, # Empty placeholder feature_ids=adata.var.index, # Gene ids ) acx_container = DatasetContainer(train=None, valid=None, test=test_dataset)

## Calculate embeddings using a pretrained model[¶](#calculate-embeddings-using-a-pretrained-model)

Now we can calculate the embeddings and visualize them

In \[7\]:

Copied!

```
print("Calculate embeddings ...")
result = loaded_ontix.predict(data=acx_container)
```

print("Calculate embeddings ...") result = loaded_ontix.predict(data=acx_container)

```
Calculate embeddings ...
Processed 10000 / 10000 samples
```

In \[8\]:

Copied!

```
# Create a latent space visualization
loaded_ontix.visualizer.show_latent_space(
	result=result,
	plot_type="Ridgeline",
	param=['cell_type'], 
	split="test",
)
```

# Create a latent space visualization

loaded_ontix.visualizer.show_latent_space( result=result, plot_type="Ridgeline", param=['cell_type'], split="test", )

## Using explain() functionality for xAI[¶](#using-explain-functionality-for-xai)

We can leverage posthoc xAI and LLMs to get a better understanding which genes and processes drive differences between malignant cells and their normal cell-of-origin, alveolar type 2 (AT2) cells

For this we will firstly determine which latent dimensions show highest class separation:

In \[9\]:

Copied!

```
# Perform Linear Discriminant Analysis (LDA) to find the latent dimension that best separates the groups
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_validate

import numpy as np
import pandas as pd

anno_col = "cell_type"
groups_list = [
	"malignant cell",
	"pulmonary alveolar type 2 cell"
]
# Get the latent space as df
df_latent = loaded_ontix.result.get_latent_df(split="test", epoch=-1)

scores = {}
# for each latent dimension
for latent_dim in df_latent.columns:
	lda = LinearDiscriminantAnalysis()
	X = df_latent[latent_dim].values.reshape(-1, 1)
	y = loaded_ontix.result.new_datasets.test.metadata[anno_col].values
	# Reduce x and y to only include samples from the groups of interest
	mask = np.isin(y, groups_list)
	X = X[mask]
	y = y[mask]
	metric = "roc_auc_ovo"
 	# Do CV-5 fold cross-validation and calculate the AUC-ROC score for this latent dimension
	scores[latent_dim] = cross_validate(lda, X, y, cv=5, scoring=metric, return_train_score=True, n_jobs=-1)

scores_df = pd.DataFrame({
	'latent_dim': list(scores.keys()),
	'test_score_mean': [scores[ld]['test_score'].mean() for ld in scores.keys()],
	'test_score_std': [scores[ld]['test_score'].std() for ld in scores.keys()],
	'train_score_mean': [scores[ld]['train_score'].mean() for ld in scores.keys()],
	'train_score_std': [scores[ld]['train_score'].std() for ld in scores.keys()],
})
scores_df = scores_df.sort_values(by='test_score_mean', ascending=False)
scores_df
```

# Perform Linear Discriminant Analysis (LDA) to find the latent dimension that best separates the groups

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis from sklearn.model_selection import cross_validate import numpy as np import pandas as pd anno_col = "cell_type" groups_list = [ "malignant cell", "pulmonary alveolar type 2 cell" ]

# Get the latent space as df

df_latent = loaded_ontix.result.get_latent_df(split="test", epoch=-1) scores = {}

# for each latent dimension

for latent_dim in df_latent.columns: lda = LinearDiscriminantAnalysis() X = df_latent[latent_dim].values.reshape(-1, 1) y = loaded_ontix.result.new_datasets.test.metadata[anno_col].values

# Reduce x and y to only include samples from the groups of interest

mask = np.isin(y, groups_list) X = X[mask] y = y[mask] metric = "roc_auc_ovo"

# Do CV-5 fold cross-validation and calculate the AUC-ROC score for this latent dimension

scores[latent_dim] = cross_validate(lda, X, y, cv=5, scoring=metric, return_train_score=True, n_jobs=-1) scores_df = pd.DataFrame({ 'latent_dim': list(scores.keys()), 'test_score_mean': \[scores[ld]['test_score'].mean() for ld in scores.keys()\], 'test_score_std': \[scores[ld]['test_score'].std() for ld in scores.keys()\], 'train_score_mean': \[scores[ld]['train_score'].mean() for ld in scores.keys()\], 'train_score_std': \[scores[ld]['train_score'].std() for ld in scores.keys()\], }) scores_df = scores_df.sort_values(by='test_score_mean', ascending=False) scores_df

Out\[9\]:

```
latent_dim
test_score_mean
test_score_std
train_score_mean
train_score_std




0
dim01_energy_production_and_central_catabolism
0.770242
0.010872
0.770508
0.002735


15
dim16_redox_detoxification_and_xenobiotic_meta...
0.755611
0.007882
0.755301
0.001928


8
dim09_cytoskeleton_motility_and_cell_adhesion
0.721482
0.006976
0.721591
0.001704


11
dim12_immune_surveillance_inflammation_and_hos...
0.718404
0.019341
0.718893
0.004999


1
dim02_anabolic_biosynthesis_and_biomass_genera...
0.712508
0.012168
0.712717
0.003078


3
dim04_cell_cycle_proliferation_and_division
0.702064
0.009938
0.702320
0.002425


2
dim03_protein_synthesis_folding_and_turnover
0.684031
0.013247
0.684122
0.003327


4
dim05_genome_integrity_and_dna_repair
0.667779
0.014805
0.667835
0.003734


14
dim15_membrane_tran
... [output truncated]
```

The highest separation of malignant cells vs. AT2 cells is observed for the top-ranked latent dimensions in the table above. Now let's calculatue the feature (gene) attribution to those dimensions and identify potential marker genes and get a functional analyses via an LLM.

To use the openrouter API store make yor API key available as environment variable `OPENROUTER_PREMIUM_API_KEY`. Accordingly, using SCADS-LLM server with API key under `SCADS_LLM_API_KEY`. See also `LLM_Setup.md`.

In \[10\]:

Copied!

```
selected_dim = list(scores_df.iloc[0:2]['latent_dim'].values) # Top latent dimensions with highest AUC-ROC score for separating the groups
top_n_genes = 15 # Number of top contributing genes per latent dimension

latent_contributions = loaded_ontix.explain(
    split="test",
    method="IntegratedGradients", # "DeepLiftShap" or "IntegratedGradients"
    n_subset=400, # Randomly sample in input and baseline space for computational efficiency
    sel_latent_dim=selected_dim, # Or specify a single latent dimension to explain    
	input_type="grouped", # We will test group 'malignant cell' (input) vs. 'pulmonary alveolar type 2 cell' (baseline)
	input_group=groups_list[0], # "malignant cell"
	baseline_type="random",
	baseline_group=groups_list[1], # "pulmonary alveolar type 2 cell"
	anno_col=anno_col,
	llm_explain=True, 	# Explain biological functions of the contributing genes using a LLM
	llm_client="openrouter",
	llm_model="openai/gpt-6-luna",
	# llm_client="scads-llm",
	# llm_model="alias-code",
	top_n_genes=top_n_genes,
)
```

selected_dim = list(scores_df.iloc[0:2]['latent_dim'].values) # Top latent dimensions with highest AUC-ROC score for separating the groups top_n_genes = 15 # Number of top contributing genes per latent dimension latent_contributions = loaded_ontix.explain( split="test", method="IntegratedGradients", # "DeepLiftShap" or "IntegratedGradients" n_subset=400, # Randomly sample in input and baseline space for computational efficiency sel_latent_dim=selected_dim, # Or specify a single latent dimension to explain input_type="grouped", # We will test group 'malignant cell' (input) vs. 'pulmonary alveolar type 2 cell' (baseline) input_group=groups_list[0], # "malignant cell" baseline_type="random", baseline_group=groups_list[1], # "pulmonary alveolar type 2 cell" anno_col=anno_col, llm_explain=True, # Explain biological functions of the contributing genes using a LLM llm_client="openrouter", llm_model="openai/gpt-6-luna",

# llm_client="scads-llm",

# llm_model="alias-code",

top_n_genes=top_n_genes, )

```
Start feature attribution calculation
Calculating attributions for latent dimension: 0
Calculating attributions for latent dimension: 15
Start LLM explanation generation
Explaining latent dimension dim01_energy_production_and_central_catabolism with LLM...
Explaining latent dimension dim16_redox_detoxification_and_xenobiotic_metabolism with LLM...
Saved explanations to: /home/ewald/Github/autoencodix_package/Tutorials/DeepDives/latent_explanations.md
```

The LLM-based verbal explaination and summary is stored as a markdown file:

In \[14\]:

Copied!

```
from pathlib import Path
from IPython.display import Markdown, display

md_path = Path("latent_explanations.md")
display(Markdown(md_path.read_text(encoding="utf-8")))
```

from pathlib import Path from IPython.display import Markdown, display md_path = Path("latent_explanations.md") display(Markdown(md_path.read_text(encoding="utf-8")))

```
Latent Dimension dim01_energy_production_and_central_catabolism¶Most influential genes¶ENSG00000198712, ENSG00000198804, ENSG00000198938, ENSG00000198727, ENSG00000198899, ENSG00000198886, ENSG00000198840, ENSG00000198763, ENSG00000198888, ENSG00000198786, ENSG00000124172, ENSG00000198695, ENSG00000228253, ENSG00000087086, ENSG00000131143
TLDR¶The gene set is dominated by mitochondrial-encoded respiratory-chain subunits, suggesting a latent dimension related to oxidative phosphorylation and mitochondrial energy metabolism.
Details¶Dominant Biological Themes¶Most supplied genes encode mitochondrial subunits of respiratory-chain complexes I, III, and IV, as well as ATP synthase. Their coordinated contribution points to mitochondrial oxidative phosphorylation and electron transport; the rema
... [output truncated]
```

## Using the .generate() functionality for synthetic data generation[¶](#using-the-generate-functionality-for-synthetic-data-generation)

Leveraging our pretrained models we can sample from the latent space artificial cells.

We can do this either randomly across the whole latent space or from pre-defined points of the latent space.

We will do here the latter by calculating the mean of our malignant cells for each latent dimension and generate 1000 synthetic cells from this point of the latent space.

In \[12\]:

Copied!

```
df_latent_malignant = df_latent.loc[
	result.new_datasets.test.metadata.cell_type == 'malignant cell',
 :
]

latent_prior_means = df_latent_malignant.mean()

# Generate 1000 latent prior samples based on means and some noise
num_samples = 1000
noise = np.random.normal(0, 0.1, (num_samples, len(latent_prior_means)))
synthetic_latent = np.tile(latent_prior_means.values, (num_samples, 1)) + noise
```

df_latent_malignant = df_latent.loc[ result.new_datasets.test.metadata.cell_type == 'malignant cell', : ] latent_prior_means = df_latent_malignant.mean()

# Generate 1000 latent prior samples based on means and some noise

num_samples = 1000 noise = np.random.normal(0, 0.1, (num_samples, len(latent_prior_means))) synthetic_latent = np.tile(latent_prior_means.values, (num_samples, 1)) + noise

In \[13\]:

Copied!

```
# Generate reconstructions (gene expression) from the synthetic latent samples
generated_reconstructions = loaded_ontix.generate(latent_prior=synthetic_latent)
print("Generated reconstructions shape:", generated_reconstructions.shape)
```

# Generate reconstructions (gene expression) from the synthetic latent samples

generated_reconstructions = loaded_ontix.generate(latent_prior=synthetic_latent) print("Generated reconstructions shape:", generated_reconstructions.shape)

```
Generated reconstructions shape: torch.Size([1000, 8357])
```
