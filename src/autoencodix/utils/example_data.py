import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_blobs  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.data._numeric_dataset import NumericDataset
from autoencodix.utils.default_config import DefaultConfig
from autoencodix.data.datapackage import DataPackage
from autoencodix.utils.default_config import DataCase
import mudata
import anndata

config = DefaultConfig()


def generate_example_data(
    n_samples=1000, n_features=30, n_clusters=5, random_seed=config.global_seed
):
    """
    Generate synthetic data for autoencoder testing and examples.

    Parameters:
        n_samples (int): Number of samples to generate
        n_features (int): Number of features for each sample
        n_clusters (int): Number of clusters to generate
        random_seed (int): Random seed for reproducibility

    Returns:
        DatasetContainer: Container with train, validation and test datasets
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    X, cluster_labels = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.8,
        random_state=random_seed,
    )

    # Create additional metadata features that correlate with the clusters
    metadata_df = pd.DataFrame(
        {
            "cluster": cluster_labels,
            "age": np.random.normal(30, 10, n_samples)
            + cluster_labels * 5,  # Age correlates with cluster
            "size": np.random.uniform(0, 10, n_samples)
            + cluster_labels * 2,  # Size correlates with cluster
            "density": np.random.exponential(1, n_samples)
            * (cluster_labels + 1)
            / 3,  # Density correlates with cluster
            "category": np.random.choice(
                ["A", "B", "C", "D", "E"], n_samples, p=[0.2, 0.2, 0.2, 0.2, 0.2]
            ),  # Random category
        }
    )
    
    # Add some noise features to metadata that don't correlate with clusters
    metadata_df["random_feature"] = np.random.normal(0, 1, n_samples)

    ids = [f"sample_{i}" for i in range(n_samples)]
    metadata_df["sample_id"] = ids
    data_tensor = torch.tensor(X, dtype=torch.float32)

    # Get split ratios from DefaultConfig
    config = DefaultConfig()
    valid_ratio = config.valid_ratio
    test_ratio = config.test_ratio

    # Calculate test_size for first split (everything except train)
    first_test_size = valid_ratio + test_ratio

    # Calculate the proportion of valid in the remaining data
    # If valid_ratio + test_ratio = 0.3 and valid_ratio = 0.1, then valid should be 1/3 of the remaining data
    valid_proportion = valid_ratio / first_test_size if first_test_size > 0 else 0

    train_idx, temp_idx = train_test_split(
        np.arange(n_samples),
        test_size=first_test_size,
        random_state=random_seed,
        stratify=cluster_labels,  # Ensure balanced classes in each split
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - valid_proportion),  # Remaining proportion goes to test
        random_state=random_seed,
        stratify=cluster_labels[temp_idx],  # Ensure balanced classes in each split
    )

    # Create split indicators as numpy arrays
    train_split = np.zeros(n_samples, dtype=bool)
    train_split[train_idx] = True

    val_split = np.zeros(n_samples, dtype=bool)
    val_split[val_idx] = True

    test_split = np.zeros(n_samples, dtype=bool)
    test_split[test_idx] = True

    train_dataset = NumericDataset(
        data=data_tensor[train_idx],
        config=DefaultConfig(),
        sample_ids=[ids[i] for i in train_idx],
        metadata=metadata_df.iloc[train_idx].reset_index(drop=True),
        split_indices=train_split,
        feature_ids=[f"feature_{i}" for i in range(n_features)],
    )

    val_dataset = NumericDataset(
        data=data_tensor[val_idx],
        config=DefaultConfig(),
        sample_ids=[ids[i] for i in val_idx],
        metadata=metadata_df.iloc[val_idx].reset_index(drop=True),
        split_indices=val_split,
        feature_ids=[f"feature_{i}" for i in range(n_features)],
    )

    test_dataset = NumericDataset(
        data=data_tensor[test_idx],
        config=DefaultConfig(),
        sample_ids=[ids[i] for i in test_idx],
        metadata=metadata_df.iloc[test_idx].reset_index(drop=True),
        split_indices=test_split,
        feature_ids=[f"feature_{i}" for i in range(n_features)],
    )

    return DatasetContainer(train=train_dataset, valid=val_dataset, test=test_dataset)


def generate_multi_bulk_example(
    n_samples=500,
    n_features_modality1=100,
    n_features_modality2=80,
    random_seed=config.global_seed,
):
    """
    Generate example data for MULTI_BULK case.

    Parameters:
        n_samples (int): Number of samples to generate
        n_features_modality1 (int): Number of features for first modality
        n_features_modality2 (int): Number of features for second modality
        random_seed (int): Random seed for reproducibility

    Returns:
        DataPackage: DataPackage with multi_bulk data
    """
    np.random.seed(random_seed)

    latent_dim = 10
    latent_representation = np.random.normal(0, 1, (n_samples, latent_dim))

    weights_mod1 = np.random.normal(0, 1, (latent_dim, n_features_modality1))
    weights_mod2 = np.random.normal(0, 1, (latent_dim, n_features_modality2))

    # Generate observed data with noise
    data_mod1 = np.dot(latent_representation, weights_mod1) + np.random.normal(
        0, 0.5, (n_samples, n_features_modality1)
    )
    data_mod2 = np.dot(latent_representation, weights_mod2) + np.random.normal(
        0, 0.5, (n_samples, n_features_modality2)
    )

    mod1_features = [f"gene_{i}" for i in range(n_features_modality1)]
    mod2_features = [f"protein_{i}" for i in range(n_features_modality2)]

    sample_ids = [f"sample_{i}" for i in range(n_samples)]

    # Create DataFrames
    df_mod1 = pd.DataFrame(data_mod1, columns=mod1_features, index=sample_ids)
    df_mod2 = pd.DataFrame(data_mod2, columns=mod2_features, index=sample_ids)

    # Create annotation DataFrame with sample information
    # Adding some metadata that correlates with the latent structure
    group_assignments = np.argmax(latent_representation[:, :3], axis=1)
    conditions = ["condition_" + str(g) for g in group_assignments]
    batch_effects = np.random.choice(["batch_1", "batch_2", "batch_3"], size=n_samples)

    annotation_df = pd.DataFrame(
        {
            "condition": conditions,
            "batch": batch_effects,
            "quality_score": np.random.uniform(0.6, 1.0, n_samples),
        },
        index=sample_ids,
    )

    data_package = DataPackage()
    data_package.multi_bulk = {"transcriptomics": df_mod1, "proteomics": df_mod2}
    data_package.annotation = {
        "transcriptomics": annotation_df.copy(),
        "proteomics": annotation_df.copy(),
    }

    return data_package


def generate_default_bulk_bulk_example(
    n_samples=500, n_features=200, random_seed=config.global_seed
):
    """Generate example data for BULK_TO_BULK case."""
    np.random.seed(random_seed)

    raise NotImplementedError("BULK_TO_BULK example generation not yet implemented")


def generate_default_sc_sc_example(
    n_cells=1000, n_features=500, random_seed=config.global_seed
):
    """Generate example data for SINGLE_CELL_TO_SINGLE_CELL case."""
    np.random.seed(random_seed)

    raise NotImplementedError(
        "SINGLE_CELL_TO_SINGLE_CELL example generation not yet implemented"
    )


def generate_default_img_bulk_example(n_samples=100, random_seed=config.global_seed):
    """Generate example data for IMG_TO_BULK case."""
    np.random.seed(random_seed)

    # Not implemented yet
    raise NotImplementedError("IMG_TO_BULK example generation not yet implemented")


def generate_default_sc_img_example(n_samples=100, random_seed=config.global_seed):
    """Generate example data for SINGLE_CELL_TO_IMG case."""
    np.random.seed(random_seed)

    raise NotImplementedError(
        "SINGLE_CELL_TO_IMG example generation not yet implemented"
    )


def generate_default_img_img_example(n_samples=100, random_seed=config.global_seed):
    """Generate example data for IMG_TO_IMG case."""
    np.random.seed(random_seed)

    raise NotImplementedError("IMG_TO_IMG example generation not yet implemented")


def generate_raw_datapackage(data_case: DataCase, **kwargs):
    """
    Generate a raw DataPackage for the specified data case.

    Parameters:
        data_case (DataCase): The type of data to generate
        **kwargs: Additional arguments to pass to the specific generator function

    Returns:
        DataPackage: Raw data package ready for preprocessing
    """
    generator_map = {
        DataCase.MULTI_BULK: generate_multi_bulk_example,
        DataCase.MULTI_SINGLE_CELL: generate_multi_sc_example,
        DataCase.BULK_TO_BULK: generate_default_bulk_bulk_example,
        DataCase.SINGLE_CELL_TO_SINGLE_CELL: generate_default_sc_sc_example,
        DataCase.IMG_TO_BULK: generate_default_img_bulk_example,
        DataCase.SINGLE_CELL_TO_IMG: generate_default_sc_img_example,
        DataCase.IMG_TO_IMG: generate_default_img_img_example,
    }

    if data_case not in generator_map:
        raise ValueError(f"Unknown data case: {data_case}")

    generator_function = generator_map[data_case]
    return generator_function(**kwargs)


def generate_multi_sc_example(
    n_cells=1000, n_genes=500, n_proteins=200, random_seed=config.global_seed
):
    """
    Generate example data for MULTI_SINGLE_CELL case.

    Parameters:
        n_cells (int): Number of cells to generate
        n_genes (int): Number of genes for RNA modality
        n_proteins (int): Number of proteins for protein modality
        random_seed (int): Random seed for reproducibility

    Returns:
        DataPackage: DataPackage with multi_sc data
    """
    np.random.seed(random_seed)

    cell_ids = [f"cell_{i}" for i in range(n_cells)]

    gene_names = [f"gene_{i}" for i in range(n_genes)]
    protein_names = [f"protein_{i}" for i in range(n_proteins)]

    n_cell_types = 5
    cell_type_probabilities = np.random.dirichlet(np.ones(n_cell_types), size=1)[0]
    cell_types = np.random.choice(
        np.arange(n_cell_types), size=n_cells, p=cell_type_probabilities
    )

    cell_metadata = pd.DataFrame(
        {
            "cell_type": [f"type_{t}" for t in cell_types],
            "batch": np.random.choice(["batch1", "batch2", "batch3"], size=n_cells),
            "donor": np.random.choice(
                ["donor1", "donor2", "donor3", "donor4"], size=n_cells
            ),
            "cell_cycle": np.random.choice(["G1", "S", "G2M"], size=n_cells),
        },
        index=cell_ids,
    )

    # Generate RNA data (sparse count matrix with negative binomial distribution)
    gene_programs = np.zeros((n_cell_types, n_genes))
    for i in range(n_cell_types):
        high_expr_genes = np.random.choice(
            n_genes, size=int(n_genes * 0.2), replace=False
        )
        gene_programs[i, high_expr_genes] = np.random.gamma(
            5, 2, size=len(high_expr_genes)
        )

    lambda_values = np.zeros((n_cells, n_genes))
    for i in range(n_cells):
        lambda_values[i] = gene_programs[cell_types[i]] + np.random.gamma(
            0.5, 0.5, n_genes
        )

    seq_depth = np.random.lognormal(mean=8, sigma=0.5, size=n_cells)
    lambda_values = lambda_values * seq_depth[:, np.newaxis]

    rna_counts = np.random.poisson(lambda_values)
    dropout_prob = 0.3
    dropout_mask = np.random.binomial(1, 1 - dropout_prob, size=rna_counts.shape)
    rna_counts = rna_counts * dropout_mask

    # Generate protein data
    protein_programs = np.zeros((n_cell_types, n_proteins))
    for i in range(n_cell_types):
        high_expr_proteins = np.random.choice(
            n_proteins, size=int(n_proteins * 0.3), replace=False
        )
        protein_programs[i, high_expr_proteins] = np.random.gamma(
            3, 1, size=len(high_expr_proteins)
        )

    protein_levels = np.zeros((n_cells, n_proteins))
    for i in range(n_cells):
        protein_levels[i] = protein_programs[cell_types[i]] + np.random.gamma(
            0.2, 0.3, n_proteins
        )

    batch_effect = np.zeros(n_cells)
    batch_map = {"batch1": 0.8, "batch2": 1.0, "batch3": 1.2}
    for i in range(n_cells):
        batch_effect[i] = batch_map[cell_metadata.iloc[i]["batch"]]

    protein_levels = protein_levels * batch_effect[:, np.newaxis]
    protein_levels = protein_levels + np.random.normal(
        0, 0.1, size=protein_levels.shape
    )
    protein_levels = np.maximum(0, protein_levels)

    var_rna = pd.DataFrame(index=gene_names)
    var_protein = pd.DataFrame(index=protein_names)

    from scipy import sparse

    # Convert to scipy sparse matrices to avoid copy issues
    rna_counts_sparse = sparse.csr_matrix(rna_counts)
    protein_levels_sparse = sparse.csr_matrix(protein_levels)

    rna_adata = anndata.AnnData(X=rna_counts_sparse, obs=cell_metadata, var=var_rna)

    protein_adata = anndata.AnnData(
        X=protein_levels_sparse, obs=cell_metadata, var=var_protein
    )

    mdata = mudata.MuData({"rna": rna_adata, "protein": protein_adata})

    data_package = DataPackage()
    data_package.multi_sc = {"multi_sc":mdata}

    return data_package


# Pre-generated example data for direct import
config = DefaultConfig()
EXAMPLE_PROCESSED_DATA = generate_example_data(random_seed=config.global_seed)
EXAMPLE_MULTI_BULK = generate_raw_datapackage(data_case=DataCase.MULTI_BULK)
EXAMPLE_MULTI_SC = generate_raw_datapackage(data_case=DataCase.MULTI_SINGLE_CELL)
