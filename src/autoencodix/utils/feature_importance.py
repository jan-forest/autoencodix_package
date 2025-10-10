import os
import warnings
from typing import Dict

import anndata as ad
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import seaborn as sns
import torch
import torch.nn as nn
from captum.attr import (
    LRP,
    DeepLiftShap,
    GradientShap,
    IntegratedGradients,
    Lime,
    LimeBase,
)
from IPython.display import HTML, Image, clear_output, display

from autoencodix.utils.adata_converter import AnnDataConverter

warnings.filterwarnings("ignore")


class Vanillix_EncoderSingleDim(nn.Module):
    def __init__(self, vae_model, dim):
        super(Vanillix_EncoderSingleDim, self).__init__()
        # Accessing the required components from the original VAE model
        self.encoder = vae_model._encoder
        self.input_dim = vae_model.input_dim
        self.dim = dim  # latent dim

    def forward(self, x):
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input with {self.input_dim} features, but got {x.shape[1]} features. "
                f"This may indicate missing data modalities or incorrect data preparation."
            )

        total_elements = x.numel()
        assert (
            total_elements % self.input_dim == 0
        ), f"Total elements {total_elements} is not a multiple of input_dim {self.input_dim}"

        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        latent = self.encoder(x)
        output = latent[:, self.dim]
        output = output.unsqueeze(1)  # Equivalent to output.reshape(output.shape[0], 1)
        return output


def make_DeepLiftShap_Vanillix_dim(model, inputs, baselines, latent_dimension):
    model_encoder_dim = Vanillix_EncoderSingleDim(model, dim=latent_dimension)
    deeplift = DeepLiftShap(model_encoder_dim)
    attributions, convergence = deeplift.attribute(
        inputs=inputs, baselines=baselines, return_convergence_delta=True
    )
    avg_abs_attributions = attributions.abs().mean(dim=0)
    return avg_abs_attributions, convergence


def make_feature_importance_Vanillix(
    input_adata,
    van,
    method="DeepLiftShap",
    n_subset=100,
    seed_int=12,
    baseline_type="mean",
    baseline_group="all",
    obs_col=None,
):
    """
    Computes attributions for a trained Vanillix model.

    Parameters
    ----------
    van : object
        Object containing the trained Vanillix model and input data as an AnnData object.

    method : str, {'DeepLiftShap', 'IntegratedGradients'}, default='DeepLiftShap'
        post-hoc feature importance assessment method
        - 'DeepLiftShap': method from pytorch Captum library, approximates SHAP values using Deeplift
        - 'IntegratedGradients': method from pytorch Captum library, attribution via Integrated gradients

    n_subset : int, default=100
        Subset of randomly selected cells to compute attributions on.

    seed_int : int, default=12
        Seed for reproducible random sampling (NumPy & PyTorch).

    baseline_type : str, {'mean', 'random_sample'}, default='mean'
        How to generate the baseline:
        - 'mean': average expression of the group
        - 'random_sample': a randomly picked cell from the group

    baseline_group : str, default='all'
        Which group of cells to use as the baseline.
        Use 'all' or a specific group label from `adata.obs[obs_col]`.

    obs_col : str or None, default=None
        Column in `adata.obs` that defines groups. Required if `baseline_group` is not 'all'.

    Returns
    -------
    df_attributions : pandas.DataFrame
        Gene-level attribution scores per latent dimension.
    """
    if baseline_group != "all" and obs_col is None:
        raise ValueError(
            "obs_col must be provided when using a group-specific baseline."
        )
    np.random.seed(seed_int)
    torch.manual_seed(seed_int)
    model = van.result.model
    # input_adata = van.raw_user_data["multi_sc"]["multi_sc"].mod["user-data"]
    inputs = torch.tensor(
        input_adata.X.toarray()
        if scipy.sparse.issparse(input_adata.X)
        else input_adata.X
    )
    if baseline_group == "all":
        if baseline_type == "mean":
            baseline_mean = inputs.mean(axis=0)  # gene_means
            baselines = torch.tensor(np.tile(baseline_mean, (inputs.shape[0], 1)))
        if baseline_type == "random_sample":
            baseline_random = inputs[torch.randint(0, inputs.size(0), (1,)).item()]
            baselines = torch.tensor(np.tile(baseline_random, (inputs.shape[0], 1)))
    else:
        input_adata_filtered = input_adata[input_adata.obs[obs_col] == baseline_group]
        inputs_filtered = torch.tensor(
            input_adata_filtered.X.toarray()
            if scipy.sparse.issparse(input_adata_filtered.X)
            else input_adata_filtered.X
        )
        if baseline_type == "mean":
            baseline_mean = inputs_filtered.mean(axis=0)  # gene_means
            baselines = torch.tensor(np.tile(baseline_mean, (inputs.shape[0], 1)))
        if baseline_type == "random_sample":
            baseline_random = inputs_filtered[
                torch.randint(0, inputs_filtered.size(0), (1,)).item()
            ]
            baselines = torch.tensor(np.tile(baseline_random, (inputs.shape[0], 1)))
    gene_names = input_adata.var_names
    cell_IDs = input_adata.obs_names

    latent_dimensions = list(range(0, van.result.adata_latent.shape[1]))
    indices_DeepLiftShap = np.random.choice(
        inputs.shape[0], size=n_subset, replace=False
    )
    all_attr = []
    for latent_dim in latent_dimensions:
        if method == "DeepLiftShap":
            avg_abs_attributions, convergence = make_DeepLiftShap_Vanillix_dim(
                model=model,
                inputs=inputs[indices_DeepLiftShap].float(),
                baselines=baselines[indices_DeepLiftShap].float(),
                latent_dimension=latent_dim,
            )
        if method == "IntegratedGradients":
            avg_abs_attributions, convergence = make_IntegratedGradients_Vanillix_dim(
                model=model,
                inputs=inputs[indices_DeepLiftShap].float(),
                baselines=baselines[indices_DeepLiftShap].float(),
                latent_dimension=latent_dim,
            )

        all_attr.append(avg_abs_attributions.detach().cpu())

    attr_matrix = torch.stack(all_attr).T.numpy()

    df_attributions = pd.DataFrame(
        attr_matrix,
        index=list(gene_names),
        columns=[f"latent_dimension_{i}" for i in latent_dimensions],
    )
    return df_attributions


def get_top_kgenes_per_latent_dimension(df_attributions, latent_dim=0, topk=10):
    """
    Returns the top-k genes with the highest attribution for a given latent dimension.

    Parameters
    ----------
    df_attributions : pandas.DataFrame
        DataFrame with genes as rows and latent dimensions as columns

    latent_dim : int, default=0
        Index of the latent dimension to extract top genes from

    topk : int, default=10
        Number of top genes to return.

    Returns
    -------
    topk_genes : list of str
        List of gene names with the highest attributions for the selected latent dimension.
    """
    topk_genes = (
        df_attributions["latent_dimension_" + str(latent_dim)]
        .nlargest(topk)
        .index.tolist()
    )
    return topk_genes


def plot_union_top_genes_heatmap(df, top_n=50, cmap="viridis", save=None):
    """
    Plots a heatmap for all latent dimensions, showing the union of the top N genes per dimension.

    Parameters:
    - df (pd.DataFrame): DataFrame with feature attributions.
                         Rows = genes/features, Columns = latent dimensions.
    - top_n (int): Number of top genes to select per latent dimension (default: 50).
    - cmap (str): Colormap for the heatmap (default: 'viridis').
    - save (str) : path for saving
    """

    # Collect top N genes per latent dimension
    top_genes_sets = []
    for col in df.columns:
        top_genes = df[col].nlargest(top_n).index
        top_genes_sets.append(set(top_genes))

    # Union of all top genes across dimensions
    union_genes = sorted(set.union(*top_genes_sets))

    # Subset the dataframe to only these genes
    data = df.loc[union_genes]

    # Plot heatmap
    plt.figure(figsize=(len(df.columns) * 0.1 + 3, max(6, 0.14 * len(union_genes))))
    ax = sns.heatmap(
        data,
        cmap=cmap,
        annot=False,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Attribution Score"},
    )

    # Format ticks (centered)
    # ax.set_xticks([i + 0.5 for i in range(len(data.columns))])
    ax.set_xticklabels(data.columns, rotation=90, ha="right", fontsize=9)

    ax.set_yticks([i + 0.5 for i in range(len(data.index))])
    ax.set_yticklabels(data.index, rotation=0, fontsize=8)

    plt.title(f"attribution of top {top_n} genes per latent dimension", fontsize=10)

    plt.tight_layout()
    if save is not None:
        plt.savefig(save, bbox_inches="tight")
    plt.show()


def get_top_genes_per_dimension(df, top_n=100):
    """
    Get top N genes per latent dimension.
    Returns a dict: {dimension_name: [list of top genes]}
    """
    top_genes = {}
    for dim in df.columns:
        top_genes[dim] = df[dim].nlargest(top_n).index.tolist()
    return top_genes


def run_go_enrichment(
    top_genes_dict,
    n_top_pathways=10,
    gene_set_library="GO_Biological_Process_2021",
    organism="Human",
):
    """
    Run GO enrichment using Enrichr (via gseapy).
    Returns a dict of DataFrames with results per latent dimension.
    """
    results = {}
    for dim, gene_list in top_genes_dict.items():
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_set_library,
            organism=organism,
            outdir=None,  # no file output
            cutoff=0.05,
        )
        results[dim] = enr.results.sort_values("Adjusted P-value").head(n_top_pathways)
    return results


def plot_GO_log_odds_all(PE_dict, top_n=10, base_save_identifier=None):
    """
    Plots log odds ratio of top enriched GO terms for each latent dimension.

    Parameters:
    - PE_dict (dict): Dict of DataFrames, keyed by latent dimension name.
    - top_n (int): Number of top GO terms to plot per dimension.
    """
    for dim, df in PE_dict.items():
        if df.empty or "Odds Ratio" not in df.columns:
            print(f"Skipping {dim}: empty or missing required columns.")
            continue

        # Compute log odds ratio
        df["log_odds_ratio"] = np.log(df["Odds Ratio"])

        # Select top_n terms
        plot_df = df.sort_values(by="log_odds_ratio", ascending=True).tail(top_n)

        # Plot
        plt.figure(figsize=(10, 0.5 * top_n))  # wider and taller
        bars = plt.barh(
            plot_df["Term"],
            plot_df["log_odds_ratio"],
            color="#D4AF3A",
            height=0.6,  # makes bars thicker
        )

        # Add value labels
        for bar in bars:
            plt.text(
                bar.get_width() - 1.05,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.2f}",
                va="center",
                ha="left",
                fontsize=8,
                color="#C40308",
            )

        plt.xlabel("log(odds Ratio)", fontsize=10)
        plt.title(f"{dim} — top {top_n} Enriched GO terms", fontsize=12)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=9)  # smaller tick labels
        plt.tight_layout()
        if base_save_identifier is not None:
            plt.savefig(
                base_save_identifier + "_" + str(dim) + ".pdf", bbox_inches="tight"
            )
        plt.show()


def do_miraculix_visualization():
    display(HTML('<p style="font-size:20px;">Calculate feature importance</p>'))
    display(Image(filename="miraculix_zaubertrank.gif"))
    clear_output(wait=True)


def do_feature_importance_Vanillix(
    van,
    method="DeepLiftShap",
    baseline_type="mean",
    baseline_group="all",
    obs_col=None,
    n_subset=100,
    seed_int=12,
    do_visualizations=True,
    top_n_genes_heatmap=50,
    top_n_foreground_pathways=30,
    gene_set_library=None,
    organism="Human",
    n_top_pathways=10,
    save_out_path=None,
    do_miraculix_vis=False,
):
    """
    Computes DeepLiftShap attributions for a trained Vanillix model and visulaization.

    Parameters
    ----------
    van : object
        Object containing the trained Vanillix model and input data as an AnnData object.

    method : str, {'DeepLiftShap', 'IntegratedGradients'}, default='DeepLiftShap'
        post-hoc feature importance assessment method
        - 'DeepLiftShap': method from pytorch Captum library, approximates SHAP values using Deeplift
        - 'IntegratedGradients': method from pytorch Captum library, attribution via Integrated gradients

    n_subset : int, default=100
        Subset of randomly selected cells to compute attributions on.

    seed_int : int, default=12
        Seed for reproducible random sampling (NumPy & PyTorch).

    baseline_type : str, {'mean', 'random_sample'}, default='mean'
        How to generate the baseline:
        - 'mean': average expression of the group
        - 'random_sample': a randomly picked cell from the group

    baseline_group : str, default='all'
        Which group of cells to use as the baseline.
        Use 'all' or a specific group label from `adata.obs[obs_col]`.

    obs_col : str or None, default=None
        Column in `adata.obs` that defines groups. Required if `baseline_group` is not 'all'.

    top_n_genes_heatmap : int, default=50
        Number of top genes (with highest attribution scores) per latent dimension to visualize in the heatmap.

    top_n_foreground_pathways : int, default=30
        Number of top genes to include as "foreground" in enrichment analysis.

    gene_set_library : str, default=None,
        The gene set library to use for enrichment analysis (from Enrichr), e.g. "GO_Biological_Process_2021".

    organism : str, default='Human'
        The organism relevant to the gene sets used in enrichment.

    n_top_pathways : int, default=10
        Number of top enriched pathways to visualize in the results per latent dimension.

    save_out_path : str or None, default=None
        File path to save the output results  figures.
        If None, results will not be saved to disk.

    do_miraculix_vis : bool, default=True
        Show Miraculix-style GIF.

    Returns
    -------
    df_attributions : pandas.DataFrame
        Gene-level attribution scores per latent dimension.
    """
    feature_importance_methods = {"DeepLiftShap", "IntegratedGradients"}
    if method not in feature_importance_methods:
        raise ValueError(
            f"Invalid method '{method}'. Must be one of: {', '.join(feature_importance_methods)}."
        )

    # baseline_type
    if do_miraculix_vis:
        do_miraculix_visualization()

    input_data: Dict[str, ad.Anndata] = AnnDataConverter.dataset_to_adata(
        datasetcontainer=van.result.datasets
    )
    for data_name, adata in input_data.items():
        df_attributions = make_feature_importance_Vanillix(
            input_adata=adata,
            van=van,
            method=method,
            n_subset=n_subset,
            seed_int=seed_int,
            baseline_type=baseline_type,
            baseline_group=baseline_group,
            obs_col=obs_col,
        )
    if save_out_path is not None:
        save_path = os.path.join(save_out_path, "df_attributions.csv")
        df_attributions.to_csv(save_path)

    """
    if not do_visualizations:
        display(HTML('<p style="font-size:20px;">Feature importance completed!</p>'))
    """

    # top genes plot
    if do_visualizations:
        if save_out_path is not None:
            save_path = os.path.join(save_out_path, "top_attributions.pdf")
            plot_union_top_genes_heatmap(
                df=df_attributions,
                top_n=top_n_genes_heatmap,
                cmap="plasma",
                save=save_path,
            )
        else:
            plot_union_top_genes_heatmap(
                df=df_attributions, top_n=top_n_genes_heatmap, cmap="plasma", save=None
            )

        dict_top_genes = get_top_genes_per_dimension(
            df=df_attributions, top_n=top_n_foreground_pathways
        )
        PE_dict = run_go_enrichment(
            top_genes_dict=dict_top_genes,
            n_top_pathways=n_top_pathways,
            gene_set_library=gene_set_library,
            organism=organism,
        )

        if save_out_path is not None:
            base_save_identifier = os.path.join(save_out_path, "GO_pathways")
            plot_GO_log_odds_all(
                PE_dict=PE_dict,
                top_n=n_top_pathways,
                base_save_identifier=base_save_identifier,
            )
        else:
            plot_GO_log_odds_all(
                PE_dict=PE_dict, top_n=n_top_pathways, base_save_identifier=None
            )
    return df_attributions


class Varix_EncoderSingleDim(nn.Module):
    def __init__(self, vae_model, dim):
        super(Varix_EncoderSingleDim, self).__init__()
        # Accessing the required components from the original VAE model
        self.encoder = vae_model._encoder
        self.mu = vae_model._mu
        self.logvar = vae_model._logvar
        self.reparameterize = vae_model.reparameterize
        self.input_dim = vae_model.input_dim
        self.dim = dim  # latent dim

    def forward(self, x):
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input with {self.input_dim} features, but got {x.shape[1]} features. "
                f"This may indicate missing data modalities or incorrect data preparation."
            )

        total_elements = x.numel()
        assert (
            total_elements % self.input_dim == 0
        ), f"Total elements {total_elements} is not a multiple of input_dim {self.input_dim}"

        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        latent = self.encoder(x)
        mu = self.mu(latent)
        logvar = self.logvar(latent)
        z = self.reparameterize(mu, logvar)
        output = z[:, self.dim]
        output = output.unsqueeze(1)  # Equivalent to output.reshape(output.shape[0], 1)
        return output


def make_DeepLiftShap_Varix_dim(model, inputs, baselines, latent_dimension):
    model_encoder_dim = Varix_EncoderSingleDim(model, dim=latent_dimension)
    deeplift = DeepLiftShap(model_encoder_dim)
    attributions, convergence = deeplift.attribute(
        inputs=inputs, baselines=baselines, return_convergence_delta=True
    )
    avg_abs_attributions = attributions.abs().mean(dim=0)
    return avg_abs_attributions, convergence


def make_feature_importance_Varix(
    van,
    method="DeepLiftShap",
    n_subset=100,
    seed_int=12,
    baseline_type="mean",
    baseline_group="all",
    obs_col=None,
):
    """
    Computes DeepLiftShap attributions for a trained Varix model .

    Parameters
    ----------
    van : object
        Object containing the trained Varix model and input data as an AnnData object.

    method : str, {'DeepLiftShap', 'IntegratedGradients'}, default='DeepLiftShap'
        post-hoc feature importance assessment method
        - 'DeepLiftShap': method from pytorch Captum library, approximates SHAP values using Deeplift
        - 'IntegratedGradients': method from pytorch Captum library, attribution via Integrated gradients

    n_subset : int, default=100
        Subset of randomly selected cells to compute attributions on.

    seed_int : int, default=12
        Seed for reproducible random sampling (NumPy & PyTorch).

    baseline_type : str, {'mean', 'random_sample'}, default='mean'
        How to generate the baseline:
        - 'mean': average expression of the group
        - 'random_sample': a randomly picked cell from the group

    baseline_group : str, default='all'
        Which group of cells to use as the baseline.
        Use 'all' or a specific group label from `adata.obs[obs_col]`.

    obs_col : str or None, default=None
        Column in `adata.obs` that defines groups. Required if `baseline_group` is not 'all'.

    Returns
    -------
    df_attributions : pandas.DataFrame
        Gene-level attribution scores per latent dimension.
    """
    if baseline_group != "all" and obs_col is None:
        raise ValueError(
            "obs_col must be provided when using a group-specific baseline."
        )
    np.random.seed(seed_int)
    torch.manual_seed(seed_int)
    model = van.result.model

    input_adata = van.raw_user_data["multi_sc"]["multi_sc"].mod["user-data"]
    inputs = torch.tensor(
        input_adata.X.toarray()
        if scipy.sparse.issparse(input_adata.X)
        else input_adata.X
    )
    if baseline_group == "all":
        if baseline_type == "mean":
            baseline_mean = inputs.mean(axis=0)  # gene_means
            baselines = torch.tensor(np.tile(baseline_mean, (inputs.shape[0], 1)))
        if baseline_type == "random_sample":
            baseline_random = inputs[torch.randint(0, inputs.size(0), (1,)).item()]
            baselines = torch.tensor(np.tile(baseline_random, (inputs.shape[0], 1)))
    else:
        input_adata_filtered = input_adata[input_adata.obs[obs_col] == baseline_group]
        inputs_filtered = torch.tensor(
            input_adata_filtered.X.toarray()
            if scipy.sparse.issparse(input_adata_filtered.X)
            else input_adata_filtered.X
        )
        if baseline_type == "mean":
            baseline_mean = inputs_filtered.mean(axis=0)  # gene_means
            baselines = torch.tensor(np.tile(baseline_mean, (inputs.shape[0], 1)))
        if baseline_type == "random_sample":
            baseline_random = inputs_filtered[
                torch.randint(0, inputs_filtered.size(0), (1,)).item()
            ]
            baselines = torch.tensor(np.tile(baseline_random, (inputs.shape[0], 1)))
    gene_names = input_adata.var_names
    cell_IDs = input_adata.obs_names

    latent_dimensions = list(range(0, van.result.adata_latent.shape[1]))
    indices_DeepLiftShap = np.random.choice(
        inputs.shape[0], size=n_subset, replace=False
    )
    all_attr = []
    for latent_dim in latent_dimensions:
        if method == "DeepLiftShap":
            avg_abs_attributions, convergence = make_DeepLiftShap_Varix_dim(
                model=model,
                inputs=inputs[indices_DeepLiftShap].float(),
                baselines=baselines[indices_DeepLiftShap].float(),
                latent_dimension=latent_dim,
            )
        if method == "IntegratedGradients":
            avg_abs_attributions, convergence = make_IntegratedGradients_Varix_dim(
                model=model,
                inputs=inputs[indices_DeepLiftShap].float(),
                baselines=baselines[indices_DeepLiftShap].float(),
                latent_dimension=latent_dim,
            )

        all_attr.append(avg_abs_attributions.detach().cpu())
    attr_matrix = torch.stack(all_attr).T.numpy()

    df_attributions = pd.DataFrame(
        attr_matrix,
        index=list(gene_names),
        columns=[f"latent_dimension_{i}" for i in latent_dimensions],
    )
    return df_attributions


def do_feature_importance_Varix(
    van,
    method="DeepLiftShap",
    baseline_type="mean",
    baseline_group="all",
    obs_col=None,
    n_subset=100,
    seed_int=12,
    do_visualizations=True,
    top_n_genes_heatmap=50,
    top_n_foreground_pathways=30,
    gene_set_library=None,
    organism="Human",
    n_top_pathways=10,
    save_out_path=None,
    do_miraculix_vis=False,
):
    """
    Computes DeepLiftShap attributions for a trained Varix model and visulaization.

    Parameters
    ----------
    van : object
        Object containing the trained Varix model and input data as an AnnData object.

    method : str, {'DeepLiftShap', 'IntegratedGradients'}, default='DeepLiftShap'
        post-hoc feature importance assessment method
        - 'DeepLiftShap': method from pytorch Captum library, approximates SHAP values using Deeplift
        - 'IntegratedGradients': method from pytorch Captum library, attribution via Integrated gradients

    n_subset : int, default=100
        Subset of randomly selected cells to compute attributions on.

    seed_int : int, default=12
        Seed for reproducible random sampling (NumPy & PyTorch).

    baseline_type : str, {'mean', 'random_sample'}, default='mean'
        How to generate the baseline:
        - 'mean': average expression of the group
        - 'random_sample': a randomly picked cell from the group

    baseline_group : str, default='all'
        Which group of cells to use as the baseline.
        Use 'all' or a specific group label from `adata.obs[obs_col]`.

    obs_col : str or None, default=None
        Column in `adata.obs` that defines groups. Required if `baseline_group` is not 'all'.

    top_n_genes_heatmap : int, default=50
        Number of top genes (with highest attribution scores) per latent dimension to visualize in the heatmap.

    top_n_foreground_pathways : int, default=30
        Number of top genes to include as "foreground" in enrichment analysis.

    gene_set_library : str, default=None,
        The gene set library to use for enrichment analysis (from Enrichr), e.g. "GO_Biological_Process_2021".

    organism : str, default='Human'
        The organism relevant to the gene sets used in enrichment.

    n_top_pathways : int, default=10
        Number of top enriched pathways to visualize in the results per latent dimension.

    save_out_path : str or None, default=None
        File path to save the output results  figures.
        If None, results will not be saved to disk.

    do_miraculix_vis : bool, default=True
        Show Miraculix-style GIF.

    Returns
    -------
    df_attributions : pandas.DataFrame
        Gene-level attribution scores per latent dimension.

    """
    feature_importance_methods = {"DeepLiftShap", "IntegratedGradients"}
    if method not in feature_importance_methods:
        raise ValueError(
            f"Invalid method '{method}'. Must be one of: {', '.join(feature_importance_methods)}."
        )

    # baseline_type
    if do_miraculix_vis:
        do_miraculix_visualization()
    df_attributions = make_feature_importance_Varix(
        van,
        method=method,
        n_subset=n_subset,
        seed_int=seed_int,
        baseline_type=baseline_type,
        baseline_group=baseline_group,
        obs_col=obs_col,
    )
    if save_out_path is not None:
        save_path = os.path.join(save_out_path, "df_attributions.csv")
        df_attributions.to_csv(save_path)

    """
    if not do_visualizations:
        display(HTML('<p style="font-size:20px;">Feature importance completed!</p>'))
    """

    # top genes plot
    if do_visualizations:
        if save_out_path is not None:
            save_path = os.path.join(save_out_path, "top_attributions.pdf")
            plot_union_top_genes_heatmap(
                df=df_attributions,
                top_n=top_n_genes_heatmap,
                cmap="plasma",
                save=save_path,
            )
        else:
            plot_union_top_genes_heatmap(
                df=df_attributions, top_n=top_n_genes_heatmap, cmap="plasma", save=None
            )

        if gene_set_library is not None:
            dict_top_genes = get_top_genes_per_dimension(
                df=df_attributions, top_n=top_n_foreground_pathways
            )
            PE_dict = run_go_enrichment(
                top_genes_dict=dict_top_genes,
                n_top_pathways=n_top_pathways,
                gene_set_library=gene_set_library,
                organism=organism,
            )

            if save_out_path is not None:
                base_save_identifier = os.path.join(save_out_path, "GO_pathways")
                plot_GO_log_odds_all(
                    PE_dict=PE_dict,
                    top_n=n_top_pathways,
                    base_save_identifier=base_save_identifier,
                )
            else:
                plot_GO_log_odds_all(
                    PE_dict=PE_dict, top_n=n_top_pathways, base_save_identifier=None
                )
    return df_attributions


def make_IntegratedGradients_Varix_dim(model, inputs, baselines, latent_dimension):
    model_encoder_dim = Varix_EncoderSingleDim(model, dim=latent_dimension)
    integrated_gradients = IntegratedGradients(model_encoder_dim)
    attributions, convergence = integrated_gradients.attribute(
        inputs=inputs, baselines=baselines, return_convergence_delta=True
    )
    avg_abs_attributions = attributions.abs().mean(dim=0)
    return avg_abs_attributions, convergence


def make_IntegratedGradients_Vanillix_dim(model, inputs, baselines, latent_dimension):
    model_encoder_dim = Vanillix_EncoderSingleDim(model, dim=latent_dimension)
    integrated_gradients = IntegratedGradients(model_encoder_dim)
    attributions, convergence = integrated_gradients.attribute(
        inputs=inputs, baselines=baselines, return_convergence_delta=True
    )
    avg_abs_attributions = attributions.abs().mean(dim=0)
    return avg_abs_attributions, convergence
