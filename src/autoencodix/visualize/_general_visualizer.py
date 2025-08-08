from dataclasses import field
from typing import Any, Dict, Optional, Union
import warnings

import matplotlib.figure
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore
import torch
from matplotlib import pyplot as plt
from umap import UMAP  # type: ignore

from autoencodix.base._base_visualizer import BaseVisualizer
from autoencodix.utils._result import Result
from autoencodix.utils._utils import nested_dict, show_figure
from autoencodix.utils.default_config import DefaultConfig


class GeneralVisualizer(BaseVisualizer):
    plots: Dict[str, Any] = field(
        default_factory=nested_dict
    )  ## Nested dictionary of plots as figure handles

    def __init__(self):
        self.plots = nested_dict()

    def __setitem__(self, key, elem):
        self.plots[key] = elem

    def visualize(self, result: Result, config: DefaultConfig) -> Result:
        ## Make Model Weights plot
        self.plots["ModelWeights"] = self.plot_model_weights(model=result.model)

        ## Make long format of losses
        loss_df_melt = self.make_loss_format(result=result, config=config)

        ## Make plot loss absolute
        self.plots["loss_absolute"] = self.make_loss_plot(
            df_plot=loss_df_melt, plot_type="absolute"
        )
        ## Make plot loss relative
        self.plots["loss_relative"] = self.make_loss_plot(
            df_plot=loss_df_melt, plot_type="relative"
        )

        return result

    ## Plotting methods ##

    def show_latent_space(
        self,
        result: Result,
        plot_type: str = "2D-scatter",
        labels: Optional[Union[list, pd.Series, None]] = None,
        param: Optional[Union[list, str]] = None,
        epoch: Optional[Union[int, None]] = None,
        split: str = "all",
        **kwargs,
    ) -> None:
        """
        Visualizes the latent space of the given result using different types of plots.

        Parameters:
        -----------
        result : Result
            The result object containing latent spaces and losses.
        plot_type : str, optional
            The type of plot to generate. Options are "2D-scatter", "Ridgeline", and "Coverage-Correlation". Default is "2D-scatter".
        labels : list, optional
            List of labels for the data points in the latent space. Default is None.
        param : str, optional
            List of parameters provided and stored as metadata. Strings must match column names. If not a list, string "all" is expected for convenient way to make plots for all parameters available. Default is None where no colored labels are plotted.
        epoch : int, optional
            The epoch number to visualize. If None, the last epoch is inferred from the losses. Default is None.
        split : str, optional
            The data split to visualize. Options are "train", "valid", "test", and "all". Default is "all".

        Returns:
        --------
        None
        """
        plt.ioff()
        if plot_type == "Coverage-Correlation":
            if "Coverage-Correlation" in self.plots:
                fig = self.plots["Coverage-Correlation"] 
                show_figure(fig)
                plt.show()
            else:
                results = []
                for epoch in range(result.model.config.checkpoint_interval, result.model.config.epochs + 1, result.model.config.checkpoint_interval):
                    for split in ["train", "valid"]:
                        latent_df = result.get_latent_df(epoch=epoch-1, split=split)
                        tc = self._total_correlation(latent_df)
                        cov = self._coverage_calc(latent_df)
                        results.append({"epoch": epoch, "split": split, "total_correlation": tc, "coverage": cov})

                df_metrics = pd.DataFrame(results)

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                # Total Correlation plot
                ax1 = sns.lineplot(data=df_metrics, x="epoch", y="total_correlation", hue="split", ax=axes[0])
                axes[0].set_title("Total Correlation")
                axes[0].set_xlabel("Epoch")
                axes[0].set_ylabel("Total Correlation")

                # Coverage plot
                ax2 = sns.lineplot(data=df_metrics, x="epoch", y="coverage", hue="split", ax=axes[1])
                axes[1].set_title("Coverage")
                axes[1].set_xlabel("Epoch")
                axes[1].set_ylabel("Coverage")

                plt.tight_layout()
                self.plots["Coverage-Correlation"] = fig
                show_figure(fig)
                plt.show()

        else:
            # Set Defaults
            if epoch is None:
                epoch = result.model.config.epochs - 1

          ## Getting clin_data
            if hasattr(result.datasets.train, "metadata"):
                # Check if metadata is a dictionary and contains 'paired'
                if isinstance(result.datasets.train.metadata, dict):
                    if "paired" in result.datasets.train.metadata:
                        clin_data = result.datasets.train.metadata['paired']
                        if hasattr(result.datasets, "test"):
                            clin_data = pd.concat(
                                [clin_data, result.datasets.test.metadata['paired']],
                                axis=0,
                            )
                        if hasattr(result.datasets, "valid"):
                            clin_data = pd.concat(
                                [clin_data, result.datasets.valid.metadata['paired']],
                                axis=0,
                            )
                    else:
                        # Raise error no annotation given
                        raise ValueError(
                            "Please provide paired annotation data with key 'paired' in metadata dictionary."
                        )
                elif isinstance(result.datasets.train.metadata, pd.DataFrame):
                    clin_data = result.datasets.train.metadata
                    if hasattr(result.datasets, "test"):
                        clin_data = pd.concat(
                            [clin_data, result.datasets.test.metadata],
                            axis=0,
                        )
                    if hasattr(result.datasets, "valid"):
                        clin_data = pd.concat(
                            [clin_data, result.datasets.valid.metadata],
                            axis=0,
                        )
                else:
                    # Raise error no annotation given
                    raise ValueError(
                        "Metadata is not a dictionary or DataFrame. Please provide a valid annotation data type."
                    )
            else:
                # Raise error no annotation given
                raise ValueError(
                    "No annotation data found. Please provide a valid annotation data type."
                )

            if split == "all":
                df_latent = pd.concat(
                    [
                        result.get_latent_df(epoch=epoch, split="train"),
                        result.get_latent_df(epoch=epoch, split="valid"),
                        result.get_latent_df(epoch=-1, split="test"),
                    ]
                )
            else:
                if split == "test":
                    df_latent = result.get_latent_df(epoch=-1, split=split)
                else:
                    df_latent = result.get_latent_df(epoch=epoch, split=split)

            if labels is None and param is None:
                labels = ["all"] * df_latent.shape[0]
            
            if labels is None and isinstance(param, str):
                if param == "all":
                    param = list(clin_data.columns)
                else:
                    raise ValueError(
                        "Please provide parameter to plot as a list not as string. If you want to plot all parameters, set param to 'all' and labels to None."
                    )
            
            if labels is not None and param is not None:
                raise ValueError(
                    "Please provide either labels or param, not both. If you want to plot all parameters, set param to 'all' and labels to None."
                )

            if labels is not None and param is None:
                if isinstance(labels, pd.Series):                                
                    param = [labels.name]
                    # Order by index of df_latent first, fill missing with "unknown"
                    labels = labels.reindex(df_latent.index, fill_value="unknown").tolist()
                else:                    
                    param = ["user_label"]  # Default label if none provided

            for p in param:
                if p in clin_data.columns:
                    labels = clin_data.loc[df_latent.index, p].tolist()

                if plot_type == "2D-scatter":
                    ## Make 2D Embedding with UMAP
                    if df_latent.shape[1] > 2:
                        reducer = UMAP(n_components=2)
                        embedding = pd.DataFrame(reducer.fit_transform(df_latent))
                    else:
                        embedding = df_latent

                    self.plots["2D-scatter"][epoch][split][p] = self.plot_2D(
                        embedding=embedding,
                        labels=labels,
                        param=p,
                        layer=f"2D latent space (epoch {epoch})",
                        figsize=(12, 8),
                        center=True,
                    )

                    fig = self.plots["2D-scatter"][epoch][split][p]
                    show_figure(fig)
                    plt.show()

                if plot_type == "Ridgeline":
                    ## Make ridgeline plot

                    self.plots["Ridgeline"][epoch][split][p] = self.plot_latent_ridge(
                        lat_space=df_latent, labels=labels, param=p
                    )

                    fig = self.plots["Ridgeline"][epoch][split][p].figure
                    show_figure(fig)
                    plt.show()

    def show_weights(self) -> None:
        """
        Display the model weights plot if it exists in the plots dictionary.
        Parameters:
        None
        Returns:
        None

        """

        if "ModelWeights" not in self.plots.keys():
            print("Model weights not found in the plots dictionary")
            print("You need to run visualize() method first")
        else:
            fig = self.plots["ModelWeights"]
            show_figure(fig)
            plt.show()


	## TODO move to BaseVisualizer? need to iterate then over each VAE
    @staticmethod
    def plot_model_weights(model: torch.nn.Module) -> matplotlib.figure.Figure:
        """
        Visualization of model weights in encoder and decoder layers as heatmap for each layer as subplot.
        Handles non-symmetrical autoencoder architectures.
        Plots _mu layer for encoder as well.
        Uses node_names for decoder layers if model has ontologies.
        ARGS:
            model (torch.nn.Module): PyTorch model instance.
        RETURNS:
            fig (matplotlib.figure): Figure handle (of last plot)
        """
        all_weights = []
        names = []
        node_names = None
        if hasattr(model, "ontologies"):
            if model.ontologies is not None:
                node_names = []
                for ontology in model.ontologies:
                    node_names.append(list(ontology.keys()))
                node_names.append(model.feature_order)

        # Collect encoder and decoder weights separately
        encoder_weights = []
        encoder_names = []
        decoder_weights = []
        decoder_names = []
        for name, param in model.named_parameters():
            # print(name)
            if "weight" in name and len(param.shape) == 2:
                if "encoder" in name and "var" not in name and "_mu" not in name:
                    encoder_weights.append(param.detach().cpu().numpy())
                    encoder_names.append(name[:-7])
                elif "_mu" in name:
                    encoder_weights.append(param.detach().cpu().numpy())
                    encoder_names.append(name[:-7])
                elif "decoder" in name and "var" not in name:
                    decoder_weights.append(param.detach().cpu().numpy())
                    decoder_names.append(name[:-7])
                elif "encoder" not in name and "decoder" not in name and "var" not in name:
                    # fallback for models without explicit encoder/decoder in name
                    all_weights.append(param.detach().cpu().numpy())
                    names.append(name[:-7])

        if encoder_weights or decoder_weights:
            n_enc = len(encoder_weights)
            n_dec = len(decoder_weights)
            n_cols = max(n_enc, n_dec)
            fig, axes = plt.subplots(2, n_cols, sharex=False, figsize=(15 * n_cols, 15))
            if n_cols == 1:
                axes = axes.reshape(2, 1)
            # Plot encoder weights
            for i in range(n_enc):
                ax = axes[0, i]
                sns.heatmap(
                    encoder_weights[i],
                    cmap=sns.color_palette("Spectral", as_cmap=True),
                    center=0,
                    ax=ax,
                ).set(title=encoder_names[i])
                ax.set_ylabel("Out Node", size=12)
            # Hide unused encoder subplots
            for i in range(n_enc, n_cols):
                axes[0, i].axis('off')
            # Plot decoder weights
            for i in range(n_dec):
                ax = axes[1, i]
                heatmap_kwargs = {}

                sns.heatmap(
                    decoder_weights[i],
                    cmap=sns.color_palette("Spectral", as_cmap=True),
                    center=0,
                    ax=ax,
                    **heatmap_kwargs
                ).set(title=decoder_names[i])
                if model.ontologies is not None:
                    axes[1, i].set_xticks(
                        ticks=range(len(node_names[i])),
                        labels=node_names[i],
                        rotation=90,
                        fontsize=8,
                    )
                    axes[1, i].set_yticks(
                        ticks=range(len(node_names[i + 1])),
                        labels=node_names[i + 1],
                        rotation=0,
                        fontsize=8,
                    )
                ax.set_xlabel("In Node", size=12)
                ax.set_ylabel("Out Node", size=12)
            # Hide unused decoder subplots
            for i in range(n_dec, n_cols):
                axes[1, i].axis('off')
        else:
            # fallback: plot all weights in order, split in half for encoder/decoder
            n_layers = len(all_weights) // 2
            fig, axes = plt.subplots(2, n_layers, sharex=False, figsize=(5 * n_layers, 10))
            for layer in range(n_layers):
                sns.heatmap(
                    all_weights[layer],
                    cmap=sns.color_palette("Spectral", as_cmap=True),
                    center=0,
                    ax=axes[0, layer],
                ).set(title=names[layer])
                sns.heatmap(
                    all_weights[n_layers + layer],
                    cmap=sns.color_palette("Spectral", as_cmap=True),
                    center=0,
                    ax=axes[1, layer],
                ).set(title=names[n_layers + layer])
                axes[1, layer].set_xlabel("In Node", size=12)
                axes[0, layer].set_ylabel("Out Node", size=12)
                axes[1, layer].set_ylabel("Out Node", size=12)

        fig.suptitle("Model Weights", size=20)
        plt.close()
        return fig

    @staticmethod
    def plot_2D(
        embedding: pd.DataFrame,
        labels: list,
        param: Optional[Union[str, None]] = None,
        layer: str = "latent space",
        figsize: tuple = (24, 15),
        center: bool = True,
        plot_numeric: bool = False,
        xlim: Optional[Union[tuple, None]] = None,
        ylim: Optional[Union[tuple, None]] = None,
        scale: Optional[Union[str, None]] = None,
        no_leg: bool = False,
    ) -> matplotlib.figure.Figure:
        """
        Plots a 2D scatter plot of the given embedding with labels.

        Parameters:
        embedding (pd.DataFrame): DataFrame containing the 2D embedding coordinates.
        labels (list): List of labels corresponding to each point in the embedding.
        param (str, optional): Title for the legend. Defaults to None.
        layer (str, optional): Title for the plot. Defaults to "latent space".
        figsize (tuple, optional): Size of the figure. Defaults to (24, 15).
        center (bool, optional): If True, centers the plot based on label means. Defaults to True.
        plot_numeric (bool, optional): If True, treats labels as numeric. Defaults to False.
        xlim (tuple, optional): Limits for the x-axis. Defaults to None.
        ylim (tuple, optional): Limits for the y-axis. Defaults to None.
        scale (str, optional): Scale for the axes (e.g., 'log'). Defaults to None.
        no_leg (bool, optional): If True, no legend is displayed. Defaults to False.

        Returns:
        plt.Figure: The resulting matplotlib figure.
        """

        numeric = False
        if not isinstance(labels[0], str):
            if len(np.unique(labels)) > 3:
                if not plot_numeric:
                    print(
                        "The provided label column is numeric and converted to categories."
                    )
                    labels = pd.qcut(
                        x=pd.Series(labels),
                        q=4,
                        labels=["1stQ", "2ndQ", "3rdQ", "4thQ"],
                    ).astype(str).to_list()
                else:
                    center = False  ## Disable centering for numeric params
                    numeric = True
            else:
                labels = [str(x) for x in labels]

        fig, ax1 = plt.subplots(figsize=figsize)

        # check if label or embedding is longerm and duplicate the shorter one
        if len(labels) < embedding.shape[0]:
            print(
                "Given labels do not have the same length as given sample size. Labels will be duplicated."
            )
            labels = [
                label
                for label in labels
                for _ in range(embedding.shape[0] // len(labels))
            ]
        elif len(labels) > embedding.shape[0]:
            labels = list(set(labels))

        if numeric:
            ax2 = sns.scatterplot(
                x=embedding.iloc[:, 0],
                y=embedding.iloc[:, 1],
                hue=labels,
                palette="bwr",
                s=40,
                alpha=0.5,
                ec="black",
            )
        else:
            ax2 = sns.scatterplot(
                x=embedding.iloc[:, 0],
                y=embedding.iloc[:, 1],
                hue=labels,
                hue_order=np.unique(labels),
                s=40,
                alpha=0.5,
                ec="black",
            )
        if center:
            means = embedding.groupby(by=labels).mean()

            ax2 = sns.scatterplot(
                x=means.iloc[:, 0],
                y=means.iloc[:, 1],
                hue=np.unique(labels),
                hue_order=np.unique(labels),
                s=200,
                ec="black",
                alpha=0.9,
                marker="*",
                legend=False,
                ax=ax2,
            )

        if xlim is not None:
            ax2.set_xlim(xlim[0], xlim[1])

        if ylim is not None:
            ax2.set_ylim(ylim[0], ylim[1])

        if scale is not None:
            plt.yscale(scale)
            plt.xscale(scale)
        ax2.set_xlabel("Dim 1")
        ax2.set_ylabel("Dim 2")
        legend_cols = 1
        if len(np.unique(labels)) > 10:
            legend_cols = 2

        if no_leg:
            plt.legend([], [], frameon=False)
        else:
            sns.move_legend(
                ax2,
                "upper left",
                bbox_to_anchor=(1, 1),
                ncol=legend_cols,
                title=param,
                frameon=False,
            )

        # Add title to the plot
        ax2.set_title(layer)

        plt.close()
        return fig

    ## TODO Might be moved to BaseVisualizer if Ridgeline per Modality is used as in notebook
    @staticmethod
    def plot_latent_ridge(
        lat_space: pd.DataFrame,
        labels: Optional[Union[list, pd.Series, None]] = None,
        param: Optional[Union[str, None]] = None,
    ) -> sns.FacetGrid:
        """
        Creates a ridge line plot of latent space dimension where each row shows the density of a latent dimension and groups (ridges).
        ARGS:
            lat_space (pd.DataFrame): DataFrame containing the latent space intensities for samples (rows) and latent dimensions (columns)
            labels (list): List of labels for each sample. If None, all samples are considered as one group.
            param (str): Clinical parameter to create groupings and coloring of ridges. Must be a column name (str) of clin_data
        RETURNS:
            g (sns.FacetGrid): FacetGrid object containing the ridge line plot
        """
        sns.set_theme(
            style="white", rc={"axes.facecolor": (0, 0, 0, 0)}
        )  ## Necessary to enforce overplotting

        df = pd.melt(lat_space, var_name="latent dim", value_name="latent intensity")
        df["sample"] = len(lat_space.columns) * list(lat_space.index)

        if labels is None:
            param = "all"
            labels = ["all"] * len(df)

        # print(labels[0])
        if not isinstance(labels[0], str):
            if len(np.unique(labels)) > 3:
                labels = pd.qcut(
                    x=pd.Series(labels),
                    q=4,
                    labels=["1stQ", "2ndQ", "3rdQ", "4thQ"],
                ).astype(str)
            else:
                labels = [str(x) for x in labels]

        df[param] = len(lat_space.columns) * labels  # type: ignore

        exclude_missing_info = (df[param] == "unknown") | (df[param] == "nan")

        xmin = (
            df.loc[~exclude_missing_info, ["latent intensity", "latent dim", param]]
            .groupby([param, "latent dim"], observed=False)
            .quantile(0.05)
            .min()
        )
        xmax = (
            df.loc[~exclude_missing_info, ["latent intensity", "latent dim", param]]
            .groupby([param, "latent dim"], observed=False)
            .quantile(0.9)
            .max()
        )

        if len(np.unique(df[param])) > 8:
            cat_pal = sns.husl_palette(len(np.unique(df[param])))
        else:
            cat_pal = sns.color_palette(n_colors=len(np.unique(df[param])))

        g = sns.FacetGrid(
            df[~exclude_missing_info],
            row="latent dim",
            hue=param,
            aspect=12,
            height=0.8,
            xlim=(xmin.iloc[0], xmax.iloc[0]),
            palette=cat_pal,
        )

        g.map_dataframe(
            sns.kdeplot,
            "latent intensity",
            bw_adjust=0.5,
            clip_on=True,
            fill=True,
            alpha=0.5,
            warn_singular=False,
            ec="k",
            lw=1,
        )

        def label(data, color, label, text="latent dim"):
            ax = plt.gca()
            label_text = data[text].unique()[0]
            ax.text(
                0.0,
                0.2,
                label_text,
                fontweight="bold",
                ha="right",
                va="center",
                transform=ax.transAxes,
            )

        g.map_dataframe(label, text="latent dim")

        g.set(xlim=(xmin.iloc[0], xmax.iloc[0]))
        # Set the subplots to overlap
        g.figure.subplots_adjust(hspace=-0.5)

        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)

        g.add_legend()

        plt.close()
        return g

    @staticmethod
    def make_loss_plot(
        df_plot: pd.DataFrame, plot_type: str
    ) -> matplotlib.figure.Figure:
        """
        Generates a plot for visualizing loss values from a DataFrame.

        Parameters:
        -----------
        df_plot : pd.DataFrame
            DataFrame containing the loss values to be plotted. It should have the columns:
            - "Loss Term": The type of loss term (e.g., "total_loss", "reconstruction_loss").
            - "Epoch": The epoch number.
            - "Loss Value": The value of the loss.
            - "Split": The data split (e.g., "train", "validation").

        plot_type : str
            The type of plot to generate. It can be either "absolute" or "relative".
            - "absolute": Generates a line plot for each unique loss term.
            - "relative": Generates a density plot for each data split, excluding the "total_loss" term.

        Returns:
        --------
        matplotlib.figure.Figure
            The generated matplotlib figure containing the loss plots.
        """
        fig_width_abs = 5*len(df_plot["Loss Term"].unique())
        fig_width_rel = 5*len(df_plot["Split"].unique())
        if plot_type == "absolute":
            fig, axes = plt.subplots(
                1, len(df_plot["Loss Term"].unique()), figsize=(fig_width_abs, 5), sharey=False
            )
            ax = 0
            for term in df_plot["Loss Term"].unique():
                axes[ax] = sns.lineplot(
                    data=df_plot[(df_plot["Loss Term"] == term)],
                    x="Epoch",
                    y="Loss Value",
                    hue="Split",
                    ax=axes[ax],
                ).set_title(term)
                ax += 1

            plt.close()

        if plot_type == "relative":
            # Check if loss values are positive
            if (df_plot["Loss Value"] < 0).any():
                # Warning
                warnings.warn(
                    "Loss values contain negative values. Check your loss function if correct. Loss will be clipped to zero for plotting."
                )
                df_plot["Loss Value"] = df_plot["Loss Value"].clip(lower=0)

            # Exclude loss terms where all Loss Value are zero or NaN over all epochs
            valid_terms = [
                term
                for term in df_plot["Loss Term"].unique()
                if (
                    (df_plot[df_plot["Loss Term"] == term]["Loss Value"].notna().any())
                    and (df_plot[df_plot["Loss Term"] == term]["Loss Value"] != 0).any()
                )
            ]
            exclude = (
                (df_plot["Loss Term"] != "total_loss")
                & ~(df_plot["Loss Term"].str.contains("_factor"))
                & (df_plot["Loss Term"].isin(valid_terms))
            )

            fig, axes = plt.subplots(1, 2, figsize=(fig_width_rel, 5), sharey=True)

            ax = 0

            for split in df_plot["Split"].unique():
                axes[ax] = sns.kdeplot(
                    data=df_plot[exclude & (df_plot["Split"] == split)],
                    x="Epoch",
                    hue="Loss Term",
                    multiple="fill",
                    weights="Loss Value",
                    clip=[0, df_plot["Epoch"].max()],
                    ax=axes[ax],
                ).set_title(split)
                ax += 1

            plt.close()

        return fig

	   
    def plot_evaluation(
            self,
            result: Result,
    ) -> dict:
        """
        Plots the evaluation results from the Result object.

        Parameters:
        result (Result): The Result object containing evaluation data.

        Returns:
        dict: The generated dictionary containing the evaluation plots.
        """
        ## Plot all results

        ml_plots = dict()
        plt.ioff()

        for c in pd.unique(result.embedding_evaluation.CLINIC_PARAM):
            ml_plots[c] = dict()
            for m in pd.unique(result.embedding_evaluation.loc[result.embedding_evaluation.CLINIC_PARAM == c, "metric"]):
                ml_plots[c][m] = dict()
                for alg in pd.unique(result.embedding_evaluation.loc[
                        (result.embedding_evaluation.CLINIC_PARAM == c) &
                        (result.embedding_evaluation.metric == m), "ML_ALG"
                    ]):
                    data = result.embedding_evaluation[
                        (result.embedding_evaluation.metric == m) &
                        (result.embedding_evaluation.CLINIC_PARAM == c) &
                        (result.embedding_evaluation.ML_ALG == alg)
                    ]

                    sns_plot = sns.catplot(
                        data=data,
                        x="score_split",
                        y="value",
                        col="ML_TASK",
                        hue="score_split",
                        kind="bar",
                    )

                    min_y = data.value.min()
                    if min_y > 0:
                        min_y = 0

                    ml_plots[c][m][alg] = sns_plot.set(ylim=(min_y, None))

        self.plots["ML_Evaluation"] = ml_plots

        return ml_plots
    
    def show_evaluation(
        self,
        param: str,
        metric: str,
        ml_alg: Optional[str] = None,
    ) -> None:
        """
        Displays the evaluation plot for a specific clinical parameter, metric, and optionally ML algorithm.
        Parameters:
        param (str): The clinical parameter to visualize.
        metric (str): The metric to visualize.
        ml_alg (str, optional): The ML algorithm to visualize. If None, plots all available algorithms.
        Returns:
        None
        """
        plt.ioff()
        if "ML_Evaluation" not in self.plots.keys():
            print("ML Evaluation plots not found in the plots dictionary")
            print("You need to run evaluate() method first")
            return None
        if param not in self.plots["ML_Evaluation"].keys():
            print(f"Parameter {param} not found in the ML Evaluation plots")
            print(f"Available parameters: {list(self.plots['ML_Evaluation'].keys())}")
            return None
        if metric not in self.plots["ML_Evaluation"][param].keys():
            print(f"Metric {metric} not found in the ML Evaluation plots for {param}")
            print(
                f"Available metrics: {list(self.plots['ML_Evaluation'][param].keys())}"
            )
            return None

        algs = list(self.plots["ML_Evaluation"][param][metric].keys())
        if ml_alg is not None:
            if ml_alg not in algs:
                print(f"ML algorithm {ml_alg} not found for {param} and {metric}")
                print(f"Available ML algorithms: {algs}")
                return None
            fig = self.plots["ML_Evaluation"][param][metric][ml_alg].figure
            show_figure(fig)
            plt.show()
        else:
            for alg in algs:
                print(f"Showing plot for ML algorithm: {alg}")
                fig = self.plots["ML_Evaluation"][param][metric][alg].figure
                show_figure(fig)
                plt.show()
    
    @staticmethod
    def _total_correlation(latent_space: pd.DataFrame) -> float:
        """ Function to compute the total correlation as described here (Equation2): https://doi.org/10.3390/e21100921
            
        Args:
            latent_space - (pd.DataFrame): latent space with dimension sample vs. latent dimensions
        Returns:
            tc - (float): total correlation across latent dimensions
        """
        lat_cov = np.cov(latent_space.T)
        tc = 0.5 * (np.sum(np.log(np.diag(lat_cov))) - np.linalg.slogdet(lat_cov)[1])
        return tc
    
    @staticmethod
    def _coverage_calc(latent_space: pd.DataFrame) -> float:
        """ Function to compute the coverage as described here (Equation3): https://doi.org/10.3390/e21100921
            
        Args:
            latent_space - (pd.DataFrame): latent space with dimension sample vs. latent dimensions
        Returns:
            cov - (float): coverage across latent dimensions
        """
        bins_per_dim = int(
            np.power(len(latent_space.index), 1 / len(latent_space.columns))
        )
        if bins_per_dim < 2:
            warnings.warn(
                "Coverage calculation fails since combination of sample size and latent dimension results in less than 2 bins."
            )
            cov = np.nan
        else:
            latent_bins = latent_space.apply(lambda x: pd.cut(x, bins=bins_per_dim))
            latent_bins = pd.Series(zip(*[latent_bins[col] for col in latent_bins]))
            cov = len(latent_bins.unique()) / np.power(
                bins_per_dim, len(latent_space.columns)
            )
            
        return cov
