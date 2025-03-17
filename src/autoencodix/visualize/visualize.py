import os
from dataclasses import field
from typing import Any, Dict, Optional, Union

import matplotlib.figure
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore
import torch
from matplotlib import pyplot as plt
from umap import UMAP  # type: ignore

from autoencodix.base._base_visualizer import BaseVisualizer
from autoencodix.utils._result import Result
from autoencodix.utils._utils import nested_dict, nested_to_tuple, show_figure
from autoencodix.utils.default_config import DefaultConfig


class Visualizer(BaseVisualizer):
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

    def save_plots(
        self, path: str, which: Union[str, list] = "all", format: str = "png"
    ) -> None:
        """
        Save specified plots to the given path in the specified format.

        Parameters:
        path (str): The directory path where the plots will be saved.
        which (list or str): A list of plot names to save or a string specifying which plots to save.
                             If 'all', all plots in the plots dictionary will be saved.
                             If a single plot name is provided as a string, only that plot will be saved.
        format (str): The file format in which to save the plots (e.g., 'png', 'jpg').

        Returns:
        None

        Raises:
        ValueError: If the 'which' parameter is not a list or a string.
        """
        if not isinstance(which, list):
            ## Case when which is a string
            if which == "all":
                ## Case when all plots are to be saved
                if len(self.plots) == 0:
                    print("No plots found in the plots dictionary")
                    print("You need to run  visualize() method first")
                else:
                    for item in nested_to_tuple(self.plots):
                        fig = item[-1]  ## Figure is in last element of the tuple
                        filename = "_".join(str(x) for x in item[0:-1])
                        fullpath = os.path.join(path, filename)
                        fig.savefig(f"{fullpath}.{format}")
            else:
                ## Case when a single plot is provided as string
                if which not in self.plots.keys():
                    print(f"Plot {which} not found in the plots dictionary")
                    print(f"All available plots are: {list(self.plots.keys())}")
                else:
                    for item in nested_to_tuple(
                        self.plots[which]
                    ):  # Plot all epochs and splits of type which
                        fig = item[-1]  ## Figure is in last element of the tuple
                        filename = which + "_" + "_".join(str(x) for x in item[0:-1])
                        fullpath = os.path.join(path, filename)
                        fig.savefig(f"{fullpath}.{format}")
        else:
            ## Case when which is a list of plot specified as strings
            for key in which:
                if key not in self.plots.keys():
                    print(f"Plot {key} not found in the plots dictionary")
                    print(f"All available plots are: {list(self.plots.keys())}")
                    continue
                else:
                    for item in nested_to_tuple(
                        self.plots[key]
                    ):  # Plot all epochs and splits of type key
                        fig = item[-1]  ## Figure is in last element of the tuple
                        filename = key + "_" + "_".join(str(x) for x in item[0:-1])
                        fullpath = os.path.join(path, filename)
                        fig.savefig(f"{fullpath}.{format}")

    def show_loss(self, plot_type: str = "absolute") -> None:
        """
        Display the loss plot.
        Parameters:
        plot_type (str): The type of loss plot to display.
                    Options are "absolute" for the absolute loss plot and
                    "relative" for the relative loss plot.
                    Defaults to "absolute".
        Returns:
        None
        """
        if plot_type == "absolute":
            if "loss_absolute" not in self.plots.keys():
                print("Absolute loss plot not found in the plots dictionary")
                print("You need to run visualize() method first")
            else:
                fig = self.plots["loss_absolute"]
                show_figure(fig)
                plt.show()
        if plot_type == "relative":
            if "relative_absolute" not in self.plots.keys():
                print("Relative loss plot not found in the plots dictionary")
                print("You need to run visualize() method first")
            else:
                fig = self.plots["loss_relative"]
                show_figure(fig)
                plt.show()

        if plot_type not in ["absolute", "relative"]:
            print(
                "Type of loss plot not recognized. Please use 'absolute' or 'relative'"
            )

    def show_latent_space(
        self,
        result: Result,
        plot_type: str = "2D-scatter",
        label_list: Optional[Union[list, None]] = None,
        param: str = "all",
        epoch: Optional[Union[int, None]] = None,
        split: str = "all",
    ) -> None:
        """
        Visualizes the latent space of the given result using different types of plots.

        Parameters:
        -----------
        result : Result
            The result object containing latent spaces and losses.
        plot_type : str, optional
            The type of plot to generate. Options are "2D-scatter", "Ridgeline", and "Coverage-Correlation". Default is "2D-scatter".
        label_list : list, optional
            List of labels for the data points in the latent space. Default is None.
        param : str, optional
            Parameter to be used for plotting. Default is "all".
        epoch : int, optional
            The epoch number to visualize. If None, the last epoch is inferred from the losses. Default is None.
        split : str, optional
            The data split to visualize. Options are "train", "valid", "test", and "all". Default is "all".

        Returns:
        --------
        None
        """
        if plot_type == "2D-scatter":
            # Set Defaults
            if epoch is None:
                # Infer total epochs from losses
                epoch = len(result.losses.get()) - 1

            if split == "all":
                df_latent = pd.DataFrame(
                    np.concatenate(
                        [
                            result.latentspaces.get(epoch=epoch, split="train"),
                            result.latentspaces.get(epoch=epoch, split="valid"),
                            result.latentspaces.get(epoch=-1, split="test"),
                        ]
                    )
                )
            else:
                if split == "test":
                    df_latent = pd.DataFrame(
                        result.latentspaces.get(epoch=-1, split=split)
                    )
                else:
                    df_latent = pd.DataFrame(
                        result.latentspaces.get(epoch=epoch, split=split)
                    )

            if label_list is None:
                label_list = ["all"] * df_latent.shape[0]

            ## Make 2D Embedding with UMAP
            if df_latent.shape[1] > 2:
                reducer = UMAP(n_components=2)
                embedding = pd.DataFrame(reducer.fit_transform(df_latent))
            else:
                embedding = df_latent

            if not self.plots["2D-scatter"][epoch][split][
                param
            ]:  ## Create when not exist
                self.plots["2D-scatter"][epoch][split][param] = self.plot_2D(
                    embedding=embedding,
                    labels=label_list,
                    param=param,
                    layer=f"2D latent space (epoch {epoch})",
                    figsize=(12, 8),
                    center=True,
                )

            fig = self.plots["2D-scatter"][epoch][split][param]
            show_figure(fig)
            plt.show()

        if plot_type == "Ridgeline":
            if epoch is None:
                # Infer total epochs from losses
                epoch = len(result.losses.get()) - 1

            if split == "all":
                df_latent = pd.DataFrame(
                    np.concatenate(
                        [
                            result.latentspaces.get(epoch=epoch, split="train"),
                            result.latentspaces.get(epoch=epoch, split="valid"),
                            result.latentspaces.get(epoch=-1, split="test"),
                        ]
                    )
                )
            else:
                if split == "test":
                    df_latent = pd.DataFrame(
                        result.latentspaces.get(epoch=-1, split=split)
                    )
                else:
                    df_latent = pd.DataFrame(
                        result.latentspaces.get(epoch=epoch, split=split)
                    )

            if label_list is None:
                label_list = ["all"] * df_latent.shape[0]

            ## Make ridgeline plot
            if not self.plots["Ridgeline"][epoch][split][
                param
            ]:  ## Create when not exist
                self.plots["Ridgeline"][epoch][split][param] = self.plot_latent_ridge(
                    lat_space=df_latent, label_list=label_list, param=param
                )

            fig = self.plots["Ridgeline"][epoch][split][param].figure
            show_figure(fig)
            plt.show()

        if plot_type == "Coverage-Correlation":
            ## TODO
            print("Not implemented yet, empty figure will be shown instead")
            fig = plt.figure()
            self.plots["Coverage-Correlation"] = fig
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

    @staticmethod
    def plot_model_weights(model: torch.nn.Module) -> matplotlib.figure.Figure:
        """
        Visualization of model weights in encoder and decoder layers as heatmap for each layer as subplot.
        ARGS:
            model (torch.nn.Module): PyTorch model instance.
            filepath (str): Path specifying save name and location.
        RETURNS:
            fig (matplotlib.figure): Figure handle (of last plot)
        """
        all_weights = []
        names = []
        for name, param in model.named_parameters():
            if "weight" in name and len(param.shape) == 2:
                if "var" not in name:  ## For VAE plot only mu weights
                    all_weights.append(param.detach().cpu().numpy())
                    names.append(name[:-7])

        layers = int(len(all_weights) / 2)
        fig, axes = plt.subplots(2, layers, sharex=False, figsize=(20, 10))

        for layer in range(layers):
            ## Encoder Layer
            if layers > 1:
                sns.heatmap(
                    all_weights[layer],
                    cmap=sns.color_palette("Spectral", as_cmap=True),
                    ax=axes[0, layer],
                ).set(title=names[layer])
                ## Decoder Layer
                sns.heatmap(
                    all_weights[layers + layer],
                    cmap=sns.color_palette("Spectral", as_cmap=True),
                    ax=axes[1, layer],
                ).set(title=names[layers + layer])
                axes[1, layer].set_xlabel("In Node", size=12)
            else:
                sns.heatmap(
                    all_weights[layer],
                    cmap=sns.color_palette("Spectral", as_cmap=True),
                    ax=axes[layer],
                ).set(title=names[layer])
                ## Decoder Layer
                sns.heatmap(
                    all_weights[layer + 2],
                    cmap=sns.color_palette("Spectral", as_cmap=True),
                    ax=axes[layer + 1],
                ).set(title=names[layer + 2])
                axes[1].set_xlabel("In Node", size=12)

        if layers > 1:
            axes[1, 0].set_ylabel("Out Node", size=12)
            axes[0, 0].set_ylabel("Out Node", size=12)
        else:
            axes[1].set_ylabel("Out Node", size=12)
            axes[0].set_ylabel("Out Node", size=12)

        ## Add title
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
                    ).astype(str)
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

    @staticmethod
    def plot_latent_ridge(
        lat_space: pd.DataFrame,
        label_list: Optional[Union[list, None]] = None,
        param: Optional[Union[str, None]] = None,
    ) -> sns.FacetGrid:
        """
        Creates a ridge line plot of latent space dimension where each row shows the density of a latent dimension and groups (ridges).
        ARGS:
            lat_space (pd.DataFrame): DataFrame containing the latent space intensities for samples (rows) and latent dimensions (columns)
            label_list (list): List of labels for each sample. If None, all samples are considered as one group.
            param (str): Clinical parameter to create groupings and coloring of ridges. Must be a column name (str) of clin_data
        RETURNS:
            g (sns.FacetGrid): FacetGrid object containing the ridge line plot
        """
        sns.set_theme(
            style="white", rc={"axes.facecolor": (0, 0, 0, 0)}
        )  ## Necessary to enforce overplotting

        df = pd.melt(lat_space, var_name="latent dim", value_name="latent intensity")
        df["sample"] = len(lat_space.columns) * list(lat_space.index)

        if label_list is None:
            param = "all"
            label_list = ["all"] * len(df)

        # print(labels[0])
        if not isinstance(label_list[0], str):
            if len(np.unique(label_list)) > 3:
                label_list = pd.qcut(
                    x=pd.Series(label_list),
                    q=4,
                    labels=["1stQ", "2ndQ", "3rdQ", "4thQ"],
                ).astype(str)
            else:
                label_list = [str(x) for x in label_list]

        df[param] = len(lat_space.columns) * label_list  # type: ignore

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

    def make_loss_plot(
        self, df_plot: pd.DataFrame, plot_type: str
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
        if plot_type == "absolute":
            fig, axes = plt.subplots(
                1, len(df_plot["Loss Term"].unique()), figsize=(15, 5), sharey=False
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
            exclude = df_plot["Loss Term"] != "total_loss"

            fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

            ax = 0

            for split in df_plot["Split"].unique():
                axes[ax] = sns.kdeplot(
                    data=df_plot[exclude & (df_plot["Split"] == split)],
                    x="Epoch",
                    hue="Loss Term",
                    multiple="fill",
                    weights="Loss Value",
                    clip=[0, 30],
                    ax=axes[ax],
                ).set_title(split)
                ax += 1

            plt.close()

        return fig

    def make_loss_format(self, result: Result, config: DefaultConfig) -> pd.DataFrame:
        loss_df_melt = pd.DataFrame()
        for term in result.sub_losses.keys():
            # Get the loss values and ensure it's a dictionary
            loss_values = result.sub_losses.get(key=term).get()

            # Add explicit type checking/conversion
            if not isinstance(loss_values, dict):
                # If it's not a dict, try to convert it or handle appropriately
                if hasattr(loss_values, "to_dict"):
                    loss_values = loss_values.to_dict()
                else:
                    # For non-convertible types, you might need a custom solution
                    # For numpy arrays, you could do something like:
                    if hasattr(loss_values, "shape"):
                        # For numpy arrays, create a dict with indices as keys
                        loss_values = {i: val for i, val in enumerate(loss_values)}

            # Now create the DataFrame
            loss_df = pd.DataFrame.from_dict(loss_values, orient="index") # type: ignore

            # Rest of your code remains the same
            if term == "var_loss":
                loss_df = loss_df * config.beta
            loss_df["Epoch"] = loss_df.index + 1
            loss_df["Loss Term"] = term

            loss_df_melt = pd.concat(
                [
                    loss_df_melt,
                    loss_df.melt(
                        id_vars=["Epoch", "Loss Term"],
                        var_name="Split",
                        value_name="Loss Value",
                    ),
                ],
                axis=0,
            ).reset_index(drop=True)

        # Similar handling for the total losses
        loss_values = result.losses.get()
        if not isinstance(loss_values, dict):
            if hasattr(loss_values, "to_dict"):
                loss_values = loss_values.to_dict()
            else:
                if hasattr(loss_values, "shape"):
                    loss_values = {i: val for i, val in enumerate(loss_values)}

        loss_df = pd.DataFrame.from_dict(loss_values, orient="index") # type: ignore
        loss_df["Epoch"] = loss_df.index + 1
        loss_df["Loss Term"] = "total_loss"

        loss_df_melt = pd.concat(
            [
                loss_df_melt,
                loss_df.melt(
                    id_vars=["Epoch", "Loss Term"],
                    var_name="Split",
                    value_name="Loss Value",
                ),
            ],
            axis=0,
        ).reset_index(drop=True)

        loss_df_melt["Loss Value"] = loss_df_melt["Loss Value"].astype(float)
        return loss_df_melt
