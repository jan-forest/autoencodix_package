import abc
import os
from typing import Optional, Union
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns  # type: ignore
import torch
import warnings

from autoencodix.utils._result import Result
from autoencodix.utils._utils import nested_dict, nested_to_tuple, show_figure
from autoencodix.configs.default_config import DefaultConfig


class BaseVisualizer(abc.ABC):
    """Defines the interface for visualizing training results.

    Attributes:
        plots: A nested dictionary to store various plots.
    """

    def __init__(self):
        self.plots = nested_dict()

    def __setitem__(self, key, elem):
        self.plots[key] = elem

    ### Abstract Methods ###
    @abc.abstractmethod
    def visualize(self, result: Result, config: DefaultConfig) -> Result:
        pass

    @abc.abstractmethod
    def show_latent_space(
        self,
        result: Result,
        plot_type: str = "2D-scatter",
        labels: Optional[Union[list, pd.Series, None]] = None,
        param: Optional[Union[list, str]] = None,
        epoch: Optional[Union[int, None]] = None,
        split: str = "all",
    ) -> None:
        pass

    @abc.abstractmethod
    def show_weights(self) -> None:
        pass

    ### General Functions used by all Visualizers in similar way ###

    def show_loss(self, plot_type: str = "absolute") -> None:
        """
        Display the loss plot.
        Args:
            plot_type: Type of loss plot to display. Options are "absolute" or "relative". Options are
                "absolute" for the absolute loss plot and
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
            if "loss_relative" not in self.plots.keys():
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

    def show_evaluation(
        self,
        param: str,
        metric: str,
        ml_alg: Optional[str] = None,
    ) -> None:
        """
        Displays the evaluation plot for a specific clinical parameter, metric, and optionally ML algorithm.
        Args:
            param: clinical parameter to visualize.
            metric: metric to visualize.
            ml_alg: ML algorithm to visualize. If None, plots all available algorithms.
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

    ## TODO move to BaseVisualizer?
    def save_plots(
        self, path: str, which: Union[str, list] = "all", format: str = "png"
    ) -> None:
        """
        Save specified plots to the given path in the specified format.

        Args:
            path: The directory path where the plots will be saved.
            which: A list of plot names to save or a string specifying which plots to save.
                   If 'all', all plots in the plots dictionary will be saved.
                   If a single plot name is provided as a string, only that plot will be saved.
            format: The file format in which to save the plots (e.g., 'png', 'jpg').

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
                        filename = which + "_" + "_".join(str(x) for x in item[0:-1]) # type: ignore
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

    ### Utilities ###

    @staticmethod
    def _make_loss_format(result: Result, config: DefaultConfig) -> pd.DataFrame:
        loss_df_melt = pd.DataFrame()
        for term in result.sub_losses.keys():
            # Get the loss values and ensure it's a dictionary
            loss_values = result.sub_losses.get(key=term).get()

            # Add explicit type checking/conversion
            if not isinstance(loss_values, dict):
                # If it's not a dict, try to convert it or handle appropriately
                if hasattr(loss_values, "to_dict"):
                    loss_values = loss_values.to_dict() # type: ignore
                else:
                    # For non-convertible types, you might need a custom solution
                    # For numpy arrays, you could do something like:
                    if hasattr(loss_values, "shape"):
                        # For numpy arrays, create a dict with indices as keys
                        loss_values = {i: val for i, val in enumerate(loss_values)}

            # Now create the DataFrame
            loss_df = pd.DataFrame.from_dict(loss_values, orient="index")  # type: ignore

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
                loss_values = loss_values.to_dict() # type: ignore
            else:
                if hasattr(loss_values, "shape"):
                    loss_values = {i: val for i, val in enumerate(loss_values)}

        loss_df = pd.DataFrame.from_dict(loss_values, orient="index")  # type: ignore
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

    @staticmethod
    def _make_loss_plot(
        df_plot: pd.DataFrame, plot_type: str
    ) -> matplotlib.figure.Figure: # type: ignore
        """
        Generates a plot for visualizing loss values from a DataFrame.

        Args:
            df_plot : DataFrame containing the loss values to be plotted. It should have the columns:
                - "Loss Term": The type of loss term (e.g., "total_loss", "reconstruction_loss").
                - "Epoch": The epoch number.
                - "Loss Value": The value of the loss.
                - "Split": The data split (e.g., "train", "validation").

            plot_type: The type of plot to generate. It can be either "absolute" or "relative".
                - "absolute": Generates a line plot for each unique loss term.
                - "relative": Generates a density plot for each data split, excluding the "total_loss" term.

        Returns:
            The generated matplotlib figure containing the loss plots.
        """
        fig_width_abs = 5 * len(df_plot["Loss Term"].unique())
        fig_width_rel = 5 * len(df_plot["Split"].unique())
        if plot_type == "absolute":
            fig, axes = plt.subplots(
                1,
                len(df_plot["Loss Term"].unique()),
                figsize=(fig_width_abs, 5),
                sharey=False,
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

    @staticmethod
    def _plot_model_weights(model: torch.nn.Module) -> matplotlib.figure.Figure: # type: ignore
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
                elif (
                    "encoder" not in name
                    and "decoder" not in name
                    and "var" not in name
                ):
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
                axes[0, i].axis("off")
            # Plot decoder weights
            for i in range(n_dec):
                ax = axes[1, i]
                heatmap_kwargs = {}

                sns.heatmap(
                    decoder_weights[i],
                    cmap=sns.color_palette("Spectral", as_cmap=True),
                    center=0,
                    ax=ax,
                    **heatmap_kwargs,
                ).set(title=decoder_names[i])
                if model.ontologies is not None:
                    axes[1, i].set_xticks(
                        ticks=range(len(node_names[i])), # type: ignore
                        labels=node_names[i], # type: ignore
                        rotation=90,
                        fontsize=8,
                    )
                    axes[1, i].set_yticks(
                        ticks=range(len(node_names[i + 1])), # type: ignore
                        labels=node_names[i + 1], # type: ignore
                        rotation=0,
                        fontsize=8,
                    )
                ax.set_xlabel("In Node", size=12)
                ax.set_ylabel("Out Node", size=12)
            # Hide unused decoder subplots
            for i in range(n_dec, n_cols):
                axes[1, i].axis("off")
        else:
            # fallback: plot all weights in order, split in half for encoder/decoder
            n_layers = len(all_weights) // 2
            fig, axes = plt.subplots(
                2, n_layers, sharex=False, figsize=(5 * n_layers, 10)
            )
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
