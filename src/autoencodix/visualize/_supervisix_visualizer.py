from dataclasses import field
from typing import Any, Dict, Optional, Union, Literal, no_type_check
import warnings

import matplotlib.figure
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore
from matplotlib import pyplot as plt
from umap import UMAP  # type: ignore
import seaborn as sns  # type: ignore
import seaborn.objects as so

from autoencodix.visualize._general_visualizer import GeneralVisualizer
from autoencodix.utils._result import Result
from autoencodix.utils._utils import nested_dict, show_figure
from autoencodix.configs.default_config import DefaultConfig


class SupervisixVisualizer(GeneralVisualizer):
    """
    Defines the interface for visualizing training results for the Supervisix architecture.

    Attributes: 
        plots: A nested dictionary to store various plots.
    """

    plots: Dict[str, Any] = field(
        default_factory=nested_dict
    )  ## Nested dictionary of plots as figure handles

    def __init__(self):
        self.plots = nested_dict()

    def __setitem__(self, key, elem):
        self.plots[key] = elem   

    @staticmethod
    def _make_loss_plot(
        df_plot: pd.DataFrame, plot_type: str
    ) -> matplotlib.figure.Figure:  # type: ignore
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
            # Get absolute values of class separation loss
            mask = df_plot["Loss Term"] == "class_separation_loss"
            df_plot.loc[mask, "Loss Value"] = df_plot.loc[mask, "Loss Value"].abs()

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

            df_plot.loc[exclude, "Relative Loss Value"] = (
                df_plot[exclude]
                .groupby(["Split", "Epoch"])["Loss Value"]
                .transform(lambda x: x / x.sum())
            )
            fig = (
                (
                    so.Plot(
                        df_plot[exclude],
                        "Epoch",
                        "Relative Loss Value",
                        color="Loss Term",
                    ).add(so.Area(alpha=0.7), so.Stack())
                )
                .facet("Split")
                .layout(size=(fig_width_rel, 5))
            )

            # fig, axes = plt.subplots(1, 2, figsize=(fig_width_rel, 5), sharey=True)

            # ax = 0

            # for split in df_plot["Split"].unique():
            #     axes[ax] = sns.kdeplot(
            #         data=df_plot[exclude & (df_plot["Split"] == split)],
            #         x="Epoch",
            #         hue="Loss Term",
            #         multiple="fill",
            #         weights="Loss Value",
            #         clip=[0, df_plot["Epoch"].max()],
            #         ax=axes[ax],
            #     ).set_title(split)
            #     ax += 1

            # plt.close()

        return fig  
     