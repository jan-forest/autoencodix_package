from dataclasses import field
from typing import Any, Dict, Optional, Union, Literal, no_type_check
import warnings

import matplotlib.figure
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore
from matplotlib import pyplot as plt
from umap import UMAP  # type: ignore

from autoencodix.base._base_visualizer import BaseVisualizer
from autoencodix.utils._result import Result
from autoencodix.utils._utils import nested_dict, show_figure
from autoencodix.configs.default_config import DefaultConfig


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
        if result.model.input_dim <= 3000:
            self.plots["ModelWeights"] = self._plot_model_weights(model=result.model)
        else:
            warnings.warn(
                f"Model weights plot is skipped since input dimension {result.model.input_dim} is larger than 3000 and heatmap would be too large."
            )

        ## Make long format of losses
        try:
            loss_df_melt = self._make_loss_format(result=result, config=config)

            ## Make plot loss absolute
            self.plots["loss_absolute"] = self._make_loss_plot(
                df_plot=loss_df_melt, plot_type="absolute"
            )
            ## Make plot loss relative
            self.plots["loss_relative"] = self._make_loss_plot(
                df_plot=loss_df_melt, plot_type="relative"
            )
        except Exception as e:
            warnings.warn(
                f"We could not create visualizations for the loss plots.\n"
                f"This usually happens if you try to visualize after saving and loading "
                f"the pipeline object with `save_all=False`. This memory-efficient saving mode "
                f"does not retain past training loss data.\n\n"
                f"Original error message: {e}"
            )

        return result

    ## Plotting methods ##
    @no_type_check
    def show_latent_space(
        self,
        result: Result,
        plot_type: Literal[
            "2D-scatter", "Ridgeline", "Coverage-Correlation"
        ] = "2D-scatter",
        labels: Optional[Union[list, pd.Series, None]] = None,
        param: Optional[Union[list, str]] = None,
        epoch: Optional[Union[int, None]] = None,
        split: str = "all",
        n_downsample: Optional[int] = 10000,
        **kwargs,
    ) -> None:
        """Visualizes the latent space of the given result using different types of plots.

        Args:
            result: The result object containing latent spaces and losses.
            plot_type: The type of plot to generate. Options are "2D-scatter", "Ridgeline", and "Coverage-Correlation". Default is "2D-scatter".
            labels: List of labels for the data points in the latent space. Default is None.
            param: List of parameters provided and stored as metadata. Strings must match column names. If not a list, string "all" is expected for convenient way to make plots for all parameters available. Default is None where no colored labels are plotted.
            epoch: The epoch number to visualize. If None, the last epoch is inferred from the losses. Default is None.
            split: The data split to visualize. Options are "train", "valid", "test", and "all". Default is "all".
            n_downsample: If provided, downsample the data to this number of samples for faster visualization. Default is 10000. Set to None to disable downsampling.
            **kwargs: additional arguments.

        """
        plt.ioff()
        if plot_type == "Coverage-Correlation":
            if "Coverage-Correlation" in self.plots:
                fig = self.plots["Coverage-Correlation"]
                show_figure(fig)
                plt.show()
            else:
                results = []
                for epoch in range(
                    result.model.config.checkpoint_interval,
                    result.model.config.epochs + 1,
                    result.model.config.checkpoint_interval,
                ):
                    for split in ["train", "valid"]:
                        latent_df = result.get_latent_df(epoch=epoch - 1, split=split)
                        tc = self._total_correlation(latent_df)
                        cov = self._coverage_calc(latent_df)
                        results.append(
                            {
                                "epoch": epoch,
                                "split": split,
                                "total_correlation": tc,
                                "coverage": cov,
                            }
                        )

                df_metrics = pd.DataFrame(results)

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                # Total Correlation plot
                _ = sns.lineplot(
                    data=df_metrics,
                    x="epoch",
                    y="total_correlation",
                    hue="split",
                    ax=axes[0],
                )
                axes[0].set_title("Total Correlation")
                axes[0].set_xlabel("Epoch")
                axes[0].set_ylabel("Total Correlation")

                # Coverage plot
                _ = sns.lineplot(
                    data=df_metrics, x="epoch", y="coverage", hue="split", ax=axes[1]
                )
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

            # ## Getting clin_data
            clin_data = self._collect_all_metadata(result=result)
            # if hasattr(result.datasets.train, "metadata"):
            #     # Check if metadata is a dictionary and contains 'paired'
            #     if isinstance(result.datasets.train.metadata, dict):
            #         if "paired" in result.datasets.train.metadata:
            #             clin_data = result.datasets.train.metadata["paired"]
            #             if hasattr(result.datasets, "test"):
            #                 clin_data = pd.concat(
            #                     [
            #                         clin_data,
            #                         result.datasets.test.metadata[  # ty: ignore
            #                             "paired"
            #                         ],  # ty: ignore
            #                     ],  # ty: ignore
            #                     axis=0,
            #                 )
            #             if hasattr(result.datasets, "valid"):
            #                 clin_data = pd.concat(
            #                     [
            #                         clin_data,
            #                         result.datasets.valid.metadata[  # ty: ignore
            #                             "paired"
            #                         ],  # ty: ignore
            #                     ],  # ty: ignore
            #                     axis=0,
            #                 )
            #         else:
            #             # Iterate over all splits and keys, concatenate if DataFrame
            #             clin_data = pd.DataFrame()
            #             for split_name in ["train", "test", "valid"]:
            #                 split_temp = getattr(result.datasets, split_name, None)
            #                 if split_temp is not None and hasattr(
            #                     split_temp, "metadata"
            #                 ):
            #                     for key in split_temp.metadata.keys():
            #                         if isinstance(
            #                             split_temp.metadata[key], pd.DataFrame
            #                         ):
            #                             clin_data = pd.concat(
            #                                 [
            #                                     clin_data,
            #                                     split_temp.metadata[key],
            #                                 ],
            #                                 axis=0,
            #                             )
            #             # remove duplicate rows
            #             clin_data = clin_data[~clin_data.index.duplicated(keep="first")]
            #             # if clin_data.empty:
            #             #     # Raise error no annotation given
            #             #     raise ValueError(
            #             #         "Please provide paired annotation data with key 'paired' in metadata dictionary."
            #             #     )
            #     elif isinstance(result.datasets.train.metadata, pd.DataFrame):
            #         clin_data = result.datasets.train.metadata
            #         if hasattr(result.datasets, "test"):
            #             clin_data = pd.concat(
            #                 [clin_data, result.datasets.test.metadata],  # ty: ignore
            #                 axis=0,
            #             )
            #         if hasattr(result.datasets, "valid"):
            #             clin_data = pd.concat(
            #                 [clin_data, result.datasets.valid.metadata],  # ty: ignore
            #                 axis=0,
            #             )
            #     else:
            #         # Raise error no annotation given
            #         raise ValueError(
            #             "Metadata is not a dictionary or DataFrame. Please provide a valid annotation data type."
            #         )
            # else:
            #     # Iterate over all splits and keys, concatenate if DataFrame
            #     clin_data = pd.DataFrame()
            #     for split_name in ["train", "test", "valid"]:
            #         split_temp = getattr(result.datasets, split_name, None)
            #         if split_temp is not None:
            #             for key in split_temp.datasets.keys():
            #                 if isinstance(
            #                     split_temp.datasets[key].metadata, pd.DataFrame
            #                 ):
            #                     clin_data = pd.concat(
            #                         [
            #                             clin_data,
            #                             split_temp.datasets[key].metadata,
            #                         ],
            #                         axis=0,
            #                     )
            #     if len(clin_data) == 0: ## New predict case
            #         for split_name in ["train", "test", "valid"]:
            #             split_temp = getattr(result.new_datasets, split_name, None)
            #             if split_temp is not None:
            #                 if len(split_temp.datasets.keys()) > 0:
            #                     for key in split_temp.datasets.keys():
            #                         if isinstance(
            #                             split_temp.datasets[key].metadata, pd.DataFrame
            #                         ):
            #                             clin_data = pd.concat(
            #                                 [
            #                                     clin_data,
            #                                     split_temp.datasets[key].metadata,
            #                                 ],
            #                                 axis=0,
            #                             )
            #                 else:
            #                     if isinstance(
            #                         split_temp.metadata, pd.DataFrame
            #                     ):
            #                         clin_data = pd.concat(
            #                             [
            #                                 clin_data,
            #                                 split_temp.metadata,
            #                             ],
            #                             axis=0,
            #                         )
            #     # remove duplicate rows
            #     clin_data = clin_data[~clin_data.index.duplicated(keep="first")]

            # # Raise error no annotation given
            # raise ValueError(
            #     "No annotation data found. Please provide a valid annotation data type."
            # )

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

            ## Label options
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
                    labels = labels.reindex(
                        df_latent.index, fill_value="unknown"
                    ).tolist()
                else:
                    param = ["user_label"]  # Default label if none provided
            if not isinstance(param, list):
                raise TypeError("Param needs to be converted to a list")
            for p in param:
                if p in clin_data.columns:
                    labels = clin_data.loc[df_latent.index, p].tolist()  # ty: ignore

                if n_downsample is not None:
                    if df_latent.shape[0] > n_downsample:
                        sample_idx = np.random.choice(
                            df_latent.shape[0], n_downsample, replace=False
                        )
                        df_latent = df_latent.iloc[sample_idx]
                        if labels is not None:
                            labels = [labels[i] for i in sample_idx]

                if plot_type == "2D-scatter":
                    ## Make 2D Embedding with UMAP
                    if df_latent.shape[1] > 2:
                        reducer = UMAP(n_components=2)
                        embedding = pd.DataFrame(reducer.fit_transform(df_latent))
                    else:
                        embedding = df_latent

                    self.plots["2D-scatter"][epoch][split][p] = self._plot_2D(
                        embedding=embedding,
                        labels=labels,
                        param=p,
                        layer=f"2D latent space (epoch {epoch+1})",  # we start counting epochs at 0, so add 1 for display
                        figsize=(12, 8),
                        center=True,
                    )

                    fig = self.plots["2D-scatter"][epoch][split][p]
                    show_figure(fig)
                    plt.show()

                if plot_type == "Ridgeline":
                    ## Make ridgeline plot

                    self.plots["Ridgeline"][epoch][split][p] = self._plot_latent_ridge(
                        lat_space=df_latent, labels=labels, param=p
                    )

                    fig = self.plots["Ridgeline"][epoch][split][p].figure
                    show_figure(fig)
                    plt.show()

                if plot_type == "Clustermap":
                    ## Make clustermap plot

                    self.plots["Clustermap"][epoch][split][p] = (
                        self._plot_latent_clustermap(
                            lat_space=df_latent, labels=labels, param=p
                        )
                    )

                    fig = self.plots["Clustermap"][epoch][split][p]
                    show_figure(fig)
                    plt.show()

    def show_weights(self) -> None:
        """Display the model weights plot if it exists in the plots dictionary."""

        if "ModelWeights" not in self.plots.keys():
            print("Model weights not found in the plots dictionary")
            print("You need to run visualize() method first")
        else:
            fig = self.plots["ModelWeights"]
            show_figure(fig)
            plt.show()

    ### Moved to Base
    # def show_evaluation(
    #     self,
    #     param: str,
    #     metric: str,
    #     ml_alg: Optional[str] = None,
    # ) -> None:

    ### Utilities ###
    @staticmethod
    def _plot_2D(
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
        """Plots a 2D scatter plot of the given embedding with labels.

        Args:
            embedding: DataFrame containing the 2D embedding coordinates.
            labels: List of labels corresponding to each point in the embedding.
            param: Title for the legend. Defaults to None.
            layer: Title for the plot. Defaults to "latent space".
            figsize: Size of the figure. Defaults to (24, 15).
            center: If True, centers the plot based on label means. Defaults to True.
            plot_numeric: If True, treats labels as numeric. Defaults to False.
            xlim: Limits for the x-axis. Defaults to None.
            ylim: Limits for the y-axis. Defaults to None.
            scale:: Scale for the axes (e.g., 'log'). Defaults to None.
            no_leg: If True, no legend is displayed. Defaults to False.

        Returns:
            The resulting matplotlib figure.
        """

        numeric = False
        if not isinstance(labels[0], str):
            if len(np.unique(labels)) > 3:
                if not plot_numeric:
                    print(
                        "The provided label column is numeric and converted to categories."
                    )
                    labels = [
                        float("nan") if not isinstance(x, float) else x for x in labels
                    ]
                    labels = (
                        pd.qcut(
                            x=pd.Series(labels),
                            q=4,
                            labels=["1stQ", "2ndQ", "3rdQ", "4thQ"],
                        )
                        .astype(str)
                        .to_list()
                    )
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
            if len(np.unique(labels)) > 8:
                cat_pal = sns.color_palette("tab20", n_colors=len(np.unique(labels)))
            else:
                cat_pal = sns.color_palette("tab10", n_colors=len(np.unique(labels)))
            ax2 = sns.scatterplot(
                x=embedding.iloc[:, 0],
                y=embedding.iloc[:, 1],
                hue=labels,
                hue_order=np.unique(labels),
                palette=cat_pal,
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
                palette=cat_pal,
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
    def _plot_latent_clustermap(
        lat_space: pd.DataFrame,
        labels: Optional[Union[list, pd.Series, None]] = None,
        param: Optional[Union[str, None]] = None,
    ) -> matplotlib.figure.Figure:
        """Creates a clustermap of the latent space dimension where each row shows the intensity of a latent dimension and columns are clustered.

        Args:
            lat_space: DataFrame containing the latent space intensities for samples (rows) and latent dimensions (columns)
            labels: List of labels for each sample. If None, all samples are considered as one group.
            param: Clinical parameter to create groupings and coloring of ridges. Must be a column name (str) of clin_data
        Returns:
            fig: Figure object containing the clustermap
        """
        lat_space[param] = labels

        cluster_figure = sns.clustermap(
            lat_space.groupby(param).mean(),
            col_cluster=False,
            row_cluster=True,
            figsize=(1 * lat_space.shape[1], 4 + 0.5 * len(set(labels))),
            dendrogram_ratio=0.1,
            cmap="icefire",
            cbar_kws={"orientation": "horizontal"},
            cbar_pos=(0.2, 0.95, 0.3, 0.02),
        ).fig

        plt.close()
        lat_space.drop(columns=[param], inplace=True)
        return cluster_figure

    @staticmethod
    def _plot_latent_ridge(
        lat_space: pd.DataFrame,
        labels: Optional[Union[list, pd.Series, None]] = None,
        param: Optional[Union[str, None]] = None,
    ) -> sns.FacetGrid:
        """Creates a ridge line plot of latent space dimension where each row shows the density of a latent dimension and groups (ridges).

        Args:
            lat_space: DataFrame containing the latent space intensities for samples (rows) and latent dimensions (columns)
            labels: List of labels for each sample. If None, all samples are considered as one group.
            param: Clinical parameter to create groupings and coloring of ridges. Must be a column name (str) of clin_data
        Returns:
            g: FacetGrid object containing the ridge line plot
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
                # Change all non-float labels to NaN
                labels = [x if isinstance(x, float) else float("nan") for x in labels]
                labels = list(
                    pd.qcut(
                        x=pd.Series(labels),
                        q=4,
                        labels=["1stQ", "2ndQ", "3rdQ", "4thQ"],
                    ).astype(str)
                )
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

        # if len(np.unique(df[param])) > 8:
        #     cat_pal = sns.husl_palette(len(np.unique(df[param])))
        # else:
        #     cat_pal = sns.color_palette(n_colors=len(np.unique(df[param])))

        if len(np.unique(labels)) > 8:
            cat_pal = sns.color_palette("tab20", n_colors=len(labels))
        else:
            cat_pal = sns.color_palette("tab10", n_colors=len(labels))

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

    def _plot_evaluation(
        self,
        result: Result,
    ) -> dict:
        """Plots the evaluation results from the Result object.

        Args:
            result: The Result object containing evaluation data.

        Returns:
            The generated dictionary containing the evaluation plots.
        """
        ## Plot all results

        ml_plots = dict()
        plt.ioff()
        if not hasattr(result.embedding_evaluation, "CLINIC_PARAM"):
            warnings.warn(
                "We could not create visualizations for the evaluation plots.\n"
                "This usually happens if you try to visualize after saving and loading "
                "the pipeline object with `save_all=False`. This memory-efficient saving mode "
                "Set save_all=True to avoid this, also this might be fixed soon."
            )
            return {}

        for c in pd.unique(result.embedding_evaluation.CLINIC_PARAM):
            ml_plots[c] = dict()
            for m in pd.unique(  # ty: ignore
                result.embedding_evaluation.loc[
                    result.embedding_evaluation.CLINIC_PARAM == c, "metric"
                ]
            ):
                ml_plots[c][m] = dict()
                for alg in pd.unique(  # ty: ignore
                    result.embedding_evaluation.loc[
                        (result.embedding_evaluation.CLINIC_PARAM == c)
                        & (result.embedding_evaluation.metric == m),
                        "ML_ALG",
                    ]
                ):
                    data = result.embedding_evaluation[
                        (result.embedding_evaluation.metric == m)
                        & (result.embedding_evaluation.CLINIC_PARAM == c)
                        & (result.embedding_evaluation.ML_ALG == alg)
                    ]

                    # Check for missing values
                    if data["value"].isnull().any():
                        warnings.warn(
                            f"Missing values found in evaluation data for parameter '{c}', metric '{m}', and algorithm '{alg}'. These will be ignored in the plot."
                        )
                        data = data.dropna()

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

    @staticmethod
    def _total_correlation(latent_space: pd.DataFrame) -> float:
        """Function to compute the total correlation as described here (Equation2): https://doi.org/10.3390/e21100921

        Args:
            latent_space: latent space with dimension sample vs. latent dimensions
        Returns:
            tc: total correlation across latent dimensions
        """
        lat_cov = np.cov(latent_space.T)
        tc = 0.5 * (np.sum(np.log(np.diag(lat_cov))) - np.linalg.slogdet(lat_cov)[1])
        return tc

    @staticmethod
    def _coverage_calc(latent_space: pd.DataFrame) -> float:
        """Function to compute the coverage as described here (Equation3): https://doi.org/10.3390/e21100921

        Args:
            latent_space: latent space with dimension sample vs. latent dimensions
        Returns:
            cov: coverage across latent dimensions
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
