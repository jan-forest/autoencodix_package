from dataclasses import field
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from umap import UMAP
import warnings
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from typing import Any, Dict, Optional, Union, List, no_type_check
from autoencodix.base._base_visualizer import BaseVisualizer
from autoencodix.utils._result import Result
from autoencodix.utils._utils import nested_dict, show_figure
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.data._datasetcontainer import DatasetContainer


class XModalVisualizer(BaseVisualizer):
    plots: Dict[str, Any] = field(
        default_factory=nested_dict
    )  ## Nested dictionary of plots as figure handles

    def __init__(self):
        self.plots = nested_dict()

    def __setitem__(self, key, elem):
        self.plots[key] = elem

    def visualize(self, result: Result, config: DefaultConfig) -> Result:
        ## Make Model Weights plot
        ## TODO needs to be adjusted for X-Modalix ##
        ## Plot Model weights for each sub-VAE ##
        # self.plots["ModelWeights"] = self._plot_model_weights(model=result.model)

        ## Make long format of losses
        loss_df_melt = self._make_loss_format(result=result, config=config)

        ## X-Modalix specific ##
        # Filter loss terms which are specific for each modality VAE
        # Plot only combined loss terms as in old autoencodix framework
        if not hasattr(result.datasets, "train"):
            raise ValueError("result.datasets has no attribute train")
        if result.datasets.train is None:
            raise ValueError("Train attribute of datasets is None")
        loss_df_melt = loss_df_melt[
            ~loss_df_melt["Loss Term"].str.startswith(
                tuple(result.datasets.train.datasets.keys())
            )
        ]
        if not result.losses._data:
            import warnings

            warnings.warn(
                "No loss data: This usually happens if you try to visualize after saving and loading the pipeline object with `save_all=False`. This memory-efficient saving mode does not retain past training loss data."
            )
            return result
        ## Make plot loss absolute
        self.plots["loss_absolute"] = self._make_loss_plot(
            df_plot=loss_df_melt, plot_type="absolute"
        )
        ## Make plot loss relative
        self.plots["loss_relative"] = self._make_loss_plot(
            df_plot=loss_df_melt, plot_type="relative"
        )

        return result

    def show_latent_space(
        self,
        result: Result,
        plot_type: str = "2D-scatter",
        labels: Optional[Union[list, pd.Series, None]] = None,
        param: Optional[Union[list, str]] = None,
        epoch: Optional[Union[int, None]] = None,
        split: str = "all",
    ) -> None:
        plt.ioff()
        if plot_type == "Coverage-Correlation":
            print("TODO: Implement Coverage-Correlation plot for X-Modalix")
            # if "Coverage-Correlation" in self.plots:
            #     fig = self.plots["Coverage-Correlation"]
            #     show_figure(fig)
            #     plt.show()
            # else:
            #     results = []
            #     for epoch in range(result.model.config.checkpoint_interval, result.model.config.epochs + 1, result.model.config.checkpoint_interval):
            #         for split in ["train", "valid"]:
            #             latent_df = result.get_latent_df(epoch=epoch-1, split=split)
            #             tc = self._total_correlation(latent_df)
            #             cov = self._coverage_calc(latent_df)
            #             results.append({"epoch": epoch, "split": split, "total_correlation": tc, "coverage": cov})

            #     df_metrics = pd.DataFrame(results)

            #     fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            #     # Total Correlation plot
            #     ax1 = sns.lineplot(data=df_metrics, x="epoch", y="total_correlation", hue="split", ax=axes[0])
            #     axes[0].set_title("Total Correlation")
            #     axes[0].set_xlabel("Epoch")
            #     axes[0].set_ylabel("Total Correlation")

            #     # Coverage plot
            #     ax2 = sns.lineplot(data=df_metrics, x="epoch", y="coverage", hue="split", ax=axes[1])
            #     axes[1].set_title("Coverage")
            #     axes[1].set_xlabel("Epoch")
            #     axes[1].set_ylabel("Coverage")

            #     plt.tight_layout()
            #     self.plots["Coverage-Correlation"] = fig
            #     show_figure(fig)
            #     plt.show()
        else:
            # Set Defaults
            if epoch is None:
                epoch = -1

            ## Collect all metadata and latent spaces from datasets
            clin_data = []
            latent_data = []

            if split == "all":
                split_list = ["train", "test", "valid"]
            else:
                split_list = [split]
            for s in split_list:
                split_ds = getattr(result.datasets, s, None)
                if split_ds is not None:
                    for key, ds in split_ds.datasets.items():
                        if s == "test":
                            df_latent = result.get_latent_df(
                                epoch=-1, split=s, modality=key
                            )
                        else:
                            df_latent = result.get_latent_df(
                                epoch=epoch, split=s, modality=key
                            )
                        df_latent["modality"] = key
                        df_latent["sample_ids"] = (
                            df_latent.index
                        )  # Each sample can occur multiple times in latent space
                        latent_data.append(df_latent)
                        if hasattr(ds, "metadata") and ds.metadata is not None:
                            df = ds.metadata.copy()
                            df["sample_ids"] = df.index.astype(str)
                            df["split"] = s
                            df["modality"] = key
                            clin_data.append(df)

            if latent_data and clin_data:
                latent_data = pd.concat(latent_data, axis=0, ignore_index=True)
                clin_data = pd.concat(clin_data, axis=0, ignore_index=True)
                if "sample_ids" in clin_data.columns:
                    clin_data = clin_data.drop_duplicates(
                        subset="sample_ids"
                    ).set_index("sample_ids")
            else:
                latent_data = pd.DataFrame()
                clin_data = pd.DataFrame()

            ## Label options
            if param is None:
                modality = list(result.model.keys())[
                    0
                ]  # Take the first since configs are same for all sub-VAEs
                model = result.model.get(modality, None)
                if model is None:
                    raise ValueError(
                        f"Model for modality {modality} not found in result.model"
                    )
                param = model.config.data_config.annotation_columns

            if labels is None and param is None:
                labels = ["all"] * latent_data["sample_ids"].unique().shape[0]

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
                    # Order by index of latent_data first, fill missing with "unknown"
                    labels = labels.reindex(  # ty: ignore
                        latent_data["sample_ids"],  # ty: ignore
                        fill_value="unknown",  # ty: ignore
                    ).tolist()
                else:
                    param = ["user_label"]  # Default label if none provided
            if not isinstance(param, list):
                raise ValueError(f"param: should be converted to list, got: {param}")
            for p in param:
                if p in clin_data.columns:
                    labels: List = clin_data.loc[
                        latent_data["sample_ids"], p
                    ].tolist()  # ty: ignore
                else:
                    if clin_data.shape[0] == len(labels):  # ty: ignore
                        clin_data[p] = labels
                    else:
                        clin_data[p] = ["all"] * clin_data.shape[0]

                if plot_type == "2D-scatter":
                    ## Make 2D Embedding with UMAP
                    if (
                        latent_data.drop(
                            columns=["sample_ids", "modality"]
                        ).shape[  # ty: ignore
                            1
                        ]  # ty: ignore
                        > 2
                    ):
                        reducer = UMAP(n_components=2)
                        embedding = pd.DataFrame(
                            reducer.fit_transform(
                                latent_data.drop(
                                    columns=["sample_ids", "modality"]
                                )  # ty: ignore
                            )
                        )
                        embedding.columns = ["DIM1", "DIM2"]
                        embedding["sample_ids"] = latent_data["sample_ids"]
                        embedding["modality"] = latent_data["modality"]
                    else:
                        embedding = latent_data

                    # Merge with clinical data via sample_ids
                    clin_data["sample_ids"] = clin_data.index.astype(str)
                    clin_data.index = clin_data.index.astype(str)  # Add this line
                    embedding["sample_ids"] = embedding["sample_ids"].astype(str)

                    embedding = embedding.merge(
                        clin_data.drop(columns=["modality"]),  # ty: ignore
                        left_on="sample_ids",
                        right_index=True,
                        how="left",
                    )

                    self.plots["2D-scatter"][epoch][split][p] = (
                        self._plot_translate_latent(
                            embedding=embedding,
                            color_param=p,
                            style_param="modality",
                        )
                    )

                    fig = self.plots["2D-scatter"][epoch][split][p].figure
                    # show_figure(fig)
                    plt.show()

                if plot_type == "Ridgeline":
                    ## Make ridgeline plot
                    if len(labels) != latent_data.shape[0]:  # ty: ignore
                        if labels[0] == "all":  # ty: ignore
                            labels = ["all"] * latent_data.shape[0]  # ty: ignore
                        else:
                            raise ValueError(
                                "Labels must match the number of samples in the latent space."
                            )

                    self.plots["Ridgeline"][epoch][split][p] = (
                        self._plot_latent_ridge_multi(
                            lat_space=latent_data.drop(
                                columns=["sample_ids"]
                            ),  # ty: ignore
                            labels=labels,
                            modality="modality",
                            param=p,
                        )
                    )

                    fig = self.plots["Ridgeline"][epoch][split][p].figure
                    show_figure(fig)
                    plt.show()

    def show_weights(self) -> None:
        ## TODO
        raise NotImplementedError(
            "Weight visualization for X-Modalix is not implemented."
        )

    @no_type_check
    def show_image_translation(  # ty: ignore
        self,
        result: Result,
        from_key: str,
        to_key: str,
        n_sample_per_class: int = 3,
        param: Optional[str] = None,
    ) -> None:  # ty: ignore
        """Visualizes image translation results for a given dataset.

        Split by displaying a grid of original, translated, and reference images,grouped by class values.
        Args:
            result:The result object containing datasets and reconstructions.
            from_key: The source modality key (not directly used in visualization, but relevant for context).
            to_key: The target modality key. Must correspond to an image dataset (must contain "IMG").
            split: The dataset split to visualize ("test", "train", or "valid"). Default is "test".
            n_sample_per_class: Number of samples to display per class value. Default is 3.
            param: The metadata column name used to group samples by class.
        Raises
            ValueError: If `to_key` does not correspond to an image dataset.
        """

        if "img" not in to_key:
            raise ValueError(
                f"You provided as 'to_key' {to_key} a non-image dataset. "
                "Image translation grid visualization is only possible for translation to IMG data type."
            )
        else:
            split = "test"  # Currently only test split is supported
            ## Get n samples per class
            if split == "test":
                meta = result.datasets.test.datasets[to_key].metadata
                paired_sample_ids = result.datasets.test.paired_sample_ids

            # Restrict meta to only paired sample ids
            meta = meta.loc[paired_sample_ids]

            if param is None:
                param = "user-label"
                meta[param] = (
                    "all"  # Default to all samples if no parameter is provided
                )

            # Get possible class values
            class_values = meta[param].unique()
            if len(class_values) > 10:
                # Make warning
                warnings.warn(
                    f"Found {len(class_values)} class values for parameter '{param}'. Only first 10 will be used to limit figure size"
                )
                class_values = class_values[:10]

            # Build dictionary of sample_ids per class value (max n_sample_per_class per class)
            sample_per_class = {
                val: meta[meta[param] == val]
                .sample(
                    n=min(n_sample_per_class, (meta[param] == val).sum()),
                    random_state=42,
                )
                .index.tolist()
                for val in class_values
            }

            print(f"Sample per class: {sample_per_class}")

            # Lookup of sample indices per modality
            sample_ids_per_key = dict()

            for key in result.sample_ids.get(epoch=-1, split="test").keys():
                sample_ids_per_key[key] = result.sample_ids.get(epoch=-1, split="test")[
                    key
                ]
            # Original
            sample_ids_per_key["original"] = result.datasets.test.datasets[
                to_key
            ].sample_ids

            ## Generate Image Grid
            # Number of test (or train or valid) samples from all values in sample_per_class dictionary
            n_test_samples = sum(len(indices) for indices in sample_per_class.values())

            # #
            col_labels = []
            for class_value in sample_per_class:
                col_labels.extend(
                    [
                        class_value + " " + split + "-sample:" + s
                        for s in sample_per_class[class_value]
                    ]
                )

            row_labels = ["Original", "Translated", "Reference"]

            fig, axes = plt.subplots(
                ncols=n_test_samples,  # Number of classes
                nrows=3,  # Original, translated, reference
                figsize=(n_test_samples * 2, 3 * 2),
            )

            for i, ax in enumerate(axes.flat):
                row = int(i / n_test_samples)
                # test_sample = sample_idx_list[i % n_test_samples]
                # print(f"Row: {row}, Column: {i % n_test_samples}")
                # print(f"Current sample: {col_labels[i % n_test_samples]}")

                if row == 0:
                    if split == "test":
                        idx_original = list(sample_ids_per_key["original"]).index(
                            col_labels[i % n_test_samples].split("sample:")[1]
                        )
                        img_temp = result.datasets.test.datasets[to_key][idx_original][
                            1
                        ].squeeze()  # Stored as Tuple (index, tensor, sample_id)

                    # Original image
                    ax.imshow(np.asarray(img_temp))
                    ax.axis("off")
                    # Sample label
                    ax.text(
                        0.5,
                        1.1,
                        col_labels[i],
                        va="bottom",
                        ha="center",
                        # rotation='vertical',
                        rotation=45,
                        transform=ax.transAxes,
                    )
                    # Row label
                    if i % n_test_samples == 0:
                        ax.text(
                            -0.1,
                            0.5,
                            row_labels[0],
                            va="center",
                            ha="right",
                            transform=ax.transAxes,
                        )

                if row == 1:
                    # Translated image
                    idx_translated = list(sample_ids_per_key["translation"]).index(
                        col_labels[i % n_test_samples].split("sample:")[1]
                    )
                    ax.imshow(
                        result.reconstructions.get(epoch=-1, split=split)[
                            "translation"
                        ][idx_translated].squeeze()
                    )
                    ax.axis("off")
                    # Row label
                    if i % n_test_samples == 0:
                        ax.text(
                            -0.1,
                            0.5,
                            row_labels[1],
                            va="center",
                            ha="right",
                            transform=ax.transAxes,
                        )

                if row == 2:
                    # Reference image reconstruction
                    idx_reference = list(
                        sample_ids_per_key[f"reference_{to_key}_to_{to_key}"]
                    ).index(col_labels[i % n_test_samples].split("sample:")[1])
                    ax.imshow(
                        result.reconstructions.get(epoch=-1, split=split)[
                            f"reference_{to_key}_to_{to_key}"
                        ][idx_reference].squeeze()
                    )
                    ax.axis("off")
                    # Row label
                    if i % n_test_samples == 0:
                        ax.text(
                            -0.1,
                            0.5,
                            row_labels[2],
                            va="center",
                            ha="right",
                            transform=ax.transAxes,
                        )

            self.plots["Image-translation"][to_key][split][param] = fig
            # show_figure(fig)
            plt.show()

    @no_type_check
    def show_2D_translation(
        self,
        result: Result,
        translated_modality: str,
        split: str = "test",
        param: Optional[str] = None,
        reducer: str = "UMAP",
    ) -> None:
        ## TODO add similar labels/param logic from other visualizations
        dataset = result.datasets

        ## Overwrite original datasets with new_datasets if available after predict with other data
        if dataset is None:
            dataset = DatasetContainer()

        if bool(result.new_datasets.test):
            dataset.test = result.new_datasets.test

        if split not in ["train", "valid", "test", "all"]:
            raise ValueError(f"Unknown split: {split}")

        if dataset.test is None:
            raise ValueError("test of dataset is None")

        if split == "test":
            df_processed = dataset.test._to_df(modality=translated_modality)
        else:
            raise NotImplementedError(
                "2D translation visualization is currently only implemented for the 'test' split since reconstruction is only performed on test-split."
            )

        # Get translated reconstruction
        tensor_list = result.reconstructions.get(epoch=-1, split=split)[  # ty: ignore
            "translation"
        ]  # ty: ignore
        print(f"len of tensor-list: {len(tensor_list)}")
        tensor_ids = result.sample_ids.get(epoch=-1, split=split)["translation"]
        print(f"len of tensor_ids: {len(tensor_ids)}")

        # Flatten each tensor and collect as rows (for image case)
        rows = [
            t.flatten().cpu().numpy() if isinstance(t, torch.Tensor) else t.flatten()
            for t in tensor_list
        ]

        # Create DataFrame
        df_translate_flat = pd.DataFrame(
            rows,
            columns=["Feature_" + str(i) for i in range(len(rows[0]))],
            index=tensor_ids,
        )

        if reducer == "UMAP":
            reducer_model = UMAP(n_components=2)
        elif reducer == "PCA":
            reducer_model = PCA(n_components=2)
        elif reducer == "TSNE":
            reducer_model = TSNE(n_components=2)

        # making sure of index alignemnt
        common_ids = df_processed.index.intersection(df_translate_flat.index)
        df_processed = df_processed.loc[common_ids]
        df_translate_flat = df_translate_flat.loc[common_ids]
        df_translate_flat = df_translate_flat.reindex(df_processed.index)
        df_translate_flat.index = pd.Index([i for i in range(len(common_ids))])
        X = np.vstack([df_processed.values, df_translate_flat.values])
        df_red_comb = pd.DataFrame(reducer_model.fit_transform(X))

        # df_comb = pd.concat(
        #     [df_processed, df_translate_flat], axis=0, ignore_index=True
        # )

        df_red_comb["origin"] = ["input"] * df_processed.shape[0] + [
            "translated"
        ] * df_translate_flat.shape[0]

        # df_red_comb = pd.DataFrame(
        #     reducer_model.fit_transform(
        #         pd.concat([df_processed, df_translate_flat], axis=0)
        #     )
        # )

        labels = (
            list(
                result.datasets.test.datasets[translated_modality].metadata[param]
            )  # ty: ignore
            * 2
        )
        df_red_comb[param] = (
            labels + labels[0 : df_red_comb.shape[0] - len(labels)]
        )  ## TODO fix for not matching lengths

        g = sns.FacetGrid(
            df_red_comb,
            col="origin",
            hue=param,
            sharex=True,
            sharey=True,
            height=8,
            aspect=1,
        )
        g.map_dataframe(sns.scatterplot, x=0, y=1, alpha=0.7)
        g.add_legend()
        g.set_axis_labels(reducer + " DIM 1", reducer + " DIM 2")
        g.set_titles(col_template="{col_name}")

        self.plots["2D-translation"][translated_modality][split][param] = g
        plt.show()

    ## Utilities specific for X-Modalix
    @staticmethod
    def _plot_translate_latent(
        embedding,
        color_param,
        style_param=None,
    ):
        """Creates a 2D visualization of the 2D embedding of the latent space.
        Args:
            embedding: embedding on which is visualized. Assumes prior 2D dimension reduction.
            color_params: Clinical parameter to color scatter plot
            style_param: Parameter e.g. "Translate" to facet scatter plot
        Returns:
            fig: Figure handle

        """
        labels = list(embedding[color_param])
        # logger = getlogger(cfg)
        numeric = False
        if not isinstance(labels[0], str):
            if len(np.unique(labels)) > 3:
                # TODO Decide if numeric to category should be optional in new Package
                # print(
                #     f"The provided label column is numeric and converted to categories."
                # )
                # labels = pd.qcut(
                #     labels, q=4, labels=["1stQ", "2ndQ", "3rdQ", "4thQ"]
                # ).astype(str)
                # else:
                numeric = True
            else:
                labels = [str(x) for x in labels]

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

        if style_param is not None:
            embedding[color_param] = labels
            if numeric:
                palette = "bwr"
            else:
                palette = None
            plot = sns.relplot(
                data=embedding,
                x="DIM1",
                y="DIM2",
                hue=color_param,
                palette=palette,
                col=style_param,
                style=style_param,
                markers=True,
                alpha=0.4,
                ec="black",
                height=10,
                aspect=1,
                s=150,
            )

        return plot

    @staticmethod
    def _plot_latent_ridge_multi(
        lat_space: pd.DataFrame,
        modality: Optional[str] = None,
        labels: Optional[Union[list, pd.Series, None]] = None,
        param: Optional[Union[str, None]] = None,
    ) -> sns.FacetGrid:
        """Creates a ridge line plot of latent space dimension where each row shows the density of a latent dimension and groups (ridges).
        Args:
            lat_space: DataFrame containing the latent space intensities for samples (rows) and latent dimensions (columns)
            labels: List of labels for each sample. If None, all samples are considered as one group.
            param: Clinical parameter to create groupings and coloring of ridges. Must be a column name (str) of clin_data
        Returns:
            g (sns.FacetGrid): FacetGrid object containing the ridge line plot
        """
        sns.set_theme(
            style="white", rc={"axes.facecolor": (0, 0, 0, 0)}
        )  ## Necessary to enforce overplotting

        df = pd.melt(
            lat_space,
            id_vars=modality,  # ty: ignore
            var_name="latent dim",
            value_name="latent intensity",
        )
        # print(df)
        df["sample"] = len(lat_space.drop(columns=modality).columns) * list(
            lat_space.index
        )

        if labels is None:
            param = "all"
            labels = ["all"] * len(df)

        # print(labels[0])
        if not isinstance(labels[0], str):
            if len(np.unique(labels)) > 3:
                # Change all non-float labels to NaN
                labels = [x if isinstance(x, float) else float("nan") for x in labels]
                labels = pd.qcut(
                    x=pd.Series(labels),
                    q=4,
                    labels=["1stQ", "2ndQ", "3rdQ", "4thQ"],
                ).astype(str)
            else:
                labels = [str(x) for x in labels]

        df[param] = len(lat_space.drop(columns=modality).columns) * labels  # type: ignore

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
            col=modality,
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

        for i, m in enumerate(df[modality].unique()):
            g.fig.get_axes()[i].set_title(m)

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

        for c in pd.unique(result.embedding_evaluation.CLINIC_PARAM):
            ml_plots[c] = dict()
            for m in pd.unique(
                result.embedding_evaluation.loc[
                    result.embedding_evaluation.CLINIC_PARAM == c, "metric"
                ]
            ):  # ty: ignore
                ml_plots[c][m] = dict()
                for alg in pd.unique(
                    result.embedding_evaluation.loc[
                        (result.embedding_evaluation.CLINIC_PARAM == c)
                        & (result.embedding_evaluation.metric == m),
                        "ML_ALG",
                    ]
                ):  # ty: ignore
                    data = result.embedding_evaluation[
                        (result.embedding_evaluation.metric == m)
                        & (result.embedding_evaluation.CLINIC_PARAM == c)
                        & (result.embedding_evaluation.ML_ALG == alg)
                    ]

                    sns_plot = sns.catplot(
                        data=data,
                        x="score_split",
                        y="value",
                        col="ML_TASK",
                        row="MODALITY",
                        hue="score_split",
                        kind="bar",
                    )

                    min_y = data.value.min()
                    if min_y > 0:
                        min_y = 0

                    ml_plots[c][m][alg] = sns_plot.set(ylim=(min_y, None))

        self.plots["ML_Evaluation"] = ml_plots

        return ml_plots
