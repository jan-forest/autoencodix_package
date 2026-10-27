from typing import Any, Dict, List, Optional, Tuple, no_type_check
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.utils._result import Result
from autoencodix.visualize._xmodal_visualizer import XModalVisualizer


class XModal3DVisualizer(XModalVisualizer):
    """Visualizer for XModalix3D.

    The class inherits all dimension-independent XModalix visualizations from
    :class:`XModalVisualizer`, including loss plots, latent-space plots and
    downstream-evaluation plots.

    Methods that operate directly on image tensors are adapted here for 3D
    volumes. Volumes are visualized through the three orthogonal central views:
    axial, coronal and sagittal.

    The class also provides a 3D-compatible implementation of
    ``show_2D_translation``. In the original XModalix implementation this method
    flattens 2D images before dimensionality reduction. For 3D volumes that can
    become prohibitively large, therefore XModal3D subsamples the flattened voxel
    grid to a configurable maximum number of voxels before applying UMAP, PCA or
    t-SNE.
    """
    
    def visualize(self, result: Result, config: DefaultConfig) -> Result:
        """
        Overwrites method implemented in XModalVisualizer
        """
        
        ## Make long format of losses
        loss_df_melt = self._make_loss_format(
            result=result, 
            config=config)

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
                "No loss data: This usually happens if you try to visualize "
                "after saving and loading the pipeline object with `save_all=False`."
                "This memory-efficient saving mode does not retain past training loss data."
            )
            return result
        
        ## Make plot loss absolute
        self.plots["loss_absolute"] = self._make_loss_plot(
            df_plot=loss_df_melt, 
            plot_type="absolute"
        )
        
        ## Relative loss plot without clf_loss
        loss_df_relative = loss_df_melt[
            loss_df_melt["Loss Term"] != "clf_loss"
        ].copy()
        
        ## Make plot loss relative
        self.plots["loss_relative"] = self._make_loss_plot(
            df_plot=loss_df_relative, 
            plot_type="relative"
        )

        return result

    @no_type_check
    def show_image_translation(
        self,
        result: Result,
        from_key: str,
        to_key: str,
        n_sample_per_class: int = 3,
        param: Optional[str] = None,
        split: str = "test",
        channel: int = 0,
        robust_percentiles: Tuple[float, float] = (1.0, 99.0),
    ) -> None:
        """Visualize 3D XModalix translation using three central orthogonal views.

        For every selected subject the figure shows:

        - original target volume,
        - translated target volume reconstructed from the source modality,
        - reference target reconstruction obtained from the target modality itself,

        for the axial, coronal and sagittal central slices.

        The method intentionally keeps the original XModalix method name so that
        existing calls such as ``pipeline.show_result()`` can use this visualizer
        without needing a separate 3D-specific API.

        Args:
            result:
                XModalix3D result object.
            from_key:
                Source modality key. Retained for API compatibility and figure
                context.
            to_key:
                Target modality key, e.g. ``"img.CBF"``.
            n_sample_per_class:
                Number of subjects to show per metadata class. If ``param`` is
                None, this is the total number of randomly selected subjects.
            param:
                Metadata column used for grouped sampling. If None, subjects are
                sampled from all paired subjects together.
            split:
                Data split containing translation output. XModalix prediction
                normally stores translations on ``"test"``.
            channel:
                Channel to display for volumes shaped ``(C, D, H, W)``.
            robust_percentiles:
                Lower and upper percentiles used for a common display range for
                original, translated and reference volumes of each subject.
        """

        if "img" not in to_key.lower():
            raise ValueError(
                f"You provided to_key={to_key!r}, which does not look like an "
                "image/volume modality."
            )

        split_data = self._get_split_dataset(result=result, split=split)

        if not hasattr(split_data, "datasets"):
            raise TypeError(
                "XModal3DVisualizer expects a MultiModalDataset-like split with "
                "a 'datasets' attribute."
            )

        if to_key not in split_data.datasets:
            raise KeyError(
                f"Target modality {to_key!r} not found. "
                f"Available modalities: {list(split_data.datasets.keys())}"
            )

        reconstructions = result.reconstructions.get(epoch=-1, split=split)
        sample_ids_result = result.sample_ids.get(epoch=-1, split=split)

        if not isinstance(reconstructions, dict):
            raise TypeError(
                "Expected result.reconstructions for XModalix3D prediction to "
                "be a dictionary."
            )

        if not isinstance(sample_ids_result, dict):
            raise TypeError(
                "Expected result.sample_ids for XModalix3D prediction to be a "
                "dictionary."
            )

        translation_key = "translation"
        reference_key = f"reference_{to_key}_to_{to_key}"

        missing_reconstruction_keys = [
            key
            for key in [translation_key, reference_key]
            if key not in reconstructions
        ]
        if missing_reconstruction_keys:
            raise KeyError(
                "Missing XModalix3D prediction reconstruction key(s): "
                f"{missing_reconstruction_keys}. Run predict() for the requested "
                "translation direction before calling show_image_translation()."
            )

        missing_sample_id_keys = [
            key
            for key in [translation_key, reference_key]
            if key not in sample_ids_result
        ]
        if missing_sample_id_keys:
            raise KeyError(
                "Missing sample IDs for reconstruction key(s): "
                f"{missing_sample_id_keys}."
            )

        target_dataset = split_data.datasets[to_key]
        original_ids = [str(sample_id) for sample_id in target_dataset.sample_ids]

        translated_ids = [
            str(sample_id) for sample_id in sample_ids_result[translation_key]
        ]
        reference_ids = [
            str(sample_id) for sample_id in sample_ids_result[reference_key]
        ]

        available_ids = set(original_ids)
        available_ids &= set(translated_ids)
        available_ids &= set(reference_ids)

        paired_ids = getattr(split_data, "paired_sample_ids", None)
        if paired_ids is not None:
            available_ids &= {str(sample_id) for sample_id in paired_ids}

        if not available_ids:
            raise ValueError(
                "No sample IDs are shared by original target volumes, translated "
                "volumes and reference reconstructions."
            )

        selected_ids, display_labels, plot_param = self._select_sample_ids(
            dataset=target_dataset,
            available_ids=available_ids,
            n_sample_per_class=n_sample_per_class,
            param=param,
        )

        original_lookup = self._make_id_lookup(original_ids)
        translation_lookup = self._make_id_lookup(translated_ids)
        reference_lookup = self._make_id_lookup(reference_ids)

        view_names = ["Axial", "Coronal", "Sagittal"]
        source_names = ["Original", "Translated", "Reference"]

        row_labels = [
            f"{source}\n{view}"
            for view in view_names
            for source in source_names
        ]

        n_samples = len(selected_ids)

        fig, axes = plt.subplots(
            nrows=9,
            ncols=n_samples,
            figsize=(max(4.0 * n_samples, 6.0), 30.0),
            squeeze=False,
            constrained_layout=True,
        )

        for column, (sample_id, display_label) in enumerate(
            zip(selected_ids, display_labels)
        ):
            original = self._extract_dataset_volume(
                dataset=target_dataset,
                index=original_lookup[sample_id],
                channel=channel,
            )
            translated = self._as_3d_volume(
                reconstructions[translation_key][translation_lookup[sample_id]],
                channel=channel,
            )
            reference = self._as_3d_volume(
                reconstructions[reference_key][reference_lookup[sample_id]],
                channel=channel,
            )

            image_min, image_max = self._common_intensity_limits(
                [original, translated, reference],
                percentiles=robust_percentiles,
            )

            original_slices = self._get_central_orthogonal_slices(original)
            translated_slices = self._get_central_orthogonal_slices(translated)
            reference_slices = self._get_central_orthogonal_slices(reference)

            rows = []
            for view in ("axial", "coronal", "sagittal"):
                rows.extend(
                    [
                        original_slices[view],
                        translated_slices[view],
                        reference_slices[view],
                    ]
                )

            for row, slice_2d in enumerate(rows):
                ax = axes[row, column]
                ax.imshow(
                    slice_2d,
                    cmap="gray",
                    origin="lower",
                    vmin=image_min,
                    vmax=image_max,
                )
                ax.axis("off")

                if column == 0:
                    ax.annotate(
                        row_labels[row],
                        xy=(-0.12, 0.5),
                        xycoords="axes fraction",
                        va="center",
                        ha="right",
                        fontsize=10,
                    )

                if row == 0:
                    ax.set_title(display_label, fontsize=10)

        fig.suptitle(
            f"XModalix3D translation: {from_key} → {to_key}",
            fontsize=14,
            fontweight="bold",
        )

        self.plots["Image-translation-3D"][to_key][split][plot_param] = fig
        plt.show()

    @no_type_check
    def show_image_recon_grid(
        self,
        result: Result,
        modality: str,
        n_samples: int = 3,
        selection: str = "random",
        metric: str = "mse",
        chosen_split: str = "test",
        reconstruction_key: Optional[str] = None,
        channel: int = 0,
        robust_percentiles: Tuple[float, float] = (1.0, 99.0),
    ) -> None:
        """Show 3D reconstruction quality for one XModalix3D modality.

        This is the multimodal counterpart of
        ``Imagix3DVisualizer.show_image_recon_grid``. Each selected subject is
        shown as original, reconstructed and absolute-error images for the
        central axial, coronal and sagittal views.

        Args:
            result:
                XModalix3D result object.
            modality:
                Modality key, e.g. ``"img.NCCT"``.
            n_samples:
                Number of subjects for ``selection="random"``.
            selection:
                ``"random"`` or ``"bmw"``. BMW selects the best, median and
                worst reconstruction according to ``metric``.
            metric:
                ``"mse"`` or ``"bce"``.
            chosen_split:
                ``"train"``, ``"valid"`` or ``"test"``.
            reconstruction_key:
                Key inside ``result.reconstructions``. If None, the modality key
                itself is used. For a target-modality reference reconstruction
                one may pass, for example,
                ``"reference_img.CBF_to_img.CBF"``.
            channel:
                Channel to display for ``(C, D, H, W)`` volumes.
            robust_percentiles:
                Display-range percentiles for original and reconstruction.
        """

        if selection not in {"random", "bmw"}:
            raise ValueError("selection must be either 'random' or 'bmw'.")

        if metric not in {"mse", "bce"}:
            raise ValueError("metric must be either 'mse' or 'bce'.")

        if chosen_split not in {"train", "valid", "test"}:
            raise ValueError(
                "chosen_split must be 'train', 'valid' or 'test'."
            )

        split_data = self._get_split_dataset(
            result=result,
            split=chosen_split,
        )

        if not hasattr(split_data, "datasets"):
            raise TypeError(
                "XModal3DVisualizer expects a MultiModalDataset-like split."
            )

        if modality not in split_data.datasets:
            raise KeyError(
                f"Modality {modality!r} not found in split {chosen_split!r}. "
                f"Available: {list(split_data.datasets.keys())}"
            )

        dataset = split_data.datasets[modality]
        epoch = self._resolve_reconstruction_epoch(
            result=result,
            split=chosen_split,
        )

        reconstructions = result.reconstructions.get(
            epoch=epoch,
            split=chosen_split,
        )
        if not isinstance(reconstructions, dict):
            raise TypeError(
                "Expected XModalix3D reconstructions to be stored as a dictionary."
            )

        if reconstruction_key is None:
            reconstruction_key = modality

        if reconstruction_key not in reconstructions:
            available = list(reconstructions.keys())
            raise KeyError(
                f"Reconstruction key {reconstruction_key!r} not found. "
                f"Available keys: {available}"
            )

        sample_ids_result = result.sample_ids.get(
            epoch=epoch,
            split=chosen_split,
        )

        dataset_ids = [str(sample_id) for sample_id in dataset.sample_ids]
        dataset_lookup = self._make_id_lookup(dataset_ids)

        if isinstance(sample_ids_result, dict) and reconstruction_key in sample_ids_result:
            reconstruction_ids = [
                str(sample_id)
                for sample_id in sample_ids_result[reconstruction_key]
            ]
        elif isinstance(sample_ids_result, dict) and modality in sample_ids_result:
            reconstruction_ids = [
                str(sample_id)
                for sample_id in sample_ids_result[modality]
            ]
        else:
            if len(reconstructions[reconstruction_key]) != len(dataset_ids):
                raise ValueError(
                    "Could not align reconstruction rows with dataset sample IDs."
                )
            reconstruction_ids = dataset_ids.copy()

        reconstruction_lookup = self._make_id_lookup(reconstruction_ids)

        common_ids = [
            sample_id
            for sample_id in dataset_ids
            if sample_id in reconstruction_lookup
        ]

        if not common_ids:
            raise ValueError(
                "No sample IDs overlap between the selected modality dataset and "
                "the stored reconstructions."
            )

        scored_samples = []
        for sample_id in common_ids:
            original = self._extract_dataset_volume(
                dataset=dataset,
                index=dataset_lookup[sample_id],
                channel=channel,
            )
            reconstructed = self._as_3d_volume(
                reconstructions[reconstruction_key][
                    reconstruction_lookup[sample_id]
                ],
                channel=channel,
            )

            score = self._reconstruction_score(
                original,
                reconstructed,
                metric=metric,
            )

            scored_samples.append(
                {
                    "sample_id": sample_id,
                    "score": score,
                }
            )

        if selection == "random":
            n_samples = min(n_samples, len(scored_samples))
            rng = np.random.default_rng(42)
            chosen_indices = rng.choice(
                len(scored_samples),
                size=n_samples,
                replace=False,
            )
            selected = [
                {
                    **scored_samples[int(index)],
                    "label": "random",
                    "show_score": False,
                }
                for index in chosen_indices
            ]
        else:
            ordered = sorted(
                scored_samples,
                key=lambda item: item["score"],
            )
            selected = [
                {**ordered[-1], "label": "worst", "show_score": True},
                {
                    **ordered[len(ordered) // 2],
                    "label": "median",
                    "show_score": True,
                },
                {**ordered[0], "label": "best", "show_score": True},
            ]
            n_samples = 3

        row_labels = [
            "Original\nAxial",
            "Recon\nAxial",
            "Error\nAxial",
            "Original\nCoronal",
            "Recon\nCoronal",
            "Error\nCoronal",
            "Original\nSagittal",
            "Recon\nSagittal",
            "Error\nSagittal",
        ]

        fig, axes = plt.subplots(
            nrows=9,
            ncols=n_samples,
            figsize=(max(4.0 * n_samples, 6.0), 30.0),
            squeeze=False,
            constrained_layout=True,
        )

        for column, item in enumerate(selected):
            sample_id = item["sample_id"]

            original = self._extract_dataset_volume(
                dataset=dataset,
                index=dataset_lookup[sample_id],
                channel=channel,
            )
            reconstructed = self._as_3d_volume(
                reconstructions[reconstruction_key][
                    reconstruction_lookup[sample_id]
                ],
                channel=channel,
            )
            error = np.abs(original - reconstructed)

            image_min, image_max = self._common_intensity_limits(
                [original, reconstructed],
                percentiles=robust_percentiles,
            )

            finite_error = error[np.isfinite(error)]
            if finite_error.size:
                error_max = float(np.nanpercentile(finite_error, 99.0))
                if not np.isfinite(error_max) or error_max <= 0:
                    error_max = float(np.nanmax(finite_error))
            else:
                error_max = 1.0

            if not np.isfinite(error_max) or error_max <= 0:
                error_max = 1.0

            original_slices = self._get_central_orthogonal_slices(original)
            reconstructed_slices = self._get_central_orthogonal_slices(
                reconstructed
            )
            error_slices = self._get_central_orthogonal_slices(error)

            rows = []
            row_is_error = []

            for view in ("axial", "coronal", "sagittal"):
                rows.extend(
                    [
                        original_slices[view],
                        reconstructed_slices[view],
                        error_slices[view],
                    ]
                )
                row_is_error.extend([False, False, True])

            for row, (slice_2d, is_error) in enumerate(
                zip(rows, row_is_error)
            ):
                ax = axes[row, column]

                if is_error:
                    ax.imshow(
                        slice_2d,
                        cmap="magma",
                        origin="lower",
                        vmin=0.0,
                        vmax=error_max,
                    )
                else:
                    ax.imshow(
                        slice_2d,
                        cmap="gray",
                        origin="lower",
                        vmin=image_min,
                        vmax=image_max,
                    )

                ax.axis("off")

                if column == 0:
                    ax.annotate(
                        row_labels[row],
                        xy=(-0.12, 0.5),
                        xycoords="axes fraction",
                        va="center",
                        ha="right",
                        fontsize=10,
                    )

                if row == 0:
                    if item["show_score"]:
                        title = (
                            f"{item['label']}: {sample_id}\n"
                            f"{metric.upper()} = {item['score']:.4g}"
                        )
                    else:
                        title = sample_id
                    ax.set_title(title, fontsize=10)

        fig.suptitle(
            f"XModalix3D reconstruction: {modality}",
            fontsize=14,
            fontweight="bold",
        )

        self.plots["Image-recon-grid-3D"][modality][chosen_split][
            reconstruction_key
        ] = fig
        plt.show()

    @no_type_check
    def show_2D_translation(
        self,
        result: Result,
        translated_modality: str,
        split: str = "test",
        param: Optional[str] = None,
        reducer: str = "UMAP",
        max_voxels: int = 20000,
        random_state: int = 42,
        channel: int = 0,
    ) -> None:
        """Compare original and translated 3D volumes in a 2D embedding.

        This overrides the XModalix implementation because flattening complete
        3D volumes can require very large amounts of memory. The same uniformly
        sampled voxel positions are used for original and translated volumes,
        after which UMAP, PCA or t-SNE reduces the vectors to two dimensions.

        The name ``show_2D_translation`` is retained for compatibility with the
        XModalix API; "2D" refers to the final embedding, not to 2D input images.

        Args:
            result:
                XModalix3D result object.
            translated_modality:
                Target modality whose real and translated volumes are compared.
            split:
                Currently translation output is normally available for ``"test"``.
            param:
                Optional metadata column used as hue.
            reducer:
                ``"UMAP"``, ``"PCA"`` or ``"TSNE"``.
            max_voxels:
                Maximum number of voxel features retained per volume before
                dimensionality reduction.
            random_state:
                Random seed used by UMAP/t-SNE where supported.
            channel:
                Channel to use for multi-channel 3D volumes.
        """

        if split != "test":
            raise NotImplementedError(
                "3D translation visualization is currently implemented for "
                "the test split, matching XModalix prediction storage."
            )

        if max_voxels < 1:
            raise ValueError("max_voxels must be >= 1.")

        split_data = self._get_split_dataset(result=result, split=split)

        if not hasattr(split_data, "datasets"):
            raise TypeError(
                "XModal3DVisualizer expects a MultiModalDataset-like split."
            )

        if translated_modality not in split_data.datasets:
            raise KeyError(
                f"Modality {translated_modality!r} not found. "
                f"Available: {list(split_data.datasets.keys())}"
            )

        dataset = split_data.datasets[translated_modality]

        reconstructions = result.reconstructions.get(epoch=-1, split=split)
        sample_ids_result = result.sample_ids.get(epoch=-1, split=split)

        if not isinstance(reconstructions, dict) or "translation" not in reconstructions:
            raise ValueError(
                "No translated volumes found. Run predict() before calling "
                "show_2D_translation()."
            )

        if not isinstance(sample_ids_result, dict) or "translation" not in sample_ids_result:
            raise ValueError(
                "No sample IDs for translated volumes were found."
            )

        original_ids = [str(sample_id) for sample_id in dataset.sample_ids]
        translated_ids = [
            str(sample_id) for sample_id in sample_ids_result["translation"]
        ]

        original_lookup = self._make_id_lookup(original_ids)
        translated_lookup = self._make_id_lookup(translated_ids)

        common_ids = [
            sample_id
            for sample_id in original_ids
            if sample_id in translated_lookup
        ]

        if len(common_ids) < 2:
            raise ValueError(
                "At least two shared samples are required for 2D translation "
                "visualization."
            )

        first_original = self._extract_dataset_volume(
            dataset=dataset,
            index=original_lookup[common_ids[0]],
            channel=channel,
        ).reshape(-1)

        n_voxels = first_original.size
        n_keep = min(max_voxels, n_voxels)

        if n_keep == n_voxels:
            voxel_indices = np.arange(n_voxels)
        else:
            voxel_indices = np.linspace(
                0,
                n_voxels - 1,
                num=n_keep,
                dtype=np.int64,
            )

        original_rows = []
        translated_rows = []

        for sample_id in common_ids:
            original = self._extract_dataset_volume(
                dataset=dataset,
                index=original_lookup[sample_id],
                channel=channel,
            ).reshape(-1)

            translated = self._as_3d_volume(
                reconstructions["translation"][
                    translated_lookup[sample_id]
                ],
                channel=channel,
            ).reshape(-1)

            if original.size != n_voxels or translated.size != n_voxels:
                raise ValueError(
                    "All original and translated volumes must have the same "
                    "spatial shape for this visualization."
                )

            original_rows.append(original[voxel_indices])
            translated_rows.append(translated[voxel_indices])

        original_matrix = np.asarray(original_rows, dtype=np.float32)
        translated_matrix = np.asarray(translated_rows, dtype=np.float32)

        X = np.vstack([original_matrix, translated_matrix])

        reducer_upper = reducer.upper()

        if reducer_upper == "UMAP":
            reducer_model = UMAP(
                n_components=2,
                random_state=random_state,
            )
        elif reducer_upper == "PCA":
            reducer_model = PCA(n_components=2)
        elif reducer_upper == "TSNE":
            perplexity = min(
                30.0,
                max(1.0, float(X.shape[0] - 1) / 3.0),
            )
            reducer_model = TSNE(
                n_components=2,
                random_state=random_state,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
            )
        else:
            raise ValueError(
                "reducer must be one of 'UMAP', 'PCA' or 'TSNE'."
            )

        embedding = reducer_model.fit_transform(X)

        embedding_df = pd.DataFrame(
            embedding,
            columns=["DIM1", "DIM2"],
        )
        embedding_df["origin"] = (
            ["input"] * len(common_ids)
            + ["translated"] * len(common_ids)
        )
        embedding_df["sample_id"] = common_ids + common_ids

        metadata = getattr(dataset, "metadata", None)

        if param is None:
            hue = "origin"
            style = None
        else:
            if metadata is None:
                raise ValueError(
                    f"param={param!r} was requested, but no metadata are available."
                )

            metadata = metadata.copy()
            metadata.index = metadata.index.astype(str)

            if param not in metadata.columns:
                raise KeyError(
                    f"Metadata column {param!r} not found. "
                    f"Available: {list(metadata.columns)}"
                )

            label_lookup = metadata[param].to_dict()
            embedding_df[param] = [
                label_lookup.get(sample_id, np.nan)
                for sample_id in embedding_df["sample_id"]
            ]
            hue = param
            style = "origin"

        fig, ax = plt.subplots(figsize=(8, 7))

        sns.scatterplot(
            data=embedding_df,
            x="DIM1",
            y="DIM2",
            hue=hue,
            style=style,
            alpha=0.75,
            s=80,
            ax=ax,
        )

        ax.set_title(
            f"{reducer_upper}: original vs translated {translated_modality}"
        )
        ax.set_xlabel(f"{reducer_upper} dimension 1")
        ax.set_ylabel(f"{reducer_upper} dimension 2")

        fig.tight_layout()

        plot_param = param if param is not None else "origin"
        self.plots["2D-translation-3D"][translated_modality][split][
            plot_param
        ] = fig

        plt.show()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _to_numpy(data: Any) -> np.ndarray:
        """Convert a tensor/array-like object to a NumPy array."""

        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()

        return np.asarray(data)

    @classmethod
    def _as_3d_volume(
        cls,
        volume: Any,
        channel: int = 0,
    ) -> np.ndarray:
        """Return one 3D volume from ``(C,D,H,W)`` or ``(D,H,W)`` input."""

        volume_np = cls._to_numpy(volume).squeeze()

        if volume_np.ndim == 3:
            return volume_np

        # ``squeeze`` only leaves four dimensions if more than one channel is
        # present. Handle this case explicitly.
        raw = cls._to_numpy(volume)

        if raw.ndim == 4:
            if channel < 0 or channel >= raw.shape[0]:
                raise ValueError(
                    f"channel={channel} is out of range for volume with "
                    f"{raw.shape[0]} channels."
                )
            volume_np = raw[channel]

        if volume_np.ndim != 3:
            raise ValueError(
                "Expected a 3D volume with shape (D,H,W) or a channel-first "
                f"volume with shape (C,D,H,W), got shape {raw.shape}."
            )

        return np.asarray(volume_np)

    @staticmethod
    def _get_central_orthogonal_slices(
        volume: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Extract axial, coronal and sagittal middle slices from a 3D volume."""

        if volume.ndim != 3:
            raise ValueError(
                f"Expected a 3D volume, got shape {volume.shape}."
            )

        d_mid, h_mid, w_mid = [
            size // 2
            for size in volume.shape
        ]

        return {
            "axial": volume[d_mid, :, :],
            "coronal": volume[:, h_mid, :],
            "sagittal": volume[:, :, w_mid],
        }

    @classmethod
    def _extract_dataset_volume(
        cls,
        dataset: Any,
        index: int,
        channel: int = 0,
    ) -> np.ndarray:
        """Extract the image tensor from one AUTOENCODIX dataset item."""

        item = dataset[index]

        if isinstance(item, tuple):
            if len(item) < 2:
                raise ValueError(
                    "Dataset tuple does not contain an image tensor at index 1."
                )
            image = item[1]
        elif isinstance(item, dict):
            if "data" in item:
                image = item["data"]
            elif "img" in item:
                image = item["img"]
            else:
                raise ValueError(
                    "Could not find image data in dataset item dictionary."
                )
        elif hasattr(item, "img"):
            image = item.img
        else:
            image = item

        return cls._as_3d_volume(
            image,
            channel=channel,
        )

    @staticmethod
    def _make_id_lookup(
        sample_ids: List[str],
    ) -> Dict[str, int]:
        """Map sample ID to its first row index."""

        lookup: Dict[str, int] = {}

        for index, sample_id in enumerate(sample_ids):
            sample_id = str(sample_id)
            if sample_id not in lookup:
                lookup[sample_id] = index

        return lookup

    @staticmethod
    def _reconstruction_score(
        original: np.ndarray,
        reconstructed: np.ndarray,
        metric: str,
    ) -> float:
        """Compute whole-volume reconstruction score."""

        if original.shape != reconstructed.shape:
            raise ValueError(
                "Original and reconstructed volumes have different shapes: "
                f"{original.shape} vs {reconstructed.shape}."
            )

        if metric == "mse":
            return float(
                np.nanmean(
                    (original - reconstructed) ** 2
                )
            )

        if metric == "bce":
            target = np.clip(
                original,
                0.0,
                1.0,
            )

            prediction = reconstructed

            finite_prediction = prediction[
                np.isfinite(prediction)
            ]

            if finite_prediction.size == 0:
                return float("nan")

            if (
                np.nanmin(finite_prediction) < 0.0
                or np.nanmax(finite_prediction) > 1.0
            ):
                prediction = 1.0 / (
                    1.0 + np.exp(-prediction)
                )

            eps = np.finfo(float).eps
            prediction = np.clip(
                prediction,
                eps,
                1.0 - eps,
            )

            bce = -(
                target * np.log(prediction)
                + (1.0 - target) * np.log(1.0 - prediction)
            )

            return float(np.nanmean(bce))

        raise ValueError(
            f"Unknown reconstruction metric: {metric!r}."
        )

    @classmethod
    def _common_intensity_limits(
        cls,
        volumes: List[np.ndarray],
        percentiles: Tuple[float, float] = (1.0, 99.0),
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate one robust display range shared by several volumes."""

        lower_percentile, upper_percentile = percentiles

        if not (
            0.0 <= lower_percentile < upper_percentile <= 100.0
        ):
            raise ValueError(
                "robust_percentiles must satisfy "
                "0 <= low < high <= 100."
            )

        finite_parts = []

        for volume in volumes:
            values = np.asarray(volume)
            values = values[np.isfinite(values)]
            if values.size:
                finite_parts.append(values)

        if not finite_parts:
            return None, None

        all_values = np.concatenate(finite_parts)

        image_min = float(
            np.nanpercentile(
                all_values,
                lower_percentile,
            )
        )
        image_max = float(
            np.nanpercentile(
                all_values,
                upper_percentile,
            )
        )

        if (
            not np.isfinite(image_min)
            or not np.isfinite(image_max)
            or image_min == image_max
        ):
            return None, None

        return image_min, image_max

    @staticmethod
    def _get_split_dataset(
        result: Result,
        split: str,
    ) -> Any:
        """Resolve original or prediction-time replacement dataset for a split."""

        if split not in {"train", "valid", "test"}:
            raise ValueError(
                "split must be 'train', 'valid' or 'test'."
            )

        dataset_container = result.datasets

        if dataset_container is None:
            dataset_container = DatasetContainer()

        new_datasets = getattr(
            result,
            "new_datasets",
            None,
        )

        if new_datasets is not None:
            new_split = getattr(
                new_datasets,
                split,
                None,
            )
            if new_split is not None:
                return new_split

        split_data = getattr(
            dataset_container,
            split,
            None,
        )

        if split_data is None:
            raise ValueError(
                f"No dataset is available for split {split!r}."
            )

        return split_data

    @staticmethod
    def _resolve_reconstruction_epoch(
        result: Result,
        split: str,
    ) -> int:
        """Resolve the stored reconstruction epoch for one split."""

        if split == "test":
            return -1

        stored_epochs = []

        for epoch_key, split_dict in result.reconstructions._data.items():
            try:
                if isinstance(split_dict, dict) and split in split_dict:
                    stored_epochs.append(int(epoch_key))
            except (TypeError, ValueError):
                continue

        # Fallback compatible with the Result storage used by current
        # AUTOENCODIX versions.
        if not stored_epochs:
            stored_epochs = [
                int(epoch)
                for epoch in result.reconstructions._data.keys()
                if epoch != -1
            ]

        if not stored_epochs:
            raise ValueError(
                f"No stored reconstructions found for split {split!r}."
            )

        return max(stored_epochs)

    @staticmethod
    def _select_sample_ids(
        dataset: Any,
        available_ids: set,
        n_sample_per_class: int,
        param: Optional[str],
    ) -> Tuple[List[str], List[str], str]:
        """Select deterministic sample IDs for translation visualization."""

        if n_sample_per_class < 1:
            raise ValueError(
                "n_sample_per_class must be >= 1."
            )

        metadata = getattr(
            dataset,
            "metadata",
            None,
        )

        available_ids = {
            str(sample_id)
            for sample_id in available_ids
        }

        if param is None:
            ordered_ids = [
                str(sample_id)
                for sample_id in dataset.sample_ids
                if str(sample_id) in available_ids
            ]

            n_select = min(
                n_sample_per_class,
                len(ordered_ids),
            )

            rng = np.random.default_rng(42)

            selected_ids = [
                ordered_ids[int(index)]
                for index in rng.choice(
                    len(ordered_ids),
                    size=n_select,
                    replace=False,
                )
            ]

            return (
                selected_ids,
                selected_ids.copy(),
                "all",
            )

        if metadata is None:
            raise ValueError(
                f"param={param!r} was requested, but the target modality "
                "contains no metadata."
            )

        metadata = metadata.copy()
        metadata.index = metadata.index.astype(str)

        if param not in metadata.columns:
            raise KeyError(
                f"Metadata column {param!r} not found. "
                f"Available columns: {list(metadata.columns)}"
            )

        metadata = metadata.loc[
            metadata.index.intersection(
                list(available_ids)
            )
        ].copy()

        if metadata.empty:
            raise ValueError(
                "No metadata rows overlap with the samples available for "
                "translation visualization."
            )

        class_values = metadata[param].drop_duplicates().tolist()

        if len(class_values) > 10:
            warnings.warn(
                f"Found {len(class_values)} class values for {param!r}. "
                "Only the first 10 are shown to keep the figure manageable."
            )
            class_values = class_values[:10]

        selected_ids: List[str] = []
        display_labels: List[str] = []

        for class_value in class_values:
            group = metadata.loc[
                metadata[param] == class_value
            ]

            n_group = min(
                n_sample_per_class,
                len(group),
            )

            sampled = group.sample(
                n=n_group,
                random_state=42,
            )

            for sample_id in sampled.index.tolist():
                sample_id = str(sample_id)
                selected_ids.append(sample_id)
                display_labels.append(
                    f"{class_value}\n{sample_id}"
                )

        if not selected_ids:
            raise ValueError(
                "No samples could be selected for translation visualization."
            )

        return (
            selected_ids,
            display_labels,
            param,
        )
    
    @no_type_check
    def show_modality_linear_probe(
        self,
        summary_df: pd.DataFrame,
        confusion_df: pd.DataFrame,
    ) -> None:
        """Visualize the result of modality linear probing.

        The visualization contains:

        1. The balanced accuracy of the modality linear probe together with
            the chance level.
        2. A row-normalized confusion matrix showing which modalities can be
            distinguished from each other.

        Args:
            summary_df:
                Summary DataFrame returned as the first output of
                XModalix3DEvaluator.modality_linear_probe().

            confusion_df:
                Row-normalized confusion matrix returned as the second output
                of XModalix3DEvaluator.modality_linear_probe().
        """

        required_summary_columns = {
            "split",
            "epoch",
            "representation",
            "linear_probe_balanced_accuracy",
            "chance_level",
            "n_modalities",
        }

        missing_columns = required_summary_columns - set(summary_df.columns)
        if missing_columns:
            raise ValueError(
                "summary_df is missing required column(s): "
                f"{sorted(missing_columns)}"
            )

        if len(summary_df) != 1:
            raise ValueError(
                "show_modality_linear_probe() expects the summary output from "
                "a single modality_linear_probe() evaluation."
            )

        if confusion_df.empty:
            raise ValueError("confusion_df must not be empty.")

        row = summary_df.iloc[0]

        balanced_accuracy = float(row["linear_probe_balanced_accuracy"])
        chance_level = float(row["chance_level"])

        representation = str(row["representation"])
        split = str(row["split"])
        epoch = int(row["epoch"])

        epoch_label = "final" if epoch == -1 else str(epoch)


        # Visual style

        probe_color = "#4C78A8"
        chance_color = "#666666"

        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(13.0, 5.4),
            constrained_layout=True,
            gridspec_kw={"width_ratios": [0.9, 1.1]},
        )

        fig.patch.set_facecolor("white")

        for ax in axes:
            ax.set_facecolor("white")
            ax.set_axisbelow(True)

            ax.yaxis.grid(
                True,
                color="0.90",
                linewidth=0.8,
            )
            ax.xaxis.grid(False)

            ax.tick_params(
                axis="both",
                colors="black",
                labelsize=9,
            )

            ax.xaxis.label.set_color("black")
            ax.yaxis.label.set_color("black")
            ax.title.set_color("black")

            for spine in ax.spines.values():
                spine.set_color("0.35")
                spine.set_linewidth(0.8)

        # Panel A: balanced accuracy as lollipop / point-range plot

        ax = axes[0]

        x_pos = 0.0

        # Stem
        ax.vlines(
            x=x_pos,
            ymin=0.0,
            ymax=balanced_accuracy,
            color=probe_color,
            linewidth=5,
            alpha=0.9,
            zorder=2,
        )

        # Point
        ax.scatter(
            [x_pos],
            [balanced_accuracy],
            s=180,
            color=probe_color,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )

        # Chance line
        ax.axhline(
            chance_level,
            color=chance_color,
            linestyle=(0, (5, 3)),
            linewidth=1.2,
            zorder=1,
        )

        # Directly label the chance line
        ax.text(
            0.98,
            chance_level,
            f"Chance = {chance_level:.3f}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8.5,
            color=chance_color,
        )

        # Accuracy value label
        ax.text(
            x_pos,
            min(balanced_accuracy + 0.04, 0.98),
            f"{balanced_accuracy:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold",
            color="black",
        )

        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([x_pos])
        ax.set_xticklabels(["Linear probe"])
        ax.set_ylabel("Balanced accuracy", fontsize=10)
        ax.set_title(
            "Modality linear probe",
            fontsize=11,
            fontweight="semibold",
            pad=10,
        )

        # Panel B: confusion matrix with cleaner blue palette
        ax = axes[1]

        cmap = sns.light_palette(
            probe_color,
            as_cmap=True,
        )

        sns.heatmap(
            confusion_df,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            square=True,
            linewidths=0.8,
            linecolor="white",
            cbar_kws={
                "label": "Proportion",
                "shrink": 0.95,
            },
            ax=ax,
        )

        # Improve annotation readability dynamically
        values = confusion_df.to_numpy().flatten()
        for text, value in zip(ax.texts, values):
            text.set_fontsize(10)
            text.set_fontweight("semibold" if value >= 0.95 else "normal")
            text.set_color("white" if value >= 0.55 else "0.15")

        ax.set_xlabel("Predicted modality", fontsize=10)
        ax.set_ylabel("True modality", fontsize=10)
        ax.set_title(
            "Modality confusion matrix",
            fontsize=11,
            fontweight="semibold",
            pad=10,
        )

        # Panel labels

        axes[0].text(
            -0.10,
            1.05,
            "A",
            transform=axes[0].transAxes,
            fontsize=13,
            fontweight="bold",
            color="black",
            ha="right",
            va="bottom",
        )

        axes[1].text(
            -0.10,
            1.05,
            "B",
            transform=axes[1].transAxes,
            fontsize=13,
            fontweight="bold",
            color="black",
            ha="right",
            va="bottom",
        )

        sns.despine(
            fig=fig,
            top=True,
            right=True,
        )

        fig.suptitle(
            (
                "Cross-modal modality separability "
                f"({representation}, {split}, epoch {epoch_label})"
            ),
            fontsize=13,
            fontweight="bold",
            color="black",
        )

        self.plots["Modality-linear-probe"][representation][split][epoch] = fig

        plt.show()
    
   
    @no_type_check
    def show_matched_unmatched_analysis(
        self,
        summary_df: pd.DataFrame,
        distances_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Visualize matched versus unmatched cross-modal latent distances.

        The first panel shows the matched/unmatched distance ratio for every
        modality pair. A value below 1 indicates that embeddings belonging
        to the same sample/patient are closer across modalities than
        embeddings belonging to different patients.

        The second panel shows either:

        - violin distributions with overlaid boxplots of matched and unmatched
        distances if ``distances_df`` is supplied, or
        - their mean distances if only ``summary_df`` is supplied.

        The plotting style is specified explicitly to produce consistent,
        publication-ready figures across different Matplotlib environments.

        Args:
            summary_df:
                Summary DataFrame returned by
                XModalix3DEvaluator.matched_unmatched_analysis().

            distances_df:
                Optional long-format distance DataFrame returned when
                matched_unmatched_analysis(..., return_distances=True)
                is used.
        """

        required_summary_columns = {
            "split",
            "epoch",
            "representation",
            "distance_metric",
            "modality_a",
            "modality_b",
            "matched_distance_mean",
            "unmatched_distance_mean",
            "matched_unmatched_ratio",
        }

        missing_columns = (
            required_summary_columns - set(summary_df.columns)
        )

        if missing_columns:
            raise ValueError(
                "summary_df is missing required column(s): "
                f"{sorted(missing_columns)}"
            )

        if summary_df.empty:
            raise ValueError("summary_df must not be empty.")

        representations = (
            summary_df["representation"].astype(str).unique()
        )
        splits = summary_df["split"].astype(str).unique()
        epochs = summary_df["epoch"].unique()
        metrics = (
            summary_df["distance_metric"].astype(str).unique()
        )

        if len(representations) != 1:
            raise ValueError(
                "summary_df must contain exactly one representation."
            )

        if len(splits) != 1:
            raise ValueError(
                "summary_df must contain exactly one split."
            )

        if len(epochs) != 1:
            raise ValueError(
                "summary_df must contain exactly one epoch."
            )

        if len(metrics) != 1:
            raise ValueError(
                "summary_df must contain exactly one distance metric."
            )

        representation = representations[0]
        split = splits[0]
        epoch = int(epochs[0])
        distance_metric = metrics[0]
        epoch_label = "final" if epoch == -1 else str(epoch)

        # Prepare summary data

        plot_df = summary_df.copy()

        # Sort modality pairs by their matched/unmatched ratio.
        plot_df = plot_df.sort_values(
            "matched_unmatched_ratio"
        ).reset_index(drop=True)

        # Multi-line labels are easier to read than strongly rotated labels.
        plot_df["modality_pair"] = (
            plot_df["modality_a"].astype(str)
            + "\n↔\n"
            + plot_df["modality_b"].astype(str)
        )

        pair_order = plot_df["modality_pair"].tolist()

        # Explicit colors make the figure independent of the active
        # Matplotlib / Seaborn style.
        matched_color = "#4C78A8"
        unmatched_color = "#F58518"
        adverse_ratio_color = "#D65F5F"

        pair_palette = {
            "matched": matched_color,
            "unmatched": unmatched_color,
        }

        # Figure

        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(15.0, 5.6),
            constrained_layout=True,
            gridspec_kw={
                "width_ratios": [1.0, 1.2],
            },
        )

        # Explicit background / axis styling prevents differences caused by
        # machine-specific Matplotlib themes.
        fig.patch.set_facecolor("white")

        for ax in axes:
            ax.set_facecolor("white")
            ax.set_axisbelow(True)

            ax.yaxis.grid(
                True,
                color="0.90",
                linewidth=0.8,
            )
            ax.xaxis.grid(False)

            ax.tick_params(
                axis="both",
                colors="black",
                labelsize=9,
            )

            ax.xaxis.label.set_color("black")
            ax.yaxis.label.set_color("black")
            ax.title.set_color("black")

            for spine in ax.spines.values():
                spine.set_color("0.35")
                spine.set_linewidth(0.8)

        # Panel A: matched / unmatched distance ratio
        ax = axes[0]

        sns.barplot(
            data=plot_df,
            x="modality_pair",
            y="matched_unmatched_ratio",
            order=pair_order,
            color=matched_color,
            edgecolor="0.25",
            linewidth=0.8,
            ax=ax,
        )

        # If a ratio >= 1 occurs, highlight the corresponding bar.
        for patch, value in zip(
            ax.patches,
            plot_df["matched_unmatched_ratio"],
        ):
            if value >= 1.0:
                patch.set_facecolor(adverse_ratio_color)

        # Reference line: ratio == 1 means no patient-specific alignment.
        ax.axhline(
            1.0,
            color="0.30",
            linestyle=(0, (5, 3)),
            linewidth=1.2,
            zorder=1,
        )

        # Label the reference line directly rather than using a legend.
        # get_yaxis_transform() means x is in axes coordinates while y is
        # in data coordinates.
        ax.text(
            0.98,
            1.0,
            "No patient-specific alignment",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8.5,
            color="0.30",
        )

        ax.set_xlabel(
            "Modality pair",
            fontsize=10,
            labelpad=8,
        )
        ax.set_ylabel(
            "Matched / unmatched distance ratio",
            fontsize=10,
        )
        ax.set_title(
            "Patient-specific cross-modal alignment",
            fontsize=11,
            fontweight="semibold",
            pad=10,
        )

        # Leave sufficient vertical space for values above the bars.
        max_ratio = float(
            np.nanmax(plot_df["matched_unmatched_ratio"])
        )
        upper_limit = max(1.10, max_ratio + 0.10)

        ax.set_ylim(
            0.0,
            upper_limit,
        )

        ax.tick_params(
            axis="x",
            rotation=0,
        )

        # Add numerical ratio above each bar.
        for patch, value in zip(
            ax.patches,
            plot_df["matched_unmatched_ratio"],
        ):
            ax.text(
                patch.get_x()
                + patch.get_width() / 2,
                patch.get_height() + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="semibold",
                color="black",
            )

        # Panel B: matched / unmatched distances
        ax = axes[1]

        if distances_df is not None:
            required_distance_columns = {
                "modality_a",
                "modality_b",
                "pair_type",
                "distance",
            }

            missing_distance_columns = (
                required_distance_columns
                - set(distances_df.columns)
            )

            if missing_distance_columns:
                raise ValueError(
                    "distances_df is missing required column(s): "
                    f"{sorted(missing_distance_columns)}"
                )

            distance_plot_df = distances_df.copy()

            distance_plot_df["modality_pair"] = (
                distance_plot_df[
                    "modality_a"
                ].astype(str)
                + "\n↔\n"
                + distance_plot_df[
                    "modality_b"
                ].astype(str)
            )

            # Ensure that both panels use exactly the same modality-pair order.
            distance_plot_df["modality_pair"] = pd.Categorical(
                distance_plot_df["modality_pair"],
                categories=pair_order,
                ordered=True,
            )

            sns.boxplot(
                data=distance_plot_df,
                x="modality_pair",
                y="distance",
                hue="pair_type",
                order=pair_order,
                hue_order=[
                    "matched",
                    "unmatched",
                ],
                palette=pair_palette,
                width=0.65,
                showfliers=False,
                linewidth=1.0,
                boxprops={
                    "edgecolor": "0.25",
                    "linewidth": 1.0,
                },
                whiskerprops={
                    "color": "0.25",
                    "linewidth": 1.0,
                },
                capprops={
                    "color": "0.25",
                    "linewidth": 1.0,
                },
                medianprops={
                    "color": "black",
                    "linewidth": 1.6,
                },
                ax=ax,
            )

            ax.set_xlabel(
                "Modality pair",
                fontsize=10,
                labelpad=8,
            )
            ax.set_ylabel(
                f"{distance_metric.capitalize()} distance",
                fontsize=10,
            )
            ax.set_title(
                "Matched vs unmatched distances",
                fontsize=11,
                fontweight="semibold",
                pad=10,
            )

            ax.tick_params(
                axis="x",
                rotation=0,
            )

            # Both violinplot() and boxplot() create legend entries.
            # Deduplicate them explicitly.
            handles, labels = ax.get_legend_handles_labels()

            unique_handles = {}

            for handle, label in zip(
                handles,
                labels,
            ):
                if (
                    label in pair_palette
                    and label not in unique_handles
                ):
                    unique_handles[label] = handle

            legend_order = [
                label
                for label in [
                    "matched",
                    "unmatched",
                ]
                if label in unique_handles
            ]

            legend = ax.legend(
                [
                    unique_handles[label]
                    for label in legend_order
                ],
                [
                    label.capitalize()
                    for label in legend_order
                ],
                title="Pair type",
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                frameon=True,
            )

            # Explicit legend colors avoid dark-theme differences between
            # machines / Matplotlib installations.
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_edgecolor("0.80")
            legend.get_frame().set_alpha(1.0)

            for text in legend.get_texts():
                text.set_color("black")

            legend.get_title().set_color("black")

        else:
            
            # Fallback when individual distances were not requested:
            # plot only mean matched/unmatched distances.
            mean_distance_df = (
                plot_df[
                    [
                        "modality_pair",
                        "matched_distance_mean",
                        "unmatched_distance_mean",
                    ]
                ]
                .melt(
                    id_vars="modality_pair",
                    value_vars=[
                        "matched_distance_mean",
                        "unmatched_distance_mean",
                    ],
                    var_name="pair_type",
                    value_name="distance",
                )
            )

            mean_distance_df["pair_type"] = (
                mean_distance_df["pair_type"].map(
                    {
                        "matched_distance_mean": "matched",
                        "unmatched_distance_mean": "unmatched",
                    }
                )
            )

            sns.barplot(
                data=mean_distance_df,
                x="modality_pair",
                y="distance",
                hue="pair_type",
                order=pair_order,
                hue_order=[
                    "matched",
                    "unmatched",
                ],
                palette=pair_palette,
                edgecolor="0.25",
                linewidth=0.8,
                ax=ax,
            )

            ax.set_xlabel(
                "Modality pair",
                fontsize=10,
                labelpad=8,
            )
            ax.set_ylabel(
                f"Mean {distance_metric} distance",
                fontsize=10,
            )
            ax.set_title(
                "Mean matched vs unmatched distances",
                fontsize=11,
                fontweight="semibold",
                pad=10,
            )

            ax.tick_params(
                axis="x",
                rotation=0,
            )

            legend = ax.legend(
                title="Pair type",
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                frameon=True,
            )

            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_edgecolor("0.80")
            legend.get_frame().set_alpha(1.0)

            for text in legend.get_texts():
                text.set_color("black")

            legend.get_title().set_color("black")

        # Panel labels
        axes[0].text(
            -0.10,
            1.05,
            "A",
            transform=axes[0].transAxes,
            fontsize=13,
            fontweight="bold",
            color="black",
            ha="right",
            va="bottom",
        )

        axes[1].text(
            -0.10,
            1.05,
            "B",
            transform=axes[1].transAxes,
            fontsize=13,
            fontweight="bold",
            color="black",
            ha="right",
            va="bottom",
        )

        # Remove the visually heavier upper/right spines.
        sns.despine(
            fig=fig,
            top=True,
            right=True,
        )

        # Overall figure title
        fig.suptitle(
            (
                "Cross-modal patient alignment "
                f"({representation}, {split}, epoch {epoch_label})"
            ),
            fontsize=13,
            fontweight="bold",
            color="black",
        )

        # Preserve the same storage convention as the original implementation.
        self.plots[
            "Matched-unmatched-alignment"
        ][representation][split][epoch] = fig

        plt.show()    