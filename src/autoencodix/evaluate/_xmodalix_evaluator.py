from typing import Union, Tuple, Optional, no_type_check


import pandas as pd
import torch
import torch.nn.functional as F

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.base import ClassifierMixin, RegressorMixin

from autoencodix.utils._result import Result
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.evaluate._general_evaluator import GeneralEvaluator


class XModalixEvaluator(GeneralEvaluator):
    def __init__(self):
        # super().__init__()
        pass

    @staticmethod
    @no_type_check
    def pure_vae_comparison(
        xmodalix_result: Result,
        pure_vae_result: Result,
        to_key: str,
        param: Optional[str] = None,
    ) -> Tuple[Figure, pd.DataFrame]:
        """Compares the reconstruction performance of a pure VAE model and a cross-modal VAE (xmodalix) model using Mean Squared Error (MSE) on test samples.

        For each sample in the test set, computes the MSE between the original and reconstructed images for:
            - Pure VAE reconstructions ("imagix")
            - xmodalix reference reconstructions ("xmodalix_reference")
            - xmodalix translated reconstructions ("xmodalix_translated")
        The results are merged with sample metadata and returned in a long-format DataFrame suitable for plotting. Optionally, boxplots are generated grouped by a specified metadata parameter.

        Args:
            xmodalix_result: The result object containing xmodalix model outputs and test datasets.
            pure_vae_result: The result object containing pure VAE model outputs and test datasets.
            to_key: The key specifying the target modality in the xmodalix dataset.
            param: Metadata column name to group boxplots by. If None, plots are grouped by model only.

        Returns:
                - The matplotlib/seaborn boxplot figure comparing MSE distributions.
                - DataFrame: Long-format DataFrame containing MSE values and associated metadata for each sample and model.
        """

        if "IMG" not in to_key:
            raise NotImplementedError(
                "Comparison is currently only implemented for the image case."
            )

        ## Pure VAE MSE calculation
        meta_imagix = pure_vae_result.datasets.test.metadata
        if meta_imagix is None:
            raise ValueError("metadata cannot be None")
        sample_ids = list(meta_imagix.index)

        all_sample_order = sample_ids  ## TODO check code, seems unnecessary
        indices = [
            all_sample_order.index(sid) for sid in sample_ids if sid in all_sample_order
        ]

        mse_records = []

        for c in range(len(indices)):
            # print(f"Sample {c+1}/{len(indices)}: {sample_ids[c]}")

            # Original image
            orig = torch.Tensor(
                pure_vae_result.datasets.test.raw_data[indices[c]].img.squeeze()
            )

            # Reconstructed image
            recon = torch.Tensor(
                pure_vae_result.reconstructions.get(split="test", epoch=-1)[
                    indices[c]
                ].squeeze()
            )

            # Calculate MSE via torch
            mse_sample = F.mse_loss(orig, recon, reduction="mean")
            # print(f"Mean Squared Error (MSE) for sample {c+1}: {mse_sample.item()}")

            # Collect results
            mse_records.append(
                {"sample_id": sample_ids[c], "mse_imagix": mse_sample.item()}
            )

        df_imagix_mse = pd.DataFrame(mse_records)
        df_imagix_mse.set_index("sample_id", inplace=True)
        # Merge with meta_imagix
        df_imagix_mse = df_imagix_mse.join(meta_imagix, on="sample_id")

        meta_xmodalix = xmodalix_result.datasets.test.datasets[to_key].metadata
        sample_ids = list(meta_xmodalix.index)

        all_sample_order = sample_ids
        indices = [
            all_sample_order.index(sid) for sid in sample_ids if sid in all_sample_order
        ]

        mse_records = []

        for c in range(len(indices)):
            # print(f"Sample {c+1}/{len(indices)}: {sample_ids[c]}")

            # Original image
            orig = torch.Tensor(
                xmodalix_result.datasets.test.datasets[to_key][indices[c]][1].squeeze()
            )
            # print(orig.shape)

            # Reference Reconstructed image
            reference = torch.Tensor(
                xmodalix_result.reconstructions.get(epoch=-1, split="test")[
                    f"reference_{to_key}_to_{to_key}"
                ][indices[c]].squeeze()
            )
            # print(reference.shape)

            # Translated Reconstructed image
            translation = torch.Tensor(
                xmodalix_result.reconstructions.get(epoch=-1, split="test")[
                    "translation"
                ][indices[c]].squeeze()
            )
            # print(translation.shape)

            # Calculate MSE via torch
            mse_sample_translated = F.mse_loss(orig, translation, reduction="mean")
            # print(f"Mean Squared Error (MSE) for sample {c+1}: {mse_sample_translated.item()}")
            mse_sample_reference = F.mse_loss(orig, reference, reduction="mean")
            # print(f"Mean Squared Error (MSE) for sample {c+1}: {mse_sample_reference.item()}")

            # Collect results
            mse_records.append(
                {
                    "sample_id": sample_ids[c],
                    "mse_xmodalix_translated": mse_sample_translated.item(),
                    "mse_xmodalix_reference": mse_sample_reference.item(),
                }
            )

        df_xmodalix_mse = pd.DataFrame(mse_records)
        df_xmodalix_mse.set_index("sample_id", inplace=True)

        # Merge with meta_xmodalix
        df_xmodalix_mse = df_xmodalix_mse.join(meta_xmodalix, on="sample_id")

        # Merge via sample_id and keep non overlapping entries
        df_both_mse = df_imagix_mse.merge(
            df_xmodalix_mse, on=list(meta_imagix.columns), how="outer"
        )

        # Make long format for plotting
        df_long = df_both_mse.melt(
            id_vars=[
                col
                for col in df_both_mse.columns
                if col
                not in [
                    "mse_imagix",
                    "mse_xmodalix_translated",
                    "mse_xmodalix_reference",
                ]
            ],
            value_vars=[
                "mse_imagix",
                "mse_xmodalix_translated",
                "mse_xmodalix_reference",
            ],
            var_name="model",
            value_name="mse_value",
        )

        df_long["model"] = df_long["model"].map(
            {
                "mse_imagix": "imagix",
                "mse_xmodalix_translated": "xmodalix_translated",
                "mse_xmodalix_reference": "xmodalix_reference",
            }
        )

        if param:
            plt.figure(figsize=(2 * len(df_long[param].unique()), 8))

            fig = sns.boxplot(data=df_long, x=param, y="mse_value", hue="model")
            sns.move_legend(
                fig,
                "lower center",
                bbox_to_anchor=(0.5, 1),
                ncol=3,
                title=None,
                frameon=False,
            )
        else:
            plt.figure(figsize=(5, 8))

            fig = sns.boxplot(data=df_long, x="model", y="mse_value")
            # Rotate tick labels
            plt.xticks(rotation=-45)
            plt.xlabel("")

        return fig, df_long

    @staticmethod
    def _get_clin_data(datasets) -> Union[pd.Series, pd.DataFrame]:
        """Retrieves the clinical annotation DataFrame (clin_data) from the provided datasets.

        Handles both standard and XModalix dataset structures.
        """
        # XModalix-Case
        if hasattr(datasets.train, "datasets"):
            clin_data = pd.DataFrame()
            splits = [datasets.train, datasets.valid, datasets.test]

            for s in splits:
                for k in s.datasets.keys():
                    print(f"Processing dataset: {k}")
                    # Merge metadata by overlapping columns
                    overlap = clin_data.columns.intersection(
                        s.datasets[k].metadata.columns
                    )
                    if overlap.empty:
                        overlap = s.datasets[k].metadata.columns
                    clin_data = pd.concat(
                        [clin_data, s.datasets[k].metadata[overlap]], axis=0
                    )

            # Remove duplicate rows
            clin_data = clin_data[~clin_data.index.duplicated(keep="first")]
        else:
            # Raise error no annotation given
            raise ValueError(
                "No annotation data found. Please provide a valid annotation data type."
            )
        return clin_data

    def _enrich_results(
        self,
        results: pd.DataFrame,
        sklearn_ml: Union[ClassifierMixin, RegressorMixin],
        ml_type: str,
        task: str,
        sub: str,
    ) -> pd.DataFrame:
        res_ml_alg = [str(sklearn_ml) for x in range(0, results.shape[0])]
        res_ml_type = [ml_type for x in range(0, results.shape[0])]
        res_ml_subtask = [sub for x in range(0, results.shape[0])]

        results["ML_ALG"] = res_ml_alg
        results["ML_TYPE"] = res_ml_type

        modality = task.split("_$_")[1]
        task_xmodal = task.split("_$_")[0]

        results["MODALITY"] = [modality for x in range(0, results.shape[0])]
        results["ML_TASK"] = [task_xmodal for x in range(0, results.shape[0])]

        results["ML_SUBTASK"] = res_ml_subtask

        return results

    @staticmethod
    def _expand_reference_methods(reference_methods: list, result: Result) -> list:
        """
        Expands each reference method by appending a suffix for every key of used data modalities.
        For each method in `reference_methods`, this function generates new method names by concatenating
        the method name with each key for the data modalities of the xmodalix.
        Args:
            reference_methods (list): A list of reference method names to be expanded.
            result (Result): An object containing latent space information.
        Returns:
            list: A list of expanded reference method names, each suffixed with a key from the latent space.
        """

        reference_methods = [
            f"{method}_$_{key}"
            for method in reference_methods
            for key in result.latentspaces.get(epoch=-1, split="train").keys()
        ]

        return reference_methods

    ## New for x-modalix
    @staticmethod
    def _load_input_for_ml(
        task: str, dataset: DatasetContainer, result: Result
    ) -> pd.DataFrame:
        """Loads and processes input data for various machine learning tasks based on the specified task type.

        Task Details:
            - "Latent": Concatenates latent representations from train, validation, and test splits at the final epoch.
            - "UMAP": Applies UMAP dimensionality reduction to the concatenated dataset splits.
            - "PCA": Applies PCA dimensionality reduction to the concatenated dataset splits.
            - "TSNE": Applies t-SNE dimensionality reduction to the concatenated dataset splits.
            - "RandomFeature": Randomly samples columns (features) from the concatenated dataset splits.


        Args:
            task: The type of ML task. Supported values are "Latent", "UMAP", "PCA", "TSNE", and "RandomFeature".
            dataset: The dataset container object holding train, validation, and test splits.
            result: The result object containing model configuration and methods to retrieve latent representations.
        Returns:
            A DataFrame containing the processed input data suitable for the specified ML task.
        Raises:
            ValueError: If the provided task is not supported.
        """

        # final_epoch = result.model.config.epochs - 1
        modality = task.split("_$_")[1]
        task = task.split("_$_")[0]

        if task == "Latent":
            df = pd.concat(
                [
                    result.get_latent_df(epoch=-1, split="train", modality=modality),
                    result.get_latent_df(epoch=-1, split="valid", modality=modality),
                    result.get_latent_df(epoch=-1, split="test", modality=modality),
                ]
            )
        elif task in ["UMAP", "PCA", "TSNE", "RandomFeature"]:
            latent_dim = result.get_latent_df(
                epoch=-1, split="train", modality=modality
            ).shape[1]
            df_processed = pd.concat(
                [
                    dataset.train._to_df(modality=modality),
                    dataset.test._to_df(modality=modality),
                    dataset.valid._to_df(modality=modality),
                ]
            )
            if task == "UMAP":
                reducer = UMAP(n_components=latent_dim)
                df = pd.DataFrame(
                    reducer.fit_transform(df_processed), index=df_processed.index
                )
            elif task == "PCA":
                reducer = PCA(n_components=latent_dim)
                df = pd.DataFrame(
                    reducer.fit_transform(df_processed), index=df_processed.index
                )
            elif task == "TSNE":
                reducer = TSNE(n_components=latent_dim)
                df = pd.DataFrame(
                    reducer.fit_transform(df_processed), index=df_processed.index
                )
            elif task == "RandomFeature":
                df = df_processed.sample(n=latent_dim, axis=1)
        else:
            raise ValueError(
                f"Your ML task {task} is not supported. Please use Latent, UMAP, PCA or RandomFeature."
            )

        return df
