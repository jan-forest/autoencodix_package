from __future__ import annotations

from itertools import combinations
from typing import Dict, Literal, Tuple, Union

import numpy as np
import pandas as pd
import torch
import gc

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, pairwise_distances
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from umap import UMAP

from autoencodix.evaluate._xmodalix_evaluator import XModalixEvaluator
from autoencodix.utils._result import Result


RepresentationType = Literal["mu", "z"]


class XModalix3DEvaluator(XModalixEvaluator):
    """Evaluator for XModalix3D.

    The class inherits all evaluation functionality from :class:`XModalixEvaluator`
    and adds two quantitative diagnostics for cross-modal latent-space alignment:

    1. Modality linear probing
       A linear classifier is trained on frozen latent representations to predict the
       modality of origin. Low balanced accuracy (close to 1 / n_modalities) indicates
       that modality identity is difficult to recover from the representation.

    2. Matched/unmatched analysis
       For each pair of modalities, latent-space distances between measurements from
       the same sample/patient are compared with distances between measurements from
       different samples/patients. A matched/unmatched distance ratio below 1 indicates
       patient-specific cross-modal alignment.

    """
    
    @staticmethod
    def refresh_final_representations(
        result: Result,
        datasets=None,
        batch_size: int = 2,
        device: str | None = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Ensure complete final latent representations for all splits/modalities.

        Multi-GPU training can leave training dynamics containing only the samples
        seen by one distributed rank. This method checks the final stored
        representations against the actual datasets. Incomplete representations
        are recomputed from the final trained models using a normal, non-distributed
        inference pass.

        Only lightweight latent information is refreshed:
            - latentspaces
            - mus
            - sigmas / latent log-variance
            - sample_ids

        Reconstructions are intentionally not recomputed here.

        Returns
        -------
        pd.DataFrame
            Diagnostic table showing whether each split/modality was taken from
            stored dynamics or recomputed.
        """

        if datasets is None:
            datasets = result.datasets

        if datasets is None:
            raise ValueError(
                "No datasets available for final representation extraction."
            )

        if not isinstance(result.model, dict) or not result.model:
            raise ValueError(
                "XModalix3D evaluation requires result.model to contain "
                "the modality-specific trained models."
            )

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        target_device = torch.device(device)

        # Determine the joint-training final epoch.
        first_model = next(iter(result.model.values()))

        while hasattr(first_model, "module"):
            first_model = first_model.module

        final_epoch = int(first_model.config.epochs) - 1

        diagnostics = []

        split_names = ("train", "valid", "test")

        for split_idx, split_name in enumerate(split_names):

            split_dataset = getattr(datasets, split_name, None)

            if split_dataset is None:
                continue

            # Existing final data. get(epoch=-1)
            try:
                stored_z = result.latentspaces.get(
                    epoch=-1,
                    split=split_name,
                )
            except Exception:
                stored_z = None

            try:
                stored_mu = result.mus.get(
                    epoch=-1,
                    split=split_name,
                )
            except Exception:
                stored_mu = None

            try:
                stored_logvar = result.sigmas.get(
                    epoch=-1,
                    split=split_name,
                )
            except Exception:
                stored_logvar = None

            try:
                stored_ids = result.sample_ids.get(
                    epoch=-1,
                    split=split_name,
                )
            except Exception:
                stored_ids = None

            z_out = {}
            mu_out = {}
            logvar_out = {}
            ids_out = {}

            for mod_idx, modality in enumerate(result.model.keys()):

                if modality not in split_dataset.datasets:
                    raise KeyError(
                        f"Modality {modality!r} is present in result.model "
                        f"but not in dataset split {split_name!r}."
                    )

                modality_dataset = split_dataset.datasets[modality]

                expected_ids = np.asarray(
                    modality_dataset.sample_ids
                )

                expected_n = len(expected_ids)


                # Check whether the stored representation is complete.
                stored_mod_ids = None

                if (
                    isinstance(stored_ids, dict)
                    and modality in stored_ids
                ):
                    stored_mod_ids = np.asarray(
                        stored_ids[modality]
                    )

                stored_n = (
                    len(stored_mod_ids)
                    if stored_mod_ids is not None
                    else 0
                )

                ids_complete = (
                    stored_mod_ids is not None
                    and stored_n == expected_n
                    and len(np.unique(stored_mod_ids)) == expected_n
                    and set(stored_mod_ids) == set(expected_ids)
                )

                z_complete = (
                    ids_complete
                    and isinstance(stored_z, dict)
                    and modality in stored_z
                    and len(stored_z[modality]) == expected_n
                )

                mu_complete = (
                    ids_complete
                    and isinstance(stored_mu, dict)
                    and modality in stored_mu
                    and len(stored_mu[modality]) == expected_n
                )

                logvar_complete = (
                    ids_complete
                    and isinstance(stored_logvar, dict)
                    and modality in stored_logvar
                    and len(stored_logvar[modality]) == expected_n
                )

                if z_complete and mu_complete and logvar_complete:

                    # Reorder stored arrays to exactly match dataset order.
                    id_to_position = {
                        sample_id: pos
                        for pos, sample_id
                        in enumerate(stored_mod_ids)
                    }

                    order = [
                        id_to_position[sample_id]
                        for sample_id in expected_ids
                    ]

                    z_out[modality] = np.asarray(
                        stored_z[modality]
                    )[order]

                    mu_out[modality] = np.asarray(
                        stored_mu[modality]
                    )[order]

                    logvar_out[modality] = np.asarray(
                        stored_logvar[modality]
                    )[order]

                    ids_out[modality] = expected_ids.copy()

                    diagnostics.append(
                        {
                            "split": split_name,
                            "modality": modality,
                            "expected_samples": expected_n,
                            "stored_samples": stored_n,
                            "final_samples": expected_n,
                            "source": "stored",
                        }
                    )

                    continue

                # Stored dynamics are incomplete -> recompute.
                print(
                    f"[XModalix3D evaluation] "
                    f"{split_name}/{modality}: stored final dynamics "
                    f"contain {stored_n}/{expected_n} samples. "
                    f"Recomputing complete final representations..."
                )

                model = result.model[modality]

                while hasattr(model, "module"):
                    model = model.module

                try:
                    original_device = next(
                        model.parameters()
                    ).device
                except StopIteration:
                    original_device = torch.device("cpu")

                was_training = model.training

                model.to(target_device)
                model.eval()

                data = modality_dataset.data

                z_parts = []
                mu_parts = []
                logvar_parts = []

                # Use a reproducible but different RNG stream for every split/modality. 
                local_seed = (
                    seed
                    + split_idx * 10_000
                    + mod_idx * 1_000
                )

                cuda_devices = (
                    [target_device.index]
                    if (
                        target_device.type == "cuda"
                        and target_device.index is not None
                    )
                    else []
                )

                try:
                    with torch.random.fork_rng(
                        devices=cuda_devices
                    ):
                        torch.manual_seed(local_seed)

                        if target_device.type == "cuda":
                            torch.cuda.manual_seed_all(
                                local_seed
                            )

                        with torch.inference_mode():

                            for start in range(
                                0,
                                expected_n,
                                batch_size,
                            ):
                                stop = min(
                                    start + batch_size,
                                    expected_n,
                                )

                                if isinstance(data, list):
                                    batch_values = data[start:stop]

                                    if not batch_values:
                                        continue

                                    if isinstance(
                                        batch_values[0],
                                        torch.Tensor,
                                    ):
                                        x = torch.stack(
                                            batch_values,
                                            dim=0,
                                        )
                                    else:
                                        x = torch.as_tensor(
                                            np.stack(
                                                batch_values,
                                                axis=0,
                                            )
                                        )

                                elif isinstance(
                                    data,
                                    torch.Tensor,
                                ):
                                    x = data[start:stop]

                                else:
                                    x = torch.as_tensor(
                                        np.asarray(
                                            data[start:stop]
                                        )
                                    )

                                x = x.to(
                                    target_device,
                                    dtype=torch.float32,
                                )

                                model_output = model(x)

                                z_parts.append(
                                    model_output
                                    .latentspace
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )

                                if (
                                    model_output.latent_mean
                                    is None
                                ):
                                    raise ValueError(
                                        f"Model for {modality!r} "
                                        "did not return latent_mean."
                                    )

                                mu_parts.append(
                                    model_output
                                    .latent_mean
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )

                                if (
                                    model_output.latent_logvar
                                    is None
                                ):
                                    raise ValueError(
                                        f"Model for {modality!r} "
                                        "did not return latent_logvar."
                                    )

                                logvar_parts.append(
                                    model_output
                                    .latent_logvar
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )

                                del x
                                del model_output

                finally:
                    model.to(original_device)
                    model.train(was_training)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    gc.collect()

                z_full = np.concatenate(
                    z_parts,
                    axis=0,
                )

                mu_full = np.concatenate(
                    mu_parts,
                    axis=0,
                )

                logvar_full = np.concatenate(
                    logvar_parts,
                    axis=0,
                )

                if z_full.shape[0] != expected_n:
                    raise RuntimeError(
                        f"Recomputed {z_full.shape[0]} latent "
                        f"vectors for {split_name}/{modality}, "
                        f"expected {expected_n}."
                    )

                z_out[modality] = z_full
                mu_out[modality] = mu_full
                logvar_out[modality] = logvar_full
                ids_out[modality] = expected_ids.copy()

                diagnostics.append(
                    {
                        "split": split_name,
                        "modality": modality,
                        "expected_samples": expected_n,
                        "stored_samples": stored_n,
                        "final_samples": z_full.shape[0],
                        "source": "recomputed",
                    }
                )


            # Replace the final representations for this split.

            result.latentspaces.add(
                epoch=final_epoch,
                split=split_name,
                data=z_out,
            )

            result.mus.add(
                epoch=final_epoch,
                split=split_name,
                data=mu_out,
            )

            result.sigmas.add(
                epoch=final_epoch,
                split=split_name,
                data=logvar_out,
            )

            try:
                existing_sample_ids = result.sample_ids.get(
                    epoch=-1,
                    split=split_name,
                )
            except Exception:
                existing_sample_ids = None

            merged_sample_ids = {}

            if isinstance(existing_sample_ids, dict):
                merged_sample_ids.update(
                    {
                        key: np.asarray(value).copy()
                        for key, value in existing_sample_ids.items()
                    }
                )

            # Refreshed modality IDs take precedence over the old
            # modality-specific entries.
            merged_sample_ids.update(ids_out)


            result.sample_ids.add(
                epoch=final_epoch,
                split=split_name,
                data=merged_sample_ids,
            )

            # result.sample_ids.add(
            #     epoch=final_epoch,
            #     split=split_name,
            #     data=ids_out,
            # )
            
            if isinstance(existing_sample_ids, dict):

                lost_keys = (
                    set(existing_sample_ids.keys())
                    - set(merged_sample_ids.keys())
                )

                if lost_keys:
                    raise RuntimeError(
                        "Refreshing final representations unexpectedly "
                        "removed sample-ID entries: "
                        f"{sorted(lost_keys)}"
                    )

        diagnostics_df = pd.DataFrame(diagnostics)

        # Final hard validation.
        for _, row in diagnostics_df.iterrows():
            if row["final_samples"] != row["expected_samples"]:
                raise RuntimeError(
                    "Final XModalix3D representation refresh "
                    "did not recover every sample."
                )

        return diagnostics_df
    
    
    @staticmethod
    def _load_input_for_ml(
        task,
        dataset,
        result,
        n_downsample=None,
        reference_reducer=None,
    ) -> pd.DataFrame:
        """Load XModalix3D representations for downstream ML evaluation.

        This implementation avoids converting complete
        volumetric datasets to very large pandas DataFrames.

        PCA and UMAP reference representations are fitted separately for each
        modality using the training split only. The fitted reducer is then used
        to transform train, validation, and test data.

        Args:
            task:
                Expanded XModalix task name
            dataset:
                DatasetContainer containing multimodal train/valid/test splits.
            result:
                XModalix3D Result object.
            n_downsample:
                Accepted for compatibility with GeneralEvaluator.evaluate().
                Downsampling itself is performed later by GeneralEvaluator.
            reference_reducer:
                Optional dictionary of pre-fitted reducers. For multimodal
                XModalix3D evaluation, reducers should be keyed by the complete
                expanded task name

        Returns:
            DataFrame containing the requested representation for all available
            splits.
        """

        if "_$_" not in task:
            raise ValueError(
                f"Expected an expanded XModalix task containing '_$_', "
                f"but received {task!r}."
            )

        task_type, modality = task.split("_$_", 1)

        # Learned latent representation
        if task_type == "Latent":
            latent_dfs = []

            for split in ("train", "valid", "test"):
                split_dataset = getattr(dataset, split, None)

                if split_dataset is None:
                    continue

                latent_dfs.append(
                    result.get_latent_df(
                        epoch=-1,
                        split=split,
                        modality=modality,
                    )
                )

            if not latent_dfs:
                raise ValueError(
                    f"No latent representations found for modality {modality!r}."
                )

            return pd.concat(latent_dfs, axis=0)

        # Reference representations
        if task_type not in ("PCA", "UMAP"):
            raise ValueError(
                f"XModalix3D downstream evaluation currently supports "
                f"'Latent', 'PCA', and 'UMAP', but received {task_type!r}."
            )

        if dataset.train is None:
            raise ValueError(
                "A training split is required to fit PCA/UMAP reference representations."
            )

        latent_dim = result.get_latent_df(
            epoch=-1,
            split="train",
            modality=modality,
        ).shape[1]

        def _get_matrix(split_dataset):
            """Return one modality as a 2D float32 array plus sample IDs."""

            if split_dataset is None:
                return None, None

            if modality not in split_dataset.datasets:
                raise ValueError(
                    f"Modality {modality!r} not found in dataset split. "
                    f"Available modalities: {list(split_dataset.datasets.keys())}"
                )

            modality_dataset = split_dataset.datasets[modality]
            data = modality_dataset.data

            # 3D image data are stored as List[Tensor].
            if isinstance(data, list):
                if len(data) == 0:
                    raise ValueError(
                        f"No samples found for modality {modality!r}."
                    )

                first = data[0]

                if isinstance(first, torch.Tensor):
                    first_flat = (
                        first.detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )
                else:
                    first_flat = np.asarray(first).reshape(-1)

                X = np.empty(
                    (len(data), first_flat.size),
                    dtype=np.float32,
                )
                X[0] = first_flat

                for i in range(1, len(data)):
                    sample = data[i]

                    if isinstance(sample, torch.Tensor):
                        sample = (
                            sample.detach()
                            .cpu()
                            .numpy()
                        )
                    else:
                        sample = np.asarray(sample)

                    X[i] = sample.reshape(-1)

            # Numeric/tensor-based modalities.
            elif isinstance(data, torch.Tensor):
                X = (
                    data.detach()
                    .cpu()
                    .numpy()
                    .reshape(data.shape[0], -1)
                    .astype(np.float32, copy=False)
                )

            else:
                X = np.asarray(data)

                if X.ndim < 2:
                    raise ValueError(
                        f"Expected at least 2 dimensions for modality "
                        f"{modality!r}, got shape {X.shape}."
                    )

                X = X.reshape(X.shape[0], -1)
                X = X.astype(np.float32, copy=False)

            sample_ids = np.asarray(modality_dataset.sample_ids)

            if X.shape[0] != len(sample_ids):
                raise ValueError(
                    f"Number of samples differs from number of sample IDs "
                    f"for modality {modality!r}: "
                    f"{X.shape[0]} vs {len(sample_ids)}."
                )

            return X, sample_ids

        # Fit reducer on training data only
        X_train, train_ids = _get_matrix(dataset.train)

        if X_train is None:
            raise ValueError("Training data could not be loaded.")

        reference_reducer = reference_reducer or {}

        # For multimodal data, use the complete expanded task name as key
        reducer = reference_reducer.get(task)

        if reducer is None:
            if task_type == "PCA":
                reducer = PCA(
                    n_components=latent_dim,
                    svd_solver="randomized",
                    random_state=42,
                )

            elif task_type == "UMAP":
                reducer = UMAP(
                    n_components=latent_dim,
                    random_state=42,
                )

            X_train_reduced = reducer.fit_transform(X_train)

        else:
            X_train_reduced = reducer.transform(X_train)

        reduced_dfs = [
            pd.DataFrame(
                X_train_reduced,
                index=train_ids,
            )
        ]

        del X_train
        del X_train_reduced
        gc.collect()

        # Transform validation and test data separately
        for split_name in ("valid", "test"):
            split_dataset = getattr(dataset, split_name, None)

            if split_dataset is None:
                continue

            X_split, sample_ids = _get_matrix(split_dataset)

            if X_split is None:
                continue

            X_reduced = reducer.transform(X_split)

            reduced_dfs.append(
                pd.DataFrame(
                    X_reduced,
                    index=sample_ids,
                )
            )

            del X_split
            del X_reduced
            gc.collect()

        del reducer
        gc.collect()

        return pd.concat(reduced_dfs, axis=0)

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        """Convert tensors/array-like objects to a NumPy array."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _get_representation_dict(
        result: Result,
        *,
        representation: RepresentationType,
        split: str,
        epoch: int,
    ) -> Dict[str, np.ndarray]:
        """Retrieve modality-specific latent representations from a Result object."""
        if representation == "z":
            store = result.latentspaces
        elif representation == "mu":
            store = result.mus
        else:
            raise ValueError(
                f"Unsupported representation '{representation}'. Use 'mu' or 'z'."
            )

        data = store.get(epoch=epoch, split=split)

        if not isinstance(data, dict) or len(data) == 0:
            raise ValueError(
                f"No '{representation}' representations found for split='{split}', "
                f"epoch={epoch}. The requested dynamics may not have been stored."
            )

        return {
            modality: XModalix3DEvaluator._to_numpy(values)
            for modality, values in data.items()
        }

    @staticmethod
    def _get_sample_id_dict(
        result: Result,
        *,
        split: str,
        epoch: int,
    ) -> Dict[str, np.ndarray]:
        """Retrieve modality-specific sample IDs from a Result object."""
        data = result.sample_ids.get(epoch=epoch, split=split)

        if not isinstance(data, dict) or len(data) == 0:
            raise ValueError(
                f"No sample IDs found for split='{split}', epoch={epoch}. "
                "Sample IDs are required for grouped probing and matched/unmatched analysis."
            )

        return {
            modality: np.asarray(values).astype(str)
            for modality, values in data.items()
        }

    @staticmethod
    def _validate_representation_and_ids(
        representation_dict: Dict[str, np.ndarray],
        sample_id_dict: Dict[str, np.ndarray],
    ) -> None:
        """Validate modality keys, shapes, and sample-ID correspondence."""
        representation_keys = set(representation_dict)
        sample_id_keys = set(sample_id_dict)

        missing_ids = representation_keys - sample_id_keys
        if missing_ids:
            raise ValueError(
                "Missing sample IDs for modality/modalities: "
                + ", ".join(sorted(missing_ids))
            )

        if len(representation_keys) < 2:
            raise ValueError(
                "At least two modalities are required for cross-modal alignment evaluation."
            )

        latent_dimensions = set()

        for modality in sorted(representation_keys):
            embeddings = np.asarray(representation_dict[modality])
            sample_ids = np.asarray(sample_id_dict[modality]).astype(str)

            if embeddings.ndim != 2:
                raise ValueError(
                    f"Expected a 2D embedding matrix for modality '{modality}', "
                    f"got shape {embeddings.shape}."
                )

            latent_dimensions.add(embeddings.shape[1])

            if not np.isfinite(embeddings).all():
                raise ValueError(
                    f"Non-finite values found in embeddings for modality '{modality}'."
                )

            if len(embeddings) != len(sample_ids):
                raise ValueError(
                    f"Number of embeddings and sample IDs differs for modality "
                    f"'{modality}': {len(embeddings)} embeddings versus "
                    f"{len(sample_ids)} sample IDs."
                )

            if len(np.unique(sample_ids)) != len(sample_ids):
                raise ValueError(
                    f"Duplicate sample IDs found for modality '{modality}'. "
                    "Alignment evaluation requires one embedding per sample and modality."
                )

        if len(latent_dimensions) != 1:
            raise ValueError(
                "All modalities must have the same latent dimensionality for cross-modal "
                f"alignment evaluation, found dimensions {sorted(latent_dimensions)}."
            )
    
    @staticmethod
    def _validate_dataset_coverage(
        result: Result,
        sample_id_dict,
        split: str,
        epoch: int,
    ) -> None:
        """Ensure stored representations cover the complete dataset split."""

        datasets = result.datasets

        if datasets is None:
            return

        split_dataset = getattr(
            datasets,
            split,
            None,
        )

        if split_dataset is None:
            return

        for modality, actual_ids in sample_id_dict.items():

            if modality not in split_dataset.datasets:
                continue

            expected_ids = np.asarray(
                split_dataset.datasets[
                    modality
                ].sample_ids
            )

            actual_ids = np.asarray(actual_ids)

            complete = (
                len(actual_ids) == len(expected_ids)
                and len(np.unique(actual_ids))
                == len(expected_ids)
                and set(actual_ids)
                == set(expected_ids)
            )

            if not complete:
                raise ValueError(
                    f"Incomplete stored dynamics for "
                    f"{split}/{modality} at epoch {epoch}: "
                    f"{len(actual_ids)} of "
                    f"{len(expected_ids)} samples. "
                    "This checkpoint must not be used for "
                    "alignment evaluation."
                )

    @staticmethod
    def modality_linear_probe(
        result: Result,
        *,
        representation: RepresentationType = "z",
        split: str = "valid",
        epoch: int = -1,
        n_splits: int = 5,
        C: float = 1.0,
        max_iter: int = 5000,
        random_state: int = 0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluate modality information using a linear probe.

        A logistic-regression classifier is trained on frozen embeddings to predict
        the modality of origin. Cross-validation is grouped by sample/patient ID so
        embeddings from the same patient cannot occur in both training and validation
        folds. Standardization is fitted independently inside every fold.

        Lower balanced accuracy indicates stronger modality invariance. For K
        modalities, the balanced-accuracy chance level is 1 / K.

        Parameters
        ----------
        result:
            XModalix/XModalix3D Result object containing latent representations and
            sample IDs.
        representation:
            ``"z"`` for sampled latent codes or ``"mu"`` for posterior means.
        split:
            Result split to evaluate, typically ``"valid"`` for model development or
            ``"test"`` for final reporting.
        epoch:
            Stored epoch to evaluate. ``-1`` follows the Result convention for the
            final stored epoch.
        n_splits:
            Number of grouped cross-validation folds. Automatically reduced if fewer
            unique sample IDs are available.
        C:
            Inverse regularization strength of logistic regression. The same value
            should be used when comparing different XModalix3D runs.
        max_iter:
            Maximum logistic-regression iterations per fold.
        random_state:
            Random state passed to logistic regression.

        Returns
        -------
        summary_df, confusion_df:
            ``summary_df`` contains one row with balanced probe accuracy, chance level,
            number of modalities, number of embeddings, number of unique patients,
            split, epoch and representation. ``confusion_df`` contains a row-normalized
            confusion matrix whose rows are true modalities and columns are predicted
            modalities.
        """
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2.")
        if C <= 0:
            raise ValueError("C must be > 0.")

        representation_dict = XModalix3DEvaluator._get_representation_dict(
            result,
            representation=representation,
            split=split,
            epoch=epoch,
        )
        sample_id_dict = XModalix3DEvaluator._get_sample_id_dict(
            result,
            split=split,
            epoch=epoch,
        )
        XModalix3DEvaluator._validate_representation_and_ids(
            representation_dict,
            sample_id_dict,
        )
        
        XModalix3DEvaluator._validate_dataset_coverage(
            result=result,
            sample_id_dict=sample_id_dict,
            split=split,
            epoch=epoch,
        )

        X_parts = []
        y_parts = []
        group_parts = []

        for modality in sorted(representation_dict):
            embeddings = np.asarray(representation_dict[modality])
            sample_ids = np.asarray(sample_id_dict[modality]).astype(str)

            X_parts.append(embeddings)
            y_parts.append(np.repeat(modality, len(embeddings)))
            group_parts.append(sample_ids)

        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        groups = np.concatenate(group_parts, axis=0)

        modalities = np.asarray(sorted(np.unique(y)))
        unique_groups = np.unique(groups)

        n_cv_splits = min(n_splits, len(unique_groups))
        if n_cv_splits < 2:
            raise ValueError(
                "At least two unique sample/patient IDs are required for grouped "
                "cross-validation."
            )

        cv = GroupKFold(n_splits=n_cv_splits)
        predictions = np.empty(len(y), dtype=object)

        for train_idx, valid_idx in cv.split(X, y, groups=groups):
            y_train = y[train_idx]

            if len(np.unique(y_train)) < len(modalities):
                raise ValueError(
                    "A grouped CV training fold does not contain all modalities. "
                    "This can occur with strongly unpaired data. Reduce n_splits or "
                    "evaluate a split with better modality coverage."
                )

            probe = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            )

            probe.fit(X[train_idx], y_train)
            predictions[valid_idx] = probe.predict(X[valid_idx])

        score = balanced_accuracy_score(y, predictions)
        chance = 1.0 / len(modalities)

        cm = confusion_matrix(
            y,
            predictions,
            labels=modalities,
            normalize="true",
        )
        confusion_df = pd.DataFrame(
            cm,
            index=pd.Index(modalities, name="true_modality"),
            columns=pd.Index(modalities, name="predicted_modality"),
        )

        summary_df = pd.DataFrame(
            [
                {
                    "split": split,
                    "epoch": epoch,
                    "representation": representation,
                    "linear_probe_balanced_accuracy": float(score),
                    "chance_level": float(chance),
                    "n_modalities": int(len(modalities)),
                    "n_embeddings": int(len(y)),
                    "n_unique_samples": int(len(unique_groups)),
                    "n_cv_splits": int(n_cv_splits),
                }
            ]
        )

        return summary_df, confusion_df

    @staticmethod
    def matched_unmatched_analysis(
        result: Result,
        *,
        representation: RepresentationType = "z",
        split: str = "valid",
        epoch: int = -1,
        distance_metric: str = "euclidean",
        return_distances: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Compare matched and unmatched cross-modal latent distances.

        For every pair of modalities, embeddings are first aligned by sample/patient
        ID. Distances between the two embeddings of the same patient are treated as
        *matched* distances. All off-diagonal cross-patient distances are treated as
        *unmatched* distances.

        The main summary statistic is

            matched_unmatched_ratio = mean(matched distance) / mean(unmatched distance)

        Values below 1 indicate that corresponding patients are closer across
        modalities than unrelated patients. Smaller values indicate stronger
        patient-specific cross-modal alignment.

        Parameters
        ----------
        result:
            XModalix/XModalix3D Result object containing latent representations and
            sample IDs.
        representation:
            ``"z"`` for sampled latent codes or ``"mu"`` for posterior means.
        split:
            Result split to evaluate.
        epoch:
            Stored epoch to evaluate. ``-1`` follows the Result convention for the
            final stored epoch.
        distance_metric:
            Any metric accepted by ``sklearn.metrics.pairwise_distances``. The default
            ``"euclidean"`` reproduces the analysis used in the XModalix3D alignment
            notebooks.
        return_distances:
            If ``True``, also return a long-format DataFrame containing the individual
            matched and unmatched distances. This is useful for violin/box plots but
            can be much larger than the summary table.

        Returns
        -------
        summary_df or (summary_df, distances_df):
            The summary contains one row per modality pair, including the number of
            paired samples, mean/median matched distance, mean/median unmatched
            distance, and the matched/unmatched ratio. If ``return_distances=True``,
            a long-format distance table is returned as the second object.
        """
        representation_dict = XModalix3DEvaluator._get_representation_dict(
            result,
            representation=representation,
            split=split,
            epoch=epoch,
        )
        sample_id_dict = XModalix3DEvaluator._get_sample_id_dict(
            result,
            split=split,
            epoch=epoch,
        )
        XModalix3DEvaluator._validate_representation_and_ids(
            representation_dict,
            sample_id_dict,
        )

        summary_rows = []
        distance_rows = []

        modalities = sorted(representation_dict)

        for modality_a, modality_b in combinations(modalities, 2):
            ids_a = np.asarray(sample_id_dict[modality_a]).astype(str)
            ids_b = np.asarray(sample_id_dict[modality_b]).astype(str)
            embeddings_a = np.asarray(representation_dict[modality_a])
            embeddings_b = np.asarray(representation_dict[modality_b])

            index_a = {sample_id: idx for idx, sample_id in enumerate(ids_a)}
            index_b = {sample_id: idx for idx, sample_id in enumerate(ids_b)}

            common_ids = sorted(set(ids_a).intersection(ids_b))

            if len(common_ids) < 2:
                raise ValueError(
                    f"Need at least two paired samples for modalities '{modality_a}' "
                    f"and '{modality_b}', found {len(common_ids)}."
                )

            A = np.stack([embeddings_a[index_a[sid]] for sid in common_ids])
            B = np.stack([embeddings_b[index_b[sid]] for sid in common_ids])

            distance_matrix = pairwise_distances(
                A,
                B,
                metric=distance_metric,
            )

            matched_distances = np.diag(distance_matrix)
            unmatched_mask = ~np.eye(len(common_ids), dtype=bool)
            unmatched_distances = distance_matrix[unmatched_mask]

            matched_mean = float(np.mean(matched_distances))
            unmatched_mean = float(np.mean(unmatched_distances))

            ratio = (
                matched_mean / unmatched_mean
                if unmatched_mean > 0
                else np.nan
            )

            summary_rows.append(
                {
                    "split": split,
                    "epoch": epoch,
                    "representation": representation,
                    "distance_metric": distance_metric,
                    "modality_a": modality_a,
                    "modality_b": modality_b,
                    "n_pairs": int(len(common_ids)),
                    "matched_distance_mean": matched_mean,
                    "matched_distance_median": float(np.median(matched_distances)),
                    "unmatched_distance_mean": unmatched_mean,
                    "unmatched_distance_median": float(np.median(unmatched_distances)),
                    "matched_unmatched_ratio": float(ratio),
                }
            )

            if return_distances:
                for sample_id, distance in zip(common_ids, matched_distances):
                    distance_rows.append(
                        {
                            "split": split,
                            "epoch": epoch,
                            "representation": representation,
                            "distance_metric": distance_metric,
                            "modality_a": modality_a,
                            "modality_b": modality_b,
                            "pair_type": "matched",
                            "sample_id_a": sample_id,
                            "sample_id_b": sample_id,
                            "distance": float(distance),
                        }
                    )

                row_idx, col_idx = np.where(unmatched_mask)
                for i, j in zip(row_idx, col_idx):
                    distance_rows.append(
                        {
                            "split": split,
                            "epoch": epoch,
                            "representation": representation,
                            "distance_metric": distance_metric,
                            "modality_a": modality_a,
                            "modality_b": modality_b,
                            "pair_type": "unmatched",
                            "sample_id_a": common_ids[i],
                            "sample_id_b": common_ids[j],
                            "distance": float(distance_matrix[i, j]),
                        }
                    )

        summary_df = pd.DataFrame(summary_rows)

        if not return_distances:
            return summary_df

        distances_df = pd.DataFrame(distance_rows)
        return summary_df, distances_df
