from __future__ import annotations

from itertools import combinations
from typing import Dict, Literal, Tuple, Union

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, pairwise_distances
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
