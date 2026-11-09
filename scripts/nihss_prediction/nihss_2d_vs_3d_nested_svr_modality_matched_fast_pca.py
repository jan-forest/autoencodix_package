#!/usr/bin/env python3
"""
Nested-CV NIHSS prediction: concatenated 2D Imagix embeddings vs Imagix3D embeddings,
including a dimension-matched PCA sensitivity analysis.

Primary analysis
----------------
For each supplied imaging modality, the script compares:
    * 2D: full concatenated patient-wise 2D embedding
    * 3D: patient-wise Imagix3D embedding

The same patient cohort and the same outer fold assignments are used for every
modality and representation. Within each outer training fold, C and epsilon of a
linear epsilon-SVR are selected by inner CV using mean absolute error.
Standardization is fitted only within the corresponding training data.

Primary statistical comparison
------------------------------
For each patient, absolute OOF prediction errors are averaged across modalities
separately for 2D and 3D. These paired patient-level mean errors are compared with
a Wilcoxon signed-rank test.

Dimension-matched sensitivity analysis
--------------------------------------
For each modality, the concatenated 2D embedding is additionally reduced by PCA
to exactly the dimensionality of the corresponding 3D embedding. The target
dimensionality is inferred from the number of LatDim_* columns in that modality's
3D embedding CSV. The leakage-safe pipeline is:

    StandardScaler -> PCA(d_m) -> linear SVR

where d_m is the modality-specific 3D latent dimensionality. PCA is therefore
fitted only on the training data inside each inner/outer CV fit. The same
nested-CV and statistical procedures are then repeated for dimension-matched
2D versus 3D.

Secondary analyses
------------------
Modality-specific paired Wilcoxon tests are saved for both the full-dimensional
2D-vs-3D comparison and the PCA-reduced-2D-vs-3D sensitivity comparison.
Both Benjamini-Hochberg and Holm adjusted p-values are reported across the
modality-specific tests within each comparison family.

Expected 2D file columns:
    patient_id, split, n_slices, latent_dim_per_slice, vector_length,
    feature_0001, feature_0002, ...

Expected 3D file columns:
    sample_id, split, LatDim_0, LatDim_1, ...

Expected clinical file columns:
    patients_id, ..., nihss_discharge, ..., custom_splits
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


DEFAULT_C_VALUES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
DEFAULT_EPSILON_VALUES = (0.1, 0.5, 1.0, 2.0)
SPLIT_NAMES = {"train", "valid", "test"}


@dataclass
class EmbeddingTable:
    modality: str
    representation: str
    path: Path
    ids: np.ndarray
    X: np.ndarray
    feature_names: List[str]
    splits: Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Nested patient-wise linear-SVR comparison of concatenated 2D "
            "Imagix embeddings with Imagix3D embeddings, including PCA sensitivity."
        )
    )
    parser.add_argument(
        "--embedding-2d",
        action="append",
        required=True,
        metavar="MODALITY=PATH",
        help=(
            "2D embedding CSV or its containing directory. Repeat once per modality. "
            "Example: CBF=C:/data/cbf/imagix_cbf_patientwise_z.csv"
        ),
    )
    parser.add_argument(
        "--embedding-3d",
        action="append",
        required=True,
        metavar="MODALITY=PATH",
        help=(
            "3D embedding CSV or its containing directory. Repeat once per modality. "
            "Example: CBF=C:/data/cbf/results/imagix3d_final_z_all_splits.csv"
        ),
    )
    parser.add_argument(
        "--clinical",
        type=Path,
        default=None,
        help=(
            "Path to sample_characteristics.csv. If omitted, the script searches "
            "<script_dir>/nihss_prediction/ and <script_dir>/nihss_prediciton/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: a timestamped directory under "
            "<script_dir>/nihss_prediction/."
        ),
    )
    parser.add_argument("--target-col", default="nihss_discharge")
    parser.add_argument("--clinical-id-col", default="patients_id")
    parser.add_argument("--clinical-split-col", default="custom_splits")
    parser.add_argument("--outer-folds", type=int, default=10)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help=(
            "Optional common PCA dimensionality override. By default, the PCA "
            "target is inferred separately for each modality from the number of "
            "LatDim_* columns in its 3D embedding file. For a true dimension-matched "
            "analysis, an override must equal every supplied 3D dimensionality."
        ),
    )
    parser.add_argument(
        "--skip-pca-sensitivity",
        action="store_true",
        help="Skip the PCA-reduced 2D sensitivity analysis.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for GridSearchCV. Use -1 for all available CPU cores.",
    )
    parser.add_argument(
        "--wilcoxon-alternative",
        choices=("two-sided", "greater", "less"),
        default="two-sided",
        help=(
            "Alternative for wilcoxon(error_2d, error_3d). "
            "'greater' tests whether 2D errors tend to be larger than 3D errors."
        ),
    )
    parser.add_argument(
        "--allow-vae-split-mismatch",
        action="store_true",
        help=(
            "Do not abort if embedding split labels disagree with clinical "
            "custom_splits. Normally this should NOT be used."
        ),
    )
    return parser.parse_args()


def _parse_mapping(values: Sequence[str], argument_name: str) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"{argument_name} must use MODALITY=PATH syntax, got {value!r}."
            )
        modality, raw_path = value.split("=", 1)
        modality = modality.strip().upper()
        raw_path = raw_path.strip()
        if not modality or not raw_path:
            raise ValueError(
                f"{argument_name} must use non-empty MODALITY=PATH syntax."
            )
        if modality in result:
            raise ValueError(
                f"Duplicate modality {modality!r} supplied for {argument_name}."
            )
        result[modality] = Path(raw_path)
    return result


def _resolve_embedding_path(path: Path, modality: str, representation: str) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Embedding path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Embedding path is neither a file nor a directory: {path}")

    if representation == "2D":
        expected = path / f"imagix_{modality.lower()}_patientwise_z.csv"
    elif representation == "3D":
        expected = path / "imagix3d_final_z_all_splits.csv"
    else:
        raise ValueError(f"Unknown representation: {representation}")

    if not expected.is_file():
        raise FileNotFoundError(
            f"Expected {representation} embedding file was not found: {expected}"
        )
    return expected


def _find_default_clinical(script_dir: Path) -> Path:
    candidates = [
        script_dir / "nihss_prediction" / "sample_characteristics.csv",
        script_dir / "nihss_prediciton" / "sample_characteristics.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find sample_characteristics.csv automatically. "
        "Pass it explicitly with --clinical. Checked:\n  "
        + "\n  ".join(str(x) for x in candidates)
    )


def _normalise_ids(series: pd.Series) -> pd.Series:
    ids = series.astype("string").str.strip()
    if ids.isna().any() or (ids == "").any():
        raise ValueError("Missing or empty patient/sample IDs were found.")
    return ids.astype(str)


def _suffix_number(name: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", str(name))
    return (int(match.group(1)) if match else 10**12, str(name))


def _validate_unique_ids(ids: pd.Series, label: str) -> None:
    duplicated = ids[ids.duplicated(keep=False)]
    if not duplicated.empty:
        examples = duplicated.drop_duplicates().tolist()[:10]
        raise ValueError(
            f"{label} contains duplicate patient IDs. Examples: {examples}"
        )


def load_clinical(
    path: Path,
    id_col: str,
    target_col: str,
    split_col: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Clinical file does not exist: {path}")

    df = pd.read_csv(path, dtype={id_col: "string"})
    missing = {id_col, target_col} - set(df.columns)
    if missing:
        raise ValueError(
            f"Clinical file is missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df[id_col] = _normalise_ids(df[id_col])
    _validate_unique_ids(df[id_col], "Clinical table")

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    n_missing_target = int(df[target_col].isna().sum())
    if n_missing_target:
        print(
            f"Clinical table: excluding {n_missing_target} patient(s) with missing "
            f"{target_col}.",
            flush=True,
        )
        df = df.loc[df[target_col].notna()].copy()

    if split_col in df.columns:
        df[split_col] = df[split_col].astype("string").str.strip().str.lower()
        invalid = set(df[split_col].dropna().unique()) - SPLIT_NAMES
        if invalid:
            raise ValueError(
                f"Clinical split column {split_col!r} has invalid labels: "
                f"{sorted(invalid)}"
            )

    return df.set_index(id_col, drop=False)


def load_embedding_table(
    path: Path,
    modality: str,
    representation: str,
) -> EmbeddingTable:
    if representation == "2D":
        id_col = "patient_id"
        prefix = "feature_"
    elif representation == "3D":
        id_col = "sample_id"
        prefix = "LatDim_"
    else:
        raise ValueError(f"Unknown representation: {representation}")

    df = pd.read_csv(path, dtype={id_col: "string"})
    if id_col not in df.columns:
        raise ValueError(
            f"{representation} {modality}: expected ID column {id_col!r} in {path}"
        )

    df[id_col] = _normalise_ids(df[id_col])
    _validate_unique_ids(df[id_col], f"{representation} {modality}")

    feature_cols = [
        col for col in df.columns
        if str(col).lower().startswith(prefix.lower())
    ]
    feature_cols = sorted(feature_cols, key=_suffix_number)

    if not feature_cols:
        raise ValueError(
            f"{representation} {modality}: no feature columns beginning with "
            f"{prefix!r} found in {path}"
        )

    X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    if X_df.isna().any().any():
        bad_cols = X_df.columns[X_df.isna().any()].tolist()[:10]
        raise ValueError(
            f"{representation} {modality}: NaN/non-numeric embedding values found. "
            f"First affected columns: {bad_cols}"
        )

    X = X_df.to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError(
            f"{representation} {modality}: non-finite embedding values found."
        )

    if representation == "2D" and "vector_length" in df.columns:
        vector_length = pd.to_numeric(df["vector_length"], errors="coerce")
        inconsistent = vector_length.notna() & (vector_length != len(feature_cols))
        if inconsistent.any():
            examples = df.loc[inconsistent, id_col].tolist()[:10]
            raise ValueError(
                f"2D {modality}: vector_length does not equal the number of "
                f"feature columns ({len(feature_cols)}) for patients: {examples}"
            )

    splits: Dict[str, str] = {}
    if "split" in df.columns:
        split_values = df["split"].astype("string").str.strip().str.lower()
        invalid = set(split_values.dropna().unique()) - SPLIT_NAMES
        if invalid:
            raise ValueError(
                f"{representation} {modality}: invalid split labels: "
                f"{sorted(invalid)}"
            )
        splits = {
            sid: str(split_value)
            for sid, split_value in zip(df[id_col].astype(str), split_values)
            if pd.notna(split_value)
        }

    return EmbeddingTable(
        modality=modality,
        representation=representation,
        path=path,
        ids=df[id_col].astype(str).to_numpy(),
        X=X,
        feature_names=[str(col) for col in feature_cols],
        splits=splits,
    )


def _check_vae_split_labels(
    clinical: pd.DataFrame,
    clinical_split_col: str,
    tables: Iterable[EmbeddingTable],
    allow_mismatch: bool,
) -> None:
    if clinical_split_col not in clinical.columns:
        print(
            f"WARNING: clinical split column {clinical_split_col!r} is absent; "
            "embedding split labels cannot be cross-checked.",
            flush=True,
        )
        return

    clinical_split = clinical[clinical_split_col].to_dict()
    mismatches: List[Tuple[str, str, str, str]] = []

    for table in tables:
        for sid, observed_split in table.splits.items():
            expected_split = clinical_split.get(sid)
            if expected_split is None or pd.isna(expected_split):
                continue
            expected_split = str(expected_split).strip().lower()
            if observed_split != expected_split:
                mismatches.append(
                    (
                        table.modality,
                        table.representation,
                        sid,
                        f"{observed_split} != {expected_split}",
                    )
                )

    if mismatches:
        message = (
            "Embedding split labels disagree with clinical custom_splits. "
            f"Number of mismatches: {len(mismatches)}. "
            f"First examples: {mismatches[:10]}"
        )
        if allow_mismatch:
            print("WARNING: " + message, flush=True)
        else:
            raise ValueError(message)


def _subset_matrix(table: EmbeddingTable, patient_ids: Sequence[str]) -> np.ndarray:
    id_to_row = {sid: idx for idx, sid in enumerate(table.ids)}
    missing = [sid for sid in patient_ids if sid not in id_to_row]
    if missing:
        raise ValueError(
            f"{table.representation} {table.modality}: cohort IDs unexpectedly "
            f"missing. Examples: {missing[:10]}"
        )
    return np.vstack([table.X[id_to_row[sid]] for sid in patient_ids])


def _benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.asarray([], dtype=float)

    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = ranked * n / np.arange(1, n + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)

    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_ranked
    return adjusted


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, preserving input order."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.asarray([], dtype=float)

    order = np.argsort(p)
    ranked = p[order]
    multipliers = n - np.arange(n)
    adjusted_ranked = ranked * multipliers
    adjusted_ranked = np.maximum.accumulate(adjusted_ranked)
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)

    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_ranked
    return adjusted


def _safe_wilcoxon(
    x: Sequence[float],
    y: Sequence[float],
    alternative: str,
) -> Tuple[float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if np.allclose(x_arr - y_arr, 0.0):
        return 0.0, 1.0

    result = wilcoxon(
        x_arr,
        y_arr,
        alternative=alternative,
        zero_method="wilcox",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def _make_pipeline(pca_components: Optional[int]) -> Pipeline:
    """Create the standard non-PCA SVR pipeline.

    The PCA sensitivity branch is handled explicitly in ``run_nested_svr`` so
    that scaler/PCA transforms are computed only once per inner split and can
    be reused across all SVR hyperparameter candidates.
    """
    steps = [("scaler", StandardScaler())]
    if pca_components is not None:
        steps.append(("pca", PCA(n_components=pca_components, svd_solver="auto")))
    steps.append(("svr", SVR(kernel="linear")))
    return Pipeline(steps=steps)


def _score_svr_candidate_on_precomputed_pca(
    C: float,
    epsilon: float,
    transformed_inner_splits: Sequence[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
) -> Tuple[float, float, float]:
    """Evaluate one SVR candidate on already scaled/PCA-transformed inner folds.

    Returns
    -------
    C, epsilon, mean_mae
        The candidate values and the mean validation MAE across inner folds.
    """
    fold_maes: List[float] = []

    for X_inner_train, y_inner_train, X_inner_valid, y_inner_valid in transformed_inner_splits:
        model = SVR(kernel="linear", C=float(C), epsilon=float(epsilon))
        model.fit(X_inner_train, y_inner_train)
        predicted = model.predict(X_inner_valid)
        fold_maes.append(float(np.mean(np.abs(y_inner_valid - predicted))))

    return float(C), float(epsilon), float(np.mean(fold_maes))


def _run_pca_inner_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    inner_cv: Sequence[Tuple[np.ndarray, np.ndarray]],
    pca_components: int,
    c_values: Sequence[float],
    epsilon_values: Sequence[float],
    n_jobs: int,
) -> Tuple[float, float, float]:
    """Leakage-safe inner-CV search with PCA preprocessing computed once per fold.

    For every inner fold, StandardScaler and PCA are fitted only on that fold's
    training subset. The transformed train/validation matrices are then reused
    for every (C, epsilon) SVR candidate. This is mathematically equivalent to
    fitting ``StandardScaler -> PCA -> SVR`` inside GridSearchCV, but avoids
    refitting the identical scaler/PCA transformation for every SVR candidate.
    """
    transformed_inner_splits: List[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []

    for inner_train_idx, inner_valid_idx in inner_cv:
        X_inner_train_raw = X_train[inner_train_idx]
        X_inner_valid_raw = X_train[inner_valid_idx]
        y_inner_train = y_train[inner_train_idx]
        y_inner_valid = y_train[inner_valid_idx]

        scaler = StandardScaler()
        X_inner_train_scaled = scaler.fit_transform(X_inner_train_raw)
        X_inner_valid_scaled = scaler.transform(X_inner_valid_raw)

        pca = PCA(n_components=pca_components, svd_solver="auto")
        X_inner_train_pca = pca.fit_transform(X_inner_train_scaled)
        X_inner_valid_pca = pca.transform(X_inner_valid_scaled)

        transformed_inner_splits.append(
            (
                X_inner_train_pca,
                y_inner_train,
                X_inner_valid_pca,
                y_inner_valid,
            )
        )

    # Keep candidate ordering identical to the original grid: C first, then epsilon.
    candidates = [
        (float(C), float(epsilon))
        for C in c_values
        for epsilon in epsilon_values
    ]

    scored = Parallel(n_jobs=n_jobs)(
        delayed(_score_svr_candidate_on_precomputed_pca)(
            C=C,
            epsilon=epsilon,
            transformed_inner_splits=transformed_inner_splits,
        )
        for C, epsilon in candidates
    )

    # Python's min is stable: in an exact tie, the first candidate in grid order
    # is selected, matching GridSearchCV's first-best-candidate behavior.
    best_C, best_epsilon, best_mae = min(scored, key=lambda row: row[2])
    return best_C, best_epsilon, best_mae


def run_nested_svr(
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    outer_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    inner_folds: int,
    random_state: int,
    n_jobs: int,
    pca_components: Optional[int] = None,
    c_values: Sequence[float] = DEFAULT_C_VALUES,
    epsilon_values: Sequence[float] = DEFAULT_EPSILON_VALUES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run leakage-safe nested CV and return OOF predictions and tuning details."""

    if len(X) != len(y) or len(y) != len(patient_ids):
        raise ValueError("X, y, and patient_ids must have identical row counts.")

    if pca_components is not None:
        if pca_components < 1:
            raise ValueError("pca_components must be >= 1.")
        if pca_components > X.shape[1]:
            raise ValueError(
                f"PCA requests {pca_components} components but X has only "
                f"{X.shape[1]} features."
            )

    oof_prediction = np.full(len(y), np.nan, dtype=float)
    oof_fold = np.full(len(y), -1, dtype=int)
    tuning_rows: List[dict] = []

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
        X_train = X[outer_train_idx]
        y_train = y[outer_train_idx]
        X_test = X[outer_test_idx]

        if inner_folds > len(outer_train_idx):
            raise ValueError(
                f"inner_folds={inner_folds} exceeds outer-training sample size "
                f"{len(outer_train_idx)}."
            )

        inner_cv = list(
            KFold(
                n_splits=inner_folds,
                shuffle=True,
                random_state=random_state + outer_fold,
            ).split(np.arange(len(outer_train_idx)))
        )

        if pca_components is not None:
            smallest_inner_train = min(len(train_idx) for train_idx, _ in inner_cv)
            max_components = min(X_train.shape[1], smallest_inner_train)
            if pca_components > max_components:
                raise ValueError(
                    f"PCA with {pca_components} components is not possible in outer "
                    f"fold {outer_fold}: the smallest inner-training fold allows at "
                    f"most {max_components} components."
                )

            fold_start = time.perf_counter()
            print(
                f"  PCA nested CV outer fold {outer_fold + 1}/{len(outer_splits)}: "
                f"precomputing {inner_folds} leakage-safe PCA transforms...",
                flush=True,
            )

            best_C, best_epsilon, best_inner_mae = _run_pca_inner_search(
                X_train=X_train,
                y_train=y_train,
                inner_cv=inner_cv,
                pca_components=int(pca_components),
                c_values=c_values,
                epsilon_values=epsilon_values,
                n_jobs=n_jobs,
            )

            # Refit preprocessing and the selected SVR on the complete outer-training
            # fold, then evaluate once on the untouched outer-test fold.
            outer_scaler = StandardScaler()
            X_train_scaled = outer_scaler.fit_transform(X_train)
            X_test_scaled = outer_scaler.transform(X_test)

            outer_pca = PCA(n_components=int(pca_components), svd_solver="auto")
            X_train_pca = outer_pca.fit_transform(X_train_scaled)
            X_test_pca = outer_pca.transform(X_test_scaled)

            best_model = SVR(
                kernel="linear",
                C=float(best_C),
                epsilon=float(best_epsilon),
            )
            best_model.fit(X_train_pca, y_train)
            predicted = best_model.predict(X_test_pca)

            fold_elapsed = time.perf_counter() - fold_start
            print(
                f"  Completed PCA outer fold {outer_fold + 1}/{len(outer_splits)} "
                f"in {fold_elapsed / 60.0:.2f} min "
                f"(best C={best_C:g}, epsilon={best_epsilon:g}, "
                f"inner MAE={best_inner_mae:.4f})",
                flush=True,
            )

            best_params = {
                "svr__C": float(best_C),
                "svr__epsilon": float(best_epsilon),
            }
            best_score_mae = float(best_inner_mae)

        else:
            # Preserve the original GridSearchCV implementation for analyses without PCA.
            search = GridSearchCV(
                estimator=_make_pipeline(pca_components=None),
                param_grid={
                    "svr__C": list(c_values),
                    "svr__epsilon": list(epsilon_values),
                },
                scoring="neg_mean_absolute_error",
                cv=inner_cv,
                refit=True,
                n_jobs=n_jobs,
                return_train_score=False,
            )
            search.fit(X_train, y_train)
            predicted = search.predict(X_test)
            best_params = search.best_params_
            best_score_mae = float(-search.best_score_)

        oof_prediction[outer_test_idx] = predicted
        oof_fold[outer_test_idx] = outer_fold

        tuning_rows.append(
            {
                "outer_fold": outer_fold,
                "n_outer_train": len(outer_train_idx),
                "n_outer_test": len(outer_test_idx),
                "pca_components": pca_components,
                "best_C": float(best_params["svr__C"]),
                "best_epsilon": float(best_params["svr__epsilon"]),
                "best_inner_MAE": best_score_mae,
            }
        )

    if np.isnan(oof_prediction).any() or (oof_fold < 0).any():
        raise RuntimeError(
            "Nested CV failed to generate exactly one OOF prediction per patient."
        )

    prediction_df = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "outer_fold": oof_fold,
            "nihss_observed": y,
            "nihss_predicted": oof_prediction,
            "absolute_error": np.abs(y - oof_prediction),
        }
    )
    return prediction_df, pd.DataFrame(tuning_rows)

def _metric_row(
    modality: str,
    representation: str,
    pred_df: pd.DataFrame,
    n_input_features: int,
    n_model_features: int,
) -> dict:
    return {
        "modality": modality,
        "representation": representation,
        "n_patients": len(pred_df),
        "n_input_features": n_input_features,
        "n_model_features": n_model_features,
        "MedAE": float(
            median_absolute_error(
                pred_df["nihss_observed"],
                pred_df["nihss_predicted"],
            )
        ),
        "R2": float(
            r2_score(
                pred_df["nihss_observed"],
                pred_df["nihss_predicted"],
            )
        ),
    }


def _representation_key(representation: str) -> str:
    return representation.lower().replace("-", "_")


def _add_modality_specific_tests(
    error_wide: pd.DataFrame,
    modalities: Sequence[str],
    rep_a: str,
    rep_b: str,
    alternative: str,
) -> pd.DataFrame:
    rows: List[dict] = []
    raw_p_values: List[float] = []

    key_a = _representation_key(rep_a)
    key_b = _representation_key(rep_b)

    for modality in modalities:
        err_a = error_wide[f"abs_error_{modality}_{key_a}"]
        err_b = error_wide[f"abs_error_{modality}_{key_b}"]
        statistic, p_value = _safe_wilcoxon(err_a, err_b, alternative)
        raw_p_values.append(p_value)
        rows.append(
            {
                "modality": modality,
                "comparison": f"{rep_a} vs {rep_b}",
                "n_patients": len(error_wide),
                f"median_abs_error_{key_a}": float(np.median(err_a)),
                f"median_abs_error_{key_b}": float(np.median(err_b)),
                f"median_difference_{key_a}_minus_{key_b}": float(
                    np.median(err_a - err_b)
                ),
                "wilcoxon_statistic": statistic,
                "p_value_raw": p_value,
            }
        )

    p_bh = _benjamini_hochberg(raw_p_values)
    p_holm = _holm_adjust(raw_p_values)

    for row, bh, holm in zip(rows, p_bh, p_holm):
        row["p_value_BH"] = float(bh)
        row["p_value_Holm"] = float(holm)
        row["significant_BH_0.05"] = bool(bh < 0.05)
        row["significant_Holm_0.05"] = bool(holm < 0.05)
        row["alternative"] = alternative

    return pd.DataFrame(rows)


def _primary_comparison(
    error_wide: pd.DataFrame,
    modalities: Sequence[str],
    rep_a: str,
    rep_b: str,
    alternative: str,
    comparison_label: str,
) -> pd.DataFrame:
    key_a = _representation_key(rep_a)
    key_b = _representation_key(rep_b)
    cols_a = [f"abs_error_{m}_{key_a}" for m in modalities]
    cols_b = [f"abs_error_{m}_{key_b}" for m in modalities]

    mean_a = error_wide[cols_a].mean(axis=1)
    mean_b = error_wide[cols_b].mean(axis=1)
    diff = mean_a - mean_b

    error_wide[f"mean_abs_error_{key_a}"] = mean_a
    error_wide[f"mean_abs_error_{key_b}"] = mean_b
    error_wide[f"error_difference_{key_a}_minus_{key_b}"] = diff

    statistic, p_value = _safe_wilcoxon(mean_a, mean_b, alternative)

    return pd.DataFrame(
        [
            {
                "comparison": comparison_label,
                "representation_a": rep_a,
                "representation_b": rep_b,
                "n_patients": len(error_wide),
                "n_modalities": len(modalities),
                f"median_patient_mean_abs_error_{key_a}": float(np.median(mean_a)),
                f"median_patient_mean_abs_error_{key_b}": float(np.median(mean_b)),
                f"median_difference_{key_a}_minus_{key_b}": float(np.median(diff)),
                "wilcoxon_statistic": statistic,
                "p_value_raw": p_value,
                "alternative": alternative,
            }
        ]
    )


def run_analysis(args: argparse.Namespace) -> Path:
    script_dir = Path(__file__).resolve().parent

    clinical_path = (
        args.clinical.expanduser().resolve()
        if args.clinical is not None
        else _find_default_clinical(script_dir)
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.output_dir is None:
        base = script_dir / "nihss_prediction"
        base.mkdir(parents=True, exist_ok=True)
        output_dir = base / f"2d_vs_3d_nested_svr_{timestamp}"
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths_2d_raw = _parse_mapping(args.embedding_2d, "--embedding-2d")
    paths_3d_raw = _parse_mapping(args.embedding_3d, "--embedding-3d")

    if set(paths_2d_raw) != set(paths_3d_raw):
        raise ValueError(
            "Exactly the same modalities must be supplied for 2D and 3D. "
            f"2D={sorted(paths_2d_raw)}, 3D={sorted(paths_3d_raw)}"
        )

    modalities = sorted(paths_2d_raw)
    if not modalities:
        raise ValueError("No modalities were supplied.")
    if args.outer_folds < 2 or args.inner_folds < 2:
        raise ValueError("outer-folds and inner-folds must both be >= 2.")
    if args.pca_components is not None and args.pca_components < 1:
        raise ValueError("--pca-components must be >= 1 when specified.")

    clinical = load_clinical(
        clinical_path,
        args.clinical_id_col,
        args.target_col,
        args.clinical_split_col,
    )

    tables: Dict[Tuple[str, str], EmbeddingTable] = {}
    resolved_paths: Dict[str, Dict[str, str]] = {}
    pca_components_by_modality: Dict[str, int] = {}

    for modality in modalities:
        path_2d = _resolve_embedding_path(paths_2d_raw[modality], modality, "2D")
        path_3d = _resolve_embedding_path(paths_3d_raw[modality], modality, "3D")

        table_2d = load_embedding_table(path_2d, modality, "2D")
        table_3d = load_embedding_table(path_3d, modality, "3D")

        if not args.skip_pca_sensitivity:
            inferred_dim = int(table_3d.X.shape[1])

            if args.pca_components is not None:
                if int(args.pca_components) != inferred_dim:
                    raise ValueError(
                        f"{modality}: --pca-components={args.pca_components}, but "
                        f"the corresponding 3D embedding has {inferred_dim} dimensions. "
                        "For modality-wise dimension matching, omit --pca-components "
                        "and let the script infer the target from each 3D file."
                    )
                target_dim = int(args.pca_components)
            else:
                target_dim = inferred_dim

            pca_components_by_modality[modality] = target_dim

        tables[(modality, "2D")] = table_2d
        tables[(modality, "3D")] = table_3d
        resolved_paths[modality] = {"2D": str(path_2d), "3D": str(path_3d)}

        pca_text = (
            f"; PCA-matched 2D target p={pca_components_by_modality[modality]}"
            if not args.skip_pca_sensitivity
            else ""
        )
        print(
            f"{modality}: 2D n={len(table_2d.ids)}, p={table_2d.X.shape[1]}; "
            f"3D n={len(table_3d.ids)}, p={table_3d.X.shape[1]}"
            f"{pca_text}",
            flush=True,
        )

    _check_vae_split_labels(
        clinical,
        args.clinical_split_col,
        tables.values(),
        args.allow_vae_split_mismatch,
    )

    # One common cohort is used across every modality and representation.
    cohort_ids = set(clinical.index.astype(str))
    for table in tables.values():
        cohort_ids &= set(table.ids)

    if not cohort_ids:
        raise ValueError(
            "No patients are common to the clinical table and all embedding files."
        )

    patient_ids = np.asarray(sorted(cohort_ids), dtype=str)
    if len(patient_ids) < args.outer_folds:
        raise ValueError(
            f"Only {len(patient_ids)} common patients are available, fewer than "
            f"outer_folds={args.outer_folds}."
        )

    clinical_cohort = clinical.loc[patient_ids].copy()
    y = clinical_cohort[args.target_col].to_numpy(dtype=float)

    print(
        f"\nGlobal common cohort: {len(patient_ids)} patients across "
        f"{len(modalities)} modalities.",
        flush=True,
    )

    outer_cv = KFold(
        n_splits=args.outer_folds,
        shuffle=True,
        random_state=args.random_state,
    )
    outer_splits = list(outer_cv.split(np.arange(len(patient_ids))))

    fold_assignment = np.full(len(patient_ids), -1, dtype=int)
    for fold, (_, test_idx) in enumerate(outer_splits):
        fold_assignment[test_idx] = fold

    fold_df = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "outer_fold": fold_assignment,
            args.target_col: y,
        }
    )
    if args.clinical_split_col in clinical_cohort.columns:
        fold_df["vae_custom_split"] = (
            clinical_cohort[args.clinical_split_col].astype(str).to_numpy()
        )
    fold_df.to_csv(output_dir / "outer_fold_assignments.csv", index=False)

    all_predictions: List[pd.DataFrame] = []
    all_tuning: List[pd.DataFrame] = []
    metric_rows: List[dict] = []

    for modality in modalities:
        table_2d = tables[(modality, "2D")]
        table_3d = tables[(modality, "3D")]
        X_2d = _subset_matrix(table_2d, patient_ids)
        X_3d = _subset_matrix(table_3d, patient_ids)

        analyses: List[Tuple[str, np.ndarray, Optional[int]]] = [
            ("2D", X_2d, None),
            ("3D", X_3d, None),
        ]
        if not args.skip_pca_sensitivity:
            analyses.insert(
                1,
                (
                    "2D_PCA_MATCHED",
                    X_2d,
                    pca_components_by_modality[modality],
                ),
            )

        for representation, X, pca_components in analyses:
            print(
                f"\nRunning {modality} {representation}: "
                f"n={X.shape[0]}, input p={X.shape[1]}",
                flush=True,
            )

            pred_df, tuning_df = run_nested_svr(
                X=X,
                y=y,
                patient_ids=patient_ids,
                outer_splits=outer_splits,
                inner_folds=args.inner_folds,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                pca_components=pca_components,
            )

            pred_df.insert(1, "modality", modality)
            pred_df.insert(2, "representation", representation)
            all_predictions.append(pred_df)

            tuning_df.insert(0, "modality", modality)
            tuning_df.insert(1, "representation", representation)
            all_tuning.append(tuning_df)

            n_model_features = pca_components if pca_components is not None else X.shape[1]
            metric_row = _metric_row(
                modality=modality,
                representation=representation,
                pred_df=pred_df,
                n_input_features=X.shape[1],
                n_model_features=n_model_features,
            )
            metric_rows.append(metric_row)

            print(
                f"Completed {modality} {representation}: "
                f"MedAE={metric_row['MedAE']:.4f}, R2={metric_row['R2']:.4f}",
                flush=True,
            )

    predictions = pd.concat(all_predictions, ignore_index=True)
    tuning = pd.concat(all_tuning, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)

    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    tuning.to_csv(
        output_dir / "svr_best_hyperparameters_by_outer_fold.csv", index=False
    )
    metrics.to_csv(output_dir / "svr_metrics.csv", index=False)

    error_wide = predictions.pivot(
        index="patient_id",
        columns=["modality", "representation"],
        values="absolute_error",
    ).sort_index()
    error_wide.columns = [
        f"abs_error_{modality}_{_representation_key(representation)}"
        for modality, representation in error_wide.columns
    ]

    # Primary full-dimensional 2D vs 3D comparison.
    primary_df = _primary_comparison(
        error_wide=error_wide,
        modalities=modalities,
        rep_a="2D",
        rep_b="3D",
        alternative=args.wilcoxon_alternative,
        comparison_label="patient mean absolute error across modalities: 2D vs 3D",
    )
    primary_df.to_csv(
        output_dir / "primary_2d_vs_3d_comparison.csv", index=False
    )

    secondary_df = _add_modality_specific_tests(
        error_wide=error_wide,
        modalities=modalities,
        rep_a="2D",
        rep_b="3D",
        alternative=args.wilcoxon_alternative,
    )
    secondary_df.to_csv(
        output_dir / "secondary_modality_specific_2d_vs_3d.csv", index=False
    )

    sensitivity_primary_df: Optional[pd.DataFrame] = None
    sensitivity_secondary_df: Optional[pd.DataFrame] = None
    if not args.skip_pca_sensitivity:
        pca_rep = "2D_PCA_MATCHED"
        sensitivity_primary_df = _primary_comparison(
            error_wide=error_wide,
            modalities=modalities,
            rep_a=pca_rep,
            rep_b="3D",
            alternative=args.wilcoxon_alternative,
            comparison_label=(
                "patient mean absolute error across modalities: "
                "modality-wise dimension-matched 2D PCA vs 3D"
            ),
        )
        sensitivity_primary_df.to_csv(
            output_dir / "sensitivity_dimension_matched_2d_vs_3d_comparison.csv",
            index=False,
        )

        sensitivity_secondary_df = _add_modality_specific_tests(
            error_wide=error_wide,
            modalities=modalities,
            rep_a=pca_rep,
            rep_b="3D",
            alternative=args.wilcoxon_alternative,
        )
        sensitivity_secondary_df["pca_components"] = (
            sensitivity_secondary_df["modality"]
            .map(pca_components_by_modality)
            .astype(int)
        )
        sensitivity_secondary_df.to_csv(
            output_dir / "sensitivity_modality_specific_dimension_matched_2d_vs_3d.csv",
            index=False,
        )

    error_wide.reset_index().to_csv(
        output_dir / "patient_level_absolute_errors.csv", index=False
    )

    # This file intentionally stores the raw primary p-value only. The thesis-level
    # Holm family contains this comparison plus a later Imagix3D-vs-XModalix3D
    # comparison, so the two p-values must be adjusted together after both exist.
    primary_family = pd.DataFrame(
        [
            {
                "hypothesis": "Imagix 2D vs Imagix3D",
                "p_value_raw": float(primary_df.loc[0, "p_value_raw"]),
                "holm_family_status": (
                    "pending second primary p-value: Imagix3D vs XModalix3D"
                ),
            }
        ]
    )
    primary_family.to_csv(
        output_dir / "primary_hypothesis_pvalue_for_future_holm.csv", index=False
    )

    metadata = {
        "timestamp": timestamp,
        "clinical_file": str(clinical_path),
        "target_column": args.target_col,
        "modalities": modalities,
        "embedding_files": resolved_paths,
        "global_common_cohort_n": int(len(patient_ids)),
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "outer_cv": {
            "class": "KFold",
            "shuffle": True,
            "random_state": args.random_state,
        },
        "inner_cv": {
            "class": "KFold",
            "shuffle": True,
            "random_state_rule": "random_state + outer_fold",
        },
        "model": "sklearn.svm.SVR(kernel='linear')",
        "primary_pipeline": "StandardScaler -> linear SVR",
        "pca_sensitivity_enabled": not args.skip_pca_sensitivity,
        "pca_components_by_modality": (
            pca_components_by_modality
            if not args.skip_pca_sensitivity
            else None
        ),
        "pca_target_rule": (
            "For each modality, infer d_m from the number of LatDim_* columns "
            "in the corresponding 3D embedding file."
            if not args.skip_pca_sensitivity
            else None
        ),
        "pca_sensitivity_pipeline": (
            "StandardScaler -> PCA(n_components=d_m) -> linear SVR, with d_m "
            "equal to the corresponding modality-specific 3D latent dimensionality"
            if not args.skip_pca_sensitivity
            else None
        ),
        "inner_scoring": "neg_mean_absolute_error",
        "C_values": list(DEFAULT_C_VALUES),
        "epsilon_values": list(DEFAULT_EPSILON_VALUES),
        "wilcoxon_alternative": args.wilcoxon_alternative,
        "modality_specific_multiple_testing": [
            "Benjamini-Hochberg FDR across modality-specific tests",
            "Holm adjustment across modality-specific tests",
        ],
        "primary_thesis_holm_family": (
            "Imagix 2D vs Imagix3D AND Imagix3D vs XModalix3D; "
            "adjust jointly once both raw p-values are available"
        ),
        "n_jobs": args.n_jobs,
    }
    with open(output_dir / "analysis_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)
    print(metrics.to_string(index=False))
    print("\nPrimary 2D-vs-3D comparison:")
    print(primary_df.to_string(index=False))
    print("\nModality-specific 2D-vs-3D comparisons:")
    print(secondary_df.to_string(index=False))
    if sensitivity_primary_df is not None:
        print("\nModality-wise dimension-matched PCA sensitivity comparison:")
        print(sensitivity_primary_df.to_string(index=False))
        print("\nDimension-matched PCA modality-specific comparisons:")
        print(sensitivity_secondary_df.to_string(index=False))
    print(f"\nResults written to: {output_dir}")

    return output_dir


def main() -> None:
    args = parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
