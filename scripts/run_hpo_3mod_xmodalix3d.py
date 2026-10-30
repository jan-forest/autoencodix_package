from __future__ import annotations

import json
from pathlib import Path

from syne_tune import StoppingCriterion, Tuner
from syne_tune.backend import PythonBackend
from syne_tune.config_space import loguniform
from syne_tune.experiments import load_experiment
from syne_tune.optimizer.baselines import CQR


def synetune_objective_function(
    # Fixed data / model settings
    data_path: str,
    modality_hparams_json: str,
    epochs: int,
    pretrain_epochs: int,
    checkpoint_interval: int,
    target_shape: str,
    batch_size: int,
    latent_dim: int,
    hidden_dim: int,
    train_normalization: str,
    anneal_function: str,
    scaling: str,
    loss_reduction: str,
    device: str,
    n_gpus: int,
    training_seed: int,

    # Tunable XModalix3D parameters
    gamma: float,
    delta_pair: float,
    classifier_learning_rate: float,
    classifier_weight_decay: float,
) -> None:
    import json
    import os
    import tempfile
    from pathlib import Path

    import mlflow
    import numpy as np
    import pandas as pd
    import torch
    import autoencodix as acx

    from syne_tune import Reporter
    from autoencodix.configs import DataCase, DataConfig, DataInfo
    from autoencodix.configs.xmodalix3d_config import XModalix3DConfig
    from autoencodix.evaluate._xmodalix3d_evaluator import XModalix3DEvaluator
    
    MODALITY_FILES = {
        "NCCT": ("ncct_flat", "ncct_anno_dst.csv"),
        "CBF": ("cbf_flat", "cbf_anno_dst.csv"),
        "CBV": ("cbv_flat", "cbv_anno_dst.csv"),
    }

    SUPPORTED_VOLUME_EXTENSIONS = (".nii", ".nii.gz")


    def _read_available_annotation(
        annotation_file: Path,
        image_dir: Path,
        modality: str,
        img_path_col: str = "filepath",
        split_col: str = "custom_splits",
    ):
        import pandas as pd

        if not annotation_file.is_file():
            raise FileNotFoundError(
                f"{modality} annotation file does not exist: {annotation_file}"
            )
        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"{modality} image directory does not exist: {image_dir}"
            )

        annotation = pd.read_csv(annotation_file, index_col=0, engine="python")
        annotation.index = annotation.index.astype(str)

        if annotation.index.has_duplicates:
            duplicate_ids = annotation.index[
                annotation.index.duplicated()
            ].unique().tolist()
            raise ValueError(
                f"{modality} annotation contains duplicate sample IDs. "
                f"First duplicates: {duplicate_ids[:10]}"
            )

        required_columns = {img_path_col, split_col}
        missing_columns = required_columns - set(annotation.columns)
        if missing_columns:
            raise ValueError(
                f"{modality} annotation is missing columns: "
                f"{sorted(missing_columns)}"
            )

        split_values = annotation[split_col].astype(str).str.strip().str.lower()
        allowed_splits = {"train", "valid", "test"}
        invalid_splits = set(split_values.unique()) - allowed_splits
        if invalid_splits:
            raise ValueError(
                f"{modality} annotation contains invalid split labels: "
                f"{sorted(invalid_splits)}"
            )

        available_files = {
            path.name
            for path in image_dir.iterdir()
            if path.is_file()
            and path.name.lower().endswith(SUPPORTED_VOLUME_EXTENSIONS)
        }

        available_mask = annotation[img_path_col].astype(str).isin(available_files)
        missing_count = int((~available_mask).sum())
        if missing_count:
            print(
                f"{modality}: ignoring {missing_count} annotation row(s) "
                "without an available volume.",
                flush=True,
            )

        annotation = annotation.loc[available_mask].copy()
        if annotation.empty:
            raise ValueError(f"No usable {modality} samples remain.")

        return annotation


    def _build_common_custom_splits(
        data_path: Path,
        paired_annotation_path: Path,
        img_path_col: str = "filepath",
        split_col: str = "custom_splits",
    ):
        import numpy as np

        annotations = {}
        for modality, (folder, anno_name) in MODALITY_FILES.items():
            annotations[modality] = _read_available_annotation(
                annotation_file=data_path / anno_name,
                image_dir=data_path / folder,
                modality=modality,
                img_path_col=img_path_col,
                split_col=split_col,
            )

        common_ids = sorted(
            set.intersection(*(set(df.index) for df in annotations.values()))
        )

        if not common_ids:
            raise ValueError(
                "The three modalities have no common sample IDs after matching "
                "annotations to available volumes."
            )

        reference_modality = "NCCT"
        reference_splits = (
            annotations[reference_modality]
            .loc[common_ids, split_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        for modality, annotation in annotations.items():
            if modality == reference_modality:
                continue

            modality_splits = (
                annotation.loc[common_ids, split_col]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            mismatch_mask = (
                reference_splits.to_numpy() != modality_splits.to_numpy()
            )
            if mismatch_mask.any():
                mismatch_positions = np.flatnonzero(mismatch_mask)
                examples = [
                    (
                        common_ids[pos],
                        reference_splits.iloc[pos],
                        modality_splits.iloc[pos],
                    )
                    for pos in mismatch_positions[:10]
                ]
                raise ValueError(
                    f"Custom split assignments differ between {reference_modality} "
                    f"and {modality} for {int(mismatch_mask.sum())} common sample(s). "
                    f"First mismatches: {examples}"
                )

        # The canonical annotation is restricted to the patients available in all
        # three modalities and ordered deterministically by the common sample ID.
        paired_annotation = annotations[reference_modality].loc[common_ids].copy()
        paired_annotation.to_csv(paired_annotation_path)

        split_values = (
            paired_annotation[split_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .to_numpy()
        )

        custom_splits = {
            "train": np.where(split_values == "train")[0].astype(int),
            "valid": np.where(split_values == "valid")[0].astype(int),
            "test": np.where(split_values == "test")[0].astype(int),
        }

        print("Paired three-modality cohort:", flush=True)
        print(f"  common samples: {len(common_ids)}", flush=True)
        for split_name in ("train", "valid", "test"):
            print(
                f"  {split_name}: {len(custom_splits[split_name])}",
                flush=True,
            )

        return custom_splits


    def _simple_modality_name(name: str) -> str:
        return str(name).split(".", 1)[-1]


    report = Reporter()
    data_path_obj = Path(data_path)
    modality_hparams = json.loads(modality_hparams_json)
    target_shape_3d = tuple(int(x) for x in target_shape.split(","))

    if set(modality_hparams) != set(MODALITY_FILES):
        raise ValueError(
            "modality_hparams_json must contain exactly: "
            f"{sorted(MODALITY_FILES)}; got {sorted(modality_hparams)}"
        )

    # Keep the stochastic training seed fixed within one HPO procedure. For
    # independent HPO repetitions, change training_seed together with the CQR seed.
    np.random.seed(training_seed)
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(
        os.environ.get(
            "MLFLOW_TRACKING_URI",
            "file:/data/horse/ws/baeuchl-imagix3d/mlruns",
        )
    )
    mlflow.set_experiment("xmodalix3d_3mod_hpo")

    with mlflow.start_run(run_name="xmodalix3d_3mod_trial"):
        trial_status = "completed"
        mlflow.set_tags(
            {
                "pipeline": "XModalix3D",
                "modalities": "NCCT,CBF,CBV",
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
                "hpo_backend": "Syne Tune",
            }
        )

        params_to_log = {
            "epochs": epochs,
            "pretrain_epochs": pretrain_epochs,
            "batch_size": batch_size,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "train_normalization": train_normalization,
            "anneal_function": anneal_function,
            "gamma": gamma,
            "delta_pair": delta_pair,
            "classifier_learning_rate": classifier_learning_rate,
            "classifier_weight_decay": classifier_weight_decay,
            "training_seed": training_seed,
        }
        for modality, hp in modality_hparams.items():
            params_to_log[f"{modality}_learning_rate"] = hp["learning_rate"]
            params_to_log[f"{modality}_weight_decay"] = hp["weight_decay"]
            params_to_log[f"{modality}_beta"] = hp["beta"]
        mlflow.log_params(params_to_log)

        try:
            with tempfile.TemporaryDirectory(prefix="xmodalix3d_3mod_hpo_") as tmpdir:
                paired_annotation_path = Path(tmpdir) / "paired_annotation.csv"
                custom_splits = _build_common_custom_splits(
                    data_path=data_path_obj,
                    paired_annotation_path=paired_annotation_path,
                )

                # Translation direction is irrelevant for HPO because we only run
                # preprocess() + fit(). It is nevertheless defined so that the config
                # remains a valid XModalix configuration.
                translation_directions = {
                    "NCCT": "from",
                    "CBF": "to",
                    "CBV": None,
                }

                data_info = {}
                for modality, (folder, anno_name) in MODALITY_FILES.items():
                    hp = modality_hparams[modality]
                    data_info[modality] = DataInfo(
                        file_path=str(data_path_obj / folder),
                        extra_anno_file=str(data_path_obj / anno_name),
                        scaling=scaling,
                        data_type="IMG",
                        translate_direction=translation_directions[modality],
                        pretrain_epochs=pretrain_epochs,
                        learning_rate=float(hp["learning_rate"]),
                        weight_decay=float(hp["weight_decay"]),
                        beta=float(hp["beta"]),
                    )

                data_info["ANNO"] = DataInfo(
                    file_path=str(paired_annotation_path),
                    data_type="ANNOTATION",
                )

                config = XModalix3DConfig(
                    data_case=DataCase.IMG_TO_IMG,
                    img_path_col="filepath",
                    spatial_shape_policy="crop_or_pad_to_shape",
                    target_shape_3d=target_shape_3d,
                    checkpoint_interval=checkpoint_interval,
                    epochs=epochs,
                    pretrain_epochs=pretrain_epochs,
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    batch_size=batch_size,
                    reconstruction_loss="mse",
                    loss_reduction=loss_reduction,
                    scaling=scaling,
                    anneal_function=anneal_function,
                    normalize_nonzero_only=True,
                    clamp_logvar=True,
                    keep_mu_positive=False,
                    train_normalization=train_normalization,
                    train_norm_groupsize=8,
                    volume_scaling_strategy="train_global",
                    device=device,
                    n_gpus=n_gpus,
                    # Global optimizer values are used by the latent modality
                    # classifier. The VAE optimizers use the DataInfo overrides.
                    learning_rate=classifier_learning_rate,
                    weight_decay=classifier_weight_decay,
                    beta=1e-5,
                    gamma=gamma,
                    delta_pair=delta_pair,
                    delta_class=0.0,
                    requires_paired=True,
                    data_config=DataConfig(data_info=data_info),
                )

                xmodalix3d = acx.XModalix3D(
                    config=config,
                    custom_splits=custom_splits,
                )

                # Do not call .run() during HPO: BasePipeline.run() additionally
                # predicts on the test set and visualizes. HPO uses train + valid only.
                xmodalix3d.preprocess()
                xmodalix3d.fit()

                result = xmodalix3d.result

                probe_summary, _ = XModalix3DEvaluator.modality_linear_probe(
                    result=result,
                    representation="z",
                    split="valid",
                    epoch=-1,
                    n_splits=5,
                )

                matched_summary = XModalix3DEvaluator.matched_unmatched_analysis(
                    result=result,
                    representation="z",
                    split="valid",
                    epoch=-1,
                )

                ratios = matched_summary["matched_unmatched_ratio"].to_numpy(
                    dtype=float
                )
                if len(ratios) != 3:
                    raise ValueError(
                        "Expected 3 pairwise alignment ratios for three modalities, "
                        f"but found {len(ratios)}."
                    )
                if not np.isfinite(ratios).all():
                    raise ValueError(
                        "Non-finite matched/unmatched ratio encountered."
                    )

                mean_ratio = float(np.mean(ratios))
                worst_ratio = float(np.max(ratios))

                probe_accuracy = float(
                    probe_summary["linear_probe_balanced_accuracy"].iloc[0]
                )
                chance_level = float(probe_summary["chance_level"].iloc[0])
                probe_excess_accuracy = max(
                    0.0,
                    (probe_accuracy - chance_level) / (1.0 - chance_level),
                )

                metrics = {
                    "worst_matched_unmatched_ratio": worst_ratio,
                    "mean_matched_unmatched_ratio": mean_ratio,
                    "modality_probe_balanced_accuracy": probe_accuracy,
                    "modality_probe_chance_level": chance_level,
                    "modality_probe_excess_accuracy": probe_excess_accuracy,
                    # Combined HPO target: patient-specific alignment plus modality invariance.
                    "alignment_score": mean_ratio + probe_excess_accuracy,
                }

                # Store all three pairwise ratios in results.csv as diagnostics.
                for _, row in matched_summary.iterrows():
                    mod_a = _simple_modality_name(row["modality_a"])
                    mod_b = _simple_modality_name(row["modality_b"])
                    metrics[f"ratio_{mod_a}_{mod_b}"] = float(
                        row["matched_unmatched_ratio"]
                    )

                valid_recon_losses = []
                train_recon_losses = []
                for modality in MODALITY_FILES:
                    key = f"img.{modality}.recon_loss"
                    valid_recon = float(
                        np.asarray(
                            result.sub_losses.get(key).get(
                                epoch=-1,
                                split="valid",
                            )
                        ).item()
                    )
                    train_recon = float(
                        np.asarray(
                            result.sub_losses.get(key).get(
                                epoch=-1,
                                split="train",
                            )
                        ).item()
                    )
                    metrics[f"valid_recon_{modality}"] = valid_recon
                    metrics[f"train_recon_{modality}"] = train_recon
                    valid_recon_losses.append(valid_recon)
                    train_recon_losses.append(train_recon)

                metrics["mean_valid_reconstruction_loss"] = float(
                    np.mean(valid_recon_losses)
                )
                metrics["max_valid_reconstruction_loss"] = float(
                    np.max(valid_recon_losses)
                )
                metrics["mean_train_reconstruction_loss"] = float(
                    np.mean(train_recon_losses)
                )
                metrics["valid_total_loss"] = float(
                    np.asarray(
                        result.losses.get(epoch=-1, split="valid")
                    ).item()
                )
                metrics["train_total_loss"] = float(
                    np.asarray(
                        result.losses.get(epoch=-1, split="train")
                    ).item()
                )

        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                mlflow.set_tag("trial_status", "failed")
                mlflow.set_tag("error", repr(error)[:5000])
                raise

            print(
                "CUDA out of memory in this trial. Reporting penalty values.",
                flush=True,
            )
            torch.cuda.empty_cache()
            metrics = {
                "worst_matched_unmatched_ratio": 1e9,
                "mean_matched_unmatched_ratio": 1e9,
                "modality_probe_balanced_accuracy": 1.0,
                "modality_probe_chance_level": 1.0 / 3.0,
                "modality_probe_excess_accuracy": 1.0,
                "alignment_score": 1e9,
                "mean_valid_reconstruction_loss": 1e9,
                "max_valid_reconstruction_loss": 1e9,
                "mean_train_reconstruction_loss": 1e9,
                "valid_total_loss": 1e9,
                "train_total_loss": 1e9,
            }
            trial_status = "oom"

        mlflow.set_tag("trial_status", trial_status)
        mlflow.log_metrics(metrics)
        report(**metrics)


def run_synetune_hpo(
    data_path: Path,
    modality_hparams: dict,
    latent_dim: int,
    hidden_dim: int = 16,
    train_normalization: str = "group",
    batch_size: int = 12,
    epochs: int = 250,
    pretrain_epochs: int = 50,
    target_shape: tuple[int, int, int] = (160, 192, 160),
    metric: str = "alignment_score",
    max_wallclock_time: int = 11 * 60 * 60,
    n_workers: int = 4,
    random_seed: int = 42,
):
    
    MODALITY_FILES = {
        "NCCT": ("ncct_flat", "ncct_anno_dst.csv"),
        "CBF": ("cbf_flat", "cbf_anno_dst.csv"),
        "CBV": ("cbv_flat", "cbv_anno_dst.csv"),
    }
    supported_metrics = {
        "worst_matched_unmatched_ratio",
        "mean_matched_unmatched_ratio",
        "modality_probe_excess_accuracy",
        "alignment_score",
    }
    if metric not in supported_metrics:
        raise ValueError(
            f"metric must be one of {sorted(supported_metrics)}, got {metric!r}."
        )

    missing_modalities = set(MODALITY_FILES) - set(modality_hparams)
    extra_modalities = set(modality_hparams) - set(MODALITY_FILES)
    if missing_modalities or extra_modalities:
        raise ValueError(
            "modality_hparams must contain exactly the three CT modalities. "
            f"Missing={sorted(missing_modalities)}, extra={sorted(extra_modalities)}"
        )

    modality_hparams_json = json.dumps(modality_hparams, sort_keys=True)

    config_space = {
        # Fixed parameters
        "data_path": str(data_path),
        "modality_hparams_json": modality_hparams_json,
        "epochs": int(epochs),
        "pretrain_epochs": int(pretrain_epochs),
        # Store only the final joint-training dynamics to keep HPO memory use down.
        "checkpoint_interval": int(epochs),
        "target_shape": ",".join(str(x) for x in target_shape),
        "batch_size": int(batch_size),
        "latent_dim": int(latent_dim),
        "hidden_dim": int(hidden_dim),
        "train_normalization": train_normalization,
        "anneal_function": "logistic-late",
        "scaling": "MINMAX",
        "loss_reduction": "mean",
        "device": "cuda",
        "n_gpus": 1,
        "training_seed": int(random_seed),

        # XModalix3D HPO parameters
        "gamma": loguniform(1e-2, 1e2),
        "delta_pair": loguniform(1e-2, 1e2),
        "classifier_learning_rate": loguniform(1e-6, 1e-2),
        "classifier_weight_decay": loguniform(1e-7, 1e-2),
    }

    # Existing hand-selected settings are included as an anchor point.
    points_to_evaluate = [
        {
            "gamma": 10.0,
            "delta_pair": 5.0,
            "classifier_learning_rate": 1e-5,
            "classifier_weight_decay": 1e-5,
        }
    ]

    scheduler = CQR(
        config_space=config_space,
        metric=metric,
        do_minimize=True,
        points_to_evaluate=points_to_evaluate,
        random_seed=random_seed,
    )

    metadata = {
        "points_to_evaluate": json.dumps(points_to_evaluate),
        "modalities": json.dumps(list(MODALITY_FILES)),
        "modality_hparams": modality_hparams_json,
        "metric": metric,
        "mode": "minimize",
        "epochs": epochs,
        "pretrain_epochs": pretrain_epochs,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "train_normalization": train_normalization,
        "batch_size": batch_size,
        "target_shape": json.dumps(target_shape),
        "random_seed": random_seed,
    }

    tuner = Tuner(
        trial_backend=PythonBackend(
            tune_function=synetune_objective_function,
            config_space=config_space,
            rotate_gpus=True,
        ),
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(
            max_wallclock_time=max_wallclock_time,
        ),
        n_workers=n_workers,
        metadata=metadata,
    )

    try:
        tuner.run()
    except Exception as error:
        print("Tuning crashed, attempting to load partial Syne Tune experiment.")
        print(repr(error))
        return load_experiment(tuner.name)

    return load_experiment(tuner.name)