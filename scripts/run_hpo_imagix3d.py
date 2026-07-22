from pathlib import Path

from syne_tune.config_space import choice, loguniform
from syne_tune.optimizer.baselines import CQR
from syne_tune import Tuner, StoppingCriterion
from syne_tune.experiments import load_experiment
from syne_tune.backend import PythonBackend


def synetune_objective_function(
    # Fixed params
    epochs: int,
    checkpoint_interval: int,
    loss_reduction: str,
    volume_root: str,
    annotation_file: str,
    tasks: str,
    device: str,
    n_gpus: int,

    # Tunable params
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    beta: float,
    latent_dim: int,
    hidden_dim: int,
    anneal_function: str,
    train_normalization: str,
    keep_mu_positive: int,
) -> None:
    import numpy as np
    import sklearn
    import autoencodix as acx

    from syne_tune import Reporter
    from sklearn import linear_model
    from autoencodix.configs.imagix3d_config import Imagix3DConfig
    from autoencodix.configs.default_config import (
        DataConfig,
        DataCase,
        DataInfo,
    )

    volconfig = Imagix3DConfig(
        # Tunable params
        beta=beta,
        batch_size=batch_size,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        train_normalization=train_normalization,
        keep_mu_positive=bool(keep_mu_positive),

        # Fixed params
        data_case=DataCase.IMG_TO_IMG,
        img_path_col="filepath",
        spatial_shape_policy="crop_or_pad_to_shape",
        target_shape_3d=(64, 64, 64),
        checkpoint_interval=checkpoint_interval,
        epochs=epochs,
        reconstruction_loss="mse",
        loss_reduction=loss_reduction,
        scaling="MINMAX",
        anneal_function=anneal_function,
        normalize_nonzero_only=False,
        clamp_logvar=True,
        train_norm_groupsize=8,
        device=device,
        n_gpus=n_gpus,
        data_config=DataConfig(
            data_info={
                "IMG": DataInfo(
                    file_path=volume_root,
                    scaling="MINMAX",
                    data_type="IMG",
                ),
                "ANNO": DataInfo(
                    file_path=annotation_file,
                    data_type="ANNOTATION",
                ),
            },
        ),
    )

    imagix3d = acx.Imagix3D(config=volconfig)
    imagix3d.run()

    valid_recon_loss = float(np.asarray(imagix3d.result.sub_losses.get("recon_loss").get(epoch=-1,split="valid",)).item())
    train_recon_loss = float(np.asarray(imagix3d.result.sub_losses.get("recon_loss").get(epoch=-1,split="train",)).item())
    valid_total_loss = float(np.asarray(imagix3d.result.losses.get(epoch=-1,split="valid",)).item())
    train_total_loss = float(np.asarray(imagix3d.result.losses.get(epoch=-1,split="train",)).item())
    valid_var_loss = float(np.asarray(imagix3d.result.sub_losses.get("var_loss").get(epoch=-1,split="valid",)).item())

    sklearn.set_config(enable_metadata_routing=True)

    sklearn_ml_class = linear_model.LogisticRegression(
        solver="sag",
        n_jobs=1,
        class_weight="balanced",
        max_iter=200,
    )

    sklearn_ml_regression = linear_model.LinearRegression()

    tasks_list = [task for task in tasks.split("$") if task]

    imagix3d.evaluate(
        ml_model_class=sklearn_ml_class,
        ml_model_regression=sklearn_ml_regression,
        params=tasks_list,
        metric_class="roc_auc_ovo",
        metric_regression="r2",
        reference_methods=[],
        split_type="use-split",
        n_downsample=None,
    )

    downstream_performance = float(
        imagix3d.result.embedding_evaluation.loc[
            imagix3d.result.embedding_evaluation.score_split == "valid",
            "value",
        ].mean()
    )

    report = Reporter()
    report(
        downstream_performance=downstream_performance,
        reconstruction_loss=valid_recon_loss,
        train_reconstruction_loss=train_recon_loss,
        valid_total_loss=valid_total_loss,
        train_total_loss=train_total_loss,
        valid_var_loss=valid_var_loss,
        recon_generalization_gap=valid_recon_loss - train_recon_loss,
    )


def run_synetune_hpo(
    data_path: Path,
    folder: str = "train",
    anno: str = "train_metadata.csv",
    tasks: str = "group",
    metric: str = "reconstruction_loss",
):
    if metric not in ["reconstruction_loss", "downstream_performance"]:
        raise ValueError(
            "metric must be either 'reconstruction_loss' or "
            "'downstream_performance'."
        )

    volume_root = data_path / folder
    annotation_file = data_path / anno

    config_space = {
        # Fixed params
        "epochs": 100,
        "checkpoint_interval": 10,
        "loss_reduction": "mean",
        "volume_root": str(volume_root),
        "annotation_file": str(annotation_file),
        "tasks": tasks,

        # Hardware params
        "device": "cuda",
        "n_gpus": 1,

        # Tunable params
        "batch_size": choice([16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 256]),
        "learning_rate": loguniform(1e-5, 1e-1),
        "weight_decay": loguniform(1e-5, 1e-1),
        "beta": loguniform(1e-5, 5e-2),
        "latent_dim": choice([16, 32, 48, 64, 128, 256]),
        "hidden_dim": choice([8, 16, 32, 48, 64]),
        "train_normalization": choice(["group", "instance", "batch"]),
        "anneal_function": choice(
            [
                "5phase-constant",
                "3phase-linear",
                "3phase-log",
                "logistic-mid",
                "logistic-early",
                "logistic-late",
            ]
        ),
        # Encoded as scalar values for Syne Tune compatibility
        "keep_mu_positive": choice([0, 1]),
    }

    points_to_evaluate = [
        {
            "batch_size": 256,
            "learning_rate": 1e-3,
            "weight_decay": 5e-3,
            "beta": 0.02,
            "latent_dim": 32,
            "hidden_dim": 8,
            "anneal_function": "logistic-late",
            "train_normalization": "instance",
            "keep_mu_positive": 0,
        }
    ]

    if metric == "downstream_performance":
        do_minimize = False
    else:
        do_minimize = True

    scheduler = CQR(
        config_space=config_space,
        metric=metric,
        do_minimize=do_minimize,
        points_to_evaluate=points_to_evaluate,
    )

    tuner = Tuner(
        trial_backend=PythonBackend(
            tune_function=synetune_objective_function,
            config_space=config_space,
            rotate_gpus=True,
        ),
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(
            max_num_trials_completed=100,
        ),
        n_workers=4,
    )

    tuner.run()

    return load_experiment(tuner.name)


def run_optuna_hpo():
    raise NotImplementedError("Optuna HPO will be added later.")