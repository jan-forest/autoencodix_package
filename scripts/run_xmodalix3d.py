from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import autoencodix as acx
from autoencodix.configs import DataCase, DataConfig, DataInfo
from autoencodix.configs.xmodalix3d_config import XModalix3DConfig
from autoencodix.utils._utils import custom_splits_from_anno


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the three-modality XModalix3D pipeline."
    )

    parser.add_argument("--workdir", type=Path, required=True)

    parser.add_argument("--ncct-images", type=Path, required=True)
    parser.add_argument("--ncct-anno", type=Path, required=True)

    parser.add_argument("--cbf-images", type=Path, required=True)
    parser.add_argument("--cbf-anno", type=Path, required=True)

    parser.add_argument("--cbv-images", type=Path, required=True)
    parser.add_argument("--cbv-anno", type=Path, required=True)

    parser.add_argument("--run-name", type=str, required=True)

    parser.add_argument(
        "--target-shape",
        nargs=3,
        type=int,
        default=(64, 64, 64),
        metavar=("D", "H", "W"),
        required=True,
    )

    # Joint training
    parser.add_argument("--epochs", type=int, default=50)

    # Unimodal pretraining
    parser.add_argument("--pretrain-epochs", type=int, default=50)

    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=12)

    parser.add_argument("--reconstruction-loss", type=str, default="mse")
    parser.add_argument("--loss-reduction", type=str, default="mean")
    parser.add_argument("--scaling", type=str, default="NONE")
    parser.add_argument("--anneal-function", type=str, default="logistic-late")
    parser.add_argument("--checkpoint-interval", type=int, default=5)

    parser.add_argument(
        "--clamp-logvar",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--keep-mu-positive",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--train-normalization", type=str, default="group")
    parser.add_argument("--train-norm-groupsize", type=int, default=8)

    # Global fallback / shared XModalix settings
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=1e-5)

    # Modality-specific optimization settings
    parser.add_argument("--ncct-learning-rate", type=float, default=3e-5)
    parser.add_argument("--ncct-weight-decay", type=float, default=8e-6)
    parser.add_argument("--ncct-beta", type=float, default=1e-6)

    parser.add_argument("--cbf-learning-rate", type=float, default=2e-5)
    parser.add_argument("--cbf-weight-decay", type=float, default=1e-5)
    parser.add_argument("--cbf-beta", type=float, default=5e-6)

    parser.add_argument("--cbv-learning-rate", type=float, default=1.5e-5)
    parser.add_argument("--cbv-weight-decay", type=float, default=1.2e-5)
    parser.add_argument("--cbv-beta", type=float, default=3e-6)

    # Cross-modal alignment weights
    parser.add_argument("--gamma", type=float, default=30.0)
    parser.add_argument("--delta-pair", type=float, default=5.0)
    parser.add_argument("--delta-class", type=float, default=0.0)

    # Translation direction. All modalities still participate in alignment.
    parser.add_argument(
        "--from-modality",
        type=str,
        choices=("NCCT", "CBF", "CBV"),
        default="NCCT",
    )
    parser.add_argument(
        "--to-modality",
        type=str,
        choices=("NCCT", "CBF", "CBV"),
        default="CBF",
    )

    # Optional downstream evaluation, analogous to run_imagix3d.py.
    parser.add_argument("--evaluate-param", type=str, default=None)

    return parser.parse_args()


def _translation_direction(
    modality: str,
    from_modality: str,
    to_modality: str,
):
    if modality == from_modality:
        return "from"
    if modality == to_modality:
        return "to"
    return None


def build_config(args: argparse.Namespace) -> XModalix3DConfig:
    if args.from_modality == args.to_modality:
        raise ValueError(
            "--from-modality and --to-modality must refer to different modalities."
        )

    return XModalix3DConfig(
        data_case=DataCase.IMG_TO_IMG,
        img_path_col="filepath",
        spatial_shape_policy="crop_or_pad_to_shape",
        target_shape_3d=tuple(args.target_shape),

        checkpoint_interval=args.checkpoint_interval,
        epochs=args.epochs,
        pretrain_epochs=args.pretrain_epochs,

        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,

        reconstruction_loss=args.reconstruction_loss,
        loss_reduction=args.loss_reduction,

        # Global fallback / shared values
        beta=args.beta,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,

        scaling=args.scaling,
        anneal_function=args.anneal_function,
        clamp_logvar=args.clamp_logvar,
        keep_mu_positive=args.keep_mu_positive,
        train_normalization=args.train_normalization,
        train_norm_groupsize=args.train_norm_groupsize,

        # Cross-modal losses
        gamma=args.gamma,
        delta_pair=args.delta_pair,
        delta_class=args.delta_class,

        requires_paired=True,

        data_config=DataConfig(
            data_info={
                "NCCT": DataInfo(
                    file_path=str(args.ncct_images),
                    extra_anno_file=str(args.ncct_anno),
                    scaling=args.scaling,
                    data_type="IMG",
                    translate_direction=_translation_direction(
                        "NCCT",
                        args.from_modality,
                        args.to_modality,
                    ),
                    pretrain_epochs=args.pretrain_epochs,
                    learning_rate=args.ncct_learning_rate,
                    weight_decay=args.ncct_weight_decay,
                    beta=args.ncct_beta,
                ),
                "CBF": DataInfo(
                    file_path=str(args.cbf_images),
                    extra_anno_file=str(args.cbf_anno),
                    scaling=args.scaling,
                    data_type="IMG",
                    translate_direction=_translation_direction(
                        "CBF",
                        args.from_modality,
                        args.to_modality,
                    ),
                    pretrain_epochs=args.pretrain_epochs,
                    learning_rate=args.cbf_learning_rate,
                    weight_decay=args.cbf_weight_decay,
                    beta=args.cbf_beta,
                ),
                "CBV": DataInfo(
                    file_path=str(args.cbv_images),
                    extra_anno_file=str(args.cbv_anno),
                    scaling=args.scaling,
                    data_type="IMG",
                    translate_direction=_translation_direction(
                        "CBV",
                        args.from_modality,
                        args.to_modality,
                    ),
                    pretrain_epochs=args.pretrain_epochs,
                    learning_rate=args.cbv_learning_rate,
                    weight_decay=args.cbv_weight_decay,
                    beta=args.cbv_beta,
                ),
                "ANNO": DataInfo(
                    file_path=str(args.ncct_anno),
                    data_type="ANNOTATION",
                ),
            },
        ),
    )


def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.workdir / "results" / f"{args.run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(args)

    # Save a run description for traceability
    run_metadata = {
        "run_name": args.run_name,
        "timestamp": timestamp,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),

        "modalities": {
            "NCCT": {
                "images": str(args.ncct_images),
                "annotation": str(args.ncct_anno),
                "learning_rate": args.ncct_learning_rate,
                "weight_decay": args.ncct_weight_decay,
                "beta": args.ncct_beta,
                "pretrain_epochs": args.pretrain_epochs,
            },
            "CBF": {
                "images": str(args.cbf_images),
                "annotation": str(args.cbf_anno),
                "learning_rate": args.cbf_learning_rate,
                "weight_decay": args.cbf_weight_decay,
                "beta": args.cbf_beta,
                "pretrain_epochs": args.pretrain_epochs,
            },
            "CBV": {
                "images": str(args.cbv_images),
                "annotation": str(args.cbv_anno),
                "learning_rate": args.cbv_learning_rate,
                "weight_decay": args.cbv_weight_decay,
                "beta": args.cbv_beta,
                "pretrain_epochs": args.pretrain_epochs,
            },
        },

        "translation_from": args.from_modality,
        "translation_to": args.to_modality,

        "target_shape": args.target_shape,
        "epochs": args.epochs,
        "pretrain_epochs": args.pretrain_epochs,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "batch_size": args.batch_size,

        "reconstruction_loss": args.reconstruction_loss,
        "loss_reduction": args.loss_reduction,
        "scaling": args.scaling,
        "anneal_function": args.anneal_function,
        "checkpoint_interval": args.checkpoint_interval,

        "clamp_logvar": args.clamp_logvar,
        "keep_mu_positive": args.keep_mu_positive,
        "train_normalization": args.train_normalization,
        "train_norm_groupsize": args.train_norm_groupsize,

        "global_learning_rate": args.learning_rate,
        "global_weight_decay": args.weight_decay,
        "global_beta": args.beta,

        "gamma": args.gamma,
        "delta_pair": args.delta_pair,
        "delta_class": args.delta_class,

        "evaluate_param": args.evaluate_param,
    }

    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=4)

    print("Starting XModalix3D three-modality run")
    print(f"Run directory: {run_dir}")
    print(f"NCCT image directory: {args.ncct_images}")
    print(f"CBF image directory: {args.cbf_images}")
    print(f"CBV image directory: {args.cbv_images}")
    print(f"Joint epochs: {args.epochs}")
    print(f"Pretraining epochs per modality: {args.pretrain_epochs}")
    print(
        f"Translation direction: "
        f"{args.from_modality} -> {args.to_modality}"
    )
    print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}")

    # The modalities are paired by patient and use the same predefined
    # patient-level train / validation / test assignments.
    custom_splits = custom_splits_from_anno(
        annotation_file=args.ncct_anno,
        split_col="custom_splits",
        sample_id_col="sample_id",
    )

    xmodalix3d = acx.XModalix3D(
        config=config,
        custom_splits=custom_splits,
    )

    result = xmodalix3d.run()

    if args.evaluate_param:
        xmodalix3d.evaluate(
            params=[args.evaluate_param]
        )

    xmodalix3d.save(
        file_path=str(run_dir / "xmodalix3d.pkl"),
        save_all=True,
    )

    print("Run finished successfully")
    print(
        f"Pipeline saved to: "
        f"{run_dir / 'xmodalix3d.pkl'}"
    )
    print(f"Final result object type: {type(result)}")


if __name__ == "__main__":
    main()
