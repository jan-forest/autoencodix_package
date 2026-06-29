from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import autoencodix as acx
from autoencodix.configs import DataCase, DataConfig, DataInfo
from autoencodix.configs.imagix3d_config import Imagix3DConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Imagix3D pipeline.")

    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--anno", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)

    parser.add_argument(
        "--target-shape",
        nargs=3,
        type=int,
        default = (64, 64, 64),
        metavar=("D", "H", "W"),
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--reconstruction-loss", type=str, default="mse")
    parser.add_argument("--loss-reduction", type=str, default="mean")
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--scaling", type=str, default="MINMAX")
    parser.add_argument("--anneal-function", type=str, default="logistic-late")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--evaluate-param", type=str, default=None)
    parser.add_argument("--clamp-logvar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-mu-positive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-normalization", type=str, default="batch")
    parser.add_argument("--train-norm-groupsize", type=int, default=8)

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Imagix3DConfig:
    return Imagix3DConfig(
        data_case=DataCase.IMG_TO_IMG,
        img_path_col="filepath",
        spatial_shape_policy="crop_or_pad_to_shape",
        target_shape_3d=tuple(args.target_shape),
        checkpoint_interval=args.checkpoint_interval,
        epochs=args.epochs,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        reconstruction_loss=args.reconstruction_loss,
        loss_reduction=args.loss_reduction,
        beta=args.beta,
        scaling=args.scaling,
        anneal_function=args.anneal_function,
        clamp_logvar=args.clamp_logvar,
        keep_mu_positive=args.keep_mu_positive,
        train_normalization=args.train_normalization,
        train_norm_groupsize=args.train_norm_groupsize,
        data_config=DataConfig(
            data_info={
                "IMG": DataInfo(
                    file_path=str(args.images),
                    scaling=args.scaling,
                    data_type="IMG",
                ),
                "ANNO": DataInfo(
                    file_path=str(args.anno),
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
        "images": str(args.images),
        "annotation": str(args.anno),
        "target_shape": args.target_shape,
        "epochs": args.epochs,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "reconstruction_loss": args.reconstruction_loss,
        "loss_reduction": args.loss_reduction,
        "beta": args.beta,
        "scaling": args.scaling,
        "anneal_function": args.anneal_function,
        "checkpoint_interval": args.checkpoint_interval,
        "clamp_logvar": args.clamp_logvar,
        "keep_mu_positive":args.keep_mu_positive,
        "train_normalization": args.train_normalization,
        "train_norm_groupsize": args.train_norm_groupsize,
        "evaluate_param": args.evaluate_param,
    }

    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=4)

    print("Starting Imagix3D run")
    print(f"Run directory: {run_dir}")
    print(f"Image directory: {args.images}")
    print(f"Annotation file: {args.anno}")
    print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}")

    imagix3d = acx.Imagix3D(config=config)
    result = imagix3d.run()

    if args.evaluate_param:
        imagix3d.evaluate(params=[args.evaluate_param])

    imagix3d.save(file_path=str(run_dir / "imagix3d.pkl"), save_all=True)

    print("Run finished successfully")
    print(f"Pipeline saved to: {run_dir / 'imagix3d.pkl'}")
    print(f"Final result object type: {type(result)}")


if __name__ == "__main__":
    main()