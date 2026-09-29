from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import autoencodix as acx
from autoencodix.configs import DataCase, DataConfig, DataInfo
from autoencodix.configs.default_config import DefaultConfig
from autoencodix.utils._utils import custom_splits_from_anno


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 2D Imagix pipeline.")

    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument( "--anno", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)

    # Imagix requires square image input.
    # The 3D -> 2D conversion used for this project pads the original
    # slices to 192 x 192 without resampling.
    parser.add_argument("--image-size", type=int, default=192)

    # Defaults follow DefaultConfig unless noted otherwise.
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--reconstruction-loss", type=str, default="mse")
    parser.add_argument("--loss-reduction", type=str, default="sum")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument( "--scaling", type=str, default="NONE")
    parser.add_argument("--anneal-function", type=str, default="logistic-late")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--evaluate-param", type=str, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    
    return parser.parse_args()


def build_config(
    args: argparse.Namespace,
) -> DefaultConfig:
    return DefaultConfig(
        data_case=DataCase.IMG_TO_IMG,
        img_path_col="filepath",
        checkpoint_interval=args.checkpoint_interval,
        epochs=args.epochs,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        reconstruction_loss=args.reconstruction_loss,
        loss_reduction=args.loss_reduction,
        beta=args.beta,
        scaling=args.scaling,
        anneal_function=args.anneal_function,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        data_config=DataConfig(
            data_info={
                "IMG": DataInfo(
                    file_path=str(args.images),
                    scaling=args.scaling,
                    data_type="IMG",
                    img_width_resize=args.image_size,
                    img_height_resize=args.image_size,
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
    run_dir = (args.workdir / "results" / f"{args.run_name}_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    config = build_config(args)

    # Save a run description for traceability.
    run_metadata = {
        "run_name": args.run_name,
        "timestamp": timestamp,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "images": str(args.images),
        "annotation": str(args.anno),
        "image_size": args.image_size,
        "epochs": args.epochs,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "batch_size": args.batch_size,
        "reconstruction_loss": (args.reconstruction_loss),
        "loss_reduction": args.loss_reduction,
        "beta": args.beta,
        "scaling": args.scaling,
        "anneal_function": (args.anneal_function),
        "checkpoint_interval": (args.checkpoint_interval),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "evaluate_param": args.evaluate_param,
    }

    with open(run_dir / "run_metadata.json", "w", encoding="utf-8",) as f:
        json.dump(run_metadata, f, indent=4)

    print("Starting Imagix run")
    print(f"Run directory: {run_dir}")
    print(f"Image directory: {args.images}")
    print(f"Annotation file: {args.anno}")
    print(
        f"Image size: "
        f"{args.image_size} x "
        f"{args.image_size}"
    )
    print(
        f"SLURM_JOB_ID: "
        f"{os.environ.get('SLURM_JOB_ID')}"
    )

    custom_splits = custom_splits_from_anno(
        annotation_file=args.anno,
        split_col="custom_splits",
        sample_id_col="sample_id",
    )

    imagix = acx.Imagix(
        config=config,
        custom_splits=custom_splits,
    )

    result = imagix.run()

    if args.evaluate_param:
        imagix.evaluate(params=[args.evaluate_param])

    imagix.save(file_path=str(run_dir / "imagix.pkl"), save_all=True)

    print("Run finished successfully")
    print(f"Pipeline saved to: {run_dir / 'imagix.pkl'}")
    print(f"Final result object type: {type(result)}")


if __name__ == "__main__":
    main()