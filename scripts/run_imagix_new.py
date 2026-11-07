from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

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
    parser.add_argument("--image-size", type=int, default=192)

    # Defaults follow DefaultConfig
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



def verify_custom_split(
    pipeline,
    annotation_file: Path,
    split_col: str = "custom_splits",
    sample_id_col: str = "sample_id",
) -> None:
    """Verify that preprocessed datasets exactly match annotation-defined splits.

    This function must be called after ``pipeline.preprocess()`` and before
    ``pipeline.fit()``. It raises ``RuntimeError`` if any discrepancy is found.
    """

    anno = pd.read_csv(
        annotation_file,
        dtype={sample_id_col: str},
    )

    if sample_id_col not in anno.columns:
        raise RuntimeError(
            f"Split verification failed: column '{sample_id_col}' not found."
        )

    if split_col not in anno.columns:
        raise RuntimeError(
            f"Split verification failed: column '{split_col}' not found."
        )

    anno[sample_id_col] = (
        anno[sample_id_col]
        .astype(str)
        .str.strip()
    )

    anno[split_col] = (
        anno[split_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    duplicated_expected = anno.loc[
        anno[sample_id_col].duplicated(keep=False),
        sample_id_col,
    ].unique()

    if len(duplicated_expected) > 0:
        raise RuntimeError(
            "Split verification failed: duplicate sample IDs found in annotation. "
            f"Examples: {duplicated_expected[:10].tolist()}"
        )

    expected = {
        split: set(
            anno.loc[
                anno[split_col] == split,
                sample_id_col,
            ]
        )
        for split in ("train", "valid", "test")
    }

    actual = {}

    for split in ("train", "valid", "test"):
        dataset = getattr(
            pipeline.result.datasets,
            split,
        )

        if dataset is None:
            ids = []
        else:
            ids = [
                str(sid).strip()
                for sid in dataset.sample_ids
            ]

        if len(ids) != len(set(ids)):
            seen = set()
            duplicates = []

            for sid in ids:
                if sid in seen and sid not in duplicates:
                    duplicates.append(sid)
                seen.add(sid)

            raise RuntimeError(
                f"Split verification failed: duplicate sample IDs inside '{split}'. "
                f"Examples: {duplicates[:10]}"
            )

        actual[split] = set(ids)

    for split in ("train", "valid", "test"):
        missing = expected[split] - actual[split]
        unexpected = actual[split] - expected[split]

        if missing or unexpected:
            raise RuntimeError(
                f"Custom split verification FAILED for '{split}'.\n"
                f"Expected: {len(expected[split])} samples\n"
                f"Actual:   {len(actual[split])} samples\n"
                f"Missing expected IDs: {sorted(missing)[:10]}\n"
                f"Unexpected IDs: {sorted(unexpected)[:10]}"
            )

    for split_a, split_b in (
        ("train", "valid"),
        ("train", "test"),
        ("valid", "test"),
    ):
        overlap = actual[split_a] & actual[split_b]

        if overlap:
            raise RuntimeError(
                "Custom split verification FAILED: "
                f"{split_a} and {split_b} overlap. "
                f"Examples: {sorted(overlap)[:10]}"
            )

    print("\nCustom split verification PASSED.")

    for split in ("train", "valid", "test"):
        print(
            f"{split}: "
            f"{len(actual[split])} samples"
        )

    print()

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

    # Preprocess first so that the actual train/valid/test membership can be checked
    # before any model training starts.
    imagix.preprocess()

    # Hard safety check: abort immediately if the preprocessed datasets do not
    # exactly match the split assignments in the annotation file.
    verify_custom_split(
        pipeline=imagix,
        annotation_file=args.anno,
        split_col="custom_splits",
        sample_id_col="sample_id",
    )

    # Continue with the remaining steps normally performed by imagix.run().
    imagix.fit()
    result = imagix.predict()
    imagix.visualize()

    if args.evaluate_param:
        imagix.evaluate(params=[args.evaluate_param])

    imagix.save(file_path=str(run_dir / "imagix.pkl"), save_all=True)

    print("Run finished successfully")
    print(f"Pipeline saved to: {run_dir / 'imagix.pkl'}")
    print(f"Final result object type: {type(result)}")


if __name__ == "__main__":
    main()