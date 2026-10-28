from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import autoencodix as acx
from autoencodix.configs import DataCase, DataConfig, DataInfo
from autoencodix.configs.xmodalix3d_config import XModalix3DConfig


SUPPORTED_VOLUME_EXTENSIONS = (".nii", ".nii.gz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the two-modality XModalix3D pipeline."
    )

    parser.add_argument("--workdir", type=Path, required=True)

    parser.add_argument("--ncct-images", type=Path, required=True)
    parser.add_argument("--ncct-anno", type=Path, required=True)

    parser.add_argument("--cbf-images", type=Path, required=True)
    parser.add_argument("--cbf-anno", type=Path, required=True)

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
    parser.add_argument(
        "--save-intermediate-recons",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pretrain-save-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

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

    parser.add_argument("--train-normalization", type=str, default="batch")
    parser.add_argument("--train-norm-groupsize", type=int, default=8)
    parser.add_argument("--n-gpus", typ=int, default=1)
    parser.add_argument(
        "--save-vram", 
        action=argparse.BooleanOptionalAction, 
        default=True
    )

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

    # Cross-modal alignment weights
    parser.add_argument("--gamma", type=float, default=30.0)
    parser.add_argument("--delta-pair", type=float, default=5.0)
    parser.add_argument("--delta-class", type=float, default=0.0)

    # Translation direction. All modalities still participate in alignment.
    parser.add_argument(
        "--from-modality",
        type=str,
        choices=("NCCT", "CBF"),
        default="NCCT",
    )
    parser.add_argument(
        "--to-modality",
        type=str,
        choices=("NCCT", "CBF"),
        default="CBF",
    )
    
    parser.add_argument(
        "--annotation-columns",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Annotation columns to retain as metadata, "
            "e.g. --annotation-columns median_split"
        ),
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


def _read_available_annotation(
    *,
    annotation_file: Path,
    image_dir: Path,
    modality: str,
    img_path_col: str = "filepath",
    split_col: str = "custom_splits",
) -> pd.DataFrame:
    """Read one modality annotation and retain samples with an available volume.

    The file is read with ``index_col=0`` to mirror ``VolumeDataReader``.
    Therefore the DataFrame index is treated as the sample identifier here,
    exactly as it is when the 3D reader creates ``ImgData`` objects.
    """

    if not annotation_file.is_file():
        raise FileNotFoundError(
            f"{modality} annotation file does not exist: {annotation_file}"
        )
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"{modality} image directory does not exist: {image_dir}"
        )

    annotation = pd.read_csv(
        annotation_file,
        index_col=0,
        engine="python",
    )
    annotation.index = annotation.index.astype(str)

    if annotation.index.has_duplicates:
        duplicate_ids = annotation.index[
            annotation.index.duplicated()
        ].unique().tolist()
        raise ValueError(
            f"{modality} annotation contains duplicate index/sample IDs. "
            f"First duplicates: {duplicate_ids[:10]}"
        )

    required_columns = {img_path_col, split_col}
    missing_columns = required_columns - set(annotation.columns)
    if missing_columns:
        raise ValueError(
            f"{modality} annotation is missing required column(s): "
            f"{sorted(missing_columns)}"
        )

    split_values = (
        annotation[split_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    allowed_splits = {"train", "valid", "test"}
    invalid_splits = set(split_values.unique()) - allowed_splits
    if invalid_splits:
        raise ValueError(
            f"{modality} annotation contains invalid split label(s): "
            f"{sorted(invalid_splits)}. "
            f"Allowed values are {sorted(allowed_splits)}."
        )

    available_files = {
        path.name
        for path in image_dir.iterdir()
        if path.is_file()
        and path.name.lower().endswith(SUPPORTED_VOLUME_EXTENSIONS)
    }

    annotation_paths = annotation[img_path_col].astype(str)
    available_mask = annotation_paths.isin(available_files)

    missing_volume_count = int((~available_mask).sum())
    if missing_volume_count:
        print(
            f"{modality}: ignoring {missing_volume_count} annotation row(s) "
            "whose volume file is not present in the image directory."
        )

    annotation = annotation.loc[available_mask].copy()

    if annotation.empty:
        raise ValueError(
            f"No usable {modality} samples remain after matching the "
            f"annotation to {image_dir}."
        )

    return annotation


def _build_common_custom_splits(
    args: argparse.Namespace,
    run_dir: Path,
    *,
    img_path_col: str = "filepath",
    split_col: str = "custom_splits",
) -> Tuple[Dict[str, np.ndarray], Path, Dict[str, int]]:
    """Build custom split indices on the actual paired two-modality cohort.

    XModalix3D uses only patients available in both modalities. The split
    indices must therefore be relative to that common patient set rather than
    to the complete annotation table of any single modality.
    """

    modality_sources = {
        "NCCT": (args.ncct_anno, args.ncct_images),
        "CBF": (args.cbf_anno, args.cbf_images),
    }

    annotations: Dict[str, pd.DataFrame] = {
        modality: _read_available_annotation(
            annotation_file=annotation_file,
            image_dir=image_dir,
            modality=modality,
            img_path_col=img_path_col,
            split_col=split_col,
        )
        for modality, (annotation_file, image_dir)
        in modality_sources.items()
    }

    common_ids = sorted(
        set(annotations["NCCT"].index)
        & set(annotations["CBF"].index)
    )

    if not common_ids:
        raise ValueError(
            "NCCT and CBF have no common sample IDs after matching "
            "annotation rows to available volume files."
        )

    reference_splits = (
        annotations["NCCT"]
        .loc[common_ids, split_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # A paired patient must have the same predefined split in every modality.
    for modality in ("CBF",):
        modality_splits = (
            annotations[modality]
            .loc[common_ids, split_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        mismatch_mask = (
            reference_splits.to_numpy()
            != modality_splits.to_numpy()
        )

        if mismatch_mask.any():
            mismatch_positions = np.flatnonzero(mismatch_mask)
            examples = [
                (
                    common_ids[position],
                    reference_splits.iloc[position],
                    modality_splits.iloc[position],
                )
                for position in mismatch_positions[:10]
            ]
            raise ValueError(
                f"Custom split assignments differ between NCCT and "
                f"{modality} for {int(mismatch_mask.sum())} common sample(s). "
                f"First mismatches (sample_id, NCCT, {modality}): {examples}"
            )

    # NCCT serves as the canonical clinical annotation. Its rows are ordered
    # exactly like common_ids. XModal3DPreprocessor uses the same deterministic
    # common-ID order before invoking the DataSplitter.
    paired_annotation = annotations["NCCT"].loc[common_ids].copy()
    paired_annotation_path = run_dir / "paired_annotation.csv"
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

    split_sizes = {
        split_name: int(len(indices))
        for split_name, indices in custom_splits.items()
    }

    print("Paired two-modality cohort:")
    print(f"  common samples: {len(common_ids)}")
    for split_name in ("train", "valid", "test"):
        print(f"  {split_name}: {split_sizes[split_name]}")
    print(f"  paired annotation: {paired_annotation_path}")

    return custom_splits, paired_annotation_path, split_sizes


def build_config(
    args: argparse.Namespace,
    paired_annotation_path: Path,
) -> XModalix3DConfig:
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
        pretrain_save_memory=args.pretrain_save_memory,
        save_intermediate_recons=args.save_intermediate_recons,

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
        n_gpus=args.n_gpus,
        save_vram=args.save_vram,

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
                "ANNO": DataInfo(
                    file_path=str(paired_annotation_path),
                    data_type="ANNOTATION",
                ),
            },
            annotation_columns=args.annotation_columns,
        ),
    )


def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.workdir / "results" / f"{args.run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    custom_splits, paired_annotation_path, split_sizes = (
        _build_common_custom_splits(
            args=args,
            run_dir=run_dir,
        )
    )

    config = build_config(
        args=args,
        paired_annotation_path=paired_annotation_path,
    )

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
        },

        "paired_annotation": str(paired_annotation_path),
        "paired_split_sizes": split_sizes,

        "translation_from": args.from_modality,
        "translation_to": args.to_modality,

        "target_shape": args.target_shape,
        "epochs": args.epochs,
        "pretrain_epochs": args.pretrain_epochs,
        "latent_dim": args.latent_dim,
        "save_intermediate_recons": args.save_intermediate_recons,
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
        
        "n_gpus": args.n_gpus,
        "save_vram": args.save_vram,

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

    print("Starting XModalix3D two-modality run")
    print(f"Run directory: {run_dir}")
    print(f"NCCT image directory: {args.ncct_images}")
    print(f"CBF image directory: {args.cbf_images}")
    print(f"Joint epochs: {args.epochs}")
    print(f"Pretraining epochs per modality: {args.pretrain_epochs}")
    print(
        f"Translation direction: "
        f"{args.from_modality} -> {args.to_modality}"
    )
    print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}")

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
