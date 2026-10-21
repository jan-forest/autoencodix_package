"""Offline exporter for the BBOmix evaluation tables used by conditional proposals.

Dev-only script -- requires `syne-tune` (see the `dev` dependency group in
`pyproject.toml`) and the `bbomix_*` blackboxes in Syne Tune's local blackbox
repository (``~/.blackbox-repository``). It is never imported by the
`autoencodix` package.

`propose_initial_config(..., fixed={...})` fits a surrogate on these tables at
call time to propose the free hyperparameters *given* the fixed ones. This script
reduces each blackbox to what that needs: one row per (task, config) with the
hyperparameters, the seed-mean downstream score at the final epoch, and the
seed-mean validation reconstruction loss at each budget in `BUDGET_GRID`.

Outputs (upload both to the `autoencodix/initial-configs` Hugging Face dataset):
    <out>/evaluations/<architecture>.parquet
    <out>/search_space.json

Usage:
    python benchmarking/tuning/export_evaluations.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from syne_tune.blackbox_repository.blackbox_tabular import (
    BlackboxTabular,
)
from syne_tune.blackbox_repository.blackbox_tabular import (
    deserialize as deserialize_tabular,
)
from syne_tune.blackbox_repository.repository import repository_path

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger("export_evaluations")

ARCHITECTURES = ["vanillix", "varix", "ontix", "disentanglix"]
DATASETS = ["tcga", "schc"]
BUDGET_GRID = [10, 25, 50, 100, 150, 200, 300]
DOWNSTREAM_METRIC = "metric_avg_ml_task_performance"
RECONSTRUCTION_METRIC = "metric_valid_recon_loss"

# Values held constant during the BBOmix sweep (see the BBOmix archive's
# `benchmarking/configs/search_space.yaml`), merged into every proposed config.
CONSTANTS: Dict[str, Any] = {
    "epochs": 300,
    "checkpoint_interval": 300,
    "loss_reduction": "sum",
}


def _load_blackbox(arch: str, dataset: str) -> Dict[str, BlackboxTabular]:
    return deserialize_tabular(repository_path / f"bbomix_{arch}_{dataset}")


def _task_frame(dataset: str, task: str, bb: BlackboxTabular) -> pd.DataFrame:
    names = bb.objectives_names
    evals = bb.objectives_evaluations  # (configs, seeds, epochs, objectives)
    downstream = evals[:, :, -1, names.index(DOWNSTREAM_METRIC)]
    recon = evals[..., names.index(RECONSTRUCTION_METRIC)]

    df = bb.hyperparameters.reset_index(drop=True).copy()
    df.insert(0, "config_id", np.arange(len(df)))
    df.insert(0, "task", task)
    df.insert(0, "dataset", dataset)
    df["downstream"] = np.nanmean(downstream, axis=1)
    for budget in BUDGET_GRID:
        df[f"reconstruction_{budget}"] = np.nanmean(recon[:, :, budget - 1], axis=1)
    return df


def _search_space(frames: List[pd.DataFrame], bb: BlackboxTabular) -> Dict[str, Any]:
    """Observed values rather than the declared domain: the sweep sampled the
    integer hyperparameters from small grids (e.g. latent_dim in 2..64 by powers
    of two) and some continuous ones from a narrower range than declared, and the
    surrogate can't extrapolate beyond the data."""
    hps = pd.concat(frames)[list(bb.hyperparameters.columns)]
    space: Dict[str, Any] = {}
    for name, domain in bb.configuration_space.items():
        if pd.api.types.is_integer_dtype(bb.hyperparameters[name]):
            choices = sorted(int(v) for v in hps[name].unique())
            space[name] = {"choices": choices, "low": choices[0], "high": choices[-1]}
        else:
            scale = (
                "log" if type(domain.sampler).__name__ == "_LogUniform" else "linear"
            )
            space[name] = {
                "scale": scale,
                "low": float(hps[name].min()),
                "high": float(hps[name].max()),
            }
    return space


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    args = parser.parse_args()

    eval_dir = args.out / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    search_space: Dict[str, Any] = {
        "constants": CONSTANTS,
        "budgets": BUDGET_GRID,
        "architectures": {},
    }

    for arch in ARCHITECTURES:
        frames: List[pd.DataFrame] = []
        for dataset in DATASETS:
            for task, bb in _load_blackbox(arch, dataset).items():
                frames.append(_task_frame(dataset, task, bb))
        table = pd.concat(frames, ignore_index=True)
        path = eval_dir / f"{arch}.parquet"
        table.to_parquet(path, index=False)
        search_space["architectures"][arch] = _search_space(frames, bb)
        logger.info(
            "%s: %d rows, %d tasks -> %s",
            arch,
            len(table),
            table["task"].nunique(),
            path,
        )

    space_path = args.out / "search_space.json"
    space_path.write_text(json.dumps(search_space, indent=2))
    logger.info("Wrote %s", space_path)


if __name__ == "__main__":
    main()
