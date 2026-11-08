#!/usr/bin/env python3
"""
Computational benchmark for Imagix3D.

Purpose
-------
Characterize the computational requirements of the FINAL Imagix3D implementation
under controlled changes in batch size, network width (hidden_dim), and optionally
latent dimensionality.

The benchmark deliberately does NOT train every configuration to convergence.
Each configuration is run for a short, standardized number of epochs. The first
epoch can be treated as warm-up and excluded from timing summaries.

Primary outputs
---------------
benchmark_runs.csv
    One row per configuration x repetition.

benchmark_epochs.csv
    Per-epoch training time, throughput, and peak GPU memory.

benchmark_summary.csv
    Configuration-level summary across successful repetitions.

batch_size_envelope.csv
    Largest fully successful tested batch size and first tested OOM batch size
    for each hidden_dim.

benchmark_environment.json
    Hardware/software and benchmark settings.

Figures (PNG + PDF)
    peak_vram_vs_batch_size
    throughput_vs_batch_size
    epoch_time_vs_batch_size

Methodological notes
--------------------
- One GPU per benchmark process.
- CUDA synchronization is used around timed training/inference sections.
- Peak GPU memory is measured with PyTorch peak-memory counters.
- Peak host RAM is sampled from /proc on Linux during fit().
- Each configuration is executed in a FRESH Python subprocess. This isolates
  GPU allocator state and process-RAM peak measurements between configurations.
- OOM is an expected benchmark outcome and is recorded instead of aborting the
  entire benchmark.
- Checkpoint/dynamics capture is disabled in the benchmark trainer so that the
  timing reflects the training/validation computation rather than report-data
  serialization at a checkpoint epoch.
- No downstream NIHSS evaluation is performed here. Computational metrics should
  later be related to reconstruction/clinical metrics from fully trained models.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Generic helpers
# =============================================================================


def parse_int_list(text: str) -> List[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def bool_from_text(value: str) -> bool:
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot interpret {value!r} as boolean.")


def cuda_sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gibibytes(n_bytes: float) -> float:
    return float(n_bytes) / (1024.0 ** 3)


def safe_float(value: Any) -> float:
    try:
        arr = np.asarray(value)
        if arr.size == 1:
            return float(arr.item())
    except Exception:
        pass
    return float("nan")


def current_rss_bytes() -> int:
    """
    Current resident-set size for this Linux process.

    Falls back to resource.ru_maxrss if /proc is unavailable. The fallback is
    a process-lifetime peak rather than an instantaneous RSS value.
    """
    status_path = Path("/proc/self/status")
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                # kB in /proc/self/status
                return int(line.split()[1]) * 1024

    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss # type: ignore 
    # Linux reports KiB; macOS reports bytes.
    if sys.platform.startswith("linux"):
        return int(value * 1024)
    return int(value)


class RSSMonitor:
    """Background sampler for peak process RSS."""

    def __init__(self, interval_s: float = 0.05):
        self.interval_s = float(interval_s)
        self.peak_bytes = 0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.peak_bytes = max(self.peak_bytes, current_rss_bytes())
            except Exception:
                pass
            self._stop_event.wait(self.interval_s)

    def start(self) -> None:
        self.peak_bytes = current_rss_bytes()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 10))
        try:
            self.peak_bytes = max(self.peak_bytes, current_rss_bytes())
        except Exception:
            pass
        return self.peak_bytes


def get_git_commit(repo_dir: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def make_config_id(batch_size: int, hidden_dim: int, latent_dim: int) -> str:
    return f"hd{hidden_dim}_ld{latent_dim}_bs{batch_size}"


# =============================================================================
# Benchmark trainer
# =============================================================================


def make_benchmark_trainer(warmup_epochs: int):
    """
    Build a GeneralTrainer subclass with per-epoch computational instrumentation.

    This factory keeps the AUTOENCODIX import inside the runtime environment on
    the HPC rather than requiring AUTOENCODIX to be installed where this script
    is merely inspected.
    """
    import torch
    from autoencodix.trainers._general_trainer import GeneralTrainer # type: ignore

    class BenchmarkGeneralTrainer(GeneralTrainer):
        benchmark_warmup_epochs = int(warmup_epochs)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.benchmark_epoch_records: List[Dict[str, Any]] = []
            self.benchmark_trainable_parameters = int(
                sum(
                    parameter.numel()
                    for parameter in self._model.parameters()
                    if parameter.requires_grad
                )
            )
            self.benchmark_inference_total_time_s = float("nan")
            self.benchmark_inference_seconds_per_volume = float("nan")
            self.benchmark_inference_volumes = 0

        def _should_checkpoint(self, epoch: int) -> bool:
            # Deliberately avoid dynamics/checkpoint capture. A final checkpoint
            # epoch otherwise includes extra copying/serialization-related work
            # that is irrelevant to steady-state training throughput.
            return False

        def train(self, epochs_overwrite=None):
            epochs = self._config.epochs
            if epochs_overwrite:
                epochs = epochs_overwrite

            with self._fabric.autocast():
                self._grad_clip_warning_sent = False

                for epoch in range(epochs):
                    # Keep GeneralTrainer behavior with respect to buffers.
                    self._init_buffers()
                    self._model.train()

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()

                    start = time.perf_counter()

                    epoch_loss, epoch_sub_losses = self._train_epoch(
                        should_checkpoint=False,
                        epoch=epoch,
                    )

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    train_elapsed_s = time.perf_counter() - start

                    if torch.cuda.is_available():
                        peak_allocated_gib = gibibytes(
                            torch.cuda.max_memory_allocated()
                        )
                        peak_reserved_gib = gibibytes(
                            torch.cuda.max_memory_reserved()
                        )
                    else:
                        peak_allocated_gib = float("nan")
                        peak_reserved_gib = float("nan")

                    n_train = len(self._trainloader.dataset)
                    throughput = (
                        float(n_train) / train_elapsed_s
                        if train_elapsed_s > 0
                        else float("nan")
                    )

                    self._log_losses(
                        epoch=epoch,
                        split="train",
                        total_loss=epoch_loss,
                        sub_losses=epoch_sub_losses,
                        should_checkpoint=False,
                    )

                    valid_loss = float("nan")
                    valid_recon_loss = float("nan")

                    if self._validset:
                        valid_loss_value, valid_sub_losses = self._validate_epoch(
                            should_checkpoint=False,
                            epoch=epoch,
                        )
                        valid_loss = float(valid_loss_value)
                        if "recon_loss" in valid_sub_losses:
                            valid_recon_loss = float(
                                valid_sub_losses["recon_loss"]
                            )

                        self._log_losses(
                            epoch=epoch,
                            split="valid",
                            total_loss=valid_loss_value,
                            sub_losses=valid_sub_losses,
                            should_checkpoint=False,
                        )

                    self.benchmark_epoch_records.append(
                        {
                            "epoch": int(epoch),
                            "is_warmup": bool(
                                epoch < self.benchmark_warmup_epochs
                            ),
                            "train_epoch_time_s": float(train_elapsed_s),
                            "train_throughput_volumes_s": float(throughput),
                            "peak_gpu_allocated_gib": float(peak_allocated_gib),
                            "peak_gpu_reserved_gib": float(peak_reserved_gib),
                            "train_total_loss": float(epoch_loss),
                            "valid_total_loss": float(valid_loss),
                            "valid_reconstruction_loss": float(
                                valid_recon_loss
                            ),
                        }
                    )

            # Measure pure forward-pass latency before returning control to
            # BasePipeline.fit(), while the Fabric-wrapped model/loaders are
            # guaranteed to still be available.
            from torch.utils.data import DataLoader

            inference_dataset = (
                self._validloader.dataset
                if self._validloader is not None
                else self._trainloader.dataset
            )
            # Keep inference batch size fixed at 1 for ALL benchmark
            # configurations so that inference latency is not confounded by
            # the training batch-size sweep.
            inference_loader = DataLoader(
                inference_dataset,
                batch_size=1,
                shuffle=False,
                pin_memory=bool(self._config.pin_memory),
            )
            self._model.eval()

            with torch.inference_mode():
                # One untimed warm-up batch initializes CUDA kernels/caches.
                first_batch = next(iter(inference_loader))
                _, first_features, _ = first_batch
                first_features = first_features.to(
                    self.device,
                    non_blocking=bool(self._config.pin_memory),
                )
                _ = self._model(first_features)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                n_inference_volumes = 0
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_start = time.perf_counter()

                for _, features, _ in inference_loader:
                    features = features.to(
                        self.device,
                        non_blocking=bool(self._config.pin_memory),
                    )
                    _ = self._model(features)
                    n_inference_volumes += int(features.shape[0])

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_elapsed = time.perf_counter() - inference_start

            self.benchmark_inference_total_time_s = float(inference_elapsed)
            self.benchmark_inference_volumes = int(n_inference_volumes)
            self.benchmark_inference_seconds_per_volume = (
                float(inference_elapsed / n_inference_volumes)
                if n_inference_volumes > 0
                else float("nan")
            )

            # Preserve the original GeneralTrainer contract.
            self._result.model = next(self._model.children())
            return self._result

    return BenchmarkGeneralTrainer


# =============================================================================
# Single benchmark run
# =============================================================================


def measure_inference(trainer) -> Tuple[float, float, int]:
    """
    Measure pure forward-pass latency on the validation split.

    Returns:
        total_seconds, seconds_per_volume, n_volumes
    """
    import torch

    loader = trainer._validloader
    if loader is None:
        loader = trainer._trainloader

    model = trainer._model
    model.eval()

    # One warm-up batch to initialize kernels/caches.
    with trainer._fabric.autocast(), torch.inference_mode():
        first_batch = next(iter(loader))
        _, features, _ = first_batch
        _ = model(features)
        cuda_sync()

    n_volumes = 0

    cuda_sync()
    start = time.perf_counter()

    with trainer._fabric.autocast(), torch.inference_mode():
        for _, features, _ in loader:
            _ = model(features)
            n_volumes += int(features.shape[0])

    cuda_sync()
    elapsed = time.perf_counter() - start

    seconds_per_volume = (
        elapsed / n_volumes if n_volumes > 0 else float("nan")
    )

    return float(elapsed), float(seconds_per_volume), int(n_volumes)


def run_single(spec_path: Path, result_json: Path) -> int:
    import torch
    import autoencodix as acx # type: ignore

    from autoencodix.configs.imagix3d_config import Imagix3DConfig # type: ignore
    from autoencodix.configs.default_config import ( # type: ignore
        DataCase,
        DataConfig,
        DataInfo,
    )
    from autoencodix.utils._utils import custom_splits_from_anno # type: ignore

    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    seed = int(spec["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output: Dict[str, Any] = {
        **spec,
        "status": "failed",
        "error": "",
    }

    if not torch.cuda.is_available():
        output["error"] = "CUDA is not available."
        result_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
        return 2

    device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)

    output.update(
        {
            "gpu_name": props.name,
            "gpu_total_memory_gib": gibibytes(props.total_memory),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
        }
    )

    BenchmarkTrainer = make_benchmark_trainer(
        warmup_epochs=int(spec["warmup_epochs"])
    )

    config = Imagix3DConfig(
        # Architecture / optimizer
        beta=float(spec["beta"]),
        latent_dim=int(spec["latent_dim"]),
        hidden_dim=int(spec["hidden_dim"]),
        weight_decay=float(spec["weight_decay"]),
        learning_rate=float(spec["learning_rate"]),
        train_normalization=str(spec["train_normalization"]),
        keep_mu_positive=bool(spec["keep_mu_positive"]),

        # Data / loss
        data_case=DataCase.IMG_TO_IMG,
        img_path_col="filepath",
        spatial_shape_policy="crop_or_pad_to_shape",
        target_shape_3d=tuple(int(x) for x in spec["target_shape"]),
        batch_size=int(spec["batch_size"]),
        checkpoint_interval=int(spec["epochs"]) + 1000,
        epochs=int(spec["epochs"]),
        reconstruction_loss="mse",
        loss_reduction=str(spec["loss_reduction"]),
        scaling=str(spec["scaling"]),
        anneal_function=str(spec["anneal_function"]),
        normalize_nonzero_only=True,
        clamp_logvar=True,
        train_norm_groupsize=8,
        volume_scaling_strategy="train_global",

        # Computational settings
        device="cuda",
        n_gpus=1,
        float_precision=str(spec["float_precision"]),
        compile_model=bool(spec["compile_model"]),
        pin_memory=bool(spec["pin_memory"]),
        save_memory=bool(spec["save_memory"]),
        save_vram=bool(spec["save_vram"]),
        reproducible=bool(spec["reproducible"]),
        global_seed=seed,

        data_config=DataConfig(
            data_info={
                "IMG": DataInfo(
                    file_path=str(spec["volume_root"]),
                    scaling=str(spec["scaling"]),
                    data_type="IMG",
                ),
                "ANNO": DataInfo(
                    file_path=str(spec["annotation_file"]),
                    data_type="ANNOTATION",
                ),
            }
        ),
    )

    custom_splits = custom_splits_from_anno(
        annotation_file=str(spec["annotation_file"]),
        split_col="custom_splits",
        sample_id_col="sample_id",
    )

    try:
        # Instantiate with the benchmark trainer. Imagix3D follows the common
        # AUTOENCODIX pipeline interface and accepts trainer_type.
        imagix3d = acx.Imagix3D(
            config=config,
            custom_splits=custom_splits,
            trainer_type=BenchmarkTrainer,
        )

        preprocess_start = time.perf_counter()
        imagix3d.preprocess()
        output["preprocess_wall_time_s"] = (
            time.perf_counter() - preprocess_start
        )

        # Measure host RAM only for model setup + training.
        rss_monitor = RSSMonitor(interval_s=0.05)
        rss_monitor.start()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        fit_start = time.perf_counter()
        imagix3d.fit()
        cuda_sync()
        fit_elapsed = time.perf_counter() - fit_start

        peak_rss_bytes = rss_monitor.stop()

        trainer = imagix3d._trainer
        epoch_records = list(trainer.benchmark_epoch_records)

        if not epoch_records:
            raise RuntimeError(
                "Benchmark trainer produced no epoch records."
            )

        measured_records = [
            row for row in epoch_records if not row["is_warmup"]
        ]
        if not measured_records:
            raise ValueError(
                "No measured epochs remain after excluding warm-up epochs. "
                "Increase --epochs or reduce --warmup-epochs."
            )

        epoch_times = np.asarray(
            [r["train_epoch_time_s"] for r in measured_records],
            dtype=float,
        )
        throughputs = np.asarray(
            [r["train_throughput_volumes_s"] for r in measured_records],
            dtype=float,
        )
        peak_allocated = np.asarray(
            [r["peak_gpu_allocated_gib"] for r in measured_records],
            dtype=float,
        )
        peak_reserved = np.asarray(
            [r["peak_gpu_reserved_gib"] for r in measured_records],
            dtype=float,
        )

        infer_total_s = float(
            trainer.benchmark_inference_total_time_s
        )
        infer_s_per_volume = float(
            trainer.benchmark_inference_seconds_per_volume
        )
        infer_n = int(trainer.benchmark_inference_volumes)

        effective_batch_size = getattr(
            trainer._trainloader,
            "batch_size",
            int(spec["batch_size"]),
        )

        final_epoch = epoch_records[-1]

        output.update(
            {
                "status": "ok",
                "fit_wall_time_s": float(fit_elapsed),
                "peak_host_rss_gib": gibibytes(peak_rss_bytes),
                "trainable_parameters": int(
                    trainer.benchmark_trainable_parameters
                ),
                "effective_train_batch_size": int(effective_batch_size),
                "n_train": int(len(trainer._trainloader.dataset)),
                "n_valid": int(
                    len(trainer._validloader.dataset)
                    if trainer._validloader is not None
                    else 0
                ),
                "measured_epochs": int(len(measured_records)),
                "mean_train_epoch_time_s": float(epoch_times.mean()),
                "median_train_epoch_time_s": float(
                    np.median(epoch_times)
                ),
                "sd_train_epoch_time_s": float(
                    epoch_times.std(ddof=1)
                    if len(epoch_times) > 1
                    else 0.0
                ),
                "mean_train_throughput_volumes_s": float(
                    throughputs.mean()
                ),
                "peak_gpu_allocated_gib": float(
                    np.nanmax(peak_allocated)
                ),
                "peak_gpu_reserved_gib": float(
                    np.nanmax(peak_reserved)
                ),
                "inference_total_time_s": float(infer_total_s),
                "inference_seconds_per_volume": float(
                    infer_s_per_volume
                ),
                "inference_volumes": int(infer_n),
                # Sanity-only short-run losses. These are NOT final
                # performance metrics for the efficiency-performance analysis.
                "short_run_final_valid_total_loss": float(
                    final_epoch["valid_total_loss"]
                ),
                "short_run_final_valid_reconstruction_loss": float(
                    final_epoch["valid_reconstruction_loss"]
                ),
                "epoch_records": epoch_records,
            }
        )

    except RuntimeError as exc:
        # CUDA OOM is an expected endpoint of the batch-size envelope.
        message = str(exc)
        if "out of memory" in message.lower():
            try:
                peak_alloc = gibibytes(torch.cuda.max_memory_allocated())
                peak_res = gibibytes(torch.cuda.max_memory_reserved())
            except Exception:
                peak_alloc = float("nan")
                peak_res = float("nan")

            output.update(
                {
                    "status": "oom",
                    "error": message[:5000],
                    "peak_gpu_allocated_gib": peak_alloc,
                    "peak_gpu_reserved_gib": peak_res,
                }
            )
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        else:
            output["error"] = traceback.format_exc()[-12000:]
            result_json.write_text(
                json.dumps(output, indent=2, allow_nan=True),
                encoding="utf-8",
            )
            return 2

    except Exception:
        output["error"] = traceback.format_exc()[-12000:]
        result_json.write_text(
            json.dumps(output, indent=2, allow_nan=True),
            encoding="utf-8",
        )
        return 2

    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(
        json.dumps(output, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    return 0


# =============================================================================
# Aggregation and plotting
# =============================================================================


RUN_METRICS = [
    "fit_wall_time_s",
    "peak_host_rss_gib",
    "trainable_parameters",
    "effective_train_batch_size",
    "n_train",
    "n_valid",
    "measured_epochs",
    "mean_train_epoch_time_s",
    "median_train_epoch_time_s",
    "sd_train_epoch_time_s",
    "mean_train_throughput_volumes_s",
    "peak_gpu_allocated_gib",
    "peak_gpu_reserved_gib",
    "inference_total_time_s",
    "inference_seconds_per_volume",
    "inference_volumes",
    "short_run_final_valid_total_loss",
    "short_run_final_valid_reconstruction_loss",
]


def flatten_run_result(data: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        k: v
        for k, v in data.items()
        if k != "epoch_records"
    }
    return row


def build_epoch_table(all_results: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for result in all_results:
        for epoch_row in result.get("epoch_records", []) or []:
            rows.append(
                {
                    "config_id": result["config_id"],
                    "sweep": result["sweep"],
                    "repetition": result["repetition"],
                    "batch_size": result["batch_size"],
                    "hidden_dim": result["hidden_dim"],
                    "latent_dim": result["latent_dim"],
                    **epoch_row,
                }
            )

    return pd.DataFrame(rows)


def build_summary(
    runs_df: pd.DataFrame,
    repetitions: int,
) -> pd.DataFrame:
    config_columns = [
        "config_id",
        "sweep",
        "batch_size",
        "hidden_dim",
        "latent_dim",
    ]

    metric_columns = [
        "mean_train_epoch_time_s",
        "mean_train_throughput_volumes_s",
        "peak_gpu_allocated_gib",
        "peak_gpu_reserved_gib",
        "peak_host_rss_gib",
        "inference_seconds_per_volume",
        "trainable_parameters",
        "short_run_final_valid_reconstruction_loss",
    ]

    rows = []

    for _, group in runs_df.groupby(config_columns, dropna=False):
        first = group.iloc[0]
        ok = group[group["status"] == "ok"]
        oom = group[group["status"] == "oom"]
        failed = group[group["status"] == "failed"]

        row: Dict[str, Any] = {
            col: first[col] for col in config_columns
        }

        row.update(
            {
                "requested_repetitions": int(repetitions),
                "successful_repetitions": int(len(ok)),
                "oom_repetitions": int(len(oom)),
                "failed_repetitions": int(len(failed)),
                "fully_feasible": bool(len(ok) == repetitions),
            }
        )

        for metric in metric_columns:
            values = pd.to_numeric(
                ok[metric],
                errors="coerce",
            ).dropna()

            row[f"{metric}_mean"] = (
                float(values.mean()) if len(values) else np.nan
            )
            row[f"{metric}_sd"] = (
                float(values.std(ddof=1))
                if len(values) > 1
                else (0.0 if len(values) == 1 else np.nan)
            )

        rows.append(row)

    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["sweep", "hidden_dim", "latent_dim", "batch_size"]
    ).reset_index(drop=True)


def build_batch_envelope(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    bw = summary_df[
        summary_df["sweep"].str.contains("batch_width", regex=False)
    ].copy()

    if bw.empty:
        return pd.DataFrame()

    for hidden_dim, group in bw.groupby("hidden_dim"):
        group = group.sort_values("batch_size")

        feasible = group[group["fully_feasible"]]
        nonfeasible = group[~group["fully_feasible"]]

        max_feasible = (
            int(feasible["batch_size"].max())
            if not feasible.empty
            else np.nan
        )

        first_tested_nonfeasible = (
            int(nonfeasible["batch_size"].min())
            if not nonfeasible.empty
            else np.nan
        )

        rows.append(
            {
                "hidden_dim": int(hidden_dim), # type: ignore
                "max_tested_fully_feasible_batch_size": max_feasible,
                "first_tested_nonfeasible_batch_size": (
                    first_tested_nonfeasible
                ),
                "note": (
                    "Bmax is bounded by tested batch sizes; if no OOM/nonfeasible "
                    "configuration was observed, the true maximum is larger than "
                    "the largest tested batch size."
                ),
            }
        )

    return pd.DataFrame(rows)


def save_batch_plots(
    summary_df: pd.DataFrame,
    output_dir: Path,
    gpu_total_memory_gib: Optional[float],
) -> None:
    import matplotlib.pyplot as plt

    data = summary_df[
        summary_df["sweep"].str.contains("batch_width", regex=False)
    ].copy()

    if data.empty:
        return

    # ---- Peak VRAM vs batch size ----
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    for hidden_dim, group in data.groupby("hidden_dim"):
        group = group.sort_values("batch_size")
        ok = group[group["fully_feasible"]]

        ax.plot(
            ok["batch_size"],
            ok["peak_gpu_allocated_gib_mean"],
            marker="o",
            label=f"hidden_dim={int(hidden_dim)}", # type: ignore
        )

        bad = group[~group["fully_feasible"]]
        if not bad.empty and gpu_total_memory_gib is not None:
            first_bad = bad.iloc[0]
            ax.scatter(
                [first_bad["batch_size"]],
                [gpu_total_memory_gib],
                marker="x",
                s=80,
            )
            ax.annotate(
                "OOM/nonfeasible",
                (
                    first_bad["batch_size"],
                    gpu_total_memory_gib,
                ),
                xytext=(4, -12),
                textcoords="offset points",
                fontsize=8,
            )

    if gpu_total_memory_gib is not None:
        ax.axhline(
            gpu_total_memory_gib,
            linestyle="--",
            linewidth=1,
            label="GPU physical VRAM",
        )

    ax.set_xlabel("Batch size")
    ax.set_ylabel("Peak allocated GPU memory [GiB]")
    ax.set_title("Imagix3D peak VRAM vs. batch size")
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_dir / "peak_vram_vs_batch_size.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "peak_vram_vs_batch_size.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # ---- Throughput vs batch size ----
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    for hidden_dim, group in data.groupby("hidden_dim"):
        group = group.sort_values("batch_size")
        ok = group[group["fully_feasible"]]

        ax.plot(
            ok["batch_size"],
            ok["mean_train_throughput_volumes_s_mean"],
            marker="o",
            label=f"hidden_dim={int(hidden_dim)}", # type: ignore
        )

    ax.set_xlabel("Batch size")
    ax.set_ylabel("Training throughput [volumes/s]")
    ax.set_title("Imagix3D training throughput vs. batch size")
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_dir / "throughput_vs_batch_size.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "throughput_vs_batch_size.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # ---- Epoch time vs batch size ----
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    for hidden_dim, group in data.groupby("hidden_dim"):
        group = group.sort_values("batch_size")
        ok = group[group["fully_feasible"]]

        ax.plot(
            ok["batch_size"],
            ok["mean_train_epoch_time_s_mean"],
            marker="o",
            label=f"hidden_dim={int(hidden_dim)}", # type: ignore
        )

    ax.set_xlabel("Batch size")
    ax.set_ylabel("Training time per epoch [s]")
    ax.set_title("Imagix3D training time vs. batch size")
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_dir / "epoch_time_vs_batch_size.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "epoch_time_vs_batch_size.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def create_tradeoff_template(
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Write a table ready to be joined with FINAL model-quality metrics.

    The short-run loss is kept in benchmark_summary.csv for diagnostics only.
    It should not be interpreted as converged reconstruction performance.
    """
    keep = [
        "config_id",
        "sweep",
        "batch_size",
        "hidden_dim",
        "latent_dim",
        "fully_feasible",
        "peak_gpu_allocated_gib_mean",
        "mean_train_epoch_time_s_mean",
        "mean_train_throughput_volumes_s_mean",
        "inference_seconds_per_volume_mean",
        "trainable_parameters_mean",
    ]

    tradeoff = summary_df[
        [col for col in keep if col in summary_df.columns]
    ].copy()

    tradeoff["final_validation_reconstruction_loss"] = np.nan
    tradeoff["final_nihss_metric"] = np.nan
    tradeoff["final_nihss_metric_name"] = ""

    tradeoff.to_csv(
        output_dir / "efficiency_performance_tradeoff_template.csv",
        index=False,
    )


# =============================================================================
# Master benchmark orchestration
# =============================================================================


def generate_configs(args) -> List[Dict[str, Any]]:
    configs: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

    # Core benchmark: batch-size envelope at two (or more) widths.
    for hidden_dim in args.hidden_dims:
        for batch_size in args.batch_sizes:
            key = (
                int(batch_size),
                int(hidden_dim),
                int(args.reference_latent_dim),
            )
            configs[key] = {
                "batch_size": key[0],
                "hidden_dim": key[1],
                "latent_dim": key[2],
                "sweep": "batch_width",
            }

    # Small latent-dimensionality sensitivity analysis.
    if args.run_latent_sweep:
        for latent_dim in args.latent_dims:
            key = (
                int(args.reference_batch_size),
                int(args.reference_hidden_dim),
                int(latent_dim),
            )
            if key in configs:
                configs[key]["sweep"] = "batch_width+latent"
            else:
                configs[key] = {
                    "batch_size": key[0],
                    "hidden_dim": key[1],
                    "latent_dim": key[2],
                    "sweep": "latent",
                }

    result = list(configs.values())

    for config in result:
        config["config_id"] = make_config_id(
            config["batch_size"],
            config["hidden_dim"],
            config["latent_dim"],
        )

    return sorted(
        result,
        key=lambda x: (
            x["hidden_dim"],
            x["latent_dim"],
            x["batch_size"],
        ),
    )


def collect_environment(args, output_dir: Path) -> Dict[str, Any]:
    import torch

    env: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "repo_git_commit": get_git_commit(Path(args.repo_dir)),
        "benchmark_settings": vars(args),
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        env.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_gib": gibibytes(props.total_memory),
                "gpu_count_visible": torch.cuda.device_count(),
            }
        )

    (output_dir / "benchmark_environment.json").write_text(
        json.dumps(env, indent=2, default=str),
        encoding="utf-8",
    )

    return env


def run_master(args) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    volume_root = Path(args.data_path) / args.folder
    annotation_file = Path(args.data_path) / args.annotation

    if not volume_root.exists():
        raise FileNotFoundError(
            f"Volume directory does not exist: {volume_root}"
        )
    if not annotation_file.is_file():
        raise FileNotFoundError(
            f"Annotation file does not exist: {annotation_file}"
        )

    if args.warmup_epochs >= args.epochs:
        raise ValueError(
            "warmup_epochs must be smaller than epochs."
        )

    env_info = collect_environment(args, output_dir)
    configs = generate_configs(args)

    design_df = pd.DataFrame(configs)
    design_df.to_csv(
        output_dir / "benchmark_design.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("Imagix3D computational benchmark")
    print("=" * 80)
    print(f"Data: {volume_root}")
    print(f"Annotation: {annotation_file}")
    print(f"Output: {output_dir}")
    print(f"Configurations: {len(configs)}")
    print(f"Repetitions/configuration: {args.repetitions}")
    print(
        f"Epochs/run: {args.epochs} "
        f"({args.warmup_epochs} warm-up excluded)"
    )
    print()

    all_results: List[Dict[str, Any]] = []

    run_specs_dir = output_dir / "run_specs"
    run_results_dir = output_dir / "run_results"
    run_specs_dir.mkdir(exist_ok=True)
    run_results_dir.mkdir(exist_ok=True)

    for config_idx, config in enumerate(configs, start=1):
        for repetition in range(1, args.repetitions + 1):
            run_seed = int(args.seed + repetition - 1)

            spec = {
                **config,
                "repetition": int(repetition),
                "seed": run_seed,
                "volume_root": str(volume_root),
                "annotation_file": str(annotation_file),
                "epochs": int(args.epochs),
                "warmup_epochs": int(args.warmup_epochs),
                "target_shape": list(args.target_shape),
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "beta": float(args.beta),
                "train_normalization": str(args.train_normalization),
                "keep_mu_positive": bool(args.keep_mu_positive),
                "anneal_function": str(args.anneal_function),
                "loss_reduction": str(args.loss_reduction),
                "scaling": str(args.scaling),
                "float_precision": str(args.float_precision),
                "compile_model": bool(args.compile_model),
                "pin_memory": bool(args.pin_memory),
                "save_memory": bool(args.save_memory),
                "save_vram": bool(args.save_vram),
                "reproducible": bool(args.reproducible),
            }

            stem = f"{config['config_id']}_rep{repetition:02d}"
            spec_path = run_specs_dir / f"{stem}.json"
            result_path = run_results_dir / f"{stem}.json"

            spec_path.write_text(
                json.dumps(spec, indent=2),
                encoding="utf-8",
            )

            print(
                f"[{config_idx}/{len(configs)}] "
                f"{config['config_id']} | repetition {repetition}/"
                f"{args.repetitions}"
            )

            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--single-run",
                "--spec",
                str(spec_path),
                "--result-json",
                str(result_path),
            ]

            completed = subprocess.run(cmd)

            if not result_path.exists():
                raise RuntimeError(
                    f"Benchmark subprocess produced no result file:\n"
                    f"{result_path}\n"
                    f"Exit code: {completed.returncode}"
                )

            result = json.loads(
                result_path.read_text(encoding="utf-8")
            )
            all_results.append(result)

            status = result.get("status")
            print(f"    status={status}")

            if status == "ok":
                print(
                    "    "
                    f"peak VRAM={result.get('peak_gpu_allocated_gib', np.nan):.2f} GiB | "
                    f"epoch={result.get('mean_train_epoch_time_s', np.nan):.2f} s | "
                    f"throughput={result.get('mean_train_throughput_volumes_s', np.nan):.3f} vol/s"
                )
            elif status == "oom":
                print("    expected benchmark endpoint: CUDA OOM")
            else:
                print(result.get("error", "Unknown error"))
                raise RuntimeError(
                    f"Unexpected benchmark failure for {stem}."
                )

    runs_df = pd.DataFrame(
        [flatten_run_result(result) for result in all_results]
    )

    runs_df.to_csv(
        output_dir / "benchmark_runs.csv",
        index=False,
    )

    epoch_df = build_epoch_table(all_results)
    epoch_df.to_csv(
        output_dir / "benchmark_epochs.csv",
        index=False,
    )

    summary_df = build_summary(
        runs_df,
        repetitions=args.repetitions,
    )
    summary_df.to_csv(
        output_dir / "benchmark_summary.csv",
        index=False,
    )

    envelope_df = build_batch_envelope(summary_df)
    envelope_df.to_csv(
        output_dir / "batch_size_envelope.csv",
        index=False,
    )

    gpu_total = env_info.get("gpu_total_memory_gib")
    save_batch_plots(
        summary_df,
        output_dir,
        float(gpu_total) if gpu_total is not None else None,
    )

    create_tradeoff_template(
        summary_df,
        output_dir,
    )

    print()
    print("=" * 80)
    print("Benchmark complete")
    print("=" * 80)
    print(output_dir / "benchmark_runs.csv")
    print(output_dir / "benchmark_epochs.csv")
    print(output_dir / "benchmark_summary.csv")
    print(output_dir / "batch_size_envelope.csv")
    print(output_dir / "benchmark_environment.json")
    print(output_dir / "efficiency_performance_tradeoff_template.csv")

    if not envelope_df.empty:
        print()
        print("Batch-size envelope:")
        print(envelope_df.to_string(index=False))

    return 0


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Imagix3D computational benchmark"
    )

    parser.add_argument("--single-run", action="store_true")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--result-json", type=Path)

    # Master-mode inputs
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--data-path")
    parser.add_argument("--folder", default="ncct_flat")
    parser.add_argument("--annotation", default="ncct_anno_dst.csv")
    parser.add_argument("--output-dir")

    # Controlled benchmark design
    parser.add_argument(
        "--batch-sizes",
        type=parse_int_list,
        default=parse_int_list("8,16,24,35,48,64"),
    )
    parser.add_argument(
        "--hidden-dims",
        type=parse_int_list,
        default=parse_int_list("16,32"),
    )
    parser.add_argument(
        "--latent-dims",
        type=parse_int_list,
        default=parse_int_list("64,80,96"),
    )
    parser.add_argument(
        "--run-latent-sweep",
        type=bool_from_text,
        default=True,
    )

    parser.add_argument("--reference-batch-size", type=int, default=35)
    parser.add_argument("--reference-hidden-dim", type=int, default=16)
    parser.add_argument("--reference-latent-dim", type=int, default=80)

    # Short-run timing protocol
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    # Fixed Imagix3D hyperparameters
    parser.add_argument("--learning-rate", type=float, required=False)
    parser.add_argument("--weight-decay", type=float, required=False)
    parser.add_argument("--beta", type=float, required=False)
    parser.add_argument("--train-normalization", default="batch")
    parser.add_argument("--keep-mu-positive", type=bool_from_text, default=False)
    parser.add_argument("--anneal-function", default="logistic-late")
    parser.add_argument("--loss-reduction", default="mean")
    parser.add_argument("--scaling", default="MINMAX")

    # Fixed volume/hardware settings
    parser.add_argument(
        "--target-shape",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default=(160, 192, 160),
    )
    parser.add_argument("--float-precision", default="32")
    parser.add_argument("--compile-model", type=bool_from_text, default=False)
    parser.add_argument("--pin-memory", type=bool_from_text, default=False)
    parser.add_argument("--save-memory", type=bool_from_text, default=False)
    parser.add_argument("--save-vram", type=bool_from_text, default=False)
    parser.add_argument("--reproducible", type=bool_from_text, default=False)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.single_run:
        if args.spec is None or args.result_json is None:
            parser.error("--single-run requires --spec and --result-json")
        return run_single(args.spec, args.result_json)

    required = {
        "--data-path": args.data_path,
        "--output-dir": args.output_dir,
        "--learning-rate": args.learning_rate,
        "--weight-decay": args.weight_decay,
        "--beta": args.beta,
    }

    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(
            "Master benchmark mode requires: " + ", ".join(missing)
        )

    return run_master(args)


if __name__ == "__main__":
    raise SystemExit(main())
