#!/bin/bash
#SBATCH --job-name=hpo_xmodalix3d_3mod_01
#SBATCH --account=p_scads_autoencodix
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=/data/horse/ws/baeuchl-imagix3d/logs/%x-%j.out
#SBATCH --error=/data/horse/ws/baeuchl-imagix3d/logs/%x-%j.err

set -euo pipefail

module --force purge
module load release/24.04
module load GCCcore/12.3.0
module load Python/3.11.3

source /data/horse/ws/baeuchl-imagix3d/venvs/alpha/bin/activate

cd /home/baeuchl/autoencodix_package

echo "===== GPU / CUDA preflight ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-not set}"

nvidia-smi

srun python - <<'PY'
import os
import torch

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch version:", torch.__version__)
print("torch CUDA version:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

export PYTHONUNBUFFERED=1
export DATA_PATH="/data/horse/ws/baeuchl-imagix3d/data/stroke_data"

# -----------------------------------------------------------------------------
# FIXED SETTINGS FOR THE THREE-MODALITY XModalix3D MODEL
# -----------------------------------------------------------------------------

export LATENT_DIM=48

# These are shared by all three 3D VAEs in the current XModalix3D implementation.
export HIDDEN_DIM=16
export TRAIN_NORMALIZATION="batch"
export BATCH_SIZE=35
export EPOCHS=250
export PRETRAIN_EPOCHS=50
export TARGET_D=160
export TARGET_H=192
export TARGET_W=160

# -----------------------------------------------------------------------------
# FIXED MODALITY-SPECIFIC HYPERPARAMETERS
# -----------------------------------------------------------------------------

export NCCT_LR=0.000031484983656998606
export NCCT_WD=0.000007737374408547984
export NCCT_BETA=0.0000006718196015256182

export CBF_LR=0.000010096108555965044
export CBF_WD=0.00001794364217052906
export CBF_BETA=0.0000000032465959108014116

export CBV_LR=0.0004628223166023
export CBV_WD=0.000005136186358487116
export CBV_BETA=0.00003832427557775615

# Fail early rather than accidentally starting a large HPO with placeholder values.
REQUIRED_VARS=(
    LATENT_DIM
    NCCT_LR NCCT_WD NCCT_BETA
    CBF_LR CBF_WD CBF_BETA
    CBV_LR CBV_WD CBV_BETA
)

for var_name in "${REQUIRED_VARS[@]}"; do
    value="${!var_name:-}"
    if [[ -z "${value}" || "${value}" == "REPLACE_ME" ]]; then
        echo "ERROR: ${var_name} has not been set. Edit run_hpo_3mod_xmodalix3d.sh first."
        exit 1
    fi
done

# Primary HPO target
export METRIC="alignment_score"

# Run 3 independent HPO procedures with the same settings but different seeds
export HPO_RUN_INDEX="01"
export HPO_RANDOM_SEED=42

export MAX_WALLCLOCK_HOURS=11.75
export N_WORKERS=4

export MLFLOW_ALLOW_FILE_STORE=true
export MLFLOW_TRACKING_URI="file:/data/horse/ws/baeuchl-imagix3d/mlruns"

HPO_ROOT="/data/horse/ws/baeuchl-imagix3d/hpo"
RUN_NAME="xmodalix3d_3mod_${HPO_RUN_INDEX}_synetune_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${HPO_ROOT}/${RUN_NAME}"

mkdir -p "${OUT_DIR}"
export OUT_DIR

echo "Starting three-modality XModalix3D Syne Tune HPO"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Repository: $(pwd)"
echo "Data path: ${DATA_PATH}"
echo "Output directory: ${OUT_DIR}"
echo "Metric: ${METRIC}"
echo "HPO random seed: ${HPO_RANDOM_SEED}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"

srun python - <<'PY'
import json
import os
import pickle
import sys
from pathlib import Path

repo = Path.cwd()
sys.path.insert(0, str(repo / "scripts"))

from run_hpo_3mod_xmodalix3d import run_synetune_hpo


def env_float(name):
    return float(os.environ[name])


data_path = Path(os.environ["DATA_PATH"])
out_dir = Path(os.environ["OUT_DIR"])
metric = os.environ["METRIC"]
random_seed = int(os.environ["HPO_RANDOM_SEED"])
max_wallclock_hours = float(os.environ.get("MAX_WALLCLOCK_HOURS", "11.75"))
max_wallclock_time = int(max_wallclock_hours * 60 * 60)
n_workers = int(os.environ.get("N_WORKERS", "4"))

modality_hparams = {
    "NCCT": {
        "learning_rate": env_float("NCCT_LR"),
        "weight_decay": env_float("NCCT_WD"),
        "beta": env_float("NCCT_BETA"),
    },
    "CBF": {
        "learning_rate": env_float("CBF_LR"),
        "weight_decay": env_float("CBF_WD"),
        "beta": env_float("CBF_BETA"),
    },
    "CBV": {
        "learning_rate": env_float("CBV_LR"),
        "weight_decay": env_float("CBV_WD"),
        "beta": env_float("CBV_BETA"),
    },
}

latent_dim = int(os.environ["LATENT_DIM"])
hidden_dim = int(os.environ["HIDDEN_DIM"])
train_normalization = os.environ["TRAIN_NORMALIZATION"]
batch_size = int(os.environ["BATCH_SIZE"])
epochs = int(os.environ["EPOCHS"])
pretrain_epochs = int(os.environ["PRETRAIN_EPOCHS"])
target_shape = (
    int(os.environ["TARGET_D"]),
    int(os.environ["TARGET_H"]),
    int(os.environ["TARGET_W"]),
)

tuning_experiment = run_synetune_hpo(
    data_path=data_path,
    modality_hparams=modality_hparams,
    latent_dim=latent_dim,
    hidden_dim=hidden_dim,
    train_normalization=train_normalization,
    batch_size=batch_size,
    epochs=epochs,
    pretrain_epochs=pretrain_epochs,
    target_shape=target_shape,
    metric=metric,
    max_wallclock_time=max_wallclock_time,
    n_workers=n_workers,
    random_seed=random_seed,
)

if tuning_experiment.results is None:
    raise RuntimeError(
        "Syne Tune produced no results. All initial trials may have failed. "
        "Check the trial stderr/stdout logs."
    )

results = tuning_experiment.results.copy()

metadata = {
    "scheduler": "CQR",
    "n_workers": n_workers,
    "metric": metric,
    "mode": "minimize",
    "data_path": str(data_path),
    "modalities": ["NCCT", "CBF", "CBV"],
    "modality_hparams": modality_hparams,
    "latent_dim": latent_dim,
    "hidden_dim": hidden_dim,
    "train_normalization": train_normalization,
    "batch_size": batch_size,
    "epochs": epochs,
    "pretrain_epochs": pretrain_epochs,
    "target_shape": target_shape,
    "random_seed": random_seed,
    "hpo_run_index": os.environ["HPO_RUN_INDEX"],
    "tuned_parameters": [
        "gamma",
        "delta_pair",
        "classifier_learning_rate",
        "classifier_weight_decay",
    ],
}

results_csv = out_dir / "results.csv"
results_pkl = out_dir / "tuning_experiment.pkl"
best_config_txt = out_dir / "best_config.txt"

results.to_csv(results_csv, index=False)

with open(results_pkl, "wb") as f:
    pickle.dump(tuning_experiment, f)

with open(best_config_txt, "w", encoding="utf-8") as f:
    f.write(str(tuning_experiment.best_config()))

with open(out_dir / "hpo_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print(f"Saved results to: {results_csv}")
print(f"Best configuration by {metric}:")
print(tuning_experiment.best_config())
PY

echo "HPO job finished."
