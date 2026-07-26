#!/bin/bash
#SBATCH --job-name=hpo_ncct
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

which python
python --version

nvidia-smi

srun python - <<'PY'
import os
import torch

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch version:", torch.__version__)
print("torch CUDA version:", torch.version.cuda)

try:
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
except Exception as e:
    print("CUDA test failed:")
    raise
PY

export PYTHONUNBUFFERED=1

export DATA_PATH="/data/horse/ws/baeuchl-imagix3d/data/medmnist/nodule3d"
export FOLDER="train"
export ANNO="train_metadata.csv"
export TASKS="group"
export METRIC="reconstruction_loss"

HPO_ROOT="/data/horse/ws/baeuchl-imagix3d/hpo"
RUN_NAME="nodule3d_synetune_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${HPO_ROOT}/${RUN_NAME}"

mkdir -p "${OUT_DIR}"
export OUT_DIR

echo "Starting Imagix3D Syne Tune HPO"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Repository: $(pwd)"
echo "Data path: ${DATA_PATH}"
echo "Output directory: ${OUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"

srun python - <<'PY'
import os
import pickle
import sys
import json
from pathlib import Path

repo = Path.cwd()
sys.path.insert(0, str(repo / "scripts"))

# load the hyperparameter optimization function
from run_hpo_imagix3d import run_synetune_hpo

data_path = Path(os.environ["DATA_PATH"])
folder = os.environ["FOLDER"]
anno = os.environ["ANNO"]
tasks = os.environ["TASKS"]
metric = os.environ["METRIC"]
out_dir = Path(os.environ["OUT_DIR"])

out_dir.mkdir(parents=True, exist_ok=True)

tuning_experiment = run_synetune_hpo(
    data_path=data_path,
    folder=folder,
    anno=anno,
    tasks=tasks,
    metric=metric,
)

results = tuning_experiment.results.copy()

metadata = {
    "scheduler": "RandomSearch",
    "n_workers": 1,
    "metric": metric,
    "mode": "minimize" if metric != "downstream_performance" else "maximize",
    "data_path": str(data_path),
    "folder": folder,
    "annotation_file": anno,
    "tasks": tasks,
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

PY

echo "HPO job finished."