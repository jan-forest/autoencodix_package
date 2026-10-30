#!/bin/bash
#SBATCH --job-name=imagix_cta_best_config_train
#SBATCH --account=p_scads_autoencodix
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --mem=50G
#SBATCH --time=01:00:00
#SBATCH --output=/data/horse/ws/baeuchl-imagix3d/logs/%x-%j.out
#SBATCH --error=/data/horse/ws/baeuchl-imagix3d/logs/%x-%j.err

set -euo pipefail

REPO="/home/baeuchl/autoencodix_package"
BASE="/data/horse/ws/baeuchl-imagix3d"
RESULTS_DIR="${BASE}/results"
REGISTRY_DIR="${BASE}/run_registry"

mkdir -p "${REGISTRY_DIR}"

module --force purge
module load release/24.04
module load GCCcore/12.3.0
module load Python/3.11.3

source "${BASE}/venvs/capella/bin/activate"

cd "${REPO}"

RUN_NAME="imagix_cta_best_config_${SLURM_JOB_ID}"

echo "Running cta Imagix pipeline"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Run name: ${RUN_NAME}"
echo "Repository: ${REPO}"
echo "Workspace: ${BASE}"

python scripts/run_imagix.py \
  --workdir "${BASE}" \
  --images "${BASE}/data/stroke_data/2D/cta" \
  --anno "${BASE}/data/stroke_data/2D/cta_2D_anno_dst.csv" \
  --run-name "${RUN_NAME}" \
  --image-size 192 \
  --epochs 250 \
  --beta 0.000011963401747530954 \
  --latent-dim 64 \
  --hidden-dim 24 \
  --reconstruction-loss "mse" \
  --loss-reduction "mean" \
  --scaling "NONE" \
  --anneal-function "logistic-late" \
  --checkpoint-interval 10 \
  --weight-decay 0.00000029591060828769537 \
  --learning-rate 0.0134606700865418 \
  --batch-size 35

echo "Pipeline finished."
echo "Searching for result directory..."

RESULT_DIR=$(ls -td "${RESULTS_DIR}/${RUN_NAME}"* | head -n 1)

if [ -z "${RESULT_DIR}" ]; then
  echo "ERROR: Could not find result directory for run name ${RUN_NAME}"
  exit 1
fi

if [ ! -f "${RESULT_DIR}/imagix.pkl" ]; then
  echo "ERROR: Result directory found, but imagix.pkl is missing:"
  echo "${RESULT_DIR}"
  exit 1
fi

echo "Detected result directory:"
echo "${RESULT_DIR}"

MARKER_FILE="${REGISTRY_DIR}/${SLURM_JOB_ID}.result_dir"
echo "${RESULT_DIR}" > "${MARKER_FILE}"

echo "Wrote result directory marker to:"
echo "${MARKER_FILE}"
