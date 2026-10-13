#!/bin/bash
#SBATCH --job-name=imagix_ncct_train
#SBATCH --account=p_scads_autoencodix
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-task=2
#SBATCH --mem=220G
#SBATCH --time=02:00:00
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

source "${BASE}/venvs/alpha/bin/activate"

cd "${REPO}"

RUN_NAME="imagix_ncct_${SLURM_JOB_ID}"

echo "Running NCCT Imagix pipeline"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Run name: ${RUN_NAME}"
echo "Repository: ${REPO}"
echo "Workspace: ${BASE}"

python scripts/run_imagix.py \
  --workdir "${BASE}" \
  --images "${BASE}/data/stroke_data/2D/ncct" \
  --anno "${BASE}/data/stroke_data/ncct_2D_anno_dst.csv" \
  --run-name "${RUN_NAME}" \
  --image-size 192 \
  --epochs 250 \
  --beta 0.000000817192 \
  --latent-dim 80 \
  --hidden-dim 32 \
  --reconstruction-loss "mse" \
  --loss-reduction "mean" \
  --scaling "NONE" \
  --anneal-function "logistic-late" \
  --checkpoint-interval 10 \
  --weight-decay 0.0000141901 \
  --learning-rate 0.0029186 \
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
