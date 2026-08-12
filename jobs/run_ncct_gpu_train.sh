#!/bin/bash
#SBATCH --job-name=imagix3d_ncct_afterHPO_03_train
#SBATCH --account=p_scads_autoencodix
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=4
#SBATCH --mem=480G
#SBATCH --time=01:30:00
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

RUN_NAME="gpu_ncct_afterHPO_03_${SLURM_JOB_ID}"

echo "Running NCCT Imagix3D pipeline"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Run name: ${RUN_NAME}"
echo "Repository: ${REPO}"
echo "Workspace: ${BASE}"

python scripts/run_imagix3d.py \
  --workdir "${BASE}" \
  --images "${BASE}/data/stroke_data/ncct_flat" \
  --anno "${BASE}/data/stroke_data/ct_anno_flat.csv" \
  --run-name "${RUN_NAME}" \
  --target-shape 160 192 160 \
  --epochs 250 \
  --beta 0.0000075 \
  --latent-dim 48 \
  --hidden-dim 16 \
  --train-normalization "group" \
  --anneal-function "logistic-late" \
  --no-keep-mu-positive \
  --weight-decay 0.00013 \
  --learning-rate 0.0006 \
  --batch-size 48

echo "Pipeline finished."
echo "Searching for result directory..."

RESULT_DIR=$(ls -td "${RESULTS_DIR}/${RUN_NAME}"* | head -n 1)

if [ -z "${RESULT_DIR}" ]; then
  echo "ERROR: Could not find result directory for run name ${RUN_NAME}"
  exit 1
fi

if [ ! -f "${RESULT_DIR}/imagix3d.pkl" ]; then
  echo "ERROR: Result directory found, but imagix3d.pkl is missing:"
  echo "${RESULT_DIR}"
  exit 1
fi

echo "Detected result directory:"
echo "${RESULT_DIR}"

MARKER_FILE="${REGISTRY_DIR}/${SLURM_JOB_ID}.result_dir"
echo "${RESULT_DIR}" > "${MARKER_FILE}"

echo "Wrote result directory marker to:"
echo "${MARKER_FILE}"
