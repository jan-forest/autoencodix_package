#!/bin/bash
#SBATCH --job-name=xmodalix3d_3mod_full
#SBATCH --account=p_scads_autoencodix
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3
#SBATCH --mem=360G
#SBATCH --time=01:30:00
#SBATCH --output=/data/horse/ws/baeuchl-imagix3d/logs/%x-%j.out
#SBATCH --error=/data/horse/ws/baeuchl-imagix3d/logs/%x-%j.err

set -euo pipefail

REPO="/home/baeuchl/autoencodix_package"
BASE="/data/horse/ws/baeuchl-imagix3d"
DATA_ROOT="${BASE}/data/stroke_data"

TEMPLATE="${REPO}/notebooks/xmodalix3d_3mod_alignment_full_hpc.ipynb"

TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_NAME="xmodalix3d_3mod_alignment_${SLURM_JOB_ID}_${TIMESTAMP}"
RESULT_DIR="${BASE}/results/${RUN_NAME}"

export DATA_ROOT
export RESULT_DIR

export NCCT_PATH="${DATA_ROOT}/ncct_flat"
export NCCT_ANNO="${DATA_ROOT}/ncct_anno_dst.csv"
export CBF_PATH="${DATA_ROOT}/cbf_flat"
export CBF_ANNO="${DATA_ROOT}/cbf_anno_dst.csv"
export CBV_PATH="${DATA_ROOT}/cbv_flat"
export CBV_ANNO="${DATA_ROOT}/cbv_anno_dst.csv"

OUTPUT_NOTEBOOK="${RESULT_DIR}/xmodalix3d_3mod_alignment_executed.ipynb"

mkdir -p "${BASE}/logs"
mkdir -p "${RESULT_DIR}"

module --force purge
module load release/24.04
module load GCCcore/12.3.0
module load Python/3.11.3

source "${BASE}/venvs/alpha/bin/activate"

# Required by PyTorch deterministic algorithms on CUDA/cuBLAS.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "${REPO}"

echo "Starting full XModalix3D 3-modality alignment run"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "Repository: ${REPO}"
echo "Data root: ${DATA_ROOT}"
echo "Template: ${TEMPLATE}"
echo "Result directory: ${RESULT_DIR}"
echo "Output notebook: ${OUTPUT_NOTEBOOK}"
echo "Deterministic CUDA configuration:"
echo "CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}"

echo
echo "Repository state:"
git branch --show-current || true
git rev-parse HEAD || true
git status --short || true

echo
echo "Python:"
which python
python --version

echo
echo "GPU allocation:"
nvidia-smi || true

echo
echo "Checking notebook and input data..."
test -f "${TEMPLATE}"

test -d "${NCCT_PATH}"
test -f "${NCCT_ANNO}"

test -d "${CBF_PATH}"
test -f "${CBF_ANNO}"

test -d "${CBV_PATH}"
test -f "${CBV_ANNO}"

ls -ld "${NCCT_PATH}" "${CBF_PATH}" "${CBV_PATH}"
ls -lh "${NCCT_ANNO}" "${CBF_ANNO}" "${CBV_ANNO}" "${TEMPLATE}"

echo
echo "Executing the complete notebook..."

python -m jupyter nbconvert \
  --to notebook \
  --execute "${TEMPLATE}" \
  --output-dir "${RESULT_DIR}" \
  --output "xmodalix3d_3mod_alignment_executed" \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3

echo
echo "Checking executed notebook..."
test -s "${OUTPUT_NOTEBOOK}"
ls -lh "${OUTPUT_NOTEBOOK}"

echo
echo "XModalix3D alignment run finished successfully."
echo "Executed notebook:"
echo "${OUTPUT_NOTEBOOK}"

echo
echo "Result directory content:"
find "${RESULT_DIR}" -maxdepth 1 -type f -ls
