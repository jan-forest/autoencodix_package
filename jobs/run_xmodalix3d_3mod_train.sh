#!/bin/bash
#SBATCH --job-name=xmodalix3d_3mod_train
#SBATCH --account=p_scads_autoencodix
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=4
#SBATCH --mem=480G
#SBATCH --time=04:00:00
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

RUN_NAME="xmodalix3d_3mod_${SLURM_JOB_ID}"

echo "Running XModalix3D three-modality pipeline"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Run name: ${RUN_NAME}"
echo "Repository: ${REPO}"
echo "Workspace: ${BASE}"

python scripts/run_xmodalix3d.py \
  --workdir "${BASE}" \
  --ncct-images "${BASE}/data/stroke_data/ncct_flat" \
  --ncct-anno "${BASE}/data/stroke_data/ncct_anno_dst.csv" \
  --cbf-images "${BASE}/data/stroke_data/cbf_flat" \
  --cbf-anno "${BASE}/data/stroke_data/cbf_anno_dst.csv" \
  --cbv-images "${BASE}/data/stroke_data/cbv_flat" \
  --cbv-anno "${BASE}/data/stroke_data/cbv_anno_dst.csv" \
  --run-name "${RUN_NAME}" \
  --target-shape 160 192 160 \
  --epochs 250 \
  --pretrain-epochs 50 \
  --pretrain-save-memory \
  --no-save-intermediate-recons \
  --latent-dim 48 \
  --hidden-dim 16 \
  --batch-size 35 \
  --reconstruction-loss "mse" \
  --loss-reduction "mean" \
  --scaling "NONE" \
  --anneal-function "logistic-late" \
  --checkpoint-interval 10 \
  --clamp-logvar \
  --no-keep-mu-positive \
  --train-normalization "batch" \
  --train-norm-groupsize 8 \
  --learning-rate 0.000010 \
  --weight-decay 0.000010 \
  --beta 0.000010 \
  --ncct-learning-rate 0.000031484 \
  --ncct-weight-decay 0.0000077837 \
  --ncct-beta 0.00000067181 \
  --cbf-learning-rate 0.000010096 \
  --cbf-weight-decay 0.000017943 \
  --cbf-beta 0.0000000032465 \
  --cbv-learning-rate 0.00046282 \
  --cbv-weight-decay 0.0000051361 \
  --cbv-beta 0.0000383242 \
  --lambda-sub 2.0 \
  --gamma 0.01 \
  --delta-pair 0.05 \
  --delta-class 0 \
  --annotation-columns median_split \
  --from-modality "NCCT" \
  --to-modality "CBF"

echo "Pipeline finished."
echo "Searching for result directory..."

RESULT_DIR=$(ls -td "${RESULTS_DIR}/${RUN_NAME}"* | head -n 1)

if [ -z "${RESULT_DIR}" ]; then
  echo "ERROR: Could not find result directory for run name ${RUN_NAME}"
  exit 1
fi

if [ ! -f "${RESULT_DIR}/xmodalix3d.pkl" ]; then
  echo "ERROR: Result directory found, but xmodalix3d.pkl is missing:"
  echo "${RESULT_DIR}"
  exit 1
fi

echo "Detected result directory:"
echo "${RESULT_DIR}"

MARKER_FILE="${REGISTRY_DIR}/${SLURM_JOB_ID}.result_dir"
echo "${RESULT_DIR}" > "${MARKER_FILE}"

echo "Wrote result directory marker to:"
echo "${MARKER_FILE}"
