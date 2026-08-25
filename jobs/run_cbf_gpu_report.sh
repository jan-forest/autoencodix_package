#!/bin/bash
#SBATCH --job-name=imagix3d_cbf_report
#SBATCH --account=p_scads_stroke
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=4
#SBATCH --mem=480G
#SBATCH --time=00:20:00
#SBATCH --output=/data/horse/ws/baeuchl-imagix3d/logs/%x-%j.out
#SBATCH --error=/data/horse/ws/baeuchl-imagix3d/logs/%x-%j.err

set -euo pipefail

REPO="/home/baeuchl/autoencodix_package"
BASE="/data/horse/ws/baeuchl-imagix3d"

REGISTRY_DIR="${BASE}/run_registry"
TEMPLATE_NOTEBOOK="${REPO}/notebooks/ncct_report_template.ipynb"

module --force purge
module load release/24.04
module load GCCcore/12.3.0
module load Python/3.11.3

source "${BASE}/venvs/alpha/bin/activate"

cd "${REPO}"

if [ -z "${TRAIN_JOBID:-}" ]; then
  echo "ERROR: TRAIN_JOBID environment variable is missing."
  echo "This report job should be submitted by submit_cbf_with_report.sh."
  exit 1
fi

RESULT_MARKER="${REGISTRY_DIR}/${TRAIN_JOBID}.result_dir"

if [ ! -f "${RESULT_MARKER}" ]; then
  echo "ERROR: Result marker file not found:"
  echo "${RESULT_MARKER}"
  exit 1
fi

RESULT_DIR=$(cat "${RESULT_MARKER}")
PKL_PATH="${RESULT_DIR}/imagix3d.pkl"
REPORT_DIR="${RESULT_DIR}/report"

# Make paths available to the Python/Jupyter process
export RESULT_DIR
export PKL_PATH
export REPORT_DIR

mkdir -p "${REPORT_DIR}"

if [ ! -f "${PKL_PATH}" ]; then
  echo "ERROR: Pickle file not found:"
  echo "${PKL_PATH}"
  exit 1
fi

if [ ! -f "${TEMPLATE_NOTEBOOK}" ]; then
  echo "ERROR: Template notebook not found:"
  echo "${TEMPLATE_NOTEBOOK}"
  exit 1
fi

echo "Creating NCCT report"
echo "Training job id: ${TRAIN_JOBID}"
echo "Result directory: ${RESULT_DIR}"
echo "Pickle path: ${PKL_PATH}"
echo "Report directory: ${REPORT_DIR}"
echo "Template notebook: ${TEMPLATE_NOTEBOOK}"

papermill \
  "${TEMPLATE_NOTEBOOK}" \
  "${REPORT_DIR}/cbf_report_executed.ipynb" \
  -k imagix3d-alpha \

jupyter nbconvert \
  --to html \
  --output-dir "${REPORT_DIR}" \
  "${REPORT_DIR}/cbf_report_executed.ipynb"

echo "Report generation finished."
echo
echo "Executed notebook:"
echo "${REPORT_DIR}/cbf_report_executed.ipynb"
echo
echo "HTML report:"
echo "${REPORT_DIR}/cbf_report_executed.html"
