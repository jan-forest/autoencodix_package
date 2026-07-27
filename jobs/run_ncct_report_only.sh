#!/bin/bash
#SBATCH --job-name=ncct_report
#SBATCH --account=p_scads_autoencodix
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH --time=0:30:00
#SBATCH --output=/data/horse/ws/baeuchl-imagix3d/logs/%x-%j.out
#SBATCH --error=/data/horse/ws/baeuchl-imagix3d/logs/%x-%j.err

set -euo pipefail

module --force purge
module load release/24.04
module load GCCcore/12.3.0
module load Python/3.11.3

source /data/horse/ws/baeuchl-imagix3d/venvs/alpha/bin/activate

cd "${BASE}"

TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"

export RESULT_DIR
export PKL_PATH="${RESULT_DIR}/imagix3d.pkl"
export REPORT_DIR="${RESULT_DIR}/report_only_${TIMESTAMP}"
export TEMPLATE

REPORT_NOTEBOOK="${REPORT_DIR}/ncct_report_executed.ipynb"
REPORT_HTML="${REPORT_DIR}/ncct_report.html"

mkdir -p "${REPORT_DIR}"

echo "Starting NCCT report-only job"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "BASE: ${BASE}"
echo "RESULT_DIR: ${RESULT_DIR}"
echo "PKL_PATH: ${PKL_PATH}"
echo "REPORT_DIR: ${REPORT_DIR}"
echo "TEMPLATE: ${TEMPLATE}"
echo "REPORT_NOTEBOOK: ${REPORT_NOTEBOOK}"
echo "REPORT_HTML: ${REPORT_HTML}"

echo "Checking input files:"
ls -lh "${RESULT_DIR}"
ls -lh "${PKL_PATH}"
ls -lh "${TEMPLATE}"

echo "Python executable:"
which python
python --version

echo "Executing notebook..."

python -m jupyter nbconvert \
  --to notebook \
  --execute "${TEMPLATE}" \
  --output-dir "${REPORT_DIR}" \
  --output "ncct_report_executed" \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3

echo "Checking executed notebook:"
test -s "${REPORT_NOTEBOOK}"
ls -lh "${REPORT_NOTEBOOK}"

echo "Converting executed notebook to HTML..."

python -m jupyter nbconvert \
  --to html \
  "${REPORT_NOTEBOOK}" \
  --output-dir "${REPORT_DIR}" \
  --output "ncct_report"

echo "Checking HTML report:"
test -s "${REPORT_HTML}"
ls -lh "${REPORT_HTML}"

echo "Report finished successfully."
echo "Executed notebook: ${REPORT_NOTEBOOK}"
echo "HTML report: ${REPORT_HTML}"

echo "Report directory content:"
find "${REPORT_DIR}" -maxdepth 1 -type f -ls