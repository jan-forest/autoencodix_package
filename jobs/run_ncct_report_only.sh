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

cd /home/h2/baeuchl/autoencodix_package

echo "Starting NCCT report-only job"
echo "RESULT_DIR: ${RESULT_DIR}"
echo "TEMPLATE: ${TEMPLATE}"

REPORT_NOTEBOOK="${RESULT_DIR}/ncct_report_executed.ipynb"
REPORT_HTML="${RESULT_DIR}/ncct_report.html"

jupyter nbconvert \
  --to notebook \
  --execute "${TEMPLATE}" \
  --output "${REPORT_NOTEBOOK}" \
  --ExecutePreprocessor.timeout=-1

jupyter nbconvert \
  --to html \
  "${REPORT_NOTEBOOK}" \
  --output "${REPORT_HTML}"

echo "Report finished."
echo "Executed notebook: ${REPORT_NOTEBOOK}"
echo "HTML report: ${REPORT_HTML}"