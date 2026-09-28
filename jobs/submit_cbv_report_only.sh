#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE="/home/h2/baeuchl/autoencodix_package"

RESULT_DIR="/data/horse/ws/baeuchl-imagix3d/results/gpu_cbv_tuned_HP_2355904_2026-09-28_16-32-34"
TEMPLATE="${BASE}/notebooks/imagix3d_full_report_template.ipynb"

echo "Submitting report-only job..."
echo "Result directory: ${RESULT_DIR}"
echo "Template: ${TEMPLATE}"
echo "Job script: ${SCRIPT_DIR}/run_cbv_report_only.sh"

sbatch \
  --export=ALL,BASE="${BASE}",RESULT_DIR="${RESULT_DIR}",TEMPLATE="${TEMPLATE}" \
  "${SCRIPT_DIR}/run_cbv_report_only.sh"