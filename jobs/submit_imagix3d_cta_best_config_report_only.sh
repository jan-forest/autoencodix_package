#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE="/home/h2/baeuchl/autoencodix_package"

RESULT_DIR="/data/horse/ws/baeuchl-imagix3d/results/imagix3d_cta_best_config_common_param_2387196_2026-11-03_20-37-34"
TEMPLATE="${BASE}/notebooks/imagix3d_full_report_template.ipynb"

echo "Submitting report-only job..."
echo "Result directory: ${RESULT_DIR}"
echo "Template: ${TEMPLATE}"
echo "Job script: ${SCRIPT_DIR}/run_cta_report_only.sh"

sbatch \
  --export=ALL,BASE="${BASE}",RESULT_DIR="${RESULT_DIR}",TEMPLATE="${TEMPLATE}" \
  "${SCRIPT_DIR}/run_cta_report_only.sh"