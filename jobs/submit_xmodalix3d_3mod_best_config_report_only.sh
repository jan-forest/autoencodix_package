#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE="/home/h2/baeuchl/autoencodix_package"

RESULT_DIR="/data/horse/ws/baeuchl-imagix3d/results/xmodalix3d_3mod_2387695_2026-11-04_02-13-09"
TEMPLATE="${BASE}/notebooks/xmodalix3d_short_report_template.ipynb"

echo "Submitting report-only job..."
echo "Result directory: ${RESULT_DIR}"
echo "Template: ${TEMPLATE}"
echo "Job script: ${SCRIPT_DIR}/run_3mod_report_only.sh"

sbatch \
  --export=ALL,BASE="${BASE}",RESULT_DIR="${RESULT_DIR}",TEMPLATE="${TEMPLATE}" \
  "${SCRIPT_DIR}/run_3mod_report_only.sh"