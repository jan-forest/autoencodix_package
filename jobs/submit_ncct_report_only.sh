#!/bin/bash
set -euo pipefail

BASE="/data/horse/ws/baeuchl-imagix3d"

RESULT_DIR="/data/horse/ws/baeuchl-imagix3d/results/gpu_ncct_afterHPO_2307348_2026-07-26_17-37-37"
TEMPLATE="${BASE}/notebooks/ncct_report_template.ipynb"

echo "Submitting report-only job..."
echo "Result directory: ${RESULT_DIR}"
echo "Template: ${TEMPLATE}"

sbatch \
  --export=ALL,BASE="${BASE}",RESULT_DIR="${RESULT_DIR}",TEMPLATE="${TEMPLATE}" \
  run_ncct_report_only.sh