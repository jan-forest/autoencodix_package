#!/bin/bash
set -euo pipefail

REPO="/home/baeuchl/autoencodix_package"
JOBS_DIR="${REPO}/jobs"

BASE="/data/horse/ws/baeuchl-imagix3d"
LOG_DIR="${BASE}/logs"
REGISTRY_DIR="${BASE}/run_registry"

mkdir -p "${LOG_DIR}"
mkdir -p "${REGISTRY_DIR}"

TRAIN_SCRIPT="${JOBS_DIR}/run_cbf_gpu_train.sh"
REPORT_SCRIPT="${JOBS_DIR}/run_cbf_gpu_report.sh"

if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "ERROR: Training script not found:"
    echo "${TRAIN_SCRIPT}"
    exit 1
fi

if [ ! -f "${REPORT_SCRIPT}" ]; then
    echo "ERROR: Report script not found:"
    echo "${REPORT_SCRIPT}"
    exit 1
fi

echo "Submitting CBF training job..."

TRAIN_JOBID_RAW=$(sbatch --parsable "${TRAIN_SCRIPT}")
TRAIN_JOBID="${TRAIN_JOBID_RAW%%;*}"

echo "Submitted training job: ${TRAIN_JOBID}"

echo "Submitting dependent report job..."

REPORT_JOBID_RAW=$(sbatch \
    --parsable \
    --dependency=afterok:${TRAIN_JOBID} \
    --export=ALL,TRAIN_JOBID="${TRAIN_JOBID}" \
    "${REPORT_SCRIPT}"
)

REPORT_JOBID="${REPORT_JOBID_RAW%%;*}"

echo "Submitted report job: ${REPORT_JOBID}"
echo
echo "The report job will start only if training job ${TRAIN_JOBID} finishes successfully."
echo
echo "Check status with:"
echo "squeue -j ${TRAIN_JOBID},${REPORT_JOBID}"
