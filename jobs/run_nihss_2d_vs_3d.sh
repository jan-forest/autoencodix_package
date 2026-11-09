#!/bin/bash
#SBATCH --job-name=nihss_2d3d_svr
#SBATCH --account=p_scads_autoencodix
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/horse/ws/baeuchl-imagix3d/logs/%x-%j.out
#SBATCH --error=/data/horse/ws/baeuchl-imagix3d/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# Environment
# =============================================================================

module --force purge
module load release/24.04
module load GCCcore/12.3.0
module load Python/3.11.3

source /data/horse/ws/baeuchl-imagix3d/venvs/alpha/bin/activate

# Prevent each joblib worker from spawning its own BLAS/OpenMP thread pool.
# Parallelism is controlled by joblib/scikit-learn via --n-jobs below.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_DYNAMIC=FALSE
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

# Use node-local storage for joblib temporary/memmap files.
export JOBLIB_TEMP_FOLDER="/tmp/${USER}/nihss_joblib_${SLURM_JOB_ID}"
mkdir -p "${JOBLIB_TEMP_FOLDER}"
trap 'rm -rf "${JOBLIB_TEMP_FOLDER}"' EXIT

N_JOBS="${SLURM_CPUS_PER_TASK}"

# =============================================================================
# USER SETTINGS
# =============================================================================

# Directory containing the two analysis scripts on Alpha.
# Upload the optimized fast-PCA script to this directory with the filename below.
SCRIPT_DIR="/home/baeuchl/autoencodix_package/scripts/nihss_prediction"
ANALYSIS_SCRIPT="${SCRIPT_DIR}/nihss_2d_vs_3d_nested_svr_modality_matched_fast_pca.py"
PLOT_SCRIPT="${SCRIPT_DIR}/plot_nihss_2d_vs_3d_results_modality_matched.py"

# -----------------------------------------------------------------------------
# 2D patient-wise embedding CSVs
# Replace REPLACE_ME paths with the actual Alpha paths.
# -----------------------------------------------------------------------------
EMBED_2D_CBF="/data/horse/ws/baeuchl-imagix3d/data/latent_embeddings/cbf/imagix_cbf_patientwise_z.csv"
EMBED_2D_CBV="/data/horse/ws/baeuchl-imagix3d/data/latent_embeddings/cbv/imagix_cbv_patientwise_z.csv"
EMBED_2D_CTA="/data/horse/ws/baeuchl-imagix3d/data/latent_embeddings/cta/imagix_cta_patientwise_z.csv"
EMBED_2D_NCCT="/data/horse/ws/baeuchl-imagix3d/data/latent_embeddings/ncct/imagix_ncct_patientwise_z.csv"
EMBED_2D_TMAX="/data/horse/ws/baeuchl-imagix3d/data/latent_embeddings/tmax/imagix_tmax_patientwise_z.csv"

# -----------------------------------------------------------------------------
# 3D Imagix3D embedding CSVs
# Each variable should point to the corresponding final all-splits CSV.
# -----------------------------------------------------------------------------
EMBED_3D_CBF="/data/horse/ws/baeuchl-imagix3d/data/latent_embeddings/cbf/imagix3d_final_z_all_splits.csv"
EMBED_3D_CBV="/data/horse/ws/baeuchl-imagix3d/data/latent_embeddings/cbv/imagix3d_final_z_all_splits.csv"
EMBED_3D_CTA="/data/horse/ws/baeuchl-imagix3d/data/latent_embeddings/cta/imagix3d_final_z_all_splits.csv"
EMBED_3D_NCCT="/data/horse/ws/baeuchl-imagix3d/data/latent_embeddings/ncct/imagix3d_final_z_all_splits.csv"
EMBED_3D_TMAX="/data/horse/ws/baeuchl-imagix3d/data/latent_embeddings/tmax/imagix3d_final_z_all_splits.csv"

# Clinical file containing patients_id, nihss_discharge and custom_splits.
CLINICAL_FILE="/home/baeuchl/autoencodix_package/scripts/nihss_prediction/sample_characteristics.csv"

# Root directory for this analysis. A unique job-specific subdirectory is created.
OUTPUT_ROOT="/data/horse/ws/baeuchl-imagix3d/nihss_prediction"
RUN_NAME="2d_vs_3d_nested_svr_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

# Analysis settings
OUTER_FOLDS=10
INNER_FOLDS=5
RANDOM_STATE=42
WILCOXON_ALTERNATIVE="two-sided"

# =============================================================================
# Preflight checks
# =============================================================================

mkdir -p "${OUT_DIR}"
mkdir -p /data/horse/ws/baeuchl-imagix3d/logs

for path in \
    "${ANALYSIS_SCRIPT}" \
    "${PLOT_SCRIPT}" \
    "${EMBED_2D_CBF}" "${EMBED_2D_CBV}" "${EMBED_2D_CTA}" "${EMBED_2D_NCCT}" "${EMBED_2D_TMAX}" \
    "${EMBED_3D_CBF}" "${EMBED_3D_CBV}" "${EMBED_3D_CTA}" "${EMBED_3D_NCCT}" "${EMBED_3D_TMAX}" \
    "${CLINICAL_FILE}"
do
    if [[ "${path}" == *REPLACE_ME* ]]; then
        echo "ERROR: Replace all REPLACE_ME paths before submitting."
        echo "First unresolved path: ${path}"
        exit 1
    fi
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: Required file does not exist: ${path}"
        exit 1
    fi
done

python - <<'PY'
import joblib, matplotlib, numpy, pandas, scipy, sklearn
print("Python/scientific stack OK")
print("scikit-learn:", sklearn.__version__)
print("joblib:", joblib.__version__)
PY

# =============================================================================
# Run information
# =============================================================================

START_EPOCH=$(date +%s)
START_TEXT=$(date '+%Y-%m-%d %H:%M:%S %Z')

echo "========================================================================"
echo "NIHSS 2D-vs-3D nested-SVR analysis"
echo "========================================================================"
echo "Job ID:             ${SLURM_JOB_ID}"
echo "Node:               ${SLURMD_NODENAME:-unknown}"
echo "Started:            ${START_TEXT}"
echo "Allocated CPUs:     ${SLURM_CPUS_PER_TASK}"
echo "joblib n_jobs:      ${N_JOBS}"
echo "Allocated GPUs:     ${SLURM_JOB_GPUS:-unknown} (not used by sklearn analysis)"
echo "Output directory:   ${OUT_DIR}"
echo "Joblib temp folder: ${JOBLIB_TEMP_FOLDER}"
echo "========================================================================"

# =============================================================================
# Main analysis
# =============================================================================

srun --cpu-bind=cores python "${ANALYSIS_SCRIPT}" \
    --embedding-2d "CBF=${EMBED_2D_CBF}" \
    --embedding-2d "CBV=${EMBED_2D_CBV}" \
    --embedding-2d "CTA=${EMBED_2D_CTA}" \
    --embedding-2d "NCCT=${EMBED_2D_NCCT}" \
    --embedding-2d "TMAX=${EMBED_2D_TMAX}" \
    --embedding-3d "CBF=${EMBED_3D_CBF}" \
    --embedding-3d "CBV=${EMBED_3D_CBV}" \
    --embedding-3d "CTA=${EMBED_3D_CTA}" \
    --embedding-3d "NCCT=${EMBED_3D_NCCT}" \
    --embedding-3d "TMAX=${EMBED_3D_TMAX}" \
    --clinical "${CLINICAL_FILE}" \
    --output-dir "${OUT_DIR}" \
    --target-col "nihss_discharge" \
    --clinical-id-col "patients_id" \
    --clinical-split-col "custom_splits" \
    --outer-folds "${OUTER_FOLDS}" \
    --inner-folds "${INNER_FOLDS}" \
    --random-state "${RANDOM_STATE}" \
    --n-jobs "${N_JOBS}" \
    --wilcoxon-alternative "${WILCOXON_ALTERNATIVE}"

# =============================================================================
# Figures
# =============================================================================

python "${PLOT_SCRIPT}" --results-dir "${OUT_DIR}"

# =============================================================================
# Timing summary
# =============================================================================

END_EPOCH=$(date +%s)
END_TEXT=$(date '+%Y-%m-%d %H:%M:%S %Z')
ELAPSED=$((END_EPOCH - START_EPOCH))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

printf -v RUNTIME '%02d:%02d:%02d' "${HOURS}" "${MINUTES}" "${SECONDS}"

{
    echo "Started: ${START_TEXT}"
    echo "Ended:   ${END_TEXT}"
    echo "Runtime: ${RUNTIME}"
    echo "CPUs:    ${SLURM_CPUS_PER_TASK}"
    echo "n_jobs:  ${N_JOBS}"
} | tee "${OUT_DIR}/runtime.txt"

echo
echo "Analysis and plotting completed successfully."
echo "Results: ${OUT_DIR}"
