#!/bin/bash
#SBATCH --job-name=imagix3d_benchmark
#SBATCH --account=p_scads_autoencodix
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=12:00:00
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

cd /home/baeuchl/autoencodix_package

export PYTHONUNBUFFERED=1

echo "===== Imagix3D benchmark preflight ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-not set}"
nvidia-smi


# =============================================================================
# USER SETTINGS
# =============================================================================

# Use ONE representative Imagix3D modality for the computational benchmark.
# NCCT is a sensible default because all modalities are resampled to the same
# target shape and the benchmark concerns architecture/training cost rather than
# modality-specific predictive performance.
export DATA_PATH="/data/horse/ws/baeuchl-imagix3d/data/stroke_data"
export FOLDER="ncct_flat"
export ANNOTATION="ncct_anno_dst.csv"


# -----------------------------------------------------------------------------
# Fixed reference hyperparameters
#
# IMPORTANT:
# Fill these from the final selected Imagix3D configuration for the chosen
# reference modality. They are deliberately held fixed during the computational
# benchmark; only batch size / hidden_dim / latent_dim are varied.
# -----------------------------------------------------------------------------

export LEARNING_RATE=0.000031484983656998606
export WEIGHT_DECAY=0.000007737374408547984
export BETA=0.0000006718196015256182

export TRAIN_NORMALIZATION="batch"
export KEEP_MU_POSITIVE="false"
export ANNEAL_FUNCTION="logistic-late"
export LOSS_REDUCTION="mean"
export SCALING="NONE"


# -----------------------------------------------------------------------------
# Fixed input size and training implementation settings
# -----------------------------------------------------------------------------

export TARGET_SHAPE="160,192,160"

# Match the precision/settings used in your actual Imagix3D experiments.
export FLOAT_PRECISION="32"
export COMPILE_MODEL="false"
export PIN_MEMORY="false"
export SAVE_MEMORY="false"
export SAVE_VRAM="false"
export REPRODUCIBLE="false"


# -----------------------------------------------------------------------------
# Benchmark design
#
# Core experiment:
#   batch-size envelope at hidden_dim = 16 and 32
#
# Optional small sensitivity analysis:
#   latent_dim = 64, 80, 96 at the reference batch size / hidden_dim
#
# Each short run has 4 epochs; epoch 1 is warm-up and excluded from timing.
# Two repetitions provide a minimal robustness check without making this a
# large benchmarking project.
# -----------------------------------------------------------------------------

export BATCH_SIZES="8,16,24,35,48,64"
export HIDDEN_DIMS="16,32"

export RUN_LATENT_SWEEP="false"
export LATENT_DIMS="64,80,96"

export REFERENCE_BATCH_SIZE=35
export REFERENCE_HIDDEN_DIM=16
export REFERENCE_LATENT_DIM=96

export BENCHMARK_EPOCHS=4
export WARMUP_EPOCHS=1
export REPETITIONS=2
export BENCHMARK_SEED=42


# =============================================================================
# Validation
# =============================================================================

for name in LEARNING_RATE WEIGHT_DECAY BETA; do
    value="${!name:-}"
    if [[ -z "${value}" || "${value}" == "REPLACE_ME" ]]; then
        echo "ERROR: ${name} is not set."
        echo "Edit run_benchmark_imagix3d.sh and enter the selected Imagix3D hyperparameter."
        exit 1
    fi
done


# =============================================================================
# Output
# =============================================================================

BENCH_ROOT="/data/horse/ws/baeuchl-imagix3d/benchmark"
RUN_NAME="imagix3d_${FOLDER}_benchmark_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${BENCH_ROOT}/${RUN_NAME}"

mkdir -p "${OUT_DIR}"

echo
echo "Output directory: ${OUT_DIR}"
echo "Reference data: ${DATA_PATH}/${FOLDER}"
echo "Batch sizes: ${BATCH_SIZES}"
echo "Hidden dims: ${HIDDEN_DIMS}"
echo "Latent sweep: ${RUN_LATENT_SWEEP} (${LATENT_DIMS})"
echo "Short-run epochs: ${BENCHMARK_EPOCHS}; warm-up epochs: ${WARMUP_EPOCHS}"
echo "Repetitions: ${REPETITIONS}"
echo


# =============================================================================
# Run benchmark
# =============================================================================

srun python scripts/benchmark_imagix3d.py \
    --repo-dir "$(pwd)" \
    --data-path "${DATA_PATH}" \
    --folder "${FOLDER}" \
    --annotation "${ANNOTATION}" \
    --output-dir "${OUT_DIR}" \
    --learning-rate "${LEARNING_RATE}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --beta "${BETA}" \
    --train-normalization "${TRAIN_NORMALIZATION}" \
    --keep-mu-positive "${KEEP_MU_POSITIVE}" \
    --anneal-function "${ANNEAL_FUNCTION}" \
    --loss-reduction "${LOSS_REDUCTION}" \
    --scaling "${SCALING}" \
    --target-shape "${TARGET_SHAPE}" \
    --float-precision "${FLOAT_PRECISION}" \
    --compile-model "${COMPILE_MODEL}" \
    --pin-memory "${PIN_MEMORY}" \
    --save-memory "${SAVE_MEMORY}" \
    --save-vram "${SAVE_VRAM}" \
    --reproducible "${REPRODUCIBLE}" \
    --batch-sizes "${BATCH_SIZES}" \
    --hidden-dims "${HIDDEN_DIMS}" \
    --run-latent-sweep "${RUN_LATENT_SWEEP}" \
    --latent-dims "${LATENT_DIMS}" \
    --reference-batch-size "${REFERENCE_BATCH_SIZE}" \
    --reference-hidden-dim "${REFERENCE_HIDDEN_DIM}" \
    --reference-latent-dim "${REFERENCE_LATENT_DIM}" \
    --epochs "${BENCHMARK_EPOCHS}" \
    --warmup-epochs "${WARMUP_EPOCHS}" \
    --repetitions "${REPETITIONS}" \
    --seed "${BENCHMARK_SEED}"

echo
echo "Imagix3D computational benchmark finished."
echo "Results: ${OUT_DIR}"
