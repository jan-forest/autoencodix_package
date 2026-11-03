#!/bin/bash

set -euo pipefail

module load rclone

SOURCE="/data/horse/ws/baeuchl-imagix3d/hpo/xmodalix3d_3mod_01_synetune_4496124_20261030_154011"
DESTINATION="scads_nextcloud:FromHPC"

echo "================================================"
echo "Copying HPC results to ScaDS Nextcloud"
echo "Source:      ${SOURCE}"
echo "Destination: ${DESTINATION}"
echo "Started:     $(date)"
echo "================================================"

rclone copy "${SOURCE}" "${DESTINATION}" \
    --progress \
    --transfers 4 \
    --checkers 8 \
    --create-empty-src-dirs

echo "================================================"
echo "Transfer completed: $(date)"
echo "================================================"