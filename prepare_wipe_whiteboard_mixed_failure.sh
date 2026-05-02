#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/public2/liushengbang/openpi_tactile"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

ORIGINAL_HDF5_DIR="/mnt/public2/liushengbang/Data/Origin/wipe_whiteboard"
FAILURE_DIR="/mnt/public2/liushengbang/idm_force_infer/stride5_failure_wipe/failure_wipe"
OUTPUT_ROOT="/mnt/public2/liushengbang/Data/processed_lerobot/wan_worldrl_stride5_posecont_192x256"
REPO_NAME="wipe_whiteboard_force_stride5_posecont_dupfront_192x256_mixed_failure"
CONFIG_NAME="pi05_wipe_whiteboard_force_stride5_posecont_dupfront_192x256_mixed_failure"
TASK_TEXT="wipe the star mark on the whiteboard"

OPENPI_DATA_HOME="/mnt/public2/liushengbang/openpi_data/openpi-assets"

export HF_LEROBOT_HOME="$OUTPUT_ROOT"
export OPENPI_DATA_HOME

cd "$ROOT_DIR"

echo "[prepare-mixed-failure] Converting original + failure data -> $OUTPUT_ROOT/$REPO_NAME"
"$PYTHON_BIN" examples/libero/convert_mixed_wipe_to_lerobot_force.py \
  --original-hdf5-dir "$ORIGINAL_HDF5_DIR" \
  --failure-dir "$FAILURE_DIR" \
  --repo-name "$REPO_NAME" \
  --output-dir "$OUTPUT_ROOT" \
  --overwrite \
  --task "$TASK_TEXT" \
  --force-history-len 8 \
  --stride 5 \
  --image-height 192 \
  --image-width 256 \
  --fps 6

echo "[prepare-mixed-failure] Computing norm stats for config $CONFIG_NAME"
"$PYTHON_BIN" scripts/compute_norm_stats.py \
  --config-name "$CONFIG_NAME"
