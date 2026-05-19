#!/usr/bin/env bash
set -euo pipefail

# ---- machine-specific paths -------------------------------------------------
ROOT_DIR="/mnt/public/gaojiahuan/liushengbang/openpi_tactile"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

RAW_DATA_DIR="/mnt/public/gaojiahuan/liushengbang/Data/Origin/wipe_whiteboard"
SYNTH_DATA_DIR_1="/mnt/public/gaojiahuan/liushengbang/Data/synthetic_data/wipe_whiteboard_0511"
SYNTH_DATA_DIR_2="/mnt/public/gaojiahuan/liushengbang/Data/synthetic_data/wipe_whiteboard1_start0_n5_ckpt7145"

OUTPUT_ROOT="/mnt/public/gaojiahuan/liushengbang/Data/processed_lerobot/wan_worldrl_stride5_posecont_192x256"
REPO_NAME="wipe_whiteboard_iteration2_force_stride5_posecont_dupfront_192x256_origin_synth12"
CONFIG_NAME="pi05_wipe_whiteboard_iteration2_force_stride5_posecont_dupfront_192x256"
TASK_TEXT="wipe the star mark on the whiteboard"

OPENPI_DATA_HOME="/mnt/public/gaojiahuan/liushengbang/openpi_data/openpi-assets"

# ---- env --------------------------------------------------------------------
export HF_LEROBOT_HOME="$OUTPUT_ROOT"
export OPENPI_DATA_HOME
export WANDB_BASE_URL="https://api.bandw.top"
export WANDB_API_KEY="7d0049c9992505326ca78a42d89dcfefa2e3d51a"
export WANDB_ENTITY="liushb9-peking-university"

cd "$ROOT_DIR"

if [[ -d "$OUTPUT_ROOT/$REPO_NAME" ]]; then
  echo "[prepare] Existing LeRobot dataset found at $OUTPUT_ROOT/$REPO_NAME"
  echo "[prepare] Skip conversion. Remove the directory first if you want to rebuild it."
else
  echo "[prepare] Converting wipe_whiteboard iteration2 data -> $OUTPUT_ROOT/$REPO_NAME"
  "$PYTHON_BIN" examples/libero/convert_wipe_whiteboard_iteration2_to_lerobot_force.py \
    --original-hdf5-dir "$RAW_DATA_DIR" \
    --failure-dir-1 "$SYNTH_DATA_DIR_1" \
    --failure-dir-2 "$SYNTH_DATA_DIR_2" \
    --repo-name "$REPO_NAME" \
    --output-dir "$OUTPUT_ROOT" \
    --overwrite \
    --task "$TASK_TEXT" \
    --force-history-len 8 \
    --stride 5 \
    --image-height 192 \
    --image-width 256 \
    --fps 6
fi

echo "[prepare] Computing norm stats for config $CONFIG_NAME"
"$PYTHON_BIN" scripts/compute_norm_stats.py \
  --config-name "$CONFIG_NAME"
