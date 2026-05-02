#!/usr/bin/env bash
set -euo pipefail

# ---- machine-specific paths -------------------------------------------------
ROOT_DIR="/mnt/public2/liushengbang/openpi_tactile"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

# Raw HDF5 source. Optional: if not present, the convert step is skipped and
# we go straight to compute_norm_stats on the already-converted dataset.
RAW_DATA_DIR="/mnt/public2/liushengbang/Data/Origin/wipe_whiteboard"

# LeRobot output root and per-task repo name.
OUTPUT_ROOT="/mnt/public2/liushengbang/Data/processed_lerobot/wan_worldrl_stride5_posecont_192x256"
REPO_NAME="wipe_whiteboard_force_stride5_posecont_dupfront_192x256"
CONFIG_NAME="pi05_wipe_whiteboard_force_stride5_posecont_dupfront_192x256"
TASK_TEXT="wipe the star mark on the whiteboard"

# OpenPI cache (where pi05_droid checkpoint lives).
OPENPI_DATA_HOME="/mnt/public2/liushengbang/openpi_data/openpi-assets"

# ---- env --------------------------------------------------------------------
export HF_LEROBOT_HOME="$OUTPUT_ROOT"
export OPENPI_DATA_HOME

# wandb (compute_norm_stats does not log; kept here for symmetry with train.sh).
export WANDB_BASE_URL="https://api.bandw.top"
export WANDB_API_KEY="7d0049c9992505326ca78a42d89dcfefa2e3d51a"
export WANDB_ENTITY="liushb9-peking-university"

cd "$ROOT_DIR"

# ---- (optional) convert raw HDF5 -> LeRobot --------------------------------
if [[ -d "$RAW_DATA_DIR" ]]; then
  echo "[prepare] Converting $RAW_DATA_DIR -> $OUTPUT_ROOT/$REPO_NAME"
  "$PYTHON_BIN" examples/libero/convert_hdf5_stride5_to_lerobot_force.py \
    --data-dir "$RAW_DATA_DIR" \
    --repo-name "$REPO_NAME" \
    --output-dir "$OUTPUT_ROOT" \
    --overwrite \
    --task "$TASK_TEXT" \
    --force-history-len 8 \
    --stride 5 \
    --use-front-as-wrist \
    --image-height 192 \
    --image-width 256 \
    --fps 6
else
  echo "[prepare] Raw HDF5 dir $RAW_DATA_DIR not found — skipping convert step."
  echo "[prepare] Using existing LeRobot dataset at $OUTPUT_ROOT/$REPO_NAME"
fi

# ---- compute norm stats (Franka data, NOT pi05_droid stats) -----------------
echo "[prepare] Computing norm stats for config $CONFIG_NAME"
"$PYTHON_BIN" scripts/compute_norm_stats.py \
  --config-name "$CONFIG_NAME"
