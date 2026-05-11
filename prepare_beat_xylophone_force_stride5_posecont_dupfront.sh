#!/usr/bin/env bash
set -euo pipefail

# ---- machine-specific paths -------------------------------------------------
ROOT_DIR="/share/project/liushengbang/openpi_tactile"
PYTHON_BIN="/share/project/chensixiang/Programs/miniconda3/envs/openpi_tactile_311/bin/python"

# Raw HDF5 source.
RAW_DATA_DIR="/share/project/liushengbang/Data/Origin/beat_xylophone"

# LeRobot output root and per-task repo name.
OUTPUT_ROOT="/share/project/liushengbang/Data/processed_lerobot/wan_worldrl_stride5_posecont_192x256"
REPO_NAME="beat_xylophone_force_stride5_posecont_dupfront_192x256"
CONFIG_NAME="pi05_beat_xylophone_force_stride5_posecont_dupfront_192x256"
TASK_TEXT="beat the xylophone"

# OpenPI cache/checkpoint root.
OPENPI_DATA_HOME="/share/project/liushengbang/openpi_tactile"

# ---- env --------------------------------------------------------------------
export HF_LEROBOT_HOME="$OUTPUT_ROOT"
export OPENPI_DATA_HOME
export http_proxy="http://hjtan:7kg-Ppp2Ay0a@10.8.36.1:3128"
export https_proxy="http://hjtan:7kg-Ppp2Ay0a@10.8.36.1:3128"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"

# wandb (compute_norm_stats does not log; kept here for symmetry with train.sh).
export WANDB_BASE_URL="https://api.bandw.top"
export WANDB_API_KEY="7d0049c9992505326ca78a42d89dcfefa2e3d51a"
export WANDB_ENTITY="liushb9-peking-university"

cd "$ROOT_DIR"

# ---- convert raw HDF5 -> LeRobot -------------------------------------------
if [[ -d "$OUTPUT_ROOT/$REPO_NAME" ]]; then
  echo "[prepare] Existing LeRobot dataset found at $OUTPUT_ROOT/$REPO_NAME"
  echo "[prepare] Skip conversion. Remove the directory first if you want to rebuild it."
else
  echo "[prepare] Converting beat_xylophone data -> $OUTPUT_ROOT/$REPO_NAME"
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
fi

# ---- compute norm stats -----------------------------------------------------
echo "[prepare] Computing norm stats for config $CONFIG_NAME"
"$PYTHON_BIN" scripts/compute_norm_stats.py \
  --config-name "$CONFIG_NAME"
