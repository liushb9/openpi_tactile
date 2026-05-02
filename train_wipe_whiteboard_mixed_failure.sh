#!/usr/bin/env bash
set -euo pipefail
trap 'echo "ERROR: command failed at line $LINENO: $BASH_COMMAND" >&2' ERR

OPENPI_DIR="/mnt/public2/liushengbang/openpi_tactile"
PYTHON_BIN="$OPENPI_DIR/.venv/bin/python"

HF_LEROBOT_HOME="/mnt/public2/liushengbang/Data/processed_lerobot/wan_worldrl_stride5_posecont_192x256"
OPENPI_DATA_HOME="/mnt/public2/liushengbang/openpi_data/openpi-assets"

TASK_NAME="pi05_wipe_whiteboard_force_stride5_posecont_dupfront_192x256_mixed_failure"
EXP_NAME="${EXP_NAME:-wipe_whiteboard_mixed_failure_from20k}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:---overwrite}"
NUM_WORKERS="${NUM_WORKERS:-8}"

export HF_LEROBOT_HOME
export OPENPI_DATA_HOME
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

export WANDB_BASE_URL="https://api.bandw.top"
export WANDB_API_KEY="7d0049c9992505326ca78a42d89dcfefa2e3d51a"
export WANDB_ENTITY="liushb9-peking-university"
export WANDB_MODE="${WANDB_MODE:-online}"

cd "$OPENPI_DIR"

"$PYTHON_BIN" scripts/train.py \
  "$TASK_NAME" \
  --exp-name "$EXP_NAME" \
  $OVERWRITE_FLAG \
  --num-workers "$NUM_WORKERS"
