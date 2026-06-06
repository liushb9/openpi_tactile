#!/usr/bin/env bash
set -euo pipefail
trap 'echo "ERROR: command failed at line $LINENO: $BASH_COMMAND" >&2' ERR

OPENPI_DIR="/mnt/public/gaojiahuan/liushengbang/openpi_tactile"
PYTHON_BIN="$OPENPI_DIR/.venv/bin/python"

HF_LEROBOT_HOME="/mnt/public/gaojiahuan/liushengbang/Data/processed_lerobot/wan_worldrl_stride5_posecont_192x256"
TASK_NAME="pi05_beat_xylophone_origin_synth0523_force_stride5_posecont_dupfront_192x256_ki_advantage"
EXP_NAME="${EXP_NAME:-beat_xylophone_origin_synth0523_success_advantage_ki_from_origin19999_h30}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:---overwrite}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FSDP_DEVICES="${FSDP_DEVICES:-8}"

export HF_LEROBOT_HOME
export OPENPI_DATA_HOME="/mnt/public/gaojiahuan/liushengbang/openpi_tactile"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"
export WANDB_API_KEY="${WANDB_API_KEY:-7d0049c9992505326ca78a42d89dcfefa2e3d51a}"
export WANDB_ENTITY="${WANDB_ENTITY:-liushb9-peking-university}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DISABLE_CODE="${WANDB_DISABLE_CODE:-true}"
export WANDB_DISABLE_GIT="${WANDB_DISABLE_GIT:-true}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export WANDB_SERVICE_TRANSPORT="${WANDB_SERVICE_TRANSPORT:-tcp}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"
export WANDB_HTTP_TIMEOUT="${WANDB_HTTP_TIMEOUT:-60}"
export WANDB_FILE_PUSHER_TIMEOUT="${WANDB_FILE_PUSHER_TIMEOUT:-60}"

cd "$OPENPI_DIR"

"$PYTHON_BIN" scripts/train.py \
  "$TASK_NAME" \
  --exp-name "$EXP_NAME" \
  $OVERWRITE_FLAG \
  --num-workers "$NUM_WORKERS" \
  --fsdp-devices "$FSDP_DEVICES"
