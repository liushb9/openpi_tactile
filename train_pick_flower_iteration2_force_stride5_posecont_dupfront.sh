#!/usr/bin/env bash
set -euo pipefail
trap 'echo "ERROR: command failed at line $LINENO: $BASH_COMMAND" >&2' ERR

OPENPI_DIR="/share/project/liushengbang/openpi_tactile"
PYTHON_BIN="/share/project/chensixiang/Programs/miniconda3/envs/openpi_tactile_311/bin/python"

HF_LEROBOT_HOME="/share/project/liushengbang/Data/processed_lerobot/wan_worldrl_stride5_posecont_192x256"
OPENPI_DATA_HOME="/share/project/liushengbang/openpi_tactile"

TASK_NAME="pi05_pick_flower_iteration2_force_stride5_posecont_dupfront_192x256"
EXP_NAME="${EXP_NAME:-pick_flower_iteration2_origin_synth12_from_iteration1_0505_h30_fscale10}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:---overwrite}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FSDP_DEVICES="${FSDP_DEVICES:-8}"

export HF_LEROBOT_HOME
export OPENPI_DATA_HOME
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export http_proxy="http://hjtan:7kg-Ppp2Ay0a@10.8.36.1:3128"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$http_proxy"

export WANDB_BASE_URL="https://api.bandw.top"
export WANDB_API_KEY="7d0049c9992505326ca78a42d89dcfefa2e3d51a"
export WANDB_ENTITY="liushb9-peking-university"
export WANDB_MODE="${WANDB_MODE:-online}"

cd "$OPENPI_DIR"

"$PYTHON_BIN" scripts/train.py \
  "$TASK_NAME" \
  --exp-name "$EXP_NAME" \
  $OVERWRITE_FLAG \
  --num-workers "$NUM_WORKERS" \
  --fsdp-devices "$FSDP_DEVICES"
