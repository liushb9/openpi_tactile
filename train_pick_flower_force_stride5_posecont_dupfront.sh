#!/usr/bin/env bash
set -euo pipefail
trap 'echo "ERROR: command failed at line $LINENO: $BASH_COMMAND" >&2' ERR

CONDA_BASE="/share/project/chensixiang/Programs/miniconda3"
ENV_NAME="openpi_tactile_311"
PYTHON_BIN="$CONDA_BASE/envs/$ENV_NAME/bin/python"

OPENPI_DIR="/share/project/liushengbang/openpi_tactile"
HF_LEROBOT_HOME="/share/project/liushengbang/Data/processed_lerobot/wan_worldrl_stride5_posecont_192x256"
OPENPI_DATA_HOME="/share/project/liushengbang/.cache/openpi"

TASK_NAME="pi05_pick_flower_force_stride5_posecont_dupfront_192x256"
EXP_NAME="${EXP_NAME:-run_stride5_posecont_dupfront_192x256}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:---overwrite}"
NUM_WORKERS="${NUM_WORKERS:-8}"

export LD_LIBRARY_PATH="$CONDA_BASE/envs/$ENV_NAME/lib:${LD_LIBRARY_PATH:-}"
export HF_LEROBOT_HOME
export OPENPI_DATA_HOME
export WANDB_MODE="${WANDB_MODE:-online}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

cd "$OPENPI_DIR"

"$PYTHON_BIN" scripts/train.py \
  "$TASK_NAME" \
  --exp-name "$EXP_NAME" \
  $OVERWRITE_FLAG \
  --num-workers "$NUM_WORKERS"
