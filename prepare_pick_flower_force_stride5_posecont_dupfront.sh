#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/share/project/liushengbang/openpi_tactile"
CONDA_BASE="/share/project/chensixiang/Programs/miniconda3"
ENV_NAME="openpi_tactile_311"
UV_BIN="$CONDA_BASE/bin/uv"
PYTHON_BIN="$CONDA_BASE/envs/$ENV_NAME/bin/python"

RAW_DATA_DIR="/share/project/liushengbang/Data/pick_flower"
OUTPUT_ROOT="/share/project/liushengbang/Data/processed_lerobot/wan_worldrl_stride5_posecont_192x256"
REPO_NAME="pick_flower_force_stride5_posecont_dupfront_192x256"
CONFIG_NAME="pi05_pick_flower_force_stride5_posecont_dupfront_192x256"
TASK_TEXT="pick up the flower and place in to the vase"

export HF_LEROBOT_HOME="$OUTPUT_ROOT"
export UV_PYTHON="$PYTHON_BIN"
export UV_PYTHON_DOWNLOADS=never

cd "$ROOT_DIR"

"$UV_BIN" run python examples/libero/convert_hdf5_stride5_to_lerobot_force.py \
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

"$UV_BIN" run python scripts/compute_norm_stats.py \
  --config-name "$CONFIG_NAME"
