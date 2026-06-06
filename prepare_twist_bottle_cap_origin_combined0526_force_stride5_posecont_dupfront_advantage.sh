#!/usr/bin/env bash
set -euo pipefail
trap 'echo "ERROR: command failed at line $LINENO: $BASH_COMMAND" >&2' ERR

ROOT_DIR="/mnt/public/gaojiahuan/liushengbang/openpi_tactile"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

RAW_DATA_DIR="/mnt/public/gaojiahuan/liushengbang/Data/Origin/twist_bottle_cap"
SYNTH_DATA_DIR="/mnt/public/gaojiahuan/liushengbang/Data/synthetic_data/twist_bottle_cap_0523_combined"
SYNTH_DATA_DIR_2="/mnt/public/gaojiahuan/liushengbang/Data/synthetic_data/twist_bottle_cap_0526_combined"
OUTPUT_ROOT="/mnt/public/gaojiahuan/liushengbang/Data/processed_lerobot/wan_worldrl_stride5_posecont_192x256"
REPO_NAME="twist_bottle_cap_origin_combined0526_force_stride5_posecont_dupfront_192x256_success_advantage"
CONFIG_NAME="pi05_twist_bottle_cap_origin_combined0526_force_stride5_posecont_dupfront_192x256_advantage"
TASK_TEXT="twist the bottle cap and pick up the bottle cap"

export HF_LEROBOT_HOME="$OUTPUT_ROOT"
export OPENPI_DATA_HOME="$ROOT_DIR"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"
export WANDB_ENTITY="${WANDB_ENTITY:-liushb9-peking-university}"

cd "$ROOT_DIR"

echo "[prepare] Rebuilding twist_bottle_cap origin + synthetic0523 + synthetic0526 LeRobot dataset -> $OUTPUT_ROOT/$REPO_NAME"
"$PYTHON_BIN" examples/libero/convert_twist_bottle_cap_origin_synth_advantage_to_lerobot_force.py \
  --original-hdf5-dir "$RAW_DATA_DIR" \
  --synthetic-dir "$SYNTH_DATA_DIR" \
  --synthetic-dir-2 "$SYNTH_DATA_DIR_2" \
  --repo-name "$REPO_NAME" \
  --output-dir "$OUTPUT_ROOT" \
  --overwrite \
  --task "$TASK_TEXT" \
  --force-history-len 8 \
  --stride 5 \
  --image-height 192 \
  --image-width 256 \
  --fps 6 \
  --advantage-value 1.0

echo "[prepare] Computing norm stats for config $CONFIG_NAME"
JAX_PLATFORMS=cpu "$PYTHON_BIN" scripts/compute_norm_stats.py \
  --config-name "$CONFIG_NAME"
