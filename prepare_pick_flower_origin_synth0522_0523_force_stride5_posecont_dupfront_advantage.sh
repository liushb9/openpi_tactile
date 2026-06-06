#!/usr/bin/env bash
set -euo pipefail
trap 'echo "ERROR: command failed at line $LINENO: $BASH_COMMAND" >&2' ERR

ROOT_DIR="/mnt/public/gaojiahuan/liushengbang/openpi_tactile"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

RAW_DATA_DIR="/mnt/public/gaojiahuan/liushengbang/Data/Origin/pick_flower"
SYNTH_DATA_DIR_0522="/mnt/public/gaojiahuan/liushengbang/Data/synthetic_data/pick_flower1_0522"
SYNTH_DATA_DIR_0523="/mnt/public/gaojiahuan/liushengbang/Data/synthetic_data/pick_flower_0523_combined"
OUTPUT_ROOT="/mnt/public/gaojiahuan/liushengbang/Data/processed_lerobot/wan_worldrl_stride5_posecont_192x256"
REPO_NAME="pick_flower_origin_synth0522_0523_force_stride5_posecont_dupfront_192x256_success_advantage"
CONFIG_NAME="pi05_pick_flower_origin_synth0522_0523_force_stride5_posecont_dupfront_192x256_advantage"
TASK_TEXT="pick up the flower and put it in the vase"

export HF_LEROBOT_HOME="$OUTPUT_ROOT"
export OPENPI_DATA_HOME="/mnt/public/gaojiahuan/liushengbang/openpi_tactile"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"
export WANDB_ENTITY="${WANDB_ENTITY:-liushb9-peking-university}"

cd "$ROOT_DIR"

if [[ "${REBUILD_DATASET:-0}" == "1" ]]; then
  echo "[prepare] Rebuilding origin + 0522 synthetic + 0523 synthetic LeRobot dataset -> $OUTPUT_ROOT/$REPO_NAME"
  "$PYTHON_BIN" examples/libero/convert_pick_flower_origin_synth0522_0523_advantage_to_lerobot_force.py \
    --original-hdf5-dir "$RAW_DATA_DIR" \
    --synthetic-dir-0522 "$SYNTH_DATA_DIR_0522" \
    --synthetic-dir-0523 "$SYNTH_DATA_DIR_0523" \
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
else
  echo "[prepare] Reusing existing LeRobot dataset -> $OUTPUT_ROOT/$REPO_NAME"
fi

echo "[prepare] Computing norm stats for config $CONFIG_NAME"
JAX_PLATFORMS=cpu "$PYTHON_BIN" scripts/compute_norm_stats.py \
  --config-name "$CONFIG_NAME"
