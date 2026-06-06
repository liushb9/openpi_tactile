#!/usr/bin/env bash
set -euo pipefail
trap 'echo "ERROR: command failed at line $LINENO: $BASH_COMMAND" >&2' ERR

ROOT_DIR="/mnt/public/gaojiahuan/liushengbang/openpi_tactile"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

RAW_DATA_DIR="/mnt/public/gaojiahuan/liushengbang/Data/Origin/pick_flower"
ROLLOUT_DATA_DIR="/mnt/public/gaojiahuan/liushengbang/Data/Rollout/pick_flower_0520_force_stride5_posecont_dupfront_chunk4"
ROLLOUT_SUCCESS_DIR="/mnt/public/gaojiahuan/liushengbang/Data/Rollout/pick_flower_0520_force_stride5_posecont_dupfront_chunk4_success_only"
OUTPUT_ROOT="/mnt/public/gaojiahuan/liushengbang/Data/processed_lerobot/wan_worldrl_stride5_posecont_192x256"
REPO_NAME="pick_flower_origin_rollout0520_force_stride5_posecont_dupfront_192x256_success_advantage"
CONFIG_NAME="pi05_pick_flower_origin_rollout0520_force_stride5_posecont_dupfront_192x256_advantage_10k"
TASK_TEXT="pick up the flower and put it in the vase"

export HF_LEROBOT_HOME="$OUTPUT_ROOT"
export OPENPI_DATA_HOME="$ROOT_DIR"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"
export WANDB_ENTITY="${WANDB_ENTITY:-liushb9-peking-university}"

cd "$ROOT_DIR"

echo "[prepare] Rebuilding success-only rollout symlink dir -> $ROLLOUT_SUCCESS_DIR"
rm -rf "$ROLLOUT_SUCCESS_DIR"
mkdir -p "$ROLLOUT_SUCCESS_DIR"
for episode in 3 4 5 6 7 9 11 13 14 15 17 19 23 24 25 27 30 31 32 33 34 35 37; do
  ln -s "$ROLLOUT_DATA_DIR/${episode}.hdf5" "$ROLLOUT_SUCCESS_DIR/${episode}.hdf5"
done
success_count="$(find "$ROLLOUT_SUCCESS_DIR" -maxdepth 1 -type l -name '*.hdf5' | wc -l)"
if [[ "$success_count" != "23" ]]; then
  echo "Expected 23 successful rollout symlinks, got $success_count" >&2
  exit 1
fi

echo "[prepare] Rebuilding origin + 0520 successful rollout LeRobot dataset -> $OUTPUT_ROOT/$REPO_NAME"
"$PYTHON_BIN" examples/libero/convert_pick_flower_origin_rollout_advantage_to_lerobot_force.py \
  --original-hdf5-dir "$RAW_DATA_DIR" \
  --rollout-hdf5-dir "$ROLLOUT_SUCCESS_DIR" \
  --repo-name "$REPO_NAME" \
  --output-dir "$OUTPUT_ROOT" \
  --overwrite \
  --task "$TASK_TEXT" \
  --force-history-len 8 \
  --original-stride 5 \
  --rollout-stride 1 \
  --image-height 192 \
  --image-width 256 \
  --fps 6 \
  --advantage-value 1.0

echo "[prepare] Computing norm stats for config $CONFIG_NAME"
JAX_PLATFORMS=cpu "$PYTHON_BIN" scripts/compute_norm_stats.py \
  --config-name "$CONFIG_NAME"
