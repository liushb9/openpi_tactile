#!/usr/bin/env bash
set -euo pipefail

# ---- machine-specific paths -------------------------------------------------
ROOT_DIR="/mnt/public2/liushengbang/openpi_tactile"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

# Raw HDF5 source. Optional: if not present, the convert step is skipped and
# we go straight to compute_norm_stats on the already-converted dataset.
RAW_DATA_DIR="/mnt/public2/liushengbang/Data/Origin/pick_flower"
SYNTH_DATA_DIR="/mnt/public2/liushengbang/Data/World-RL/Data/idm_infer/WM_Rollout_merged/pick_flower1_nosplus20_n5_ckpt7145"

# LeRobot output root and per-task repo name.
OUTPUT_ROOT="/mnt/public2/liushengbang/Data/processed_lerobot/wan_worldrl_stride5_posecont_192x256"
REPO_NAME="pick_flower_force_stride5_posecont_dupfront_192x256_origin_wmrollout"
CONFIG_NAME="pi05_pick_flower_force_stride5_posecont_dupfront_192x256"
TASK_TEXT="pick up the flower and place in to the vase"

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
if [[ -d "$RAW_DATA_DIR" && -d "$SYNTH_DATA_DIR" ]]; then
  echo "[prepare] Converting original + synthetic pick_flower data -> $OUTPUT_ROOT/$REPO_NAME"
  "$PYTHON_BIN" examples/libero/convert_mixed_wipe_to_lerobot_force.py \
    --original-hdf5-dir "$RAW_DATA_DIR" \
    --failure-dir "$SYNTH_DATA_DIR" \
    --repo-name "$REPO_NAME" \
    --output-dir "$OUTPUT_ROOT" \
    --overwrite \
    --task "$TASK_TEXT" \
    --force-history-len 8 \
    --stride 5 \
    --image-height 192 \
    --image-width 256 \
    --fps 6
else
  echo "[prepare] Missing source dir(s):"
  echo "  RAW_DATA_DIR=$RAW_DATA_DIR"
  echo "  SYNTH_DATA_DIR=$SYNTH_DATA_DIR"
  echo "[prepare] Using existing LeRobot dataset at $OUTPUT_ROOT/$REPO_NAME"
fi

# ---- compute norm stats (Franka data, NOT pi05_droid stats) -----------------
echo "[prepare] Computing norm stats for config $CONFIG_NAME"
"$PYTHON_BIN" scripts/compute_norm_stats.py \
  --config-name "$CONFIG_NAME"
