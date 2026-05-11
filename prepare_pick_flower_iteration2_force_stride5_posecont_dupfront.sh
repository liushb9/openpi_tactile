#!/usr/bin/env bash
set -euo pipefail

# ---- machine-specific paths -------------------------------------------------
ROOT_DIR="/share/project/liushengbang/openpi_tactile"
PYTHON_BIN="/share/project/chensixiang/Programs/miniconda3/envs/openpi_tactile_311/bin/python"

RAW_DATA_DIR="/share/project/liushengbang/Data/Origin/pick_flower"
SYNTH_DATA_DIR_1="/share/project/liushengbang/Data/Synthetic_data/synthetic_data/pick_flower1_nosplus20_n5_ckpt7145"
SYNTH_DATA_DIR_2="/share/project/liushengbang/Data/Synthetic_data/synthetic_data/pick_flower2_failure_ckpt7145"

OUTPUT_ROOT="/share/project/liushengbang/Data/processed_lerobot/wan_worldrl_stride5_posecont_192x256"
REPO_NAME="pick_flower_iteration2_force_stride5_posecont_dupfront_192x256_origin_synth12"
CONFIG_NAME="pi05_pick_flower_iteration2_force_stride5_posecont_dupfront_192x256"
TASK_TEXT="pick up the flower and put it in the vase"

OPENPI_DATA_HOME="/share/project/liushengbang/openpi_tactile"

# ---- env --------------------------------------------------------------------
export HF_LEROBOT_HOME="$OUTPUT_ROOT"
export OPENPI_DATA_HOME
export http_proxy="http://hjtan:7kg-Ppp2Ay0a@10.8.36.1:3128"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$http_proxy"

export WANDB_BASE_URL="https://api.bandw.top"
export WANDB_API_KEY="7d0049c9992505326ca78a42d89dcfefa2e3d51a"
export WANDB_ENTITY="liushb9-peking-university"

cd "$ROOT_DIR"

if [[ -d "$OUTPUT_ROOT/$REPO_NAME" ]]; then
  echo "[prepare] Existing LeRobot dataset found at $OUTPUT_ROOT/$REPO_NAME"
  echo "[prepare] Skip conversion. Remove the directory first if you want to rebuild it."
else
  echo "[prepare] Converting pick_flower iteration2 data -> $OUTPUT_ROOT/$REPO_NAME"
  "$PYTHON_BIN" examples/libero/convert_pick_flower_iteration2_to_lerobot_force.py \
    --original-hdf5-dir "$RAW_DATA_DIR" \
    --failure-dir-1 "$SYNTH_DATA_DIR_1" \
    --failure-dir-2 "$SYNTH_DATA_DIR_2" \
    --repo-name "$REPO_NAME" \
    --output-dir "$OUTPUT_ROOT" \
    --overwrite \
    --task "$TASK_TEXT" \
    --force-history-len 8 \
    --stride 5 \
    --image-height 192 \
    --image-width 256 \
    --fps 6
fi

echo "[prepare] Computing norm stats for config $CONFIG_NAME"
"$PYTHON_BIN" scripts/compute_norm_stats.py \
  --config-name "$CONFIG_NAME"
