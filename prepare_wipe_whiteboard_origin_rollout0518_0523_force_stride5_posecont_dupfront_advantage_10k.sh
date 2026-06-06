#!/usr/bin/env bash
set -euo pipefail
trap 'echo "ERROR: command failed at line $LINENO: $BASH_COMMAND" >&2' ERR

ROOT_DIR="/mnt/public/gaojiahuan/liushengbang/openpi_tactile"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

RAW_DATA_DIR="/mnt/public/gaojiahuan/liushengbang/Data/Origin/wipe_whiteboard"
ROLLOUT_0518_DIR="/mnt/public/gaojiahuan/liushengbang/Data/Rollout/wipe_whiteboard_0518_force_stride5_posecont_dupfront_chunk4"
ROLLOUT_0518_CSV="$ROLLOUT_0518_DIR/annotation .csv"
ROLLOUT_0523_DIR="/mnt/public/gaojiahuan/liushengbang/Data/Rollout/wipe_whiteboard_0523_force_stride5_posecont_dupfront_chunk4"
ROLLOUT_0523_CSV="$ROLLOUT_0523_DIR/annotation .csv"
ROLLOUT_SUCCESS_DIR="/mnt/public/gaojiahuan/liushengbang/Data/Rollout/wipe_whiteboard_0518_0523_force_stride5_posecont_dupfront_chunk4_success_only"
OUTPUT_ROOT="/mnt/public/gaojiahuan/liushengbang/Data/processed_lerobot/wan_worldrl_stride5_posecont_192x256"
REPO_NAME="wipe_whiteboard_origin_rollout0518_0523_force_stride5_posecont_dupfront_192x256_success_advantage"
CONFIG_NAME="pi05_wipe_whiteboard_origin_rollout0518_0523_force_stride5_posecont_dupfront_192x256_advantage_10k"
TASK_TEXT="wipe the star mark on the whiteboard"

export HF_LEROBOT_HOME="$OUTPUT_ROOT"
export OPENPI_DATA_HOME="$ROOT_DIR"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"
export WANDB_ENTITY="${WANDB_ENTITY:-liushb9-peking-university}"

cd "$ROOT_DIR"

echo "[prepare] Rebuilding success-only rollout symlink dir -> $ROLLOUT_SUCCESS_DIR"
rm -rf "$ROLLOUT_SUCCESS_DIR"
mkdir -p "$ROLLOUT_SUCCESS_DIR"
"$PYTHON_BIN" - "$ROLLOUT_0518_DIR" "$ROLLOUT_0518_CSV" "$ROLLOUT_SUCCESS_DIR" "0518" <<'PY'
import csv
import sys
from pathlib import Path

src_dir = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
dst_dir = Path(sys.argv[3])
prefix = sys.argv[4]

marked = set()
with csv_path.open(newline="") as f:
    for row in csv.DictReader(f):
        episode = (row.get("episode") or "").strip()
        if episode:
            marked.add(episode)

count = 0
for hdf5_path in sorted(src_dir.glob("*.hdf5"), key=lambda p: int(p.stem)):
    if hdf5_path.stem in marked:
        continue
    (dst_dir / f"{prefix}_{hdf5_path.stem}.hdf5").symlink_to(hdf5_path)
    count += 1
print(f"{prefix}: success={count}, marked={len(marked)}")
PY
"$PYTHON_BIN" - "$ROLLOUT_0523_DIR" "$ROLLOUT_0523_CSV" "$ROLLOUT_SUCCESS_DIR" "0523" <<'PY'
import csv
import sys
from pathlib import Path

src_dir = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
dst_dir = Path(sys.argv[3])
prefix = sys.argv[4]

marked = set()
with csv_path.open(newline="") as f:
    for row in csv.DictReader(f):
        episode = (row.get("episode") or "").strip()
        if episode:
            marked.add(episode)

count = 0
for hdf5_path in sorted(src_dir.glob("*.hdf5"), key=lambda p: int(p.stem)):
    if hdf5_path.stem in marked:
        continue
    (dst_dir / f"{prefix}_{hdf5_path.stem}.hdf5").symlink_to(hdf5_path)
    count += 1
print(f"{prefix}: success={count}, marked={len(marked)}")
PY
success_count="$(find "$ROLLOUT_SUCCESS_DIR" -maxdepth 1 -type l -name '*.hdf5' | wc -l)"
if [[ "$success_count" != "20" ]]; then
  echo "Expected 20 successful rollout symlinks, got $success_count" >&2
  exit 1
fi

echo "[prepare] Rebuilding origin + 0518/0523 successful rollout LeRobot dataset -> $OUTPUT_ROOT/$REPO_NAME"
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
