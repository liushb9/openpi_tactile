#!/usr/bin/env bash
set -euo pipefail
trap 'echo "ERROR: command failed at line $LINENO: $BASH_COMMAND" >&2' ERR

ROOT_DIR="/mnt/public/gaojiahuan/liushengbang/openpi_tactile"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

RAW_DATA_DIR="/mnt/public/gaojiahuan/liushengbang/Data/Origin/twist_bottle_cap"
ROLLOUT1_DIR="/mnt/public/gaojiahuan/liushengbang/Data/Rollout/twist_bottle_cap1"
ROLLOUT1_CSV="$ROLLOUT1_DIR/annotations.csv"
ROLLOUT_0525_DIR="/mnt/public/gaojiahuan/liushengbang/Data/Rollout/twist_bottle_cap_0525_force_stride5_posecont_dupfront_chunk4"
ROLLOUT_0525_CSV="$ROLLOUT_0525_DIR/annotation.csv"
ROLLOUT_SUCCESS_DIR="/mnt/public/gaojiahuan/liushengbang/Data/Rollout/twist_bottle_cap1_0525_force_stride5_posecont_dupfront_chunk4_success_only"
OUTPUT_ROOT="/mnt/public/gaojiahuan/liushengbang/Data/processed_lerobot/wan_worldrl_stride5_posecont_192x256"
REPO_NAME="twist_bottle_cap_origin_rollout1_0525_force_stride5_posecont_dupfront_192x256_success_advantage"
CONFIG_NAME="pi05_twist_bottle_cap_origin_rollout1_0525_force_stride5_posecont_dupfront_192x256_advantage_10k"
TASK_TEXT="twist the bottle cap and pick up the bottle cap"

export HF_LEROBOT_HOME="$OUTPUT_ROOT"
export OPENPI_DATA_HOME="$ROOT_DIR"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"
export WANDB_ENTITY="${WANDB_ENTITY:-liushb9-peking-university}"

cd "$ROOT_DIR"

echo "[prepare] Rebuilding success-only rollout symlink dir -> $ROLLOUT_SUCCESS_DIR"
rm -rf "$ROLLOUT_SUCCESS_DIR"
mkdir -p "$ROLLOUT_SUCCESS_DIR"
"$PYTHON_BIN" - "$ROLLOUT1_DIR" "$ROLLOUT1_CSV" "$ROLLOUT_SUCCESS_DIR" "rollout1" <<'PY'
import csv
import sys
from pathlib import Path

src_dir = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
dst_dir = Path(sys.argv[3])
prefix = sys.argv[4]

success_ids = []
fail_count = 0
with csv_path.open(newline="") as f:
    for row in csv.DictReader(f):
        episode = (row.get("traj_id") or row.get("episode") or "").strip()
        label = (row.get("label") or "").strip().lower()
        if not episode:
            continue
        if label == "success":
            success_ids.append(episode)
        else:
            fail_count += 1

for episode in success_ids:
    src = src_dir / f"{episode}.hdf5"
    if not src.exists():
        raise FileNotFoundError(src)
    (dst_dir / f"{prefix}_{episode}.hdf5").symlink_to(src)

print(f"{prefix}: success={len(success_ids)}, marked_fail={fail_count}")
PY
"$PYTHON_BIN" - "$ROLLOUT_0525_DIR" "$ROLLOUT_0525_CSV" "$ROLLOUT_SUCCESS_DIR" "0525" <<'PY'
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
        episode = (row.get("episode") or row.get("traj_id") or "").strip()
        if episode:
            marked.add(episode)

count = 0
for hdf5_path in sorted(src_dir.glob("*.hdf5"), key=lambda p: (not p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else p.stem)):
    if hdf5_path.stem in marked:
        continue
    (dst_dir / f"{prefix}_{hdf5_path.stem}.hdf5").symlink_to(hdf5_path)
    count += 1
print(f"{prefix}: success={count}, marked_fail={len(marked)}")
PY
success_count="$(find "$ROLLOUT_SUCCESS_DIR" -maxdepth 1 -type l -name '*.hdf5' | wc -l)"
if [[ "$success_count" != "55" ]]; then
  echo "Expected 55 successful rollout symlinks, got $success_count" >&2
  exit 1
fi

echo "[prepare] Rebuilding origin + twist_bottle_cap1/0525 successful rollout LeRobot dataset -> $OUTPUT_ROOT/$REPO_NAME"
"$PYTHON_BIN" examples/libero/convert_twist_bottle_cap_origin_rollout_advantage_to_lerobot_force.py \
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
