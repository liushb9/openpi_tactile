#!/usr/bin/env bash
set -euo pipefail
trap 'echo "ERROR: command failed at line $LINENO: $BASH_COMMAND" >&2' ERR

ROOT_DIR="/mnt/public/gaojiahuan/liushengbang/openpi_tactile"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

RAW_DATA_DIR="/mnt/public/gaojiahuan/liushengbang/Data/Origin/hanoi_tower"
ROLLOUT_0520_DIR="/mnt/public/gaojiahuan/liushengbang/Data/Rollout/hanoi_tower_0520_force_stride5_posecont_dupfront_chunk4"
ROLLOUT_0526_DIR="/mnt/public/gaojiahuan/liushengbang/Data/Rollout/hanoi_tower_0526_force_stride5_posecont_dupfront_chunk4"
ROLLOUT_SUCCESS_DIR="/mnt/public/gaojiahuan/liushengbang/Data/Rollout/hanoi_tower_0520_0526_force_stride5_posecont_dupfront_chunk4_success_only"
OUTPUT_ROOT="/mnt/public/gaojiahuan/liushengbang/Data/processed_lerobot/wan_worldrl_stride5_posecont_192x256"
REPO_NAME="hanoi_tower_origin_rollout0520_0526_force_stride5_posecont_dupfront_192x256_success_advantage"
CONFIG_NAME="pi05_hanoi_tower_origin_rollout0520_0526_force_stride5_posecont_dupfront_192x256_advantage_10k"
TASK_TEXT="pick up the rings one by one from the center stack and place them on the side pegs"

export HF_LEROBOT_HOME="$OUTPUT_ROOT"
export OPENPI_DATA_HOME="$ROOT_DIR"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"
export WANDB_ENTITY="${WANDB_ENTITY:-liushb9-peking-university}"

cd "$ROOT_DIR"

echo "[prepare] Rebuilding hanoi combined success-only rollout symlink dir -> $ROLLOUT_SUCCESS_DIR"
rm -rf "$ROLLOUT_SUCCESS_DIR"
mkdir -p "$ROLLOUT_SUCCESS_DIR"
"$PYTHON_BIN" - "$ROLLOUT_SUCCESS_DIR" "0520:$ROLLOUT_0520_DIR" "0526:$ROLLOUT_0526_DIR" <<'PY'
import csv
import sys
from pathlib import Path

dst_dir = Path(sys.argv[1])
expected = {"0520": 10, "0526": 23}
total_success = 0

for spec in sys.argv[2:]:
    label, src = spec.split(":", 1)
    src_dir = Path(src)
    csv_path = src_dir / "annotation.csv"
    marked = set()
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            episode = (row.get("episode") or row.get("traj_id") or "").strip()
            if episode:
                marked.add(episode)

    hdf5_files = sorted(src_dir.glob("*.hdf5"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    success = 0
    for hdf5_path in hdf5_files:
        if hdf5_path.stem in marked:
            continue
        (dst_dir / f"{label}_{hdf5_path.name}").symlink_to(hdf5_path)
        success += 1

    print(f"hanoi_tower rollout{label}: success={success}, marked={len(marked)}, total_hdf5={len(hdf5_files)}")
    if success != expected[label]:
        raise SystemExit(f"Expected {expected[label]} successful rollout{label} episodes, found {success}.")
    total_success += success

if total_success != 33:
    raise SystemExit(f"Expected 33 combined successful rollout episodes, found {total_success}.")
print(f"combined_success={total_success}")
PY

success_count="$(find "$ROLLOUT_SUCCESS_DIR" -maxdepth 1 -type l -name '*.hdf5' | wc -l)"
echo "[prepare] Combined success rollout symlinks: $success_count"

echo "[prepare] Rebuilding origin + 0520/0526 successful rollout LeRobot dataset -> $OUTPUT_ROOT/$REPO_NAME"
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
