
uv run /media/jiayueru/Ckpt/WRL/openpi/examples/libero/convert_own_data_to_lerobot_abs.py \
    --data-dir /media/jiayueru/Ckpt/WRL/data/WorldRL-pickplace-bread-full/position0123_20260331 \
    --repo-name pickplace_bread_pos0123 \
    --output-dir /media/jiayueru/Ckpt/WRL/data/processed_lerobot/WorldRL-pickplace-bread-full/position012_20260331 \
    --action-interval 1 \
    --overwrite \
    --task "pick up the bread and put it in the plate"

cd /media/jiayueru/Ckpt/WRL/openpi

export HF_LEROBOT_HOME=/media/jiayueru/Ckpt/WRL/data/processed_lerobot/WorldRL-pickplace-bread-full/position012_20260331

uv run scripts/compute_norm_stats.py \
    --config-name pi05_bread_pos0123