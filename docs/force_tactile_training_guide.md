# Pi0.5 Force/Tactile 微调训练全流程

**适用场景**: 在 pi0.5 模型中加入力/触觉 (force_history) 条件，基于 Franka 机械臂 HDF5 演示数据微调  
**工作目录**: `/media/jiayueru/Ckpt/WRL/openpi_tactile`  
**数据根目录**: `/media/jiayueru/Ckpt/WRL/data`

---

## 0. 流程总览

```
原始 HDF5 数据
  ↓
[可选] 抽帧 → framestep5 格式 (video.mp4 + npy)
  ↓
数据格式转换 → LeRobot 格式 (parquet + images + meta)
  ↓
在 config.py 中注册训练配置
  ↓
计算归一化统计量 (norm_stats: state + actions + force_history)
  ↓
启动训练
```

---

## 1. 环境变量设置 (所有命令通用前缀)

```bash
export CONDA_ROOT="/media/jiayueru/Ckpt/miniconda3"
export UV_BIN="$CONDA_ROOT/bin/uv"
export UV_PYTHON="$CONDA_ROOT/envs/openpi311/bin/python"
export UV_PYTHON_DOWNLOADS=never
export LD_LIBRARY_PATH="$CONDA_ROOT/envs/openpi311/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="/media/jiayueru/Ckpt/.cache/huggingface"

cd /media/jiayueru/Ckpt/WRL/openpi_tactile
```

> **注意**: `LD_LIBRARY_PATH` 必须包含 conda 环境的 lib 目录，否则会出现 `CXXABI_1.3.15` 错误。

---

## 2. 源数据格式

### 2.1 HDF5 格式 (原始采集数据)

位于 `/media/jiayueru/Ckpt/WRL/data/WorldRL/<task_name>/`，每个文件一个 episode。

```
pick_flower/
├── 0.hdf5
├── 1.hdf5
└── ...  (共 99 个)
```

每个 HDF5 内部结构:

| 字段 | Shape | 说明 |
|------|-------|------|
| `observations/images/cam_front` | `(T,) object` | 正面相机 JPEG 字节流 |
| `observations/images/cam_high` | `(T,) object` | 高位/腕部相机 JPEG 字节流 |
| `puppet/joint` | `(T, 7) float32` | 关节角 |
| `puppet/pose` | `(T, 6) float32` | 末端位姿 [x,y,z,rx,ry,rz] |
| `puppet/gripper` | `(T, 2) float32` | [:,0] 实际宽度，[:,1] 二值状态 |
| `tactile/force_left` | `(T, 6) float32` | 左手力矩 [Fx,Fy,Fz,Tx,Ty,Tz] |
| `tactile/force_right` | `(T, 6) float32` | 右手力矩 [Fx,Fy,Fz,Tx,Ty,Tz] |

### 2.2 Framestep5 格式 (抽帧后数据)

位于 `/media/jiayueru/Ckpt/WRL/data/WorldRL/<task_name>_framestep5/`，每个子目录一个 episode。

```
pick_flower_framestep5/
├── metadata.csv
├── 0/
│   ├── video.mp4           ← cam_front 视频
│   ├── wrist_video.mp4     ← cam_front 的副本 (dupfront)
│   ├── actions.npy          ← (T', 7) pose(6) + gripper_norm(1)
│   ├── state.npy            ← (T', 8) joint(7) + gripper_norm(1)
│   └── tactile.npy          ← (T', 12) force_left(6) + force_right(6)
├── 1/
└── ...
```

---

## 3. 数据集一览

| 数据集名称 | 来源 | 腕部相机 | Episodes | 转换脚本 |
|---|---|---|---|---|
| `pick_flower_force` | HDF5 | cam_high | 99 | `convert_own_data_to_lerobot_force.py` |
| `wipe_whiteboard_force` | HDF5 | cam_high | 108 | 同上 |
| `pick_flower_force_dupfront` | HDF5 | cam_front 复制 | 99 | 同上 (`--use-front-as-wrist`) |
| `wipe_whiteboard_force_dupfront` | HDF5 | cam_front 复制 | 108 | 同上 (`--use-front-as-wrist`) |
| `pick_flower_force_framestep5` | framestep5 | cam_front 复制 | 99 | `convert_framestep5_to_lerobot_force.py` |
| `wipe_whiteboard_force_framestep5` | framestep5 | cam_front 复制 | 108 | 同上 |

### LeRobot 输出的公共 feature 定义

| Feature | Shape | 说明 |
|---------|-------|------|
| `image` | `(256, 256, 3)` | 正面相机 RGB |
| `wrist_image` | `(256, 256, 3)` | 腕部相机 RGB |
| `state` | `(8,)` | joint(7) + gripper_norm(1) |
| `actions` | `(7,)` | pose(6) + gripper_norm(1) |
| `force_history` | `(8, 12)` | 滑动窗口 8 帧 x (left_wrench(6) + right_wrench(6))，已归一化 |

**gripper_norm**: `raw_gripper > 0.6 → 0.0 (open), else → 1.0 (closed)`  
**force 归一化**: Fx/Fy/Fz 除以 20，Tx/Ty/Tz 除以 2

---

## 4. Step-by-Step: 从 HDF5 转换 (force / force_dupfront)

### 4.1 数据转换

**使用 cam_high 作为腕部相机 (force)**:

```bash
cd /media/jiayueru/Ckpt/WRL/openpi_tactile

# pick_flower_force
$UV_BIN run python examples/libero/convert_own_data_to_lerobot_force.py \
  --data-dir /media/jiayueru/Ckpt/WRL/data/WorldRL/pick_flower \
  --repo-name pick_flower_force \
  --output-dir /media/jiayueru/Ckpt/WRL/data/processed_lerobot/WorldRL/pick_flower_force \
  --overwrite \
  --task "pick up the flower and put it in the vase" \
  --force-history-len 8

# wipe_whiteboard_force
$UV_BIN run python examples/libero/convert_own_data_to_lerobot_force.py \
  --data-dir /media/jiayueru/Ckpt/WRL/data/WorldRL/wipe_whiteboard \
  --repo-name wipe_whiteboard_force \
  --output-dir /media/jiayueru/Ckpt/WRL/data/processed_lerobot/WorldRL/wipe_whiteboard_force \
  --overwrite \
  --task "wipe the whiteboard" \
  --force-history-len 8
```

**使用 cam_front 复制作为腕部相机 (force_dupfront)**:

```bash
# pick_flower_force_dupfront
$UV_BIN run python examples/libero/convert_own_data_to_lerobot_force.py \
  --data-dir /media/jiayueru/Ckpt/WRL/data/WorldRL/pick_flower \
  --repo-name pick_flower_force_dupfront \
  --output-dir /media/jiayueru/Ckpt/WRL/data/processed_lerobot/WorldRL/pick_flower_force_dupfront \
  --overwrite \
  --task "pick up the flower and put it in the vase" \
  --force-history-len 8 \
  --use-front-as-wrist

# wipe_whiteboard_force_dupfront
$UV_BIN run python examples/libero/convert_own_data_to_lerobot_force.py \
  --data-dir /media/jiayueru/Ckpt/WRL/data/WorldRL/wipe_whiteboard \
  --repo-name wipe_whiteboard_force_dupfront \
  --output-dir /media/jiayueru/Ckpt/WRL/data/processed_lerobot/WorldRL/wipe_whiteboard_force_dupfront \
  --overwrite \
  --task "wipe the whiteboard" \
  --force-history-len 8 \
  --use-front-as-wrist
```

### 4.2 验证数据集完整性

```bash
# 检查 meta 文件是否齐全 (必须有这 4 个文件)
ls /media/jiayueru/Ckpt/WRL/data/processed_lerobot/WorldRL/pick_flower_force/pick_flower_force/meta/
# 期望输出: episodes.jsonl  episodes_stats.jsonl  info.json  tasks.jsonl

# 检查 parquet 数量是否等于 episode 数
find /media/jiayueru/Ckpt/WRL/data/processed_lerobot/WorldRL/pick_flower_force/pick_flower_force/data/ -name "*.parquet" | wc -l
# 期望输出: 99
```

> **重要**: 如果 `episodes.jsonl` 缺失，训练会报 `FileNotFoundError`，需要重新跑转换。

---

## 5. Step-by-Step: 从 Framestep5 转换

### 5.1 抽帧 (HDF5 → framestep5)

抽帧脚本: `/media/jiayueru/Codes/DiffSythn-Action/examples/wanvideo/data_conversion/sample_worldrl_pick_flower_every_n.py`

```bash
cd /media/jiayueru/Codes/DiffSythn-Action

# pick_flower
python examples/wanvideo/data_conversion/sample_worldrl_pick_flower_every_n.py \
  --src_dir /media/jiayueru/Ckpt/WRL/data/WorldRL/pick_flower \
  --out_dir /media/jiayueru/Ckpt/WRL/data/WorldRL/pick_flower_framestep5 \
  --prompt "pick up the flower and put it in the vase"

# wipe_whiteboard
python examples/wanvideo/data_conversion/sample_worldrl_pick_flower_every_n.py \
  --src_dir /media/jiayueru/Ckpt/WRL/data/WorldRL/wipe_whiteboard \
  --out_dir /media/jiayueru/Ckpt/WRL/data/WorldRL/wipe_whiteboard_framestep5 \
  --prompt "wipe the whiteboard"
```

抽帧脚本输出:
- `actions.npy`: pose(6) + gripper_norm(1) = 7 维
- `state.npy`: joint(7) + gripper_norm(1) = 8 维
- `tactile.npy`: force_left(6) + force_right(6) = 12 维 (raw)
- `video.mp4`: cam_front 视频
- `wrist_video.mp4`: cam_front 副本

### 5.2 转换为 LeRobot 格式

```bash
cd /media/jiayueru/Ckpt/WRL/openpi_tactile

# pick_flower_force_framestep5
$UV_BIN run python examples/libero/convert_framestep5_to_lerobot_force.py \
  --data-dir /media/jiayueru/Ckpt/WRL/data/WorldRL/pick_flower_framestep5 \
  --repo-name pick_flower_force_framestep5 \
  --output-dir /media/jiayueru/Ckpt/WRL/data/processed_lerobot/WorldRL/pick_flower_force_framestep5 \
  --overwrite \
  --task "pick up the flower and put it in the vase" \
  --force-history-len 8

# wipe_whiteboard_force_framestep5
$UV_BIN run python examples/libero/convert_framestep5_to_lerobot_force.py \
  --data-dir /media/jiayueru/Ckpt/WRL/data/WorldRL/wipe_whiteboard_framestep5 \
  --repo-name wipe_whiteboard_force_framestep5 \
  --output-dir /media/jiayueru/Ckpt/WRL/data/processed_lerobot/WorldRL/wipe_whiteboard_force_framestep5 \
  --overwrite \
  --task "wipe the whiteboard" \
  --force-history-len 8
```

---

## 6. 注册训练配置 (config.py)

编辑 `src/openpi/training/config.py`，在 `_CONFIGS` 列表中添加配置。

**Force 训练配置模板** (以 pick_flower_force 为例):

```python
TrainConfig(
    name="pi05_pick_flower_force",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=30,
        discrete_state_input=False,
        use_force_condition=True,      # 启用力条件
        force_dim=12,                  # left(6) + right(6)
        force_history_len=8,           # 滑动窗口长度
        force_hidden_dim=256,          # ForceEncoder 隐藏层维度
        force_scale=1.0,               # force 注入权重
    ),
    data=LeRobotLiberoDataConfig(
        repo_id="pick_flower_force",   # 必须与 --repo-name 一致
        base_config=DataConfig(
            prompt_from_task=True,
        ),
        extra_delta_transform=False,
        action_dim=7,                  # pose(6) + gripper(1)
        include_force_history=True,    # 数据加载时包含 force_history
    ),
    batch_size=32,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=300,
        peak_lr=5e-5,
        decay_steps=1_000_000,
        decay_lr=5e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=0.999,
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/media/jiayueru/Ckpt/WRL/openpi/checkpoints/hairuoliu/pi05_base/params",
        extra_missing_regex=".*force_encoder.*",  # ForceEncoder 是新模块，允许缺失
    ),
    num_train_steps=30_000,
),
```

**已注册的 6 个配置名称**:

| config name | repo_id |
|---|---|
| `pi05_pick_flower_force` | `pick_flower_force` |
| `pi05_wipe_whiteboard_force` | `wipe_whiteboard_force` |
| `pi05_pick_flower_force_dupfront` | `pick_flower_force_dupfront` |
| `pi05_wipe_whiteboard_force_dupfront` | `wipe_whiteboard_force_dupfront` |
| `pi05_pick_flower_force_framestep5` | `pick_flower_force_framestep5` |
| `pi05_wipe_whiteboard_force_framestep5` | `wipe_whiteboard_force_framestep5` |

---

## 7. 计算归一化统计量

> **必须在训练前完成**。`compute_norm_stats.py` 已修改，会计算 `state`、`actions`、`force_history` 三个 key 的统计量。

```bash
cd /media/jiayueru/Ckpt/WRL/openpi_tactile

# ---- pick_flower_force ----
HF_LEROBOT_HOME="/media/jiayueru/Ckpt/WRL/data/processed_lerobot/WorldRL/pick_flower_force" \
$UV_BIN run python scripts/compute_norm_stats.py --config-name pi05_pick_flower_force

# ---- wipe_whiteboard_force ----
HF_LEROBOT_HOME="/media/jiayueru/Ckpt/WRL/data/processed_lerobot/WorldRL/wipe_whiteboard_force" \
$UV_BIN run python scripts/compute_norm_stats.py --config-name pi05_wipe_whiteboard_force

# ---- pick_flower_force_dupfront ----
HF_LEROBOT_HOME="/media/jiayueru/Ckpt/WRL/data/processed_lerobot/WorldRL/pick_flower_force_dupfront" \
$UV_BIN run python scripts/compute_norm_stats.py --config-name pi05_pick_flower_force_dupfront

# ---- wipe_whiteboard_force_dupfront ----
HF_LEROBOT_HOME="/media/jiayueru/Ckpt/WRL/data/processed_lerobot/WorldRL/wipe_whiteboard_force_dupfront" \
$UV_BIN run python scripts/compute_norm_stats.py --config-name pi05_wipe_whiteboard_force_dupfront

# ---- pick_flower_force_framestep5 ----
HF_LEROBOT_HOME="/media/jiayueru/Ckpt/WRL/data/processed_lerobot/WorldRL/pick_flower_force_framestep5" \
$UV_BIN run python scripts/compute_norm_stats.py --config-name pi05_pick_flower_force_framestep5

# ---- wipe_whiteboard_force_framestep5 ----
HF_LEROBOT_HOME="/media/jiayueru/Ckpt/WRL/data/processed_lerobot/WorldRL/wipe_whiteboard_force_framestep5" \
$UV_BIN run python scripts/compute_norm_stats.py --config-name pi05_wipe_whiteboard_force_framestep5
```

**验证**:

```bash
python3 -c "
import json
for name in [
    'pi05_pick_flower_force/pick_flower_force',
    'pi05_wipe_whiteboard_force/wipe_whiteboard_force',
    'pi05_pick_flower_force_dupfront/pick_flower_force_dupfront',
    'pi05_wipe_whiteboard_force_dupfront/wipe_whiteboard_force_dupfront',
    'pi05_pick_flower_force_framestep5/pick_flower_force_framestep5',
    'pi05_wipe_whiteboard_force_framestep5/wipe_whiteboard_force_framestep5',
]:
    path = f'assets/{name}/norm_stats.json'
    d = json.load(open(path))
    keys = list(d['norm_stats'].keys())
    print(f'{name}: {keys}')
"
# 每行都应输出: ['state', 'actions', 'force_history']
```

**norm_stats 输出路径规则**: `assets/<config_name>/<repo_id>/norm_stats.json`

---

## 8. 启动训练

### 8.1 使用一键训练脚本 (推荐，适用于远程训练节点)

已有 6 个训练脚本 (位于项目根目录):

| 脚本 | 对应 config |
|---|---|
| `train_pick_flower_force.sh` | `pi05_pick_flower_force` |
| `train_wipe_whiteboard_force.sh` | `pi05_wipe_whiteboard_force` |
| `train_pick_flower_force_dupfront.sh` | `pi05_pick_flower_force_dupfront` |
| `train_wipe_whiteboard_force_dupfront.sh` | `pi05_wipe_whiteboard_force_dupfront` |
| `train_pick_flower_force_framestep5.sh` | `pi05_pick_flower_force_framestep5` |
| `train_wipe_whiteboard_force_framestep5.sh` | `pi05_wipe_whiteboard_force_framestep5` |

训练脚本包含 8 个阶段: conda env → system deps → ffmpeg7 → clash proxy → python/uv → venv/av → calc_workers → run_train。直接运行即可:

```bash
bash /media/jiayueru/Ckpt/WRL/openpi_tactile/train_pick_flower_force.sh
```

**训练脚本中的关键变量** (按需修改):

```bash
HF_LEROBOT_HOME="..."    # LeRobot 数据集所在目录
TASK_NAME="..."           # config.py 中的 name
EXP_NAME="run_20260414"   # 实验名称，checkpoint 保存子目录
OVERWRITE_FLAG="--overwrite"  # 改为 "--resume" 可断点续训
```

### 8.2 手动运行训练命令

```bash
cd /media/jiayueru/Ckpt/WRL/openpi_tactile

HF_LEROBOT_HOME="/media/jiayueru/Ckpt/WRL/data/processed_lerobot/WorldRL/pick_flower_force" \
WANDB_BASE_URL="https://api.bandw.top" \
WANDB_API_KEY="7d0049c9992505326ca78a42d89dcfefa2e3d51a" \
WANDB_ENTITY="liushb9-peking-university" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
$UV_BIN run scripts/train.py \
  pi05_pick_flower_force \
  --exp-name run_20260414 \
  --overwrite \
  --num-workers 16
```

**Checkpoint 保存目录**: `checkpoints_/<config_name>/<exp_name>/`

---

## 9. 代码修改记录

### 9.1 `scripts/compute_norm_stats.py` (line 103)

```python
# 修改前:
keys = ["state", "actions"]

# 修改后:
keys = ["state", "actions", "force_history"]
```

不加 `force_history` 的话，训练时 force_history 不会被归一化，ForceEncoder 输入数据分布不对。

### 9.2 `examples/libero/convert_own_data_to_lerobot_force.py`

相比原版 `convert_own_data_to_lerobot_abs.py` 的主要区别:
- 新增 `tactile/force_left` + `tactile/force_right` 读取
- 新增 `force_history` feature (滑动窗口 + 归一化)
- 新增 `--use-front-as-wrist` 参数 (dupfront 方案)
- action 使用绝对 pose(6) 而非 delta pose

### 9.3 `examples/libero/convert_framestep5_to_lerobot_force.py`

从 framestep5 格式 (video.mp4 + npy) 转换为 LeRobot 格式。与 HDF5 版本的区别:
- 图像从 MP4 视频读取 (OpenCV)，不是 HDF5 JPEG 字节
- actions/state/tactile 从 npy 文件读取，已是最终格式
- fps=15 (匹配抽帧脚本的输出 fps)

### 9.4 `sample_worldrl_pick_flower_every_n.py` (抽帧脚本)

相比原版的修改:
- `actions.npy`: 从 joint(7)+gripper(2)=9维 改为 pose(6)+gripper_norm(1)=7维
- 新增 `state.npy`: joint(7)+gripper_norm(1)=8维
- 新增 `wrist_video.mp4`: cam_front 的副本
- `metadata.csv` 新增 `wrist_video` 和 `state_sequence` 列

---

## 10. 路径速查表

### 源数据

| 路径 | 说明 |
|---|---|
| `/media/jiayueru/Ckpt/WRL/data/WorldRL/pick_flower/` | pick_flower 原始 HDF5 (99 eps) |
| `/media/jiayueru/Ckpt/WRL/data/WorldRL/wipe_whiteboard/` | wipe_whiteboard 原始 HDF5 (108 eps) |
| `/media/jiayueru/Ckpt/WRL/data/WorldRL/pick_flower_framestep5/` | pick_flower 抽帧数据 |
| `/media/jiayueru/Ckpt/WRL/data/WorldRL/wipe_whiteboard_framestep5/` | wipe_whiteboard 抽帧数据 |

### LeRobot 数据集

均位于 `/media/jiayueru/Ckpt/WRL/data/processed_lerobot/WorldRL/` 下。  
注意: LeRobot 使用 `HF_LEROBOT_HOME / repo_id` 拼接路径，所以实际数据在 **双层嵌套** 目录中:

```
<dataset_name>/<dataset_name>/
├── data/chunk-000/episode_XXXXXX.parquet
├── images/
│   ├── image/episode_XXXXXX/frame_XXXXXX.png
│   └── wrist_image/episode_XXXXXX/frame_XXXXXX.png
└── meta/
    ├── info.json
    ├── tasks.jsonl
    ├── episodes.jsonl
    └── episodes_stats.jsonl
```

**HF_LEROBOT_HOME 设置规则**: 指向外层目录 (不含 repo_id)。

| config name | HF_LEROBOT_HOME |
|---|---|
| `pi05_pick_flower_force` | `.../WorldRL/pick_flower_force` |
| `pi05_wipe_whiteboard_force` | `.../WorldRL/wipe_whiteboard_force` |
| `pi05_pick_flower_force_dupfront` | `.../WorldRL/pick_flower_force_dupfront` |
| `pi05_wipe_whiteboard_force_dupfront` | `.../WorldRL/wipe_whiteboard_force_dupfront` |
| `pi05_pick_flower_force_framestep5` | `.../WorldRL/pick_flower_force_framestep5` |
| `pi05_wipe_whiteboard_force_framestep5` | `.../WorldRL/wipe_whiteboard_force_framestep5` |

### 预训练权重

```
/media/jiayueru/Ckpt/WRL/openpi/checkpoints/hairuoliu/pi05_base/params
```

### 脚本位置

| 文件 | 说明 |
|---|---|
| `examples/libero/convert_own_data_to_lerobot_force.py` | HDF5 → LeRobot 转换 |
| `examples/libero/convert_framestep5_to_lerobot_force.py` | framestep5 → LeRobot 转换 |
| `scripts/compute_norm_stats.py` | 计算归一化统计量 |
| `scripts/train.py` | 训练入口 |
| `src/openpi/training/config.py` | 训练配置注册 |

---

## 11. 常见问题

### Q1: `FileNotFoundError: episodes.jsonl`

数据集不完整，需要重新跑转换脚本 (加 `--overwrite`)。

### Q2: `401 Unauthorized` / `RepositoryNotFoundError`

数据集本地加载失败后回退到 HuggingFace Hub。检查:
1. `HF_LEROBOT_HOME` 是否正确设置
2. 数据集目录结构是否完整 (特别是 `episodes.jsonl`)

### Q3: `ImportError: CXXABI_1.3.15`

缺少正确的 `libstdc++`。确保设置:
```bash
export LD_LIBRARY_PATH="$CONDA_ROOT/envs/openpi311/lib:${LD_LIBRARY_PATH:-}"
```

### Q4: norm_stats 缺少 force_history

`scripts/compute_norm_stats.py` 的 `keys` 列表没有包含 `"force_history"`。确认 line 103 为:
```python
keys = ["state", "actions", "force_history"]
```

### Q5: 磁盘空间不足

HuggingFace datasets 缓存默认在 `/root/.cache/huggingface/datasets/`，可能占用大量空间。解决:
```bash
# 清理缓存
rm -rf /root/.cache/huggingface/datasets/*

# 重定向缓存到大磁盘
export HF_HOME="/media/jiayueru/Ckpt/.cache/huggingface"
```
