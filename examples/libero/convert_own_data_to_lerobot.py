"""
将自定义数据转为 LeRobot 格式的示例脚本（已按 Franka HDF5 数据适配）。

HDF5 数据结构：
  observations/images/cam_front   : (T,) object  — JPEG 字节流
  observations/images/cam_high    : (T,) object  — JPEG 字节流（腕部/高位相机）
  puppet/joint    : (T, 7)  float32 — 关节角
  puppet/pose     : (T, 6)  float32 — 末端执行器位姿 [x,y,z,rx,ry,rz]
  puppet/gripper  : (T, 2)  float32 — [:,0] 实际宽度，[:,1] 二值状态(1=open,0=closed)

state (8D)   = joint(7) + gripper_binary(1)，其中 0=open，1=closed（Pi0 约定）
action (7D)  = delta_pose(6) + gripper_binary(1)
               delta_pose[i] = pose[min(i+N, T-1)] - pose[i]，N = action_interval

Usage:
  uv run examples/libero/convert_own_data_to_lerobot.py \\
      --data-dir /path/to/position1_20260331 \\
      --repo-name my_pickplace \\
      --output-dir /path/to/processed_data \\
      --overwrite \\
      --task "pick up the bread and put it in the plate"

  uv run examples/libero/convert_own_data_to_lerobot.py \\
      --data-dir /path/to/data \\
      --action-interval 3
"""

import io
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import tyro

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

try:
    import cv2
except ImportError:
    cv2 = None


def _decode_jpeg(buf: np.ndarray) -> np.ndarray:
    """将 HDF5 中存储的 JPEG 字节流解码为 (H, W, 3) uint8 RGB 数组。"""
    raw = bytes(buf.tobytes()) if isinstance(buf, np.ndarray) else bytes(buf)
    if cv2 is not None:
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    from PIL import Image
    return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))


def _resize_image(img: np.ndarray, target_hw=(256, 256)) -> np.ndarray:
    """将图像缩放到 target_hw = (H, W)，输出 (H, W, 3)。"""
    h, w = target_hw
    if img.shape[:2] == (h, w):
        return img
    if cv2 is not None:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    from PIL import Image
    return np.array(Image.fromarray(img).resize((w, h), Image.BILINEAR))


def main(
    data_dir: str,
    repo_name: str = "my_dataset",
    action_interval: int = 1,
    output_dir: str | None = None,
    overwrite: bool = False,
    push_to_hub: bool = False,
    task: str = "",
    image_size: int = 256,
):
    """将 Franka HDF5 演示数据批量转换为 LeRobot 格式。

    Args:
        data_dir: 包含 *.hdf5 文件的目录（也会递归搜索一层子目录）。
        repo_name: 输出数据集名称，默认 my_dataset。
        action_interval: delta action 帧间隔 N，默认 1（相邻帧差）。
        output_dir: 写入目录，不指定则写到当前目录。
        overwrite: 输出目录已存在时覆盖，默认 False。
        push_to_hub: 是否上传到 HuggingFace Hub，默认 False。
        task: 任务语言描述，写入每帧的 task 字段，默认空字符串。
        image_size: 输出图像边长（正方形），默认 256。
    """
    root = Path(output_dir) / repo_name if output_dir else None
    if root is not None and root.exists():
        if overwrite:
            shutil.rmtree(root)
        else:
            raise FileExistsError(
                f"输出目录已存在: {root}。请删除或指定 --overwrite 覆盖。"
            )

    hw = (image_size, image_size)
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=10,
        root=root,
        features={
            "image": {
                "dtype": "image",
                "shape": (image_size, image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (image_size, image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    action_interval = max(1, int(action_interval))

    # 收集所有 hdf5 文件（当前目录 + 一层子目录）
    data_path = Path(data_dir)
    hdf5_files = sorted(data_path.glob("*.hdf5")) + sorted(data_path.glob("*/*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"在 {data_dir} 下未找到任何 *.hdf5 文件。")
    print(f"共找到 {len(hdf5_files)} 个 HDF5 episode。")

    for ep_idx, hdf5_path in enumerate(hdf5_files):
        print(f"[{ep_idx + 1}/{len(hdf5_files)}] 处理: {hdf5_path.name}")
        with h5py.File(hdf5_path, "r") as f:
            joints   = f["puppet/joint"][:]         # (T, 7)
            poses    = f["puppet/pose"][:]           # (T, 6)
            grippers = f["puppet/gripper"][:]        # (T, 2)
            imgs_front = f["observations/images/cam_front"][:]   # (T,) object
            # cam_high 作为腕部相机；不存在时回退到 cam_front
            if "observations/images/cam_high" in f:
                imgs_wrist = f["observations/images/cam_high"][:]
            else:
                imgs_wrist = imgs_front

        T = len(joints)
        # gripper[:,1] 在实际 HDF5 数据中恒为 1.0（不可靠），改用实际宽度列 gripper[:,0]：
        #   ≈1.01（张开）→ 0.0 (open)，≈0.08（闭合）→ 1.0 (closed)，阈值 0.5
        gripper_state = np.where(grippers[:, 0] > 0.5, 0.0, 1.0).astype(np.float32)  # (T,)

        for i in range(T):
            state = np.concatenate([joints[i], [gripper_state[i]]]).astype(np.float32)

            # delta pose：当前帧与 i+N 帧之差，末尾帧用自身（delta=0）
            j = min(i + action_interval, T - 1)
            delta_pose = (poses[j] - poses[i]).astype(np.float32)
            actions = np.concatenate([delta_pose, [gripper_state[i]]]).astype(np.float32)

            image       = _resize_image(_decode_jpeg(imgs_front[i]), hw)
            wrist_image = _resize_image(_decode_jpeg(imgs_wrist[i]), hw)

            dataset.add_frame(
                {
                    "image": image,
                    "wrist_image": wrist_image,
                    "state": state,
                    "actions": actions,
                    "task": task,
                }
            )
        dataset.save_episode()
        print(f"  → 保存完成，共 {T} 帧。")

    print(f"\n转换完成！数据集写入: {root or '当前目录'}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
