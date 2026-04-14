"""
将自定义数据转为 LeRobot 格式的示例脚本（已按 Franka 数据适配，HDF5 版本）。

数据约定（HDF5 结构）：
  puppet/joint     (T, 7)   关节角
  puppet/pose      (T, 6)   末端姿态
  puppet/gripper   (T, 2)   夹爪状态
  observations/images/cam_front  (T,) 压缩图像字节
  observations/images/cam_high   (T,) 压缩图像字节

state   = joint(7) + gripper[0](1) = 8 维
actions = pose(6)  + gripper[0](1) = 7 维

Usage:
  python convert_own_data_to_lerobot_abs.py \
      --data-dir /path/to/hdf5_folder \
      --repo-name my_dataset \
      --output-dir /path/to/output \
      --action-interval 1 \
      --overwrite \
      --task "pick up the bread and put it in the plate"
"""

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


def _decode_and_resize(img_bytes: bytes, target_shape=(256, 256, 3)) -> np.ndarray:
    """从压缩字节（JPEG/PNG）解码图像，缩放到 target_shape，输出 RGB uint8。"""
    assert cv2 is not None, "需要安装 opencv-python: pip install opencv-python"
    arr = np.frombuffer(bytes(img_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("cv2.imdecode 失败，图像字节可能损坏。")
    if img_bgr.shape[:2] != target_shape[:2]:
        img_bgr = cv2.resize(
            img_bgr, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR
        )
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def main(
    data_dir: str,
    repo_name: str = "my_dataset",
    action_interval: int = 1,
    output_dir: str | None = None,
    overwrite: bool = False,
    push_to_hub: bool = False,
    task: str = "",
):
    """
    data_dir       : 包含 *.hdf5 的目录（每个文件对应一条 episode）
    repo_name      : 数据集名称
    action_interval: 保留参数，当前版本使用绝对 pose 作为 action，暂不使用
    output_dir     : 输出根目录（None 则写入当前目录下的 repo_name 子目录）
    overwrite      : 若输出目录已存在则覆盖
    push_to_hub    : 是否推送到 Hugging Face Hub
    task           : 任务语言描述
    """
    root = Path(output_dir) / repo_name if output_dir else None
    if root is not None and root.exists():
        if overwrite:
            shutil.rmtree(root)
        else:
            raise FileExistsError(
                f"输出目录已存在: {root}。请删除或指定 --overwrite 覆盖。"
            )

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=10,
        root=root,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
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

    hdf5_files = sorted(Path(data_dir).glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"在 {data_dir} 下未找到任何 .hdf5 文件。")

    print(f"共找到 {len(hdf5_files)} 个 episode 文件。")
    print(f"任务描述: {task!r}")

    gripper_open_threshold = 0.6  # 大于此值视为"打开"(0.0)，否则视为"闭合"(1.0)

    for hdf5_path in hdf5_files:
        print(f"处理: {hdf5_path}")
        try:
            with h5py.File(hdf5_path, "r") as f:
                joints   = f["puppet/joint"][:]                          # (T, 7)
                poses    = f["puppet/pose"][:]                           # (T, 6)
                grippers = f["puppet/gripper"][:]                        # (T, 2)
                front_imgs = f["observations/images/cam_front"][:]       # (T,) bytes
                high_imgs  = f["observations/images/cam_high"][:]        # (T,) bytes
        except Exception as e:
            print(f"  加载失败，跳过: {e}")
            continue

        T = len(joints)
        frame_count = 0

        for i in range(T):
            joint = joints[i].astype(np.float32)      # (7,)
            pose  = poses[i].astype(np.float32)       # (6,)

            # 用夹爪第 0 维判断开合；根据实际数据范围可调整阈值
            raw_gripper = float(grippers[i][0])
            # print(raw_gripper)
            gripper_norm = 0.0 if raw_gripper > gripper_open_threshold else 1.0
            # print(gripper_norm,".")
            state   = np.concatenate([joint, [gripper_norm]]).astype(np.float32)   # (8,)
            actions = np.concatenate([pose,  [gripper_norm]]).astype(np.float32)   # (7,)
            print(actions)

            try:
                image       = _decode_and_resize(front_imgs[i], (256, 256, 3))
                wrist_image = _decode_and_resize(high_imgs[i],  (256, 256, 3))
            except Exception as e:
                print(f"  第 {i} 帧图像解码失败，跳过该帧: {e}")
                continue

            dataset.add_frame(
                {
                    "image":       image,
                    "wrist_image": wrist_image,
                    "state":       state,
                    "actions":     actions,
                    "task":        task,
                }
            )
            frame_count += 1

        if frame_count == 0:
            print(f"  跳过空 episode（无有效帧）: {hdf5_path}")
            continue

        dataset.save_episode()
        print(f"  已保存 {frame_count} 帧。")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)