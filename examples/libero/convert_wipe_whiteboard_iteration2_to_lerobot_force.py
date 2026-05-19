"""Convert wipe_whiteboard iteration-2 data to LeRobot format.

This combines:
  - raw HDF5 wipe_whiteboard episodes sampled with stride=5
  - two synthetic failure rollout directories with video.mp4/actions.npy/force.npy

The data convention is inherited from convert_mixed_wipe_to_lerobot_force.py:
  state/actions = xyz + Euler/RPY + continuous_gripper
  force_history = normalized left/right 6D wrench history
"""

import shutil
from pathlib import Path
import sys

import tyro

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ModuleNotFoundError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.libero.convert_mixed_wipe_to_lerobot_force import add_hdf5_episodes, add_video_episodes


def main(
    original_hdf5_dir: str,
    failure_dir_1: str,
    failure_dir_2: str,
    repo_name: str = "wipe_whiteboard_iteration2",
    output_dir: str | None = None,
    overwrite: bool = False,
    task: str = "wipe the star mark on the whiteboard",
    force_history_len: int = 8,
    stride: int = 5,
    image_height: int = 192,
    image_width: int = 256,
    fps: int = 6,
):
    root = Path(output_dir) / repo_name if output_dir else None
    if root is not None and root.exists():
        if overwrite:
            shutil.rmtree(root)
        else:
            raise FileExistsError(f"Output directory exists: {root}. Use --overwrite to replace.")

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=fps,
        root=root,
        features={
            "image": {"dtype": "image", "shape": (image_height, image_width, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (image_height, image_width, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (7,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
            "force_history": {"dtype": "float32", "shape": (force_history_len, 12), "names": ["force_history"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    original_path = Path(original_hdf5_dir)
    original_count = add_hdf5_episodes(
        dataset,
        original_path,
        task=task,
        force_history_len=force_history_len,
        stride=stride,
        image_height=image_height,
        image_width=image_width,
    )
    failure1_count = add_video_episodes(
        dataset,
        Path(failure_dir_1),
        task=task,
        force_history_len=force_history_len,
        image_height=image_height,
        image_width=image_width,
        source="failure",
    )
    failure2_count = add_video_episodes(
        dataset,
        Path(failure_dir_2),
        task=task,
        force_history_len=force_history_len,
        image_height=image_height,
        image_width=image_width,
        source="failure",
    )
    print(f"Saved episodes: original={original_count}, failure1={failure1_count}, failure2={failure2_count}")


if __name__ == "__main__":
    tyro.cli(main)
