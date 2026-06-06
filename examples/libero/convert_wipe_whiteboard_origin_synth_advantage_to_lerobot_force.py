"""Convert wipe_whiteboard original HDF5 plus synthetic rollout data to LeRobot.

Sources:
  - original_hdf5_dir: raw World-RL HDF5 episodes, sampled with stride=5.
  - synthetic_dir: episode dirs with video.mp4/actions.npy/tactile.npy.

All frames receive the same scalar advantage label, defaulting to 1.0.
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
    synthetic_dir: str,
    repo_name: str = "wipe_whiteboard_origin_synth0522_force_stride5_posecont_dupfront_192x256_success_advantage",
    output_dir: str | None = None,
    overwrite: bool = False,
    task: str = "wipe the star mark on the whiteboard",
    force_history_len: int = 8,
    stride: int = 5,
    image_height: int = 192,
    image_width: int = 256,
    fps: int = 6,
    advantage_value: float = 1.0,
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
            "advantage": {"dtype": "float32", "shape": (1,), "names": ["advantage"]},
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
        advantage_value=advantage_value,
    )
    synthetic_count = add_video_episodes(
        dataset,
        Path(synthetic_dir),
        task=task,
        force_history_len=force_history_len,
        image_height=image_height,
        image_width=image_width,
        source="failure",
        advantage_value=advantage_value,
    )
    print(f"Saved episodes: original={original_count}, synthetic={synthetic_count}, advantage={advantage_value}")


if __name__ == "__main__":
    tyro.cli(main)
