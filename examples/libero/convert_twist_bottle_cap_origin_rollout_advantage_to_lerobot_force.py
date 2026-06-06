"""Convert twist_bottle_cap origin HDF5 plus successful rollout HDF5 to LeRobot.

The origin data is sampled with stride=5. The rollout directory is already
produced at the rollout rate requested by the experiment, so it is imported
with stride=1. All frames receive the same scalar advantage label.
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

from examples.libero.convert_mixed_wipe_to_lerobot_force import add_hdf5_episodes


def main(
    original_hdf5_dir: str,
    rollout_hdf5_dir: str,
    repo_name: str = "twist_bottle_cap_origin_rollout1_force_stride5_posecont_dupfront_192x256_success_advantage",
    output_dir: str | None = None,
    overwrite: bool = False,
    task: str = "twist the bottle cap and pick up the bottle cap",
    force_history_len: int = 8,
    original_stride: int = 5,
    rollout_stride: int = 1,
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

    original_count = add_hdf5_episodes(
        dataset,
        Path(original_hdf5_dir),
        task=task,
        force_history_len=force_history_len,
        stride=original_stride,
        image_height=image_height,
        image_width=image_width,
        advantage_value=advantage_value,
    )
    rollout_count = add_hdf5_episodes(
        dataset,
        Path(rollout_hdf5_dir),
        task=task,
        force_history_len=force_history_len,
        stride=rollout_stride,
        image_height=image_height,
        image_width=image_width,
        advantage_value=advantage_value,
    )
    print(
        "Saved episodes: "
        f"original={original_count}, rollout={rollout_count}, "
        f"advantage={advantage_value}, original_stride={original_stride}, rollout_stride={rollout_stride}"
    )


if __name__ == "__main__":
    tyro.cli(main)
