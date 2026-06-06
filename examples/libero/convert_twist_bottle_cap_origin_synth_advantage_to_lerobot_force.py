"""Convert twist_bottle_cap original HDF5 plus synthetic rollout data to LeRobot.

Sources:
  - original_hdf5_dir: raw World-RL HDF5 episodes, sampled with stride=5.
  - synthetic_dir: episode dirs with video.mp4/actions.npy/tactile.npy.
  - synthetic_dir_2: optional second synthetic dir. It can either contain the
    same episode-dir layout or flat *_action.npy files named like
    000000__8_s35__cam_front__f0_p000_action.npy. For flat action files,
    images and force are recovered from the matching original HDF5 episode.

All frames receive the same scalar advantage label, defaulting to 1.0.
"""

import re
import shutil
from pathlib import Path
import sys

import h5py
import numpy as np
import tyro

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ModuleNotFoundError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.libero.convert_mixed_wipe_to_lerobot_force import (
    _decode_and_resize,
    add_episode,
    add_hdf5_episodes,
    add_video_episodes,
)


ACTION_FILE_RE = re.compile(r"^\d+__(?P<episode>\d+)_s(?P<start>\d+)__.*_action\.npy$")


def add_flat_action_episodes(
    dataset,
    data_dir: Path,
    *,
    original_hdf5_dir: Path,
    task: str,
    force_history_len: int,
    image_height: int,
    image_width: int,
    advantage_value: float | None = None,
) -> int:
    count = 0
    action_files = sorted(data_dir.glob("*_action.npy"))
    for action_path in action_files:
        print(f"[flat_action] Processing {action_path.name}")
        match = ACTION_FILE_RE.match(action_path.name)
        if match is None:
            print("  [SKIP] filename does not match expected *_action.npy pattern")
            continue
        episode_idx = int(match.group("episode"))
        start_idx = int(match.group("start"))
        hdf5_path = original_hdf5_dir / f"{episode_idx}.hdf5"
        if not hdf5_path.exists():
            print(f"  [SKIP] matching HDF5 episode not found: {hdf5_path}")
            continue

        try:
            actions = np.load(action_path).astype(np.float32)
            if actions.ndim != 2 or actions.shape[1] < 7:
                raise ValueError(f"Expected action shape (T, >=7), got {actions.shape}")
            with h5py.File(hdf5_path, "r") as f:
                front_imgs = f["observations/images/cam_front"]
                if "tactile/force_left" in f and "tactile/force_right" in f:
                    force = np.concatenate(
                        [f["tactile/force_left"][:].astype(np.float32), f["tactile/force_right"][:].astype(np.float32)],
                        axis=-1,
                    )
                else:
                    force = np.zeros((len(front_imgs), 12), dtype=np.float32)

                end_idx = min(start_idx + len(actions), len(front_imgs), len(force))
                if end_idx - start_idx < 2:
                    print(f"  [SKIP] too few available frames from start={start_idx}")
                    continue
                indices = np.arange(start_idx, end_idx, dtype=np.int64)
                frames = np.stack([_decode_and_resize(front_imgs[i], image_height, image_width) for i in indices], axis=0)

            pose_gripper = np.concatenate([actions[: len(indices), :6], actions[: len(indices), 6:7]], axis=-1)
            if add_episode(
                dataset,
                frames,
                pose_gripper,
                force[indices],
                task,
                force_history_len,
                advantage_value,
            ):
                count += 1
        except Exception as exc:
            print(f"  Load failed, skipping: {exc}")
            continue
    return count


def main(
    original_hdf5_dir: str,
    synthetic_dir: str,
    synthetic_dir_2: str = "",
    repo_name: str = "twist_bottle_cap_origin_synth0523_force_stride5_posecont_dupfront_192x256_success_advantage",
    output_dir: str | None = None,
    overwrite: bool = False,
    task: str = "twist the bottle cap and pick up the bottle cap",
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

    original_count = add_hdf5_episodes(
        dataset,
        Path(original_hdf5_dir),
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
    synthetic_count_2 = 0
    if synthetic_dir_2:
        second_dir = Path(synthetic_dir_2)
        if any(second_dir.glob("*_action.npy")):
            synthetic_count_2 = add_flat_action_episodes(
                dataset,
                second_dir,
                original_hdf5_dir=Path(original_hdf5_dir),
                task=task,
                force_history_len=force_history_len,
                image_height=image_height,
                image_width=image_width,
                advantage_value=advantage_value,
            )
        else:
            synthetic_count_2 = add_video_episodes(
                dataset,
                second_dir,
                task=task,
                force_history_len=force_history_len,
                image_height=image_height,
                image_width=image_width,
                source="failure",
                advantage_value=advantage_value,
            )
    print(
        "Saved episodes: "
        f"original={original_count}, synthetic={synthetic_count}, "
        f"synthetic_2={synthetic_count_2}, advantage={advantage_value}"
    )


if __name__ == "__main__":
    tyro.cli(main)
