"""Convert put_bread_twice original HDF5 plus synthetic rollout data to LeRobot.

Sources:
  - original_hdf5_dir: raw World-RL HDF5 episodes, sampled with stride=5.
  - synthetic_dir: episode dirs with video.mp4/actions.npy/tactile.npy.
  - synthetic_dir_2: optional second synthetic episode dir with the same layout.

All frames receive the same scalar advantage label, defaulting to 1.0.
"""

import shutil
from pathlib import Path
import sys

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
    add_episode,
    add_hdf5_episodes,
    add_video_episodes,
    normalize_force_shape,
    read_video_frames,
)


def _synthetic_episode_dirs(*data_dirs: str) -> list[Path]:
    episode_dirs: list[Path] = []
    for data_dir in data_dirs:
        if not data_dir:
            continue
        root = Path(data_dir)
        episode_dirs.extend(d for d in root.iterdir() if d.is_dir())
    return sorted(episode_dirs, key=lambda p: (str(p.parent), p.name))


def _sample_episode_dirs(ep_dirs: list[Path], sample_count: int, seed: int) -> list[Path]:
    if sample_count <= 0:
        return ep_dirs
    if sample_count > len(ep_dirs):
        raise ValueError(f"Requested {sample_count} synthetic episodes, but only found {len(ep_dirs)}")
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(len(ep_dirs), size=sample_count, replace=False)
    return [ep_dirs[i] for i in sorted(selected_indices)]


def _add_video_episode_dirs(
    dataset,
    ep_dirs: list[Path],
    *,
    task: str,
    force_history_len: int,
    image_height: int,
    image_width: int,
    advantage_value: float,
) -> int:
    count = 0
    for ep_dir in ep_dirs:
        print(f"[failure] Processing {ep_dir}")
        video_path = ep_dir / "video.mp4"
        actions_path = ep_dir / "actions.npy"
        force_path = ep_dir / "force.npy"
        if not force_path.exists():
            force_path = ep_dir / "tactile.npy"
        if not (video_path.exists() and actions_path.exists() and force_path.exists()):
            print("  [SKIP] missing video.mp4/actions.npy/force.npy or tactile.npy")
            continue
        try:
            frames = read_video_frames(video_path, image_height, image_width)
            actions = np.load(actions_path).astype(np.float32)
            force = normalize_force_shape(np.load(force_path))
            pose_gripper = np.concatenate([actions[:, :6], actions[:, 6:7]], axis=-1).astype(np.float32)
        except Exception as exc:
            print(f"  Load failed, skipping: {exc}")
            continue
        if add_episode(dataset, frames, pose_gripper, force, task, force_history_len, advantage_value):
            count += 1
    return count


def main(
    original_hdf5_dir: str,
    synthetic_dir: str,
    synthetic_dir_2: str = "",
    repo_name: str = "put_bread_twice_origin_synth_force_stride5_posecont_dupfront_192x256_success_advantage",
    output_dir: str | None = None,
    overwrite: bool = False,
    task: str = "pick up the bread slices one by one and place them into the toaster",
    force_history_len: int = 8,
    stride: int = 5,
    image_height: int = 192,
    image_width: int = 256,
    fps: int = 6,
    advantage_value: float = 1.0,
    synthetic_sample_count: int = 0,
    synthetic_sample_seed: int = 20260603,
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
    if synthetic_sample_count > 0:
        all_synthetic_dirs = _synthetic_episode_dirs(synthetic_dir, synthetic_dir_2)
        selected_synthetic_dirs = _sample_episode_dirs(
            all_synthetic_dirs,
            sample_count=synthetic_sample_count,
            seed=synthetic_sample_seed,
        )
        if root is not None:
            manifest_path = root / "synthetic_sample_manifest.txt"
            manifest_path.write_text(
                "\n".join(str(path) for path in selected_synthetic_dirs) + "\n",
                encoding="utf-8",
            )
            print(f"[sample] Wrote synthetic sample manifest: {manifest_path}")
        synthetic_count = _add_video_episode_dirs(
            dataset,
            selected_synthetic_dirs,
            task=task,
            force_history_len=force_history_len,
            image_height=image_height,
            image_width=image_width,
            advantage_value=advantage_value,
        )
        synthetic_count_2 = 0
    else:
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
            synthetic_count_2 = add_video_episodes(
                dataset,
                Path(synthetic_dir_2),
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
