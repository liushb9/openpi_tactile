"""
Convert pre-strided hanoi_tower data to LeRobot format.

Input format per episode directory:
  video.mp4       - front camera video, already sampled at fps=6
  actions.npy     - (T, 7) eef xyz + rpy + gripper
  tactile.npy     - (T, 12) left 6D wrench + right 6D wrench, raw

The dataset does not contain a separate wrist camera or state file, so this
converter duplicates the front image as wrist_image and uses actions.npy as
both state and action targets.
"""

import shutil
from pathlib import Path

import numpy as np
import tyro

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ModuleNotFoundError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

try:
    import cv2
except ImportError:
    cv2 = None


FORCE_NORM = np.array([20.0, 20.0, 20.0, 2.0, 2.0, 2.0], dtype=np.float32)


def build_force_history(tactile: np.ndarray, history_len: int) -> np.ndarray:
    tactile = tactile.astype(np.float32, copy=True)
    tactile[:, :6] /= FORCE_NORM
    tactile[:, 6:] /= FORCE_NORM

    pad = np.zeros((history_len - 1, 12), dtype=np.float32)
    padded = np.concatenate([pad, tactile], axis=0)
    return np.stack([padded[t : t + history_len] for t in range(len(tactile))], axis=0)


def read_video_frames(video_path: Path, target_height: int, target_width: int) -> np.ndarray:
    assert cv2 is not None, "Need opencv-python: pip install opencv-python"
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_bgr.shape[:2] != (target_height, target_width):
            frame_bgr = cv2.resize(frame_bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    return np.stack(frames, axis=0)


def main(
    data_dir: str,
    repo_name: str = "hanoi_tower_force_stride5_posecont_dupfront_192x256_success_advantage",
    output_dir: str | None = None,
    overwrite: bool = False,
    push_to_hub: bool = False,
    task: str = "pick up the rings one by one from the center stack and place them on the side pegs",
    force_history_len: int = 8,
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

    features = {
        "image": {
            "dtype": "image",
            "shape": (image_height, image_width, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": (image_height, image_width, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["actions"],
        },
        "force_history": {
            "dtype": "float32",
            "shape": (force_history_len, 12),
            "names": ["force_history"],
        },
        "advantage": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["advantage"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=fps,
        root=root,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    data_path = Path(data_dir)
    ep_dirs = sorted(
        [p for p in data_path.iterdir() if p.is_dir()],
        key=lambda p: int(p.name) if p.name.isdigit() else p.name,
    )
    if not ep_dirs:
        raise FileNotFoundError(f"No episode directories found in {data_dir}.")

    print(f"Found {len(ep_dirs)} episode directories.")
    print(f"Task: {task!r}")
    print(f"Force history len: {force_history_len}")
    print(f"FPS: {fps}")
    print(f"Image HxW: {image_height}x{image_width}")
    print(f"Advantage value: {advantage_value}")

    for ep_dir in ep_dirs:
        video_path = ep_dir / "video.mp4"
        actions_path = ep_dir / "actions.npy"
        tactile_path = ep_dir / "tactile.npy"
        missing = [p.name for p in (video_path, actions_path, tactile_path) if not p.exists()]
        if missing:
            print(f"  [SKIP] {ep_dir.name}: missing {missing}")
            continue

        print(f"Processing: {ep_dir.name}")
        try:
            frames = read_video_frames(video_path, image_height, image_width)
            actions = np.load(actions_path).astype(np.float32)
            tactile = np.load(tactile_path).astype(np.float32)
        except Exception as exc:
            print(f"  Load failed, skipping: {exc}")
            continue

        frame_count = min(len(frames), len(actions), len(tactile))
        if frame_count < 2:
            print(f"  [SKIP] {ep_dir.name}: too few frames ({frame_count})")
            continue
        if len(frames) != len(actions) or len(actions) != len(tactile):
            print(
                f"  [WARN] length mismatch video/actions/tactile="
                f"{len(frames)}/{len(actions)}/{len(tactile)}; using {frame_count}"
            )

        force_hist = build_force_history(tactile[:frame_count], force_history_len)
        for i in range(frame_count):
            state = actions[i].astype(np.float32)
            action = actions[i].astype(np.float32)
            dataset.add_frame(
                {
                    "image": frames[i],
                    "wrist_image": frames[i],
                    "state": state,
                    "actions": action,
                    "force_history": force_hist[i],
                    "advantage": np.array([advantage_value], dtype=np.float32),
                    "task": task,
                }
            )

        dataset.save_episode()
        print(f"  Saved {frame_count} frames.")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
