"""
Convert framestep5 data (video.mp4 + npy files) to LeRobot format.

Input format (per episode directory):
  video.mp4         - front camera video
  wrist_video.mp4   - wrist camera video (cam_front duplicate)
  actions.npy       - (T, 7) pose(6) + gripper_norm(1)
  state.npy         - (T, 8) joint(7) + gripper_norm(1)
  tactile.npy       - (T, 12) force_left(6) + force_right(6), raw

Output LeRobot features:
  image             - (256, 256, 3) from video.mp4
  wrist_image       - (256, 256, 3) from wrist_video.mp4
  state             - (8,) from state.npy
  actions           - (7,) from actions.npy
  force_history     - (K, 12) sliding window of normalized tactile

Usage:
  python convert_framestep5_to_lerobot_force.py \
      --data-dir /path/to/framestep5_folder \
      --repo-name my_dataset \
      --output-dir /path/to/output \
      --overwrite \
      --task "pick up the flower" \
      --force-history-len 8
"""

import shutil
from pathlib import Path

import numpy as np
import tyro

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

try:
    import cv2
except ImportError:
    cv2 = None


# Normalization constants for 6D wrench [Fx, Fy, Fz, Tx, Ty, Tz].
FORCE_NORM = np.array([20.0, 20.0, 20.0, 2.0, 2.0, 2.0], dtype=np.float32)


def build_force_history(tactile: np.ndarray, history_len: int) -> np.ndarray:
    """Build sliding-window force history from concatenated tactile.

    Args:
        tactile: (T, 12) raw force_left(6) + force_right(6).
        history_len: number of past timesteps to include.

    Returns:
        (T, history_len, 12) float32 array, normalized.
    """
    T = tactile.shape[0]
    # Normalize: first 6 dims are left hand, last 6 are right hand.
    normalized = tactile.copy()
    normalized[:, :6] /= FORCE_NORM
    normalized[:, 6:] /= FORCE_NORM

    pad = np.zeros((history_len - 1, 12), dtype=np.float32)
    padded = np.concatenate([pad, normalized], axis=0)

    history = np.stack(
        [padded[t : t + history_len] for t in range(T)], axis=0
    )
    return history


def read_video_frames(video_path: Path, target_size=(256, 256)) -> np.ndarray:
    """Read all frames from an MP4 file, resize to target_size, return as RGB."""
    assert cv2 is not None, "Need opencv-python: pip install opencv-python"
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_bgr.shape[:2] != target_size:
            frame_bgr = cv2.resize(
                frame_bgr, (target_size[1], target_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames, axis=0)


def main(
    data_dir: str,
    repo_name: str = "my_dataset",
    output_dir: str | None = None,
    overwrite: bool = False,
    push_to_hub: bool = False,
    task: str = "",
    force_history_len: int = 8,
):
    """
    data_dir          : Directory with episode subdirectories (0/, 1/, ...)
    repo_name         : Dataset name
    output_dir        : Output root directory
    overwrite         : Overwrite existing output
    push_to_hub       : Push to Hugging Face Hub
    task              : Language task description
    force_history_len : Number of past timesteps for force history window
    """
    root = Path(output_dir) / repo_name if output_dir else None
    if root is not None and root.exists():
        if overwrite:
            shutil.rmtree(root)
        else:
            raise FileExistsError(
                f"Output directory exists: {root}. Use --overwrite to replace."
            )

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=15,
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
            "force_history": {
                "dtype": "float32",
                "shape": (force_history_len, 12),
                "names": ["force_history"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    data_path = Path(data_dir)
    ep_dirs = sorted(
        [d for d in data_path.iterdir() if d.is_dir()],
        key=lambda p: int(p.name) if p.name.isdigit() else p.name,
    )
    if not ep_dirs:
        raise FileNotFoundError(f"No episode directories found in {data_dir}.")

    print(f"Found {len(ep_dirs)} episode directories.")
    print(f"Task: {task!r}")
    print(f"Force history len: {force_history_len}")

    for ep_dir in ep_dirs:
        video_path = ep_dir / "video.mp4"
        wrist_video_path = ep_dir / "wrist_video.mp4"
        actions_path = ep_dir / "actions.npy"
        state_path = ep_dir / "state.npy"
        tactile_path = ep_dir / "tactile.npy"

        # Check required files exist.
        missing = [
            p for p in [video_path, wrist_video_path, actions_path, state_path, tactile_path]
            if not p.exists()
        ]
        if missing:
            print(f"  [SKIP] {ep_dir.name}: missing {[p.name for p in missing]}")
            continue

        print(f"Processing: {ep_dir.name}")
        try:
            front_frames = read_video_frames(video_path)
            wrist_frames = read_video_frames(wrist_video_path)
            actions = np.load(str(actions_path)).astype(np.float32)
            state = np.load(str(state_path)).astype(np.float32)
            tactile = np.load(str(tactile_path)).astype(np.float32)
        except Exception as e:
            print(f"  Load failed, skipping: {e}")
            continue

        T = min(len(front_frames), len(wrist_frames), len(actions), len(state), len(tactile))
        if T < 2:
            print(f"  [SKIP] {ep_dir.name}: too few frames ({T})")
            continue

        force_hist = build_force_history(tactile[:T], force_history_len)

        frame_count = 0
        for i in range(T):
            dataset.add_frame(
                {
                    "image": front_frames[i],
                    "wrist_image": wrist_frames[i],
                    "state": state[i],
                    "actions": actions[i],
                    "force_history": force_hist[i],
                    "task": task,
                }
            )
            frame_count += 1

        if frame_count == 0:
            print(f"  Skipping empty episode: {ep_dir.name}")
            continue

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
