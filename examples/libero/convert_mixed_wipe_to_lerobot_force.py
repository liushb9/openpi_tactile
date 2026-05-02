"""Convert mixed wipe data to LeRobot format.

Output convention:
  state   = pose(6) + continuous_gripper(1)
  actions = pose(6) + continuous_gripper(1)
  pose(6) = xyz(3) + Euler/RPY(3), matching the raw HDF5 puppet/pose convention.

Sources:
  - original_hdf5_dir: raw HDF5 episodes, using puppet/pose and puppet/gripper[:, 0].
  - success_dir: optional video/action/force episode dirs whose actions are joint-space;
    pose/gripper are read from the matching raw HDF5 episode id.
  - failure_dir: video/action/force episode dirs from EefIDM; actions[:, :6] must
    use the same xyz + Euler/RPY convention and actions[:, 6] is used as the continuous gripper.
"""

import shutil
from pathlib import Path

import h5py
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


def _resize_with_pad(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    if image.shape[:2] == (target_height, target_width):
        return image
    cur_height, cur_width = image.shape[:2]
    ratio = max(cur_width / target_width, cur_height / target_height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
    pad_top = (target_height - resized_height) // 2
    pad_left = (target_width - resized_width) // 2
    padded[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized
    return padded


def _decode_and_resize(img_bytes: bytes, target_height: int, target_width: int) -> np.ndarray:
    assert cv2 is not None, "Need opencv-python"
    arr = np.frombuffer(bytes(img_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("cv2.imdecode failed")
    img_bgr = _resize_with_pad(img_bgr, target_height, target_width)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def read_video_frames(video_path: Path, target_height: int, target_width: int) -> np.ndarray:
    assert cv2 is not None, "Need opencv-python"
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_bgr = _resize_with_pad(frame_bgr, target_height, target_width)
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from {video_path}")
    return np.stack(frames, axis=0)


def build_force_history(force: np.ndarray, history_len: int) -> np.ndarray:
    normalized = force.astype(np.float32).copy()
    normalized[:, :6] /= FORCE_NORM
    normalized[:, 6:] /= FORCE_NORM
    pad = np.zeros((history_len - 1, 12), dtype=np.float32)
    padded = np.concatenate([pad, normalized], axis=0)
    return np.stack([padded[t : t + history_len] for t in range(len(normalized))], axis=0)


def hdf5_pose_gripper(hdf5_path: Path, stride: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(hdf5_path, "r") as f:
        poses = f["puppet/pose"][:].astype(np.float32)
        grippers = f["puppet/gripper"][:, 0].astype(np.float32)
    indices = np.arange(0, len(poses), stride, dtype=np.int64)
    return poses[indices], grippers[indices, None]


def add_episode(dataset, frames, pose_gripper: np.ndarray, force: np.ndarray, task: str, force_history_len: int) -> bool:
    t = min(len(frames), len(pose_gripper), len(force))
    if t < 2:
        return False
    force_hist = build_force_history(force[:t], force_history_len)
    for i in range(t):
        action = pose_gripper[i].astype(np.float32)
        dataset.add_frame(
            {
                "image": frames[i],
                "wrist_image": frames[i],
                "state": action,
                "actions": action,
                "force_history": force_hist[i],
                "task": task,
            }
        )
    dataset.save_episode()
    return True


def add_hdf5_episodes(
    dataset,
    original_hdf5_dir: Path,
    *,
    task: str,
    force_history_len: int,
    stride: int,
    image_height: int,
    image_width: int,
) -> int:
    count = 0
    hdf5_files = sorted(original_hdf5_dir.glob("*.hdf5"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    for hdf5_path in hdf5_files:
        print(f"[original] Processing {hdf5_path.name}")
        try:
            with h5py.File(hdf5_path, "r") as f:
                poses = f["puppet/pose"][:].astype(np.float32)
                grippers = f["puppet/gripper"][:, 0].astype(np.float32)
                front_imgs = f["observations/images/cam_front"][:]
                if "tactile/force_left" in f and "tactile/force_right" in f:
                    force = np.concatenate(
                        [f["tactile/force_left"][:].astype(np.float32), f["tactile/force_right"][:].astype(np.float32)],
                        axis=-1,
                    )
                else:
                    force = np.zeros((len(poses), 12), dtype=np.float32)
        except Exception as exc:
            print(f"  Load failed, skipping: {exc}")
            continue

        indices = np.arange(0, len(poses), stride, dtype=np.int64)
        if len(indices) < 2:
            print(f"  [SKIP] too few sampled frames: {len(indices)}")
            continue
        try:
            frames = np.stack([_decode_and_resize(front_imgs[i], image_height, image_width) for i in indices], axis=0)
        except Exception as exc:
            print(f"  Image decode failed, skipping: {exc}")
            continue
        pose_gripper = np.concatenate([poses[indices], grippers[indices, None]], axis=-1).astype(np.float32)
        if add_episode(dataset, frames, pose_gripper, force[indices], task, force_history_len):
            count += 1
    return count


def add_video_episodes(
    dataset,
    data_dir: Path,
    *,
    task: str,
    force_history_len: int,
    image_height: int,
    image_width: int,
    source: str,
    original_hdf5_dir: Path | None = None,
    stride: int = 5,
) -> int:
    count = 0
    ep_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()], key=lambda p: int(p.name) if p.name.isdigit() else p.name)
    for ep_dir in ep_dirs:
        print(f"[{source}] Processing {ep_dir.name}")
        video_path = ep_dir / "video.mp4"
        actions_path = ep_dir / "actions.npy"
        force_path = ep_dir / "force.npy"
        if not (video_path.exists() and actions_path.exists() and force_path.exists()):
            print("  [SKIP] missing video.mp4/actions.npy/force.npy")
            continue
        try:
            frames = read_video_frames(video_path, image_height, image_width)
            actions = np.load(actions_path).astype(np.float32)
            force = np.load(force_path).astype(np.float32)
            if source == "success":
                if original_hdf5_dir is None:
                    raise ValueError("success source requires original_hdf5_dir")
                hdf5_path = original_hdf5_dir / f"{ep_dir.name}.hdf5"
                poses, grippers = hdf5_pose_gripper(hdf5_path, stride)
                pose_gripper = np.concatenate([poses, grippers], axis=-1).astype(np.float32)
            elif source == "failure":
                pose_gripper = np.concatenate([actions[:, :6], actions[:, 6:7]], axis=-1).astype(np.float32)
            else:
                raise ValueError(f"Unknown source: {source}")
        except Exception as exc:
            print(f"  Load failed, skipping: {exc}")
            continue
        if add_episode(dataset, frames, pose_gripper, force, task, force_history_len):
            count += 1
    return count


def main(
    original_hdf5_dir: str,
    failure_dir: str,
    success_dir: str = "",
    repo_name: str = "wipe_whiteboard_mixed",
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
    success_count = 0
    if success_dir:
        success_count = add_video_episodes(
            dataset,
            Path(success_dir),
            task=task,
            force_history_len=force_history_len,
            image_height=image_height,
            image_width=image_width,
            source="success",
            original_hdf5_dir=original_path,
            stride=stride,
        )
    failure_count = add_video_episodes(
        dataset,
        Path(failure_dir),
        task=task,
        force_history_len=force_history_len,
        image_height=image_height,
        image_width=image_width,
        source="failure",
    )
    print(f"Saved episodes: original={original_count}, success={success_count}, failure={failure_count}")


if __name__ == "__main__":
    tyro.cli(main)
