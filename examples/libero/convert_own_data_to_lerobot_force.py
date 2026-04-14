"""
Convert custom HDF5 data (with force/torque) to LeRobot format.

Extends the base conversion script to include:
  - tactile/force_left  (T, 6)  left hand wrench
  - tactile/force_right (T, 6)  right hand wrench

Concatenated into force_history (K, 12) per frame using a sliding window
of the last K timesteps.  Force is normalized: Fx/Fy/Fz divided by 20,
Tx/Ty/Tz divided by 2.

Usage:
  python convert_own_data_to_lerobot_force.py \
      --data-dir /path/to/hdf5_folder \
      --repo-name my_dataset \
      --output-dir /path/to/output \
      --overwrite \
      --task "pick up the flower and put it in the vase" \
      --force-history-len 8
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


# Normalization constants for 6D wrench [Fx, Fy, Fz, Tx, Ty, Tz].
FORCE_NORM = np.array([20.0, 20.0, 20.0, 2.0, 2.0, 2.0], dtype=np.float32)


def _decode_and_resize(img_bytes: bytes, target_shape=(256, 256, 3)) -> np.ndarray:
    assert cv2 is not None, "Need opencv-python: pip install opencv-python"
    arr = np.frombuffer(bytes(img_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("cv2.imdecode failed.")
    if img_bgr.shape[:2] != target_shape[:2]:
        img_bgr = cv2.resize(
            img_bgr, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR
        )
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def normalize_wrench(wrench: np.ndarray) -> np.ndarray:
    """Normalize a (*, 6) wrench array."""
    return wrench / FORCE_NORM


def build_force_history(
    force_left: np.ndarray,
    force_right: np.ndarray,
    history_len: int,
) -> np.ndarray:
    """Build sliding-window force history for every timestep.

    Args:
        force_left:  (T, 6) left hand wrench, raw.
        force_right: (T, 6) right hand wrench, raw.
        history_len: number of past timesteps to include.

    Returns:
        (T, history_len, 12) float32 array. For t < history_len, earlier
        slots are zero-padded.
    """
    T = force_left.shape[0]
    # Normalize each hand separately, then concatenate.
    fl = normalize_wrench(force_left)   # (T, 6)
    fr = normalize_wrench(force_right)  # (T, 6)
    force_concat = np.concatenate([fl, fr], axis=-1)  # (T, 12)

    # Zero-pad at the beginning so indexing is simple.
    pad = np.zeros((history_len - 1, 12), dtype=np.float32)
    padded = np.concatenate([pad, force_concat], axis=0)  # (history_len - 1 + T, 12)

    # Build sliding window: for frame t, we want [t, t+1, ..., t+history_len-1] in padded.
    history = np.stack(
        [padded[t : t + history_len] for t in range(T)], axis=0
    )  # (T, history_len, 12)
    return history


def main(
    data_dir: str,
    repo_name: str = "my_dataset",
    action_interval: int = 1,
    output_dir: str | None = None,
    overwrite: bool = False,
    push_to_hub: bool = False,
    task: str = "",
    force_history_len: int = 8,
    use_front_as_wrist: bool = False,
):
    """
    data_dir          : Directory with *.hdf5 episode files
    repo_name         : Dataset name
    action_interval   : Reserved (absolute pose used as action)
    output_dir        : Output root directory
    overwrite         : Overwrite existing output
    push_to_hub       : Push to Hugging Face Hub
    task              : Language task description
    force_history_len : Number of past timesteps for force history window
    use_front_as_wrist: Use cam_front for both image and wrist_image (ignore cam_high)
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
            "force_history": {
                "dtype": "float32",
                "shape": (force_history_len, 12),
                "names": ["force_history"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    hdf5_files = sorted(Path(data_dir).glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found in {data_dir}.")

    print(f"Found {len(hdf5_files)} episode files.")
    print(f"Task: {task!r}")
    print(f"Force history len: {force_history_len}")

    gripper_open_threshold = 0.6

    for hdf5_path in hdf5_files:
        print(f"Processing: {hdf5_path}")
        try:
            with h5py.File(hdf5_path, "r") as f:
                joints     = f["puppet/joint"][:]                      # (T, 7)
                poses      = f["puppet/pose"][:]                       # (T, 6)
                grippers   = f["puppet/gripper"][:]                    # (T, 2)
                front_imgs = f["observations/images/cam_front"][:]     # (T,) bytes
                high_imgs  = front_imgs if use_front_as_wrist else f["observations/images/cam_high"][:]  # (T,) bytes

                # Force data.
                has_force = "tactile/force_left" in f and "tactile/force_right" in f
                if has_force:
                    force_left  = f["tactile/force_left"][:].astype(np.float32)   # (T, 6)
                    force_right = f["tactile/force_right"][:].astype(np.float32)  # (T, 6)
                else:
                    force_left  = np.zeros((len(joints), 6), dtype=np.float32)
                    force_right = np.zeros((len(joints), 6), dtype=np.float32)
        except Exception as e:
            print(f"  Load failed, skipping: {e}")
            continue

        T = len(joints)

        # Build force history for all frames at once.
        force_hist = build_force_history(force_left, force_right, force_history_len)  # (T, K, 12)

        frame_count = 0
        for i in range(T):
            joint = joints[i].astype(np.float32)
            pose  = poses[i].astype(np.float32)

            raw_gripper = float(grippers[i][0])
            gripper_norm = 0.0 if raw_gripper > gripper_open_threshold else 1.0
            state   = np.concatenate([joint, [gripper_norm]]).astype(np.float32)
            actions = np.concatenate([pose,  [gripper_norm]]).astype(np.float32)

            try:
                image       = _decode_and_resize(front_imgs[i], (256, 256, 3))
                wrist_image = _decode_and_resize(high_imgs[i],  (256, 256, 3))
            except Exception as e:
                print(f"  Frame {i} image decode failed, skipping: {e}")
                continue

            dataset.add_frame(
                {
                    "image":         image,
                    "wrist_image":   wrist_image,
                    "state":         state,
                    "actions":       actions,
                    "force_history": force_hist[i],  # (K, 12)
                    "task":          task,
                }
            )
            frame_count += 1

        if frame_count == 0:
            print(f"  Skipping empty episode: {hdf5_path}")
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
