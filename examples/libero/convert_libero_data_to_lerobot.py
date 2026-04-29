"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
import tensorflow_datasets as tfds
import tyro

REPO_NAME = "your_hf_username/libero"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]  # For simplicity we will combine multiple Libero datasets into one training dataset


def _to_uint8_hwc(image: np.ndarray) -> np.ndarray:
    """Convert an image to uint8 HWC format before storing it in LeRobot."""
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image to be 3D, got shape {image.shape}")

    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))

    if np.issubdtype(image.dtype, np.floating):
        if np.max(image) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def _resize_with_pad(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """Match openpi resize_with_pad: keep aspect ratio, then pad with black."""
    image = _to_uint8_hwc(image)
    if image.shape[:2] == (target_height, target_width):
        return image

    cur_height, cur_width = image.shape[:2]
    ratio = max(cur_width / target_width, cur_height / target_height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized = Image.fromarray(image).resize((resized_width, resized_height), resample=Image.BILINEAR)
    padded = Image.new(resized.mode, (target_width, target_height), 0)
    pad_top = (target_height - resized_height) // 2
    pad_left = (target_width - resized_width) // 2
    padded.paste(resized, (pad_left, pad_top))
    return np.asarray(padded)


def main(data_dir: str, *, push_to_hub: bool = False, image_height: int = 224, image_width: int = 224):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`.
    # Images are resized with pad to match openpi preprocessing instead of being stretched.
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
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

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        for episode in raw_dataset:
            for step in episode["steps"].as_numpy_iterator():
                dataset.add_frame(
                    {
                        "image": _resize_with_pad(step["observation"]["image"], image_height, image_width),
                        "wrist_image": _resize_with_pad(step["observation"]["wrist_image"], image_height, image_width),
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                        "task": step["language_instruction"].decode(),
                    }
                )
            dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
