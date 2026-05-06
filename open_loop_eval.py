"""Open-loop evaluation: feed training-set frames to a trained policy and
compare its predicted action chunks against the ground-truth action chunks
stored in the LeRobot dataset.

Usage example:
    python open_loop_eval.py \
        --config pi05_wipe_whiteboard_force_stride5_posecont_dupfront_192x256 \
        --ckpt /mnt/public2/liushengbang/openpi_tactile/checkpoints_/pi05_wipe_whiteboard_force_stride5_posecont_dupfront_192x256/wipe_whiteboard_euler_stride5_108_droidinit_h30_fscale10/20000 \
        --num-episodes 20 --frames-per-episode 8 --out open_loop_results
"""

import argparse
import io
import json
import os
import pathlib
import random
import time
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


def _to_array(value, dtype=np.float32) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = np.stack(arr)
    return arr.astype(dtype)


def _decode_image(value) -> np.ndarray:
    if isinstance(value, dict) and value.get("bytes") is not None:
        return np.asarray(Image.open(io.BytesIO(value["bytes"])).convert("RGB"))
    return np.asarray(value)


def _load_episode_meta(repo_root: pathlib.Path) -> tuple[list[dict], dict[int, str]]:
    with (repo_root / "meta" / "episodes.jsonl").open(encoding="utf-8") as f:
        episodes = [json.loads(line) for line in f]
    with (repo_root / "meta" / "tasks.jsonl").open(encoding="utf-8") as f:
        tasks = {int(item["task_index"]): item["task"] for item in map(json.loads, f)}
    return episodes, tasks


def _episode_path(repo_root: pathlib.Path, ep_idx: int) -> pathlib.Path:
    return repo_root / "data" / f"chunk-{ep_idx // 1000:03d}" / f"episode_{ep_idx:06d}.parquet"


def _build_obs(df: pd.DataFrame, frame_idx: int, prompt: str) -> dict[str, Any]:
    return {
        "observation/image": _decode_image(df["image"].iloc[frame_idx]),
        "observation/wrist_image": _decode_image(df["wrist_image"].iloc[frame_idx]),
        "observation/state": _to_array(df["state"].iloc[frame_idx]),
        "observation/force_history": _to_array(df["force_history"].iloc[frame_idx]),
        "prompt": prompt,
    }


def _gt_chunk(df: pd.DataFrame, frame_idx: int, horizon: int) -> np.ndarray:
    rows = [min(frame_idx + t, len(df) - 1) for t in range(horizon)]
    return np.stack([_to_array(df["actions"].iloc[i]) for i in rows]).astype(np.float32)


def _select_frames(length: int, num: int) -> list[int]:
    if num >= length:
        return list(range(length))
    if num == 1:
        return [0]
    # Evenly spaced, including start, excluding the very last (which would just
    # repeat the final action via padding).
    last = max(length - 1, 1)
    return [int(round(i * last / (num - 1))) for i in range(num)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="pi05_wipe_whiteboard_force_stride5_posecont_dupfront_192x256",
        help="Training config name registered in openpi.training.config",
    )
    parser.add_argument(
        "--ckpt",
        default=(
            "/mnt/public2/liushengbang/openpi_tactile/checkpoints_/"
            "pi05_wipe_whiteboard_force_stride5_posecont_dupfront_192x256/"
            "wipe_whiteboard_euler_stride5_108_droidinit_h30_fscale10/20000"
        ),
    )
    parser.add_argument(
        "--lerobot-root",
        default="/mnt/public2/liushengbang/Data/processed_lerobot/wan_worldrl_stride5_posecont_192x256",
        help="Root directory containing the LeRobot repo (HF_LEROBOT_HOME).",
    )
    parser.add_argument(
        "--repo-id",
        default="wipe_whiteboard_force_stride5_posecont_dupfront_192x256",
    )
    parser.add_argument(
        "--openpi-data-home",
        default="/mnt/public2/liushengbang/openpi_data/openpi-assets",
    )
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--frames-per-episode", type=int, default=8,
                        help="How many frames per episode to evaluate (evenly spaced).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value (single id).")
    parser.add_argument(
        "--out",
        default="/mnt/public2/liushengbang/openpi_tactile/open_loop_results",
    )
    parser.add_argument("--prompt", default=None, help="Override task prompt.")
    parser.add_argument("--episodes", type=str, default=None,
                        help="Comma-separated episode indices to use instead of random sampling.")
    args = parser.parse_args()

    # Env vars must be set before importing JAX/openpi.
    os.environ.setdefault("HF_LEROBOT_HOME", args.lerobot_root)
    os.environ.setdefault("OPENPI_DATA_HOME", args.openpi_data_home)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

    from openpi.policies import policy_config as _policy_config
    from openpi.training import checkpoints as _checkpoints
    from openpi.training import config as _config
    from openpi import transforms as _ot

    train_config = _config.get_config(args.config)
    horizon = train_config.model.action_horizon
    print(f"[info] config={args.config} action_horizon={horizon} action_dim(model)={train_config.model.action_dim}")

    repo_root = pathlib.Path(args.lerobot_root) / args.repo_id
    if not repo_root.exists():
        raise FileNotFoundError(f"LeRobot repo not found: {repo_root}")
    episodes_meta, tasks_meta = _load_episode_meta(repo_root)
    print(f"[info] dataset has {len(episodes_meta)} episodes")

    # Pick episodes.
    if args.episodes:
        chosen_indices = [int(x) for x in args.episodes.split(",") if x.strip()]
    else:
        rng = random.Random(args.seed)
        all_indices = [int(ep["episode_index"]) for ep in episodes_meta]
        chosen_indices = sorted(rng.sample(all_indices, min(args.num_episodes, len(all_indices))))
    print(f"[info] evaluating {len(chosen_indices)} episodes: {chosen_indices}")

    # Load policy.
    print(f"[info] loading policy from {args.ckpt} ...")
    t0 = time.time()
    policy = _policy_config.create_trained_policy(train_config, args.ckpt)
    print(f"[info] policy loaded in {time.time() - t0:.1f}s")

    # Patch the output transform: the upstream Unnormalize is strict and fails
    # because norm_stats contains "force_history" while the model only outputs
    # {state, actions}. Rebuild output transforms with norm_stats stripped of
    # force_history. Inputs path is unaffected (Normalize is non-strict).
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    raw_norm = _checkpoints.load_norm_stats(pathlib.Path(args.ckpt) / "assets", data_config.asset_id)
    stripped_norm = {k: v for k, v in raw_norm.items() if k != "force_history"}
    new_output = _ot.compose([
        *data_config.model_transforms.outputs,
        _ot.Unnormalize(stripped_norm, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
    ])
    policy._output_transform = new_output  # type: ignore[attr-defined]
    print(f"[info] patched Unnormalize keys -> {list(stripped_norm.keys())}")

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_episode = []
    all_pred = []
    all_gt = []
    all_ep = []
    all_frame = []

    for ep_idx in chosen_indices:
        path = _episode_path(repo_root, ep_idx)
        df = pd.read_parquet(path)
        ep_len = len(df)
        task_idx = int(df["task_index"].iloc[0])
        prompt = args.prompt or tasks_meta[task_idx]
        frame_indices = _select_frames(ep_len, args.frames_per_episode)
        print(f"[ep {ep_idx:03d}] len={ep_len} frames={frame_indices} prompt={prompt!r}")

        ep_pred, ep_gt = [], []
        for t_idx, frame in enumerate(frame_indices):
            obs = _build_obs(df, frame, prompt)
            t0 = time.time()
            result = policy.infer(obs)
            actions = np.asarray(result["actions"], dtype=np.float32)  # (H, action_dim)
            gt = _gt_chunk(df, frame, horizon)  # (H, 7)
            actions = actions[: gt.shape[0], : gt.shape[1]]
            l1 = float(np.mean(np.abs(actions - gt)))
            print(f"  frame {frame:03d} L1={l1:.4f} infer_ms={(time.time()-t0)*1000:.0f}")
            ep_pred.append(actions)
            ep_gt.append(gt)
            all_pred.append(actions)
            all_gt.append(gt)
            all_ep.append(ep_idx)
            all_frame.append(frame)

        ep_pred_arr = np.stack(ep_pred)  # (F, H, 7)
        ep_gt_arr = np.stack(ep_gt)
        ep_abs = np.abs(ep_pred_arr - ep_gt_arr)
        per_episode.append(
            {
                "episode_index": ep_idx,
                "length": ep_len,
                "frames": [int(f) for f in frame_indices],
                "prompt": prompt,
                "mean_l1": float(ep_abs.mean()),
                "per_dim_mean_l1": ep_abs.mean(axis=(0, 1)).tolist(),
                "per_horizon_mean_l1": ep_abs.mean(axis=(0, 2)).tolist(),
            }
        )

    pred_arr = np.stack(all_pred)  # (N, H, 7)
    gt_arr = np.stack(all_gt)
    abs_err = np.abs(pred_arr - gt_arr)

    summary = {
        "config": args.config,
        "ckpt": args.ckpt,
        "num_episodes": len(chosen_indices),
        "num_frames_total": pred_arr.shape[0],
        "horizon": int(pred_arr.shape[1]),
        "action_dim": int(pred_arr.shape[2]),
        "mean_l1_overall": float(abs_err.mean()),
        "median_l1_overall": float(np.median(abs_err)),
        "rmse_overall": float(np.sqrt(((pred_arr - gt_arr) ** 2).mean())),
        "per_dim_mean_l1": abs_err.mean(axis=(0, 1)).tolist(),
        "per_dim_rmse": np.sqrt(((pred_arr - gt_arr) ** 2).mean(axis=(0, 1))).tolist(),
        "per_horizon_mean_l1": abs_err.mean(axis=(0, 2)).tolist(),
        "first_step_mean_l1": float(abs_err[:, 0, :].mean()),
        "last_step_mean_l1": float(abs_err[:, -1, :].mean()),
    }

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "per_episode.json").write_text(json.dumps(per_episode, indent=2))
    np.savez_compressed(
        out_dir / "predictions.npz",
        pred=pred_arr,
        gt=gt_arr,
        episode_index=np.asarray(all_ep, dtype=np.int64),
        frame_index=np.asarray(all_frame, dtype=np.int64),
    )
    print(f"\n[done] results written to {out_dir}")


if __name__ == "__main__":
    main()
