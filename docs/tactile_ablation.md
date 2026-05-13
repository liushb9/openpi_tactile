# Tactile Ablation Training

This repo now supports four force-conditioned pi0.5 training modes from config:

| Mode | Config name | Behavior |
| --- | --- | --- |
| `baseline` | `pi05_pick_flower_force_baseline` | Original openpi_tactile behavior: `force_history` is encoded by `ForceEncoder` and added to the action expert adaRMS condition. |
| `ki` | `pi05_pick_flower_force_ki` | Baseline plus Knowledge Insulation. The tactile/action loss reads frozen VLM prefix KV features and the optimizer excludes VLM parameters. |
| `advantage` | `pi05_pick_flower_force_advantage` | Baseline plus scalar advantage/success conditioning and CFG-style advantage dropout. VLM gradients are unchanged from baseline. |
| `ki_advantage` | `pi05_pick_flower_force_ki_advantage` | KI plus advantage-conditioned CFG. Action expert, `ForceEncoder`, `AdvantageEncoder`, and learned null advantage embedding remain trainable. |

## Launch commands

```bash
uv run scripts/train.py --config-name pi05_pick_flower_force_baseline --exp-name <exp>
uv run scripts/train.py --config-name pi05_pick_flower_force_ki --exp-name <exp>
uv run scripts/train.py --config-name pi05_pick_flower_force_advantage --exp-name <exp>
uv run scripts/train.py --config-name pi05_pick_flower_force_ki_advantage --exp-name <exp>
```

The original `pi05_pick_flower_force_stride5_posecont_dupfront_192x256` config is left in place.

## Binary Advantage Fields

The model checks the batch in this order:

1. `advantage`
2. `return`
3. `reward`
4. `success`

For our experiments, use a binary label: `0.0` for failure / low-quality samples and `1.0` for success / preferred samples. Put it in `advantage` when possible; `success` is also supported directly. The value is not normalized by the data pipeline.

If none exists, the model uses `advantage = 1.0`, so old LeRobot datasets continue to run. The Libero-style repack transform keeps these fields only when present.

During training, `adv_dropout_prob` randomly replaces the advantage condition with a null condition. The default null condition is a learned embedding initialized to zero (`null_advantage_type="learned"`). During inference, set `cfg_scale != 1.0` in the model config to run conditional and unconditional velocity predictions and combine them as:

```text
pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
```

The combination happens at the flow-matching velocity prediction (`v_t`) before the sampler updates `x_t`.

## Verifying KI

At training startup, `print_trainable_params=True` logs both trainable and frozen parameter paths. In KI modes:

- `PaliGemma/img/...` should appear under `FROZEN`.
- `PaliGemma/llm/...` paths without `_1` should appear under `FROZEN`.
- Action expert paths such as `PaliGemma/llm/..._1...`, `action_in_proj`, `action_out_proj`, `time_mlp_*`, `force_encoder`, `advantage_encoder`, and `null_advantage` should appear under `TRAINABLE`.

This is paired with forward-level `jax.lax.stop_gradient` on the prefix KV cache used by the action expert.

## Mixing teleop and synthetic data

The current LeRobot loader trains from a single `repo_id`. It does not natively sample multiple LeRobot repos with weights. To mix original teleop and synthetic correction data, create one merged LeRobot dataset before training, then point `repo_id` at that merged dataset and recompute norm stats.

RLDS/DROID configs already have weighted multi-dataset support through `DataConfig.datasets`, but the pick_flower force configs use the LeRobot path.

## Smoke tests

```bash
uv run pytest src/openpi/models/tactile_ablation_test.py
```

These tests cover baseline force loss, missing advantage default, advantage dropout with `adv_dropout_prob=1.0`, and KI parameter filtering.
