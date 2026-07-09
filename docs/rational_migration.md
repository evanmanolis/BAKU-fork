# Rational/MatrixPolicy LIBERO-90 Migration

This repo now has a first BAKU-native port of the RationalOPT comparison path:

- `policy_activation=relu` for the ReLU + AdamW baseline.
- `policy_activation=rlb_fused_global_rational` for RLB activations.
- `actor_optimizer=rational_matrix_policy` for AdamW-only vanilla MatrixPolicy on RLB `W_in` and `W_out`.

The default BAKU config remains `policy_activation=gelu` and `actor_optimizer=adamw`.

## Lambda Smoke Test

Run this before any long LIBERO jobs:

```bash
cd /path/to/BAKU
python scripts/check_rational_integration.py
```

This verifies:

- ReLU GPT forward pass.
- RLB GPT forward/backward.
- RLB matrix-group collection.
- One MatrixPolicy optimizer step.

For LIBERO rendering on Lambda, the smoke-tested stack used:

- `libnvidia-gl-580-server` so GLVND exposes NVIDIA EGL.
- Ubuntu user membership in `render` and `video`.
- `mujoco==2.3.7` with `robosuite==1.4.0`.
- `diffusers==0.11.1`, matching this repo's original environment pin.
- `einops==0.7.0`, required by the imported VQ-BET utilities.

## LIBERO-90 Comparison Grid

Convert the raw LIBERO HDF5 demonstrations into the BAKU pickle format first:

```bash
cd /path/to/BAKU
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python baku/data_generation/generate_libero.py \
  --dataset-path /path/to/LIBERO/libero/datasets \
  --save-path /path/to/BAKU/expert_demos/libero \
  --benchmarks libero_90
```

For a smoke conversion, add `--max-tasks-per-benchmark 1` and write to a
temporary `--save-path`.

The sweep script uses LIBERO-90 explicitly and saves checkpoints at every 5k step
through 100k:

```bash
cd /path/to/BAKU
python scripts/libero90_rational_sweep.py train \
  --python /path/to/env/bin/python
```

It launches:

- `relu_adamw`
- `rlb_adamw`
- `rlb_matrix_policy`

Training outputs go to:

```text
exp_local/libero90_rational_sweep/train/{variant}/snapshot/{step}.pt
```

Evaluate the checkpoint grid:

```bash
python scripts/libero90_rational_sweep.py eval \
  --python /path/to/env/bin/python
```

Evaluation outputs go to:

```text
exp_local/libero90_rational_sweep/eval/{variant}/step_0005000/eval.csv
```

The script also writes:

```text
exp_local/libero90_rational_sweep/summary.csv
```

## Single Command

```bash
python scripts/libero90_rational_sweep.py all \
  --python /path/to/env/bin/python
```

## Notes

- `suite.num_train_steps=100001` is intentional. BAKU trains while
  `global_step < num_train_steps`; using `100001` lets it save `100000.pt`.
- MatrixPolicy currently disables Muon by default for compatibility with BAKU's
  pinned environment. If Lambda uses a PyTorch build with `torch.optim.Muon`,
  Muon can be enabled through `matrix_policy.*` overrides.
- **Because Muon is disabled, `rlb_matrix_policy` is "MatrixPolicy (AdamW-only)"
  — a reduced variant of the reference repo's headline result, which uses an
  early Muon phase before switching to the AdamW matrix policy. Label it
  accordingly in writeups.**
- Eval requires matching model architecture overrides. The sweep script passes
  the same `policy_activation` and `actor_optimizer` settings for train/eval.
- Unlike the reference LM harness (warmup + cosine decay to `min_lr`), BAKU
  trains the actor at a constant `lr=1e-4` with no scheduler, so
  `adam_lr_scale=3.0` holds the RLB matrices at ~3e-4 for the entire run.
  This regime mismatch is the leading hypothesis for the 100k-step eval
  regression observed for `rlb_matrix_policy` (train loss still improving,
  rollout success dropping after the 75k peak in both eval seeds).
- `matrix_policy.adam_lr_scale_final`, `matrix_policy.adam_decay_start`, and
  `matrix_policy.adam_decay_end` are now plumbed through `BCAgent` to test
  that hypothesis. Two prebuilt ablation variants exist in the sweep script:
  - `rlb_matrix_policy_decay`: anneals the multiplier 3x -> 1x over progress
    0.60-0.95 (steps 60k-95k of 100k), mimicking the reference cosine tail.
  - `rlb_matrix_policy_flat`: `adam_lr_scale=1.0`, keeping role/group shaping
    but removing the LR boost, to separate magnitude from shaping.
- The sweep script now records `seed=<N>` explicitly in hydra overrides
  (`--seed`, default 2) so run provenance is auditable from `.hydra/`.
- Training checkpoints now store `max_state_dim`; `eval.py` uses
  `max(payload value, text_only_max_state_dim)` so a stale stored width can
  never shrink the live-sim feature buffer (LIBERO-90 sim state is 123-dim;
  the policy ignores this buffer in pixels mode).
- BAKU training wraps the encoder and actor forward passes in bf16 autocast
  (all variants uniformly). Note this when comparing against stock fp32 BAKU
  or external baselines.

## Follow-up eval protocol (100k regression investigation)

Dense multi-episode eval of existing checkpoints, with `rlb_adamw` as the
control arm (~55 min per checkpoint per episode-count=1 on an A10; scale
accordingly):

```bash
python scripts/libero90_rational_sweep.py eval \
  --variants rlb_adamw rlb_matrix_policy \
  --steps 75000 85000 95000 100000 \
  --eval-episodes 5 --text-only-eval
```

Ablation training runs (each ~2h on an A10; eval dominates cost):

```bash
python scripts/libero90_rational_sweep.py train \
  --variants rlb_matrix_policy_decay rlb_matrix_policy_flat
```

Analyze per-task: eval.csv has one `success_env{i}` column per LIBERO-90 task,
so compare 75k vs 100k within a variant (and variants against each other)
as paired per-task outcomes, not just the aggregate mean.
