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
- Eval requires matching model architecture overrides. The sweep script passes
  the same `policy_activation` and `actor_optimizer` settings for train/eval.
