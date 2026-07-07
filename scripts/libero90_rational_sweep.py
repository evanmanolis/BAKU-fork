#!/usr/bin/env python3
"""Train and evaluate the LIBERO-90 Rational/MatrixPolicy comparison grid."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BAKU_DIR = REPO_ROOT / "baku"

DEFAULT_STEPS = tuple(range(5000, 100001, 5000))

VARIANTS = {
    "relu_adamw": {
        "policy_activation": "relu",
        "actor_optimizer": "adamw",
    },
    "rlb_adamw": {
        "policy_activation": "rlb_fused_global_rational",
        "actor_optimizer": "adamw",
    },
    "rlb_matrix_policy": {
        "policy_activation": "rlb_fused_global_rational",
        "actor_optimizer": "rational_matrix_policy",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run BAKU LIBERO-90 comparison jobs for ReLU+AdamW, RLB+AdamW, "
            "and RLB+MatrixPolicy."
        )
    )
    parser.add_argument(
        "mode",
        choices=("train", "eval", "all"),
        help="Run training, checkpoint evaluation, or both.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable inside the BAKU/LIBERO environment.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "exp_local" / "libero90_rational_sweep",
        help="Directory for train/eval outputs.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANTS),
        default=sorted(VARIANTS),
        help="Subset of variants to run.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=list(DEFAULT_STEPS),
        help="Checkpoint steps to evaluate. Defaults to 5000..100000 by 5000.",
    )
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=100001,
        help="Training loop upper bound. 100001 is required to save 100000.pt.",
    )
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=5000,
        help="Snapshot cadence.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Evaluation episodes per LIBERO-90 task.",
    )
    parser.add_argument(
        "--text-only-eval",
        action="store_true",
        help=(
            "Evaluate prompt=text checkpoints without loading converted expert demos. "
            "Task embeddings are generated from LIBERO language annotations."
        ),
    )
    parser.add_argument(
        "--text-only-max-state-dim",
        type=int,
        default=123,
        help="Feature padding width to use with --text-only-eval for LIBERO-90.",
    )
    parser.add_argument(
        "--num-demos-per-task",
        type=int,
        default=50,
        help="Training demos per LIBERO-90 task.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def hydra_overrides(args, variant, run_dir):
    settings = VARIANTS[variant]
    return [
        "agent=baku",
        "suite=libero",
        "dataloader=libero",
        "suite/task=libero_90",
        f"root_dir={REPO_ROOT}",
        "suite.hidden_dim=256",
        f"suite.num_train_steps={args.num_train_steps}",
        f"suite.save_every_steps={args.save_every_steps}",
        f"suite.num_eval_episodes={args.eval_episodes}",
        f"num_demos_per_task={args.num_demos_per_task}",
        "save_video=false",
        "use_tb=true",
        "policy_type=gpt",
        "policy_head=deterministic",
        "use_proprio=true",
        "use_language=true",
        "prompt=text",
        "temporal_agg=true",
        "num_queries=10",
        f"policy_activation={settings['policy_activation']}",
        f"actor_optimizer={settings['actor_optimizer']}",
        f"experiment_label={variant}",
        f"hydra.run.dir={run_dir}",
    ]


def run_command(cmd, cwd, dry_run):
    print("\n" + " ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def train_variant(args, variant):
    run_dir = args.output_root / "train" / variant
    cmd = [
        args.python,
        str(BAKU_DIR / "train.py"),
        *hydra_overrides(args, variant, run_dir),
        "eval=false",
    ]
    run_command(cmd, cwd=BAKU_DIR, dry_run=args.dry_run)


def eval_checkpoint(args, variant, step):
    train_dir = args.output_root / "train" / variant
    snapshot = train_dir / "snapshot" / f"{step}.pt"
    eval_dir = args.output_root / "eval" / variant / f"step_{step:07d}"
    if not args.dry_run and not snapshot.exists():
        raise FileNotFoundError(f"missing checkpoint: {snapshot}")

    cmd = [
        args.python,
        str(BAKU_DIR / "eval.py"),
        *hydra_overrides(args, variant, eval_dir),
        "eval=true",
        f"bc_weight={snapshot}",
    ]
    if args.text_only_eval:
        cmd.append("text_only_eval=true")
        cmd.append(f"text_only_max_state_dim={args.text_only_max_state_dim}")
    run_command(cmd, cwd=BAKU_DIR, dry_run=args.dry_run)


def collect_results(args):
    rows = []
    for variant in args.variants:
        for step in args.steps:
            eval_csv = args.output_root / "eval" / variant / f"step_{step:07d}" / "eval.csv"
            if not eval_csv.exists():
                continue
            with eval_csv.open() as handle:
                records = list(csv.DictReader(handle))
            if not records:
                continue
            row = records[-1]
            rows.append(
                {
                    "variant": variant,
                    "step": step,
                    "success": row.get("success", ""),
                    "episode_reward": row.get("episode_reward", ""),
                    "eval_csv": str(eval_csv),
                }
            )
    if not rows:
        return
    summary = args.output_root / "summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("variant", "step", "success", "episode_reward", "eval_csv"),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {summary}")


def main():
    args = parse_args()
    args.output_root = args.output_root.resolve()
    if args.mode in {"train", "all"}:
        for variant in args.variants:
            train_variant(args, variant)
    if args.mode in {"eval", "all"}:
        for variant in args.variants:
            for step in args.steps:
                eval_checkpoint(args, variant, step)
        if not args.dry_run:
            collect_results(args)


if __name__ == "__main__":
    main()
