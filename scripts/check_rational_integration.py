#!/usr/bin/env python3
"""Smoke-test BAKU's Rational/MatrixPolicy integration."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BAKU_DIR = REPO_ROOT / "baku"
if str(BAKU_DIR) not in sys.path:
    sys.path.insert(0, str(BAKU_DIR))


def collect_matrix_groups(model):
    groups = []
    blocks = model.transformer.h
    for layer_index, block in enumerate(blocks):
        mlp = block.mlp
        activation = getattr(mlp, "rlb_activation", None)
        if activation is None:
            continue
        setattr(activation, "_rlb_optimizer_track_stats", True)
        groups.append(
            {
                "module": activation,
                "in_weight": mlp.in_proj.weight,
                "out_weight": mlp.out_proj.weight,
                "numerator": activation.numerator,
                "denominator": activation.denominator,
                "groups": int(activation.groups),
                "hidden_dim": int(activation.hidden_dim),
                "layer_index": layer_index,
                "num_layers": len(blocks),
            }
        )
    return groups


def main():
    import torch
    from agent.networks.gpt import GPT, GPTConfig
    from agent.optim import RationalMatrixPolicyOptimizer

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    relu_model = GPT(
        GPTConfig(
            block_size=8,
            input_dim=16,
            output_dim=8,
            n_layer=2,
            n_head=2,
            n_embd=32,
            dropout=0.0,
            activation="relu",
        )
    ).to(device)
    relu_out = relu_model(torch.randn(4, 5, 16, device=device))
    assert relu_out.shape == (4, 5, 8), relu_out.shape

    rlb_model = GPT(
        GPTConfig(
            block_size=8,
            input_dim=16,
            output_dim=8,
            n_layer=2,
            n_head=2,
            n_embd=32,
            dropout=0.0,
            activation="rlb_fused_global_rational",
            rational_group_size=16,
            rational_max_groups=8,
        )
    ).to(device)
    x = torch.randn(4, 5, 16, device=device)
    out = rlb_model(x)
    assert out.shape == (4, 5, 8), out.shape
    loss = out.square().mean()
    loss.backward()

    rlb_groups = collect_matrix_groups(rlb_model)
    assert len(rlb_groups) == 2, len(rlb_groups)
    matrix_groups = []
    for selector_index, group in enumerate(rlb_groups):
        common = {
            "weight_decay": 1e-4,
            "layer_index": group["layer_index"],
            "num_layers": group["num_layers"],
            "selector_index": selector_index,
        }
        matrix_groups.append(
            {"params": [group["in_weight"]], "matrix_role": "in", **common}
        )
        matrix_groups.append(
            {"params": [group["out_weight"]], "matrix_role": "out", **common}
        )

    optimizer = RationalMatrixPolicyOptimizer(
        matrix_groups,
        lr=1e-4,
        weight_decay=1e-4,
        total_steps=10,
        selector_groups=rlb_groups,
        muon_strength=0.0,
        muon_lr_scale=0.0,
        max_muon=0.0,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    print(
        "PASS rational integration",
        {
            "device": device,
            "rlb_groups": len(rlb_groups),
            "torch": torch.__version__,
            "torch_optim_muon": hasattr(torch.optim, "Muon"),
        },
    )


if __name__ == "__main__":
    main()
