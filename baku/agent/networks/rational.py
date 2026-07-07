"""Rational Latent Basis modules adapted for BAKU.

This implements the `rlb_fused_global_rational` branch from RationalOPT in
pure PyTorch so BAKU does not need the optional CUDA extension at import time.
"""

import math

import torch
from torch import nn


_VERSION_A_INIT_TABLE = {
    ("relu", 5.0): (
        [
            0.05678566882939273,
            0.499999998601522,
            1.001174819993277,
            0.7162719746045968,
            0.20331357925790883,
            0.019549369530291263,
        ],
        [
            1e-08,
            1.432543927483962,
            1e-08,
            0.039098737789319785,
        ],
    ),
    ("gelu", 5.0): (
        [
            -0.01708892945210021,
            0.5385611231879953,
            0.5024356148637584,
            0.18681350949076436,
            0.03243657130432953,
            0.0021540413477319988,
        ],
        [
            0.19015085308074264,
            0.23125739259275593,
            0.0406782234638605,
            0.00041446716044879055,
        ],
    ),
    ("silu", 5.0): (
        [
            2.4915714620520875e-05,
            0.5000668663485823,
            0.2499441908995771,
            0.0526200910151016,
            0.005525765640115002,
            0.00024199321178747251,
        ],
        [
            0.00022554017910995938,
            0.10511420350691994,
            2.8656992394109812e-05,
            0.0004816962110562025,
        ],
    ),
}


def resolve_group_count(hidden_dim, group_size=256, max_groups=32):
    hidden_dim = int(hidden_dim)
    group_size = max(1, int(group_size))
    max_groups = max(1, int(max_groups))
    groups = max(1, min(max_groups, math.ceil(hidden_dim / group_size)))
    while hidden_dim % groups != 0 and groups > 1:
        groups -= 1
    return groups


def _load_init(init, fit_range):
    key = (str(init), float(fit_range))
    if key not in _VERSION_A_INIT_TABLE:
        allowed = ", ".join(sorted({name for name, _ in _VERSION_A_INIT_TABLE}))
        raise ValueError(f"unknown rational init {init!r}; expected one of {allowed}")
    return _VERSION_A_INIT_TABLE[key]


def _repeat_init(numerator, denominator, groups):
    numerator = torch.tensor(numerator, dtype=torch.float32).view(1, 6)
    denominator = torch.tensor(denominator, dtype=torch.float32).view(1, 4)
    return numerator.repeat(groups, 1), denominator.repeat(groups, 1)


def rational_version_a5_4(x, numerator, denominator):
    a = numerator.to(device=x.device, dtype=x.dtype)
    b = denominator.abs().to(device=x.device, dtype=x.dtype)
    x_abs = x.abs()
    x2 = x.square()
    x3 = x2 * x
    x4 = x2.square()
    x5 = x4 * x
    p = (
        a[..., 0]
        + a[..., 1] * x
        + a[..., 2] * x2
        + a[..., 3] * x3
        + a[..., 4] * x4
        + a[..., 5] * x5
    )
    q = 1.0 + b[..., 0] * x_abs + b[..., 1] * x2 + b[..., 2] * x_abs * x2 + b[..., 3] * x4
    return p / q.clamp_min(torch.finfo(x.dtype).tiny)


class RationalFusedGlobalA5_4(nn.Module):
    """Grouped Version A rational activation without local atoms."""

    def __init__(self, hidden_dim, groups=None, init="silu", fit_range=5.0, eps=1e-6):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.groups = resolve_group_count(hidden_dim) if groups is None else int(groups)
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.groups <= 0:
            raise ValueError("groups must be positive")
        if self.hidden_dim % self.groups != 0:
            raise ValueError("hidden_dim must be divisible by groups")

        numerator, denominator = _load_init(init, fit_range)
        numerator, denominator = _repeat_init(numerator, denominator, self.groups)
        self.numerator = nn.Parameter(numerator)
        self.denominator = nn.Parameter(denominator)
        self.init_name = str(init)
        self.fit_range = float(fit_range)
        self.eps = float(eps)

    @torch.no_grad()
    def _update_optimizer_stats(self, x):
        if not bool(getattr(self, "_rlb_optimizer_track_stats", False)):
            return

        flat = x.detach().reshape(-1, self.hidden_dim)
        max_samples = int(getattr(self, "_rlb_optimizer_stat_samples", 512))
        if max_samples > 0 and flat.size(0) > max_samples:
            index = torch.linspace(0, flat.size(0) - 1, max_samples, device=flat.device).long()
            flat = flat.index_select(0, index)

        grouped = flat.float().view(-1, self.groups, self.hidden_dim // self.groups)
        rms = torch.sqrt(grouped.square().mean(dim=-1, keepdim=True) + self.eps)
        t = grouped / rms
        abs_t = t.abs()
        t2 = t.square()
        t3 = t2 * t
        t4 = t2.square()
        t5 = t4 * t
        ax3 = abs_t * t2

        numerator = self.numerator.detach().float().view(1, self.groups, 1, 6)
        denominator = self.denominator.detach().float()
        denominator_abs = denominator.abs().view(1, self.groups, 1, 4)
        powers = torch.stack((torch.ones_like(t), t, t2, t3, t4, t5), dim=-1)
        q = (
            1.0
            + denominator_abs[..., 0] * abs_t
            + denominator_abs[..., 1] * t2
            + denominator_abs[..., 2] * ax3
            + denominator_abs[..., 3] * t4
        )
        poly = (numerator * powers).sum(dim=-1)
        dpoly = (
            numerator[..., 1]
            + 2.0 * numerator[..., 2] * t
            + 3.0 * numerator[..., 3] * t2
            + 4.0 * numerator[..., 4] * t3
            + 5.0 * numerator[..., 5] * t4
        )
        dq = (
            denominator_abs[..., 0] * torch.sign(t)
            + 2.0 * denominator_abs[..., 1] * t
            + 3.0 * denominator_abs[..., 2] * t * abs_t
            + 4.0 * denominator_abs[..., 3] * t3
        )
        output = poly / q.clamp_min(self.eps)
        derivative = (dpoly * q - poly * dq) / q.square().clamp_min(self.eps)
        self._rlb_optimizer_stats = {
            "output_rms": torch.sqrt(output.square().mean(dim=(0, 2)) + self.eps).detach(),
            "derivative_rms": torch.sqrt(derivative.square().mean(dim=(0, 2)) + self.eps).detach(),
        }

    def forward(self, x):
        if x.size(-1) != self.hidden_dim:
            raise ValueError(f"expected last dimension {self.hidden_dim}, got {x.size(-1)}")
        self._update_optimizer_stats(x)
        shape = x.shape
        grouped = x.view(*shape[:-1], self.groups, self.hidden_dim // self.groups)
        rms = torch.sqrt(grouped.square().mean(dim=-1, keepdim=True) + self.eps)
        t = grouped / rms
        coeff_shape = (1,) * (grouped.dim() - 2) + (self.groups, 1)
        y = rational_version_a5_4(
            t,
            self.numerator.view(coeff_shape + (6,)),
            self.denominator.view(coeff_shape + (4,)),
        )
        return (y * rms).reshape(shape)


class RationalFusedGlobalFFN(nn.Module):
    """Single-branch FFN using grouped P5/Q4 rational activations."""

    def __init__(
        self,
        dim,
        hidden_dim,
        dropout=0.0,
        rational_group_size=256,
        rational_max_groups=32,
        rational_init="silu",
        eps=1e-6,
    ):
        super().__init__()
        groups = resolve_group_count(hidden_dim, rational_group_size, rational_max_groups)
        self.in_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.rlb_activation = RationalFusedGlobalA5_4(
            hidden_dim,
            groups=groups,
            init=rational_init,
            fit_range=5.0,
            eps=eps,
        )
        self.out_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.out_proj(self.rlb_activation(self.in_proj(x))))
