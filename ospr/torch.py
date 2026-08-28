"""
ospr.torch — PyTorch OSPR layers.

Provides OSPRLinear (2 weights -> 1 theta + per-layer halflength params)
and differentiable Bhaskara I approximations.

Example
-------
>>> import torch
>>> from ospr.torch import OSPRLinear
>>> layer = OSPRLinear(10, 8)  # 80 weights -> 40 theta + 2 halflength params
>>> x = torch.randn(4, 10)
>>> y = layer(x)  # (4, 8)

Compression
-----------
n_weights = in_features * out_features (must be even)
n_theta   = n_weights // 2
stored    = n_theta + hl_params  (hl_params depends on halflength_mode)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import C_PI, C_2PI

# Re-export constants
__all__ = [
    "C_PI",
    "C_2PI",
    "bhaskara_sin_torch",
    "bhaskara_cos_torch",
    "half_length_torch",
    "half_length_torch_pos",
    "half_length_torch_expressive",
    "half_length_torch_exp",
    "OSPRLinear",
    "compression_stats",
]


def bhaskara_sin_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Differentiable Bhaskara I approx for sin(x), vectorized.

    Handles any real x via periodic reduction to [-pi, pi] and odd symmetry.
    Formula: 16|xn|(pi-|xn|)/(5pi^2 -4|xn|(pi-|xn|)) + 0.00001343*|xn|(pi-|xn|)(|xn|-pi/2)^2

    Parameters
    ----------
    x : torch.Tensor
        Input angles in radians.

    Returns
    -------
    torch.Tensor
        Approximate sin(x), same shape/dtype/device as x, preserves autograd.
    """
    x_norm = torch.remainder(x + C_PI, C_2PI) - C_PI
    ax = torch.abs(x_norm)
    denom = 5 * (C_PI * C_PI) - 4 * ax * (C_PI - ax)
    base = 16 * ax * (C_PI - ax) / denom
    corr = 0.00001343 * ax * (C_PI - ax) * (ax - C_PI / 2) ** 2
    val = base + corr
    return torch.where(x_norm >= 0, val, -val)


def bhaskara_cos_torch(x: torch.Tensor) -> torch.Tensor:
    """Differentiable Bhaskara I cos(x) = sin(pi/2 - x)."""
    return bhaskara_sin_torch(C_PI / 2 - x)


def half_length_torch(theta: torch.Tensor, r_min: torch.Tensor, r_range: torch.Tensor) -> torch.Tensor:
    """
    Dynamic halflength: R_hat = r_min + r_range * smoothstep(|theta|/pi)

    smoothstep(t) = t^2 (3 - 2t), t in [0,1] clamped.

    Note: unconstrained — can go negative if optimizer drives r_min negative.
    Use :func:`half_length_torch_pos` for positivity constraint.
    """
    x = torch.abs(theta) / C_PI
    x = torch.clamp(x, 0, 1)
    cubic = x * x * (3 - 2 * x)
    return r_min + r_range * cubic


def half_length_torch_pos(
    theta: torch.Tensor, r_min_raw: torch.Tensor, r_range_raw: torch.Tensor
) -> torch.Tensor:
    """Positive-constrained: r_min=softplus(r_min_raw)*0.5+0.05, r_range=softplus(r_range_raw)."""
    r_min = F.softplus(r_min_raw) * 0.5 + 0.05
    r_range = F.softplus(r_range_raw) * 1.0
    x = torch.abs(theta) / C_PI
    x = torch.clamp(x, 0, 1)
    cubic = x * x * (3 - 2 * x)
    return r_min + r_range * cubic


def half_length_torch_expressive(theta: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Expressive per-layer: R_hat = softplus(c0)*0.5+0.05 + c1*x + c2*x^2 + c3*x^3

    coeffs: [4] vector per layer. 4 params overhead vs 2 for smoothstep.
    """
    x = torch.abs(theta) / C_PI
    x = torch.clamp(x, 0, 1)
    c0 = F.softplus(coeffs[0]) * 0.5 + 0.05
    c1 = coeffs[1]
    c2 = coeffs[2]
    c3 = coeffs[3]
    return c0 + c1 * x + c2 * x * x + c3 * x * x * x


def half_length_torch_exp(theta: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Positive large-range: R_hat = exp(a + b*x) where x=|theta|/pi, 2 params."""
    x = torch.abs(theta) / C_PI
    return torch.exp(a + b * x)


class OSPRLinear(nn.Module):
    """
    OSPR 1.6 Linear: 2 weights -> 1 theta + per-layer halflength params.

    Compression ~2x: n_weights = out*in (even) -> n_theta = n_weights//2 + 2 params per layer.
    Bias is stored uncompressed.

    Parameters
    ----------
    in_features, out_features : int
        Standard Linear dims. Product must be even.
    use_bhaskara : bool
        If True uses differentiable Bhaskara I instead of torch.sin/cos.
    bias : bool
        Whether to include bias.
    halflength_mode : str
        One of "smoothstep" (2 params, unconstrained),
        "smoothstep_pos" (2 params, positive via softplus),
        "exp" (2 params, exp),
        "cubic4" (4 params),
        "free" (per-pair logR, no compression — upper-bound baseline).

    Example
    -------
    >>> layer = OSPRLinear(10, 8, halflength_mode="smoothstep_pos", use_bhaskara=True)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bhaskara: bool = False,
        bias: bool = True,
        halflength_mode: str = "smoothstep",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bhaskara = use_bhaskara
        self.halflength_mode = halflength_mode
        n_weights = in_features * out_features
        assert n_weights % 2 == 0, f"OSPR requires even n_weights, got {n_weights} ({in_features}*{out_features})"
        self.n_theta = n_weights // 2
        self.theta = nn.Parameter((torch.rand(self.n_theta) * 2 - 1) * C_PI)
        if halflength_mode == "smoothstep":
            self.r_min = nn.Parameter(torch.tensor(0.4))
            self.r_range = nn.Parameter(torch.tensor(0.4))
        elif halflength_mode == "smoothstep_pos":
            self.r_min_raw = nn.Parameter(torch.tensor(0.0))
            self.r_range_raw = nn.Parameter(torch.tensor(0.5))
        elif halflength_mode == "exp":
            self.a = nn.Parameter(torch.tensor(-0.7))
            self.b = nn.Parameter(torch.tensor(0.0))
        elif halflength_mode == "cubic4":
            self.coeffs = nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.0]))
        elif halflength_mode == "free":
            self.logR = nn.Parameter(torch.randn(self.n_theta) * 0.2)
        else:
            raise ValueError(f"unknown halflength_mode {halflength_mode}")
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def get_halflength(self) -> torch.Tensor:
        if self.halflength_mode == "smoothstep":
            return half_length_torch(self.theta, self.r_min, self.r_range)
        elif self.halflength_mode == "smoothstep_pos":
            return half_length_torch_pos(self.theta, self.r_min_raw, self.r_range_raw)
        elif self.halflength_mode == "exp":
            return half_length_torch_exp(self.theta, self.a, self.b)
        elif self.halflength_mode == "cubic4":
            return half_length_torch_expressive(self.theta, self.coeffs)
        elif self.halflength_mode == "free":
            return torch.exp(self.logR)
        else:
            raise ValueError(self.halflength_mode)

    def reconstruct_weight(self) -> torch.Tensor:
        half = self.get_halflength()  # [n_theta]
        if self.use_bhaskara:
            s = bhaskara_sin_torch(self.theta)
            c = bhaskara_cos_torch(self.theta)
        else:
            s = torch.sin(self.theta)
            c = torch.cos(self.theta)
        w0 = half * c  # x
        w1 = half * s  # y
        w_pairs = torch.stack([w0, w1], dim=1).reshape(-1)
        weight = w_pairs.view(self.out_features, self.in_features)
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.reconstruct_weight()
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        if self.halflength_mode == "smoothstep":
            return (
                f"in={self.in_features}, out={self.out_features}, n_theta={self.n_theta}, "
                f"bhaskara={self.use_bhaskara}, mode={self.halflength_mode}, "
                f"r_min={self.r_min.item():.3f}, r_range={self.r_range.item():.3f}"
            )
        elif self.halflength_mode == "smoothstep_pos":
            r_min = F.softplus(self.r_min_raw).item() * 0.5 + 0.05
            r_range = F.softplus(self.r_range_raw).item()
            return (
                f"in={self.in_features}, out={self.out_features}, n_theta={self.n_theta}, "
                f"bhaskara={self.use_bhaskara}, mode={self.halflength_mode}, "
                f"r_min~{r_min:.3f}, r_range~{r_range:.3f}"
            )
        elif self.halflength_mode == "exp":
            return (
                f"in={self.in_features}, out={self.out_features}, n_theta={self.n_theta}, "
                f"bhaskara={self.use_bhaskara}, mode={self.halflength_mode}, "
                f"a={self.a.item():.3f}, b={self.b.item():.3f}"
            )
        elif self.halflength_mode == "cubic4":
            return (
                f"in={self.in_features}, out={self.out_features}, n_theta={self.n_theta}, "
                f"bhaskara={self.use_bhaskara}, mode={self.halflength_mode}, "
                f"coeffs={self.coeffs.data.tolist()}"
            )
        else:
            return (
                f"in={self.in_features}, out={self.out_features}, n_theta={self.n_theta}, "
                f"bhaskara={self.use_bhaskara}, mode={self.halflength_mode}"
            )


def compression_stats(module: OSPRLinear) -> dict:
    """Compute compression statistics for an OSPRLinear layer."""
    if module.halflength_mode in ("smoothstep", "smoothstep_pos", "exp"):
        hl_params = 2
    elif module.halflength_mode == "cubic4":
        hl_params = 4
    elif module.halflength_mode == "free":
        hl_params = module.n_theta
    else:
        hl_params = 2
    n_raw = module.in_features * module.out_features
    n_stored = module.n_theta + hl_params
    if module.bias is not None:
        n_raw_b = n_raw + module.bias.numel()
        n_stored_b = n_stored + module.bias.numel()
        return {
            "raw_weights": n_raw,
            "stored_weights": n_stored,
            "raw_with_bias": n_raw_b,
            "stored_with_bias": n_stored_b,
            "rate_weights": n_raw / n_stored,
            "rate_with_bias": n_raw_b / n_stored_b,
            "hl_params": hl_params,
            "mode": module.halflength_mode,
        }
    return {"raw": n_raw, "stored": n_stored, "rate": n_raw / n_stored, "hl_params": hl_params}
