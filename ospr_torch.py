import math
import torch
import torch.nn as nn
import torch.nn.functional as F

C_PI = 3.141592653589793
C_2PI = 2 * C_PI

def bhaskara_sin_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Differentiable Bhaskara I approx for sin(x).
    Handles any real x via periodic reduction to [-pi, pi] and odd symmetry.
    Vectorized, no recursion, preserves autograd.
    Formula: 16|x|(pi-|x|)/(5pi^2 -4|x|(pi-|x|)) + 0.00001343*|x|(pi-|x|)(|x|-pi/2)^2
    """
    # Map to [-pi, pi] via remainder: ((x + pi) % 2pi) - pi
    # torch.remainder is differentiable (grad 1) except at wrap points
    x_norm = torch.remainder(x + C_PI, C_2PI) - C_PI
    # Bhaskara on abs
    ax = torch.abs(x_norm)
    denom = 5 * (C_PI * C_PI) - 4 * ax * (C_PI - ax)
    # denom in [4*pi^2, 5*pi^2] >0, safe
    base = 16 * ax * (C_PI - ax) / denom
    corr = 0.00001343 * ax * (C_PI - ax) * (ax - C_PI / 2) ** 2
    val = base + corr
    # odd
    return torch.where(x_norm >= 0, val, -val)

def bhaskara_cos_torch(x: torch.Tensor) -> torch.Tensor:
    """cos(x) = sin(pi/2 - x) reuse sin"""
    return bhaskara_sin_torch(C_PI / 2 - x)

def half_length_torch(theta: torch.Tensor, r_min: torch.Tensor, r_range: torch.Tensor) -> torch.Tensor:
    """
    Dynamic halflength per-layer: R_hat = r_min + r_range * smoothstep(|theta|/pi)
    smoothstep: x^2 (3 - 2x) with x=|theta|/pi in [0,1] (clamped)
    All differentiable, r_min/r_range are nn.Parameter per-layer.
    NOTE: unconstrained version can go negative (optimizer cheat). Caller should use
    half_length_torch_pos if positivity required (see below). Kept for backward compat.
    """
    x = torch.abs(theta) / C_PI
    x = torch.clamp(x, 0, 1)
    cubic = x * x * (3 - 2 * x)
    return r_min + r_range * cubic

def half_length_torch_pos(theta: torch.Tensor, r_min_raw: torch.Tensor, r_range_raw: torch.Tensor) -> torch.Tensor:
    """Positive-constrained version: r_min=softplus(r_min_raw)*0.5, r_range=softplus(r_range_raw)"""
    r_min = F.softplus(r_min_raw) * 0.5 + 0.05  # ensure ~0.05..inf
    r_range = F.softplus(r_range_raw) * 1.0
    x = torch.abs(theta) / C_PI
    x = torch.clamp(x, 0, 1)
    cubic = x * x * (3 - 2 * x)
    return r_min + r_range * cubic

def half_length_torch_expressive(theta: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Expressive per-layer: R_hat = softplus(c0) + softplus(c1)*x + softplus(c2)*x^2 + c3*x^3
    coeffs: [4] vector per layer. Allows more flexible R(theta) than 2-param smoothstep.
    Still shared per-layer (4 params overhead) vs per-pair.
    """
    # coeffs[0..2] positive via softplus to keep magnitude positive-ish, c3 free for shape
    x = torch.abs(theta) / C_PI
    x = torch.clamp(x, 0, 1)
    c0 = F.softplus(coeffs[0]) * 0.5 + 0.05
    c1 = coeffs[1]
    c2 = coeffs[2]
    c3 = coeffs[3]
    return c0 + c1 * x + c2 * x * x + c3 * x * x * x

def half_length_torch_exp(theta: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Even simpler positive large-range: R_hat = exp(a + b*x) where x=|theta|/pi, 2 params but exp gives large dynamic range"""
    x = torch.abs(theta) / C_PI
    return torch.exp(a + b * x)

class OSPRLinear(nn.Module):
    """
    OSPR 1.6 Linear: 2 weights -> 1 theta + per-layer halflength params
    Compression ~2x: n_weights = out*in (even) -> n_theta = n_weights//2 + 2 params per layer
    Example 100 weights -> 50 theta +2 =52 vs 100 =1.92x
    Bias is stored uncompressed (as in origin-dev/lib/ospr.py)

    use_bhaskara=False -> torch.sin/cos exact
    use_bhaskara=True  -> differentiable Bhaskara I approx
    halflength_mode: 'smoothstep' (default, 2 params, can go negative), 'smoothstep_pos' (positive), 'exp' (2 params exp), 'cubic4' (4 params)
    """
    def __init__(self, in_features: int, out_features: int, use_bhaskara: bool = False, bias: bool = True, halflength_mode: str = "smoothstep"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bhaskara = use_bhaskara
        self.halflength_mode = halflength_mode
        n_weights = in_features * out_features
        assert n_weights % 2 == 0, f"OSPR requires even n_weights, got {n_weights} ({in_features}*{out_features})"
        self.n_theta = n_weights // 2
        # One theta per pair: uniform [-pi, pi] to match vanilla weight direction distribution
        # Previous: randn*0.5 => clustered near 0, poor coverage, initial half small
        self.theta = nn.Parameter((torch.rand(self.n_theta) * 2 - 1) * C_PI)
        # Per-layer dynamic halflength params: init to give half ~0.5 mean (matches vanilla R ~0.6)
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
            # per-pair magnitude: no compression, for upper-bound baseline
            self.logR = nn.Parameter(torch.randn(self.n_theta) * 0.2)
        else:
            raise ValueError(f"unknown halflength_mode {halflength_mode}")
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def get_halflength(self):
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
            raise ValueError

    def reconstruct_weight(self):
        half = self.get_halflength()  # [n_theta]
        if self.use_bhaskara:
            s = bhaskara_sin_torch(self.theta)
            c = bhaskara_cos_torch(self.theta)
        else:
            s = torch.sin(self.theta)
            c = torch.cos(self.theta)
        # h1 = half*sin = y, h2 = half*cos = x, pairs are (x,y) = (h2,h1) to match ospr_method.py:50-51
        w0 = half * c  # x
        w1 = half * s  # y
        # interleave as [w0_0, w1_0, w0_1, w1_1, ...]
        w_pairs = torch.stack([w0, w1], dim=1).reshape(-1)  # [n_weights]
        weight = w_pairs.view(self.out_features, self.in_features)
        return weight

    def forward(self, x):
        w = self.reconstruct_weight()
        return F.linear(x, w, self.bias)

    def extra_repr(self):
        if self.halflength_mode == "smoothstep":
            return f"in={self.in_features}, out={self.out_features}, n_theta={self.n_theta}, bhaskara={self.use_bhaskara}, mode={self.halflength_mode}, r_min={self.r_min.item():.3f}, r_range={self.r_range.item():.3f}"
        elif self.halflength_mode == "smoothstep_pos":
            r_min = F.softplus(self.r_min_raw).item()*0.5+0.05
            r_range = F.softplus(self.r_range_raw).item()
            return f"in={self.in_features}, out={self.out_features}, n_theta={self.n_theta}, bhaskara={self.use_bhaskara}, mode={self.halflength_mode}, r_min~{r_min:.3f}, r_range~{r_range:.3f}"
        elif self.halflength_mode == "exp":
            return f"in={self.in_features}, out={self.out_features}, n_theta={self.n_theta}, bhaskara={self.use_bhaskara}, mode={self.halflength_mode}, a={self.a.item():.3f}, b={self.b.item():.3f}"
        elif self.halflength_mode == "cubic4":
            return f"in={self.in_features}, out={self.out_features}, n_theta={self.n_theta}, bhaskara={self.use_bhaskara}, mode={self.halflength_mode}, coeffs={self.coeffs.data.tolist()}"
        else:
            return f"in={self.in_features}, out={self.out_features}, n_theta={self.n_theta}, bhaskara={self.use_bhaskara}, mode={self.halflength_mode}"

def compression_stats(module: OSPRLinear):
    # stored = theta + halflength params (varies by mode)
    if module.halflength_mode in ("smoothstep", "smoothstep_pos", "exp"):
        hl_params = 2
    elif module.halflength_mode == "cubic4":
        hl_params = 4
    elif module.halflength_mode == "free":
        hl_params = module.n_theta  # per-pair R
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
