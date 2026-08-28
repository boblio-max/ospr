"""
ospr.core — pure-Python / NumPy OSPR primitives.

Reference implementation of Bhaskara I sine/cosine approximation and
the OSPR seesaw (polar) encoding.

Formulas
--------
Bhaskara I on [0, pi]:
    sin(x) ~= 16 x (pi - x) / (5 pi^2 - 4 x (pi - x))
            + 0.00001343 * x (pi - x) (x - pi/2)^2   # small correction

Extended to all reals via odd symmetry and 2pi periodicity.

half_length:  R = r_min + r_range * smoothstep(|theta|/pi) + delta_r
              smoothstep(t) = t^2 (3 - 2t),  t in [0,1]

seesaw:       (h_x, h_y) = (R * cos(theta), R * sin(theta))
"""

from __future__ import annotations

import math

try:
    import numpy as np  # optional
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

C_PI = math.pi
C_2PI = 2 * math.pi
_CORR = 0.00001343


def _bhaskara_scalar(x: float) -> float:
    """Scalar Bhaskara I sin(x) on unbounded x."""
    if x < 0:
        return -_bhaskara_scalar(-x)
    if x > C_PI:
        nx = x % C_2PI
        if nx > C_PI:
            return -_bhaskara_scalar(C_2PI - nx)
        denom = 5 * (C_PI * C_PI) - 4 * nx * (C_PI - nx)
        base = (16 * nx) * (C_PI - nx) / denom
        corr = _CORR * nx * (C_PI - nx) * (nx - C_PI / 2) ** 2
        return base + corr
    denom = 5 * (C_PI * C_PI) - 4 * x * (C_PI - x)
    base = (16 * x) * (C_PI - x) / denom
    corr = _CORR * x * (C_PI - x) * (x - C_PI / 2) ** 2
    return base + corr


def bhaskara_sin(x):
    """
    Bhaskara I sine approximation.

    Supports scalar float, and (if NumPy is installed) np.ndarray.
    For torch.Tensor use ``ospr.torch.bhaskara_sin_torch``.

    Parameters
    ----------
    x : float | np.ndarray
        Angle in radians.

    Returns
    -------
    float | np.ndarray
        Approximate sin(x).
    """
    if _HAS_NUMPY and isinstance(x, np.ndarray):
        return _bhaskara_numpy(x, sin=True)
    if isinstance(x, (list, tuple)):
        if _HAS_NUMPY:
            return _bhaskara_numpy(np.asarray(x, dtype=float), sin=True)
        return type(x)(_bhaskara_scalar(float(v)) for v in x)
    return _bhaskara_scalar(float(x))


def bhaskara_cos(x):
    """
    Bhaskara I cosine via sin(pi/2 - x).

    Supports scalar and NumPy arrays. For torch use ``ospr.torch.bhaskara_cos_torch``.
    """
    if _HAS_NUMPY and isinstance(x, np.ndarray):
        return _bhaskara_numpy(x, sin=False)
    if isinstance(x, (list, tuple)):
        if _HAS_NUMPY:
            return _bhaskara_numpy(np.asarray(x, dtype=float), sin=False)
        return type(x)(_bhaskara_scalar(float(C_PI / 2 - v)) for v in x)
    return _bhaskara_scalar(float(C_PI / 2 - float(x)))


def _bhaskara_numpy(x: "np.ndarray", sin: bool = True) -> "np.ndarray":
    """Vectorized NumPy Bhaskara. If sin=False, computes cos(x)=sin(pi/2 - x)."""
    if not sin:
        x = C_PI / 2 - x
    # Map to [-pi, pi] : ((x + pi) % 2pi) - pi
    x_norm = np.remainder(x + C_PI, C_2PI) - C_PI
    ax = np.abs(x_norm)
    denom = 5 * (C_PI * C_PI) - 4 * ax * (C_PI - ax)
    base = 16 * ax * (C_PI - ax) / denom
    corr = _CORR * ax * (C_PI - ax) * (ax - C_PI / 2) ** 2
    val = base + corr
    return np.where(x_norm >= 0, val, -val)


# Backwards-compatible names (original API)
def bhaskaraI_smethod(x):
    """Legacy alias for :func:`bhaskara_sin`."""
    return bhaskara_sin(x)


def bhaskaraI_cmethod(x):
    """Legacy alias for :func:`bhaskara_cos`."""
    return bhaskara_cos(x)


def half_length(angle, delta_r: float = 0.0, r_min: float = 0.1, r_range: float = 0.9):
    """
    Smoothstep half-length.

    R = r_min + r_range * smoothstep(|angle|/pi) + delta_r
    smoothstep(t) = t^2 * (3 - 2t),  t = |angle|/pi clamped to [0,1]

    Parameters
    ----------
    angle : float | np.ndarray
        Polar angle theta in radians.
    delta_r : float | np.ndarray
        Optional residual (learned delta). Default 0.
    r_min, r_range : float
        Per-layer parameters (defaults match original ospr_method.py).

    Returns
    -------
    float | np.ndarray
    """
    if _HAS_NUMPY and isinstance(angle, np.ndarray):
        x = np.abs(angle) / C_PI
        x = np.clip(x, 0, 1)
        cubic = x * x * (3 - 2 * x)
        return r_min + r_range * cubic + delta_r
    # scalar path (also handles numpy scalar)
    x = abs(float(angle)) / C_PI
    if x > 1:
        x = 1.0
    elif x < 0:
        x = 0.0
    cubic = x * x * (3 - 2 * x)
    return r_min + r_range * cubic + float(delta_r)


def half_length_calculations(angle, delta_r):
    """Legacy alias: half_length(angle, delta_r, r_min=0.1, r_range=0.9)."""
    return half_length(angle, delta_r, r_min=0.1, r_range=0.9)


def seesaw(angle, delta_r: float = 0.0, r_min: float = 0.1, r_range: float = 0.9):
    """
    OSPR seesaw: convert (theta, delta_r) -> (weight_x, weight_y).

    Each theta encodes two weights as a polar pair:
        half = half_length(theta, ...)
        w_x = half * cos(theta)   (height_2 in original)
        w_y = half * sin(theta)   (height_1 in original)

    Returns (w_x, w_y) to match ospr_method.py height_2, height_1 ordering
    as (x=cos, y=sin). For backward compat, also returns (h1, h2) ordering
    via ``ospr_seesaw_method`` wrapper.

    Parameters
    ----------
    angle : float | np.ndarray
    delta_r, r_min, r_range : float

    Returns
    -------
    tuple[float, float] | tuple[np.ndarray, np.ndarray]
    """
    hl = half_length(angle, delta_r, r_min=r_min, r_range=r_range)
    # use scalar Bhaskara if NumPy not involved, else vectorized
    if _HAS_NUMPY and isinstance(angle, np.ndarray):
        s = bhaskara_sin(angle)
        c = bhaskara_cos(angle)
    else:
        s = bhaskara_sin(angle)
        c = bhaskara_cos(angle)
    w_x = hl * c
    w_y = hl * s
    return w_x, w_y


def ospr_seesaw_method(angle, delta_r):
    """
    Legacy wrapper: prints and returns (height_1, height_2) = (y, x).

    Original ospr_method.py printed ``"{h1} {h2}"`` where
    h1 = half*sin, h2 = half*cos. Kept for backward compat; prefer :func:`seesaw`.
    """
    hl = half_length_calculations(angle, delta_r)
    h1 = hl * bhaskaraI_smethod(angle)
    h2 = hl * bhaskaraI_cmethod(angle)
    print(f"{h1} {h2}")
    return h1, h2


def polar_encode(weights, r_min: float = 0.1, r_range: float = 0.9):
    """
    Encode a flat weight array (even length) into (theta, R, delta_r).

    Useful for analysis / fitting half_length. Mirrors half_length_test.py.

    Parameters
    ----------
    weights : np.ndarray | list[float]
        Flat array of even length.
    r_min, r_range : float

    Returns
    -------
    dict with keys theta, R, R_hat, delta_r
    """
    if not _HAS_NUMPY:
        raise ImportError("polar_encode requires NumPy")
    w = np.asarray(weights, dtype=float).reshape(-1, 2)
    x = w[:, 0]
    y = w[:, 1]
    R = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    R_hat = half_length(theta, delta_r=0.0, r_min=r_min, r_range=r_range)
    delta_r = R - R_hat
    return {"theta": theta, "R": R, "R_hat": R_hat, "delta_r": delta_r}


def cubic_fit(weights):
    """
    Fit R = a*theta^3 + b*theta + c via least squares (from half_length_test.py).

    Parameters
    ----------
    weights : array-like, even length

    Returns
    -------
    tuple[float, float, float]
        (a, b, c)
    """
    if not _HAS_NUMPY:
        raise ImportError("cubic_fit requires NumPy")
    w = np.asarray(weights, dtype=float)
    assert w.size % 2 == 0, "weights length must be even"
    pairs = w.reshape(-1, 2)
    x = pairs[:, 0]
    y = pairs[:, 1]
    R = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    X = np.column_stack([theta**3, theta, np.ones_like(theta)])
    coeffs, *_ = np.linalg.lstsq(X, R, rcond=None)
    return tuple(coeffs.tolist())
