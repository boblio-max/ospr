"""
Backward-compat shim: ``import ospr_torch`` still works.
New code should use ``from ospr.torch import ...`` or ``from ospr import OSPRLinear``.
"""

import warnings

warnings.warn(
    "ospr_torch is deprecated, use `from ospr.torch import OSPRLinear, ...` or `from ospr import OSPRLinear`",
    DeprecationWarning,
    stacklevel=2,
)

from ospr.torch import (  # noqa: F401
    C_2PI,
    C_PI,
    OSPRLinear,
    bhaskara_cos_torch,
    bhaskara_sin_torch,
    compression_stats,
    half_length_torch,
    half_length_torch_exp,
    half_length_torch_expressive,
    half_length_torch_pos,
)

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
