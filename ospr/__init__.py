"""
ospr — Opposite Seesaw Parameter Reduction Method library.

Pure-Python core in :mod:`ospr.core`, PyTorch layers in :mod:`ospr.torch`.

Quick start
-----------
>>> from ospr import bhaskara_sin, bhaskara_cos, half_length, seesaw
>>> from ospr.torch import OSPRLinear
>>> import torch
>>> layer = OSPRLinear(10, 8, use_bhaskara=True)
>>> y = layer(torch.randn(2, 10))
"""

from .__version__ import __version__
from .core import (
    C_2PI,
    C_PI,
    bhaskara_cos,
    bhaskara_sin,
    bhaskaraI_cmethod,
    bhaskaraI_smethod,
    cubic_fit,
    half_length,
    half_length_calculations,
    ospr_seesaw_method,
    polar_encode,
    seesaw,
)

# Lazy torch import — don't require torch for core usage
try:
    from .torch import (
        OSPRLinear,
        bhaskara_cos_torch,
        bhaskara_sin_torch,
        compression_stats,
        half_length_torch,
        half_length_torch_exp,
        half_length_torch_expressive,
        half_length_torch_pos,
    )

    _has_torch = True
except ImportError:  # torch not installed
    _has_torch = False
    OSPRLinear = None  # type: ignore
    bhaskara_sin_torch = None  # type: ignore
    bhaskara_cos_torch = None  # type: ignore
    half_length_torch = None  # type: ignore
    half_length_torch_pos = None  # type: ignore
    half_length_torch_expressive = None  # type: ignore
    half_length_torch_exp = None  # type: ignore
    compression_stats = None  # type: ignore

__all__ = [
    "__version__",
    "C_PI",
    "C_2PI",
    # core
    "bhaskara_sin",
    "bhaskara_cos",
    "bhaskaraI_smethod",
    "bhaskaraI_cmethod",
    "half_length",
    "half_length_calculations",
    "seesaw",
    "ospr_seesaw_method",
    "polar_encode",
    "cubic_fit",
    # torch (optional)
    "OSPRLinear",
    "bhaskara_sin_torch",
    "bhaskara_cos_torch",
    "half_length_torch",
    "half_length_torch_pos",
    "half_length_torch_expressive",
    "half_length_torch_exp",
    "compression_stats",
]
