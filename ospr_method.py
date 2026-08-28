"""
Backward-compat shim: ``import ospr_method`` still works.
New code should use ``from ospr.core import ...`` or ``from ospr import ...``.
"""

import warnings

warnings.warn(
    "ospr_method is deprecated, use `from ospr.core import bhaskara_sin, ...` or `from ospr import ...`",
    DeprecationWarning,
    stacklevel=2,
)

from ospr.core import (
    bhaskara_cos as bhaskaraI_cmethod,
    bhaskara_sin as bhaskaraI_smethod,
    half_length as _half_length,
    half_length_calculations,
    ospr_seesaw_method,
    seesaw,
)

# Keep original function names exported
__all__ = [
    "bhaskaraI_smethod",
    "bhaskaraI_cmethod",
    "half_length_calculations",
    "ospr_seesaw_method",
]

# Original file executed ospr_seesaw_method(0.7, 5) at import time.
# We no longer do that on import — run only if executed as script.
if __name__ == "__main__":
    ospr_seesaw_method(0.7, 5)
