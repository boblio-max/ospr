import math
import torch
import pytest

try:
    from ospr.torch import OSPRLinear, bhaskara_sin_torch, compression_stats
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


def test_bhaskara_torch_accuracy():
    xs = torch.linspace(-math.pi, math.pi, 25)
    for x in xs:
        bs = bhaskara_sin_torch(x).item()
        assert abs(bs - math.sin(x.item())) < 0.002


def test_bhaskara_torch_grad():
    x = torch.tensor([0.7], requires_grad=True)
    y = bhaskara_sin_torch(x).sum()
    y.backward()
    assert x.grad is not None
    assert x.grad.item() != 0


def test_ospr_linear_forward():
    layer = OSPRLinear(10, 8, halflength_mode="smoothstep")
    x = torch.randn(4, 10)
    y = layer(x)
    assert y.shape == (4, 8)
    # grad
    loss = y.sum()
    loss.backward()
    assert layer.theta.grad is not None


def test_ospr_linear_bhaskara():
    layer = OSPRLinear(10, 8, use_bhaskara=True, halflength_mode="smoothstep_pos")
    x = torch.randn(2, 10)
    y = layer(x)
    assert y.shape == (2, 8)
    y.sum().backward()
    assert layer.theta.grad is not None


def test_compression_stats():
    layer = OSPRLinear(10, 8)
    s = compression_stats(layer)
    assert s["raw_weights"] == 80
    assert s["stored_weights"] == 42
    assert abs(s["rate_weights"] - 1.90) < 0.01


def test_halflength_modes():
    for mode in ["smoothstep", "smoothstep_pos", "exp", "cubic4", "free"]:
        layer = OSPRLinear(4, 4, halflength_mode=mode)
        hl = layer.get_halflength()
        assert hl.shape == (8,)
        assert torch.isfinite(hl).all()
