import math
import numpy as np
from ospr.core import bhaskara_sin, bhaskara_cos, half_length, seesaw, polar_encode, cubic_fit


def test_bhaskara_accuracy():
    for x in np.linspace(-math.pi, math.pi, 25):
        assert abs(bhaskara_sin(x) - math.sin(x)) < 0.002, f"sin {x}"
        assert abs(bhaskara_cos(x) - math.cos(x)) < 0.002, f"cos {x}"


def test_bhaskara_vectorized():
    xs = np.array([0.0, 0.7, -1.2, math.pi])
    s = bhaskara_sin(xs)
    assert s.shape == xs.shape
    assert abs(s[1] - math.sin(0.7)) < 0.002


def test_half_length_monotonic():
    # R should increase with |theta|
    assert half_length(0.0) < half_length(1.0) < half_length(math.pi)


def test_seesaw_reconstruct():
    wx, wy = seesaw(0.7, delta_r=0.1)
    import math as m
    hl = half_length(0.7, 0.1)
    # wx ~ hl*cos, wy ~ hl*sin within Bhaskara error
    assert abs(wx - hl * m.cos(0.7)) < 0.002
    assert abs(wy - hl * m.sin(0.7)) < 0.002


def test_polar_encode():
    w = np.random.randn(10)
    info = polar_encode(w)
    assert info["theta"].shape == (5,)
    assert np.allclose(info["R"], info["R_hat"] + info["delta_r"])


def test_cubic_fit():
    w = np.random.randn(20)
    a, b, c = cubic_fit(w)
    assert isinstance(a, float)
