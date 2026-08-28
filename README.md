# OSPR — Oscillating Seesaw Polar Representation

> **Compress every 2 weights into 1 angle.** A `2:1` polar encoding for neural network weights using a smoothstep half-length and a 7th-century Bhaskara I sine approximation — fully differentiable in PyTorch.

```
(w0, w1)  ──►  (theta, R)  ──►  (theta) + 2 params/layer  ──►  reconstruct (w0, w1)
  2 floats        polar          stored: 1 theta/pair         R(theta) * [cos(theta), sin(theta)]
                 atan2, sqrt     vs 2 raw weights             with Bhaskara or torch.sin
```

---

## Why OSPR?

Standard linear layers store every weight independently. OSPR observes that **pairs of weights can be represented in polar coordinates** — an angle `θ` and a radius `R` — and that `R` is highly predictable from `|θ|` alone.

Instead of storing `R` per pair, OSPR learns **one smooth curve `R(θ)` per layer** (just 2 parameters). Each pair then costs **1 float (`θ`) instead of 2**, giving ~**1.9× compression** including biases, or **2.0×** on weights alone.

The "seesaw" name comes from the reconstruction: `θ` tilts the seesaw, `R(θ)` sets its half-length, and the two ends are the two weights.

---

## How it works

### 1. Pair → Polar

Take a flat weight vector of even length (e.g. `10×8 = 80` weights) and reshape into pairs:

```python
import numpy as np

weights = np.random.randn(80)          # raw weights
pairs = weights.reshape(-1, 2)         # 40 pairs
x, y = pairs[:, 0], pairs[:, 1]

R = np.sqrt(x**2 + y**2)               # radius / half-length
theta = np.arctan2(y, x)               # angle in [-pi, pi]
```

### 2. Half-length curve `R(θ)`

Empirically, `R` correlates with `|θ|`. OSPR fits it with a **smoothstep cubic**:

```python
# R_hat(θ) = r_min + r_range * smoothstep(|θ|/π) + delta_r
# smoothstep(t) = t² * (3 - 2t),  t = |θ|/π ∈ [0, 1]

from ospr import half_length

theta = 0.7
R_hat = half_length(theta, delta_r=0.0, r_min=0.1, r_range=0.9)
# 0.1 + 0.9 * smoothstep(0.7/π) ≈ 0.214
```

`r_min` and `r_range` are **learned per layer** (2 parameters total). `delta_r = R - R_hat` is the residual — discarded for compression, or kept for analysis:

```python
from ospr import polar_encode

info = polar_encode(weights)  # {theta, R, R_hat, delta_r}
# info["delta_r"] is what you'd need for lossless reconstruction
```

> The library also provides `cubic_fit()` which reproduces `half_length_test.py`'s `R = a·θ³ + b·θ + c` least-squares fit for analysis.

### 3. Seesaw reconstruction

Each stored `θ` is expanded back to two weights:

```python
from ospr import seesaw

# single pair
wx, wy = seesaw(theta=0.7, delta_r=0.0)
# wx = R_hat * cos(θ), wy = R_hat * sin(θ)
# wx ≈ 0.163, wy ≈ 0.137  (with Bhaskara approx)

# vectorized over many angles
import numpy as np
thetas = np.array([0.0, 0.7, -1.2, 3.14])
wx, wy = seesaw(thetas)   # both are np.ndarray shape (4,)
```

This is `ospr_seesaw_method` from `ospr_method.or:26` — historically it printed `height_1 = R·sin(θ)` and `height_2 = R·cos(θ)`:

```python
from ospr import ospr_seesaw_method

h1, h2 = ospr_seesaw_method(0.7, delta_r=5)
# prints: "3.34... 3.98..."
# h1 = wy, h2 = wx
```

### 4. Bhaskara I sine (no `math.sin` needed)

OSPR can reconstruct without calling `sin`/`cos` at all, using **Bhaskara I's 7th-century rational approximation** plus a tiny correction:

```
sin(x) ≈ 16·x·(π - x) / (5π² - 4·x·(π - x))  +  0.00001343·x·(π - x)·(x - π/2)²   for x ∈ [0, π]
```

Extended to all reals by odd symmetry and `2π` periodicity. Max error < `0.002` vs `math.sin` — and it's **30% faster in micro-benchmarks** (`trig_speedup_test.py`) while remaining differentiable.

```python
from ospr import bhaskara_sin, bhaskara_cos
import math

bhaskara_sin(0.7)          # 0.64327...
math.sin(0.7)              # 0.64421...  err ≈ 0.00094

# vectorized (NumPy)
bhaskara_sin(np.linspace(-math.pi, math.pi, 7))

# PyTorch (differentiable, preserves autograd)
from ospr.torch import bhaskara_sin_torch
import torch
x = torch.tensor([0.7], requires_grad=True)
y = bhaskara_sin_torch(x)
y.backward()  # x.grad is populated
```

Cosine reuses the sine: `cos(x) = sin(π/2 - x)` (`core.py:80`), avoiding the inaccurate `(π² - 4x²)/(π² + x²)` form.

---

## Installation

```bash
pip install ospr                  # from PyPI (once published)
# or from source:
pip install -e .                  # core only, needs numpy
pip install -e ".[torch]"         # with PyTorch layers
pip install -e ".[dev]"           # plus pytest, ruff, mypy
```

- Python ≥3.9
- `numpy>=1.20` (required)
- `torch>=2.0` (optional, only for `ospr.torch`)

```bash
# verify
python -c "import ospr; print(ospr.__version__)"
# 0.1.0
```

---

## Quick start

### Pure Python / NumPy

```python
from ospr import bhaskara_sin, bhaskara_cos, half_length, seesaw
import numpy as np

# 1. trig without math
print(bhaskara_sin(0.7), bhaskara_cos(0.7))

# 2. half-length for one angle or a whole layer
thetas = np.random.uniform(-np.pi, np.pi, size=40)
R_hat = half_length(thetas, r_min=0.4, r_range=0.4)  # shape (40,)

# 3. seesaw: theta -> (w0, w1) pairs
wx, wy = seesaw(thetas)
weights_reconstructed = np.column_stack([wx, wy]).reshape(-1)  # 80 weights

# 4. analyze an existing weight tensor
from ospr import polar_encode, cubic_fit

weights = np.random.randn(80)
info = polar_encode(weights)
print(info["theta"][:3], info["R"][:3], info["delta_r"][:3])

a, b, c = cubic_fit(weights)
print(f"R ≈ {a:.4f}·θ³ + {b:.4f}·θ + {c:.4f}")
```

### PyTorch — drop-in `OSPRLinear`

`OSPRLinear` (`torch.py:126`) is a drop-in for `nn.Linear` that stores `n/2` thetas + 2 params/layer.

```python
import torch
import torch.nn as nn
from ospr.torch import OSPRLinear, compression_stats

class VanillaMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 8)   # 80 weights
        self.fc2 = nn.Linear(8, 3)    # 24 weights

class OSPRMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 80 -> 40 theta +2  and  24 -> 12 theta +2
        self.fc1 = OSPRLinear(10, 8, halflength_mode="smoothstep_pos", use_bhaskara=False)
        self.fc2 = OSPRLinear(8, 3, halflength_mode="smoothstep_pos", use_bhaskara=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

m = OSPRMLP()
print(m.fc1)  # OSPRLinear(in=10, out=8, n_theta=40, bhaskara=False, mode=smoothstep_pos, r_min~0.39, r_range~1.0)
print(compression_stats(m.fc1))
# {'raw_weights': 80, 'stored_weights': 42, 'rate_weights': 1.90, 'rate_with_bias': 1.76, ...}

x = torch.randn(4, 10)
y = m(x)           # (4, 3)
loss = y.sum()
loss.backward()    # theta, r_min, r_range all get grads
```

**With Bhaskara (no `torch.sin` at inference):**

```python
layer = OSPRLinear(10, 8, use_bhaskara=True, halflength_mode="smoothstep_pos")
# forward now uses bhaskara_sin_torch / bhaskara_cos_torch (differentiable)
```

**Inspecting reconstruction:**

```python
layer = OSPRLinear(4, 4)
W = layer.reconstruct_weight()   # (4, 4)  — the materialized weight matrix
R = layer.get_halflength()       # (8,)    — half-lengths for each theta
print(W.shape, R.shape)
```

### Training example

`examples/mnist_ospr.py` trains vanilla vs OSPR on synthetic data (same setup as `train_100.py:13`):

```python
import torch, torch.nn as nn
from ospr.torch import OSPRLinear

torch.manual_seed(42)
# ... make_data() as in train_100.py ...

class Model(nn.Module):
    def __init__(self, bhaskara=False):
        super().__init__()
        self.fc1 = OSPRLinear(10, 8, use_bhaskara=bhaskara, halflength_mode="smoothstep_pos")
        self.fc2 = OSPRLinear(8, 3, use_bhaskara=bhaskara, halflength_mode="smoothstep_pos")
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

for name, bh in [("sin", False), ("bhaskara", True)]:
    m = Model(bhaskara=bh)
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    for epoch in range(30):
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(m(X_train), y_train)
        loss.backward()
        opt.step()
    acc = (m(X_test).argmax(1) == y_test).float().mean().item()
    print(name, f"{acc:.3f}")
```

Run it:

```bash
python examples/mnist_ospr.py
# OSPR sin: test acc 0.610  ...
# OSPR bhaskara: test acc 0.610  ...
```

---

## API reference

### `ospr.core` — pure Python / NumPy

| Function | Signature | Description |
|---|---|---|
| `bhaskara_sin` | `(x: float \| np.ndarray) -> same` | Bhaskara I sin, scalar or vectorized (`core.py:54`) |
| `bhaskara_cos` | `(x) -> same` | `sin(π/2 - x)` (`core.py:80`) |
| `bhaskaraI_smethod` | alias of `bhaskara_sin` | legacy name from `ospr_method.py:1` |
| `bhaskaraI_cmethod` | alias of `bhaskara_cos` | legacy name |
| `half_length` | `(angle, delta_r=0, r_min=0.1, r_range=0.9)` | smoothstep `R(θ)` (`core.py:120`) |
| `half_length_calculations` | `(angle, delta_r)` | legacy `half_length` with defaults `0.1, 0.9` |
| `seesaw` | `(angle, delta_r, r_min, r_range) -> (wx, wy)` | `θ -> (R·cos, R·sin)` (`core.py:160`) |
| `ospr_seesaw_method` | `(angle, delta_r) -> (h1, h2)` | legacy print wrapper `h1=R·sin, h2=R·cos` (`core.py:195`) |
| `polar_encode` | `(weights) -> {theta,R,R_hat,delta_r}` | analyze existing weights (`core.py:209`) |
| `cubic_fit` | `(weights) -> (a,b,c)` | least-squares `R = aθ³+bθ+c` (`core.py:237`) |

### `ospr.torch` — PyTorch

| Symbol | Description |
|---|---|
| `bhaskara_sin_torch(x: Tensor) -> Tensor` | differentiable Bhaskara (`torch.py:47`) |
| `bhaskara_cos_torch(x)` | `sin(π/2 - x)` |
| `half_length_torch(theta, r_min, r_range)` | unconstrained smoothstep (`torch.py:78`) |
| `half_length_torch_pos(theta, r_min_raw, r_range_raw)` | `softplus` positive variant (`torch.py:93`) |
| `half_length_torch_expressive(theta, coeffs[4])` | cubic, 4 params (`torch.py:105`) |
| `half_length_torch_exp(theta, a, b)` | `exp(a + b·|θ|/π)` (`torch.py:120`) |
| `OSPRLinear(in, out, use_bhaskara, bias, halflength_mode)` | compressed linear (`torch.py:126`) |
| `compression_stats(layer) -> dict` | `raw/stored/rate` (`torch.py:256`) |

`OSPRLinear` methods: `.get_halflength()`, `.reconstruct_weight()`, `.forward(x)`

### Halflength modes

| Mode | Params/layer | Formula | When to use |
|---|---|---|---|
| `smoothstep` | 2 | `r_min + r_range·smoothstep(|θ|/π)` | simple, but can go negative |
| `smoothstep_pos` | 2 | `softplus(r_min_raw)*0.5+0.05 + ...` | **default recommended** — always positive |
| `exp` | 2 | `exp(a + b·|θ|/π)` | large dynamic range |
| `cubic4` | 4 | `softplus(c0)+c1·x+c2·x²+c3·x³` | more expressive, 2 extra params |
| `free` | `n_theta` | `exp(logR)` per pair | no compression — upper-bound baseline |

---

## Compression math

```
n_weights = in_features * out_features   # must be even
n_theta   = n_weights // 2
hl_params = 2  (or 4 for cubic4, n_theta for free)
stored    = n_theta + hl_params
rate      = n_weights / stored

Example: 10×8 layer
  raw 80  ->  stored 40 +2 = 42  ->  1.90×
  with bias 80+8=88 -> 50 -> 1.76×
```

Check any layer:

```python
from ospr.torch import OSPRLinear, compression_stats
layer = OSPRLinear(10, 8)
print(compression_stats(layer)["rate_weights"])  # 1.904...
```

---

## Project layout

```
ospr/
├─ pyproject.toml              # build + deps (numpy required, torch optional)
├─ README.md                   # this file
├─ ospr/
│  ├─ __init__.py              # top-level re-exports
│  ├─ __version__.py           # 0.1.0
│  ├─ core.py                  # Bhaskara + half-length + seesaw (scalar + NumPy)
│  └─ torch.py                 # OSPRLinear + torch Bhaskara (differentiable)
├─ ospr_method.py              # deprecated shim -> ospr.core
├─ ospr_torch.py               # deprecated shim -> ospr.torch
├─ ospr_method.or              # original spec language version
├─ tests/
│  ├─ test_core.py
│  └─ test_torch.py
└─ examples/
   └─ mnist_ospr.py
```

Migrating old scripts:

```python
# old
from ospr_method import bhaskaraI_smethod
from ospr_torch import OSPRLinear

# new
from ospr import bhaskara_sin
from ospr.torch import OSPRLinear
# or
from ospr import OSPRLinear  # re-exported at top level
```

---

## Design notes & limitations

* **Even weights required.** `in_features * out_features` must be even — each `θ` encodes 2 weights. Odd layers need padding or fallback to `nn.Linear`.
* **`R(θ)` is an approximation.** The smoothstep curve cannot fit arbitrary `R` distributions perfectly; `diagnose.py` shows the MSE vs a trained model. Use `cubic4` or `free` if you need tighter fit.
* **Bhaskara error is small but non-zero.** <0.002 absolute. Fine for inference, but if you need exact `sin`, set `use_bhaskara=False`.
* **Initialization matters.** `OSPRLinear` samples `θ ~ Uniform(-π, π)` (`torch.py:169`) — earlier `randn*0.5` clustering near 0 hurts coverage.

---

## Testing

```bash
pip install -e ".[dev]"
pytest -v          # 12 tests: core + torch
python -m ruff check ospr/
```

---

## License

MIT — see `LICENSE` (or `pyproject.toml:10`).

Original OSPR concept and `ospr_method.or` by the OSPR contributors.
