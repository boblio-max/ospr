import numpy as np


# ============================================================
# 1. Create a toy NN with exactly 50 randomized weights
# ============================================================

rng = np.random.default_rng(42)

weights = rng.normal(
    loc=0.0,
    scale=1.0,
    size=50
)

print("Original weights:")
print(weights)


# ============================================================
# 2. Convert OSPR weight pairs into (theta, R)
# ============================================================

def get_polar_pairs(weights):
    assert len(weights) % 2 == 0

    pairs = weights.reshape(-1, 2)

    x = pairs[:, 0]
    y = pairs[:, 1]

    # Radius / half-length
    R = np.sqrt(x**2 + y**2)

    # Angle in radians
    theta = np.arctan2(y, x)

    return theta, R


# ============================================================
# 3. Calculate a, b, c for:
#
#       R = a*theta^3 + b*theta + c
#
# ============================================================

def calculate_abc(weights):
    theta, R = get_polar_pairs(weights)

    # Design matrix:
    #
    # [theta^3, theta, 1]
    #
    # We solve:
    #
    # X @ [a,b,c] ≈ R
    X = np.column_stack([
        theta**3,
        theta,
        np.ones_like(theta)
    ])

    coefficients, residuals, rank, singular_values = np.linalg.lstsq(
        X,
        R,
        rcond=None
    )

    a, b, c = coefficients

    return a, b, c


# ============================================================
# 4. Calculate coefficients
# ============================================================

a, b, c = calculate_abc(weights)

print("\nFitted cubic:")
print(f"f(theta) = {a:.8f} * theta^3 + {b:.8f} * theta + {c:.8f}")


# ============================================================
# 5. Show how well the cubic predicts the actual radii
# ============================================================

theta, actual_R = get_polar_pairs(weights)

predicted_R = (
    a * theta**3
    + b * theta
    + c
)

delta_R = actual_R - predicted_R

print("\nPair results:")
print("Pair | theta       | Actual R   | Predicted R | delta_R")

for i in range(len(theta)):
    print(
        f"{i:4d} | "
        f"{theta[i]:10.6f} | "
        f"{actual_R[i]:10.6f} | "
        f"{predicted_R[i]:11.6f} | "
        f"{delta_R[i]:10.6f}"
    )