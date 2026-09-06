def bhaskaraI_smethod(x):
    c_pi = 3.141592653589793

    # Bhaskara I approx is defined on [0, pi]: 16x(pi-x)/(5pi^2 -4x(pi-x))
    # Handle sign and out-of-[0,pi] via odd symmetry and periodicity for
    # OSPR theta in [-pi, pi] (and pi/2 - x shift for cosine).
    if x < 0:
        return -bhaskaraI_smethod(-x)
    if x > c_pi:
        c_2pi = 2 * c_pi
        nx = x % c_2pi
        if nx > c_pi:
            return -bhaskaraI_smethod(c_2pi - nx)
        # nx in [0, pi] — compute directly
        denom = 5 * (c_pi * c_pi) - 4 * nx * (c_pi - nx)
        base = (16 * nx) * (c_pi - nx) / denom
        correction = 0.00001343 * nx * (c_pi - nx) * (nx - c_pi / 2) ** 2
        return base + correction

    denom = 5 * (c_pi * c_pi) - 4 * x * (c_pi - x)
    base = (16 * x) * (c_pi - x) / denom
    correction = 0.00001343 * x * (c_pi - x) * (x - c_pi / 2) ** 2
    return base + correction


def bhaskaraI_cmethod(x):
    c_pi = 3.141592653589793
    # cos(x) = sin(pi/2 - x) — reuse smethod for full-range accuracy
    # instead of the inaccurate (pi^2 -4x^2)/(pi^2 + x^2) which diverges past pi/2
    return bhaskaraI_smethod(c_pi / 2 - x)


def half_length_calculations(angle, delta_r):
    c_pi = 3.141592653589793

    r_min = 0.1
    r_range = 0.9

    x = abs(angle) / c_pi
    x2 = x * x

    cubic = x2 * (3 - 2 * x)

    return r_min + r_range * cubic + delta_r


def ospr_seesaw_method(angle, delta_r):
    half_length = half_length_calculations(angle, delta_r)

    height_1 = half_length * bhaskaraI_smethod(angle)
    height_2 = half_length * bhaskaraI_cmethod(angle)

    print(f"{height_1} {height_2}")
    return height_1, height_2


ospr_seesaw_method(0.7, 5)