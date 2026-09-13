import math


def number_to_polar(x):
    theta = 2 * math.pi * (x - math.floor(x))
    radius = abs(x)

    px = radius * math.cos(theta)
    py = radius * math.sin(theta)

    return px, py

def circle_to_number(px, py):
    theta = math.atan2(py, px)

    if theta < 0:
        theta += 2 * math.pi

    x = theta / (2 * math.pi)

    return x
