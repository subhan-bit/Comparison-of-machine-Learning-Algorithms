"""Artificial data sets with known answers, for testing the classifiers."""

import numpy as np

from ml_comparison.dataset import Dataset


def make_blobs(n, seed):
    """Two clouds of points: class 0 around (0, 0) and class 1 around (5, 5)."""
    rng = np.random.default_rng(seed)

    # n points around each centre, with a spread (standard deviation) of 1
    class_0 = rng.normal(loc=[0, 0], scale=1, size=(n, 2))
    class_1 = rng.normal(loc=[5, 5], scale=1, size=(n, 2))

    # put the two clouds together: n zeros first, then n ones
    X = np.vstack([class_0, class_1])
    y = [0] * n + [1] * n
    return Dataset(X, y)


def make_xor(n, seed):
    """n points in a square; label 1 when x and y have the same sign."""
    rng = np.random.default_rng(seed)

    # n points spread evenly between -1 and 1 in both directions
    X = rng.uniform(-1, 1, size=(n, 2))

    # same sign means both positive or both negative
    y = []
    for x_value, y_value in X:
        if (x_value > 0) == (y_value > 0):
            y.append(1)
        else:
            y.append(0)
    return Dataset(X, y)


def make_circles(n, seed):
    """n points on a circle of radius 1 (class 0) and n on radius 3 (class 1)."""
    rng = np.random.default_rng(seed)

    # a random angle for each point, then its position on the circle
    inner_angles = rng.uniform(0, 2 * np.pi, size=n)
    inner = np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])

    outer_angles = rng.uniform(0, 2 * np.pi, size=n)
    outer = 3 * np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])

    # put the two circles together: n zeros first, then n ones
    X = np.vstack([inner, outer])
    y = [0] * n + [1] * n
    return Dataset(X, y)
