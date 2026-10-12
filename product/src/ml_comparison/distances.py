"""Distances between dots. a and b can be single dots or many rows at once."""

import numpy as np


def euclidean(a, b):
    """Straight-line distance: square root of the sum of squared differences."""
    diff = np.asarray(a) - np.asarray(b)
    return np.sqrt((diff ** 2).sum(axis=-1))


def manhattan(a, b):
    """City-block distance: sum of the absolute differences."""
    diff = np.asarray(a) - np.asarray(b)
    return np.abs(diff).sum(axis=-1)
