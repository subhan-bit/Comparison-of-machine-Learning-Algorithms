"""Decision tree: predict a label by asking yes/no questions about the features."""

import numpy as np


def class_shares(codes):
    """The share of each class in a group. codes are the labels as numbers."""
    counts = np.bincount(codes)
    counts = counts[counts > 0]  # classes with no rows don't count
    return counts / counts.sum()


def gini(codes):
    """Gini index: 1 - sum of p^2. 0 for a pure group."""
    p = class_shares(codes)
    return 1 - (p ** 2).sum()


def entropy(codes):
    """Entropy: - sum of p * log2(p). 0 for a pure group."""
    p = class_shares(codes)
    return -(p * np.log2(p)).sum()


def misclassification(codes):
    """Misclassification error: 1 - the biggest share. 0 for a pure group."""
    p = class_shares(codes)
    return 1 - p.max()
