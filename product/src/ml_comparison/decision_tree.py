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


def split_impurity(left, right, measure):
    """The impurity of a split: each group's impurity, weighted by its size."""
    n = len(left) + len(right)
    return len(left) / n * measure(left) + len(right) / n * measure(right)


def best_split(X, codes, measure):
    """Try every feature and threshold, and return (feature, threshold) of the purest split.

    Returns None if no split is possible, because every row has the same values.
    """
    X = np.asarray(X, dtype=float)
    codes = np.asarray(codes)
    best = None
    best_score = np.inf
    for feature in range(X.shape[1]):
        values = np.unique(X[:, feature])  # each value once, sorted
        # try a threshold halfway between each pair of neighbouring values
        for threshold in (values[:-1] + values[1:]) / 2:
            goes_left = X[:, feature] <= threshold
            score = split_impurity(codes[goes_left], codes[~goes_left], measure)
            # only a strictly better score wins, so a tie keeps the first split found
            if score < best_score:
                best = (feature, float(threshold))
                best_score = score
    return best
