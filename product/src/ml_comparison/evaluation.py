"""The evaluation framework: split the data, train and test, and measure how well a classifier did."""

import numpy as np

from ml_comparison.dataset import Dataset


def subset(data, rows):
    """The given rows of a data set, as a new Dataset with the same class list and positive class.

    Keeping the class list of the whole data set means a classifier gives the same score columns for
    every split, even when a rare class has no rows in one of them.
    """
    return Dataset(data.X[rows], data.y[rows], data.positive_label, data.classes)


def train_test_split(data, test_fraction=0.3, seed=0):
    """Split a data set into a training part and a test part. The same seed gives the same split.

    Each part keeps at least one row.
    """
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be between 0 and 1, not {test_fraction}")
    if len(data) < 2:
        raise ValueError("the data set is too small to split")
    rng = np.random.default_rng(seed)
    rows = rng.permutation(len(data))
    n_test = int(test_fraction * len(rows) + 0.5)  # rounded to the nearest row
    n_test = min(max(n_test, 1), len(rows) - 1)  # at least one row in each part
    return subset(data, rows[n_test:]), subset(data, rows[:n_test])
