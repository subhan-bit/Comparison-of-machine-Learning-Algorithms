"""The evaluation framework: split the data, train and test, and measure how well a classifier did."""

import numpy as np

from ml_comparison.dataset import Dataset


def subset(data, rows):
    """The given rows of a data set, as a new Dataset with the same class list and positive class.

    Keeping the class list of the whole data set means a classifier gives the same score columns for
    every split, even when a rare class has no rows in one of them.
    """
    return Dataset(data.X[rows], data.y[rows], data.positive_label, data.classes)


def train_test_split(data, test_fraction=0.3, seed=0, stratify=False):
    """Split a data set into a training part and a test part. The same seed gives the same split.

    With stratify=True every class is split on its own, so the test part has the same share of each
    class as the whole data set (to within a row). That matters for a rare class such as fraud: a plain
    random split can leave it almost out of the test part. A class with two or more rows keeps at least
    one row in each part.
    """
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be between 0 and 1, not {test_fraction}")
    rng = np.random.default_rng(seed)

    if stratify:
        groups = [np.flatnonzero(data.codes == code) for code in range(len(data.classes))]
    else:
        groups = [np.arange(len(data))]

    train, test = [], []
    for rows in groups:
        rows = rng.permutation(rows)
        n_test = int(test_fraction * len(rows) + 0.5)  # rounded to the nearest row
        if len(rows) >= 2:
            n_test = min(max(n_test, 1), len(rows) - 1)  # at least one row in each part
        else:
            n_test = 0
        test.append(rows[:n_test])
        train.append(rows[n_test:])

    # shuffle again, so the rows are not in blocks of one class
    train = rng.permutation(np.concatenate(train))
    test = rng.permutation(np.concatenate(test))
    if len(train) == 0 or len(test) == 0:
        raise ValueError("the data set is too small to split")
    return subset(data, train), subset(data, test)
