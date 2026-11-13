"""The evaluation framework: split the data, train and test, and measure how well a classifier did."""

import numpy as np

from ml_comparison.checks import check_whole_number
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


def stratified_k_fold(data, k=5, seed=0):
    """Split the rows into k folds, and give each fold a turn as the test part.

    Returns the k pairs (training part, test part), one at a time. Every class is dealt out to the
    folds in turn, so each fold has about the same share of each class, to within one row. The same
    seed gives the same folds.
    """
    k = check_whole_number("k", k)
    if k < 2:
        raise ValueError(f"k must be at least 2, not {k}")
    if k > len(data):
        raise ValueError(f"{k} folds is more folds than rows ({len(data)})")
    rng = np.random.default_rng(seed)

    fold_of = np.empty(len(data), dtype=int)  # which fold each row is in
    next_fold = 0
    for code in range(len(data.classes)):
        rows = rng.permutation(np.flatnonzero(data.codes == code))
        fold_of[rows] = (next_fold + np.arange(len(rows))) % k
        # the next class carries on from where this one stopped, so the folds stay the same size
        next_fold = (next_fold + len(rows)) % k

    # the rows of each fold are worked out now, so the same seed always gives the same folds
    test_rows = [rng.permutation(np.flatnonzero(fold_of == i)) for i in range(k)]
    train_rows = [rng.permutation(np.flatnonzero(fold_of != i)) for i in range(k)]
    return ((subset(data, train_rows[i]), subset(data, test_rows[i])) for i in range(k))
