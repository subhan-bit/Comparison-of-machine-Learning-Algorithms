"""Decision tree: predict a label by asking yes/no questions about the features."""

import numpy as np

from ml_comparison.classifier import Classifier
from ml_comparison.dataset import Dataset


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


MEASURES = {"gini": gini, "entropy": entropy, "misclassification": misclassification}


class Node:
    """One node of the tree. A leaf has no question and no children."""

    def __init__(self, shares):
        self.shares = shares   # the share of each class among the training rows here
        self.feature = None    # the question is: is X[feature] <= threshold?
        self.threshold = None
        self.left = None       # rows where the answer is yes
        self.right = None      # rows where the answer is no

    def is_leaf(self):
        return self.left is None


class DecisionTree(Classifier):
    def __init__(self, measure="gini", max_depth=None, min_samples_split=2):
        if measure not in MEASURES:
            raise ValueError(f"measure must be one of {list(MEASURES)}, not {measure!r}")
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be 0 or more")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        self.measure = MEASURES[measure]
        self.max_depth = max_depth                  # None means no limit
        self.min_samples_split = min_samples_split  # a node needs this many rows to split

    def _train(self, data: Dataset) -> None:
        self.n_classes = len(data.classes)
        self.root = self._grow(data.X, data.codes, depth=0)

    def _grow(self, X, codes, depth):
        """Make a node for these rows, and keep splitting until a stopping rule says stop."""
        node = Node(np.bincount(codes, minlength=self.n_classes) / len(codes))
        if self._should_stop(codes, depth):
            return node
        split = best_split(X, codes, self.measure)
        if split is None:  # every row is the same, so there is nothing to ask
            return node

        node.feature, node.threshold = split
        goes_left = X[:, node.feature] <= node.threshold
        node.left = self._grow(X[goes_left], codes[goes_left], depth + 1)
        node.right = self._grow(X[~goes_left], codes[~goes_left], depth + 1)
        return node

    def _should_stop(self, codes, depth):
        is_pure = len(np.unique(codes)) == 1
        too_deep = self.max_depth is not None and depth >= self.max_depth
        too_small = len(codes) < self.min_samples_split
        return is_pure or too_deep or too_small

    def _leaf_for(self, point):
        """Follow the questions from the root down to a leaf."""
        node = self.root
        while not node.is_leaf():
            node = node.left if point[node.feature] <= node.threshold else node.right
        return node

    def _predict_scores(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._leaf_for(point).shares for point in X])

    def count_leaves(self, node=None):
        node = node or self.root
        if node.is_leaf():
            return 1
        return self.count_leaves(node.left) + self.count_leaves(node.right)

    def prune(self, validation: Dataset) -> None:
        """Reduced-error pruning: turn a node into a leaf if that is no worse on the validation set."""
        if self.classes_ is None:
            raise RuntimeError("call train() before prune()")
        # give each validation label the same number it had in training
        position = {label: i for i, label in enumerate(self.classes_)}
        codes = np.array([position[label] for label in validation.y])
        self._prune(self.root, validation.X, codes)

    def _prune(self, node, X, codes):
        """Prune from the bottom up, and return how many validation rows this node gets wrong."""
        if node.is_leaf():
            return np.sum(codes != node.shares.argmax())

        goes_left = X[:, node.feature] <= node.threshold
        wrong_below = (self._prune(node.left, X[goes_left], codes[goes_left])
                       + self._prune(node.right, X[~goes_left], codes[~goes_left]))
        wrong_as_leaf = np.sum(codes != node.shares.argmax())

        # a tie goes to the leaf, because the smaller tree is simpler.
        # So a node that no validation row reaches (0 wrong either way) is also cut.
        if wrong_as_leaf <= wrong_below:
            node.feature = node.threshold = node.left = node.right = None
            return wrong_as_leaf
        return wrong_below
