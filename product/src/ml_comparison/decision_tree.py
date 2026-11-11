"""Decision tree: predict a label by asking yes/no questions about the features."""

import numpy as np

from ml_comparison.classifier import Classifier
from ml_comparison.dataset import Dataset


def class_shares(codes):
    """The share of each class in a group. codes are the labels as numbers."""
    counts = np.bincount(codes)
    counts = counts[counts > 0]  # classes with no rows don't count
    return counts / counts.sum()


def gini_of(p):
    """Gini index from class shares p (the last axis): 1 - sum of p^2."""
    return 1 - (p ** 2).sum(axis=-1)


def entropy_of(p):
    """Entropy from class shares p (the last axis): - sum of p * log2(p)."""
    safe = np.where(p > 0, p, 1)  # log2(1) is 0, so a class with no rows adds nothing
    return -(p * np.log2(safe)).sum(axis=-1)


def misclassification_of(p):
    """Misclassification error from class shares p (the last axis): 1 - the biggest share."""
    return 1 - p.max(axis=-1)


def gini(codes):
    """Gini index of a group of labels. 0 for a pure group."""
    return gini_of(class_shares(codes))


def entropy(codes):
    """Entropy of a group of labels. 0 for a pure group."""
    return entropy_of(class_shares(codes))


def misclassification(codes):
    """Misclassification error of a group of labels. 0 for a pure group."""
    return misclassification_of(class_shares(codes))


# the same three measures, for many groups at once (used by the fast split search)
MEASURES_OF_SHARES = {gini: gini_of, entropy: entropy_of, misclassification: misclassification_of}

# two scores closer than this count as equal, so rounding errors cannot pick a winner
TINY = 1e-12


def split_impurity(left, right, measure):
    """The impurity of a split: each group's impurity, weighted by its size."""
    n = len(left) + len(right)
    return len(left) / n * measure(left) + len(right) / n * measure(right)


def best_split(X, codes, measure):
    """Try every feature and threshold, and return (feature, threshold) of the purest split.

    Returns None if no split is possible (every row has the same values) or if no
    split makes the groups purer than the rows are now.
    """
    X = np.asarray(X, dtype=float)
    codes = np.asarray(codes)
    impurity_of = MEASURES_OF_SHARES[measure]
    n = len(codes)
    one_hot = np.eye(codes.max() + 1)[codes]  # one row per example, a 1 in its class column

    best = None
    best_score = measure(codes)  # a split has to beat the impurity before splitting
    for feature in range(X.shape[1]):
        # sort the rows by this feature once, then sweep along them
        order = np.argsort(X[:, feature], kind="stable")
        values = X[order, feature]
        # left_counts[i] = how many rows of each class are among the first i + 1 rows
        left_counts = np.cumsum(one_hot[order], axis=0)

        # a threshold only makes sense between two different values
        between_different = values[:-1] < values[1:]
        if not between_different.any():
            continue
        n_left = np.arange(1, n)[between_different]
        left = left_counts[:-1][between_different]
        right = left_counts[-1] - left
        thresholds = ((values[:-1] + values[1:]) / 2)[between_different]

        # the impurity of every split on this feature at once, weighted by group size
        scores = (n_left * impurity_of(left / n_left[:, None])
                  + (n - n_left) * impurity_of(right / (n - n_left)[:, None])) / n

        # only a clearly better score wins, so a tie keeps the first split found
        lowest = scores.min()
        if lowest < best_score - TINY:
            first_lowest = np.argmax(scores <= lowest + TINY)
            best = (feature, float(thresholds[first_lowest]))
            best_score = scores[first_lowest]
    return best


MEASURES = {"gini": gini, "entropy": entropy, "misclassification": misclassification}


class Node:
    """One node of the tree. A leaf has no question and no children."""

    def __init__(self, scores):
        self.scores = scores   # the smoothed share of each class among the training rows here
        self.feature = None    # the question is: is X[feature] <= threshold?
        self.threshold = None
        self.left = None       # rows where the answer is yes
        self.right = None      # rows where the answer is no

    def is_leaf(self):
        return self.left is None


def check_whole_number(name, value):
    """Raise a clear error if a setting is not a whole number."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a whole number, not {value!r}")


class DecisionTree(Classifier):
    def __init__(self, measure="gini", max_depth=None, min_samples_split=2):
        if measure not in MEASURES:
            raise ValueError(f"measure must be one of {list(MEASURES)}, not {measure!r}")
        if max_depth is not None:
            check_whole_number("max_depth", max_depth)
        check_whole_number("min_samples_split", min_samples_split)
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
        # Smoothed share: (count + 1) / (rows + classes). A leaf of 1 row is then
        # less sure than a leaf of 50 rows, so a full tree gives more than 0 and 1.
        counts = np.bincount(codes, minlength=self.n_classes)
        node = Node((counts + 1) / (len(codes) + self.n_classes))
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
        return np.array([self._leaf_for(point).scores for point in X])

    def count_leaves(self, node=None):
        if node is None:
            node = self.root
        if node.is_leaf():
            return 1
        return self.count_leaves(node.left) + self.count_leaves(node.right)

    def prune(self, validation: Dataset) -> None:
        """Reduced-error pruning: turn a node into a leaf if that is no worse on the validation set."""
        if self.classes_ is None:
            raise RuntimeError("call train() before prune()")
        if len(validation) == 0:
            raise ValueError("the validation set is empty, so there is nothing to prune with")
        if validation.n_features != self.n_features_:
            raise ValueError(f"validation has {validation.n_features} features "
                             f"but the model was trained on {self.n_features_}")

        # give each validation label the same number it had in training
        position = {label: i for i, label in enumerate(self.classes_)}
        codes = []
        for label in validation.y:
            if label not in position:
                raise ValueError(f"validation label {label!r} was not in the training data")
            codes.append(position[label])
        self._prune(self.root, validation.X, np.array(codes, dtype=int))

    def _prune(self, node, X, codes):
        """Prune from the bottom up, and return how many validation rows this node gets wrong."""
        if node.is_leaf():
            return np.sum(codes != node.scores.argmax())
        if len(codes) == 0:
            return 0  # no validation row gets here, so there is no evidence to prune it

        goes_left = X[:, node.feature] <= node.threshold
        wrong_below = (self._prune(node.left, X[goes_left], codes[goes_left])
                       + self._prune(node.right, X[~goes_left], codes[~goes_left]))
        wrong_as_leaf = np.sum(codes != node.scores.argmax())

        # a tie goes to the leaf, because the smaller tree is simpler
        if wrong_as_leaf <= wrong_below:
            node.feature = node.threshold = node.left = node.right = None
            return wrong_as_leaf
        return wrong_below
