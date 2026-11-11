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

        lower, upper = values[:-1], values[1:]
        with np.errstate(over="ignore"):  # a huge gap overflows to inf, and is skipped below
            midpoints = lower + (upper - lower) / 2  # (lower + upper) / 2 can overflow

        # a threshold only makes sense between two different values
        usable = lower < upper
        if not usable.any():
            continue
        n_left = np.arange(1, n)[usable]
        left = left_counts[:-1][usable]
        right = left_counts[-1] - left
        # Two values one float step apart can have a midpoint that rounds up to the upper
        # value, which would send every row left. Then the lower value itself separates them.
        thresholds = np.where(midpoints < upper, midpoints, lower)[usable]

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
    """A decision tree that follows the Classifier interface.

    measure is "gini", "entropy" or "misclassification". max_depth (None means no
    limit) and min_samples_split stop the tree growing. The scores are the smoothed
    class shares in the leaf. prune() cuts the tree back using a validation set.
    """

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
        self.root = self._new_node(data.codes)

        # Nodes still to split, as (node, its rows, their labels, its depth). A list used
        # as a stack, not recursion, so a deep tree cannot hit Python's recursion limit.
        to_split = [(self.root, data.X, data.codes, 0)]
        while to_split:
            node, X, codes, depth = to_split.pop()
            if self._should_stop(codes, depth):
                continue
            split = best_split(X, codes, self.measure)
            if split is None:  # every row is the same, or no split helps
                continue

            node.feature, node.threshold = split
            goes_left = X[:, node.feature] <= node.threshold
            node.left = self._new_node(codes[goes_left])
            node.right = self._new_node(codes[~goes_left])
            to_split.append((node.left, X[goes_left], codes[goes_left], depth + 1))
            to_split.append((node.right, X[~goes_left], codes[~goes_left], depth + 1))

    def _new_node(self, codes):
        """A leaf for these rows. Its scores are the smoothed share of each class."""
        # Smoothed share: (count + 1) / (rows + classes). A leaf of 1 row is then
        # less sure than a leaf of 50 rows, so a full tree gives more than 0 and 1.
        counts = np.bincount(codes, minlength=self.n_classes)
        return Node((counts + 1) / (len(codes) + self.n_classes))

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

    def count_leaves(self):
        """How many leaves the tree has (a smaller tree is a simpler one)."""
        leaves = 0
        to_visit = [self.root]
        while to_visit:
            node = to_visit.pop()
            if node.is_leaf():
                leaves += 1
            else:
                to_visit.extend([node.left, node.right])
        return leaves

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
        self._prune(validation.X, np.array(codes, dtype=int))

    def _prune(self, X, codes):
        """Prune from the bottom up, using a stack instead of recursion."""
        # Going down: find which validation rows reach each node. A node is added to the
        # list before its children, so going through the list backwards does children first.
        reached = []  # (node, labels of the validation rows that reach it)
        to_visit = [(self.root, X, codes)]
        while to_visit:
            node, X, codes = to_visit.pop()
            reached.append((node, codes))
            if node.is_leaf() or len(codes) == 0:
                continue  # no validation row gets below here, so there is no evidence to prune it
            goes_left = X[:, node.feature] <= node.threshold
            to_visit.append((node.left, X[goes_left], codes[goes_left]))
            to_visit.append((node.right, X[~goes_left], codes[~goes_left]))

        # Going up: wrong[node] is how many validation rows this node gets wrong
        wrong = {}
        for node, codes in reversed(reached):
            wrong_as_leaf = np.sum(codes != node.scores.argmax())
            if node.is_leaf() or len(codes) == 0:
                wrong[node] = wrong_as_leaf
                continue
            wrong_below = wrong[node.left] + wrong[node.right]
            # a tie goes to the leaf, because the smaller tree is simpler
            if wrong_as_leaf <= wrong_below:
                node.feature = node.threshold = node.left = node.right = None
                wrong[node] = wrong_as_leaf
            else:
                wrong[node] = wrong_below
