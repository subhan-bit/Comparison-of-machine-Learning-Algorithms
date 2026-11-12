"""k-nearest neighbours: predict a label from the labels of the closest training points."""

import numpy as np

from ml_comparison.checks import check_whole_number
from ml_comparison.classifier import Classifier
from ml_comparison.dataset import Dataset
from ml_comparison.distances import euclidean, manhattan

DISTANCES = {"euclidean": euclidean, "manhattan": manhattan}
TIE_BREAKS = ["random", "nearest", "reduce_k", "weighted"]


def same(a, b):
    """True where a and b are equal apart from rounding errors (0.1 + 0.2 and 0.3 are)."""
    return np.isclose(a, b, rtol=1e-9, atol=1e-12)


def has_tie(scores):
    """For each row of scores, True if two or more classes share the top score."""
    scores = np.asarray(scores)
    top = scores.max(axis=1, keepdims=True)
    return same(scores, top).sum(axis=1) > 1


class KNN(Classifier):
    def __init__(self, k=1, distance="euclidean", tie_break="nearest", seed=None):
        k = check_whole_number("k", k)
        if k < 1:
            raise ValueError("k must be at least 1")
        if distance not in DISTANCES:
            raise ValueError(f"distance must be one of {list(DISTANCES)}, not {distance!r}")
        if tie_break not in TIE_BREAKS:
            raise ValueError(f"tie_break must be one of {TIE_BREAKS}, not {tie_break!r}")
        self.k = k
        self.distance = DISTANCES[distance]
        self.tie_break = tie_break
        self.seed = seed  # only used by the "random" strategy

    def _train(self, data: Dataset) -> None:
        # k-NN does no learning up front: it just remembers the training data
        if self.k > len(data):
            raise ValueError(f"k is {self.k} but there are only {len(data)} training points")
        # keep copies, so changing the original data later cannot change the model
        self.train_X = data.X.copy()
        self.train_codes = data.codes.copy()  # each label as a number: its position in the classes
        self.n_classes = len(data.classes)

    def _neighbours(self, point):
        """The positions and distances of the k closest training points, closest first.

        If several points are tied at the k-th distance they all count, so there can be
        more than k. Otherwise the row order of the data would decide which of them is
        the k-th neighbour.
        """
        distances = self.distance(self.train_X, point)
        kth_distance = np.partition(distances, self.k - 1)[self.k - 1]  # no need to sort them all
        close_enough = np.flatnonzero((distances < kth_distance) | same(distances, kth_distance))
        # closest first; points at the same distance stay in row order
        order = np.argsort(distances[close_enough], kind="stable")
        nearest = close_enough[order]
        return nearest, distances[nearest]

    def _predict_scores(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros((len(X), self.n_classes))
        for row, point in enumerate(X):
            nearest, _ = self._neighbours(point)
            # each neighbour gives one vote to its class
            for i in nearest:
                scores[row, self.train_codes[i]] += 1
            # turn the vote counts into shares of the vote (usually k votes, more if tied)
            scores[row] /= len(nearest)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Like the base class, but a tie for the top score is broken by the chosen strategy."""
        scores = self.predict_scores(X)  # also checks the input
        X = np.asarray(X, dtype=float)
        # A new generator for every call, so with a seed the same data always gives the same
        # answers. (One generator kept for the whole model would give a different answer each time.)
        rng = np.random.default_rng(self.seed)
        winners = []
        for row, point in enumerate(X):
            # the classes that share the highest score
            tied = np.flatnonzero(same(scores[row], scores[row].max()))
            if len(tied) == 1:
                winners.append(tied[0])
            else:
                winners.append(self._break_tie(point, tied, rng))
        assert self.classes_ is not None  # predict_scores has already checked that train() was called
        return self.classes_[winners]

    def _break_tie(self, point, tied, rng):
        """Pick one class from the tied ones, using the chosen strategy. rng is for "random"."""
        nearest, distances = self._neighbours(point)
        codes = self.train_codes[nearest]  # the neighbours' classes, closest first

        if self.tie_break == "random":
            return rng.choice(tied)

        if self.tie_break == "reduce_k":
            # Drop the whole farthest group of equally far voters and vote again, until one class wins.
            # (Dropping just one voter of a group would let the row order decide which of them stays.)
            still_tied = tied
            size = len(codes)
            while True:
                # keep only the voters that are nearer than the farthest group
                size = np.count_nonzero(~same(distances[:size], distances[size - 1]))
                if size == 0:
                    break  # everything left is equally far, so there is no smaller vote to try
                votes = np.bincount(codes[:size], minlength=self.n_classes)
                top = np.flatnonzero(votes == votes.max())
                if len(top) == 1:
                    return top[0]
                still_tied = top
            return still_tied[0]  # still a tie: the class that comes first in the sorted classes wins

        if self.tie_break == "weighted":
            # Dudani's rule: the nearest neighbour has weight 1, the k-th has weight 0
            d_first, d_last = distances[0], distances[-1]
            if same(d_last, d_first):
                weights = np.ones(len(distances))
            else:
                weights = (d_last - distances) / (d_last - d_first)
            # add up the weights for each tied class and keep the biggest
            totals = [weights[codes == c].sum() for c in tied]
            best = [c for c, total in zip(tied, totals, strict=True) if same(total, max(totals))]
            if len(best) == 1:
                return best[0]
            tied = best  # the weights tie too, so fall back to the nearest label

        # "nearest" (and the fallback): the tied class of the closest neighbour
        for code in codes:
            if code in tied:
                return code
