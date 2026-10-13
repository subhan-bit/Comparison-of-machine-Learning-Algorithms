"""k-nearest neighbours: predict a label from the labels of the closest training points."""

import numpy as np

from ml_comparison.classifier import Classifier
from ml_comparison.dataset import Dataset
from ml_comparison.distances import euclidean, manhattan

DISTANCES = {"euclidean": euclidean, "manhattan": manhattan}
TIE_BREAKS = ["random", "nearest", "reduce_k", "weighted"]


class KNN(Classifier):
    def __init__(self, k=1, distance="euclidean", tie_break="nearest", seed=None):
        if k < 1:
            raise ValueError("k must be at least 1")
        if distance not in DISTANCES:
            raise ValueError(f"distance must be one of {list(DISTANCES)}, not {distance!r}")
        if tie_break not in TIE_BREAKS:
            raise ValueError(f"tie_break must be one of {TIE_BREAKS}, not {tie_break!r}")
        self.k = k
        self.distance = DISTANCES[distance]
        self.tie_break = tie_break
        self.rng = np.random.default_rng(seed)  # only used by the "random" strategy

    def _train(self, data: Dataset) -> None:
        # k-NN does no learning up front: it just remembers the training data
        if self.k > len(data):
            raise ValueError(f"k is {self.k} but there are only {len(data)} training points")
        self.train_X = data.X
        self.train_codes = data.codes  # each label as a number: its position in the classes
        self.n_classes = len(data.classes)

    def _neighbours(self, point):
        """The positions and distances of the k closest training points, closest first."""
        distances = self.distance(self.train_X, point)
        nearest = np.argsort(distances, kind="stable")[:self.k]
        return nearest, distances[nearest]

    def _predict_scores(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros((len(X), self.n_classes))
        for row, point in enumerate(X):
            nearest, _ = self._neighbours(point)
            # each of the k neighbours gives one vote to its class
            for i in nearest:
                scores[row, self.train_codes[i]] += 1
        # turn the vote counts into shares of the vote
        return scores / self.k

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Like the base class, but a tie for the top score is broken by the chosen strategy."""
        scores = self.predict_scores(X)  # also checks the input
        X = np.asarray(X, dtype=float)
        winners = []
        for row, point in enumerate(X):
            # the classes that share the highest score
            tied = np.flatnonzero(scores[row] == scores[row].max())
            if len(tied) == 1:
                winners.append(tied[0])
            else:
                winners.append(self._break_tie(point, tied))
        return self.classes_[winners]

    def _break_tie(self, point, tied):
        """Pick one class from the tied ones, using the chosen strategy."""
        nearest, distances = self._neighbours(point)
        codes = self.train_codes[nearest]  # the neighbours' classes, closest first

        if self.tie_break == "random":
            return self.rng.choice(tied)

        if self.tie_break == "reduce_k":
            # drop the farthest neighbour and vote again, until one class wins
            for smaller_k in range(self.k - 1, 0, -1):
                votes = np.bincount(codes[:smaller_k], minlength=self.n_classes)
                top = np.flatnonzero(votes == votes.max())
                if len(top) == 1:
                    return top[0]

        if self.tie_break == "weighted":
            # Dudani's rule: the nearest neighbour has weight 1, the k-th has weight 0
            d_first, d_last = distances[0], distances[-1]
            if d_last == d_first:
                weights = np.ones(len(distances))
            else:
                weights = (d_last - distances) / (d_last - d_first)
            # add up the weights for each tied class and keep the biggest
            totals = [weights[codes == c].sum() for c in tied]
            best = [c for c, total in zip(tied, totals) if total == max(totals)]
            if len(best) == 1:
                return best[0]
            tied = best  # the weights tie too, so fall back to the nearest label

        # "nearest" (and the fallback): the tied class of the closest neighbour
        for code in codes:
            if code in tied:
                return code
