"""k-nearest neighbours: predict a label from the labels of the closest training points."""

import numpy as np

from ml_comparison.classifier import Classifier
from ml_comparison.dataset import Dataset
from ml_comparison.distances import euclidean, manhattan

DISTANCES = {"euclidean": euclidean, "manhattan": manhattan}


class KNN(Classifier):
    def __init__(self, k=1, distance="euclidean"):
        if k < 1:
            raise ValueError("k must be at least 1")
        if distance not in DISTANCES:
            raise ValueError(f"distance must be one of {list(DISTANCES)}, not {distance!r}")
        self.k = k
        self.distance = DISTANCES[distance]

    def _train(self, data: Dataset) -> None:
        # k-NN does no learning up front: it just remembers the training data
        if self.k > len(data):
            raise ValueError(f"k is {self.k} but there are only {len(data)} training points")
        self.train_X = data.X
        self.train_codes = data.codes  # each label as a number: its position in the classes
        self.n_classes = len(data.classes)

    def _predict_scores(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros((len(X), self.n_classes))
        for row, point in enumerate(X):
            # 1. distance from this point to every training point
            distances = self.distance(self.train_X, point)
            # 2. the positions of the k closest training points
            nearest = np.argsort(distances, kind="stable")[:self.k]
            # 3. each of them gives one vote to its class
            for i in nearest:
                scores[row, self.train_codes[i]] += 1
        # turn the vote counts into shares of the vote
        return scores / self.k
