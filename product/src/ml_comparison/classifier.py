"""The rulebook that every algorithm follows."""

from abc import ABC, abstractmethod

import numpy as np

from ml_comparison.dataset import Dataset


class Classifier(ABC):
    """Every algorithm (k-NN, tree, SVM) fills in _train and _predict_scores.

    The public methods train, predict_scores and predict do the checks,
    so each algorithm doesn't have to repeat them.
    """

    classes_ = None      # the labels, in the same order as the score columns
    n_features_ = None   # how many columns the training data had

    def train(self, data: Dataset) -> None:
        # Forget anything from an earlier train() call first
        self.classes_ = None
        self.n_features_ = None
        if len(data) == 0:
            raise ValueError("cannot train on an empty dataset")

        self._train(data)
        self.classes_ = data.classes
        self.n_features_ = data.n_features

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """One row per dot, one column per class. A higher score means more likely."""
        if self.classes_ is None:
            raise RuntimeError("call train() before predict()")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X has {X.shape[1]} features but the model was trained on {self.n_features_}")

        scores = np.asarray(self._predict_scores(X))
        expected = (len(X), len(self.classes_))
        if scores.shape != expected:
            raise ValueError(f"scores must have shape {expected}, got {scores.shape}")
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """For each dot, pick the class with the highest score."""
        best_column = self.predict_scores(X).argmax(axis=1)
        return self.classes_[best_column]

    @abstractmethod
    def _train(self, data: Dataset) -> None:
        """Learn from the data."""

    @abstractmethod
    def _predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Return the scores for each row of X."""
