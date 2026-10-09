"""The rulebook that every algorithm follows."""

from abc import ABC, abstractmethod


class Classifier(ABC):
    @abstractmethod
    def train(self, data):
        """Learn from a Dataset of dots with known labels."""

    @abstractmethod
    def predict(self, X):
        """Return one guessed label for each row of X."""