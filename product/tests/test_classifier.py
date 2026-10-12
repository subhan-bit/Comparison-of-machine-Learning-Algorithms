import numpy as np
import pytest

from ml_comparison.classifier import Classifier
from ml_comparison.dataset import Dataset


class AlwaysFirst(Classifier):
    """A fake algorithm for testing: gives the first class the top score every time."""

    def _train(self, data):
        pass

    def _predict_scores(self, X):
        scores = np.zeros((len(X), len(self.classes_)))
        scores[:, 0] = 1
        return scores


DATA = Dataset([[1, 1], [2, 2]], ["red", "blue"])


def test_predict_picks_the_highest_score():
    model = AlwaysFirst()
    model.train(DATA)
    # classes are sorted, so "blue" comes first
    assert model.predict_scores([[1, 1]]).tolist() == [[1.0, 0.0]]
    assert model.predict([[1, 1], [5, 5]]).tolist() == ["blue", "blue"]


def test_training_again_forgets_the_old_classes():
    model = AlwaysFirst()
    model.train(DATA)
    model.train(Dataset([[1], [2]], ["x", "y"]))
    assert model.classes_.tolist() == ["x", "y"]


def test_cannot_create_the_base_class():
    with pytest.raises(TypeError):
        Classifier()


def test_wrong_use_is_rejected():
    model = AlwaysFirst()
    with pytest.raises(RuntimeError):
        model.predict([[1, 1]])        # not trained yet
    with pytest.raises(ValueError):
        model.train(Dataset([], []))  # nothing to learn from
    model.train(DATA)
    with pytest.raises(ValueError, match="2-D"):
        model.predict([1, 1])         # not a table
    with pytest.raises(ValueError, match="features"):
        model.predict([[1, 2, 3]])    # wrong number of columns


def test_wrong_score_shape_is_rejected():
    class Broken(AlwaysFirst):
        def _predict_scores(self, X):
            return np.zeros((len(X), 5))

    model = Broken()
    model.train(DATA)
    with pytest.raises(ValueError, match="scores"):
        model.predict([[1, 1]])
