import pytest

from ml_comparison.classifier import Classifier


def test_cannot_make_a_plain_classifier():
    with pytest.raises(TypeError):
        Classifier()


def test_a_class_with_both_methods_works():
    class AlwaysRed(Classifier):
        def train(self, data):
            pass

        def predict(self, X):
            return ["red" for _ in X]

    model = AlwaysRed()
    assert model.predict([[1, 1], [2, 2]]) == ["red", "red"]


def test_a_class_missing_a_method_fails():
    class Lazy(Classifier):
        def train(self, data):
            pass

    with pytest.raises(TypeError):
        Lazy()