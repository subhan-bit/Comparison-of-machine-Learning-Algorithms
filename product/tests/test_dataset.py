import numpy as np
import pytest

from ml_comparison.dataset import Dataset


def test_keeps_x_and_y():
    data = Dataset([[1, 1], [4, 4]], ["red", "blue"])
    assert data.X.tolist() == [[1.0, 1.0], [4.0, 4.0]]
    assert list(data.y) == ["red", "blue"]
    assert len(data) == 2
    assert data.n_features == 2
    assert data.positive_label is None


def test_labels_are_numbered_in_sorted_order():
    data = Dataset([[1], [2], [3]], ["spam", "ham", "spam"], positive_label="spam")
    assert list(data.classes) == ["ham", "spam"]
    assert list(data.codes) == [1, 0, 1]
    assert data.positive_label == "spam"


def test_empty_dataset():
    assert Dataset([], []).X.shape == (0, 0)
    assert Dataset(np.zeros((0, 3)), []).n_features == 3


@pytest.mark.parametrize("X, y, positive, message", [
    ([[1, 2]], ["a", "b"], None, "1 rows but y has 2 labels"),
    ([[1, 2], [3]], ["a", "b"], None, "rectangular"),
    ([["a", "b"]], ["red"], None, "rectangular"),
    ([1, 2, 3], ["a", "b", "c"], None, "2-D"),
    ([[1.0, np.inf]], ["a"], None, "finite"),
    ([[1], [2]], [["a"], ["b"]], None, "y must be 1-D"),
    ([[1], [2]], ["ham", "spam"], "fraud", "not one of the labels"),
    ([[1], [2]], ["ham", "spam"], 1, "not one of the labels"),
])
def test_bad_data_is_rejected(X, y, positive, message):
    with pytest.raises(ValueError, match=message):
        Dataset(X, y, positive_label=positive)
