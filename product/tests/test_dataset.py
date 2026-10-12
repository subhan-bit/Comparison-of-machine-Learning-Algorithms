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


def test_bad_data_is_rejected():
    with pytest.raises(ValueError):
        Dataset([[1, 2]], ["a", "b"])         # 1 row but 2 labels
    with pytest.raises(ValueError):
        Dataset([[1, 2], [3]], ["a", "b"])    # rows of different lengths
    with pytest.raises(ValueError):
        Dataset([["a", "b"]], ["red"])        # text instead of numbers
    with pytest.raises(ValueError):
        Dataset([1, 2, 3], ["a", "b", "c"])   # a flat list, not a table
    with pytest.raises(ValueError):
        Dataset([[1.0, np.inf]], ["a"])       # infinity
    with pytest.raises(ValueError):
        Dataset([[1], [2]], [["a"], ["b"]])   # labels inside lists


def test_positive_label_must_be_one_of_the_labels():
    with pytest.raises(ValueError):
        Dataset([[1], [2]], ["ham", "spam"], positive_label="fraud")
    with pytest.raises(ValueError):
        Dataset([[1], [2]], ["ham", "spam"], positive_label=1)
