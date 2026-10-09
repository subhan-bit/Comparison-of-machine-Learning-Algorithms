import numpy as np

from ml_comparison.dataset import Dataset


def test_dataset_keeps_positions_and_labels():
    data = Dataset([[1, 1], [4, 4]], ["red", "blue"])
    assert data.X.shape == (2, 2)
    assert list(data.y) == ["red", "blue"]


def test_len_counts_the_dots():
    data = Dataset([[1, 1], [2, 1], [4, 4]], ["red", "red", "blue"])
    assert len(data) == 3


def test_x_is_a_numpy_array_of_floats():
    data = Dataset([[1, 1]], ["red"])
    assert isinstance(data.X, np.ndarray)
    assert data.X.dtype == float