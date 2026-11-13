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


def test_values_that_are_not_numbers_give_a_clear_error_not_a_raw_type_error():
    for bad in ({}, {"a": 1}, object(), 1j, {1, 2}):
        with pytest.raises(ValueError, match="rectangular table of numbers"):
            Dataset([[bad]], [0])
    with pytest.raises(ValueError, match="rectangular table of numbers"):
        Dataset([{}], [0])  # a dict where a row should be


def test_blank_labels_are_rejected():
    for blank in ("", "   ", None):
        with pytest.raises(ValueError, match="label 1 is blank"):
            Dataset([[1], [2]], ["red", blank])
    assert Dataset([[1]], [0]).y.tolist() == [0]  # the number 0 is a fine label


def test_positive_label_must_be_one_of_the_labels():
    with pytest.raises(ValueError):
        Dataset([[1], [2]], ["ham", "spam"], positive_label="fraud")
    with pytest.raises(ValueError):
        Dataset([[1], [2]], ["ham", "spam"], positive_label=1)


def test_the_bad_table_error_does_not_also_show_numpys_own_error():
    with pytest.raises(ValueError, match="rectangular table") as error:
        Dataset([[1, "a"]], [0])
    # "raise ... from None": the traceback shows only our message, not numpy's as well
    assert error.value.__suppress_context__


def test_a_dataset_can_be_given_its_class_list():
    # a split of a bigger data set keeps the classes of the whole data set, even one with no rows in the split
    data = Dataset([[1], [2]], ["a", "c"], classes=["a", "b", "c"])
    assert data.classes.tolist() == ["a", "b", "c"]
    assert data.codes.tolist() == [0, 2]  # "c" is the third class, though "b" has no rows
    # a class with no rows can still be the positive one
    assert Dataset([[1]], ["a"], positive_label="b", classes=["a", "b"]).positive_label == "b"


def test_labels_must_be_in_the_given_class_list():
    with pytest.raises(ValueError, match="'z' is not one of the classes"):
        Dataset([[1]], ["z"], classes=["a", "b"])
    with pytest.raises(ValueError, match="repeat"):
        Dataset([[1]], ["a"], classes=["a", "a"])
