import numpy as np
import pytest

from ml_comparison.dataset import Dataset
from ml_comparison.evaluation import accuracy, confusion_matrix, stratified_k_fold, train_test_split


def rare_class_data(n_rare=10, n_common=190):
    """A data set with a rare class. The number in each row says which row it is."""
    X = np.arange(n_rare + n_common, dtype=float).reshape(-1, 1)
    return Dataset(X, ["rare"] * n_rare + ["common"] * n_common, positive_label="rare")


def count(data, label):
    return int(np.sum(data.y == label))


def test_a_split_uses_every_row_once():
    data = rare_class_data()
    train, test = train_test_split(data, test_fraction=0.3, seed=1)
    assert (len(train), len(test)) == (140, 60)
    assert sorted(train.X.ravel().tolist() + test.X.ravel().tolist()) == data.X.ravel().tolist()


def test_the_same_seed_gives_the_same_split_and_another_seed_a_different_one():
    data = rare_class_data()
    first = train_test_split(data, seed=7)
    again = train_test_split(data, seed=7)
    other = train_test_split(data, seed=8)
    assert first[1].X.tolist() == again[1].X.tolist()
    assert first[0].X.tolist() == again[0].X.tolist()
    assert first[1].X.tolist() != other[1].X.tolist()


def test_a_stratified_split_keeps_the_share_of_each_class():
    data = rare_class_data(10, 190)  # 5% of the rows are rare
    train, test = train_test_split(data, test_fraction=0.3, seed=1, stratify=True)
    assert (count(test, "rare"), count(test, "common")) == (3, 57)  # 30% of each class
    assert (count(train, "rare"), count(train, "common")) == (7, 133)


def test_a_stratified_split_puts_a_rare_class_in_both_parts():
    data = rare_class_data(2, 198)
    train, test = train_test_split(data, test_fraction=0.1, seed=1, stratify=True)
    assert (count(train, "rare"), count(test, "rare")) == (1, 1)  # 10% of 2 rows rounds to 0, but each part keeps one


def test_a_class_with_one_row_stays_in_the_training_part():
    data = Dataset(np.arange(11, dtype=float).reshape(-1, 1), ["one"] + ["many"] * 10)
    train, test = train_test_split(data, test_fraction=0.3, seed=1, stratify=True)
    assert (count(train, "one"), count(test, "one")) == (1, 0)


def test_both_parts_keep_the_full_class_list_and_the_positive_class():
    # with one row of a class, the class can land on one side only, but it is still a class in both parts
    data = Dataset(np.arange(11, dtype=float).reshape(-1, 1), ["one"] + ["many"] * 10, positive_label="one")
    for seed in range(6):
        for part in train_test_split(data, seed=seed):
            assert part.classes.tolist() == ["many", "one"]
            assert part.positive_label == "one"


def test_bad_split_settings_are_rejected():
    data = rare_class_data()
    for bad in (0, 1, -0.1, 1.5):
        with pytest.raises(ValueError, match="test_fraction"):
            train_test_split(data, test_fraction=bad)
    with pytest.raises(ValueError, match="too small"):
        train_test_split(Dataset([[1]], ["a"]))


def test_every_row_is_in_exactly_one_test_fold():
    data = rare_class_data(13, 187)
    seen = []
    for train, test in stratified_k_fold(data, k=5, seed=1):
        assert len(train) + len(test) == len(data)
        seen += test.X.ravel().tolist()
    assert sorted(seen) == data.X.ravel().tolist()


def test_each_fold_keeps_each_class_within_one_row_of_its_share():
    data = rare_class_data(13, 187)  # 6.5% rare, and 13 does not divide by 5
    for train, test in stratified_k_fold(data, k=5, seed=1):
        assert abs(count(test, "rare") - 13 / 5) < 1  # 2 or 3 rare rows in every test fold
        assert abs(count(test, "common") - 187 / 5) < 1
        assert abs(len(test) - 200 / 5) <= 1  # and the folds are the same size, to within a row


def test_the_same_seed_gives_the_same_folds():
    data = rare_class_data()

    def test_parts(seed):
        return [test.X.ravel().tolist() for _, test in stratified_k_fold(data, k=4, seed=seed)]

    assert test_parts(3) == test_parts(3)
    assert test_parts(3) != test_parts(4)


def test_a_class_with_fewer_rows_than_folds_is_still_a_class_in_every_fold():
    data = Dataset(np.arange(21, dtype=float).reshape(-1, 1), ["tiny"] + ["many"] * 20)  # one "tiny" row, 5 folds
    folds = list(stratified_k_fold(data, k=5, seed=0))
    without_tiny = [train for train, _ in folds if count(train, "tiny") == 0]
    assert len(without_tiny) == 1  # in one fold the only "tiny" row is in the test part
    assert without_tiny[0].classes.tolist() == ["many", "tiny"]  # but the class list is still complete


def test_bad_fold_settings_are_rejected():
    data = rare_class_data()
    for bad in (1, 0, -2):
        with pytest.raises(ValueError, match="k must be at least 2"):
            stratified_k_fold(data, k=bad)
    with pytest.raises(ValueError, match="more folds than rows"):
        stratified_k_fold(Dataset([[1], [2], [3]], ["a", "b", "a"]), k=5)
    with pytest.raises(TypeError, match="k must be a whole number"):
        stratified_k_fold(data, k=2.5)


CLASSES = ["fraud", "genuine"]
TRUE = ["fraud", "fraud", "fraud", "genuine", "genuine", "genuine", "genuine", "genuine"]
PREDICTED = ["fraud", "fraud", "genuine", "fraud", "genuine", "genuine", "genuine", "genuine"]


def test_the_confusion_matrix_counts_true_classes_against_predicted_classes():
    # rows are the true class and columns the predicted class, in the order of the class list
    matrix = confusion_matrix(TRUE, PREDICTED, CLASSES)
    assert matrix.tolist() == [[2, 1],   # 3 frauds: 2 caught, 1 missed
                               [1, 4]]   # 5 genuine: 1 false alarm, 4 right


def test_a_class_with_no_rows_still_has_a_row_and_a_column():
    matrix = confusion_matrix(["a", "a", "b"], ["a", "c", "b"], ["a", "b", "c"])
    assert matrix.tolist() == [[1, 0, 1], [0, 1, 0], [0, 0, 0]]


def test_accuracy_is_the_share_of_right_answers():
    assert accuracy(TRUE, PREDICTED) == 6 / 8
    assert accuracy(["a"], ["a"]) == 1.0


def test_bad_input_to_the_confusion_matrix_is_rejected():
    with pytest.raises(ValueError, match="'q' is not one of the classes"):
        confusion_matrix(["a"], ["q"], ["a", "b"])
    with pytest.raises(ValueError, match="same length"):
        confusion_matrix(["a", "b"], ["a"], ["a", "b"])


def test_bad_input_to_accuracy_is_rejected():
    with pytest.raises(ValueError, match="same length"):
        accuracy(["a", "b"], ["a"])
    with pytest.raises(ValueError, match="no rows"):
        accuracy([], [])
