import numpy as np
import pytest

from ml_comparison.dataset import Dataset
from ml_comparison.evaluation import train_test_split


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
