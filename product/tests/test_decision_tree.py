import pytest

from ml_comparison.decision_tree import (
    best_split,
    entropy,
    gini,
    misclassification,
    split_impurity,
)

# Labels as numbers, like Dataset.codes. 6 red (0) and 4 blue (1), as in
# documents/decision_trees/decision_trees.md
SIX_RED_FOUR_BLUE = [0] * 6 + [1] * 4


def test_a_pure_group_has_no_impurity():
    for measure in [gini, entropy, misclassification]:
        assert measure([1, 1, 1]) == 0


def test_impurity_of_six_red_four_blue_like_in_the_notes():
    assert gini(SIX_RED_FOUR_BLUE) == pytest.approx(0.48)
    assert entropy(SIX_RED_FOUR_BLUE) == pytest.approx(0.971, abs=0.001)
    assert misclassification(SIX_RED_FOUR_BLUE) == pytest.approx(0.4)


def test_a_half_and_half_group_is_the_most_impure():
    assert gini([0, 1]) == 0.5
    assert entropy([0, 1]) == 1
    assert misclassification([0, 1]) == 0.5


def test_split_impurity_weights_each_group_by_its_size():
    # the split from the notes: left = 4 red, right = 2 red and 4 blue
    left, right = [0] * 4, [0] * 2 + [1] * 4
    assert split_impurity(left, right, gini) == pytest.approx(0.267, abs=0.001)


def test_misclassification_cannot_tell_these_splits_apart_but_gini_can():
    # 400 of each class, from the notes (Hastie et al., section 9.2)
    split_a = ([0] * 300 + [1] * 100, [0] * 100 + [1] * 300)
    split_b = ([0] * 200 + [1] * 400, [0] * 200)
    assert split_impurity(*split_a, misclassification) == pytest.approx(0.25)
    assert split_impurity(*split_b, misclassification) == pytest.approx(0.25)
    assert split_impurity(*split_a, gini) == pytest.approx(0.375)
    assert split_impurity(*split_b, gini) == pytest.approx(0.333, abs=0.001)


def test_best_split_is_halfway_between_the_two_groups():
    X = [[1], [2], [3], [10], [11]]
    codes = [0, 0, 0, 1, 1]
    assert best_split(X, codes, gini) == (0, 6.5)  # feature 0, threshold 6.5


def test_best_split_picks_the_useful_feature():
    # feature 0 is the same for every row, so only feature 1 can split
    X = [[5, 1], [5, 2], [5, 8], [5, 9]]
    codes = [0, 0, 1, 1]
    assert best_split(X, codes, entropy) == (1, 5.0)


def test_no_split_when_all_rows_are_the_same():
    assert best_split([[1, 2], [1, 2]], [0, 1], gini) is None
