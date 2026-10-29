import pytest

from ml_comparison.decision_tree import entropy, gini, misclassification

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
