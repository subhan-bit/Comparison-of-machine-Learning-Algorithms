import numpy as np
import pytest

from ml_comparison.dataset import Dataset
from ml_comparison.evaluation import (
    accuracy,
    auc,
    confusion_matrix,
    roc_auc,
    roc_curve,
    standardise,
    stratified_k_fold,
    train_test_split,
    true_and_false_positive_rates,
)


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


def test_true_and_false_positive_rates():
    matrix = confusion_matrix(TRUE, PREDICTED, CLASSES)
    # fraud is positive: 2 of 3 frauds caught, and 1 of 5 genuine rows flagged
    assert true_and_false_positive_rates(matrix, CLASSES, "fraud") == pytest.approx((2 / 3, 1 / 5))
    # looking from the other side, genuine is positive: 4 of 5 found, and 1 of 3 frauds called genuine
    assert true_and_false_positive_rates(matrix, CLASSES, "genuine") == pytest.approx((4 / 5, 1 / 3))


def test_with_more_than_two_classes_the_other_classes_count_together_as_negative():
    matrix = np.array([[5, 1, 0],    # true a
                       [2, 3, 1],    # true b
                       [0, 2, 6]])   # true c
    # positive is b: TP = 3, FN = 2 + 1 = 3, FP = 1 + 2 = 3, and TN is the rest: 5 + 0 + 0 + 6 = 11
    assert true_and_false_positive_rates(matrix, ["a", "b", "c"], "b") == pytest.approx((3 / 6, 3 / 14))


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


def test_bad_input_to_the_rates_is_rejected():
    matrix = np.array([[0, 0], [1, 2]])  # there are no true "a" rows
    with pytest.raises(ValueError, match="no rows of 'a'"):
        true_and_false_positive_rates(matrix, ["a", "b"], "a")
    matrix = np.array([[3, 1], [0, 0]])  # every row is an "a"
    with pytest.raises(ValueError, match="no rows of the other classes"):
        true_and_false_positive_rates(matrix, ["a", "b"], "a")
    with pytest.raises(ValueError, match="'z' is not one of the classes"):
        true_and_false_positive_rates(matrix, ["a", "b"], "z")


def test_standardising_uses_the_training_rows_only():
    train = Dataset([[1, 10], [3, 10], [5, 10]], ["a", "b", "a"])
    new_train, new_far = standardise(train, Dataset([[100, 10]], ["a"]))
    again_train, _ = standardise(train, Dataset([[3, 10]], ["a"]))
    # the test rows do not change the mean and spread, so the training part comes out the same
    assert new_train.X.tolist() == again_train.X.tolist()
    # the training part has mean 0 and spread 1 (a feature that never changes is just 0)
    assert new_train.X.mean(axis=0) == pytest.approx([0, 0])
    assert new_train.X[:, 0].std() == pytest.approx(1)
    assert new_train.X[:, 1].tolist() == [0, 0, 0]
    # the test row is changed with the training mean (3) and spread (the square root of 8/3)
    assert new_far.X[0, 0] == pytest.approx((100 - 3) / np.sqrt(8 / 3))
    assert new_far.X[0, 1] == 0


def test_standardising_keeps_the_labels_the_class_list_and_the_positive_class():
    classes = ["a", "b", "c"]
    train = Dataset([[1], [2]], ["a", "b"], positive_label="b", classes=classes)
    test = Dataset([[3]], ["a"], positive_label="b", classes=classes)
    new_train, new_test = standardise(train, test)
    assert (new_train.y.tolist(), new_test.y.tolist()) == (["a", "b"], ["a"])
    assert new_train.classes.tolist() == classes and new_test.classes.tolist() == classes
    assert (new_train.positive_label, new_test.positive_label) == ("b", "b")


def test_standardising_does_not_change_the_original_data_sets():
    train, test = Dataset([[1.0], [3.0]], ["a", "b"]), Dataset([[2.0]], ["a"])
    standardise(train, test)
    assert (train.X.tolist(), test.X.tolist()) == ([[1.0], [3.0]], [[2.0]])


def test_standardising_needs_training_rows():
    with pytest.raises(ValueError, match="no training rows"):
        standardise(Dataset([], []), Dataset([[1]], ["a"]))


# the worked example in documents/research/class_scores_and_roc.md: five transactions and their fraud scores
WORKED_TRUE = ["fraud", "genuine", "fraud", "genuine", "genuine"]
WORKED_SCORES = [0.9, 0.8, 0.7, 0.4, 0.2]


def test_the_roc_curve_matches_the_worked_example_in_the_notes():
    fpr, tpr, thresholds = roc_curve(WORKED_TRUE, WORKED_SCORES, "fraud")
    # the table in the notes: at each threshold, the share of frauds caught and of genuine rows flagged
    assert thresholds.tolist() == [np.inf, 0.9, 0.8, 0.7, 0.4, 0.2]  # the first point is "flag nothing"
    assert tpr.tolist() == pytest.approx([0, 0.5, 0.5, 1, 1, 1])
    assert fpr.tolist() == pytest.approx([0, 0, 1 / 3, 1 / 3, 2 / 3, 1])


def test_rows_with_the_same_score_make_one_step_of_the_roc_curve():
    # two frauds and two genuine rows all score 0.5: one threshold flags all four at once
    fpr, tpr, thresholds = roc_curve(["a", "a", "b", "b"], [0.5, 0.5, 0.5, 0.5], "a")
    assert thresholds.tolist() == [np.inf, 0.5]
    assert (fpr.tolist(), tpr.tolist()) == ([0, 1], [0, 1])  # straight from (0, 0) to (1, 1)


def test_the_roc_curve_does_not_depend_on_the_row_order():
    true = np.array(["a", "b", "a", "b", "b", "a", "b"])
    scores = np.array([0.8, 0.8, 0.6, 0.3, 0.6, 0.1, 0.3])  # several ties
    order = np.random.default_rng(3).permutation(len(true))
    first, second = roc_curve(true, scores, "a"), roc_curve(true[order], scores[order], "a")
    for one, other in zip(first, second, strict=True):
        assert one.tolist() == other.tolist()


def test_bad_input_to_the_roc_curve_is_rejected():
    with pytest.raises(ValueError, match="same length"):
        roc_curve(["a", "b"], [0.1], "a")
    with pytest.raises(ValueError, match="no rows of 'a'"):
        roc_curve(["b", "b"], [0.1, 0.2], "a")
    with pytest.raises(ValueError, match="no rows of the other classes"):
        roc_curve(["a", "a"], [0.1, 0.2], "a")
    for bad in (float("nan"), float("inf"), -float("inf")):  # a NaN would quietly give a wrong AUC
        with pytest.raises(ValueError, match="finite"):
            roc_curve(["a", "b"], [0.5, bad], "a")
    # one column of scores is needed, not the whole table that predict_scores gives
    with pytest.raises(ValueError, match="scores must be 1-D"):
        roc_curve(["a", "b"], [[0.9, 0.1], [0.2, 0.8]], "a")
    with pytest.raises(ValueError, match="true must be 1-D"):
        roc_curve([["a"], ["b"]], [0.9, 0.2], "a")


def test_the_auc_of_the_worked_example_is_five_sixths():
    # of the 6 (fraud, genuine) pairs, the fraud has the higher score in 5
    fpr, tpr, _ = roc_curve(WORKED_TRUE, WORKED_SCORES, "fraud")
    assert auc(fpr, tpr) == pytest.approx(5 / 6)
    assert roc_auc(WORKED_TRUE, WORKED_SCORES, "fraud") == pytest.approx(5 / 6)


def test_the_auc_is_1_for_a_perfect_scorer_and_0_for_a_reversed_one():
    true = ["a", "a", "b", "b"]
    assert roc_auc(true, [0.9, 0.8, 0.2, 0.1], "a") == pytest.approx(1)
    assert roc_auc(true, [0.1, 0.2, 0.8, 0.9], "a") == pytest.approx(0)


def test_tied_scores_count_as_half_a_win():
    # the positives score 0.8 and 0.5 and the negatives 0.5 and 0.2. Of the 4 (positive, negative) pairs,
    # the positive wins 3 and ties 1, so the AUC is (3 + 0.5) / 4
    assert roc_auc(["a", "a", "b", "b"], [0.8, 0.5, 0.5, 0.2], "a") == pytest.approx(0.875)
    # a scorer that gives every row the same score is no better than guessing
    assert roc_auc(["a", "a", "b", "b", "b"], [0.3] * 5, "a") == pytest.approx(0.5)


def test_the_auc_is_the_share_of_pairs_that_the_positive_wins_with_ties_counting_half():
    rng = np.random.default_rng(0)
    true = rng.choice(["a", "b"], size=200)
    scores = rng.integers(0, 6, size=200) / 5  # only six different scores, so there are many ties
    positive, negative = scores[true == "a"], scores[true == "b"]
    wins = (positive[:, None] > negative[None, :]).sum() + 0.5 * (positive[:, None] == negative[None, :]).sum()
    assert roc_auc(true, scores, "a") == pytest.approx(wins / (len(positive) * len(negative)))


def test_a_random_scorer_has_an_auc_of_about_half():
    rng = np.random.default_rng(1)
    true = rng.choice(["a", "b"], size=5000)
    assert roc_auc(true, rng.random(5000), "a") == pytest.approx(0.5, abs=0.03)


def test_bad_input_to_the_auc_is_rejected():
    with pytest.raises(ValueError, match="same length"):
        auc([0, 1], [0])
    with pytest.raises(ValueError, match="at least two points"):
        auc([0.0], [0.0])
