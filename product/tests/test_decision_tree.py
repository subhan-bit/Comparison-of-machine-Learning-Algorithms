import time

import numpy as np
import pytest

from ml_comparison.artificial import make_blobs, make_circles, make_xor
from ml_comparison.dataset import Dataset
from ml_comparison.decision_tree import (
    DecisionTree,
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


def sides_of(split, X):
    """How many rows go left and right for a (feature, threshold) split."""
    feature, threshold = split
    goes_left = np.asarray(X, dtype=float)[:, feature] <= threshold
    return goes_left.sum(), (~goes_left).sum()


def test_a_split_never_leaves_one_side_empty_for_values_one_float_step_apart():
    # halfway between two neighbouring floats rounds to one of them, so a careless
    # midpoint can send every row left and make train() repeat the same split forever
    a = np.nextafter(1.0, 2)
    b = np.nextafter(a, 2)
    X = [[a], [b]]
    split = best_split(X, [0, 1], gini)
    assert sides_of(split, X) == (1, 1)  # still told apart, one row on each side

    model = DecisionTree()
    model.train(Dataset(X, [0, 1]))  # must finish
    assert model.count_leaves() == 2


def test_the_midpoint_of_very_large_values_does_not_overflow():
    for X in ([[1e308], [1.7e308]], [[-1.7e308], [1.7e308]]):
        split = best_split(X, [0, 1], gini)
        assert np.isfinite(split[1])
        assert sides_of(split, X) == (1, 1)


def test_no_split_when_no_split_lowers_the_impurity():
    # one odd label among five: every split leaves the misclassification error at 0.2
    X = [[1], [2], [3], [4], [5]]
    codes = [0, 0, 1, 0, 0]
    assert best_split(X, codes, misclassification) is None

    model = DecisionTree(measure="misclassification")
    model.train(Dataset(X, ["a", "a", "b", "a", "a"]))
    assert model.count_leaves() == 1

    # gini can see that isolating the odd label helps, so it still splits
    assert best_split(X, codes, gini) is not None


def slow_best_split(X, codes, measure):
    """The simple search the code used to do: rescan every row for every threshold."""
    X = np.asarray(X, dtype=float)
    codes = np.asarray(codes)
    best = None
    best_score = measure(codes)
    for feature in range(X.shape[1]):
        values = np.unique(X[:, feature])
        for threshold in (values[:-1] + values[1:]) / 2:
            goes_left = X[:, feature] <= threshold
            score = split_impurity(codes[goes_left], codes[~goes_left], measure)
            # scores within 1e-12 count as equal, so rounding cannot pick a winner
            if score < best_score - 1e-12:
                best = (feature, float(threshold))
                best_score = score
    return best


def test_the_fast_search_finds_the_same_split_as_the_slow_one():
    rng = np.random.default_rng(0)
    for measure in [gini, entropy, misclassification]:
        for n_classes in [2, 3]:
            # whole numbers from 0 to 9, so values repeat
            X = rng.integers(0, 10, size=(60, 3))
            codes = rng.integers(0, n_classes, size=60)
            assert best_split(X, codes, measure) == slow_best_split(X, codes, measure)


def test_best_split_copes_with_many_rows():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20000, 5))
    codes = (X[:, 3] > 0.5).astype(int)  # only feature 3 matters

    start = time.perf_counter()
    feature, threshold = best_split(X, codes, gini)
    seconds = time.perf_counter() - start

    assert feature == 3
    assert threshold == pytest.approx(0.5, abs=0.01)
    assert seconds < 3  # the slow search took about 10 seconds


def accuracy(model, data):
    return np.mean(model.predict(data.X) == data.y)


def test_a_full_tree_gets_every_training_row_right():
    train = make_circles(200, seed=1)
    model = DecisionTree()
    model.train(train)
    assert accuracy(model, train) == 1


def test_works_on_blobs_xor_and_circles_with_every_measure():
    for make in [make_blobs, make_xor, make_circles]:
        for measure in ["gini", "entropy", "misclassification"]:
            model = DecisionTree(measure=measure)
            model.train(make(200, seed=1))
            assert accuracy(model, make(100, seed=2)) >= 0.85, (make.__name__, measure)


def test_scores_are_the_smoothed_class_shares_in_the_leaf():
    # rows 0-2 can't be told apart, so they end up in one leaf: 2 of class a, 1 of class b
    # smoothed share = (count + 1) / (rows + number of classes)
    data = Dataset([[1], [1], [1], [5]], ["a", "a", "b", "b"])
    model = DecisionTree()
    model.train(data)
    assert model.predict_scores([[1]])[0] == pytest.approx([3 / 5, 2 / 5])
    assert model.predict([[1], [5]]).tolist() == ["a", "b"]


def test_scores_of_a_full_tree_are_not_only_0_and_1():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 2))
    y = np.zeros(500, dtype=int)
    y[rng.choice(500, 10, replace=False)] = 1  # 2% positive
    model = DecisionTree()
    model.train(Dataset(X, y))

    scores = model.predict_scores(rng.normal(size=(200, 2)))[:, 1]
    assert len(np.unique(scores)) > 2  # leaves of different sizes give different scores
    assert scores.min() > 0 and scores.max() < 1  # no leaf is completely sure


def test_max_depth_0_just_predicts_the_smoothed_shares_of_the_whole_training_set():
    data = Dataset([[1], [2], [3], [4]], ["a", "a", "a", "b"])
    model = DecisionTree(max_depth=0)
    model.train(data)
    assert model.predict_scores([[4]])[0] == pytest.approx([4 / 6, 2 / 6])


def test_max_depth_1_makes_one_split():
    data = Dataset([[1], [2], [3], [4]], ["a", "b", "a", "b"])
    model = DecisionTree(max_depth=1)
    model.train(data)
    assert model.root.left.is_leaf() and model.root.right.is_leaf()


def test_min_samples_split_stops_small_nodes_from_splitting():
    data = Dataset([[1], [2], [3], [4]], ["a", "a", "b", "b"])
    model = DecisionTree(min_samples_split=5)  # only 4 rows, so no split
    model.train(data)
    assert model.root.is_leaf()


def test_bad_tree_settings_are_rejected():
    with pytest.raises(ValueError):
        DecisionTree(measure="variance")
    with pytest.raises(ValueError):
        DecisionTree(max_depth=-1)
    with pytest.raises(ValueError):
        DecisionTree(min_samples_split=1)


def test_depth_and_split_size_must_be_whole_numbers():
    with pytest.raises(TypeError, match="whole number"):
        DecisionTree(max_depth=2.5)
    with pytest.raises(TypeError, match="whole number"):
        DecisionTree(max_depth="a")
    with pytest.raises(TypeError, match="whole number"):
        DecisionTree(min_samples_split=2.5)


def test_pruning_removes_a_split_that_only_fits_noise():
    # the "b" at x=3 is noise: the full tree makes extra splits just to fit it
    train = Dataset([[1], [2], [3], [4], [5]], ["a", "a", "b", "a", "a"])
    model = DecisionTree()
    model.train(train)
    assert model.count_leaves() == 3

    # on the validation set, x=3 is really "a", so those splits only make things worse
    model.prune(Dataset([[3], [5]], ["a", "a"]))
    assert model.root.is_leaf()
    assert model.predict([[3]]).tolist() == ["a"]


def test_pruning_keeps_a_split_that_helps():
    model = DecisionTree()
    model.train(Dataset([[1], [2], [3], [4]], ["a", "a", "b", "b"]))
    model.prune(Dataset([[1], [4]], ["a", "b"]))
    assert model.count_leaves() == 2


def test_a_very_deep_tree_can_be_trained_counted_and_pruned():
    # alternating labels on one feature need a split between every pair of rows,
    # so the tree is far deeper than Python's recursion limit
    X = [[i] for i in range(2000)]
    labels = [i % 2 for i in range(2000)]
    model = DecisionTree()
    model.train(Dataset(X, labels))
    assert model.count_leaves() == 2000
    assert accuracy(model, Dataset(X, labels)) == 1

    # pruning against the same rows keeps every split, because each one is right
    model.prune(Dataset(X, labels))
    assert model.count_leaves() == 2000


def test_pruning_rejects_an_empty_validation_set():
    model = DecisionTree()
    model.train(Dataset([[1], [2]], ["a", "b"]))
    with pytest.raises(ValueError, match="empty"):
        model.prune(Dataset([], []))


def test_pruning_keeps_a_split_that_no_validation_row_reaches():
    # left of 6 is "b"; right of 6 splits again at 12.5 (3 leaves in all)
    train = Dataset([[1], [2], [10], [11], [12], [13]], ["b", "b", "a", "a", "a", "b"])
    model = DecisionTree()
    model.train(train)
    assert model.count_leaves() == 3

    # the validation rows are all on the left, so nothing shows the right split is bad
    model.prune(Dataset([[1], [2]], ["b", "b"]))
    assert model.count_leaves() == 3


def test_pruning_a_noisy_tree_makes_it_smaller_and_no_worse():
    # blobs where 15% of the training labels have been flipped
    rng = np.random.default_rng(0)
    blobs = make_blobs(200, seed=1)
    flipped = rng.random(len(blobs)) < 0.15
    noisy = Dataset(blobs.X, np.where(flipped, 1 - blobs.y, blobs.y))
    validation, test = make_blobs(50, seed=2), make_blobs(100, seed=3)

    model = DecisionTree()
    model.train(noisy)
    leaves_before, accuracy_before = model.count_leaves(), accuracy(model, test)
    model.prune(validation)
    assert model.count_leaves() < leaves_before
    assert accuracy(model, test) >= accuracy_before


def test_cannot_prune_before_training():
    with pytest.raises(RuntimeError):
        DecisionTree().prune(Dataset([[1]], ["a"]))


def test_bad_validation_sets_are_rejected():
    model = DecisionTree()
    model.train(Dataset([[1], [2]], ["a", "b"]))
    with pytest.raises(ValueError, match="not in the training data"):
        model.prune(Dataset([[1]], ["c"]))        # a label the tree never saw
    with pytest.raises(ValueError, match="features"):
        model.prune(Dataset([[1, 2]], ["a"]))     # wrong number of columns
