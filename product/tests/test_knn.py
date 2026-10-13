import numpy as np
import pytest

from ml_comparison.artificial import make_blobs, make_xor
from ml_comparison.dataset import Dataset
from ml_comparison.knn import KNN

# The dots from documents/knn/knn.md: A, B red and C, D, E blue
DOTS = Dataset([[1, 1], [2, 1], [4, 4], [5, 4], [4, 5]],
               ["red", "red", "blue", "blue", "blue"])
Q = [[2, 2]]


def test_1nn_gives_the_label_of_the_closest_point():
    model = KNN(k=1)
    model.train(DOTS)
    assert model.predict(Q).tolist() == ["red"]  # closest is B, red


def test_k_changes_the_answer_like_in_the_notes():
    for k, expected in [(3, "red"), (5, "blue")]:
        model = KNN(k=k)
        model.train(DOTS)
        assert model.predict(Q).tolist() == [expected]


def test_scores_are_the_share_of_votes():
    model = KNN(k=5)
    model.train(DOTS)
    # classes are sorted: blue first, then red. 3 blue votes and 2 red votes out of 5
    assert model.predict_scores(Q).tolist() == [[0.6, 0.4]]


def test_manhattan_distance_can_be_chosen():
    model = KNN(k=3, distance="manhattan")
    model.train(DOTS)
    assert model.predict(Q).tolist() == ["red"]


def test_almost_perfect_on_blobs():
    model = KNN(k=3)
    model.train(make_blobs(100, seed=1))
    test = make_blobs(50, seed=2)
    accuracy = np.mean(model.predict(test.X) == test.y)
    assert accuracy >= 0.95


def test_learns_xor_which_no_straight_line_can_split():
    model = KNN(k=5)
    model.train(make_xor(400, seed=1))
    test = make_xor(200, seed=2)
    accuracy = np.mean(model.predict(test.X) == test.y)
    assert accuracy >= 0.9


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError):
        KNN(k=0)                        # need at least one neighbour
    with pytest.raises(ValueError):
        KNN(distance="cosine")          # only euclidean or manhattan
    with pytest.raises(ValueError):
        KNN(k=10).train(DOTS)           # more neighbours than training points


def test_k4_is_a_tie_and_the_scores_show_it():
    model = KNN(k=4)
    model.train(DOTS)
    # 2 blue votes and 2 red votes: the scores stay as plain vote shares
    assert model.predict_scores(Q).tolist() == [[0.5, 0.5]]


def test_nearest_reduce_k_and_weighted_all_pick_red():
    # the same answers as the worked example in documents/knn/knn_tie_breaking.md
    for strategy in ["nearest", "reduce_k", "weighted"]:
        model = KNN(k=4, tie_break=strategy)
        model.train(DOTS)
        assert model.predict(Q).tolist() == ["red"], strategy


def test_random_is_repeatable_with_a_seed_and_can_pick_either():
    answers = []
    for seed in range(20):
        model = KNN(k=4, tie_break="random", seed=seed)
        model.train(DOTS)
        answers.append(model.predict(Q)[0])
        # the same seed always gives the same answer
        again = KNN(k=4, tie_break="random", seed=seed)
        again.train(DOTS)
        assert again.predict(Q)[0] == answers[-1]
    assert set(answers) == {"red", "blue"}


def test_weighted_falls_back_to_nearest_when_the_weights_tie():
    # two points exactly as far from Q: their weights are equal too
    data = Dataset([[0, 1], [0, -1]], ["red", "blue"])
    model = KNN(k=2, tie_break="weighted")
    model.train(data)
    assert model.predict([[0, 0]]).tolist() == ["red"]  # red comes first in the data


def test_unknown_tie_break_is_rejected():
    with pytest.raises(ValueError):
        KNN(tie_break="coin flip")


def test_follows_the_classifier_rules():
    model = KNN(k=1)
    with pytest.raises(RuntimeError):
        model.predict(Q)                # not trained yet
    model.train(DOTS)
    with pytest.raises(ValueError):
        model.predict([[1, 2, 3]])      # wrong number of columns
