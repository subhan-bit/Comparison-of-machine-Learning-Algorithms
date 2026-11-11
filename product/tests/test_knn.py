import itertools

import numpy as np
import pytest

from ml_comparison.artificial import make_blobs, make_xor
from ml_comparison.dataset import Dataset
from ml_comparison.knn import KNN, has_tie

# The dots from documents/knn/knn.md: A, B red and C, D, E blue.
# E is at (4, 6), not (4, 5): (4, 5) is exactly as far from Q as D is, so the 4th neighbour
# would be a distance tie that only the row order could settle.
DOTS = Dataset([[1, 1], [2, 1], [4, 4], [5, 4], [4, 6]],
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


def test_every_point_tied_at_the_kth_distance_votes_whatever_order_the_rows_are_in():
    # Q is (0, 0) and all three points are exactly 1 away: one red and two blue
    points = [[0, 1], [1, 0], [0, -1]]
    labels = ["red", "blue", "blue"]
    for order in itertools.permutations(range(3)):
        data = Dataset([points[i] for i in order], [labels[i] for i in order])
        model = KNN(k=1)
        model.train(data)
        # the classes are sorted: blue, red. All three tied points vote, so 2 blue and 1 red
        assert model.predict_scores([[0, 0]])[0] == pytest.approx([2 / 3, 1 / 3]), order
        assert model.predict([[0, 0]]).tolist() == ["blue"], order


def test_the_old_example_point_was_a_hidden_distance_tie():
    # with E at (4, 5), D and E are both exactly sqrt(13) from Q, so k=4 has five voters:
    # 2 red and 3 blue, and there is no vote tie. The old code let row order drop E.
    data = Dataset([[1, 1], [2, 1], [4, 4], [5, 4], [4, 5]],
                   ["red", "red", "blue", "blue", "blue"])
    model = KNN(k=4)
    model.train(data)
    assert model.predict_scores(Q)[0] == pytest.approx([0.6, 0.4])  # blue, red


def test_the_tie_can_be_further_out_than_the_nearest_point():
    # distances from (0, 0): 1 (red), then 2 (blue), 2 (blue), then 3 (red)
    data = Dataset([[1, 0], [2, 0], [0, 2], [3, 0]], ["red", "blue", "blue", "red"])
    model = KNN(k=2)
    model.train(data)
    # the 2nd distance is 2, which two points share, so three points vote: 1 red, 2 blue
    assert model.predict_scores([[0, 0]])[0] == pytest.approx([2 / 3, 1 / 3])


def test_without_tied_distances_the_votes_match_a_plain_full_sort():
    rng = np.random.default_rng(0)
    data = Dataset(rng.normal(size=(200, 3)), rng.integers(0, 4, size=200))
    queries = rng.normal(size=(30, 3))
    for k in (1, 5, 20):
        model = KNN(k=k)
        model.train(data)
        expected = np.zeros((30, 4))
        for row, q in enumerate(queries):
            nearest = np.argsort(((data.X - q) ** 2).sum(axis=1))[:k]
            for i in nearest:
                expected[row, data.codes[i]] += 1
        assert model.predict_scores(queries) == pytest.approx(expected / k)


def test_the_default_k_is_one():
    assert KNN().k == 1


def test_k_must_be_a_whole_number():
    for bad in (2.5, True, "3"):
        with pytest.raises(TypeError, match="k must be a whole number"):
            KNN(k=bad)
    assert KNN(k=3.0).k == 3  # a whole float is fine


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


def test_random_gives_the_same_answers_every_time_predict_is_called():
    # the same tied point, asked for 12 times: a seeded model must always answer the same
    model = KNN(k=4, tie_break="random", seed=5)
    model.train(DOTS)
    answers = [model.predict(Q)[0] for _ in range(12)]
    assert len(set(answers)) == 1

    # and the same batch of tied points gives the same list of answers each time
    batch = Q * 8
    assert model.predict(batch).tolist() == model.predict(batch).tolist()


def test_random_without_a_seed_can_differ_between_calls():
    model = KNN(k=4, tie_break="random")
    model.train(DOTS)
    answers = {model.predict(Q)[0] for _ in range(60)}
    assert answers == {"red", "blue"}


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


def test_has_tie_finds_rows_where_the_top_score_is_shared():
    scores = [[0.5, 0.5], [0.75, 0.25], [0.25, 0.75]]
    assert has_tie(scores).tolist() == [True, False, False]
    # three classes: two of them share the top score
    assert has_tie([[0.4, 0.4, 0.2]]).tolist() == [True]


def test_has_tie_ignores_tiny_rounding_differences():
    # 0.1 + 0.2 is 0.30000000000000004, so == says these differ, but they are the same score
    assert has_tie([[0.1 + 0.2, 0.3]]).tolist() == [True]
    assert has_tie([[0.3, 0.30001]]).tolist() == [False]  # a real difference still counts


def test_changing_the_original_data_does_not_change_a_trained_model():
    data = Dataset([[1, 1], [2, 1], [4, 4], [5, 4], [4, 6]],
                   ["red", "red", "blue", "blue", "blue"])
    model = KNN(k=1)
    model.train(data)
    assert model.predict(Q).tolist() == ["red"]

    data.codes[:] = 1 - data.codes  # swap red and blue in the original data
    assert model.predict(Q).tolist() == ["red"]  # the model kept its own copy

    data.X[:] = 0  # and wipe the original dots (all the same distance now)
    assert model.predict(Q).tolist() == ["red"]


def test_has_tie_on_the_example_from_the_notes():
    # k=4 gives 2 red and 2 blue votes for Q, but k=3 does not tie
    for k, expected in [(3, False), (4, True)]:
        model = KNN(k=k)
        model.train(DOTS)
        assert has_tie(model.predict_scores(Q)).tolist() == [expected]
