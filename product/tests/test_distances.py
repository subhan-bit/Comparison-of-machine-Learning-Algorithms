from ml_comparison.distances import euclidean, manhattan


def test_distances_between_two_dots():
    assert euclidean([0, 0], [3, 4]) == 5
    assert manhattan([0, 0], [3, 4]) == 7 
    assert euclidean([2, 5], [2, 5]) == 0


def test_distances_for_many_rows_at_once():
    rows = [[3, 4], [0, 1]]
    assert euclidean(rows, [0, 0]).tolist() == [5, 1]
    assert manhattan(rows, [0, 0]).tolist() == [7, 1]
