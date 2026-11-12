import numpy as np
import pytest

from ml_comparison.artificial import make_blobs, make_circles, make_xor


def test_blobs_has_two_classes_of_the_same_size():
    data = make_blobs(50, seed=1)
    assert data.X.shape == (100, 2)
    assert list(data.y).count(0) == 50
    assert list(data.y).count(1) == 50


def test_blobs_points_are_near_their_centre():
    data = make_blobs(200, seed=1)
    class_0 = data.X[data.y == 0]
    class_1 = data.X[data.y == 1]
    # the average point of each class should be close to its centre
    assert np.allclose(class_0.mean(axis=0), [0, 0], atol=0.5)
    assert np.allclose(class_1.mean(axis=0), [5, 5], atol=0.5)


def test_xor_label_is_1_when_x_and_y_have_the_same_sign():
    data = make_xor(100, seed=1)
    assert data.X.shape == (100, 2)
    for (x, y), label in zip(data.X, data.y, strict=True):
        same_sign = (x > 0) == (y > 0)
        assert label == (1 if same_sign else 0)


def test_circles_are_at_radius_1_and_3():
    data = make_circles(50, seed=1)
    radius = np.sqrt((data.X ** 2).sum(axis=1))
    assert np.allclose(radius[data.y == 0], 1)
    assert np.allclose(radius[data.y == 1], 3)


def test_n_means_n_per_class_for_blobs_and_circles_but_n_in_all_for_xor():
    # the XOR labels come from where each random point lands, so the classes can't be
    # made the same size; blobs and circles make n points for each of the 2 classes
    assert len(make_blobs(30, seed=1)) == 60
    assert len(make_circles(30, seed=1)) == 60
    assert len(make_xor(30, seed=1)) == 30


def test_the_artificial_data_has_the_shape_the_docstrings_promise():
    blobs = make_blobs(2000, seed=1)
    for label in (0, 1):
        assert blobs.X[blobs.y == label].std(axis=0) == pytest.approx([1.0, 1.0], abs=0.1)   # a spread of 1
    xor = make_xor(2000, seed=1)
    assert -1 <= xor.X.min() < -0.95 and 0.95 < xor.X.max() <= 1      # the whole square, and no more
    circles = make_circles(2000, seed=1)
    assert set(circles.y) == {0, 1}
    for label in (0, 1):                                              # each circle on its own
        angles = np.arctan2(circles.X[circles.y == label, 1], circles.X[circles.y == label, 0])
        assert angles.min() < -3.0 and angles.max() > 3.0             # all the way round ...
        assert np.histogram(angles, bins=8, range=(-np.pi, np.pi))[0].min() > 200   # ... evenly (250 per bin expected)


def test_same_seed_gives_the_same_data():
    for make in [make_blobs, make_xor, make_circles]:
        first = make(20, seed=7)
        second = make(20, seed=7)
        assert np.array_equal(first.X, second.X)


def test_different_seed_gives_different_data():
    for make in [make_blobs, make_xor, make_circles]:
        assert not np.array_equal(make(20, seed=1).X, make(20, seed=2).X)
