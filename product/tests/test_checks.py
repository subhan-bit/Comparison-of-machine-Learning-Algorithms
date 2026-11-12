import numpy as np
import pytest

from ml_comparison.checks import check_whole_number


def test_whole_numbers_are_returned_as_ints():
    assert check_whole_number("k", 3) == 3
    assert check_whole_number("k", np.int64(3)) == 3
    assert check_whole_number("k", 3.0) == 3  # a whole float, such as one read from a config file
    assert isinstance(check_whole_number("k", 3.0), int)


def test_other_values_are_rejected_with_a_clear_error():
    for bad in (2.5, "3", True, None, [3], float("nan")):
        with pytest.raises(TypeError, match="k must be a whole number"):
            check_whole_number("k", bad)


def test_absurdly_large_values_are_rejected():
    # 1e300 is a whole number, but nobody has that many rows or levels: it is a mistake
    for big in (1e300, 10**12, np.int64(10**12), -1e300):
        with pytest.raises(ValueError, match="k is too large"):
            check_whole_number("k", big)
    assert check_whole_number("k", 10**9) == 10**9  # the limit itself is fine
