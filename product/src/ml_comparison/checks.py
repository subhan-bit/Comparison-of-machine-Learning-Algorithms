"""Small checks shared by the classifiers."""

import numpy as np


MOST = 10**9  # no k, depth or number of rows is ever this big, so a bigger value is a mistake such as 1e300


def check_whole_number(name, value):
    """Return value as an int, or raise a clear error if it is not a whole number.

    A float such as 3.0 is fine, because settings read from a file often arrive that way.
    A number bigger than MOST is rejected too.
    """
    whole = (isinstance(value, (int, np.integer))
             or (isinstance(value, (float, np.floating)) and float(value).is_integer()))
    if isinstance(value, bool) or not whole:  # True is an int in Python, but not a count
        raise TypeError(f"{name} must be a whole number, not {value!r}")
    if abs(value) > MOST:
        raise ValueError(f"{name} is too large: {value!r} (the most allowed is {MOST:,})")
    return int(value)
