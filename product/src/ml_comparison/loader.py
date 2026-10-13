"""Read a CSV file (like the ones from Kaggle) into a Dataset."""

import csv
import math

import numpy as np

from ml_comparison.dataset import Dataset


def load_csv(path, label_column, drop_columns=(), column_names=None, whitespace=False):
    """The label column becomes y and the other columns become X.

    drop_columns lists columns to leave out, such as an id column.
    column_names gives the names for a file with no header row (like the UCI files).
    whitespace=True reads values separated by spaces instead of commas (like German Credit).
    """
    if isinstance(drop_columns, str):
        raise TypeError("drop_columns must be a list of column names, not a string")

    # utf-8-sig removes the hidden "byte order mark" some files start with,
    # otherwise it gets stuck to the first column name
    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = f
        if whitespace:
            # Turn "  1   6  good" into "1,6,good" so the csv reader can handle it
            lines = (",".join(line.split()) + "\n" for line in f)
        reader = csv.DictReader(lines, fieldnames=column_names)
        headers = reader.fieldnames
        if not headers:
            raise ValueError(f"{path} has no header row")
        for name in [label_column, *drop_columns]:
            if name not in headers:
                raise KeyError(name)

        # Every other named column is a feature (blank header names are skipped)
        features = [h for h in headers if h and h != label_column and h not in drop_columns]

        X = []
        y = []
        for row in reader:
            line = reader.line_num  # the real line number in the file, for error messages

            # DictReader puts extra cells under the key None, and fills missing cells with None.
            # A missing cell under a blank header name is fine: some files (like the Breast
            # Cancer one) end only the header row with a comma.
            if None in row:
                raise ValueError(f"{path} line {line}: row has more fields than the header")
            if any(value is None for name, value in row.items() if name):
                raise ValueError(f"{path} line {line}: row has fewer fields than the header")

            numbers = []
            for name in features:
                text = row[name]
                try:
                    value = float(text)
                except ValueError:
                    raise ValueError(f"{path} line {line}, column {name!r}: {text!r} is not a number")
                if not math.isfinite(value):
                    raise ValueError(f"{path} line {line}, column {name!r}: {text!r} is not a finite number")
                numbers.append(value)

            X.append(numbers)
            y.append(row[label_column])

    # reshape keeps the right number of columns even when the file has no rows
    return Dataset(np.array(X, dtype=float).reshape(len(X), len(features)), y)
