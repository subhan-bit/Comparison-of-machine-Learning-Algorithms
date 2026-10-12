"""Read a CSV file (like the ones from Kaggle) into a Dataset."""

import csv
import math

import numpy as np

from ml_comparison.dataset import Dataset


def load_csv(path, label_column, drop_columns=()):
    """The label column becomes y and the other columns become X.

    drop_columns lists columns to leave out, such as an id column.
    """
    if isinstance(drop_columns, str):
        raise TypeError("drop_columns must be a list of column names, not a string")

    # utf-8-sig removes the hidden "byte order mark" some files start with,
    # otherwise it gets stuck to the first column name
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
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

            # DictReader puts extra cells under the key None, and fills missing cells with None
            if None in row:
                raise ValueError(f"{path} line {line}: row has more fields than the header")
            if None in row.values():
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
