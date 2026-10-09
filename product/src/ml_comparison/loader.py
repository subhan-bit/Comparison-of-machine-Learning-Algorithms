"""Read a CSV file (like the ones from Kaggle) into a Dataset."""

import csv

from ml_comparison.dataset import Dataset


def load_csv(path, label_column):
    X = []
    y = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            y.append(row.pop(label_column))
            X.append([float(value) for value in row.values()])
    return Dataset(X, y)