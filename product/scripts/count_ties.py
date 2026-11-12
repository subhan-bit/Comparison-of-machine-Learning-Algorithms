"""Count how often the k-NN vote ends in a tie on each real data set (RQ1).

Run it from the product folder:  python scripts/count_ties.py
The data sets must be in the data/ folder at the top of the project.
"""

from pathlib import Path

import numpy as np

from ml_comparison.dataset import Dataset
from ml_comparison.knn import KNN, has_tie
from ml_comparison.loader import load_csv

DATA = Path(__file__).resolve().parents[2] / "data"
K_VALUES = range(1, 11)
SEED = 0  # fixed, so every run gives the same numbers
FRAUD = "Credit Card Fraud"
GENUINE_ROWS = 3000  # how many genuine rows go with the 492 frauds


def rng_for(name):
    """The random generator for one data set. Its seed is made from the data set's name, so a
    data set gets the same random numbers whatever the other data sets are, or in what order."""
    return np.random.default_rng([SEED, *name.encode()])


def names(n):
    """Made-up column names x1, x2, ... for the files with no header row."""
    return [f"x{i}" for i in range(1, n + 1)]


def keep_all_rare_class(data, n_others, rng):
    """Every row of the rarest class, plus n_others random rows of the other classes.

    The fraud data is too big for brute-force k-NN, and only 0.17% of it is fraud. A plain random
    subsample of 3,000 rows holds about 5 frauds, so its tie counts say nothing about fraud.
    Keeping every fraud gives a sample with far more fraud than the real data (an enriched sample).
    """
    rare = np.bincount(data.codes).argmin()
    rare_rows = np.flatnonzero(data.codes == rare)
    other_rows = rng.choice(np.flatnonzero(data.codes != rare), size=n_others, replace=False)
    rows = np.concatenate([rare_rows, other_rows])
    return Dataset(data.X[rows], data.y[rows])


def split(data, rng):
    """A random 70% for training and 30% for testing, with every class split the same way.

    Splitting each class on its own (a stratified split) keeps a rare class, such as fraud,
    in both sets in the same share. A plain random split can leave it almost out of the test set.
    """
    train, test = [], []
    for code in range(len(data.classes)):
        rows = rng.permutation(np.flatnonzero(data.codes == code))
        cut = int(0.7 * len(rows))
        train.append(rows[:cut])
        test.append(rows[cut:])
    # shuffle again, so the rows are not in blocks of one class
    train = rng.permutation(np.concatenate(train))
    test = rng.permutation(np.concatenate(test))
    return Dataset(data.X[train], data.y[train]), Dataset(data.X[test], data.y[test])


def standardise(train, test):
    """Give every feature mean 0 and spread 1, using the training rows only."""
    mean = train.X.mean(axis=0)
    spread = train.X.std(axis=0)
    spread[spread == 0] = 1  # a feature that never changes would divide by 0
    return (Dataset((train.X - mean) / spread, train.y),
            Dataset((test.X - mean) / spread, test.y))


def load_all():
    return {
        "Iris": load_csv(DATA / "iris.csv", "Species", drop_columns=["Id"]),
        "Wine": load_csv(DATA / "wine" / "wine.data", "class",
                         column_names=["class", *names(13)]),
        "Breast Cancer": load_csv(DATA / "breast_cancer.csv", "diagnosis", drop_columns=["id"]),
        "Spambase": load_csv(DATA / "spambase" / "spambase.data", "spam",
                             column_names=[*names(57), "spam"]),
        "German Credit": load_csv(DATA / "german_credit" / "german.data-numeric", "credit",
                                  column_names=[*names(24), "credit"], whitespace=True),
        FRAUD: load_csv(DATA / "creditcard.csv", "Class"),
    }


def count_row(name, data):
    """For one data set: the % of test points with a tied vote for each k.

    Only the name and the data decide the result, so it does not depend on the other data sets.
    """
    rng = rng_for(name)
    if name == FRAUD:
        data = keep_all_rare_class(data, GENUINE_ROWS, rng)
    train, test = standardise(*split(data, rng))
    ties = []
    for k in K_VALUES:
        model = KNN(k=k)
        model.train(train)
        ties.append(100 * has_tie(model.predict_scores(test.X)).mean())
    per_class = zip(test.classes.tolist(), np.bincount(test.codes).tolist(), strict=True)
    return {"classes": len(data.classes), "rows": len(data), "train": len(train), "test": len(test),
            "test_per_class": ", ".join(f"{label}: {n}" for label, n in per_class), "ties": ties}


def main():
    results = {name: count_row(name, data) for name, data in load_all().items()}

    print("% of test points where the k-NN vote is a tie\n")
    print("| Data set | Classes | " + " | ".join(f"k={k}" for k in K_VALUES) + " |")
    print("|---|---|" + "---|" * len(K_VALUES))
    for name, row in results.items():
        print(f"| {name} | {row['classes']} | " + " | ".join(f"{t:.1f}" for t in row["ties"]) + " |")

    print(f"\nRows used. The fraud row is an enriched sample: every fraud plus {GENUINE_ROWS:,} genuine rows.\n")
    print("| Data set | Rows | Training rows | Test rows | Test rows per class |")
    print("|---|---|---|---|---|")
    for name, row in results.items():
        print(f"| {name} | {row['rows']:,} | {row['train']:,} | {row['test']:,} | {row['test_per_class']} |")


if __name__ == "__main__":
    main()
