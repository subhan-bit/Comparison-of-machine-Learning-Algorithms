"""A data set: the dots (X) and the label of each dot (y)."""

import numpy as np


class Dataset:
    def __init__(self, X, y, positive_label=None, classes=None):
        # X must be a table of numbers. Uneven rows or text make this fail.
        try:
            X = np.array(X, dtype=float)
        except (ValueError, TypeError):  # text or uneven rows, or something that is not a number
            raise ValueError("X must be a rectangular table of numbers") from None

        # Dataset([], []) gives a flat empty array, so make it 0 rows by 0 columns
        if X.ndim == 1 and len(X) == 0:
            X = X.reshape(0, 0)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite numbers (no NaN or inf)")

        y = np.array(y)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)} labels")
        for row, label in enumerate(y):
            # a blank label would become a class called '', which is almost always a mistake
            if label is None or str(label).strip() == "":
                raise ValueError(f"label {row} is blank")

        self.X = X
        self.y = y

        # classes: each label once, sorted. codes: the position of each row's label in classes.
        # Example: y = [spam, ham, spam] -> classes = [ham, spam], codes = [1, 0, 1]
        if classes is None:
            self.classes, self.codes = np.unique(y, return_inverse=True)
        else:
            # The classes were given, for example the classes of the whole data set for one of its
            # splits. A class with no rows here is still a class, so the score columns of a
            # classifier do not depend on which classes landed in a split.
            self.classes = np.array(classes)
            position = {label: i for i, label in enumerate(self.classes)}
            if len(position) != len(self.classes):
                raise ValueError("classes must not repeat a label")
            try:
                self.codes = np.array([position[label] for label in y.tolist()], dtype=int)
            except KeyError as error:
                raise ValueError(f"label {error.args[0]!r} is not one of the classes") from None

        # The "positive" class is the one we care about finding, e.g. fraud or spam
        if positive_label is not None and positive_label not in list(self.classes):
            raise ValueError(f"positive_label {positive_label!r} is not one of the labels")
        self.positive_label = positive_label

    def __len__(self):
        return len(self.y)

    @property
    def n_features(self):
        return self.X.shape[1]
