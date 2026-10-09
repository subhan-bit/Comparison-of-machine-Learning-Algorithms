"""A data set: dots (X) and the label of each dot (y)."""

import numpy as np


class Dataset:
    def __init__(self, X, y):
        self.X = np.array(X, dtype=float)  
        self.y = np.array(y)               

    def __len__(self):
        return len(self.y)