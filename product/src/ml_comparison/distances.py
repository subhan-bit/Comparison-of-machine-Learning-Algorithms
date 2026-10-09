import math


def euclidean_distance(point_a, point_b):
    """
    Calculate the straight-line distance between two points.
    """

    if len(point_a) != len(point_b):
        raise ValueError("Points must have the same number of dimensions.")

    squared_differences = []

    for i in range(len(point_a)):
        difference = point_a[i] - point_b[i]
        squared_differences.append(difference ** 2)

    total = sum(squared_differences)

    return math.sqrt(total)


def manhattan_distance(point_a, point_b):
    """
    Calculate the Manhattan distance between two points.
    """

    if len(point_a) != len(point_b):
        raise ValueError("Points must have the same number of dimensions.")

    differences = []

    for i in range(len(point_a)):
        difference = abs(point_a[i] - point_b[i])
        differences.append(difference)

    return sum(differences)