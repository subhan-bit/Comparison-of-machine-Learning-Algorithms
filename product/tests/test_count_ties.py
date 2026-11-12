import importlib.util
from pathlib import Path

import numpy as np

from ml_comparison.dataset import Dataset

# scripts/count_ties.py is a script, not part of the package, so load it from its file
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "count_ties.py"
spec = importlib.util.spec_from_file_location("count_ties", SCRIPT)
count_ties = importlib.util.module_from_spec(spec)
spec.loader.exec_module(count_ties)


def random_data(n, seed):
    """n rows with 3 features and two classes, made up for the tests."""
    rng = np.random.default_rng(seed)
    return Dataset(rng.normal(size=(n, 3)), rng.integers(0, 2, size=n))


def test_a_data_sets_generator_depends_only_on_its_name():
    assert count_ties.rng_for("Wine").random() == count_ties.rng_for("Wine").random()
    assert count_ties.rng_for("Wine").random() != count_ties.rng_for("Iris").random()


def test_a_data_sets_row_does_not_depend_on_the_other_data_sets():
    sets = {"A": random_data(60, 1), "B": random_data(80, 2), "C": random_data(70, 3)}
    alone = count_ties.count_row("B", sets["B"])["ties"]
    for order in (["A", "B", "C"], ["C", "B", "A"], ["B", "C"]):
        rows = {name: count_ties.count_row(name, sets[name])["ties"] for name in order}
        assert rows["B"] == alone, order
    assert count_ties.count_row("B", sets["B"])["ties"] == alone  # and a second run gives the same row

