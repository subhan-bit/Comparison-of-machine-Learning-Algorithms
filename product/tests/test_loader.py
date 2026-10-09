import pytest

from ml_comparison.loader import load_csv


def test_load_csv_reads_features_and_labels(tmp_path):
    file = tmp_path / "dots.csv"
    file.write_text("x1,x2,colour\n1,1,red\n4,4,blue\n")

    data = load_csv(file, "colour")

    assert len(data) == 2
    assert data.X.tolist() == [[1.0, 1.0], [4.0, 4.0]]
    assert list(data.y) == ["red", "blue"]


def test_load_csv_with_unknown_label_column_fails(tmp_path):
    file = tmp_path / "dots.csv"
    file.write_text("x1,x2,colour\n1,1,red\n")

    with pytest.raises(KeyError):
        load_csv(file, "nope")