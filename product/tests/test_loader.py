import pytest

from ml_comparison.loader import load_csv


def make_csv(tmp_path, text):
    file = tmp_path / "data.csv"
    file.write_text(text, encoding="utf-8")
    return file


def test_reads_features_and_labels(tmp_path):
    data = load_csv(make_csv(tmp_path, "x1,x2,colour\n1,1,red\n4,4,blue\n"), "colour")
    assert data.X.tolist() == [[1.0, 1.0], [4.0, 4.0]]
    assert list(data.y) == ["red", "blue"]


def test_drops_id_and_blank_columns(tmp_path):
    # Like the Breast Cancer file: a byte order mark, an id column and an empty last column
    file = make_csv(tmp_path, "﻿id,x1,diagnosis,\n842302,1,M,\n")
    data = load_csv(file, "diagnosis", drop_columns=["id"])
    assert data.X.tolist() == [[1.0]]


def test_header_only_file_gives_an_empty_dataset(tmp_path):
    data = load_csv(make_csv(tmp_path, "x1,x2,colour\n"), "colour")
    assert data.X.shape == (0, 2)


def load_text(tmp_path, text):
    return load_csv(make_csv(tmp_path, text), "colour")


def test_bad_files_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_text(tmp_path, "")                              # empty file
    with pytest.raises(ValueError):
        load_text(tmp_path, "x1,x2,colour\n1,abc,red\n")     # text instead of a number
    with pytest.raises(ValueError):
        load_text(tmp_path, "x1,x2,colour\n1,,red\n")        # empty cell
    with pytest.raises(ValueError):
        load_text(tmp_path, "x1,x2,colour\nnan,1,red\n")     # NaN
    with pytest.raises(ValueError):
        load_text(tmp_path, "x1,x2,colour\n1,inf,red\n")     # infinity
    with pytest.raises(ValueError):
        load_text(tmp_path, "x1,x2,colour\n1,red\n")         # a cell missing
    with pytest.raises(ValueError):
        load_text(tmp_path, "x1,x2,colour\n1,1,red,9\n")     # an extra cell


def test_error_says_the_real_line_and_column(tmp_path):
    # The label on line 2 runs onto line 3, so the bad value is on line 4
    with pytest.raises(ValueError, match="line 4, column 'x2'"):
        load_text(tmp_path, 'x1,x2,colour\n1,1,"two\nlines"\n1,abc,blue\n')


def test_unknown_column_names_are_rejected(tmp_path):
    file = make_csv(tmp_path, "id,x1,colour\n1,1,red\n")
    with pytest.raises(KeyError):
        load_csv(file, "nope")
    with pytest.raises(KeyError):
        load_csv(file, "colour", drop_columns=["idd"])
    with pytest.raises(TypeError):
        load_csv(file, "colour", drop_columns="id")
