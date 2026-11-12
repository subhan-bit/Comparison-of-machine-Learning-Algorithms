import pytest

from ml_comparison.loader import load_csv


def make_csv(tmp_path, text):
    file = tmp_path / "data.csv"
    file.write_text(text, encoding="utf-8")
    return file


def load_text(tmp_path, text):
    return load_csv(make_csv(tmp_path, text), "colour")


def test_reads_features_and_labels(tmp_path):
    data = load_csv(make_csv(tmp_path, "x1,x2,colour\n1,1,red\n4,4,blue\n"), "colour")
    assert data.X.tolist() == [[1.0, 1.0], [4.0, 4.0]]
    assert list(data.y) == ["red", "blue"]


def test_drops_id_and_blank_columns(tmp_path):
    # Like the Breast Cancer file: a byte order mark, an id column and an empty last column
    file = make_csv(tmp_path, "﻿id,x1,diagnosis,\n842302,1,M,\n")
    data = load_csv(file, "diagnosis", drop_columns=["id"])
    assert data.X.tolist() == [[1.0]]


def test_blank_column_only_in_the_header_is_ignored(tmp_path):
    # The real Breast Cancer file: only the header ends with a comma, the rows don't
    data = load_text(tmp_path, "x1,x2,colour,\n1,1,red\n4,4,blue\n")
    assert data.X.tolist() == [[1.0, 1.0], [4.0, 4.0]]


def test_file_without_a_header_uses_the_given_column_names(tmp_path):
    # Like the UCI files (Wine, Spambase), which have no header row
    file = make_csv(tmp_path, "1,1,red\n4,4,blue\n")
    data = load_csv(file, "colour", column_names=["x1", "x2", "colour"])
    assert data.X.tolist() == [[1.0, 1.0], [4.0, 4.0]]
    assert list(data.y) == ["red", "blue"]


def test_values_separated_by_spaces(tmp_path):
    # Like German Credit: spaces between values, and spaces at the start of each line
    file = make_csv(tmp_path, "   1   6  good\n  4  4   bad\n")
    data = load_csv(file, "label", column_names=["x1", "x2", "label"], whitespace=True)
    assert data.X.tolist() == [[1.0, 6.0], [4.0, 4.0]]
    assert list(data.y) == ["good", "bad"]


def test_line_numbers_start_at_1_without_a_header(tmp_path):
    file = make_csv(tmp_path, "1,1,red\n1,abc,blue\n")
    with pytest.raises(ValueError, match="line 2"):
        load_csv(file, "colour", column_names=["x1", "x2", "colour"])


def test_header_only_file_gives_an_empty_dataset(tmp_path):
    data = load_csv(make_csv(tmp_path, "x1,x2,colour\n"), "colour")
    assert data.X.shape == (0, 2)


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


def test_a_blank_label_is_rejected_with_its_line_number(tmp_path):
    # the empty label would otherwise become a class called ''
    with pytest.raises(ValueError, match="line 3: the label in column 'colour' is blank"):
        load_text(tmp_path, "x1,x2,colour\n1,1,red\n4,4,\n")
    with pytest.raises(ValueError, match="line 2: the label in column 'colour' is blank"):
        load_text(tmp_path, "x1,x2,colour\n1,1,   \n")


def test_repeated_column_names_are_rejected(tmp_path):
    # without a check, the second x1 would silently replace the first one
    with pytest.raises(ValueError, match="repeated column names.*'x1'"):
        load_text(tmp_path, "x1,x1,colour\n1,2,red\n")
    # the same goes for the names given for a file that has no header row
    file = make_csv(tmp_path, "1,2,red\n")
    with pytest.raises(ValueError, match="repeated column names.*'x1'"):
        load_csv(file, "colour", column_names=["x1", "x1", "colour"])


def test_several_blank_column_names_are_not_repeats(tmp_path):
    data = load_text(tmp_path, "x1,colour,,\n1,red,,\n")
    assert data.X.tolist() == [[1.0]]


def test_unknown_column_names_are_rejected(tmp_path):
    file = make_csv(tmp_path, "id,x1,colour\n1,1,red\n")
    with pytest.raises(KeyError):
        load_csv(file, "nope")
    with pytest.raises(KeyError):
        load_csv(file, "colour", drop_columns=["idd"])
    with pytest.raises(TypeError):
        load_csv(file, "colour", drop_columns="id")


def test_spaces_around_a_label_are_not_part_of_the_label(tmp_path):
    # "M " and "M" must be one class, not two
    data = load_text(tmp_path, "x1,colour\n1,M \n2, M\n3,M\n4,B\n")
    assert list(data.y) == ["M", "M", "M", "B"]
    assert data.classes.tolist() == ["B", "M"]


def test_only_ordinary_numbers_are_accepted(tmp_path):
    # float() also takes "1_0" (as 10) and digits from other scripts, which a data file should not have
    for bad in ["1_0", "١٢٣", "１２"]:  # the last two are Arabic-Indic and fullwidth digits
        with pytest.raises(ValueError, match="line 2, column 'x1'.*is not a number"):
            load_text(tmp_path, f"x1,colour\n{bad},red\n")
    # the ordinary ways to write a number are fine, with or without spaces around them
    data = load_text(tmp_path, "x1,colour\n1e3,a\n-2.5,a\n.5,a\n+7,a\n 3 ,a\n")
    assert data.X.ravel().tolist() == [1000.0, -2.5, 0.5, 7.0, 3.0]


def test_a_file_that_is_not_utf8_gives_a_clear_error(tmp_path):
    file = tmp_path / "latin1.csv"
    file.write_bytes("x1,colour\n1,café\n".encode("latin-1"))  # here the é is one byte, which is not valid UTF-8
    with pytest.raises(ValueError, match="latin1.csv is not a UTF-8 text file"):
        load_csv(file, "colour")
