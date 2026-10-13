# Preparing CSV Data

## What it is
The Kaggle data sets come as CSV files: a header row with the column names,
then one row per example. Before an algorithm can use a file, every feature
has to be a number and every row has to be complete. load_csv does this
and stops with a clear error when something is wrong.

## Problems to watch for
- Id column. A number that names the patient or transaction but says
  nothing about the label. If it stays in, it swamps the distances in k-NN
  (an id like 842302 against features around 10 to 20). Drop it.
- Blank column name in the header. In the real Breast Cancer file only the
  header row ends with a comma, so it names 33 columns (the last one blank)
  while every data row has 32 values. I first assumed every row ended with a
  comma, but checking the downloaded file showed it does not. Kaggle notebooks
  call this column "Unnamed: 32" [3]. load_csv skips columns with a blank name.
- Byte order mark (BOM). An invisible character some programs (for example
  Excel) put at the very start of a UTF-8 file [2]. It sticks to the first
  column name, so "id" is read as "﻿id" and dropping "id" fails.
  Opening the file with encoding "utf-8-sig" removes it [1].
- Missing or extra cells. Python's csv.DictReader fills missing cells with
  None and puts extra cells under the key None [1]. Both are errors.
- NaN and infinity. float("nan") and float("inf") are accepted by Python,
  but they break distances and splits, so they are rejected.
- Text labels. Labels like "M" and "B" are given numbers (sorted, so B = 0,
  M = 1) so every algorithm uses the same order.

## Example
A file in the same layout as the Breast Cancer data (values made up):

    id,diagnosis,radius_mean,texture_mean,
    842302,M,17.99,10.38
    842517,B,13.54,14.36

load_csv(file, "diagnosis", drop_columns=["id"]) gives:
- X = [[17.99, 10.38], [13.54, 14.36]] (id and the blank column are gone)
- y = ["M", "B"], classes = ["B", "M"], codes = [1, 0]

If the second row were "842517,B,abc,14.36" the error would say
"line 3, column 'radius_mean': 'abc' is not a number".

## Why it matters for my project
- Bad data gives wrong results with no warning, so it is better to stop
  at load time and say where the problem is.
- ROC and cost need one fixed positive class (malignant, fraud, spam), so
  the labels need a fixed numbering.
- This step only reads the file. Normalising the features (RQ3) comes later
  and must be fitted inside each cross-validation fold, or the test data
  leaks into training.

## Still to learn
- Normalisation methods (min-max and z-score) and how each affects k-NN.
- Turning text attributes into numbers. German Credit has categories like
  "own house" that need encoding (for example one-hot).
- (Done) Checked the real files once downloaded. What each one looks like,
  and how to load it, is in data_sets.md.

## Sources
[1] Python documentation: csv (https://docs.python.org/3/library/csv.html)
    and codecs, "utf-8-sig" (https://docs.python.org/3/library/codecs.html).
[2] Unicode FAQ: UTF-8, UTF-16, UTF-32 and BOM
    (https://www.unicode.org/faq/utf_bom.html).
[3] UCI Machine Learning. Breast Cancer Wisconsin (Diagnostic) Data Set.
    Kaggle. https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
