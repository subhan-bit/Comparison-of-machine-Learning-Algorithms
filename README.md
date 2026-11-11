# Comparison of Machine Learning Algorithms

Cost-sensitive evaluation and multi-class reduction strategies.
Final Year Individual Project, Department of Computer Science,
Royal Holloway, University of London.

**Author:** Muhammad Subhan Ahmad
**Supervisor:** Anand Subramoney

## What this project is
A Python program, written from scratch with no machine learning library, that
learns from labelled examples, tests how well each algorithm predicts labels it
has never seen, and compares the algorithms fairly.

Algorithms: k-NN (with tie-breaking strategies and a k-d tree), decision trees
(Gini index, entropy, misclassification error), kernel nearest neighbours, and
a multi-class SVM trained with SMO. They are compared with a hold-out test set,
k-fold cross-validation and ROC analysis.

Extension: how the algorithms behave when some mistakes cost more than others
(spam, credit risk, fraud, medical diagnosis). Secondary study: ways of
reducing multi-class problems to binary ones.

## Status
Done: the Python base (Dataset, a CSV loader, the abstract Classifier with a
score for each class), Euclidean and Manhattan distance, artificial data
(blobs, XOR, circles), the six real data sets checked and documented, k-NN
with four tie-breaking strategies, and decision trees with three impurity
measures, stopping rules and reduced-error pruning.
Next: the evaluation framework (hold-out split, cross-validation, ROC).
See `diary.md` for the weekly log.

## Repository layout
```
documents/   project plan and research notes
  knn/       notes on k-NN and tie-breaking
  decision_trees/  notes on decision trees, impurity and pruning
  research/  notes on data, class scores and ROC curves
product/     Python source code and tests
  src/ml_comparison/   the package
  tests/               pytest tests, one file per module
AI.md        declaration of AI tool use
diary.md     weekly progress log
```

## Getting started
Requires Python 3.12 or newer. First check which version `python3` is:
```
python3 --version
```
If it says 3.12 or newer, use `python3` below. If it is older (the Python that
comes with macOS is 3.9, and some Linux systems have 3.10 or 3.11), install a
newer Python and use its name in the first command instead, for example
`python3.12 -m venv .venv`. With an older Python the install fails with a
confusing "No matching distribution found for numpy" error. The other commands
stay the same. Then, from the `product` folder:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
pytest
```
Inside the environment, `python --version` should show the same version.

## Data
Data sets are not stored in this repository. Download them into a `data/`
folder at the top of the project, which Git ignores:
- Breast Cancer Wisconsin (Diagnostic): https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- Credit Card Fraud Detection: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Iris: https://www.kaggle.com/datasets/uciml/iris
- Wine: https://archive.ics.uci.edu/dataset/109/wine
- Spambase: https://archive.ics.uci.edu/dataset/94/spambase
- German Credit: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

What each file contains, how to load it and its licence are in
`documents/research/data_sets.md`.
