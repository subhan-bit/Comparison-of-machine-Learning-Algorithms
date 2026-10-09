# Comparison of Machine Learning Algorithms

Cost-sensitive evaluation and multi-class reduction strategies.
Final Year Individual Project, Department of Computer Science,
Royal Holloway, University of London.

**Author:** Muhammad Subhan Ahmad
**Supervisor:** [supervisor's name]

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
Week 2. The project plan is submitted and the k-NN research notes are written.
Next: Python project structure, then 1-NN and k-NN with tie-breaking.
See `diary.md` for the weekly log.

## Repository layout
```
documents/   project plan, research notes and reports
product/     source code and tests (currently an early Java skeleton,
             to be replaced by the Python implementation)
AI.md        declaration of AI tool use
diary.md     weekly progress log
```

## Getting started (planned, will be updated when the Python code is added)
Requires Python 3.11 or newer.
```
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib pytest
```

## Data
Data sets are not stored in this repository. They are downloaded from Kaggle
into a local `data/` folder, which Git ignores:
- Credit Card Fraud Detection: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Breast Cancer Wisconsin (Diagnostic): https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- Others from Kaggle or the UCI repository (https://archive.ics.uci.edu)

Check each data set's licence before use.

