# Diary

## Week 2
- Switched from Java to Python, with my supervisor's agreement. NumPy, Matplotlib and pytest let me spend more time on the algorithms (see section 5 of the plan).
- Rebuilt the base in Python: Dataset, the abstract Classifier and load_csv, each with pytest tests.
- Had the code audited with Claude Code (see AI.md) and fixed what it found:
  - Dataset and load_csv now reject bad input (uneven rows, missing values, NaN) with clear errors, and can drop the id column in the Breast Cancer data.
  - Classifier now gives a score for each class, which ROC and cost curves will need later.
  - Brought back distances.py (Euclidean and Manhattan).
- Caught up on a Week 1 task (issue #3): made a Kaggle account and downloaded the six data sets (Breast Cancer, Iris, Wine, Spambase, German Credit, Credit Card Fraud) into data/, which Git ignores.
- Checked every file loads, with Claude Code's help. The real Breast Cancer file has a blank column name only in the header, which my loader rejected, so I fixed that. The UCI files have no header row and German Credit uses spaces, so load_csv can now read both.
- Wrote research notes on the data sets (data_sets.md) and corrected the CSV note, which had assumed the wrong layout.
- Found useful details: German Credit has its own cost matrix (5 to 1), fraud is only 0.17% of transactions, and Wine's features have very different scales.
- Behind plan: k-NN, the tie-breaking strategies and the artificial data are not done yet.
- Next: research note on tie-breaking, artificial data (blobs, XOR, circles), then k-NN with the four tie-breaking strategies, then decision trees.

## Week 1
- Finalised project plan (Java, cost-sensitive extension, multi-class secondary study).
- Set up Java/Maven, GitLab repo structure, Classifier interface and first test.
- Reading: Mitchell ch. 1, 8; started notes on kNN.
- Next: implement 1-NN and kNN with tie-breaking (Week 2).
