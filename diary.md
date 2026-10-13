# Diary

## Week 3
- Finished the Week 2 work (issue #4), with Claude Code's help (see AI.md):
  - Research note on the four tie-breaking strategies, with a worked example where k=4 gives a 2-2 tie.
  - Artificial data sets (blobs, XOR, circles) with a fixed seed, so the same seed always gives the same data.
  - A k-NN classifier that follows the Classifier interface. Its scores are the share of the vote for each class. It gets 100% on blobs and 97.5% on XOR, which no straight line can split.
  - The four tie-breaking strategies: random (seeded), nearest label, reduce k and distance-weighted (Dudani).
- What I learnt: before adding the strategies, a 2-2 tie at k=4 was decided by alphabetical order, so "blue" won only because it comes before "red". Three of the four strategies pick red on the same example. This showed me why RQ1 matters: a hidden default can decide the answer.
- Decided to keep the scores as plain vote shares and let the strategy only decide ties, so the ROC curves later use the real votes.
- Moved the k-NN notes into documents/knn and brought the README up to date.
- Change from the plan: I am writing the report in Word using the department's template, not Overleaf.
- Next: count how often ties happen on each data set (RQ1), then decision trees (Weeks 3 to 4).

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
