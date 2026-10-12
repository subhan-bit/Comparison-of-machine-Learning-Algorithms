# Diary

## Week 2
- Switched from Java to Python, with my supervisor's agreement. NumPy, Matplotlib and pytest let me spend more time on the algorithms (see section 5 of the plan).
- Rebuilt the base in Python: Dataset, the abstract Classifier and load_csv, each with pytest tests.
- Had the code audited with Claude Code (see AI.md) and fixed what it found:
  - Dataset and load_csv now reject bad input (uneven rows, missing values, NaN) with clear errors, and can drop the id column in the Breast Cancer data.
  - Classifier now gives a score for each class, which ROC and cost curves will need later.
  - Brought back distances.py (Euclidean and Manhattan).
- Behind plan: k-NN, the tie-breaking strategies and the artificial data are not done yet, and the Kaggle data is not downloaded.
- Next: k-NN with the four tie-breaking strategies, artificial data (blobs, XOR, circles), download the Kaggle data, then start decision trees.

## Week 1
- Finalised project plan (Java, cost-sensitive extension, multi-class secondary study).
- Set up Java/Maven, GitLab repo structure, Classifier interface and first test.
- Reading: Mitchell ch. 1, 8; started notes on kNN.
- Next: implement 1-NN and kNN with tie-breaking (Week 2).
