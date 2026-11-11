# Diary

## Week 7
- Received an audit of the repository (3 November) that listed 19 problems: 10 older ones that were still open and 9 new ones. Its method section says it used AI reviewers (see AI.md). I split the problems into four groups and started with the decision tree, because the evaluation framework and the ROC curves will depend on it (issue #9, branch feature/tree-fixes).
- Fixed the decision tree with Claude Code's help (see AI.md), writing a failing test first for each change, in seven small commits:
  - Settings: max_depth and min_samples_split must be whole numbers, with a clear error. Ruff said to use TypeError for a wrong type, which also removed the raw TypeError the audit found.
  - Pruning: an empty validation set is now an error, and a split that no validation row reaches is kept. Before, such a split was always cut, because it got 0 wrong either way and a tie goes to the leaf.
  - Splitting: the tree now splits only when the impurity drops. Before, misclassification error on the labels 0, 0, 1, 0, 0 made 4 leaves although no split helped.
  - Scores: leaves now give smoothed shares, (count + 1) / (rows + classes). Before, a full tree gave only 0 and 1, which would make ROC curves useless. It never changes which class wins, so the predictions are the same.
  - Speed: the threshold search sorts each feature once and scores all thresholds from running class counts. A full tree on 2,000 rows with 10 features went from about 6 seconds to 0.09 seconds, so the fraud data may not need a subsample for the tree.
  - Recursion: building, counting and pruning use a list of nodes still to visit instead of recursion, so 2,000 alternating rows no longer give a RecursionError.
  - Notes: updated decision_trees.md and added a docstring.
- What I learnt:
  - The tree was right on small data and the problems only showed up at scale. The artificial data sets I tested on were too small to show slow search, deep trees or all-0-or-1 scores. Next time I will test a new algorithm on one large or awkward case as well.
  - Two splits that are equal in theory can differ in the 16th decimal place, and the old code then picked the later one by accident. I made scores within 1e-12 count as equal and kept the first, as my notes say, and checked the fast search against the slow one on 900 random cases.
  - I did the speed fix before the recursion fix, because a deep-tree test took too long to run with the slow search. Fixing things in the order that keeps the tests fast saved time.
- Effect on my results: one number in the report changes (blobs with misclassification error now has 4 leaves, not 5; the accuracy is still 100%). The pruning result (85 leaves down to 2, 81.0% up to 99.5%) and the other table values are the same. k-NN and the tie counts are not affected.
- Behind plan: the audit fixes took the time, so the evaluation framework has not started and the two early reports are still to be written.
- Next: the other audit groups (k-NN, the tie-count script and its uninformative fraud row, set-up problems), then the early reports and the evaluation framework.

## Week 6
- Decision trees (issue #7) merged into main on GitLab and mirrored to GitHub.
- Counted how often the k-NN vote ends in a tie on each data set (issue #8), with Claude Code's help (see AI.md):
  - Added has_tie, which finds the rows where two or more classes share the top score, with tests that use the k=4 tie from my notes.
  - Wrote scripts/count_ties.py: a random 70/30 split (seed 0), features standardised on the training rows only, then k = 1 to 10. The fraud data uses a 3,000-row subsample because brute-force k-NN is too slow on 284,807 rows.
  - Results are in documents/knn/tie_counts.md. With two classes and an odd k there are never ties, as expected. With an even k, German Credit ties the most: 38.7% of test points at k=2, still 8.7% at k=10. Spambase is next (12.2% at k=2). Iris and the fraud subsample had no ties.
- This changes how I will test RQ1: the tie-breaking strategy can only matter at an even k, so the comparison should focus on German Credit and Spambase at k=2 and k=4.
- The counts come from one random split, so I will repeat them with cross-validation once it is built.
- Next: the two early reports (k-NN, decision trees), then the evaluation framework (hold-out split, stratified cross-validation, confusion matrix, ROC and AUC, normalisation).

## Week 5
- Checked progress against the plan: about two weeks behind. Decision trees (Weeks 3 to 4) had not been started, and the two early reports are due this week.
- Built decision trees (issue #7), with Claude Code's help (see AI.md), writing the tests first for each step:
  - Research note on decision trees, with a worked example (6 red, 4 blue gives Gini 0.48, entropy 0.971, misclassification 0.4).
  - The three impurity measures, and a threshold search that tries every feature and every point halfway between two values.
  - A DecisionTree class that follows the Classifier interface, with three stopping rules (pure node, max depth, min samples to split). Its scores are the class shares in the leaf, for ROC curves later.
  - Reduced-error pruning, which turns a node into a leaf if that is no worse on a validation set. On blobs with 15% of the labels flipped, pruning made the tree smaller and its test accuracy no worse.
- Found an example for RQ2 (Hastie et al.): two splits that misclassification error scores the same (0.25), while Gini prefers the one that makes a pure group (0.333 against 0.375).
- Known limitation: the threshold search is slow on large data, so the fraud data set will need a subsample.
- Had the code reviewed (see AI.md). It found that prune() crashed with an unclear error when the validation set had a label the tree never saw in training, or the wrong number of columns. Fixed both, with a test for each, and corrected a comment that said the XOR points were spread evenly when they are random.
- Feedback from my supervisor on the project plan: the abstract needs more motivation, and I should build a basic GUI in Term 1 so there is a demo. I will build the GUI at the end of Term 1, after the other Term 1 work.
- Next: write the two early reports (k-NN tie-breaking, decision trees), count ties on each data set (RQ1), then the evaluation framework (Weeks 6 to 7).

## Week 4
- No project commits this week. I am working on other modules because I have a marked quiz and other assignment deadlines this week.
- Next: when I come back, count how often ties happen on each data set (RQ1), then start decision trees.

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
