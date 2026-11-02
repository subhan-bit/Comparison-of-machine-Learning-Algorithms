# How often k-NN ties happen

## What it is
RQ1 asks whether the tie-breaking strategy changes k-NN's results. A strategy
only does something when the vote is a tie, so the first step is to count how
often that happens on the real data sets. A tie here means two or more classes
share the most votes (has_tie in knn.py). Ties in distance, where several
points are equally far for the k-th place, are not counted.

## How I counted
scripts/count_ties.py does this for each data set:
1. Split the rows at random: 70% for training, 30% for testing (seed 0).
2. Standardise every feature (mean 0, spread 1), using the training rows only.
3. For k = 1 to 10, train k-NN and count the test points whose vote is a tie.
The fraud data has 284,807 rows, which is too many for brute-force k-NN, so
it uses a random subsample of 3,000 rows. This is one random split, so the
numbers would move a little with a different seed.

## Results
% of test points where the vote is a tie:

| Data set | Classes | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 | k=10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Iris | 3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Wine | 3 | 0.0 | 3.7 | 0.0 | 3.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.9 |
| Breast Cancer | 2 | 0.0 | 4.7 | 0.0 | 1.8 | 0.0 | 2.3 | 0.0 | 1.8 | 0.0 | 0.6 |
| Spambase | 2 | 0.0 | 12.2 | 0.0 | 6.2 | 0.0 | 4.3 | 0.0 | 3.5 | 0.0 | 3.3 |
| German Credit | 2 | 0.0 | 38.7 | 0.0 | 22.0 | 0.0 | 15.0 | 0.0 | 11.7 | 0.0 | 8.7 |
| Credit Card Fraud | 2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## What the numbers show
- With two classes and an odd k there can never be a tie, because an odd
  number of votes cannot split evenly. Every odd column for the two-class
  data sets is 0, as expected.
- With an even k, ties are common, and German Credit has by far the most:
  at k=2, 38.7% of test points tie. This suggests its two classes overlap a
  lot, so a point's nearest neighbours often disagree.
- Ties get rarer as k grows, because an exact split of more votes is less
  likely (German Credit: 38.7% at k=2, down to 8.7% at k=10).
- With three classes a tie can happen at an odd k (1-1-1 at k=3), but it
  never did here, and Iris had no ties at all on this split.
- The fraud subsample has no ties, because fraud is only 0.17% of the rows,
  so almost every neighbour is genuine and the vote is nearly always
  unanimous.

## Why it matters for my project
- For RQ1, the tie-breaking strategy can only change results at even k
  (and, in theory, any k with three classes). The best places to compare
  the four strategies are German Credit and Spambase at k=2 and k=4.
- At k=2 on German Credit, the tie-breaking rule decides the answer for
  more than a third of the test points, so it could change the accuracy
  a lot. A library's hidden default would matter here.

## Still to learn
- Whether the counts change much with other random splits. Cross-validation
  (Weeks 6 to 7) will give an average over several splits.
- How often ties in distance happen, especially in data with repeated values.
- How much each strategy changes the accuracy at even k (the actual RQ1
  experiment, Week 8).

## Sources
[1] T. M. Mitchell. Machine Learning. McGraw-Hill, 1997. Chapter 8.
[2] S. A. Dudani. The distance-weighted k-nearest-neighbor rule. IEEE
    Transactions on Systems, Man, and Cybernetics SMC-6, 325-327, 1976.
