# How often k-NN ties happen

## What it is
RQ1 asks whether the tie-breaking strategy changes k-NN's results. A strategy
only does something when the vote is a tie, so the first step is to count how
often that happens on the real data sets. A tie here means two or more classes
share the most votes (has_tie in knn.py). If several points are equally far
for the k-th place, they all vote, so there can be more than k votes (see
knn_tie_breaking.md). The table below is from the script after issue #15, which
changed how the rows are split (see below), so every row moved a little
compared with the first version.

## How I counted
scripts/count_ties.py does this for each data set:
1. Split the rows 70% for training and 30% for testing, with each class split
   on its own (a stratified split), so a rare class is in both sets in the
   same share.
2. Standardise every feature (mean 0, spread 1), using the training rows only.
3. For k = 1 to 10, train k-NN and count the test points whose vote is a tie.

Each data set has its own random generator, seeded from its name. A data set
therefore gives the same numbers whatever the other data sets are and in what
order they come, and running the script twice gives the same table.

The fraud data has 284,807 rows, which is too many for brute-force k-NN, and
only 0.17% of them are fraud. A plain random subsample of 3,000 rows holds about
5 frauds, so its tie counts said nothing about fraud. The fraud row now uses
every one of the 492 frauds plus 3,000 random genuine rows: 3,492 rows, of which
14% are fraud. This is an enriched sample, with far more fraud than the real
data. It is one random split, so the numbers would move a little with a
different seed.

## Results
% of test points where the vote is a tie:

| Data set | Classes | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 | k=10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Iris | 3 | 0.0 | 4.4 | 0.0 | 4.4 | 0.0 | 2.2 | 0.0 | 2.2 | 0.0 | 2.2 |
| Wine | 3 | 0.0 | 3.6 | 0.0 | 3.6 | 0.0 | 1.8 | 0.0 | 1.8 | 0.0 | 0.0 |
| Breast Cancer | 2 | 0.0 | 7.0 | 0.0 | 1.7 | 0.0 | 2.3 | 0.0 | 1.2 | 0.0 | 1.7 |
| Spambase | 2 | 0.1 | 9.8 | 0.1 | 5.5 | 0.3 | 4.5 | 0.3 | 3.3 | 0.0 | 3.4 |
| German Credit | 2 | 0.0 | 32.2 | 0.0 | 21.6 | 0.0 | 16.6 | 0.0 | 14.0 | 0.0 | 13.6 |
| Credit Card Fraud (enriched) | 2 | 0.0 | 4.0 | 0.0 | 1.1 | 0.0 | 1.1 | 0.0 | 0.4 | 0.0 | 0.5 |

Rows used. The fraud row is an enriched sample: every fraud plus 3,000 genuine rows.

| Data set | Rows | Training rows | Test rows | Test rows per class |
|---|---|---|---|---|
| Iris | 150 | 105 | 45 | Iris-setosa: 15, Iris-versicolor: 15, Iris-virginica: 15 |
| Wine | 178 | 123 | 55 | 1: 18, 2: 22, 3: 15 |
| Breast Cancer | 569 | 397 | 172 | B: 108, M: 64 |
| Spambase | 4,601 | 3,220 | 1,381 | 0: 837, 1: 544 |
| German Credit | 1,000 | 699 | 301 | 1: 211, 2: 90 |
| Credit Card Fraud | 3,492 | 2,444 | 1,048 | 0: 900, 1: 148 |

## What the numbers show
- With two classes and k voters, an odd k cannot tie, because an odd number
  of votes cannot split evenly. So the odd columns of the two-class data sets
  are 0, with one exception: Spambase has points at exactly the same
  distance, so more than k points can vote and the number of voters can be
  even. Its odd columns are small but not always 0 (up to 0.3%).
- With an even k, ties are common, and German Credit has by far the most:
  at k=2, 32.2% of test points tie. This suggests its two classes overlap a
  lot, so a point's nearest neighbours often disagree.
- Ties get rarer as k grows, because an exact split of more votes is less
  likely (German Credit: 32.2% at k=2, down to 13.6% at k=10).
- With three classes a tie can happen at an odd k (1-1-1 at k=3), but it
  never did here. At an even k Iris and Wine tie like the two-class data
  sets do (4.4% and 3.6% at k=2). The first version of this table had no ties
  at all for Iris, which was only a feature of that one split.
- The fraud row is an enriched sample: 14% fraud, with 148 frauds in the test
  set. It shows ties at an even k (4.0% at k=2, 1.1% at k=4 and k=6, 0.4% at
  k=8). On the real data, with 0.17% fraud, almost every neighbour is genuine
  and the vote is nearly always unanimous, so the real tie rate is expected to
  be lower than in this row.

## Why it matters for my project
- For RQ1, the tie-breaking strategy can only change results at even k
  (and, in theory, any k with three classes). The best places to compare
  the four strategies are German Credit and Spambase at k=2 and k=4.
- At k=2 on German Credit, the tie-breaking rule decides the answer for
  about a third of the test points (32%), so it could change the accuracy
  a lot. A library's hidden default would matter here.

## Still to learn
- Whether the counts change much with other random splits. Cross-validation
  (Weeks 6 to 7) will give an average over several splits.
- How often ties in distance happen. In the first run only Spambase was
  affected by that rule; the other data sets gave the same counts as before it.
- The tie rate on the real fraud data (0.17% fraud). That needs a faster k-NN
  than brute force, such as the k-d tree (Week 9).
- How much each strategy changes the accuracy at even k (the actual RQ1
  experiment, Week 8).

## Sources
[1] T. M. Mitchell. Machine Learning. McGraw-Hill, 1997. Chapter 8.
[2] S. A. Dudani. The distance-weighted k-nearest-neighbor rule. IEEE
    Transactions on Systems, Man, and Cybernetics SMC-6, 325-327, 1976.
