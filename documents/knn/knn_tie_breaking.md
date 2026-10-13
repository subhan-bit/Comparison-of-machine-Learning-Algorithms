# Tie-Breaking in k-NN

## What it is
k-NN lets the k nearest neighbours vote, and the label with the most votes
wins. A tie happens when two or more labels share the highest number of
votes, so the vote alone cannot decide. My plan compares four ways to break
a tie (RQ1).

Ties are common:
- An even k with two classes (k=4: 2 red, 2 blue).
- More than two classes, even with an odd k (k=3: 1 red, 1 blue, 1 green).
- Data with repeated values, where several points are the same distance away.

## The four strategies
- Random. Pick one of the tied labels at random. Simple and unbiased, but the
  answer can change between runs, so the random generator needs a fixed seed.
- Nearest label. Pick the tied label of the single closest neighbour.
- Reduce k. Drop the farthest neighbour and vote again with k-1. Repeat
  until there is one winner. It always ends, because k=1 has only one vote.
- Distance-weighted. Closer neighbours count more. Dudani's rule [1] gives
  the neighbour at distance d a weight of (d_k - d) / (d_k - d_1), where d_1
  is the nearest distance and d_k the k-th. So the nearest neighbour has
  weight 1 and the k-th has weight 0 (all weights are 1 if every distance is
  equal). Mitchell describes a similar idea with weight 1/d^2 [2]. Weighted
  votes rarely tie, but if they do, one of the other strategies is needed.

## Example
The dots from knn.md: A (1,1) red, B (2,1) red, C (4,4) blue,
D (5,4) blue, E (4,5) blue. New dot Q = (2,2).

Euclidean distances from Q: B = 1, A = 1.41, C = 2.83, D = 3.61, E = 3.61.

With k=4 the neighbours are B, A, C and one of D or E (they are equally far,
but both are blue, so it does not matter which). The vote is 2 red, 2 blue:
a tie.

- Random: red or blue, each with chance 1/2.
- Nearest label: the closest neighbour is B, which is red, so red.
- Reduce k: with k=3 the neighbours are B, A, C, so 2 red, 1 blue: red.
- Distance-weighted (Dudani): d_1 = 1 and d_4 = 3.61, so the weights are
  B = 1, A = 0.84, C = 0.30, D = 0. Red = 1.84, blue = 0.30: red.

Here three strategies agree, but random would be wrong half the time.

## Why it matters for my project
- RQ1 asks whether the choice of strategy changes the results. To answer it,
  I should also count how often ties happen for each data set and each k,
  because a strategy cannot matter if there are no ties.
- The vote shares are also k-NN's scores for ROC curves. A tie gives two
  labels the same score, so the ROC code must handle equal scores.
- Random tie-breaking must use a seeded random generator so experiments can
  be repeated exactly (audit item 7).

## Still to learn
- How to handle ties in distance, where several points are equally far for
  the k-th place (D and E above). This is different from a tie in the vote.
- Which strategy scikit-learn and Weka use, since I will check my results
  against them and need to know why answers might differ.

## Sources
[1] S. A. Dudani. The distance-weighted k-nearest-neighbor rule. IEEE
    Transactions on Systems, Man, and Cybernetics SMC-6, 325-327, 1976.
[2] T. M. Mitchell. Machine Learning. McGraw-Hill, 1997. Chapter 8.
[3] T. Cover and P. Hart. Nearest neighbor pattern classification.
    IEEE Transactions on Information Theory 13(1), 21-27, 1967.
