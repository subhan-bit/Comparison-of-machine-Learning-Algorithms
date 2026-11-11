# Tie-breaking in k-NN

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
D (5,4) blue, E (4,6) blue. New dot Q = (2,2).

Euclidean distances from Q: B = 1, A = 1.41, C = 2.83, D = 3.61, E = 4.47.

With k=4 the neighbours are B, A, C and D. The vote is 2 red, 2 blue: a tie.

- Random: red or blue, each with chance 1/2.
- Nearest label: the closest neighbour is B, which is red, so red.
- Reduce k: with k=3 the neighbours are B, A, C, so 2 red, 1 blue: red.
- Distance-weighted (Dudani): d_1 = 1 and d_4 = 3.61, so the weights are
  B = 1, A = 0.84, C = 0.30, D = 0. Red = 1.84, blue = 0.30: red.

Here three strategies pick red, and random picks red only half the time.

## Ties in distance
This is different from a tie in the vote. E used to be at (4,5), exactly as
far from Q as D (both 3.61), so the 4th neighbour could be D or E. The first
version of the code kept whichever came first in the data, so the "2 red,
2 blue" tie in this example depended on the row order. With both counted, the
vote is 2 red and 3 blue, and there is no tie at all. I only noticed because
the fix below broke my own test. So I moved E to (4,6).

The rule now: find the k-th smallest distance, and let every point at that
distance or closer vote. There can be more than k voters, and the vote shares
are divided by the number of voters, so they still add up to 1. When no
distances are tied, nothing changes. Two effects:
- With two classes and an odd k, the number of voters can be even, so a tie
  is possible after all (Spambase has points at exactly the same distance).
- The strategies work on the voters ordered closest first. If the closest
  points are equally far and in different classes, "nearest label" and
  "reduce k" fall back to the order of the rows in the data. This is a last
  resort, and a hidden default, so it is worth remembering for RQ1.

## Why it matters for my project
- RQ1 asks whether the choice of strategy changes the results. To answer it,
  I should also count how often ties happen for each data set and each k,
  because a strategy cannot matter if there are no ties.
- The vote shares are also k-NN's scores for ROC curves. A tie gives two
  labels the same score, so the ROC code must handle equal scores.
- Random tie-breaking must use a seeded random generator so experiments can
  be repeated exactly (audit item 7). A new generator is made from the seed on
  every predict call, so the same data always gives the same answers. A
  generator kept for the whole model gave a different answer each call.

## Still to learn
- Whether another rule for ties in distance (such as choosing among the
  tied points at random) would change the results. My rule is that all of
  them vote.
- Which strategy scikit-learn and Weka use, since I will check my results
  against them and need to know why answers might differ.

## Sources
[1] S. A. Dudani. The distance-weighted k-nearest-neighbor rule. IEEE
    Transactions on Systems, Man, and Cybernetics SMC-6, 325-327, 1976.
[2] T. M. Mitchell. Machine Learning. McGraw-Hill, 1997. Chapter 8.
[3] T. Cover and P. Hart. Nearest neighbor pattern classification.
    IEEE Transactions on Information Theory 13(1), 21-27, 1967.
