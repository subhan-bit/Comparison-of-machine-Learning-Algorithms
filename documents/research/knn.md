# k-Nearest Neighbours (kNN)

## What it is
kNN guesses the label of a new example by looking at the k training examples
that are most like it, and going with the label most of them have [1][2].
It does no real learning in advance: it stores the training data and does
the work when asked to predict. Even the simplest version, 1-NN, is a strong
method given enough data [3].

## How "most like it" is measured
Each example is a list of numbers, so we can measure the distance between two
examples. A small distance means they are similar.
- Euclidean distance: the straight-line distance.
  sqrt((a1-b1)^2 + (a2-b2)^2 + ...)
- Manhattan distance: the distance if you can only move along a grid.
  |a1-b1| + |a2-b2| + ...

## Example
Training dots: A (1,1) red, B (2,1) red, C (4,4) blue, D (5,4) blue, E (4,5) blue.
New dot Q = (2,2). Closest to farthest: B, A, C, D, E.
- k=1: B is red, so Q is red.
- k=3: red, red, blue, so Q is red.
- k=5: 2 red, 3 blue, so Q is blue.

## Choosing k
- k too small: one odd example can fool the vote.
- k too large: far-away examples get a vote and can outvote the close ones,
  as with k=5 above.
A good k is in between, and I will find it by testing on data.

## Why it matters for my project
- Ties (for example k=4 with 2 red and 2 blue) need a rule. My plan compares
  four: random, nearest label, reduce k, and distance-weighted (RQ1).
- Distance depends on the size of each feature, so features should be
  normalised first (RQ3).

## Still to learn
- How each tie-breaking strategy works in detail (next lesson).
- Why normalising the features matters for distance.
- What goes wrong when there are many features (curse of dimensionality).

## Sources
[1] T. M. Mitchell. Machine Learning. McGraw-Hill, 1997. Chapter 8.
[2] T. Hastie, R. Tibshirani and J. Friedman. The Elements of Statistical
    Learning, 2nd ed. Springer, 2009. Section 13.3.
[3] T. Cover and P. Hart. Nearest neighbor pattern classification.
    IEEE Transactions on Information Theory 13(1), 21-27, 1967.