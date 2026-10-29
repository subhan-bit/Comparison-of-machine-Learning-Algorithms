# Decision trees

## What it is
A decision tree predicts a label by asking a short list of yes/no questions
about the features, such as "is x <= 3?" [1][2]. Each answer leads down one
branch until a leaf is reached, and the leaf gives the label. The questions
are learnt from the training data, and the finished tree can be read as
if-then rules.

## How the questions are chosen
The tree is built from the top down. At each node:
1. Try every feature, and every threshold between two neighbouring values.
2. For each try, split the rows into a left group (<= threshold) and a right
   group (> threshold).
3. Keep the split that makes the two groups the purest.
4. Do the same again inside each group, until a stopping rule says stop.

"Pure" means most rows in a group have the same label. With p_c the share of
class c in a group, there are three common ways to measure impurity [2]:
- Gini index: 1 - sum of p_c^2
- Entropy: - sum of p_c * log2(p_c)
- Misclassification error: 1 - the largest p_c
All three are 0 for a pure group. A split is scored by the impurity of each
group, weighted by how many rows it has.

## Example
10 dots: 6 red, 4 blue. Before splitting:
- Gini = 1 - (0.6^2 + 0.4^2) = 0.48
- Entropy = -(0.6 log2 0.6 + 0.4 log2 0.4) = 0.971
- Misclassification = 1 - 0.6 = 0.4

Split "x <= 3" gives left = 4 red, right = 2 red and 4 blue.
- Left Gini = 0 (pure). Right Gini = 1 - (1/9 + 4/9) = 0.444.
- Weighted Gini = 0.4 * 0 + 0.6 * 0.444 = 0.267, down from 0.48.

## Why the measure matters
Misclassification error can miss a good split. Take 800 dots, 400 of each
class, and two possible splits [2]:
- Split A: (300, 100) and (100, 300).
- Split B: (200, 400) and (200, 0), where the second group is pure.
Both splits get 200 dots wrong, so misclassification error scores them the
same (0.25). Gini scores A as 0.375 and B as 0.333, so it prefers B, which
made a pure group. This is why Gini and entropy are normally used to grow
the tree.

## Stopping and pruning
A tree grown until every leaf is pure fits the training data perfectly,
including its noise, so it overfits [1].
- Stopping rules end the growth early: a maximum depth, a minimum number of
  rows needed to split, or stop when a node is already pure.
- Pruning grows the full tree and then cuts it back. Reduced-error pruning
  [1][4] turns a node into a leaf if that does not make the accuracy on a
  separate validation set worse.

## Why it matters for my project
- The plan compares the three impurity measures (RQ2). The example above
  shows they can pick different splits.
- The class shares in a leaf are the tree's scores, which ROC curves need.
- Normalising the features should make no difference to a tree, because a
  threshold only cares about order. For k-NN it changes the distances, so
  RQ3 can compare the two.

## Still to learn
- How entropy and Gini compare on the real data sets.
- Other pruning methods, such as cost-complexity pruning [2][3].
- How to make the impurity cost-sensitive (Term 2 extension).

## Sources
[1] T. M. Mitchell. Machine Learning. McGraw-Hill, 1997. Chapter 3.
[2] T. Hastie, R. Tibshirani and J. Friedman. The Elements of Statistical
    Learning, 2nd ed. Springer, 2009. Section 9.2.
[3] L. Breiman, J. Friedman, R. Olshen and C. Stone. Classification and
    Regression Trees. Wadsworth, 1984.
[4] J. R. Quinlan. Simplifying decision trees. International Journal of
    Man-Machine Studies 27(3), 221-234, 1987.
