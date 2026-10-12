# Class Scores and ROC Curves

## What it is
A classifier can give back just a label ("fraud"), or a score for each class
that shows how strongly it leans towards that class. The label is then simply
the class with the highest score. Scores are more useful, because they let us
move the line between "fraud" and "genuine" and see what happens [1].

## How the scores are used
For two classes, take the score for the positive class (the one we care about,
like fraud or malignant). Pick a threshold t and say "positive" when the score
is at least t. Each threshold gives two numbers:
- True positive rate (TPR): the share of real positives we catch.
  TP / (TP + FN)
- False positive rate (FPR): the share of real negatives we wrongly flag.
  FP / (FP + TN)

Trying every threshold and plotting TPR against FPR gives the ROC curve.
The area under it (AUC) is the chance that a random positive gets a higher
score than a random negative [1]. Guessing gives the diagonal line, AUC 0.5.

When mistakes cost different amounts, the scores also give the cheapest
decision. If p is the score for "positive", say positive when the expected
cost of saying negative is higher:
p * cost(missed positive) > (1 - p) * cost(false alarm) [2].
This only works if p behaves like a probability.

## Example
Five test transactions, with their fraud scores. F = fraud, G = genuine.
0.9 F, 0.8 G, 0.7 F, 0.4 G, 0.2 G (2 frauds, 3 genuine).

| Threshold | Caught frauds (TPR) | False alarms (FPR) |
|-----------|---------------------|--------------------|
| 0.9       | 1 of 2 = 0.5        | 0 of 3 = 0         |
| 0.8       | 1 of 2 = 0.5        | 1 of 3 = 0.33      |
| 0.7       | 2 of 2 = 1          | 1 of 3 = 0.33      |
| 0.4       | 2 of 2 = 1          | 2 of 3 = 0.67      |
| 0.2       | 2 of 2 = 1          | 3 of 3 = 1         |

AUC: of the 6 (fraud, genuine) pairs, the fraud has the higher score in 5,
so AUC = 5/6 = 0.83.

Cost: if missing a fraud costs 10 and a false alarm costs 1, flag anything
with p * 10 > (1 - p) * 1, which is p > 1/11, about 0.09.

## Why it matters for my project
- The fraud data is about 0.17% fraud, so accuracy is misleading. The plan
  judges the algorithms by ROC, AUC and expected cost instead.
- So every algorithm must give scores, not only labels. This is why my
  Classifier has predict_scores, and predict just picks the highest score.
- Where each algorithm's score comes from (from the plan):
  - k-NN: the share of the k neighbours voting for each class.
  - Decision tree: the class proportions in the leaf.
  - SVM: the signed distance from the boundary. This is not a probability.

## Still to learn
- Precision-recall curves, which may show more than ROC when positives are
  very rare (suggested in my audit for the fraud data).
- How to turn SVM distances into something like a probability, so the
  cost rule above can be used.
- Cost curves, which the plan uses to show expected cost over many cost ratios.

## Sources
[1] T. Fawcett. An introduction to ROC analysis. Pattern Recognition Letters
    27(8), 2006.
[2] C. Elkan. The foundations of cost-sensitive learning. IJCAI, 2001.
