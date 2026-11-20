# The evaluation framework

## What it is
The code that tests a classifier fairly. It splits the data, trains a model on
one part, tests it on the other, counts the right and wrong answers, and turns
the class scores into a ROC curve and an AUC. k-NN, the decision tree and later
the SVM all go through the same functions, so the comparison between them is
fair. The basic rule is to test on rows the model has not seen, because the
accuracy on its own training rows says little [1][2].

## The pieces
- Split. train_test_split cuts the data into a training part and a test part.
  The same seed gives the same split. With stratify=True every class is split
  on its own, so the test part has the same share of each class as the whole
  data (to within a row), and a class with two or more rows has at least one
  row on each side.
- Folds. stratified_k_fold cuts the data into k folds. Each fold takes a turn
  as the test part and the other folds are the training part, so every row is
  tested exactly once. Each class is dealt out to the folds in turn, so every
  fold has about the same share of each class (to within one row). Stratified
  folds are the usual advice for estimating accuracy [2]. The same seed gives
  the same folds, so two algorithms can be tested on exactly the same folds.
- Confusion matrix and accuracy. The matrix counts the rows by true class
  (down) and predicted class (across), so the right answers are on the
  diagonal. Accuracy is the diagonal over all the rows.
- TPR and FPR. The true and false positive rates of the positive class (the
  one we care about, such as fraud). Every other class counts together as
  negative.
- ROC and AUC. roc_curve takes the true labels and the score of the positive
  class and gives one point for each different score. auc is the area under
  the curve, by the trapezoid rule. Rows with the same score are one step,
  and count as half a win (see class_scores_and_roc.md) [3].
- Standardising. standardise gives every feature mean 0 and spread 1. The mean
  and spread come from the training rows only, and the test rows are changed
  with them.
- evaluate. Trains a model on a training part, tests it on a test part, and
  returns the accuracy and the confusion matrix, and, if there is a positive
  class, the TPR, FPR and AUC. The score column is found by the name of the
  class, and every split keeps the class list of the whole data, so it is the
  right column even when a rare class has no rows in the training part.
- cross_validate. Runs evaluate on every fold and gives the results of each
  fold, the mean accuracy and the mean AUC. It takes a function that makes a
  new model, so every fold gets a fresh one. With normalise=True it
  standardises inside each fold.

      results = cross_validate(lambda: KNN(k=5), data, k=5, seed=0,
                               normalise=True)
      results["mean_accuracy"], results["mean_auc"]

## Example
1. Folds. Ten transactions: rows 0 and 1 are fraud, the other eight are
   genuine. I want 2 folds of 5 rows. If I picked the 5 test rows at random,
   112 of the 252 possible choices (44%) would leave one side with no fraud at
   all, and the TPR and AUC of that side could not be worked out.
   stratified_k_fold deals the two frauds to different folds, then the genuine
   rows, so with seed 0 the folds are
   - fold 0: rows 0, 3, 4, 5, 7 (1 fraud, 4 genuine)
   - fold 1: rows 1, 2, 6, 8, 9 (1 fraud, 4 genuine)
   Seeds 1 and 2 pick other rows, but again 1 fraud and 4 genuine in each fold.

2. Measuring one test part. The five test transactions of
   class_scores_and_roc.md with their fraud scores: 0.9 F, 0.8 G, 0.7 F,
   0.4 G, 0.2 G (F = fraud, G = genuine). The scores of k-NN and of the tree
   add up to 1 for each row, so with two classes predict says "fraud" when
   the fraud score is above 0.5. It says F, F, F, G, G, which gives this
   confusion matrix:

                     predicted fraud   predicted genuine
       true fraud           2                  0
       true genuine         1                  2

   - accuracy = (2 + 2) / 5 = 0.8
   - TPR = 2 / (2 + 0) = 1 and FPR = 1 / (1 + 2) = 1/3
   - AUC = 5/6 = 0.83, the same as the table in class_scores_and_roc.md
   Accuracy and the rates describe one threshold (0.5). The AUC only depends
   on the order of the scores, so it does not depend on any threshold.

3. Standardising inside a fold. Training amounts 10, 20, 30 and a test amount
   40. The training mean is 20 and the spread (the standard deviation, dividing
   by the number of rows) is sqrt(200/3) = 8.16. The training amounts become
   -1.22, 0, 1.22 and the test amount becomes (40 - 20) / 8.16 = 2.45.
   If all four amounts were standardised together, the mean would be 25 and
   the spread 11.18, the test amount would become 1.34, and the training
   amounts would change too (-1.34, -0.45, 0.45). The training would have
   been shaped by a row it should not have seen.

4. k-NN and the tree through the same function. cross_validate with 5 folds
   and seed 0 on 200 artificial points each. Mean accuracy / mean AUC. The
   tree has the default settings (Gini, no depth limit).

   | Data set | k-NN (k=5)     | Decision tree  |
   |----------|----------------|----------------|
   | blobs    | 100.0% / 1.000 | 98.5% / 0.985  |
   | circles  | 100.0% / 1.000 | 100.0% / 1.000 |
   | xor      | 96.0% / 0.993  | 94.5% / 0.949  |

5. Why standardise at all. The blobs again, plus one useless feature of random
   numbers on a scale of about 1000. k-NN with k=3 and 5 folds:

   | Folds                    | Accuracy | AUC   |
   |--------------------------|----------|-------|
   | as they are              | 55.8%    | 0.562 |
   | standardised (normalise) | 100.0%   | 1.000 |

   The useless feature decides every distance until it is scaled down. The real
   Wine data has the same problem: proline runs from 278 to 1680 and hue from
   0.48 to 1.71 (data_sets.md).

## Sources
[1] T. Hastie, R. Tibshirani and J. Friedman. The Elements of Statistical
    Learning, 2nd ed. Springer, 2009. Section 7.10 (cross-validation, with
    7.10.2 on doing it the wrong and the right way).
[2] R. Kohavi. A study of cross-validation and bootstrap for accuracy
    estimation and model selection. IJCAI, 1995.
[3] T. Fawcett. An introduction to ROC analysis. Pattern Recognition Letters
    27(8), 2006.
