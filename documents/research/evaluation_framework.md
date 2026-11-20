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

## Sources
[1] T. Hastie, R. Tibshirani and J. Friedman. The Elements of Statistical
    Learning, 2nd ed. Springer, 2009. Section 7.10 (cross-validation, with
    7.10.2 on doing it the wrong and the right way).
[2] R. Kohavi. A study of cross-validation and bootstrap for accuracy
    estimation and model selection. IJCAI, 1995.
[3] T. Fawcett. An introduction to ROC analysis. Pattern Recognition Letters
    27(8), 2006.
