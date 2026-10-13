# The Data Sets

## What it is
The six real data sets from section 4 of my plan. They are kept in a local
data/ folder that Git ignores, so they never go into the repository.
I checked every file on 12 October 2026 by loading it with load_csv.

## What is in each one
| Data set | Rows x features | Classes | Source |
|---|---|---|---|
| Breast Cancer Wisconsin (Diagnostic) | 569 x 30 | 357 benign (B), 212 malignant (M) | Kaggle [1] |
| Iris | 150 x 4 | 3 species, 50 each | Kaggle [3] |
| Wine | 178 x 13 | 3 cultivars: 59, 71, 48 | UCI [4] |
| Spambase | 4,601 x 57 | 2,788 not spam (0), 1,813 spam (1) | UCI [5] |
| German Credit (numeric) | 1,000 x 24 | 700 good (1), 300 bad (2) | UCI [6] |
| Credit Card Fraud | 284,807 x 30 | 284,315 genuine (0), 492 fraud (1) | Kaggle [2] |

Licences (copied from each page):
- Breast Cancer: CC BY-NC-SA 4.0 (non-commercial use, give credit, share alike)
- Iris: CC0, public domain
- Credit Card Fraud: Database: Open Database, Contents: Database Contents
- Wine: CC BY 4.0 (any use, give credit)
- Spambase: CC BY 4.0 (any use, give credit)
- German Credit: CC BY 4.0 (any use, give credit)

- Breast Cancer: the 30 features are the mean, standard error and "worst"
  value of 10 measurements of cell nuclei. There is an id column to drop,
  and the header has one blank column name (see preparing_csv_data.md).
- Iris: sepal and petal length and width in cm. There is an Id column to drop.
- Wine: amounts of 13 chemicals in wines from three grape types. No header
  row, and the class is the first column, not the last [4].
- Spambase: 48 word frequencies, 6 character frequencies and 3 measures of
  runs of capital letters. No header row; the class is the last column [5].
- German Credit: the original file has 20 attributes, 13 of them text
  categories. The numeric file turns these into 24 numbers, separated by
  spaces. Its documentation does not list names for the 24 columns [6].
- Credit Card Fraud: two days of card transactions. Time is the seconds
  since the first transaction (0 to 172,792), Amount is the amount, and
  V1 to V28 are already transformed by PCA to hide the real details [2].

## Example
Loading each file:

    load_csv("data/breast_cancer.csv", "diagnosis", drop_columns=["id"])
    load_csv("data/iris.csv", "Species", drop_columns=["Id"])
    load_csv("data/wine/wine.data", "class", column_names=["class", ...13 names])
    load_csv("data/spambase/spambase.data", "spam", column_names=[...57 names, "spam"])
    load_csv("data/german_credit/german.data-numeric", "credit",
             column_names=[...24 names, "credit"], whitespace=True)
    load_csv("data/creditcard.csv", "Class")

## Why it matters for my project
- Costs are real, not made up. German Credit comes with a cost matrix:
  calling a bad customer good costs 5, calling a good customer bad costs 1 [6].
  The Spambase authors say marking good mail as spam is "very undesirable" [5].
  These give the cost-sensitive extension real cost ratios to start from.
- Imbalance. Only 0.17% of transactions are fraud, so always answering
  "genuine" is 99.83% accurate and useless. I need stratified splits and
  ROC or cost, not accuracy (see class_scores_and_roc.md).
- Scale. Feature ranges differ a lot: in Wine, hue is 0.48 to 1.71 but
  proline is 278 to 1680, so proline would dominate k-NN distances. The Wine
  notes suggest standardising, and report 96.1% for 1-NN on standardised
  data [4]. This is a good check for my k-NN and for RQ3 (normalisation).
- Size. The fraud data has 284,807 rows, so brute-force k-NN will be slow.
  This is why the plan has the k-d tree and uses subsamples while developing.
- Sanity checks. Iris and Wine are small and well behaved [4], so they are
  good first tests for each new algorithm.

## Still to learn
- Names for the 24 German Credit numeric columns, which the documentation
  does not give.
- Whether Time should be a feature in the fraud data, since it only says
  when a transaction happened.
- Spambase's non-spam mail came from the authors' own mail, so the word
  "george" and the number "650" point to non-spam [5]. This could make
  results look better than they would be on other people's mail.

## Sources
[1] UCI Machine Learning. Breast Cancer Wisconsin (Diagnostic) Data Set.
    Kaggle. https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
[2] Machine Learning Group, ULB. Credit Card Fraud Detection. Kaggle.
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
[3] UCI Machine Learning. Iris Species. Kaggle.
    https://www.kaggle.com/datasets/uciml/iris
[4] M. Forina et al. Wine recognition data. UCI Machine Learning
    Repository, https://archive.ics.uci.edu/dataset/109/wine (file wine.names).
[5] M. Hopkins, E. Reeber, G. Forman and J. Suermondt. Spambase. Hewlett-
    Packard Labs, 1999. UCI Machine Learning Repository,
    https://archive.ics.uci.edu/dataset/94/spambase (file spambase.DOCUMENTATION).
[6] H. Hofmann. Statlog (German Credit Data). University of Hamburg.
    UCI Machine Learning Repository,
    https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
    (file german.doc).
