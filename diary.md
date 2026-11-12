# Diary

## Week 7
- Plan: write the two early reports and start the evaluation framework.
- Done: fixed the problems the audits found (decision tree, tree hang and XOR rule, k-NN, data loading, set-up; issues #9, #11 to #14); reviewed the project with mutation testing and added the missing tests (#17); CI, ruff and README updates (#18); loader clean-ups (#19); moved the report to the new template. The audits took the time, so the framework and the reports did not start.
- Next week: start the evaluation framework (#10), then write the two early reports.

## Week 6
- Plan: the two early reports, count k-NN ties on each data set.
- Done: decision trees merged (#7); counted how often the k-NN vote ties on each data set (#8, scripts/count_ties.py); German Credit ties most (38.7% at k=2).
- Next week: the two early reports, then the evaluation framework.

## Week 5
- Plan: decision trees and the two early reports.
- Done: built decision trees (#7) with three impurity measures, stopping rules, leaf scores and pruning; a code review fixed prune(); supervisor feedback: more motivation in the abstract and a basic GUI in Term 1.
- Next week: the two early reports, count ties on each data set, then the evaluation framework.

## Week 4
- Plan: count ties on each data set, start decision trees.
- Done: no project work, because of deadlines in other modules.
- Next week: count ties, start decision trees.

## Week 3
- Plan: finish k-NN with tie-breaking and the artificial data.
- Done: finished the Week 2 work (#4): tie-breaking notes, artificial data (blobs, XOR, circles) and k-NN with four tie-breaking strategies.
- Next week: count ties on each data set, then decision trees.

## Week 2
- Plan: 1-NN and k-NN with tie-breaking.
- Done: switched from Java to Python (supervisor agreed); rebuilt Dataset, Classifier and load_csv with tests; downloaded and checked the six data sets (#3). k-NN was not done.
- Next week: artificial data, then k-NN with four tie-breaking strategies.

## Week 1
- Plan: finalise the project plan and set up the project.
- Done: finalised the plan; set up Java/Maven, the GitLab repository and the Classifier interface; read Mitchell chapters 1 and 8.
- Next week: 1-NN and k-NN with tie-breaking.
