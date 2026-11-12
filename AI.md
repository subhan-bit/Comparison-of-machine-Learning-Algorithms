# Use of AI

- Tool: Claude Code and Claude (claude.ai), models Claude Opus 5.5, Sonnet 5.5 and Haiku 5.5
- Provider: Anthropic
- URL: https://claude.com/claude-code

## What the AI did

Claude Code worked in this repository from my instructions. It wrote the first version of most of the code, the tests and the notes, and I then changed, ran and reviewed them.

- Code: Dataset, the CSV loader, the abstract Classifier, the distance functions, the artificial data sets, k-NN with its four tie-breaking strategies, decision trees with three impurity measures and pruning, and scripts/count_ties.py.
- Tests: the pytest tests for all of the above, written before each change.
- Fixes: the problems found by the audits of 3 and 11 November (issues #9 and #11 to #14), and by the project review (issue #17).
- Reviews: an audit of the repository (Week 2), code reviews (Weeks 5 and 7), and a mutation test, which is a script that makes small deliberate changes to the code and checks that a test fails. The review was written up as a PDF kept outside the repository.
- Set-up: the CI pipeline (.gitlab-ci.yml), the README and the ruff and coverage settings.
- Writing: first drafts of the research notes, the diary entries, the merge request texts, and the facts, tables and outline of the interim report. Claude Code also put the interim report into the department's template.

## What I did myself

- I chose the project, the plan, the algorithms and the research questions, and agreed the switch from Java to Python with my supervisor.
- I decided what to keep. For example, k-NN keeps its scores as plain vote shares and the tie-breaking strategy only decides ties, and in pruning a tie goes to the leaf.
- I created the GitLab issues and merge requests, approved every commit and pushed the code.

## How I checked the output

- I ran every test and ruff before each commit (105 tests, 100% line coverage).
- I wrote or asked for a failing test before each change, and checked each new test against the broken version of the code it was written for.
- I read the changes until I could explain them.
- The text of the report is my own writing, built on the facts and drafts above.
