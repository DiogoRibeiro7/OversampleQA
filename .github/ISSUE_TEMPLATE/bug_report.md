---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Import oversampleqa
2. Load dataset with '...'
3. Run validation with '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Code Example**
```python
# Please provide a minimal code example that reproduces the issue
from oversampleqa import validate_oversampling
# ... your code here
```

**Error Output**
```
# Paste the full error message here
```

**Environment**

Run `oversampleqa doctor` and paste the table. It reports the OversampleQA,
Python, platform, numpy, pandas, scikit-learn, imbalanced-learn, scipy and
matplotlib versions, which is everything below and less work than filling it in
by hand.

```
# paste `oversampleqa doctor` output here
```

**Run parameters**

An error rate cannot be reproduced without these. If you have a
`<artifact>.metadata.json` sidecar or a `ValidationReport`, paste that instead
and skip the list -- both already contain all of it.

 - Oversampler and its settings: [e.g. SMOTE(k_neighbors=5)]
 - Distance metric: [e.g. hassanat]
 - `hidden_ratio`: [e.g. 0.1]
 - `random_state` / seeds: [e.g. 42; leave blank if unset, which is itself
   the answer to why a result did not reproduce]
 - Repeats or folds: [e.g. n_repeats=5]
 - Reference mode: [e.g. hidden_minority]
 - Calibration or fidelity run? [yes/no]

**Dataset**

We rarely need the data itself, but we do need its shape:

 - Rows and features: [e.g. 4800 x 30]
 - Class balance: [e.g. 4700 majority / 100 minority]
 - Minority label: [e.g. 1]
 - Feature types: [numeric only / mixed / categorical encoded how?]
 - Source: [public dataset and version, generated with a seed, or private]

**Additional context**
Add any other context about the problem here.
