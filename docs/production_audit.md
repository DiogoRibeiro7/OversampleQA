# Production Audit Workflow

Use this workflow when an oversampling decision affects a real model, report or
operational process. The goal is to leave an audit trail that another person can
rerun and challenge.

## 1. Freeze the Evaluation Setup

Record the dataset version, row filtering, preprocessing, target definition,
class counts and feature list. Keep preprocessing inside the model-selection
pipeline so held-out rows do not leak into encoders, scalers, imputers or
feature selectors.

Create a checked-in OversampleQA manifest:

```yaml
version: 1
output: audit
defaults:
  target: target
  minority_label: 1
  oversampler: SMOTE
  metric: hassanat
  hidden_ratio: 0.1
  random_state: 42
  n_repeats: 20
  export:
    - json
    - markdown
datasets:
  production_sample:
    path: data.csv
experiments:
  - name: smote-calibrated
    dataset: production_sample
    calibrate: true
```

Then run it with:

```bash
oversampleqa run oversampleqa-experiment.yaml
```

## 2. Calibrate the Baseline

Run validation with null calibration before comparing samplers:

```bash
oversampleqa validate data.csv \
  --target target \
  --minority-label 1 \
  --oversampler SMOTE \
  --metric hassanat \
  --hidden-ratio 0.1 \
  --calibrate \
  --export json \
  --export markdown \
  --output audit/validation
```

The calibration asks how real held-out minority points behave under the same
nearest-neighbor protocol. A sampler should be judged against that reference,
not against a universal threshold.

## 3. Compare Candidate Samplers

Compare samplers with the same preprocessing, metric, `hidden_ratio`,
validator seed and sampler seeds. Use repeats when the decision matters:

```python
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from oversampleqa import validate_oversampling

samplers = {
    "smote": SMOTE(random_state=0),
    "borderline": BorderlineSMOTE(random_state=0),
}

for name, sampler in samplers.items():
    details = validate_oversampling(
        X,
        y,
        minority_label=1,
        oversampler=sampler,
        metric="hassanat",
        hidden_ratio=0.1,
        random_state=42,
        n_repeats=20,
        return_details=True,
    )
    print(name, details.mean, details.interval)
```

If the intervals overlap heavily, treat the methods as indistinguishable under
this diagnostic.

## 4. Check Fidelity Failure Modes

Run the fidelity report for the leading sampler and any duplicate-producing
baseline:

```bash
oversampleqa fidelity data.csv \
  --target target \
  --minority-label 1 \
  --oversampler SMOTE \
  --metric hassanat \
  --output audit/fidelity.json
```

Read memorisation, boundary violations, precision, recall, density and coverage
together. A sampler can have a low validation error and still be unsuitable if
it copies the training minority or collapses diversity.

## 5. Confirm Utility Separately

OversampleQA is not a substitute for model evaluation. Validate the downstream
model with the metric that matters for the application, such as minority recall,
precision-recall AUC, calibration error or cost-weighted loss. Keep the
oversampler inside the cross-validation pipeline so synthetic samples are
generated only from the training fold.

## 6. Store the Audit Bundle

Keep these artifacts together:

- Dataset version or immutable query.
- OversampleQA config and package version.
- Dependency lock file or environment export.
- Validation JSON and Markdown exports.
- Null calibration output.
- Fidelity output.
- Model-validation summary.
- Any plots used for review.
- A short decision note explaining why the chosen sampler was accepted,
  rejected or deferred.

## Accept, Reject or Defer

Accept a sampler only when the validation result is close to the calibrated
minority reference, repeat variability is acceptable, fidelity does not show
copying or boundary failures, and downstream model validation improves the
metric that matters.

Reject or defer when the evidence conflicts. In particular, do not accept a
sampler because it wins one scalar while the fidelity report or model metrics
show a practical regression.
