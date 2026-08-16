"""
When a Good Error Rate Hides Memorisation
=========================================

The hidden-majority error rate is one number, and it cannot distinguish two
failures that need opposite remedies: generating implausible points, and merely
copying the training data.

``RandomOverSampler`` is the clean demonstration. It duplicates real minority
points, so it is maximally realistic and contributes no information whatsoever.
Its error rate looks comparable to SMOTE's. Only the memorisation ratio
separates them.
"""

import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.datasets import make_classification

from oversampleqa.fidelity import fidelity_report

# %%
# A moderately imbalanced dataset.
X, y = make_classification(
    n_samples=800,
    n_features=6,
    n_informative=4,
    n_redundant=1,
    n_clusters_per_class=1,
    weights=[0.85, 0.15],
    random_state=0,
)

# %%
# Score both samplers across the full fidelity suite.
samplers = {
    "SMOTE": SMOTE(random_state=0),
    "RandomOverSampler": RandomOverSampler(random_state=0),
}
reports = {
    name: fidelity_report(X, y, 1, sampler, metric="hassanat")
    for name, sampler in samplers.items()
}

for name, report in reports.items():
    print(f"\n{name}")
    print(f"  error rate         {report.error_rate:.3f}")
    print(f"  precision          {report.manifold.precision:.3f}")
    print(f"  coverage           {report.manifold.coverage:.3f}")
    print(f"  memorisation ratio {report.memorisation.distance_ratio:.3f}")
    for note in report.interpret():
        print(f"  - {note}")

# %%
# The error rate, precision and coverage are close for both. The memorisation
# ratio is not: it is the median distance from a synthetic point to its nearest
# training point, divided by the median nearest-neighbour distance *within* the
# real minority. A value near zero means the generator sits on top of its
# training data.
metrics = ["error_rate", "precision", "coverage", "memorisation"]
values = {
    name: [
        report.error_rate,
        report.manifold.precision,
        report.manifold.coverage,
        report.memorisation.distance_ratio,
    ]
    for name, report in reports.items()
}

fig, ax = plt.subplots(figsize=(8, 4.5))
positions = np.arange(len(metrics))
width = 0.36
for offset, (name, series) in zip((-width / 2, width / 2), values.items()):
    bars = ax.bar(positions + offset, series, width, label=name)
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

ax.set_xticks(positions)
ax.set_xticklabels(["error rate", "precision", "coverage", "memorisation\nratio"])
ax.set_ylabel("value")
ax.set_ylim(0, 1.25)
ax.legend(loc="upper left")
ax.set_title(
    "Three of these four metrics cannot tell the samplers apart",
    fontsize=11,
)
ax.axvspan(2.5, 3.5, color="0.92", zorder=0)
ax.text(
    3.0,
    1.15,
    "only this one separates them",
    ha="center",
    fontsize=9,
    style="italic",
)
fig.tight_layout()
plt.show()

# %%
# The reading: a single scalar rewards memorisation. A generator that copies its
# training data is perfectly "realistic" and completely useless, and the error
# rate has no way to say so. Report the pair, not the scalar.
