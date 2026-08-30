"""Regression tests for the diagnostic decision guide."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_decision_guide_covers_the_diagnostic_stack():
    guide = (ROOT / "docs/decision_guide.md").read_text(encoding="utf-8")

    for term in (
        "validate_oversampling",
        "null_error_rate",
        "nn_two_sample_test",
        "mst_two_sample_test",
        "cross_match_test",
        "fidelity_report",
        "downstream_utility",
        "StatisticalBenchmark",
        "fold_results()",
        "*.metadata.json",
    ):
        assert term in guide


def test_decision_guide_names_operational_outcomes():
    guide = (ROOT / "docs/decision_guide.md").read_text(encoding="utf-8")
    lower = guide.lower()

    assert "Accept a sampler" in guide
    assert "Reject a sampler" in guide
    assert "Defer the decision" in guide
    assert "high p-value treated as proof of equality" in lower
    assert "resampling happened before splitting" in guide


def test_roadmap_marks_documentation_debt_done():
    roadmap = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    compact = " ".join(roadmap.split())

    assert "the examples refresh and the decision guide are done" in compact
    assert "validation error rate, calibrated error rate, two-sample tests" in compact
