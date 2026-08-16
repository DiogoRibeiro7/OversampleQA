"""Tests for the fidelity radar chart.

The chart is only readable if every axis points the same way, so the orientation
and clipping rules are the substance here -- not whether a PNG appeared.
"""

from __future__ import annotations

import numpy as np
import pytest

from oversampleqa.plotting import (
    _FIDELITY_AXES,
    _fidelity_axis_values,
    plot_fidelity_radar,
)

AXES = {spec[0]: spec for spec in _FIDELITY_AXES}


def _payload(**overrides):
    base = {
        "precision": 0.9,
        "recall": 0.8,
        "coverage": 0.7,
        "density": 0.6,
        "memorisation_distance_ratio": 0.5,
        "boundary_violation_strict": 0.1,
    }
    base.update(overrides)
    return base


def test_every_axis_points_outward_for_better():
    """A radar whose spokes disagree on direction has a meaningless area."""
    good = _payload(
        precision=1.0,
        recall=1.0,
        coverage=1.0,
        density=1.0,
        memorisation_distance_ratio=1.0,
        boundary_violation_strict=0.0,  # no violations == best
    )
    values, _ = _fidelity_axis_values(good, _FIDELITY_AXES)
    assert all(v == pytest.approx(1.0) for v in values)


def test_boundary_safety_is_the_complement_of_the_violation_rate():
    """Raw rate is better when small, so it must be inverted to be plotted."""
    values, _ = _fidelity_axis_values(
        _payload(boundary_violation_strict=0.25), [AXES["boundary safety"]]
    )
    assert values[0] == pytest.approx(0.75)


def test_a_worse_sampler_gets_a_shorter_spoke():
    safe, _ = _fidelity_axis_values(
        _payload(boundary_violation_strict=0.05), [AXES["boundary safety"]]
    )
    unsafe, _ = _fidelity_axis_values(
        _payload(boundary_violation_strict=0.60), [AXES["boundary safety"]]
    )
    assert safe[0] > unsafe[0]


def test_density_above_one_is_clipped_and_reported():
    """Density is not bounded by 1; 2.4 and 1.0 would otherwise render alike."""
    values, notes = _fidelity_axis_values(
        _payload(density=2.4), [AXES["density"]], name="SMOTE"
    )
    assert values[0] == pytest.approx(1.0)
    assert notes and "SMOTE" in notes[0] and "2.40" in notes[0]


def test_bounded_axes_are_never_reported_as_clipped():
    _, notes = _fidelity_axis_values(_payload(precision=1.0), [AXES["precision"]])
    assert notes == []


def test_nan_is_preserved_not_zeroed():
    """A zero would be indistinguishable from a real measurement of failure."""
    values, notes = _fidelity_axis_values(
        {"precision": float("nan")}, [AXES["precision"]]
    )
    assert np.isnan(values[0])
    assert notes == []


def test_missing_key_becomes_nan():
    values, _ = _fidelity_axis_values({}, [AXES["coverage"]])
    assert np.isnan(values[0])


def test_inverting_a_nan_does_not_produce_a_number():
    """1.0 - nan is nan, but the guard must not turn it into 1.0."""
    values, _ = _fidelity_axis_values(
        {"boundary_violation_strict": float("nan")}, [AXES["boundary safety"]]
    )
    assert np.isnan(values[0])


def test_error_rate_is_not_an_axis():
    """It answers a different question and is not commensurable with these."""
    assert "error_rate" not in {spec[1] for spec in _FIDELITY_AXES}


def test_writes_a_file(tmp_path):
    out = tmp_path / "radar.png"
    plot_fidelity_radar({"SMOTE": _payload(), "ROS": _payload()}, save_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_accepts_a_fidelity_report_object(tmp_path):
    class Fake:
        def to_dict(self):
            return _payload()

    out = tmp_path / "radar.png"
    plot_fidelity_radar({"SMOTE": Fake()}, save_path=str(out))
    assert out.exists()


def test_empty_reports_raises():
    with pytest.raises(ValueError, match="empty"):
        plot_fidelity_radar({})


def test_unknown_metric_raises():
    with pytest.raises(ValueError, match="unknown metric"):
        plot_fidelity_radar({"a": _payload()}, metrics=["precision", "nope", "recall"])


def test_fewer_than_three_axes_raises():
    """Two spokes is a line, one is a point."""
    with pytest.raises(ValueError, match="at least 3"):
        plot_fidelity_radar({"a": _payload()}, metrics=["precision", "recall"])


def test_metrics_subset_is_respected(tmp_path):
    out = tmp_path / "radar.png"
    plot_fidelity_radar(
        {"a": _payload()},
        save_path=str(out),
        metrics=["precision", "recall", "coverage"],
    )
    assert out.exists()


def test_no_file_written_without_a_path():
    plot_fidelity_radar({"a": _payload()})  # must not raise
