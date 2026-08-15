from pathlib import Path

import pandas as pd

from oversampleqa.cli_enhanced import (
    analyze_dataset,
    explain_ratio,
    generate_recommendations,
    interpret_error_rate,
    load_checkpoint,
    save_checkpoint,
)


def test_analyze_dataset(tmp_path: Path):
    df = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 10.0],
            "target": [0, 0, 1, 1],
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    info = analyze_dataset(csv_path, target="target", minority_label=1)
    assert info["n_samples"] == 4
    assert info["minority"] == 2
    assert info["majority"] == 2


def test_checkpoint_roundtrip(tmp_path: Path):
    payload = {"status": "completed", "results": {"error_rate": 0.12}}
    save_checkpoint(tmp_path, payload)
    loaded = load_checkpoint(tmp_path)
    assert loaded == payload


def test_interpretation_helpers():
    assert "Excellent" in interpret_error_rate(0.05)
    assert "Acceptable" in interpret_error_rate(0.2)
    assert "Risky" in interpret_error_rate(0.4)

    assert "Highly imbalanced" in explain_ratio(0.05)
    assert "Moderately" in explain_ratio(0.2)
    assert "Near-balanced" in explain_ratio(0.6)

    recs = generate_recommendations(0.4, 0.05)
    assert any("advanced oversamplers" in r for r in recs)
