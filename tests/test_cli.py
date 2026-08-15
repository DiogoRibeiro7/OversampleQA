import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification


def test_cli_runs(tmp_path):
    X, y = make_classification(n_samples=600, weights=[0.8, 0.2], random_state=0)
    df = pd.DataFrame(X)
    df["label"] = y
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    env = dict(**os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent / "src")
    out_path = tmp_path / "report.txt"
    plot_path = tmp_path / "plot.png"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "oversampleqa.cli",
            str(csv_path),
            "--target",
            "label",
            "--oversampler",
            "SMOTE",
            "--out",
            str(out_path),
            "--plot",
            str(plot_path),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "Error rate" in result.stdout
    assert out_path.exists() and plot_path.exists()
