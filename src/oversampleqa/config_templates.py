"""Predefined configuration templates for the enhanced CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

CONFIG_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "research": {
        "description": "Comprehensive validation for research purposes",
        "params": {
            "hidden_ratios": [0.1, 0.25, 0.5],
            "n_runs": 50,
            "metrics": ["hassanat", "euclidean", "mahalanobis"],
            "include_plots": True,
            "statistical_tests": True,
            "export": ["json", "markdown"],
        },
    },
    "production": {
        "description": "Fast validation for production monitoring",
        "params": {
            "hidden_ratios": [0.1],
            "n_runs": 5,
            "metrics": ["euclidean"],
            "include_plots": False,
            "cache_results": True,
            "resume": True,
        },
    },
    "education": {
        "description": "Educational demonstrations with detailed explanations",
        "params": {
            "hidden_ratios": [0.25],
            "n_runs": 10,
            "metrics": ["hassanat", "euclidean"],
            "include_plots": True,
            "verbose_output": True,
            "include_tutorials": True,
        },
    },
}


def generate_config_file(template_name: str, output_path: str) -> Path:
    """Generate a YAML configuration file from a named template.

    Parameters
    ----------
    template_name:
        Name of the template (``research``, ``production``, ``education``).
    output_path:
        Destination path for the generated YAML file.

    Returns
    -------
    pathlib.Path
        Path to the generated configuration file.
    """

    if template_name not in CONFIG_TEMPLATES:
        available = ", ".join(sorted(CONFIG_TEMPLATES))
        raise ValueError(f"Unknown template '{template_name}'. Choose from: {available}")

    payload: Dict[str, Any] = {
        "template": template_name,
        "description": CONFIG_TEMPLATES[template_name]["description"],
        "profiles": {
            template_name: CONFIG_TEMPLATES[template_name]["params"],
        },
    }

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)
    return destination
