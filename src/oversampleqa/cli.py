import argparse
import logging
import pandas as pd
from importlib import import_module

from .validator import validate_oversampling, extract_synthetic_samples
from .plotting import plot_sample_distribution

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for validation.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Validate oversampling using the hidden majority approach"
    )
    parser.add_argument("csv", help="Path to CSV dataset containing features and target column")
    parser.add_argument(
        "--target",
        default="target",
        help="Name of the target/label column in the CSV file",
    )
    parser.add_argument(
        "--minority-label",
        type=int,
        default=1,
        help="Label value representing the minority class",
    )
    parser.add_argument(
        "--oversampler",
        default="SMOTE",
        help="Name of imbalanced-learn oversampler class to use",
    )
    parser.add_argument(
        "--hidden-ratio",
        type=float,
        default=0.1,
        help="Fraction of majority samples to hide during validation",
    )
    parser.add_argument(
        "--distance",
        default="hassanat",
        help="Distance metric to use (hassanat, euclidean, etc.)",
    )
    parser.add_argument(
        "--out",
        help="Optional path to save a text report with the error rate",
    )
    parser.add_argument(
        "--plot",
        help="Optional path to save a PCA plot of the resampled data",
    )
    return parser.parse_args()


def main() -> None:
    """Run the CLI validation workflow.

    This entry point loads the dataset, configures the oversampler, runs the
    validation, and optionally writes a report or plot.
    """
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    try:
        df = pd.read_csv(args.csv)
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.exception("Failed to read CSV: %s", exc)
        raise
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in CSV")
    X = df.drop(columns=[args.target]).values
    y = df[args.target].values

    mod = import_module("imblearn.over_sampling")
    oversampler_cls = getattr(mod, args.oversampler)
    oversampler = oversampler_cls()

    try:
        error = validate_oversampling(
            X,
            y,
            minority_label=args.minority_label,
            oversampler=oversampler,
            hidden_ratio=args.hidden_ratio,
            metric=args.distance,
        )
    except Exception:
        logger.exception("Validation failed")
        raise
    print(f"Error rate: {error:.3f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(f"Error rate: {error:.3f}\n")

    if args.plot:
        # Refit oversampler on the full dataset for visualization
        vis_os = oversampler_cls()
        X_res, y_res = vis_os.fit_resample(X, y)
        mask = y == args.minority_label
        minority = X[mask]
        majority = X[~mask]
        synthetic = extract_synthetic_samples(X, X_res, y_res, args.minority_label)
        plot_sample_distribution(majority, minority, synthetic, save_path=args.plot)


if __name__ == "__main__":
    main()
