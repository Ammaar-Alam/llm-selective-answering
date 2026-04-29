from __future__ import annotations

import argparse
import os

os.environ["MPLCONFIGDIR"] = "/tmp/sml312-mpl"

import pandas as pd

from abstention.config import ensure_project_dirs, load_config, processed_path, results_path
from abstention.plots import (
    make_calibration_plot,
    make_confidence_correctness_plot,
    make_feature_importance_plot,
    make_risk_coverage_plot,
    make_roc_plot,
    make_utility_plot,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_test.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_project_dirs(config)
    metrics = pd.read_csv(results_path(config, "tables", "policy_metrics.csv"))
    features = pd.read_csv(processed_path(config, "features.csv"))
    predictions = pd.read_csv(processed_path(config, "predictions.csv"))
    importances = pd.read_csv(results_path(config, "tables", "feature_importance.csv"))
    make_utility_plot(metrics, results_path(config, "figures", "utility_vs_lambda.png"))
    make_risk_coverage_plot(metrics, results_path(config, "figures", "risk_coverage.png"))
    make_roc_plot(predictions, results_path(config, "figures", "roc_curve.png"))
    make_calibration_plot(predictions, results_path(config, "figures", "calibration.png"))
    make_confidence_correctness_plot(features, results_path(config, "figures", "confidence_correctness.png"))
    make_feature_importance_plot(importances, results_path(config, "figures", "feature_importance.png"))
    print(f"wrote figures to {results_path(config, 'figures')}")


if __name__ == "__main__":
    main()
