from __future__ import annotations

import argparse

from abstention.config import ensure_project_dirs, load_config, processed_path, results_path
from abstention.data import load_dataframe, save_dataframe
from abstention.models import save_feature_importance, save_model_summary, train_predictors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_test.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_project_dirs(config)
    features = load_dataframe(processed_path(config, "features.csv"))
    predictions, summary, importances = train_predictors(features, config)
    save_dataframe(predictions, processed_path(config, "predictions.csv"))
    save_model_summary(summary, results_path(config, "metrics", "model_summary.json"))
    save_feature_importance(importances, results_path(config, "tables", "feature_importance.csv"))
    print(f"wrote predictions to {processed_path(config, 'predictions.csv')}")


if __name__ == "__main__":
    main()
