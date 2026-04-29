from __future__ import annotations

import argparse

from abstention.config import ensure_project_dirs, load_config, processed_path
from abstention.data import load_dataframe, save_dataframe
from abstention.features import build_feature_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_test.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_project_dirs(config)
    items = load_dataframe(processed_path(config, "items.csv"))
    outputs = load_dataframe(processed_path(config, "model_outputs.csv"))
    features = build_feature_table(items, outputs)
    save_dataframe(features, processed_path(config, "features.csv"))
    print(f"wrote {len(features)} feature rows to {processed_path(config, 'features.csv')}")


if __name__ == "__main__":
    main()

