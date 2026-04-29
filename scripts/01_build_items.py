from __future__ import annotations

import argparse

from abstention.config import ensure_project_dirs, load_config, processed_path
from abstention.data import build_items, save_dataframe
from abstention.utils import set_random_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_test.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_random_seed(int(config.get("random_seed", 312)))
    ensure_project_dirs(config)
    items = build_items(config)
    save_dataframe(items, processed_path(config, "items.csv"))
    print(f"wrote {len(items)} items to {processed_path(config, 'items.csv')}")


if __name__ == "__main__":
    main()

