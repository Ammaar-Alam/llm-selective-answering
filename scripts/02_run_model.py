from __future__ import annotations

import argparse

from abstention.config import ensure_project_dirs, load_config, processed_path
from abstention.data import load_dataframe, save_dataframe
from abstention.inference import cache_key, run_inference
from abstention.utils import set_random_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_test.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_random_seed(int(config.get("random_seed", 312)))
    ensure_project_dirs(config)
    items = load_dataframe(processed_path(config, "items.csv"))
    cache_file = processed_path(config, f"model_outputs_cache_{cache_key(config)}.csv")
    outputs = run_inference(items, config, cache_file)
    save_dataframe(outputs, processed_path(config, "model_outputs.csv"))
    print(f"wrote {len(outputs)} outputs to {processed_path(config, 'model_outputs.csv')}")


if __name__ == "__main__":
    main()
