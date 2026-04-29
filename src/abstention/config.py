from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(config_path)
    return config


def ensure_project_dirs(config: dict[str, Any]) -> None:
    # keep generated outputs in the same few folders
    paths = config.get("paths", {})
    for key in ("processed_dir", "results_dir"):
        if key in paths:
            Path(paths[key]).mkdir(parents=True, exist_ok=True)
    results_dir = Path(paths.get("results_dir", "results"))
    for child in ("figures", "tables", "metrics"):
        (results_dir / child).mkdir(parents=True, exist_ok=True)


def processed_path(config: dict[str, Any], filename: str) -> Path:
    # one place for intermediate csv paths
    return Path(config.get("paths", {}).get("processed_dir", "data/processed")) / filename


def results_path(config: dict[str, Any], *parts: str) -> Path:
    # keeps scripts from hardcoding result locations
    return Path(config.get("paths", {}).get("results_dir", "results")).joinpath(*parts)
