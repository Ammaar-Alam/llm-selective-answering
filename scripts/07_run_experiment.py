from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from abstention.config import load_config


STAGES = [
    ("build_items", "scripts/01_build_items.py"),
    ("run_model", "scripts/02_run_model.py"),
    ("build_features", "scripts/03_build_features.py"),
    ("train_models", "scripts/04_train_models.py"),
    ("evaluate_policies", "scripts/05_evaluate_policies.py"),
    ("make_figures", "scripts/06_make_figures.py"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    run_name = args.run_name or config.get("run_name", Path(args.config).stem)
    run_dir = Path("results") / "runs" / run_name
    log_dir = run_dir / "logs"
    artifact_dir = run_dir / "artifacts"
    log_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_name": run_name,
        "config": args.config,
        "started_at": _now(),
        "git_commit": _git_commit(),
        "stages": [],
    }
    shutil.copy2(args.config, run_dir / "config.yaml")

    failed_returncode = 0
    for stage, script in STAGES:
        if args.skip_inference and stage == "run_model":
            continue
        started = time.time()
        command = [sys.executable, script, "--config", args.config]
        env = _stage_env(stage)
        result = _run(command, log_dir / f"{stage}.log", env=env)
        manifest["stages"].append(
            {
                "stage": stage,
                "command": " ".join(command),
                "returncode": result.returncode,
                "runtime_seconds": round(time.time() - started, 3),
            }
        )
        if result.returncode != 0:
            failed_returncode = result.returncode
            break

    _copy_artifacts(artifact_dir)
    manifest["finished_at"] = _now()
    manifest["summary"] = _summarize_artifacts()
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_notes(run_dir / "RUN_NOTES.md", manifest)
    if failed_returncode:
        sys.exit(failed_returncode)


def _stage_env(stage: str) -> dict[str, str] | None:
    if stage != "make_figures":
        return None
    env = os.environ.copy()
    env.update(
        {
            "MPLCONFIGDIR": "/tmp/sml312-mpl",
            "MPLBACKEND": "Agg",
            "XDG_CACHE_HOME": "/tmp/sml312-xdg",
        }
    )
    return env


def _run(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, env=env, check=False)
    log_path.write_text(result.stdout + "\n--- stderr ---\n" + result.stderr, encoding="utf-8")
    return result


def _copy_artifacts(artifact_dir: Path) -> None:
    for source in [
        "data/processed/items.csv",
        "data/processed/model_outputs.csv",
        "data/processed/features.csv",
        "data/processed/predictions.csv",
        "results/metrics/model_summary.json",
        "results/tables/policy_metrics.csv",
        "results/tables/thresholds.csv",
        "results/tables/feature_importance.csv",
    ]:
        path = Path(source)
        if path.exists():
            shutil.copy2(path, artifact_dir / path.name)
    figures = Path("results/figures")
    if figures.exists():
        figure_dir = artifact_dir / "figures"
        figure_dir.mkdir(exist_ok=True)
        for png in figures.glob("*.png"):
            shutil.copy2(png, figure_dir / png.name)


def _summarize_artifacts() -> dict[str, object]:
    summary: dict[str, object] = {}
    for name, path in {
        "items": "data/processed/items.csv",
        "outputs": "data/processed/model_outputs.csv",
        "features": "data/processed/features.csv",
        "predictions": "data/processed/predictions.csv",
        "policy_metrics": "results/tables/policy_metrics.csv",
    }.items():
        csv_path = Path(path)
        if csv_path.exists():
            summary[f"{name}_rows"] = int(len(pd.read_csv(csv_path)))
    metrics_path = Path("results/tables/policy_metrics.csv")
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        test = metrics[metrics["split"] == "test"].copy()
        if not test.empty:
            idx = test["mean_utility"].idxmax()
            summary["best_test_policy"] = test.loc[idx].to_dict()
    return summary


def _write_notes(path: Path, manifest: dict[str, object]) -> None:
    lines = [
        f"# {manifest['run_name']}",
        "",
        f"- Started: {manifest['started_at']}",
        f"- Finished: {manifest['finished_at']}",
        f"- Git commit: {manifest['git_commit']}",
        f"- Config: {manifest['config']}",
        "",
        "## Stage runtimes",
        "",
    ]
    for stage in manifest["stages"]:
        lines.append(f"- {stage['stage']}: {stage['runtime_seconds']}s, return {stage['returncode']}")
    lines.extend(["", "## Summary", "", "```json", json.dumps(manifest["summary"], indent=2), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], text=True, capture_output=True, check=False)
    return result.stdout.strip()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
