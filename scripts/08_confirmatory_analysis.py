from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from abstention.metrics import compute_rewards, policy_metrics
from abstention.policies import always_abstain, always_answer, decide_from_probability, prompt_only_abstention


BASELINES = {
    "always_answer": always_answer,
    "always_abstain": always_abstain,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["results/runs/draft_500_no_category", "results/runs/draft_1000_strict_json"],
    )
    parser.add_argument("--lambda-cost", type=float, default=0.5)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=312)
    parser.add_argument("--out-dir", default="results/confirmatory")
    args = parser.parse_args()

    selected_rows = []
    bootstrap_rows = []
    for run in args.runs:
        selected, bootstraps = analyze_run(Path(run), args.lambda_cost, args.bootstrap_reps, args.seed)
        selected_rows.append(selected)
        bootstrap_rows.extend(bootstraps)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_frame = pd.DataFrame(selected_rows)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    selected_frame.to_csv(out_dir / "selected_policies.csv", index=False)
    bootstrap_frame.to_csv(out_dir / "paired_bootstrap_tests.csv", index=False)
    _write_summary(out_dir / "confirmatory_summary.md", selected_frame, bootstrap_frame, args)
    print(f"wrote confirmatory results to {out_dir}")


def analyze_run(
    run_dir: Path, lambda_cost: float, bootstrap_reps: int, seed: int
) -> tuple[dict[str, object], list[dict[str, object]]]:
    artifact_dir = run_dir / "artifacts" if (run_dir / "artifacts").exists() else run_dir
    predictions = pd.read_csv(artifact_dir / "predictions.csv")
    features = pd.read_csv(artifact_dir / "features.csv")
    merged = predictions.merge(features[["item_id", "model_abstained"]], on="item_id", how="left")
    merged["model_abstained"] = _coerce_bool_series(merged["model_abstained"])

    model_cols = sorted(col for col in predictions.columns if col.endswith("_prob_correct"))
    if not model_cols:
        raise ValueError(f"no model probability columns found in {artifact_dir / 'predictions.csv'}")

    validation = merged[merged["split"] == "validation"].copy()
    test = merged[merged["split"] == "test"].copy()
    selected_model, validation_auc = _select_model_by_validation_auc(validation, model_cols)
    prob_col = f"{selected_model}_prob_correct"

    y_true = test["answer_is_correct"].astype(bool).to_numpy()
    policy_actions = decide_from_probability(test[prob_col].to_numpy(), lambda_cost)
    selected_metrics = policy_metrics(policy_actions, y_true, lambda_cost)
    policy_rewards = compute_rewards(policy_actions, y_true, lambda_cost)

    selected = {
        "run": run_dir.name,
        "selection_rule": "highest validation AUC",
        "selected_model": selected_model,
        "validation_auc": validation_auc,
        "lambda": lambda_cost,
        "test_utility": selected_metrics["mean_utility"],
        "coverage": selected_metrics["coverage"],
        "risk": selected_metrics["risk"],
        "n_test": int(len(test)),
    }

    baseline_actions = {
        "always_answer": always_answer(len(test)),
        "prompt_only": prompt_only_abstention(test["model_abstained"].to_numpy()),
        "always_abstain": always_abstain(len(test)),
    }
    bootstrap_rows = []
    for baseline, actions in baseline_actions.items():
        baseline_rewards = compute_rewards(actions, y_true, lambda_cost)
        stats = paired_bootstrap_difference(policy_rewards, baseline_rewards, bootstrap_reps, seed)
        bootstrap_rows.append(
            {
                "run": run_dir.name,
                "selected_model": selected_model,
                "lambda": lambda_cost,
                "baseline": baseline,
                "policy_utility": float(policy_rewards.mean()),
                "baseline_utility": float(baseline_rewards.mean()),
                "mean_utility_difference": float((policy_rewards - baseline_rewards).mean()),
                "n_test": int(len(test)),
                **stats,
            }
        )
    return selected, bootstrap_rows


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    def coerce(value: object) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "1", "yes"}

    return series.map(coerce).astype(bool)


def _select_model_by_validation_auc(validation: pd.DataFrame, model_cols: list[str]) -> tuple[str, float]:
    y_true = validation["answer_is_correct"].astype(int)
    if y_true.nunique() < 2:
        raise ValueError("validation split must contain both classes to select by AUC")
    best_model = None
    best_auc = float("-inf")
    for col in model_cols:
        auc = float(roc_auc_score(y_true, validation[col]))
        if auc > best_auc:
            best_auc = auc
            best_model = col.replace("_prob_correct", "")
    if best_model is None:
        raise ValueError("could not select a model")
    return best_model, best_auc


def paired_bootstrap_difference(
    policy_rewards: np.ndarray, baseline_rewards: np.ndarray, reps: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    delta = np.asarray(policy_rewards, dtype=float) - np.asarray(baseline_rewards, dtype=float)
    n = len(delta)
    draws = np.empty(reps, dtype=float)
    for idx in range(reps):
        sample = rng.integers(0, n, size=n)
        draws[idx] = float(delta[sample].mean())
    lower, upper = np.quantile(draws, [0.025, 0.975])
    left = float(np.mean(draws <= 0.0))
    right = float(np.mean(draws >= 0.0))
    return {
        "ci_low": float(lower),
        "ci_high": float(upper),
        "p_bootstrap_two_sided": min(1.0, 2.0 * min(left, right)),
    }


def _write_summary(path: Path, selected: pd.DataFrame, bootstraps: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Confirmatory Results",
        "",
        f"- Lambda: {args.lambda_cost}",
        "- Model selection: highest validation AUC",
        "- Evaluation split: held-out test only",
        f"- Bootstrap reps: {args.bootstrap_reps}",
        "",
        "## Selected policies",
        "",
        "```csv",
        selected.to_csv(index=False).strip(),
        "```",
        "",
        "## Paired bootstrap tests",
        "",
        "```csv",
        bootstraps.to_csv(index=False).strip(),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
