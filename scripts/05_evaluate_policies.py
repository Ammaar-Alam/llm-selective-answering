from __future__ import annotations

import argparse

import pandas as pd

from abstention.config import ensure_project_dirs, load_config, processed_path, results_path
from abstention.data import load_dataframe, save_dataframe
from abstention.metrics import policy_metrics
from abstention.policies import (
    always_abstain,
    always_answer,
    candidate_thresholds,
    decide_from_probability,
    prompt_only_abstention,
    threshold_policy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_test.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_project_dirs(config)
    features = load_dataframe(processed_path(config, "features.csv"))
    predictions = load_dataframe(processed_path(config, "predictions.csv"))
    policy_frame = features.merge(predictions, on=["item_id", "split", "answer_is_correct"])
    metrics, thresholds = evaluate_policies(policy_frame, config)
    save_dataframe(metrics, results_path(config, "tables", "policy_metrics.csv"))
    save_dataframe(thresholds, results_path(config, "tables", "thresholds.csv"))
    print(f"wrote metrics to {results_path(config, 'tables', 'policy_metrics.csv')}")


def evaluate_policies(policy_frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    threshold_rows = []
    lambdas = config.get("evaluation", {}).get("lambdas", [1, 5, 10])
    model_cols = [col for col in policy_frame.columns if col.endswith("_prob_correct")]
    validation = policy_frame[policy_frame["split"] == "validation"].copy()
    threshold_features = ["verbal_confidence", "self_consistency", "choice_margin"]
    for lambda_ in lambdas:
        tuned_thresholds = {
            feature: tune_threshold(validation, feature, float(lambda_)) for feature in threshold_features
        }
        tuned_model_thresholds = {
            col: tune_threshold(validation, col, float(lambda_)) for col in model_cols
        }
        threshold_rows.extend(
            {
                "lambda": float(lambda_),
                "policy": f"threshold_{feature}",
                "feature": feature,
                "threshold": threshold,
            }
            for feature, threshold in tuned_thresholds.items()
        )
        threshold_rows.extend(
            {
                "lambda": float(lambda_),
                "policy": f"tuned_{col.replace('_prob_correct', '')}",
                "feature": col,
                "threshold": threshold,
            }
            for col, threshold in tuned_model_thresholds.items()
        )
        for split, split_frame in policy_frame[policy_frame["split"].isin(["validation", "test"])].groupby("split"):
            y_true = split_frame["answer_is_correct"].astype(bool).to_numpy()
            baseline_actions = {
                "always_answer": always_answer(len(split_frame)),
                "always_abstain": always_abstain(len(split_frame)),
                "prompt_only": prompt_only_abstention(split_frame["model_abstained"].to_numpy()),
            }
            for policy, actions in baseline_actions.items():
                rows.append(
                    {"split": split, "policy": policy, **policy_metrics(actions, y_true, float(lambda_))}
                )
            for feature, threshold in tuned_thresholds.items():
                actions = threshold_policy(split_frame[feature].to_numpy(), threshold)
                rows.append(
                    {
                        "split": split,
                        "policy": f"threshold_{feature}",
                        **policy_metrics(actions, y_true, float(lambda_)),
                    }
                )
            for col in model_cols:
                policy = col.replace("_prob_correct", "")
                actions = decide_from_probability(split_frame[col].to_numpy(), float(lambda_))
                rows.append(
                    {"split": split, "policy": policy, **policy_metrics(actions, y_true, float(lambda_))}
                )
                tuned_actions = threshold_policy(
                    split_frame[col].to_numpy(), tuned_model_thresholds[col]
                )
                rows.append(
                    {
                        "split": split,
                        "policy": f"tuned_{policy}",
                        **policy_metrics(tuned_actions, y_true, float(lambda_)),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(threshold_rows)


def tune_threshold(validation: pd.DataFrame, feature: str, lambda_: float) -> float:
    # validation only so test remains untouched
    if validation.empty:
        return 0.5
    y_true = validation["answer_is_correct"].astype(bool).to_numpy()
    best_threshold = 0.5
    best_utility = float("-inf")
    for threshold in candidate_thresholds(validation[feature].to_numpy()):
        actions = threshold_policy(validation[feature].to_numpy(), float(threshold))
        utility = policy_metrics(actions, y_true, lambda_)["mean_utility"]
        if utility is not None and utility > best_utility:
            best_utility = float(utility)
            best_threshold = float(threshold)
    return best_threshold


if __name__ == "__main__":
    main()
