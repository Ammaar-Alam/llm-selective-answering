from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .metrics import expected_calibration_error


NUMERIC_FEATURES = [
    "verbal_confidence",
    "self_consistency",
    "choice_margin",
    "choice_entropy",
    "question_length",
    "context_length",
    "choice_count",
    "mean_choice_length",
]
CATEGORICAL_FEATURES = ["benchmark", "subject_or_category", "contains_number", "model_abstained"]


def train_predictors(
    features: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    # fit only on train and score held out rows
    train = features[features["split"] == "train"].copy()
    testable = features[features["split"].isin(["validation", "test"])].copy()
    if train.empty or testable.empty:
        raise ValueError("features must include train and validation/test rows")

    enabled = config.get("models", {})
    models: dict[str, Pipeline] = {}
    if enabled.get("logistic_regression", True):
        models["logistic_regression"] = _make_pipeline(
            LogisticRegression(max_iter=1000, random_state=int(config.get("random_seed", 312)))
        )
    if enabled.get("random_forest", True):
        models["random_forest"] = _make_pipeline(
            RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=2,
                random_state=int(config.get("random_seed", 312)),
            )
        )
    if enabled.get("gradient_boosting", False):
        models["gradient_boosting"] = _make_pipeline(
            GradientBoostingClassifier(random_state=int(config.get("random_seed", 312)))
        )

    predictions = testable[["item_id", "split", "answer_is_correct"]].copy()
    summary: dict[str, Any] = {}
    importance_frames = []
    for name, model in models.items():
        model.fit(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train["answer_is_correct"])
        proba = model.predict_proba(testable[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
        predictions[f"{name}_prob_correct"] = proba
        summary[name] = {}
        for split, group in predictions.groupby("split"):
            split_proba = group[f"{name}_prob_correct"]
            summary[name][split] = _summarize_predictions(group["answer_is_correct"], split_proba)
        importance_frames.append(_feature_importance_frame(name, model))
    importances = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    return predictions, summary, importances


def save_model_summary(summary: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def save_feature_importance(importances: pd.DataFrame, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    importances.to_csv(out, index=False)


def _make_pipeline(classifier: Any) -> Pipeline:
    # sklearn handles train only scaling inside the pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def _summarize_predictions(y_true: pd.Series, prob_correct: pd.Series) -> dict[str, float | None]:
    hard = prob_correct >= 0.5
    auc = None
    if len(set(y_true.astype(int))) > 1:
        auc = float(roc_auc_score(y_true, prob_correct))
    return {
        "accuracy_at_0_5": float(accuracy_score(y_true, hard)),
        "auc": auc,
        "ece": expected_calibration_error(prob_correct.to_numpy(), y_true.to_numpy()),
    }


def _feature_importance_frame(model_name: str, pipeline: Pipeline) -> pd.DataFrame:
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
        kind = "importance"
    elif hasattr(classifier, "coef_"):
        values = classifier.coef_[0]
        kind = "coefficient"
    else:
        values = [0.0] * len(names)
        kind = "unavailable"
    return pd.DataFrame(
        {
            "model": model_name,
            "feature": names,
            "value": values,
            "abs_value": [abs(float(value)) for value in values],
            "kind": kind,
        }
    ).sort_values(["model", "abs_value"], ascending=[True, False])
