from __future__ import annotations

from pathlib import Path

import pandas as pd


def make_utility_plot(metrics: pd.DataFrame, path: str | Path) -> None:
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for policy, group in metrics.groupby("policy"):
        ax.plot(group["lambda"], group["mean_utility"], marker="o", label=policy)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("wrong-answer cost")
    ax.set_ylabel("mean utility")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def make_risk_coverage_plot(metrics: pd.DataFrame, path: str | Path) -> None:
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for policy, group in metrics.groupby("policy"):
        ax.plot(group["coverage"], group["risk"].fillna(0), marker="o", label=policy)
    ax.set_xlabel("coverage")
    ax.set_ylabel("risk among answered")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def make_roc_plot(predictions: pd.DataFrame, path: str | Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    for col in _prediction_columns(predictions):
        for split, group in predictions.groupby("split"):
            y = group["answer_is_correct"].astype(int)
            if y.nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(y, group[col])
            ax.plot(fpr, tpr, label=f"{col.replace('_prob_correct', '')} {split} auc={auc(fpr, tpr):.2f}")
    ax.plot([0, 1], [0, 1], color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def make_calibration_plot(predictions: pd.DataFrame, path: str | Path, n_bins: int = 10) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    bins = np.linspace(0, 1, n_bins + 1)
    for col in _prediction_columns(predictions):
        test = predictions[predictions["split"] == "test"]
        xs = []
        ys = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (test[col] >= lo) & (test[col] < hi)
            if not mask.any():
                continue
            xs.append(test.loc[mask, col].mean())
            ys.append(test.loc[mask, "answer_is_correct"].mean())
        ax.plot(xs, ys, marker="o", label=col.replace("_prob_correct", ""))
    ax.plot([0, 1], [0, 1], color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("mean predicted correctness")
    ax.set_ylabel("empirical correctness")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def make_confidence_correctness_plot(features: pd.DataFrame, path: str | Path) -> None:
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    test = features[features["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    test["confidence_bin"] = pd.cut(test["verbal_confidence"], bins=10, include_lowest=True)
    grouped = test.groupby("confidence_bin", observed=True).agg(
        verbal_confidence=("verbal_confidence", "mean"),
        answer_is_correct=("answer_is_correct", "mean"),
        n=("item_id", "count"),
    )
    ax.plot(grouped["verbal_confidence"], grouped["answer_is_correct"], marker="o")
    for _, row in grouped.iterrows():
        ax.annotate(str(int(row["n"])), (row["verbal_confidence"], row["answer_is_correct"]), fontsize=8)
    ax.set_xlabel("verbal confidence")
    ax.set_ylabel("answer correctness rate")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def make_feature_importance_plot(importances: pd.DataFrame, path: str | Path, top_n: int = 12) -> None:
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if importances.empty:
        return
    frames = []
    for model, group in importances.groupby("model"):
        top = group.sort_values("abs_value", ascending=False).head(top_n).copy()
        top["label"] = top["feature"].str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False)
        top["model"] = model
        frames.append(top)
    plot_data = pd.concat(frames, ignore_index=True)
    fig, axes = plt.subplots(len(frames), 1, figsize=(8, max(3, 2.8 * len(frames))))
    if len(frames) == 1:
        axes = [axes]
    for ax, (model, group) in zip(axes, plot_data.groupby("model")):
        group = group.sort_values("abs_value")
        ax.barh(group["label"], group["value"])
        ax.set_title(model)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _prediction_columns(predictions: pd.DataFrame) -> list[str]:
    return [col for col in predictions.columns if col.endswith("_prob_correct")]
