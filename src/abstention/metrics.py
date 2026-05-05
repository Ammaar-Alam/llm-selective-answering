from __future__ import annotations

import numpy as np


def compute_rewards(actions: np.ndarray, y_true: np.ndarray, lambda_: float) -> np.ndarray:
    # abstentions get zero by construction
    actions = np.asarray(actions)
    y_true = np.asarray(y_true).astype(bool)
    rewards = np.zeros(len(actions), dtype=float)
    answered = actions == "answer"
    rewards[answered & y_true] = 1.0
    rewards[answered & ~y_true] = -float(lambda_)
    return rewards


def policy_metrics(actions: np.ndarray, y_true: np.ndarray, lambda_: float) -> dict[str, float | None]:
    actions = np.asarray(actions)
    y_true = np.asarray(y_true).astype(bool)
    answered = actions == "answer"
    rewards = compute_rewards(actions, y_true, lambda_)
    coverage = float(answered.mean()) if len(actions) else 0.0
    risk = None
    if answered.any():
        risk = float((answered & ~y_true).sum() / answered.sum())
    return {
        "lambda": float(lambda_),
        "mean_utility": float(rewards.mean()) if len(rewards) else 0.0,
        "coverage": coverage,
        "risk": risk,
    }


def expected_calibration_error(prob_correct: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    # small transparent ECE implementation
    p = np.asarray(prob_correct, dtype=float)
    y = np.asarray(y_true, dtype=float)
    if len(p) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)
