from __future__ import annotations

import numpy as np


def utility_threshold(lambda_: float) -> float:
    # answer only when expected answer utility is positive
    if lambda_ < 0:
        raise ValueError("lambda must be nonnegative")
    return lambda_ / (1.0 + lambda_)


def decide_from_probability(prob_correct: np.ndarray, lambda_: float) -> np.ndarray:
    threshold = utility_threshold(lambda_)
    return np.where(np.asarray(prob_correct) > threshold, "answer", "abstain")


def always_answer(n_items: int) -> np.ndarray:
    return np.repeat("answer", n_items)


def always_abstain(n_items: int) -> np.ndarray:
    return np.repeat("abstain", n_items)


def prompt_only_abstention(model_abstained: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(model_abstained), "abstain", "answer")

def threshold_policy(values: np.ndarray, threshold: float, higher_is_better: bool = True) -> np.ndarray:
    # single signal baseline for confidence style features
    scores = np.asarray(values, dtype=float)
    answer = scores >= threshold if higher_is_better else scores <= threshold
    return np.where(answer, "answer", "abstain")


def candidate_thresholds(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return np.array([0.5])
    candidates = sorted(set(float(value) for value in finite))
    return np.array([min(candidates) - 1e-9, *candidates, max(candidates) + 1e-9])
