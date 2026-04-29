import numpy as np

from abstention.metrics import compute_rewards, expected_calibration_error, policy_metrics


def test_compute_rewards() -> None:
    rewards = compute_rewards(np.array(["answer", "answer", "abstain"]), np.array([1, 0, 0]), 5)
    assert rewards.tolist() == [1.0, -5.0, 0.0]


def test_policy_metrics() -> None:
    metrics = policy_metrics(np.array(["answer", "abstain"]), np.array([0, 1]), 2)
    assert metrics["mean_utility"] == -1.0
    assert metrics["coverage"] == 0.5
    assert metrics["risk"] == 1.0


def test_expected_calibration_error_perfect() -> None:
    ece = expected_calibration_error(np.array([0.1, 0.9]), np.array([0, 1]), n_bins=2)
    assert ece < 0.11

