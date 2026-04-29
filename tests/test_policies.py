import numpy as np
import pytest

from abstention.policies import decide_from_probability, prompt_only_abstention, utility_threshold


def test_utility_threshold() -> None:
    assert utility_threshold(1) == 0.5
    assert utility_threshold(9) == 0.9


def test_negative_lambda_rejected() -> None:
    with pytest.raises(ValueError):
        utility_threshold(-1)


def test_decide_from_probability() -> None:
    actions = decide_from_probability(np.array([0.4, 0.8]), lambda_=1)
    assert actions.tolist() == ["abstain", "answer"]


def test_prompt_only_abstention() -> None:
    actions = prompt_only_abstention(np.array([True, False]))
    assert actions.tolist() == ["abstain", "answer"]

