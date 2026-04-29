from abstention.parsing import ParsedResponse
from abstention.scoring import exact_match, score_mmlu, score_squad_abstention, score_squad_answer, token_f1


def test_score_mmlu_choice_letter() -> None:
    parsed = ParsedResponse(answer="B", confidence=0.9, abstained=False)
    assert score_mmlu(parsed, "B", ["a", "b", "c", "d"])


def test_score_mmlu_choice_text() -> None:
    parsed = ParsedResponse(answer="Paris", confidence=0.7, abstained=False)
    assert score_mmlu(parsed, "C", ["Rome", "London", "Paris", "Berlin"])


def test_score_squad_unanswerable_abstention() -> None:
    parsed = ParsedResponse(answer="IDK", confidence=0.8, abstained=True)
    assert not score_squad_answer(parsed, [], False)
    assert score_squad_abstention(parsed, False)


def test_answer_normalization() -> None:
    assert exact_match("The quick, fox!", "quick fox")
    assert token_f1("alpha beta", "alpha gamma") > 0
