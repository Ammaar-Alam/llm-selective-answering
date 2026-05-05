from abstention.prompts import build_prompt


def test_strict_json_mmlu_prompt_requests_json() -> None:
    item = {
        "benchmark": "mmlu",
        "question": "Which option is correct?",
        "choices": ["one", "two", "three", "four"],
    }
    prompt = build_prompt(item, "strict_json")
    assert "Return JSON only" in prompt
    assert "A, B, C, or D" in prompt


def test_strict_json_squad_prompt_allows_idk() -> None:
    item = {"benchmark": "squad_v2", "question": "When?", "context": "No date is given."}
    prompt = build_prompt(item, "strict_json")
    assert "answer IDK" in prompt
    assert "Return JSON only" in prompt
