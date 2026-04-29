from abstention.parsing import extract_choice, is_abstention, parse_response


def test_parse_json_response() -> None:
    parsed = parse_response('{"answer": "B", "confidence": 0.8, "abstain": false}')
    assert parsed.answer == "B"
    assert parsed.confidence == 0.8
    assert parsed.abstained is False


def test_parse_abstention_text() -> None:
    parsed = parse_response("IDK")
    assert parsed.abstained is True


def test_parse_empty_response() -> None:
    parsed = parse_response("")
    assert parsed.answer == ""
    assert parsed.confidence is None
    assert parsed.abstained is False


def test_extract_choice() -> None:
    assert extract_choice("The answer is C.") == "C"
    assert extract_choice("choice z") is None


def test_abstention_patterns() -> None:
    assert is_abstention("Cannot answer from the context")
