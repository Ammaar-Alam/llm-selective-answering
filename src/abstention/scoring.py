from __future__ import annotations

import re
import string
from collections import Counter

from .parsing import ParsedResponse, extract_choice
from .utils import normalize_text


def score_mmlu(parsed: ParsedResponse, gold_answer: str, choices: list[str] | None = None) -> bool:
    # abstention is tracked separately from raw correctness
    if parsed.abstained:
        return False
    parsed_choice = extract_choice(parsed.answer)
    if parsed_choice:
        return parsed_choice == gold_answer.upper()

    if choices:
        gold_index = ord(gold_answer.upper()) - ord("A")
        if 0 <= gold_index < len(choices):
            return normalize_text(parsed.answer) == normalize_text(choices[gold_index])
    return normalize_text(parsed.answer) == normalize_text(gold_answer)


def score_squad_answer(parsed: ParsedResponse, gold_answers: list[str], gold_is_answerable: bool) -> bool:
    # target is answer correctness not abstention correctness
    if parsed.abstained or not gold_is_answerable:
        return False
    return max((token_f1(parsed.answer, gold) for gold in gold_answers), default=0.0) >= 0.8


def score_squad_abstention(parsed: ParsedResponse, gold_is_answerable: bool) -> bool:
    # useful diagnostic for prompt only abstention
    return parsed.abstained and not gold_is_answerable


def exact_match(prediction: str, gold: str) -> bool:
    return _normalize_answer(prediction) == _normalize_answer(gold)


def token_f1(prediction: str, gold: str) -> float:
    # simple squad style token overlap
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    same = sum(overlap.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _normalize_answer(text: str) -> str:
    lowered = text.lower()
    no_punc = "".join(ch for ch in lowered if ch not in string.punctuation)
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punc)
    return " ".join(no_articles.split())
