from __future__ import annotations

import json
import re
from dataclasses import dataclass


ABSTENTION_PATTERNS = (
    "idk",
    "i don't know",
    "i do not know",
    "cannot answer",
    "can't answer",
    "not enough information",
    "unknown",
)


@dataclass(frozen=True)
class ParsedResponse:
    answer: str
    confidence: float | None
    abstained: bool


def is_abstention(text: str) -> bool:
    lowered = text.lower().strip()
    return any(pattern in lowered for pattern in ABSTENTION_PATTERNS)


def clamp_confidence(value: object) -> float | None:
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence > 1:
        confidence = confidence / 100.0
    return max(0.0, min(1.0, confidence))


def parse_response(text: str) -> ParsedResponse:
    raw = text.strip()
    if not raw:
        return ParsedResponse(answer="", confidence=None, abstained=False)
    # prefer structured output when the prompt gets it
    parsed_json = _parse_json(raw)
    if parsed_json is not None:
        answer = str(parsed_json.get("answer", "")).strip()
        confidence = clamp_confidence(parsed_json.get("confidence"))
        abstained = bool(parsed_json.get("abstain", False)) or is_abstention(answer)
        return ParsedResponse(answer=answer, confidence=confidence, abstained=abstained)

    answer = _parse_answer(raw)
    confidence = _parse_confidence(raw)
    return ParsedResponse(answer=answer, confidence=confidence, abstained=is_abstention(raw))


def extract_choice(answer: str) -> str | None:
    # handles answers like b or the answer is b
    match = re.search(r"\b([A-D])\b", answer.upper())
    return match.group(1) if match else None


def _parse_json(text: str) -> dict[str, object] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _parse_answer(text: str) -> str:
    match = re.search(r"answer\s*[:\-]\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).splitlines()[0].strip().strip('"')
    return text.splitlines()[0].strip()


def _parse_confidence(text: str) -> float | None:
    match = re.search(r"confidence\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?%?)", text, re.IGNORECASE)
    if not match:
        return None
    value = match.group(1).rstrip("%")
    confidence = clamp_confidence(value)
    if "%" in match.group(1) and confidence is not None:
        return confidence / 100.0 if confidence > 1 else confidence
    return confidence
