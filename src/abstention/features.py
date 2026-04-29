from __future__ import annotations

import json
import re

import pandas as pd

from .parsing import ParsedResponse
from .scoring import score_mmlu, score_squad_abstention, score_squad_answer


def build_feature_table(items: pd.DataFrame, outputs: pd.DataFrame) -> pd.DataFrame:
    # one row is one answer or abstain decision
    merged = items.merge(outputs, on="item_id", how="left")
    rows = []
    for _, row in merged.iterrows():
        choices = _loads(row.get("choices", "[]"))
        parsed = ParsedResponse(
            answer=str(row.get("parsed_answer", "")),
            confidence=_none_if_nan(row.get("verbal_confidence")),
            abstained=bool(row.get("model_abstained", False)),
        )
        answer_is_correct = score_item(row, parsed, choices)
        abstention_is_correct = score_abstention(row, parsed)
        rows.append(
            {
                "item_id": row["item_id"],
                "split": row["split"],
                "benchmark": row["benchmark"],
                "subject_or_category": row.get("subject_or_category", ""),
                "answer_is_correct": answer_is_correct,
                "abstention_is_correct": abstention_is_correct,
                "model_abstained": parsed.abstained,
                "verbal_confidence": parsed.confidence if parsed.confidence is not None else 0.5,
                "self_consistency": float(row.get("self_consistency", 0.0)),
                "choice_margin": float(row.get("choice_margin", 0.0)),
                "choice_entropy": float(row.get("choice_entropy", 0.0)),
                "question_length": len(str(row.get("question", "")).split()),
                "context_length": len(str(row.get("context", "")).split()),
                "contains_number": bool(re.search(r"\d", str(row.get("question", "")))),
                "choice_count": len(choices),
                "mean_choice_length": _mean_choice_length(choices),
            }
        )
    return pd.DataFrame(rows)


def score_item(row: pd.Series, parsed: ParsedResponse, choices: list[str]) -> bool:
    # dataset specific scoring stays behind one helper
    if row["benchmark"] == "mmlu":
        return score_mmlu(parsed, str(row["gold_answer"]), choices)
    if row["benchmark"] == "squad_v2":
        gold_answers = _loads(row.get("gold_answers", "[]")) or [str(row.get("gold_answer", ""))]
        return score_squad_answer(parsed, gold_answers, bool(row["gold_is_answerable"]))
    return False


def score_abstention(row: pd.Series, parsed: ParsedResponse) -> bool:
    # this stays separate from answer correctness
    if row["benchmark"] == "squad_v2":
        return score_squad_abstention(parsed, bool(row["gold_is_answerable"]))
    return False


def _loads(value: object) -> list[str]:
    if isinstance(value, list):
        return value
    if value is None or pd.isna(value):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _none_if_nan(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _mean_choice_length(choices: list[str]) -> float:
    if not choices:
        return 0.0
    return float(sum(len(choice.split()) for choice in choices) / len(choices))
