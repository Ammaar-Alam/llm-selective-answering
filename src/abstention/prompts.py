from __future__ import annotations

import json
from typing import Any


def build_prompt(item: dict[str, Any], prompt_type: str = "confidence") -> str:
    benchmark = item.get("benchmark", "")
    if benchmark == "mmlu":
        return build_mmlu_prompt(item, prompt_type)
    if benchmark == "squad_v2":
        return build_squad_prompt(item, prompt_type)
    return build_generic_prompt(item, prompt_type)


def build_mmlu_prompt(item: dict[str, Any], prompt_type: str = "confidence") -> str:
    choices = json.loads(item.get("choices", "[]")) if isinstance(item.get("choices"), str) else item["choices"]
    choice_lines = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
    return (
        "Answer the multiple-choice question.\n"
        f"Question: {item['question']}\n"
        f"{choice_lines}\n"
        f"{_response_instruction(prompt_type)}"
    )


def build_squad_prompt(item: dict[str, Any], prompt_type: str = "confidence") -> str:
    return (
        "Answer the question from the context. If the context does not contain the answer, say IDK.\n"
        f"Context: {item.get('context', '')}\n"
        f"Question: {item['question']}\n"
        f"{_response_instruction(prompt_type)}"
    )


def build_generic_prompt(item: dict[str, Any], prompt_type: str = "confidence") -> str:
    return f"Question: {item['question']}\n{_response_instruction(prompt_type)}"


def _response_instruction(prompt_type: str) -> str:
    if prompt_type == "confidence":
        return 'Respond as JSON with keys "answer", "confidence", and "abstain".'
    return "Give only the answer."

