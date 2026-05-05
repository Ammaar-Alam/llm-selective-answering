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
    if prompt_type == "strict_json":
        return (
            "Choose the best answer. The answer must be one of A, B, C, or D.\n"
            f"Question: {item['question']}\n"
            f"{choice_lines}\n"
            'Return JSON only: {"answer": "A", "confidence": 0.0, "abstain": false}.'
        )
    if prompt_type == "abstain_json":
        return (
            "Choose the best answer only if enough evidence is available. If not, answer IDK.\n"
            f"Question: {item['question']}\n"
            f"{choice_lines}\n"
            'Return JSON only with keys "answer", "confidence", and "abstain".'
        )
    return (
        "Answer the multiple-choice question.\n"
        f"Question: {item['question']}\n"
        f"{choice_lines}\n"
        f"{_response_instruction(prompt_type)}"
    )


def build_squad_prompt(item: dict[str, Any], prompt_type: str = "confidence") -> str:
    if prompt_type == "strict_json":
        return (
            "Answer using the shortest exact phrase from the context. If the answer is not in the context, answer IDK.\n"
            f"Context: {item.get('context', '')}\n"
            f"Question: {item['question']}\n"
            'Return JSON only: {"answer": "IDK", "confidence": 0.0, "abstain": true}.'
        )
    if prompt_type == "abstain_json":
        return (
            "Read the context and decide whether the question can be answered. Use IDK when the context is insufficient.\n"
            f"Context: {item.get('context', '')}\n"
            f"Question: {item['question']}\n"
            'Return JSON only with keys "answer", "confidence", and "abstain".'
        )
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
