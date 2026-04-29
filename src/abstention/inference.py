from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .parsing import ParsedResponse, parse_response
from .prompts import build_prompt


OUTPUT_COLUMNS = [
    "item_id",
    "raw_response",
    "parsed_answer",
    "verbal_confidence",
    "model_abstained",
    "self_consistency",
    "choice_margin",
    "choice_entropy",
]


def run_inference(
    items: pd.DataFrame, config: dict[str, Any], cache_path: str | Path | None = None
) -> pd.DataFrame:
    provider = config.get("inference", {}).get("provider", "mock")
    cached = _load_cache(cache_path)
    rows = []
    missing = []
    for _, item in items.iterrows():
        cached_row = cached.get(str(item["item_id"]))
        if cached_row is not None:
            rows.append(cached_row)
        else:
            missing.append(item.to_dict())

    if not missing:
        return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    if provider == "mock":
        new_rows = [_run_mock_item(item) for item in missing]
    elif provider == "transformers":
        new_rows = _run_transformers_items(missing, config)
    else:
        raise ValueError(f"unsupported inference provider: {provider}")

    rows.extend(new_rows)
    outputs = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    _write_cache(cache_path, outputs)
    return outputs


def cache_key(config: dict[str, Any]) -> str:
    inference_cfg = config.get("inference", {})
    payload = {
        "provider": inference_cfg.get("provider", "mock"),
        "model_name": inference_cfg.get("model_name", "mock-deterministic"),
        "prompt_type": inference_cfg.get("prompt_type", "confidence"),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]


def _run_mock_item(item: dict[str, Any]) -> dict[str, Any]:
    raw = mock_response(item)
    parsed = parse_response(raw)
    return _output_row(item["item_id"], raw, parsed, item)


def _run_transformers_items(items: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    _configure_transformers_cache(config)
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    inference_cfg = config.get("inference", {})
    model_name = inference_cfg.get("model_name", "google/flan-t5-small")
    max_new_tokens = int(inference_cfg.get("max_new_tokens", 80))
    batch_size = int(inference_cfg.get("batch_size", 4))
    prompt_type = inference_cfg.get("prompt_type", "confidence")

    # seq2seq keeps the draft colab path modest
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    rows: list[dict[str, Any]] = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        prompts = [build_prompt(item, prompt_type) for item in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )
        decoded = tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)
        confidences = _sequence_confidences(generated.scores, generated.sequences, tokenizer)
        for item, raw_text, confidence in zip(batch, decoded, confidences):
            raw = _coerce_confidence_response(raw_text, confidence)
            parsed = parse_response(raw)
            rows.append(_output_row(item["item_id"], raw, parsed, item))
    return rows


def _output_row(item_id: str, raw: str, parsed: ParsedResponse, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "raw_response": raw,
        "parsed_answer": parsed.answer,
        "verbal_confidence": parsed.confidence,
        "model_abstained": parsed.abstained,
        "self_consistency": mock_self_consistency(item, parsed),
        "choice_margin": mock_choice_margin(item, parsed),
        "choice_entropy": mock_choice_entropy(item),
    }


def _coerce_confidence_response(raw_text: str, confidence: float | None) -> str:
    parsed = parse_response(raw_text)
    if parsed.confidence is not None:
        return raw_text
    return json.dumps(
        {
            "answer": parsed.answer or raw_text.strip(),
            "confidence": 0.5 if confidence is None or math.isnan(confidence) else confidence,
            "abstain": parsed.abstained,
        }
    )


def _sequence_confidences(scores: tuple[Any, ...], sequences: Any, tokenizer: Any) -> list[float | None]:
    if not scores:
        return [None] * len(sequences)
    import torch

    confidences: list[float | None] = []
    score_tensor = torch.stack(scores, dim=1)
    probs = torch.softmax(score_tensor, dim=-1)
    pad_id = tokenizer.pad_token_id
    for batch_idx, seq in enumerate(sequences):
        token_probs = []
        generated = seq[-score_tensor.shape[1] :]
        for step, token_id in enumerate(generated):
            if pad_id is not None and int(token_id) == int(pad_id):
                continue
            token_probs.append(float(probs[batch_idx, step, int(token_id)]))
        confidences.append(float(sum(token_probs) / len(token_probs)) if token_probs else None)
    return confidences


def _load_cache(cache_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if cache_path is None:
        return {}
    path = Path(cache_path)
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return {str(row["item_id"]): row.to_dict() for _, row in frame.iterrows()}


def _write_cache(cache_path: str | Path | None, outputs: pd.DataFrame) -> None:
    if cache_path is None:
        return
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    outputs.drop_duplicates("item_id").sort_values("item_id").to_csv(path, index=False)


def mock_response(item: dict[str, Any]) -> str:
    # deterministic mix of right wrong and idk responses
    item_id = str(item["item_id"])
    number = int(item_id.rsplit("_", 1)[-1])
    if item["benchmark"] == "mmlu":
        answer = item["gold_answer"] if number % 3 != 1 else "D"
        confidence = 0.82 if answer == item["gold_answer"] else 0.42
        return json.dumps({"answer": answer, "confidence": confidence, "abstain": False})
    if not bool(item["gold_is_answerable"]) and number % 3 != 0:
        return json.dumps({"answer": "IDK", "confidence": 0.78, "abstain": True})
    answer = item["gold_answer"] if bool(item["gold_is_answerable"]) else "not in context"
    confidence = 0.76 if bool(item["gold_is_answerable"]) else 0.28
    return json.dumps({"answer": answer, "confidence": confidence, "abstain": False})


def mock_self_consistency(item: dict[str, Any], parsed: ParsedResponse) -> float:
    if parsed.abstained:
        return 0.66
    return 0.85 if parsed.confidence and parsed.confidence >= 0.7 else 0.45


def mock_choice_margin(item: dict[str, Any], parsed: ParsedResponse) -> float:
    if item["benchmark"] != "mmlu":
        return 0.0
    return 0.55 if parsed.confidence and parsed.confidence >= 0.7 else 0.12


def mock_choice_entropy(item: dict[str, Any]) -> float:
    return 1.15 if item["benchmark"] == "mmlu" else 0.0


def _configure_transformers_cache(config: dict[str, Any]) -> None:
    # colab reruns should reuse downloaded weights
    raw_dir = Path(config.get("paths", {}).get("raw_dir", "data/raw"))
    cache_dir = raw_dir / "huggingface"
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    cache_dir.mkdir(parents=True, exist_ok=True)
