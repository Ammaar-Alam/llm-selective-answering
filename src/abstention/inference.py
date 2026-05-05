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
    "choice_top_prob",
    "choice_margin",
    "choice_entropy",
    "token_confidence_min",
    "token_confidence_std",
    "raw_response_length",
    "parsed_answer_length",
    "response_empty",
    "confidence_missing",
    "parsed_choice_valid",
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
        "max_new_tokens": inference_cfg.get("max_new_tokens", 80),
        "batch_size": inference_cfg.get("batch_size", 4),
        "feature_version": 2,
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
        token_stats = _sequence_confidences(generated.scores, generated.sequences, tokenizer)
        choice_stats = _choice_stats_for_batch(model, tokenizer, encoded, batch)
        for item, raw_text, stats, choice_stat in zip(batch, decoded, token_stats, choice_stats):
            raw = _coerce_confidence_response(raw_text, stats["mean"])
            parsed = parse_response(raw)
            rows.append(_output_row(item["item_id"], raw, parsed, item, stats, choice_stat))
    return rows


def _output_row(
    item_id: str,
    raw: str,
    parsed: ParsedResponse,
    item: dict[str, Any],
    token_stats: dict[str, float | None] | None = None,
    choice_stats: dict[str, float] | None = None,
) -> dict[str, Any]:
    token_stats = token_stats or {"mean": parsed.confidence, "min": parsed.confidence, "std": 0.0}
    choice_stats = choice_stats or {
        "choice_top_prob": mock_choice_top_prob(item, parsed),
        "choice_margin": mock_choice_margin(item, parsed),
        "choice_entropy": mock_choice_entropy(item),
    }
    parsed_choice = _extract_choice(parsed.answer)
    raw_stripped = raw.strip()
    return {
        "item_id": item_id,
        "raw_response": raw,
        "parsed_answer": parsed.answer,
        "verbal_confidence": parsed.confidence,
        "model_abstained": parsed.abstained,
        "self_consistency": mock_self_consistency(item, parsed),
        "choice_top_prob": choice_stats["choice_top_prob"],
        "choice_margin": choice_stats["choice_margin"],
        "choice_entropy": choice_stats["choice_entropy"],
        "token_confidence_min": token_stats["min"],
        "token_confidence_std": token_stats["std"],
        "raw_response_length": len(raw_stripped),
        "parsed_answer_length": len(parsed.answer.strip()),
        "response_empty": raw_stripped == "",
        "confidence_missing": parsed.confidence is None,
        "parsed_choice_valid": item.get("benchmark") != "mmlu" or parsed_choice in {"A", "B", "C", "D"},
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


def _sequence_confidences(scores: tuple[Any, ...], sequences: Any, tokenizer: Any) -> list[dict[str, float | None]]:
    if not scores:
        return [{"mean": None, "min": None, "std": None} for _ in range(len(sequences))]
    import torch

    confidences: list[dict[str, float | None]] = []
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
        if token_probs:
            mean = float(sum(token_probs) / len(token_probs))
            variance = sum((value - mean) ** 2 for value in token_probs) / len(token_probs)
            confidences.append({"mean": mean, "min": min(token_probs), "std": math.sqrt(variance)})
        else:
            confidences.append({"mean": None, "min": None, "std": None})
    return confidences


def _choice_stats_for_batch(model: Any, tokenizer: Any, encoded: Any, items: list[dict[str, Any]]) -> list[dict[str, float]]:
    import torch

    labels = ["A", "B", "C", "D"]
    token_ids = [_first_token_id(tokenizer, label) for label in labels]
    decoder_start = model.config.decoder_start_token_id
    if decoder_start is None:
        decoder_start = tokenizer.pad_token_id
    decoder_input_ids = torch.full(
        (len(items), 1), int(decoder_start), dtype=torch.long, device=model.device
    )
    with torch.no_grad():
        logits = model(**encoded, decoder_input_ids=decoder_input_ids).logits[:, -1, token_ids]
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    stats = []
    for item, row in zip(items, probs):
        if item.get("benchmark") != "mmlu":
            stats.append({"choice_top_prob": 0.0, "choice_margin": 0.0, "choice_entropy": 0.0})
            continue
        sorted_probs = sorted((float(value) for value in row), reverse=True)
        entropy = -sum(float(value) * math.log(max(float(value), 1e-12)) for value in row)
        stats.append(
            {
                "choice_top_prob": sorted_probs[0],
                "choice_margin": sorted_probs[0] - sorted_probs[1],
                "choice_entropy": entropy,
            }
        )
    return stats


def _first_token_id(tokenizer: Any, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return int(ids[0])


def _load_cache(cache_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if cache_path is None:
        return {}
    path = Path(cache_path)
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if any(column not in frame.columns for column in OUTPUT_COLUMNS):
        return {}
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


def mock_choice_top_prob(item: dict[str, Any], parsed: ParsedResponse) -> float:
    if item["benchmark"] != "mmlu":
        return 0.0
    return 0.7 if parsed.confidence and parsed.confidence >= 0.7 else 0.35


def mock_choice_entropy(item: dict[str, Any]) -> float:
    return 1.15 if item["benchmark"] == "mmlu" else 0.0


def _extract_choice(answer: str) -> str | None:
    text = str(answer).upper()
    for label in ("A", "B", "C", "D"):
        if label in text.split() or text.strip().startswith(label):
            return label
    return None


def _configure_transformers_cache(config: dict[str, Any]) -> None:
    # colab reruns should reuse downloaded weights
    raw_dir = Path(config.get("paths", {}).get("raw_dir", "data/raw"))
    cache_dir = raw_dir / "huggingface"
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    cache_dir.mkdir(parents=True, exist_ok=True)
