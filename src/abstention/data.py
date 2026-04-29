from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd


CHOICE_LABELS = ["A", "B", "C", "D"]


def build_items(config: dict[str, Any]) -> pd.DataFrame:
    # mock mode is the fast colab and ci path
    if config.get("mode") == "mock":
        return build_mock_items(config)
    frames: list[pd.DataFrame] = []
    if config.get("datasets", {}).get("use_mmlu", True):
        frames.append(load_mmlu(config))
    if config.get("datasets", {}).get("use_squad_v2", True):
        frames.append(load_squad_v2(config))
    if not frames:
        return pd.DataFrame()
    items = pd.concat(frames, ignore_index=True)
    return assign_splits(items, config)


def build_mock_items(config: dict[str, Any]) -> pd.DataFrame:
    # small fixed table that exercises both task types
    seed = int(config.get("random_seed", 312))
    rows: list[dict[str, Any]] = []
    for i in range(8):
        rows.append(
            {
                "item_id": f"mmlu_mock_{i}",
                "benchmark": "mmlu",
                "source_split": "mock",
                "subject_or_category": "college_computer_science" if i % 2 else "abstract_algebra",
                "question": f"Which option solves mock multiple-choice item {i}?",
                "context": "",
                "choices": json.dumps(["zero", "one", "two", "three"]),
                "gold_answer": CHOICE_LABELS[i % 4],
                "gold_answer_index": i % 4,
                "gold_is_answerable": True,
            }
        )
    for i in range(8):
        answerable = i % 2 == 0
        rows.append(
            {
                "item_id": f"squad_mock_{i}",
                "benchmark": "squad_v2",
                "source_split": "mock",
                "subject_or_category": "mock_context",
                "question": f"What is the key phrase for mock reading item {i}?",
                "context": f"The key phrase for item {i} is alpha {i}." if answerable else "",
                "choices": json.dumps([]),
                "gold_answer": f"alpha {i}" if answerable else "",
                "gold_answer_index": -1,
                "gold_is_answerable": answerable,
            }
        )
    return assign_splits(pd.DataFrame(rows), {**config, "random_seed": seed})


def load_mmlu(config: dict[str, Any]) -> pd.DataFrame:
    configure_hf_cache(config)
    from datasets import load_dataset

    # subject configs make later subject level splits possible
    sample_size = int(config.get("datasets", {}).get("sample_size_per_dataset", 1000))
    subjects = config.get("datasets", {}).get("mmlu_subjects", ["abstract_algebra"])
    per_subject = max(1, math.ceil(sample_size / max(1, len(subjects))))
    rows: list[dict[str, Any]] = []
    for subject in subjects:
        ds = load_dataset("cais/mmlu", subject, split="test")
        for idx, row in enumerate(ds):
            subject_count = sum(existing["subject_or_category"] == subject for existing in rows)
            if len(rows) >= sample_size or subject_count >= per_subject:
                break
            answer_index = int(row["answer"])
            rows.append(
                {
                    "item_id": f"mmlu_{subject}_{idx}",
                    "benchmark": "mmlu",
                    "source_split": "test",
                    "subject_or_category": subject,
                    "question": row["question"],
                    "context": "",
                    "choices": json.dumps(list(row["choices"])),
                    "gold_answer": CHOICE_LABELS[answer_index],
                    "gold_answer_index": answer_index,
                    "gold_is_answerable": True,
                }
            )
    return assign_splits(pd.DataFrame(rows), config)


def load_squad_v2(config: dict[str, Any]) -> pd.DataFrame:
    configure_hf_cache(config)
    from datasets import load_dataset

    # validation has answerable and unanswerable examples
    sample_size = int(config.get("datasets", {}).get("sample_size_per_dataset", 1000))
    target_answerable = int(round(sample_size * float(config.get("datasets", {}).get("squad_answerable_fraction", 0.5))))
    target_unanswerable = sample_size - target_answerable
    ds = load_dataset("rajpurkar/squad_v2", split="validation").shuffle(
        seed=int(config.get("random_seed", 312))
    )
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        if len(rows) >= sample_size:
            break
        answers = row.get("answers", {})
        answer_texts = list(answers.get("text", []))
        answerable = bool(answer_texts)
        have_answerable = sum(bool(existing["gold_is_answerable"]) for existing in rows)
        have_unanswerable = len(rows) - have_answerable
        if answerable and have_answerable >= target_answerable:
            continue
        if not answerable and have_unanswerable >= target_unanswerable:
            continue
        rows.append(
            {
                "item_id": f"squad_v2_{row['id']}",
                "benchmark": "squad_v2",
                "source_split": "validation",
                "subject_or_category": row.get("title", ""),
                "question": row["question"],
                "context": row["context"],
                "choices": json.dumps([]),
                "gold_answer": answer_texts[0] if answer_texts else "",
                "gold_answers": json.dumps(answer_texts),
                "gold_answer_index": -1,
                "gold_is_answerable": answerable,
            }
        )
    return assign_splits(pd.DataFrame(rows), config)


def assign_splits(items: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    # split after row construction so every downstream table agrees
    if items.empty:
        items["split"] = []
        return items
    seed = int(config.get("random_seed", 312))
    split_cfg = config.get("splits", {})
    train_frac = float(split_cfg.get("train", 0.6))
    val_frac = float(split_cfg.get("validation", 0.2))
    shuffled = items.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    train_end = int(round(n * train_frac))
    val_end = train_end + int(round(n * val_frac))
    split = ["train"] * train_end + ["validation"] * max(0, val_end - train_end)
    split += ["test"] * max(0, n - len(split))
    shuffled["split"] = split[:n]
    return shuffled.sort_values("item_id").reset_index(drop=True)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def load_dataframe(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def configure_hf_cache(config: dict[str, Any]) -> None:
    # keeps downloads out of home dir and out of git
    raw_dir = Path(config.get("paths", {}).get("raw_dir", "data/raw"))
    cache_dir = raw_dir / "huggingface"
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))
    cache_dir.mkdir(parents=True, exist_ok=True)
