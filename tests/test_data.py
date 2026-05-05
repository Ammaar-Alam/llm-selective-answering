import sys
import types

from abstention.data import load_mmlu


def test_load_mmlu_respects_requested_sample_size(monkeypatch, tmp_path) -> None:
    rows = [{"answer": 0, "question": f"q{i}", "choices": ["a", "b", "c", "d"]} for i in range(4)]

    def fake_load_dataset(*_args, **_kwargs):
        return rows

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    config = {
        "random_seed": 312,
        "paths": {"raw_dir": str(tmp_path / "raw")},
        "datasets": {
            "sample_size_per_dataset": 3,
            "mmlu_subjects": ["subject_a", "subject_b"],
        },
        "splits": {"train": 0.6, "validation": 0.2, "test": 0.2},
    }

    items = load_mmlu(config)

    assert len(items) == 3
