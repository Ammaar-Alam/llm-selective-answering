# LLM Selective Answering

This project studies whether item-level features and uncertainty signals can predict when a language-model answer will be correct, then uses those predictions to decide whether to answer or abstain under different wrong-answer costs.

The supervised target is whether the model's answer is correct; the downstream decision is whether to answer or abstain under a specified wrong-answer cost.

## Research Question

Can measurable uncertainty and item-level features predict when an LLM answer will be correct, and can those predictions be converted into cost-sensitive answer/abstain decisions that improve expected utility across varying wrong-answer penalties?

## Project Structure

```text
.
├── configs/              # Run settings for smoke, draft, and full experiments
├── data/                 # Raw and processed data, not tracked by Git
├── scripts/              # Pipeline entrypoints
├── src/abstention/       # Reusable project package
└── tests/                # Unit tests for core logic
```

## Setup

Local Python setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Conda setup:

```bash
conda env create -f environment.yml
conda activate sml312-abstention
pip install -e .
```

## Smoke Test

The smoke test uses deterministic mock data and mock model outputs, so it does not download full benchmarks or run model inference.

```bash
pytest -q
python3 scripts/01_build_items.py --config configs/smoke_test.yaml
python3 scripts/02_run_model.py --config configs/smoke_test.yaml
python3 scripts/03_build_features.py --config configs/smoke_test.yaml
python3 scripts/04_train_models.py --config configs/smoke_test.yaml
python3 scripts/05_evaluate_policies.py --config configs/smoke_test.yaml
python3 scripts/06_make_figures.py --config configs/smoke_test.yaml
```

## Full Pipeline

Use the draft config first:

```bash
python3 scripts/01_build_items.py --config configs/draft.yaml
```

Then run each subsequent numbered script in order. The draft and full configs run the same pipeline at larger sample sizes.

The larger current run uses:

```bash
python3 scripts/01_build_items.py --config configs/draft_500.yaml
python3 scripts/02_run_model.py --config configs/draft_500.yaml
python3 scripts/03_build_features.py --config configs/draft_500.yaml
python3 scripts/04_train_models.py --config configs/draft_500.yaml
python3 scripts/05_evaluate_policies.py --config configs/draft_500.yaml
MPLCONFIGDIR=/tmp/sml312-mpl python3 scripts/06_make_figures.py --config configs/draft_500.yaml
```

The `MPLCONFIGDIR` prefix avoids local Matplotlib font-cache permission issues on some Macs.

## Data

Primary datasets:

- MMLU for multiple-choice knowledge questions.
- SQuAD 2.0 for answerability and abstention-style reading comprehension.

TruthfulQA is reserved as an optional held-out stress test. Raw data and generated processed files are excluded from Git.

## Metrics

The main evaluation reports:

- Mean utility across wrong-answer penalties.
- Coverage, the fraction of items answered.
- Risk, the error rate among answered items.
- ROC/AUC for correctness prediction.
- Calibration of predicted correctness probabilities.

The answer decision uses the cutoff `lambda / (1 + lambda)`, where `lambda` is the cost of a wrong answer.
