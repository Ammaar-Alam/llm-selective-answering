# LLM Selective Answering

This project studies whether item-level features and uncertainty signals can predict when a language-model answer will be correct, then uses those predictions to decide whether to answer or abstain under different wrong-answer costs.

The supervised target is whether the model's answer is correct; the downstream decision is whether to answer or abstain under a specified wrong-answer cost.

## Research Question

Can measurable uncertainty and item-level features predict when an LLM answer will be correct, and can those predictions be converted into cost-sensitive answer/abstain decisions that improve expected utility across varying wrong-answer penalties?

## Project Structure

```text
.
├── configs/              # Run settings for smoke, baseline, and confirmatory experiments
├── data/                 # Raw and processed data, not tracked by Git
├── scripts/              # Pipeline and analysis entrypoints
├── src/abstention/       # Reusable project package
└── tests/                # Unit tests for core logic
```

Generated datasets, model outputs, metrics, and figures are excluded from Git. The public repository tracks the code and configs needed to regenerate them.

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

The smoke test uses deterministic mock data and mock model outputs, so it does not download full benchmarks or run transformer inference.

```bash
pytest -q
python3 scripts/07_run_experiment.py --config configs/smoke_test.yaml --run-name smoke_test
```

The equivalent manual pipeline is:

```bash
python3 scripts/01_build_items.py --config configs/smoke_test.yaml
python3 scripts/02_run_model.py --config configs/smoke_test.yaml
python3 scripts/03_build_features.py --config configs/smoke_test.yaml
python3 scripts/04_train_models.py --config configs/smoke_test.yaml
python3 scripts/05_evaluate_policies.py --config configs/smoke_test.yaml
python3 scripts/06_make_figures.py --config configs/smoke_test.yaml
```

## Main Experiments

The final writeup uses these public configs:

| Purpose | Config |
|---|---|
| Baseline 500-per-dataset run | `configs/draft_500.yaml` |
| No-category confirmatory run | `configs/draft_500_no_category.yaml` |
| Strict-format robustness run | `configs/draft_1000_strict_json.yaml` |

Run the baseline pipeline with:

```bash
python3 scripts/07_run_experiment.py --config configs/draft_500.yaml --run-name draft_500
```

Run the two headline confirmatory experiments with:

```bash
python3 scripts/07_run_experiment.py \
  --config configs/draft_500_no_category.yaml \
  --run-name draft_500_no_category
python3 scripts/07_run_experiment.py \
  --config configs/draft_1000_strict_json.yaml \
  --run-name draft_1000_strict_json
python3 scripts/08_confirmatory_analysis.py \
  --runs results/runs/draft_500_no_category results/runs/draft_1000_strict_json \
  --lambda-cost 0.5 \
  --bootstrap-reps 10000
```

The confirmatory analysis selects the classifier by validation AUC, evaluates once on the held-out test split, and computes paired-bootstrap comparisons against always-answer, prompt-only, and always-abstain baselines.

Fresh transformer inference can vary slightly across package, model, and dataset versions. The repository therefore emphasizes reproducible code, fixed configs, fixed random seeds, and explicit generated-output paths rather than committing generated artifacts.

## Data

Primary datasets:

- MMLU for multiple-choice knowledge questions.
- SQuAD 2.0 for answerability and abstention-style reading comprehension.

TruthfulQA is reserved as an optional held-out stress test. Raw data and generated processed files are excluded from Git.

## Features

The feature table combines response-level signals and item-level metadata. The self-consistency field is a proxy stability signal in this implementation; it is not a full multi-sample self-consistency decoding experiment.

## Metrics

The main evaluation reports:

- Mean utility across wrong-answer penalties.
- Coverage, the fraction of items answered.
- Risk, the error rate among answered items.
- ROC/AUC for correctness prediction.
- Calibration of predicted correctness probabilities.

The answer decision uses the cutoff `lambda / (1 + lambda)`, where `lambda` is the cost of a wrong answer.
