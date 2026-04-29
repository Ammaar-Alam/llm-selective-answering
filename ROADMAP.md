# Roadmap

## Phase 1: Baseline Pipeline

- Load MMLU and SQuAD 2.0 into one normalized item table.
- Keep deterministic train, validation, and test splits.
- Run deterministic mock inference for smoke tests.
- Score MMLU choices and SQuAD answerability.
- Build item-level and uncertainty-style features.
- Train logistic regression and random forest correctness predictors.
- Evaluate utility, coverage, risk, ROC/AUC, and calibration.

## Phase 2: Draft Experiment

- Replace mock data with sampled Hugging Face datasets.
- Keep mock inference available as a fallback.
- Add cached real model outputs for a small instruction-tuned model.
- Generate draft figures and tables.
- Write the first complete paper draft from actual outputs.

## Phase 3: Final Experiment

- Increase dataset sizes if Colab runtime allows.
- Add ablations by feature group.
- Add gradient boosting.
- Add qualitative error examples.
- Finalize paper and presentation figures.

## Phase 4: Extension

- Add TruthfulQA as a stress test.
- Test cross-subject or cross-benchmark generalization.
- Add stronger calibration diagnostics.
