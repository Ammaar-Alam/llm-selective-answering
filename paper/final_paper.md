# Risk-Sensitive Abstention for Language Models

## Abstract

Large language models can produce useful answers, but the cost of a wrong answer depends on the setting. This project frames abstention as a supervised data science problem: predict whether a model's answer will be correct from measurable uncertainty and item-level features, then answer only when the predicted utility is positive. I evaluate this idea on 1,000 real model-output rows from MMLU and SQuAD 2.0 using `google/flan-t5-small`. The strongest correctness predictors in the current run are logistic regression and random forest models, with test AUC values of 0.644 and 0.641. Under a low wrong-answer cost, logistic regression improves test utility over always answering, but for larger penalties the learned policies often choose abstention, matching the conservative always-abstain baseline. These results support the project framing while also showing a key limitation: with this small model and simple features, useful risk-sensitive answering is possible only in the low-penalty region.

## 1. Introduction

Not every question should be answered. If a model is unsure, a wrong answer may be worse than no answer at all. This project studies that tradeoff by asking whether observable signals can predict when a language-model answer is worth giving.

The supervised prediction target is whether the model's answer is correct; the downstream decision is whether to answer or abstain under a specified wrong-answer cost. For a wrong-answer penalty `lambda`, answering has expected reward

```text
E[R(answer)] = p(correct) - lambda * (1 - p(correct))
```

and abstention has reward zero. Therefore the policy answers only when

```text
p(correct) > lambda / (1 + lambda)
```

This turns abstention into a standard prediction-and-decision pipeline.

## 2. Related Work

MMLU measures broad multiple-choice knowledge across many academic subjects. SQuAD 2.0 adds unanswerable reading-comprehension questions, making it useful for studying when a model should avoid answering. Prior work on selective prediction, calibration, and LLM abstention motivates the central idea: confidence should be evaluated not only as a descriptive score but also as an input to a decision rule.

This project does not train a reinforcement-learning policy. It instead uses a held-out supervised setup: estimate answer correctness, tune simple baselines on validation data, and report test-set utility, coverage, risk, ROC/AUC, and calibration.

## 3. Data

The current real run uses 1,000 item-level rows:

| Benchmark | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| MMLU | 313 | 95 | 92 | 500 |
| SQuAD 2.0 | 287 | 105 | 108 | 500 |

The MMLU sample is spread evenly across five subjects: abstract algebra, college computer science, high school US history, professional law, and moral scenarios. The SQuAD 2.0 sample is balanced between 250 answerable and 250 unanswerable examples and is shuffled before splitting.

Each row is normalized into a common schema with an item id, benchmark name, split, subject or category, question, optional context, choices, gold answer fields, and answerability. Model outputs are stored separately and joined by `item_id`.

## 4. Exploratory Data Analysis

The model used for this run is `google/flan-t5-small`. It produced no explicit abstentions in the 1,000-row run, so prompt-only abstention is equivalent to always answering in the current results. This is an important empirical finding: simply asking for abstention was not enough to make this model say IDK on the sampled items.

Answer correctness rates were:

| Benchmark | Train | Validation | Test |
|---|---:|---:|---:|
| MMLU | 0.217 | 0.274 | 0.174 |
| SQuAD 2.0 | 0.258 | 0.390 | 0.380 |

The figures in `paper/figures/` show confidence versus correctness, calibration, ROC curves, feature importance, utility by penalty, and risk-coverage behavior.

## 5. Methods

The pipeline has six stages:

1. Load and normalize MMLU and SQuAD 2.0.
2. Run cached transformers inference with `google/flan-t5-small`.
3. Parse the model answer and sequence-derived confidence.
4. Score answer correctness without counting correct abstention as answer correctness.
5. Train correctness predictors on the train split.
6. Tune policy thresholds on validation and evaluate final metrics separately on validation and test.

Features include benchmark, subject/category, question length, context length, choice count, mean choice length, verbal confidence, self-consistency proxy, choice margin, and choice entropy. The learned models are logistic regression, random forest, and gradient boosting. Threshold baselines are tuned on validation for verbal confidence, self-consistency, and choice margin.

## 6. Results

Correctness-prediction results:

| Model | Validation AUC | Validation ECE | Test AUC | Test ECE |
|---|---:|---:|---:|---:|
| Logistic regression | 0.619 | 0.096 | 0.644 | 0.072 |
| Random forest | 0.612 | 0.093 | 0.641 | 0.056 |
| Gradient boosting | 0.485 | 0.195 | 0.528 | 0.120 |

At `lambda = 0.5`, logistic regression is the best learned test policy in the current run. It answers 18.0% of test items with mean utility 0.0375 and risk 0.528. Always answering has full coverage but negative utility, -0.0725, because 71.5% of answered test items are wrong. At `lambda = 1`, the safest policies mostly abstain, and the best nonnegative utilities are zero. At `lambda = 5` and above, always answering collapses sharply, reaching -3.29 at `lambda = 5` and -6.865 at `lambda = 10`.

The main result is therefore conditional: correctness prediction is informative enough to improve low-penalty utility, but the current model-output quality is too low for high-penalty answering. In high-risk settings, the learned policy often chooses not to answer.

Key generated figures:

- `paper/figures/utility_vs_lambda.png`
- `paper/figures/risk_coverage.png`
- `paper/figures/roc_curve.png`
- `paper/figures/calibration.png`
- `paper/figures/confidence_correctness.png`
- `paper/figures/feature_importance.png`

## 7. Error Analysis

The current run shows three clear failure modes.

First, prompt-only abstention failed because the model did not produce explicit abstentions. This means the model's generated text alone did not adapt to the answerability risk.

Second, confidence is noisy. Validation-tuned verbal-confidence thresholds helped at low penalty, but the test risk among answered items remained high. At `lambda = 0.5`, the verbal-confidence threshold policy answered 70.5% of test items with risk 0.667 and zero mean utility.

Third, learned models improved ranking but not enough for high-stakes answering. Logistic regression and random forest had moderate test AUC, but once the utility cutoff rose above 0.5, many predicted probabilities were not high enough to justify answering.

## 8. Discussion

The data-science framing works: the project produces a concrete decision table, explicit features, a binary correctness target, train/validation/test evaluation, and interpretable utility curves. The results also show why abstention is necessary. Always answering looks acceptable only if wrong answers are cheap. Once wrong answers are costly, full coverage creates large negative utility.

The strongest current predictors are logistic regression and random forest. Their test AUC values are similar, and both are better calibrated than gradient boosting on this run. Feature importance suggests that choice-margin style signals, self-consistency proxy, verbal confidence, benchmark/category indicators, and text-length features all contribute.

## 9. Limitations and Ethics

This is a small open-model experiment. `google/flan-t5-small` is useful for a reproducible classroom-scale pipeline, but its answer quality is not representative of larger modern LLMs. The current self-consistency and margin features are simple proxies, not full sampled-consistency or true logit-margin features. SQuAD 2.0 and MMLU are benchmark datasets, not real deployment settings.

Abstention reduces the harm of wrong answers, but it also withholds potentially useful answers. A real system would need stronger calibration, domain-specific costs, and careful monitoring of who is denied useful responses.

## 10. Conclusion

This project shows how risk-sensitive abstention can be studied as a supervised prediction problem followed by a utility cutoff. On 1,000 real FLAN-T5 output rows, learned correctness predictors achieve moderate discrimination and improve low-penalty utility over always answering. For higher penalties, the best safe behavior is usually abstention, which honestly reflects the limits of the current model and feature set. The next step is to strengthen the uncertainty features, run a larger Colab experiment, and test whether a stronger instruction-tuned model produces better calibrated answerability behavior.
