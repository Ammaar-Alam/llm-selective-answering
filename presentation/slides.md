# Risk-Sensitive Abstention for Language Models

## 1. Question

Can uncertainty and item-level features predict when a model answer will be correct, and can that prediction decide when to abstain?

## 2. Decision Rule

- Reward for correct answer: `+1`
- Penalty for wrong answer: `-lambda`
- Reward for abstaining: `0`
- Answer when `p(correct) > lambda / (1 + lambda)`

## 3. Data

- 1,000 real model-output rows
- 500 MMLU questions across five subjects
- 500 SQuAD 2.0 questions
- SQuAD sample balanced: 250 answerable, 250 unanswerable
- Train/validation/test split kept separate

## 4. Model Output

- Model: `google/flan-t5-small`
- Inference was cached for reproducibility
- The model produced no explicit abstentions in this run
- Correct abstention is tracked separately from answer correctness

## 5. Features

- Benchmark and subject/category
- Question length and context length
- Choice count and mean choice length
- Verbal confidence
- Self-consistency proxy
- Choice margin and entropy proxies

## 6. Correctness Prediction

| Model | Validation AUC | Test AUC | Test ECE |
|---|---:|---:|---:|
| Logistic regression | 0.619 | 0.644 | 0.072 |
| Random forest | 0.612 | 0.641 | 0.056 |
| Gradient boosting | 0.485 | 0.528 | 0.120 |

Figure: `presentation/figures/roc_curve.png`

## 7. Policy Results

At `lambda = 0.5`:

- Logistic regression test utility: `0.0375`
- Logistic regression coverage: `0.180`
- Always-answer test utility: `-0.0725`
- Always-answer risk: `0.715`

At `lambda >= 1`, most safe policies abstain heavily.

Figures:

- `presentation/figures/utility_vs_lambda.png`
- `presentation/figures/risk_coverage.png`

## 8. Main Takeaway

The supervised framing works, but the current small model is not reliable enough for high-penalty answering.

For cheap mistakes, learned correctness prediction can improve utility. For expensive mistakes, abstention is usually the safer decision.

## 9. Limitations

- Small open model
- Simple uncertainty proxies
- No explicit abstentions from the model
- Benchmark tasks are not real deployment settings
- Larger or better calibrated models may change the tradeoff

## 10. Next Steps

- Run stronger model outputs in Colab
- Add true self-consistency sampling
- Add logit-based choice probabilities where available
- Add ablation table
- Test TruthfulQA as a stress test
