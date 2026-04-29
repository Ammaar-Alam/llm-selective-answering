# Google Colab Setup

Use this path when running the project from Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/SML312
!pip install -e ".[dev]"
```

Run the smoke pipeline first:

```python
!python scripts/01_build_items.py --config configs/smoke_test.yaml
!python scripts/02_run_model.py --config configs/smoke_test.yaml
!python scripts/03_build_features.py --config configs/smoke_test.yaml
!python scripts/04_train_models.py --config configs/smoke_test.yaml
!python scripts/05_evaluate_policies.py --config configs/smoke_test.yaml
!python scripts/06_make_figures.py --config configs/smoke_test.yaml
```

After the smoke run works, switch to:

```python
!python scripts/01_build_items.py --config configs/draft.yaml
```

Then run the remaining numbered scripts in order.

For the larger run used by the current draft:

```python
!python scripts/01_build_items.py --config configs/draft_500.yaml
!python scripts/02_run_model.py --config configs/draft_500.yaml
!python scripts/03_build_features.py --config configs/draft_500.yaml
!python scripts/04_train_models.py --config configs/draft_500.yaml
!python scripts/05_evaluate_policies.py --config configs/draft_500.yaml
!python scripts/06_make_figures.py --config configs/draft_500.yaml
```

Generated tables and plots appear in:

```text
results/tables/
results/figures/
results/metrics/
```
