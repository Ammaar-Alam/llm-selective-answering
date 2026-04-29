# Data Directory

`raw/` stores downloaded benchmark data or model caches. `processed/` stores normalized item tables, model outputs, feature matrices, and intermediate CSV files. Both directories are excluded from Git because they can be regenerated from the pipeline.

The normalized item schema is:

```text
item_id
benchmark
source_split
split
subject_or_category
question
context
choices
gold_answer
gold_answer_index
gold_is_answerable
```

