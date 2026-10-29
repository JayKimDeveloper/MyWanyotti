# Model Training and Evaluation

## Purpose and status

The training pipeline consumes `training_eligible.parquet` under data-contract v0.3 and uses chronological splits. Model inputs, early stopping, and threshold selection use training/validation data only. The final test is evaluated once after the protocol is frozen.

A small pilot was blocked in preflight because required behavior classes were absent from training/validation. No valid model performance should be claimed from that pilot. Synthetic data may be used only to test engineering paths, not to report model quality.

## Example commands

```bash
.venv/bin/python scripts/train_catboost.py \
  --input data/processed/training/full/training_eligible.parquet \
  --dataset-report data/audit/training_pipeline/full/training_dataset_report.json \
  --output-dir models/catboost/full_v1 \
  --train-end 2021-01-01T00:00:00Z \
  --validation-end 2021-07-01T00:00:00Z
```

## Time split and purging

For boundaries `t1` and `t2`:

- train decisions precede `t1`, and their target end must not exceed `t1`;
- validation decisions are in `[t1,t2)`, and their target end must not exceed `t2`;
- test decisions are at or after `t2`.

Remove labels crossing a boundary. If an order appears in more than one split, purge every row containing that order. Verify that boundary violations and cross-split order overlap are zero after purging. If no explicit boundaries are supplied, the implementation uses a time-ordered 70/15/15 split by unique decision times; this does not replace a genuinely untouched future holdout.

## Preflight requirements

Require the v0.3 schema, target order identifiers, eligible rows only, unique `(symbol, decision_time_utc)` keys, strict-as-of market/position evidence, labels contained in their target interval, a fixed feature allowlist without target/PnL/future information, required class representation in training and validation, and zero post-purge boundary/order leakage. Record class counts and imbalance before fitting.

## Initial model configuration

The example uses `CatBoostClassifier` on CPU, 300 iterations, depth 5, seed 42, multiclass loss, balanced class weights derived from training data, validation early stopping, and the fixed `MODEL_FEATURE_COLUMNS` list. Convert decision time using the documented UTC representation and encode symbol as categorical. Do not impute from test statistics.

The majority-class baseline predicts the training majority label everywhere. Any confidence threshold is selected on validation only; test predictions are generated only after model and threshold are fixed. Confusion-matrix ordering must come from `model.classes_`.

## Evaluation and artifacts

Record per-class precision/recall/F1/support, macro-F1, accuracy, confusion matrices, baseline results, run durations, and maximum resident memory. Record input hashes, split boundaries, purge results, class order, model settings, and feature schema.

Expected successful outputs include the model, schema, class labels, training configuration, split and provenance reports, evaluation report/predictions, a reload probe, and a runtime contract. If preflight blocks training, do not create a model or performance report; save only the preflight and data-contract diagnostics.

Do not describe scores from unverified source data or synthetic tests as real-world performance. A behavior model is not a trading edge unless a separate, cost-aware evaluation establishes one.
