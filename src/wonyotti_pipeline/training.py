"""Leakage-safe temporal splitting and evaluation helpers for CatBoost."""

from __future__ import annotations

import hashlib
import json
import math
import resource
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .dataset import MODEL_FEATURE_COLUMNS


REQUIRED_ACTION_LABELS = [
    "LONG_INCREASE",
    "LONG_DECREASE",
    "SHORT_INCREASE",
    "SHORT_DECREASE",
    "NO_FILL",
]
FORBIDDEN_FEATURE_TOKENS = (
    "target_",
    "next_",
    "future_",
    "pnl",
    "profit",
    "lastpx",
)
FEATURE_DEFINITIONS = {
    "symbol": {
        "input_type": "string",
        "model_type": "categorical_string",
        "unit": "SYMBOL",
        "nullable": False,
        "transform": "missing->__MISSING__",
    },
    "decision_time_utc": {
        "input_type": "timezone_aware_datetime",
        "model_type": "float64",
        "unit": "UTC",
        "nullable": False,
        "transform": "UTC timestamp to Unix seconds",
    },
    "position_before_qty": {
        "input_type": "integer",
        "model_type": "float64",
        "unit": "CONTRACTS",
        "nullable": False,
        "transform": "numeric cast; NaN preserved for CatBoost",
    },
    "market_trade_count": {
        "input_type": "integer",
        "model_type": "float64",
        "unit": "COUNT",
        "nullable": False,
        "transform": "numeric cast; NaN preserved for CatBoost",
    },
    "market_volume": {
        "input_type": "number",
        "model_type": "float64",
        "unit": "CONTRACTS",
        "nullable": False,
        "transform": "numeric cast; NaN preserved for CatBoost",
    },
    "market_buy_volume": {
        "input_type": "number",
        "model_type": "float64",
        "unit": "CONTRACTS",
        "nullable": False,
        "transform": "numeric cast; NaN preserved for CatBoost",
    },
    "market_sell_volume": {
        "input_type": "number",
        "model_type": "float64",
        "unit": "CONTRACTS",
        "nullable": False,
        "transform": "numeric cast; NaN preserved for CatBoost",
    },
    "market_signed_volume": {
        "input_type": "number",
        "model_type": "float64",
        "unit": "CONTRACTS",
        "nullable": False,
        "transform": "numeric cast; NaN preserved for CatBoost",
    },
    "market_vwap": {
        "input_type": "number",
        "model_type": "float64",
        "unit": "SYMBOL_PRICE_UNITS",
        "nullable": False,
        "transform": "numeric cast; NaN preserved for CatBoost",
    },
    "market_last_price": {
        "input_type": "number",
        "model_type": "float64",
        "unit": "SYMBOL_PRICE_UNITS",
        "nullable": False,
        "transform": "numeric cast; NaN preserved for CatBoost",
    },
    "market_return_lookback": {
        "input_type": "number",
        "model_type": "float64",
        "unit": "RATIO",
        "nullable": True,
        "transform": "numeric cast; NaN preserved for CatBoost",
    },
    "bid_price": {
        "input_type": "number",
        "model_type": "float64",
        "unit": "SYMBOL_PRICE_UNITS",
        "nullable": False,
        "transform": "numeric cast; NaN preserved for CatBoost",
    },
    "ask_price": {
        "input_type": "number",
        "model_type": "float64",
        "unit": "SYMBOL_PRICE_UNITS",
        "nullable": False,
        "transform": "numeric cast; NaN preserved for CatBoost",
    },
    "bid_size": {
        "input_type": "number",
        "model_type": "float64",
        "unit": "CONTRACTS",
        "nullable": False,
        "transform": "numeric cast; NaN preserved for CatBoost",
    },
    "ask_size": {
        "input_type": "number",
        "model_type": "float64",
        "unit": "CONTRACTS",
        "nullable": False,
        "transform": "numeric cast; NaN preserved for CatBoost",
    },
    "mid_price": {
        "input_type": "number",
        "model_type": "float64",
        "unit": "SYMBOL_PRICE_UNITS",
        "nullable": False,
        "transform": "numeric cast; NaN preserved for CatBoost",
    },
    "spread": {
        "input_type": "number",
        "model_type": "float64",
        "unit": "SYMBOL_PRICE_UNITS",
        "nullable": False,
        "transform": "numeric cast; NaN preserved for CatBoost",
    },
    "spread_bps": {
        "input_type": "number",
        "model_type": "float64",
        "unit": "BASIS_POINTS",
        "nullable": False,
        "transform": "numeric cast; NaN preserved for CatBoost",
    },
    "top_book_imbalance": {
        "input_type": "number",
        "model_type": "float64",
        "unit": "RATIO",
        "nullable": False,
        "transform": "numeric cast; NaN preserved for CatBoost",
    },
    "quote_age_ms": {
        "input_type": "number",
        "model_type": "float64",
        "unit": "MILLISECONDS",
        "nullable": False,
        "transform": "numeric cast; NaN preserved for CatBoost",
    },
}


class TrainingDataError(ValueError):
    """Raised when the training contract cannot be satisfied."""


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def data_version(path: Path, frame: pd.DataFrame) -> dict[str, object]:
    stat = path.stat()
    return {
        "input_path": str(path.resolve()),
        "sha256": file_sha256(path),
        "size_bytes": stat.st_size,
        "modified_time_utc": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat(),
        "eligible_frame_row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "decision_start_utc": frame["decision_time_utc"].min().isoformat(),
        "decision_end_utc": frame["decision_time_utc"].max().isoformat(),
        "target_end_utc": frame["target_end_time_utc"].max().isoformat(),
        "data_contract_version": "v0.3",
    }


def load_training_data(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = pd.read_parquet(path)
    required = {
        *MODEL_FEATURE_COLUMNS,
        "target_end_time_utc",
        "target_label",
        "target_order_ids",
        "training_eligible",
        "market_feature_max_time_utc",
        "position_asof_time_utc",
        "target_first_event_time_utc",
        "target_last_event_time_utc",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise TrainingDataError(
            "input Parquet does not satisfy the training contract; missing columns: "
            + ", ".join(missing)
        )

    for column in (
        "decision_time_utc",
        "target_end_time_utc",
        "market_feature_max_time_utc",
        "position_asof_time_utc",
        "target_first_event_time_utc",
        "target_last_event_time_utc",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True)

    input_rows = len(frame)
    frame = frame.loc[frame["training_eligible"].eq(True)].copy()
    frame = frame.sort_values(["decision_time_utc", "symbol"], kind="stable")
    frame = frame.reset_index(drop=True)
    return frame, {
        "input_row_count": int(input_rows),
        "eligible_row_count": int(len(frame)),
        "ineligible_row_count": int(input_rows - len(frame)),
    }


def validate_input_contract(frame: pd.DataFrame) -> dict[str, object]:
    duplicate_keys = frame.duplicated(["symbol", "decision_time_utc"])
    selected_forbidden = [
        column
        for column in MODEL_FEATURE_COLUMNS
        if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    unexpected_labels = sorted(
        set(frame["target_label"].dropna().astype(str)) - set(REQUIRED_ACTION_LABELS)
    )
    checks = {
        "duplicate_decision_key_count": int(duplicate_keys.sum()),
        "null_decision_time_count": int(frame["decision_time_utc"].isna().sum()),
        "null_target_end_time_count": int(frame["target_end_time_utc"].isna().sum()),
        "null_target_label_count": int(frame["target_label"].isna().sum()),
        "unexpected_labels": unexpected_labels,
        "forbidden_selected_features": selected_forbidden,
        "market_feature_at_or_after_decision_count": int(
            (
                frame["market_feature_max_time_utc"].notna()
                & frame["market_feature_max_time_utc"].ge(frame["decision_time_utc"])
            ).sum()
        ),
        "position_event_at_or_after_decision_count": int(
            (
                frame["position_asof_time_utc"].notna()
                & frame["position_asof_time_utc"].ge(frame["decision_time_utc"])
            ).sum()
        ),
        "target_before_decision_count": int(
            (
                frame["target_first_event_time_utc"].notna()
                & frame["target_first_event_time_utc"].lt(
                    frame["decision_time_utc"]
                )
            ).sum()
        ),
        "target_at_or_after_end_count": int(
            (
                frame["target_last_event_time_utc"].notna()
                & frame["target_last_event_time_utc"].ge(
                    frame["target_end_time_utc"]
                )
            ).sum()
        ),
        "nonpositive_target_interval_count": int(
            frame["target_end_time_utc"].le(frame["decision_time_utc"]).sum()
        ),
    }
    blocking_counts = [
        checks["duplicate_decision_key_count"],
        checks["null_decision_time_count"],
        checks["null_target_end_time_count"],
        checks["null_target_label_count"],
        checks["market_feature_at_or_after_decision_count"],
        checks["position_event_at_or_after_decision_count"],
        checks["target_before_decision_count"],
        checks["target_at_or_after_end_count"],
        checks["nonpositive_target_interval_count"],
    ]
    checks["passed"] = bool(
        not any(blocking_counts)
        and not unexpected_labels
        and not selected_forbidden
        and len(frame) > 0
    )
    return checks


def choose_split_boundaries(
    frame: pd.DataFrame,
    train_end: pd.Timestamp | None,
    validation_end: pd.Timestamp | None,
    train_ratio: float,
    validation_ratio: float,
) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    if (train_end is None) != (validation_end is None):
        raise TrainingDataError("train_end and validation_end must be provided together")
    if train_end is not None and validation_end is not None:
        if train_end >= validation_end:
            raise TrainingDataError("train_end must be before validation_end")
        return train_end, validation_end, "EXPLICIT"

    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1:
        raise TrainingDataError("split ratios must be between zero and one")
    if train_ratio + validation_ratio >= 1:
        raise TrainingDataError("train_ratio + validation_ratio must be less than one")

    times = pd.DatetimeIndex(frame["decision_time_utc"].drop_duplicates().sort_values())
    if len(times) < 3:
        raise TrainingDataError("at least three distinct decision times are required")
    train_index = max(1, math.floor(len(times) * train_ratio))
    validation_index = max(
        train_index + 1,
        math.floor(len(times) * (train_ratio + validation_ratio)),
    )
    if validation_index >= len(times):
        raise TrainingDataError("ratios leave no decision time for the test split")
    return times[train_index], times[validation_index], "RATIO_BY_DECISION_TIME"


def _order_ids(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return {item for item in str(value).split("|") if item}


def chronological_split(
    frame: pd.DataFrame,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    working = frame.copy()
    decision = working["decision_time_utc"]
    working["_split"] = np.select(
        [decision.lt(train_end), decision.lt(validation_end)],
        ["train", "validation"],
        default="test",
    )

    crosses_train = working["_split"].eq("train") & working[
        "target_end_time_utc"
    ].gt(train_end)
    crosses_validation = working["_split"].eq("validation") & working[
        "target_end_time_utc"
    ].gt(validation_end)
    boundary_crossing = crosses_train | crosses_validation
    retained = working.loc[~boundary_crossing].copy()

    order_splits: dict[str, set[str]] = {}
    for split_name, target_order_ids in retained[
        ["_split", "target_order_ids"]
    ].itertuples(index=False, name=None):
        for order_id in _order_ids(target_order_ids):
            order_splits.setdefault(order_id, set()).add(split_name)
    crossing_orders = {
        order_id for order_id, splits in order_splits.items() if len(splits) > 1
    }
    order_crossing = retained["target_order_ids"].map(
        lambda value: bool(_order_ids(value) & crossing_orders)
    )
    purged_orders = retained.loc[order_crossing].copy()
    retained = retained.loc[~order_crossing].copy()

    splits = {
        name: retained.loc[retained["_split"].eq(name)]
        .drop(columns="_split")
        .reset_index(drop=True)
        for name in ("train", "validation", "test")
    }
    report = {
        "train_end_utc": train_end.isoformat(),
        "validation_end_utc": validation_end.isoformat(),
        "boundary_rule": (
            "train decision<t1 and target_end<=t1; validation t1<=decision<t2 "
            "and target_end<=t2; test decision>=t2"
        ),
        "boundary_crossing_row_count": int(boundary_crossing.sum()),
        "boundary_crossing_counts": {
            name: int((boundary_crossing & working["_split"].eq(name)).sum())
            for name in ("train", "validation", "test")
        },
        "cross_split_order_count": len(crossing_orders),
        "cross_split_order_ids_sample": sorted(crossing_orders)[:20],
        "order_purged_row_count": int(len(purged_orders)),
        "order_purged_counts": {
            name: int(purged_orders["_split"].eq(name).sum())
            for name in ("train", "validation", "test")
        },
        "retained_counts": {name: int(len(split)) for name, split in splits.items()},
    }
    return splits, report


def split_leakage_checks(
    splits: dict[str, pd.DataFrame],
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> dict[str, object]:
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]
    order_sets = {
        name: set().union(*(_order_ids(value) for value in split["target_order_ids"]))
        if len(split)
        else set()
        for name, split in splits.items()
    }
    overlap = (
        (order_sets["train"] & order_sets["validation"])
        | (order_sets["train"] & order_sets["test"])
        | (order_sets["validation"] & order_sets["test"])
    )
    checks = {
        "train_target_crossing_count": int(
            train["target_end_time_utc"].gt(train_end).sum()
        ),
        "validation_target_crossing_count": int(
            validation["target_end_time_utc"].gt(validation_end).sum()
        ),
        "train_decision_at_or_after_boundary_count": int(
            train["decision_time_utc"].ge(train_end).sum()
        ),
        "validation_decision_outside_boundary_count": int(
            (
                validation["decision_time_utc"].lt(train_end)
                | validation["decision_time_utc"].ge(validation_end)
            ).sum()
        ),
        "test_decision_before_boundary_count": int(
            test["decision_time_utc"].lt(validation_end).sum()
        ),
        "cross_split_order_count_after_purge": len(overlap),
        "cross_split_order_ids_after_purge_sample": sorted(overlap)[:20],
    }
    checks["passed"] = not any(
        value for key, value in checks.items() if key.endswith("_count")
    )
    return checks


def class_distribution(
    frame: pd.DataFrame, labels: Iterable[str] = REQUIRED_ACTION_LABELS
) -> dict[str, object]:
    counts = frame["target_label"].value_counts().to_dict()
    total = len(frame)
    ordered_counts = {label: int(counts.get(label, 0)) for label in labels}
    nonzero = [count for count in ordered_counts.values() if count > 0]
    return {
        "row_count": int(total),
        "counts": ordered_counts,
        "proportions": {
            label: (count / total if total else 0.0)
            for label, count in ordered_counts.items()
        },
        "majority_to_minority_ratio": (
            max(nonzero) / min(nonzero) if nonzero else None
        ),
    }


def class_coverage_issues(
    splits: dict[str, pd.DataFrame],
    required_labels: Iterable[str],
    minimum_count: int,
) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    for split_name, split in splits.items():
        counts = split["target_label"].value_counts()
        for label in required_labels:
            count = int(counts.get(label, 0))
            if count < minimum_count:
                issues.append(
                    {
                        "split": split_name,
                        "label": label,
                        "count": count,
                        "minimum_required": minimum_count,
                    }
                )
    return issues


def feature_schema(frame: pd.DataFrame) -> dict[str, object]:
    fields: list[dict[str, object]] = []
    for column in MODEL_FEATURE_COLUMNS:
        definition = FEATURE_DEFINITIONS[column]
        fields.append(
            {
                "name": column,
                "source_dtype": str(frame[column].dtype),
                **definition,
            }
        )
    return {
        "data_contract_version": "v0.3",
        "feature_selection": "FIXED_CONTRACT_ALLOWLIST",
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "categorical_features": ["symbol"],
        "forbidden_feature_tokens": list(FORBIDDEN_FEATURE_TOKENS),
        "fields": fields,
    }


def prepare_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame[MODEL_FEATURE_COLUMNS].copy()
    prepared["symbol"] = prepared["symbol"].astype("string").fillna("__MISSING__")
    prepared["symbol"] = prepared["symbol"].astype(str)
    decision = pd.to_datetime(prepared["decision_time_utc"], utc=True)
    prepared["decision_time_utc"] = decision.astype("int64") / 1_000_000_000.0
    for column in MODEL_FEATURE_COLUMNS:
        if column in {"symbol", "decision_time_utc"}:
            continue
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce").astype(
            "float64"
        )
    numeric = prepared.drop(columns="symbol")
    if np.isinf(numeric.to_numpy(dtype="float64")).any():
        raise TrainingDataError("infinite numeric feature value found")
    return prepared


def serialize_feature_row(row: pd.Series) -> dict[str, object]:
    record: dict[str, object] = {}
    for column in MODEL_FEATURE_COLUMNS:
        value = row[column]
        definition = FEATURE_DEFINITIONS[column]
        if pd.isna(value):
            record[column] = None
        elif definition["input_type"] == "timezone_aware_datetime":
            record[column] = pd.Timestamp(value).tz_convert("UTC").isoformat()
        elif definition["input_type"] == "integer":
            record[column] = int(value)
        elif definition["input_type"] == "number":
            record[column] = float(value)
        else:
            record[column] = str(value)
    return record


def majority_label(labels: pd.Series) -> str:
    counts = labels.astype(str).value_counts().to_dict()
    if not counts:
        raise TrainingDataError("cannot select a majority class from an empty split")
    return sorted(counts, key=lambda label: (-counts[label], label))[0]


def classification_metrics(
    truth: Iterable[str], predictions: Iterable[str], class_labels: list[str]
) -> dict[str, object]:
    truth_values = [str(value) for value in truth]
    prediction_values = [str(value) for value in predictions]
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    unknown = sorted(
        (set(truth_values) | set(prediction_values)) - set(class_labels)
    )
    if unknown:
        raise TrainingDataError(f"metrics contain classes absent from model: {unknown}")
    matrix = np.zeros((len(class_labels), len(class_labels)), dtype="int64")
    for actual, predicted in zip(truth_values, prediction_values, strict=True):
        matrix[label_to_index[actual], label_to_index[predicted]] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for index, label in enumerate(class_labels):
        true_positive = int(matrix[index, index])
        support = int(matrix[index, :].sum())
        predicted_count = int(matrix[:, index].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted_count": predicted_count,
        }
    total = int(matrix.sum())
    return {
        "row_count": total,
        "accuracy": float(np.trace(matrix) / total) if total else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": {
            "row_order_actual": class_labels,
            "column_order_predicted": class_labels,
            "values": matrix.tolist(),
        },
    }


def threshold_predictions(
    probabilities: np.ndarray,
    class_labels: list[str],
    threshold: float,
    fallback_label: str,
) -> list[str]:
    best_indices = probabilities.argmax(axis=1)
    confidence = probabilities[np.arange(len(probabilities)), best_indices]
    return [
        class_labels[index] if confidence[row] >= threshold else fallback_label
        for row, index in enumerate(best_indices)
    ]


def choose_validation_threshold(
    probabilities: np.ndarray,
    truth: pd.Series,
    class_labels: list[str],
    fallback_label: str,
    grid_step: float,
) -> tuple[float, list[dict[str, float]]]:
    if not 0 < grid_step <= 1:
        raise TrainingDataError("threshold_grid_step must be in (0, 1]")
    thresholds = np.arange(0.0, 1.0 + grid_step / 2, grid_step)
    candidates: list[dict[str, float]] = []
    for threshold in thresholds:
        predictions = threshold_predictions(
            probabilities, class_labels, float(threshold), fallback_label
        )
        metrics = classification_metrics(truth, predictions, class_labels)
        candidates.append(
            {"threshold": float(threshold), "macro_f1": metrics["macro_f1"]}
        )
    best = sorted(
        candidates, key=lambda item: (-item["macro_f1"], item["threshold"])
    )[0]
    return best["threshold"], candidates


def max_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(value)
    return int(value * 1024)
