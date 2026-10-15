"""Load-once CatBoost inference with strict artifact and feature validation."""

from __future__ import annotations

import json
import math
import numbers
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .dataset import MODEL_FEATURE_COLUMNS
from .training import FEATURE_DEFINITIONS, prepare_model_features


class InferenceContractError(ValueError):
    """Raised when model artifacts or an inference request violate the contract."""


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise InferenceContractError(f"required artifact not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise InferenceContractError(f"JSON artifact must contain an object: {path}")
    return value


def validate_feature_schema(schema: Mapping[str, object]) -> None:
    columns = schema.get("feature_columns")
    if columns != MODEL_FEATURE_COLUMNS:
        raise InferenceContractError(
            "feature_schema.json column order does not match MODEL_FEATURE_COLUMNS"
        )
    if schema.get("categorical_features") != ["symbol"]:
        raise InferenceContractError("categorical feature contract must be ['symbol']")
    if schema.get("data_contract_version") != "v0.3":
        raise InferenceContractError("feature schema must use data contract v0.3")

    fields = schema.get("fields")
    if not isinstance(fields, list):
        raise InferenceContractError("feature schema fields must be a list")
    field_names = [field.get("name") for field in fields if isinstance(field, dict)]
    if field_names != MODEL_FEATURE_COLUMNS:
        raise InferenceContractError("feature schema field order is invalid")
    for field in fields:
        if not isinstance(field, dict):
            raise InferenceContractError("feature schema field must be an object")
        name = str(field["name"])
        expected = FEATURE_DEFINITIONS[name]
        for key in ("input_type", "model_type", "unit", "nullable", "transform"):
            if field.get(key) != expected[key]:
                raise InferenceContractError(
                    f"feature schema mismatch for {name}.{key}: "
                    f"expected {expected[key]!r}, got {field.get(key)!r}"
                )


def validate_feature_record(record: Mapping[str, object]) -> pd.DataFrame:
    provided = set(record)
    expected = set(MODEL_FEATURE_COLUMNS)
    missing = sorted(expected - provided)
    extra = sorted(provided - expected)
    if missing or extra:
        raise InferenceContractError(
            f"feature columns mismatch; missing={missing}, extra={extra}"
        )

    normalized: dict[str, object] = {}
    for name in MODEL_FEATURE_COLUMNS:
        value = record[name]
        definition = FEATURE_DEFINITIONS[name]
        if value is None:
            if not definition["nullable"]:
                raise InferenceContractError(f"feature {name} is not nullable")
            normalized[name] = np.nan
            continue

        input_type = definition["input_type"]
        if input_type == "string":
            if not isinstance(value, str) or not value:
                raise InferenceContractError(f"feature {name} must be a nonempty string")
            normalized[name] = value
        elif input_type == "timezone_aware_datetime":
            timestamp = pd.Timestamp(value)
            if timestamp.tzinfo is None:
                raise InferenceContractError(
                    f"feature {name} must include a UTC offset or Z suffix"
                )
            normalized[name] = timestamp.tz_convert("UTC")
        elif input_type == "integer":
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise InferenceContractError(f"feature {name} must be an integer")
            normalized[name] = int(value)
        elif input_type == "number":
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise InferenceContractError(f"feature {name} must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise InferenceContractError(f"feature {name} must be finite")
            normalized[name] = numeric
        else:
            raise InferenceContractError(f"unsupported input type for {name}: {input_type}")

    raw = pd.DataFrame([normalized], columns=MODEL_FEATURE_COLUMNS)
    return prepare_model_features(raw)


class CatBoostActionPredictor:
    """A process-local predictor that loads model artifacts exactly once."""

    def __init__(self, model_dir: Path):
        started_at = time.perf_counter()
        self.model_dir = model_dir
        required_paths = [
            model_dir / "model.cbm",
            model_dir / "feature_schema.json",
            model_dir / "class_labels.json",
            model_dir / "training_config.json",
        ]
        missing_paths = [str(path) for path in required_paths if not path.exists()]
        if missing_paths:
            raise InferenceContractError(
                "required artifacts not found: " + ", ".join(missing_paths)
            )
        self.schema = _read_json(model_dir / "feature_schema.json")
        validate_feature_schema(self.schema)

        labels_document = _read_json(model_dir / "class_labels.json")
        labels = labels_document.get("class_labels")
        if labels_document.get("source") != "model.classes_" or not isinstance(
            labels, list
        ):
            raise InferenceContractError(
                "class_labels.json must contain model.classes_ order"
            )
        self.class_labels = [str(label) for label in labels]

        model_path = model_dir / "model.cbm"
        if not model_path.exists():
            raise InferenceContractError(f"required artifact not found: {model_path}")
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)
        model_classes = [str(label) for label in self.model.classes_]
        if model_classes != self.class_labels:
            raise InferenceContractError(
                "class_labels.json order differs from loaded model.classes_"
            )
        if list(self.model.feature_names_) != MODEL_FEATURE_COLUMNS:
            raise InferenceContractError(
                "loaded model feature names differ from feature_schema.json"
            )

        training_config = _read_json(model_dir / "training_config.json")
        frozen = training_config.get("frozen_selection")
        if not isinstance(frozen, dict):
            raise InferenceContractError(
                "training_config.json has no frozen_selection"
            )
        frozen_classes = [str(label) for label in frozen.get("class_labels", [])]
        if frozen_classes != self.class_labels:
            raise InferenceContractError(
                "training_config class order differs from model.classes_"
            )
        self.threshold = float(frozen["selected_threshold"])
        self.fallback_label = str(frozen["majority_baseline_label"])
        if self.fallback_label not in self.class_labels:
            raise InferenceContractError("fallback label is absent from model classes")
        self.loaded_at_utc = pd.Timestamp.now(tz="UTC").isoformat()
        self.load_seconds = time.perf_counter() - started_at

    def predict_probabilities(self, record: Mapping[str, object]) -> np.ndarray:
        features = validate_feature_record(record)
        probabilities = np.asarray(self.model.predict_proba(features), dtype="float64")
        if probabilities.shape != (1, len(self.class_labels)):
            raise InferenceContractError(
                f"unexpected probability shape: {probabilities.shape}"
            )
        return probabilities[0]

    def predict(self, record: Mapping[str, object]) -> dict[str, object]:
        probabilities = self.predict_probabilities(record)
        best_index = int(probabilities.argmax())
        confidence = float(probabilities[best_index])
        argmax_action = self.class_labels[best_index]
        policy_action = (
            argmax_action if confidence >= self.threshold else self.fallback_label
        )
        return {
            "policy_action": policy_action,
            "argmax_action": argmax_action,
            "confidence": confidence,
            "threshold": self.threshold,
            "fallback_action": self.fallback_label,
            "class_order": self.class_labels,
            "probabilities": {
                label: float(probabilities[index])
                for index, label in enumerate(self.class_labels)
            },
            "model_loaded_at_utc": self.loaded_at_utc,
        }
