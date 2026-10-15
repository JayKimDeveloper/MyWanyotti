"""Unweighted binary recorded-activity research contract and evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .side_research import ASSUMPTIONS, FEATURE_COLUMNS, RUNTIME_CONTRACT as SIDE_CONTRACT
from .side_research import CLASSES as SIDE_CLASSES, feature_schema, prepare_model_features
from .training import classification_metrics

VERSION = "candle_activity_research_v3_unweighted"
CLASSES = ["NO_RECORDED_FILL", "ACTIVE_RECORDED"]
CONTRACT = dict(SIDE_CONTRACT, data_contract_version=VERSION, labels=CLASSES,
                activity_score="P(ACTIVE_RECORDED); not established as calibrated",
                threshold_rule="score >= frozen validation-F1 threshold",
                target_mapping={label: "NO_RECORDED_FILL" if label == "NO_RECORDED_FILL"
                                else "ACTIVE_RECORDED" for label in SIDE_CLASSES})


def binary_labels(labels: pd.Series) -> pd.Series:
    if not labels.isin(SIDE_CLASSES).all():
        raise ValueError("invalid or missing observed-side label")
    return labels.map(CONTRACT["target_mapping"])


def activity_curve(actual, scores) -> tuple[pd.DataFrame, float | None]:
    truth = np.asarray(actual, dtype=bool)
    values = np.asarray(scores, dtype=float)
    if len(truth) != len(values) or not len(truth) or not np.isfinite(values).all():
        raise ValueError("invalid activity score arrays")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("activity scores must be in [0, 1]")
    groups = pd.DataFrame({"score": values, "positive": truth.astype(int)}).groupby("score").positive.agg(["sum", "count"]).sort_index(ascending=False)
    groups["tp"] = groups["sum"].cumsum()
    groups["flagged"] = groups["count"].cumsum()
    groups["precision"] = groups.tp / groups.flagged
    groups["recall"] = groups.tp / truth.sum() if truth.sum() else 0.0
    groups["f1"] = 2 * groups.tp / (groups.flagged + truth.sum())
    ap = float((groups.precision * groups["sum"] / truth.sum()).sum()) if truth.sum() else None
    return groups, ap


def choose_activity_threshold(actual, scores) -> float:
    if not np.asarray(actual, dtype=bool).any():
        raise ValueError("validation has no observed activity")
    curve, _ = activity_curve(actual, scores)
    # Descending scores break maximum-F1 ties in favor of the greatest threshold.
    return float(curve.f1.idxmax())


def activity_metrics(actual, scores, threshold: float) -> dict:
    truth = np.asarray(actual, dtype=bool)
    values = np.asarray(scores, dtype=float)
    _, ap = activity_curve(truth, values)
    flagged = values >= threshold
    result = classification_metrics(np.where(truth, CLASSES[1], CLASSES[0]),
                                    np.where(flagged, CLASSES[1], CLASSES[0]), CLASSES)
    active = result["per_class"][CLASSES[1]]
    result.update(average_precision=ap, threshold=threshold,
                  precision=active["precision"], recall=active["recall"], binary_f1=active["f1"],
                  flagged_count=int(flagged.sum()), false_positives=int((flagged & ~truth).sum()),
                  active_count=int(truth.sum()), activity_rate=float(truth.mean()))
    return result


def load_activity_model(path: Path) -> tuple[CatBoostClassifier, list[str], float]:
    if json.loads((path / "runtime_contract.json").read_text()) != CONTRACT:
        raise ValueError("binary research runtime contract mismatch")
    if json.loads((path / "feature_schema.json").read_text()) != feature_schema(VERSION):
        raise ValueError("binary research feature schema mismatch")
    labels = json.loads((path / "class_labels.json").read_text())["class_labels"]
    threshold = json.loads((path / "activity_threshold.json").read_text())["threshold"]
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("invalid frozen activity threshold")
    model = CatBoostClassifier()
    model.load_model(path / "model.cbm")
    if list(model.classes_) != labels or set(labels) != set(CLASSES) or list(model.feature_names_) != FEATURE_COLUMNS:
        raise ValueError("binary research model class/feature mismatch")
    return model, labels, threshold
