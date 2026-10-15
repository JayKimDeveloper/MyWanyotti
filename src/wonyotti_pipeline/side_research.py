"""Position-free observed-side contract; explicitly unverified research assumptions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .candles import FEATURE_COLUMNS as CANDLE_FEATURES, FEATURE_UNITS, HOUR
from .candles import build_market_candle_features, prepare_model_features as prepare_candle_features
from .executions import ZERO_UUID
from .training import chronological_split, split_leakage_checks

VERSION = "candle_side_research_v1"
FEATURE_COLUMNS = CANDLE_FEATURES[:8]
CLASSES = ["BUY_ONLY", "SELL_ONLY", "MIXED", "NO_RECORDED_FILL"]
START = pd.Timestamp("2018-03-07T00:00:00Z")
END = START + pd.Timedelta(days=90)
TRAIN_END = START + pd.Timedelta(days=54)
VALIDATION_END = START + pd.Timedelta(days=72)
ASSUMPTIONS = {
    "status": "UNVERIFIED_USER_AUTHORIZED_RESEARCH_ASSUMPTIONS",
    "execution_timezone": "Interpret naive execution clocks as UTC for this experiment only",
    "account_records": "Treat supplied records as complete only within the selected experiment interval for observed activity labels",
    "negative_label": "NO_RECORDED_FILL is absence of a recorded fill, not verified inactivity",
    "position_state": "No position reconstruction or funding-based signed snapshots",
    "production_evidence_modified": False,
}
RUNTIME_CONTRACT = {
    "data_contract_version": VERSION, "research_only": True,
    "compatible_with_five_action_predictor": False, "actual_order_api_enabled": False,
    "symbols": ["XBTUSD"], "candle_interval": "1h", "decision_interval": "1h",
    "completed_candle_rule": "candle_end_utc < decision_time_utc",
    "latest_candle_end": "decision_time_utc - 1h",
    "feature_dependency": "[decision_time_utc - 26h, decision_time_utc - 1h]",
    "label_window": "[decision_time_utc, decision_time_utc + 1h)",
    "labels": CLASSES, "position_features": False,
    "price_unit": "USD_PER_XBT", "volume_unit": "CONTRACTS", "assumptions": ASSUMPTIONS,
}


def feature_schema(version: str = VERSION) -> dict:
    return {"data_contract_version": version, "research_only": True, "columns": FEATURE_COLUMNS,
            "features": {name: {"input_type": "number", "model_type": "float64",
                                "nullable": False, "unit": FEATURE_UNITS[name]} for name in FEATURE_COLUMNS}}


def prepare_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    return prepare_candle_features(frame, FEATURE_COLUMNS)


def build_dataset(decisions: pd.DataFrame, candles: pd.DataFrame, executions: pd.DataFrame) -> pd.DataFrame:
    if not decisions.symbol.eq("XBTUSD").all():
        raise ValueError("side research supports XBTUSD only")
    rows = build_market_candle_features(decisions, candles)
    trades = executions[executions.symbol.eq("XBTUSD") & executions.exectype.eq("Trade")].copy()
    trades = trades.sort_values("event_time_utc", kind="stable")
    times = trades.event_time_utc.dt.as_unit("ns").astype("int64").to_numpy()
    labels = []
    for row in rows.itertuples(index=False):
        end = row.decision_time_utc + HOUR
        left = np.searchsorted(times, row.decision_time_utc.value, side="left")
        right = np.searchsorted(times, end.value, side="left")
        target = trades.iloc[left:right]
        invalid = (~target.side.isin(["Buy", "Sell"]) | target.lastqty.isna() | target.lastpx.isna()
                   | target.lastqty.le(0) | target.lastpx.le(0)).any()
        sides = set(target.side.dropna())
        label = ("NO_RECORDED_FILL" if not sides else "MIXED" if len(sides) == 2
                 else "BUY_ONLY" if sides == {"Buy"} else "SELL_ONLY")
        reasons = []
        if row.market_feature_status != "OK":
            reasons.append(row.market_feature_status)
        if invalid:
            reasons.append("INVALID_RECORDED_FILL")
        ids = set(target.orderid.dropna()) - {ZERO_UUID, ""}
        labels.append({"target_label": label, "target_end_time_utc": end,
                       "target_fill_count": len(target), "target_order_ids": "|".join(sorted(ids)),
                       "target_unidentified_order_fill_count": int((target.orderid.isna() | target.orderid.eq(ZERO_UUID)).sum()),
                       "target_first_event_time_utc": target.event_time_utc.min(),
                       "target_last_event_time_utc": target.event_time_utc.max(),
                       "exclusion_reasons": "|".join(reasons), "research_eligible": not reasons})
    rows = pd.concat([rows.reset_index(drop=True), pd.DataFrame(labels)], axis=1)
    rows["feature_start_time_utc"] = rows.decision_time_utc - 26 * HOUR
    rows["dependency_end_time_utc"] = rows.target_end_time_utc
    return rows


def research_splits(frame: pd.DataFrame, train_end: pd.Timestamp, validation_end: pd.Timestamp) -> tuple[dict, dict]:
    splits, report = chronological_split(frame, train_end, validation_end)
    purged = {}
    for name, boundary in (("validation", train_end), ("test", validation_end)):
        original = len(splits[name])
        splits[name] = splits[name][splits[name].feature_start_time_utc.ge(boundary)].reset_index(drop=True)
        purged[name] = original - len(splits[name])
    checks = split_leakage_checks(splits, train_end, validation_end)
    overlaps = 0
    for earlier, later in (("train", "validation"), ("validation", "test"), ("train", "test")):
        if not splits[earlier].empty and not splits[later].empty:
            overlaps += int(splits[earlier].dependency_end_time_utc.max() > splits[later].feature_start_time_utc.min())
    checks["dependency_overlap_pairs"] = overlaps
    checks["passed"] = checks["passed"] and overlaps == 0
    report.update(feature_dependency_purged_counts=purged, leakage_checks=checks,
                  retained_counts={name: len(value) for name, value in splits.items()})
    return splits, report


def load_research_model(path: Path, version: str = VERSION) -> tuple[CatBoostClassifier, list[str]]:
    contract = dict(RUNTIME_CONTRACT, data_contract_version=version)
    if json.loads((path / "runtime_contract.json").read_text()) != contract:
        raise ValueError("research runtime contract mismatch")
    if json.loads((path / "feature_schema.json").read_text()) != feature_schema(version):
        raise ValueError("research feature schema mismatch")
    labels = json.loads((path / "class_labels.json").read_text())["class_labels"]
    model = CatBoostClassifier()
    model.load_model(path / "model.cbm")
    if list(model.classes_) != labels or list(model.feature_names_) != FEATURE_COLUMNS:
        raise ValueError("research model class/feature order mismatch")
    return model, labels
