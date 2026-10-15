"""Separate candle_v1 contract, causal account state, and hourly features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .executions import (
    EXECUTION_COLUMNS, audit_orders, execution_paths, funding_anchor_candidates,
    load_executions, positions_before_decisions, validate_funding_anchors, ZERO_UUID,
)
from .training import file_sha256, write_json
from .account_evidence import AccountEvidence, VALIDITY_POLICY, load_account_evidence, utc
from .dataset import build_target_labels, apply_eligibility_policy


CONTRACT_VERSION = "candle_v1"
HOUR = pd.Timedelta(hours=1)
FEATURE_UNITS = {
    "return_1h": "RATIO", "return_3h": "RATIO", "return_6h": "RATIO",
    "return_24h": "RATIO", "volatility_6h": "RATIO",
    "volatility_24h": "RATIO", "relative_volume_24h": "RATIO",
    "candle_range_ratio": "RATIO", "position_before_qty": "CONTRACTS",
    "position_direction": "SIGN", "holding_hours": "HOURS",
    "unrealized_return": "RATIO",
}
FEATURE_COLUMNS = list(FEATURE_UNITS)
NULLABLE_FEATURES = {"holding_hours", "unrealized_return"}
RUNTIME_CONTRACT = {
    "data_contract_version": CONTRACT_VERSION, "exchange": "BitMEX",
    "symbols": ["XBTUSD"], "candle_interval": "1h", "decision_interval": "1h",
    "label_window": "[decision_time_utc, decision_time_utc + 1h)",
    "lookback": "25h", "candle_delay": "1h",
    "strict_asof_rule": "candle_end_utc < decision_time_utc; account_event_time_utc < decision_time_utc",
    "price_unit": "USD_PER_XBT", "volume_unit": "CONTRACTS",
    "position_unit": "CONTRACTS", "source": "https://www.bitmex.com/api/v1/trade/bucketed",
    "position_anchor_policy": "verified snapshot or two preceding exact covered funding intervals; gaps invalidate state",
    "account_validity_policy": VALIDITY_POLICY,
    "no_fill_semantics": "no recorded Trade within independently evidenced account coverage; UTC provenance separately required",
    "actual_order_api_enabled": False,
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def feature_schema() -> dict[str, object]:
    return {
        "data_contract_version": CONTRACT_VERSION,
        "columns": FEATURE_COLUMNS,
        "features": {
            name: {"input_type": "number", "model_type": "float64",
                   "unit": unit, "nullable": name in NULLABLE_FEATURES}
            for name, unit in FEATURE_UNITS.items()
        },
    }


def prepare_model_features(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    columns = FEATURE_COLUMNS if columns is None else columns
    if not set(columns).issubset(FEATURE_COLUMNS):
        raise ValueError("unknown candle feature column")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"candle_v1 missing features: {sorted(missing)}")
    selected = frame[columns].apply(pd.to_numeric, errors="raise").astype("float64")
    if np.isinf(selected.to_numpy()).any():
        raise ValueError("candle_v1 infinite feature")
    if selected[[name for name in columns if name not in NULLABLE_FEATURES]].isna().any().any():
        raise ValueError("candle_v1 nonnullable feature is missing")
    return selected


def feature_record(row: pd.Series) -> dict[str, float | None]:
    return {name: None if pd.isna(row[name]) else float(row[name]) for name in FEATURE_COLUMNS}


def load_candle_model(model_dir: Path) -> tuple[CatBoostClassifier, list[str]]:
    schema = json.loads((model_dir / "feature_schema.json").read_text())
    if schema != feature_schema():
        raise ValueError("candle_v1 feature schema mismatch")
    contract = json.loads((model_dir / "runtime_contract.json").read_text())
    if contract != RUNTIME_CONTRACT:
        raise ValueError("candle_v1 runtime contract mismatch")
    labels = json.loads((model_dir / "class_labels.json").read_text())["class_labels"]
    model = CatBoostClassifier()
    model.load_model(model_dir / "model.cbm")
    if list(model.feature_names_) != FEATURE_COLUMNS or list(model.classes_) != labels:
        raise ValueError("candle_v1 model feature/class order mismatch")
    return model, labels


def validate_candles(records: list[dict], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Validate a complete hourly page, whose API timestamps are candle ends."""
    frame = pd.DataFrame(records)
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume", "foreignNotional"}
    if not required.issubset(frame):
        raise ValueError(f"missing candle fields for {start}: {sorted(required - set(frame))}")
    raw_times = frame["timestamp"].map(pd.Timestamp)
    if any(t.tzinfo is None for t in raw_times):
        raise ValueError("candle timestamps must include a UTC offset")
    frame["candle_end_utc"] = pd.to_datetime(frame["timestamp"], utc=True).dt.as_unit("ns")
    frame = frame.sort_values("candle_end_utc").reset_index(drop=True)
    expected = pd.date_range(start + HOUR, end, freq=HOUR).as_unit("ns")
    if frame["candle_end_utc"].tolist() != expected.tolist():
        missing = expected.difference(pd.DatetimeIndex(frame["candle_end_utc"]))
        raise ValueError(f"candle coverage mismatch {start} to {end}; missing {missing[:5].tolist()}")
    if not frame["symbol"].eq("XBTUSD").all():
        raise ValueError("non-XBTUSD candle")
    for name in ("open", "high", "low", "close", "volume", "foreignNotional"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if not np.isfinite(frame[name]).all():
            raise ValueError(f"invalid candle {name}")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any() or (frame["volume"] < 0).any():
        raise ValueError("nonpositive price or negative volume")
    # BitMEX open is the previous close and need not lie within this bucket's trade range.
    if (frame["high"] < frame["low"]).any() or (~frame["close"].between(frame["low"], frame["high"])).any():
        raise ValueError("invalid candle high/low/close")
    if not np.allclose(frame["volume"], frame["foreignNotional"], rtol=0, atol=1e-6):
        raise ValueError("XBTUSD volume is not equal to quote notional contracts")
    frame["candle_start_utc"] = frame["candle_end_utc"] - HOUR
    return frame[["symbol", "candle_start_utc", "candle_end_utc", "open", "high", "low", "close", "volume"]]


def deduplicate_executions(executions: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if executions["execid"].isna().any() or executions["execid"].eq("").any():
        raise ValueError("execution ID missing; cannot safely deduplicate")
    duplicate = executions.duplicated("execid", keep=False)
    repeated = executions.loc[duplicate, EXECUTION_COLUMNS]
    if not repeated.empty and repeated.groupby("execid", dropna=False).nunique(dropna=False).gt(1).any().any():
        raise ValueError("conflicting records share an execution ID")
    unique = executions.drop_duplicates("execid").reset_index(drop=True)
    return unique, len(executions) - len(unique)


def causal_position_anchors(executions: pd.DataFrame, evidence: AccountEvidence | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Qualify only the current anchor, never backdate later validation to earlier anchors."""
    evidence = evidence or AccountEvidence()
    candidates = funding_anchor_candidates(executions)
    detail, _ = validate_funding_anchors(executions, candidates)
    accepted = set()
    availability = {}
    if not detail.empty:
        detail["coverage_verified"] = False
        detail["causal_prefix_intervals"] = 0
        for symbol, intervals in detail.groupby("symbol", sort=False):
            count = 0
            prefix_start = None
            symbol_events = executions[executions.symbol.eq(symbol)]
            event_times = symbol_events.event_time_utc.dt.as_unit("ns").astype("int64").to_numpy()
            processed_times = symbol_events.processed_time_utc.dt.as_unit("ns").astype("int64").to_numpy()
            for index, interval in intervals.sort_values("anchor_time_utc").iterrows():
                start, end = interval.previous_anchor_time_utc, interval.anchor_time_utc
                covered = evidence.continuous(symbol, start, end + pd.Timedelta(1, unit="ns"))
                if count == 0:
                    prefix_start = start
                count = count + 1 if covered and interval.position_error == 0 else 0
                detail.loc[index, ["coverage_verified", "causal_prefix_intervals"]] = [covered, count]
                if count >= 2:
                    # Qualification needs only this prefix; later rows never qualify its first anchor.
                    left = np.searchsorted(event_times, prefix_start.value, side="left")
                    right = np.searchsorted(event_times, end.value, side="right")
                    available = max(end, pd.Timestamp(int(processed_times[left:right].max()), tz="UTC"))
                    accepted.add((symbol, end))
                    availability[(symbol, end)] = available
    funding = candidates.loc[pd.Series([(r.symbol, r.event_time_utc) in accepted
                          for r in candidates.itertuples(index=False)], index=candidates.index, dtype=bool)].copy()
    funding["anchor_source"] = "FUNDING_CAUSAL_COVERED_PREFIX"
    funding["available_at_utc"] = [availability[(r.symbol, r.event_time_utc)]
                                   for r in funding.itertuples(index=False)]
    funding["position_evidence_id"] = funding["execid"]
    snapshots = []
    for symbol in executions.symbol.unique():
        for record in evidence.scoped("positions", symbol):
            at = utc(record["event_time_utc"])
            if evidence.continuous(symbol, at, at):
                snapshots.append({"symbol": symbol, "event_time_utc": at,
                                  "execid": record["id"], "position_qty": record["position_qty"],
                                  "anchor_source": "VERIFIED_POSITION_SNAPSHOT",
                                  "available_at_utc": utc(record["available_at_utc"]),
                                  "position_evidence_id": record["id"]})
    anchors = pd.concat([funding, pd.DataFrame(snapshots)], ignore_index=True)
    # A same-time fill/snapshot has no verified before/after ordering.
    trade_keys = set(zip(executions.loc[executions.exectype.eq("Trade"), "symbol"],
                         executions.loc[executions.exectype.eq("Trade"), "event_time_utc"]))
    anchors = anchors.loc[pd.Series([(r.symbol, r.event_time_utc) not in trade_keys
                        for r in anchors.itertuples(index=False)], index=anchors.index, dtype=bool)].copy()
    if not anchors.empty and anchors.groupby(["symbol", "event_time_utc"]).position_qty.nunique().gt(1).any():
        raise ValueError("conflicting position anchors at the same timestamp")
    anchors["available_at_utc"] = pd.to_datetime(anchors["available_at_utc"], utc=True)
    return anchors.sort_values("event_time_utc"), detail


def build_validated_labels(decisions: pd.DataFrame, executions: pd.DataFrame,
                           evidence: AccountEvidence, invalid_orders: set[str]) -> pd.DataFrame:
    """Keep recorded fills while withholding labels unsupported by account evidence."""
    # A sentinel identifies no actual order; never join unrelated fills for split purging.
    if not decisions.symbol.eq("XBTUSD").all():
        raise ValueError("candle_v1 labels support XBTUSD only")
    fills = executions.copy()
    for column in ("event_time_utc", "processed_time_utc"):
        fills[column] = pd.to_datetime(fills[column], utc=True).dt.as_unit("ns")
    unidentified = fills.orderid.isna() | fills.orderid.eq(ZERO_UUID)
    fills.loc[unidentified, "orderid"] = pd.NA
    labels = build_target_labels(decisions, fills, HOUR, evidence.intervals("coverage", "XBTUSD"), invalid_orders)
    rows = decisions.merge(labels, on=["symbol", "decision_time_utc"], validate="one_to_one")
    timestamp_status, coverage_ids, timestamp_ids = [], [], []
    for r in rows.itertuples(index=False):
        coverage_ids.append("|".join(evidence.supporting_ids("coverage", r.symbol, r.decision_time_utc, r.target_end_time_utc)))
        ids = evidence.supporting_ids("timestamps", r.symbol, r.decision_time_utc, r.target_end_time_utc)
        timestamp_ids.append("|".join(ids))
        timestamp_status.append("VERIFIED" if ids else "UNKNOWN")
    rows["timestamp_provenance_status"] = timestamp_status
    rows["coverage_evidence_ids"] = coverage_ids
    rows["timestamp_evidence_ids"] = timestamp_ids
    rows = apply_eligibility_policy(rows)
    unknown = rows.timestamp_provenance_status.ne("VERIFIED")
    rows.loc[unknown, "exclusion_reasons"] = rows.loc[unknown, "exclusion_reasons"].map(
        lambda value: "|".join(filter(None, [value, "TIMESTAMP_PROVENANCE_UNKNOWN"])))
    rows.loc[unknown, "training_eligible"] = False
    counts = fills[fills.exectype.eq("Trade") & unidentified].set_index("event_time_utc")
    times = counts.index.astype("int64").to_numpy()
    rows["target_unidentified_order_fill_count"] = [
        int(np.searchsorted(times, r.target_end_time_utc.value) - np.searchsorted(times, r.decision_time_utc.value))
        for r in rows.itertuples(index=False)]
    rows["order_validation_status"] = np.where(rows.target_unidentified_order_fill_count.gt(0),
                                                 "UNIDENTIFIED_ORDER_FILLS_RETAINED", "IDENTIFIED_ORDERS_ONLY")
    rows["validity_policy"] = VALIDITY_POLICY
    return rows


def reconstruct_account(raw_dir: Path, decisions: pd.DataFrame, audit_dir: Path, evidence: AccountEvidence | None = None) -> tuple[pd.DataFrame, pd.DataFrame, set[str], dict]:
    evidence = evidence or AccountEvidence()
    paths = execution_paths(raw_dir)
    executions, removed = deduplicate_executions(load_executions(paths, symbols={"XBTUSD"}))
    orders = audit_orders(executions)
    checks = ["starts_at_first_fill", "cumqty_monotonic", "increments_match_lastqty",
              "sum_matches_max_cumqty", "symbol_consistent", "side_consistent"]
    invalid = set(orders.loc[~orders[checks].all(axis=1), "order_id"]) if not orders.empty else set()
    anchors, detail = causal_position_anchors(executions, evidence)
    positions = positions_before_decisions(decisions, executions, anchors)
    events = executions[executions["exectype"].eq("Trade")].copy()
    events["anchor_qty"] = np.nan
    resets = anchors.copy()
    resets["exectype"] = "Anchor"
    resets["anchor_qty"] = resets["position_qty"]
    resets["processed_time_utc"] = resets["available_at_utc"]
    timeline = pd.concat([events, resets], ignore_index=True).sort_values(
        ["event_time_utc", "processed_time_utc", "orderid", "cumqty", "execid"], kind="stable")
    iterator = iter(timeline.itertuples(index=False))
    event = next(iterator, None)
    qty = None
    entry = None
    entry_asof = None
    opened_at = None
    state_time = None
    anchor_time = None
    anchor_available = None
    anchor_source = None
    anchor_id = None
    snapshots = []
    changes = []
    ambiguous_times = set(events.groupby("event_time_utc")["side"].nunique().loc[lambda s: s.gt(1)].index)
    for decision in positions.itertuples(index=False):
        while event is not None and event.event_time_utc < decision.decision_time_utc:
            if event.processed_time_utc >= decision.decision_time_utc:
                newer = anchors[(anchors.event_time_utc > event.event_time_utc)
                                & (anchors.event_time_utc < decision.decision_time_utc)
                                & (anchors.available_at_utc < decision.decision_time_utc)]
                if newer.empty:
                    break
                # A newer available snapshot can restore state despite an older unavailable event.
                qty, entry, opened_at, entry_asof = None, None, None, None
                event = next(iterator, None)
                continue
            if state_time is not None and not evidence.continuous("XBTUSD", state_time, event.event_time_utc):
                qty, entry, opened_at, entry_asof = None, None, None, None
            before = qty
            if event.exectype == "Anchor":
                new_qty = int(event.anchor_qty)
                if qty != new_qty:
                    entry, opened_at = None, None
                    entry_asof = None
                qty = new_qty
                anchor_time, anchor_available = event.event_time_utc, event.processed_time_utc
                anchor_source, anchor_id = event.anchor_source, event.position_evidence_id
            elif qty is not None:
                delta = int(event.trade_signed_qty)
                qty += delta
                price = float(event.lastpx)
                if qty == 0:
                    entry, opened_at = None, None
                    entry_asof = None
                elif before == 0 or np.sign(before) != np.sign(qty):
                    entry, opened_at = price, event.event_time_utc
                    entry_asof = event.event_time_utc
                elif np.sign(delta) == np.sign(before) and entry is not None:
                    entry = abs(qty) / (abs(before) / entry + abs(delta) / price)
                    entry_asof = event.event_time_utc
                if event.event_time_utc in ambiguous_times:
                    entry, opened_at = None, None
                    entry_asof = None
                if event.orderid in invalid:
                    qty, entry, opened_at = None, None, None
                    entry_asof = None
            state_time = event.event_time_utc
            if event.exectype == "Trade":
                changes.append({"event_time_utc": event.event_time_utc, "execid": event.execid,
                                "orderid": event.orderid, "signed_qty": int(event.trade_signed_qty),
                                "position_before_qty": before, "position_after_qty": qty,
                                "entry_price": entry, "holding_start_utc": opened_at})
            event = next(iterator, None)
        unavailable = event is not None and event.event_time_utc < decision.decision_time_utc
        if (unavailable or state_time is None or not evidence.continuous("XBTUSD", state_time, decision.decision_time_utc)):
            qty, entry, opened_at, entry_asof = None, None, None, None
        if qty is not None and pd.notna(decision.position_before_qty) and qty != decision.position_before_qty:
            raise ValueError("position reconstruction disagrees with shared anchor reconstruction")
        holding = 0.0 if qty == 0 else ((decision.decision_time_utc - opened_at).total_seconds() / 3600 if opened_at is not None else np.nan)
        snapshots.append({"position_before_qty": qty, "entry_price": entry,
                          "position_anchor_time_utc": anchor_time if qty is not None else pd.NaT,
                          "position_anchor_available_at_utc": anchor_available if qty is not None else pd.NaT,
                          "position_anchor_source": anchor_source if qty is not None else None,
                          "position_evidence_id": anchor_id if qty is not None else None,
                          "position_asof_time_utc": state_time if qty is not None else pd.NaT,
                          "entry_asof_time_utc": entry_asof,
                          "holding_hours": holding, "position_status": "KNOWN" if qty is not None else "UNKNOWN"})
    state = pd.DataFrame(snapshots)
    for column in state:
        positions[column] = state[column].to_numpy()
    positions["position_before_qty"] = pd.array(positions["position_before_qty"], dtype="Int64")
    for column in ("entry_asof_time_utc", "position_asof_time_utc", "position_anchor_time_utc", "position_anchor_available_at_utc"):
        positions[column] = pd.to_datetime(positions[column], utc=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    atomic_parquet(audit_dir / "position_changes.parquet", pd.DataFrame(changes))
    atomic_parquet(audit_dir / "position_anchors.parquet", anchors)
    atomic_parquet(audit_dir / "funding_anchor_validation.parquet", detail)
    atomic_parquet(audit_dir / "order_audit.parquet", orders)
    report = {"xbtusd_event_rows": len(executions), "duplicate_executions_removed": removed,
              "trade_rows": len(events),
              "zero_uuid_trade_fills_retained": int(events.orderid.eq(ZERO_UUID).sum()),
              "missing_orderid_trade_fills_retained": int(events.orderid.isna().sum()),
              "funding_candidates": int(executions.exectype.eq("Funding").sum()),
              "anchor_policy": "Only current covered prefix; no retrospective backdating; gaps reset validation",
              "order_count": len(orders), "invalid_orders": len(invalid),
              "multi_fill_orders": int(orders["fill_row_count"].gt(1).sum()) if not orders.empty else 0,
              "accepted_causal_anchors": len(anchors), "ambiguous_execution_timestamps": len(ambiguous_times),
              "unknown_decisions": int(positions["position_status"].eq("UNKNOWN").sum()),
              "source_files": {str(p): file_sha256(p) for p in paths}}
    atomic_json(audit_dir / "position_report.json", report)
    return executions, positions, invalid, report


def build_market_candle_features(decisions: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    indexed = candles.sort_values("candle_end_utc").set_index("candle_end_utc")
    if indexed.index.has_duplicates:
        raise ValueError("duplicate candle end")
    indexed = indexed.reindex(pd.date_range(indexed.index.min(), indexed.index.max(), freq=HOUR))
    close = indexed["close"]
    for hours in (1, 3, 6, 24):
        indexed[f"return_{hours}h"] = close / close.shift(hours) - 1
    returns = close / close.shift(1) - 1
    for hours in (6, 24):
        indexed[f"volatility_{hours}h"] = returns.rolling(hours, min_periods=hours).std(ddof=0)
    indexed["relative_volume_24h"] = indexed["volume"] / indexed["volume"].shift(1).rolling(24, min_periods=24).mean()
    indexed["candle_range_ratio"] = (indexed["high"] - indexed["low"]) / close
    indexed["candle_window_complete"] = indexed["close"].notna().rolling(25, min_periods=25).sum().eq(25)
    indexed.index.name = "candle_end_utc"
    rows = decisions.copy()
    rows["candle_end_utc"] = rows["decision_time_utc"] - HOUR
    rows = rows.merge(indexed.drop(columns="symbol", errors="ignore").reset_index(), on="candle_end_utc", how="left", validate="many_to_one")
    market_columns = FEATURE_COLUMNS[:8]
    good = rows["candle_window_complete"].eq(True) & np.isfinite(rows[market_columns]).all(axis=1)
    rows["market_feature_status"] = np.where(good, "OK", "CANDLE_WINDOW_INCOMPLETE")
    return rows


def build_candle_features(decisions: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    rows = build_market_candle_features(decisions, candles)
    rows["position_direction"] = np.sign(rows["position_before_qty"].astype("float64"))
    rows["unrealized_return"] = rows["position_direction"] * (1 - rows["entry_price"] / rows["close"])
    rows.loc[rows["entry_asof_time_utc"].isna() | rows["entry_asof_time_utc"].gt(rows["candle_end_utc"]), "unrealized_return"] = np.nan
    rows.loc[rows["position_before_qty"].eq(0), "unrealized_return"] = 0.0
    return rows
