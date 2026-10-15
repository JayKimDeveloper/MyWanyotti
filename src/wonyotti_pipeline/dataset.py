"""Decision-time rows, observed behavior labels, and eligibility policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .executions import ExecutionCoverage, coverage_status


MODEL_FEATURE_COLUMNS = [
    "symbol",
    "decision_time_utc",
    "position_before_qty",
    "market_trade_count",
    "market_volume",
    "market_buy_volume",
    "market_sell_volume",
    "market_signed_volume",
    "market_vwap",
    "market_last_price",
    "market_return_lookback",
    "bid_price",
    "ask_price",
    "bid_size",
    "ask_size",
    "mid_price",
    "spread",
    "spread_bps",
    "top_book_imbalance",
    "quote_age_ms",
]


def decision_grid(
    symbols: Iterable[str],
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    interval: pd.Timedelta,
) -> pd.DataFrame:
    if interval <= pd.Timedelta(0):
        raise ValueError("interval must be positive")
    last_start = end_utc - interval
    if last_start < start_utc:
        times = pd.DatetimeIndex([], tz="UTC", dtype="datetime64[ns, UTC]")
    else:
        times = pd.date_range(start_utc, last_start, freq=interval).as_unit("ns")
    return pd.DataFrame(
        [(symbol, timestamp) for symbol in symbols for timestamp in times],
        columns=["symbol", "decision_time_utc"],
    )


def _atomic_actions(position: int, signed_qty: int) -> tuple[list[str], int]:
    if signed_qty > 0:
        if position < 0:
            if signed_qty <= -position:
                return ["SHORT_DECREASE"], position + signed_qty
            return ["SHORT_DECREASE", "LONG_INCREASE"], position + signed_qty
        return ["LONG_INCREASE"], position + signed_qty
    if signed_qty < 0:
        sell_qty = -signed_qty
        if position > 0:
            if sell_qty <= position:
                return ["LONG_DECREASE"], position + signed_qty
            return ["LONG_DECREASE", "SHORT_INCREASE"], position + signed_qty
        return ["SHORT_INCREASE"], position + signed_qty
    return [], position


def build_target_labels(
    decisions: pd.DataFrame,
    executions: pd.DataFrame,
    interval: pd.Timedelta,
    coverage: Iterable[ExecutionCoverage],
    invalid_order_ids: set[str] | None = None,
) -> pd.DataFrame:
    invalid_order_ids = invalid_order_ids or set()
    output_rows: list[dict[str, object]] = []
    for symbol, symbol_decisions in decisions.groupby("symbol", sort=False):
        trades = executions[
            executions["symbol"].eq(symbol) & executions["exectype"].eq("Trade")
        ].sort_values(
            ["event_time_utc", "processed_time_utc", "orderid", "cumqty", "execid"],
            kind="stable",
        )
        trade_times = trades["event_time_utc"].astype("int64").to_numpy()
        for decision in symbol_decisions.itertuples(index=False):
            start = decision.decision_time_utc
            end = start + interval
            left = np.searchsorted(trade_times, int(start.value), side="left")
            right = np.searchsorted(trade_times, int(end.value), side="left")
            target = trades.iloc[left:right]
            target_coverage = coverage_status(start, end, coverage)
            position_value = decision.position_before_qty
            position_known = pd.notna(position_value)
            components: list[str] = []
            position_after = int(position_value) if position_known else None

            if target.empty:
                label = "NO_FILL" if target_coverage != "UNVERIFIED" else "MISSING"
            elif not position_known:
                label = "POSITION_UNKNOWN"
            else:
                for fill in target.itertuples(index=False):
                    actions, position_after = _atomic_actions(
                        int(position_after), int(fill.trade_signed_qty)
                    )
                    components.extend(actions)
                unique_components = sorted(set(components))
                label = unique_components[0] if len(unique_components) == 1 else "MIXED"

            same_time_sides = (
                target.groupby("event_time_utc")["side"].nunique()
                if not target.empty
                else pd.Series(dtype="int64")
            )
            sequence_ambiguous = bool(same_time_sides.gt(1).any())
            order_counts = target["orderid"].value_counts() if not target.empty else pd.Series()
            partial_orders = set(order_counts[order_counts.gt(1)].index)
            if not target.empty:
                partial_orders.update(
                    target.loc[target["ordstatus"].eq("PartiallyFilled"), "orderid"]
                    .dropna()
                    .tolist()
                )

            output_rows.append(
                {
                    "symbol": symbol,
                    "decision_time_utc": start,
                    "target_end_time_utc": end,
                    "target_label": label,
                    "target_action_components": "|".join(sorted(set(components))),
                    "target_fill_count": int(len(target)),
                    "target_order_count": int(target["orderid"].nunique()),
                    "target_order_ids": "|".join(
                        sorted(set(target["orderid"].dropna().astype(str)))
                    ),
                    "target_partial_fill_order_count": len(partial_orders),
                    "target_order_integrity_failed": bool(
                        target["orderid"].isin(invalid_order_ids).any()
                    ),
                    "target_buy_qty": int(
                        target.loc[target["side"].eq("Buy"), "lastqty"].sum()
                    ),
                    "target_sell_qty": int(
                        target.loc[target["side"].eq("Sell"), "lastqty"].sum()
                    ),
                    "target_net_qty": int(target["trade_signed_qty"].sum()),
                    "target_position_after_qty": position_after,
                    "target_sequence_ambiguous": sequence_ambiguous,
                    "target_coverage_status": target_coverage,
                    "target_first_event_time_utc": (
                        target["event_time_utc"].min() if not target.empty else pd.NaT
                    ),
                    "target_last_event_time_utc": (
                        target["event_time_utc"].max() if not target.empty else pd.NaT
                    ),
                    "target_exec_ids": "|".join(target["execid"].head(50).tolist()),
                    "target_exec_ids_truncated": bool(len(target) > 50),
                }
            )
    result = pd.DataFrame(output_rows)
    for column in ("target_first_event_time_utc", "target_last_event_time_utc"):
        result[column] = pd.to_datetime(result[column], utc=True)
    return result


def apply_eligibility_policy(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = dataset.copy()
    reasons: list[str] = []
    eligible: list[bool] = []
    for row in rows.itertuples(index=False):
        current: list[str] = []
        if row.target_coverage_status == "UNVERIFIED":
            current.append("EXECUTION_COVERAGE_UNVERIFIED")
        if row.position_status != "KNOWN":
            current.append("POSITION_UNKNOWN")
        if row.market_feature_status != "OK":
            current.append(row.market_feature_status)
        if row.target_label == "MIXED":
            current.append("MIXED_ACTION")
        if row.target_label == "MISSING":
            current.append("TARGET_MISSING")
        if row.target_sequence_ambiguous:
            current.append("TARGET_SEQUENCE_AMBIGUOUS")
        if row.target_order_integrity_failed:
            current.append("ORDER_INTEGRITY_FAILED")
        reasons.append("|".join(current))
        eligible.append(not current)
    rows["exclusion_reasons"] = reasons
    rows["training_eligible"] = eligible
    return rows


def load_instrument_snapshot(path: Path) -> pd.DataFrame:
    records = json.loads(path.read_text(encoding="utf-8"))
    metadata = pd.DataFrame(records)
    selected = metadata[
        [
            "symbol",
            "listing",
            "expiry",
            "positionCurrency",
            "quoteCurrency",
            "settlCurrency",
            "lotSize",
            "tickSize",
            "multiplier",
            "isInverse",
            "isQuanto",
        ]
    ].copy()
    selected = selected.rename(
        columns={
            "listing": "metadata_listing_snapshot",
            "expiry": "metadata_expiry_snapshot",
            "positionCurrency": "metadata_position_currency_snapshot",
            "quoteCurrency": "metadata_quote_currency_snapshot",
            "settlCurrency": "metadata_settlement_currency_snapshot",
            "lotSize": "metadata_lot_size_snapshot",
            "tickSize": "metadata_tick_size_snapshot",
            "multiplier": "metadata_multiplier_snapshot",
            "isInverse": "metadata_is_inverse_snapshot",
            "isQuanto": "metadata_is_quanto_snapshot",
        }
    )
    selected["instrument_metadata_status"] = "CURRENT_SNAPSHOT_NOT_POINT_IN_TIME"
    return selected


def leakage_report(dataset: pd.DataFrame) -> dict[str, int | list[str]]:
    feature_violations = dataset["market_feature_max_time_utc"].notna() & dataset[
        "market_feature_max_time_utc"
    ].ge(dataset["decision_time_utc"])
    position_violations = dataset["position_asof_time_utc"].notna() & dataset[
        "position_asof_time_utc"
    ].ge(dataset["decision_time_utc"])
    first_target_violations = dataset["target_first_event_time_utc"].notna() & dataset[
        "target_first_event_time_utc"
    ].lt(dataset["decision_time_utc"])
    last_target_violations = dataset["target_last_event_time_utc"].notna() & dataset[
        "target_last_event_time_utc"
    ].ge(dataset["target_end_time_utc"])
    duplicate_keys = dataset.duplicated(["symbol", "decision_time_utc"])
    return {
        "market_feature_at_or_after_decision_count": int(feature_violations.sum()),
        "position_event_at_or_after_decision_count": int(position_violations.sum()),
        "target_before_decision_count": int(first_target_violations.sum()),
        "target_at_or_after_end_count": int(last_target_violations.sum()),
        "duplicate_decision_key_count": int(duplicate_keys.sum()),
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
    }
