"""Execution loading, order audits, anchors, and position reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXECUTION_COLUMNS = [
    "date",
    "execid",
    "orderid",
    "symbol",
    "side",
    "lastqty",
    "lastpx",
    "lastliquidityind",
    "orderqty",
    "leavesqty",
    "cumqty",
    "currency",
    "settlcurrency",
    "exectype",
    "execinst",
    "ordstatus",
    "tradepublishindicator",
    "trdmatchid",
    "homenotional",
    "foreignnotional",
    "transacttime",
    "timestamp",
]
ZERO_UUID = "00000000-0000-0000-0000-000000000000"


@dataclass(frozen=True)
class ExecutionCoverage:
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp
    source_file: str
    status: str = "COMPLETE_BY_EXPORT_RANGE"


def execution_paths(raw_dir: Path) -> list[Path]:
    paths = sorted(raw_dir.glob("aoa-execution-*.csv"))
    if not paths:
        raise FileNotFoundError(f"no execution CSV files found under {raw_dir}")
    return paths


def parse_utc(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.replace("D", "T", regex=False)
    return pd.to_datetime(
        normalized, utc=True, format="mixed", errors="raise"
    ).dt.as_unit("ns")


def load_executions(
    paths: Iterable[Path],
    symbols: set[str] | None = None,
    end_utc: pd.Timestamp | None = None,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        reader = pd.read_csv(
            path,
            encoding="utf-8-sig",
            usecols=EXECUTION_COLUMNS,
            chunksize=chunk_size,
            low_memory=False,
        )
        for chunk in reader:
            if symbols is not None:
                chunk = chunk[chunk["symbol"].isin(symbols)]
            if chunk.empty:
                continue
            chunk = chunk.copy()
            chunk["event_time_utc"] = parse_utc(chunk["transacttime"])
            chunk["processed_time_utc"] = parse_utc(chunk["timestamp"])
            if end_utc is not None:
                chunk = chunk[chunk["event_time_utc"] < end_utc]
            if not chunk.empty:
                frames.append(chunk)
    if not frames:
        return pd.DataFrame(columns=EXECUTION_COLUMNS)

    executions = pd.concat(frames, ignore_index=True)
    executions["event_kind"] = executions["exectype"].str.upper()
    executions["trade_signed_qty"] = np.select(
        [
            executions["exectype"].eq("Trade") & executions["side"].eq("Buy"),
            executions["exectype"].eq("Trade") & executions["side"].eq("Sell"),
        ],
        [executions["lastqty"], -executions["lastqty"]],
        default=0,
    ).astype("int64")
    return executions.sort_values(
        [
            "symbol",
            "event_time_utc",
            "processed_time_utc",
            "orderid",
            "cumqty",
            "execid",
        ],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)


def coverage_from_filenames(paths: Iterable[Path]) -> list[ExecutionCoverage]:
    coverage: list[ExecutionCoverage] = []
    for path in paths:
        stem = path.stem
        parts = stem.split("-")
        if len(parts) < 8:
            continue
        start_text = "-".join(parts[-6:-3])
        end_text = "-".join(parts[-3:])
        try:
            start = pd.Timestamp(start_text, tz="UTC")
            end = pd.Timestamp(end_text, tz="UTC") + pd.Timedelta(days=1)
        except ValueError:
            continue
        coverage.append(ExecutionCoverage(start, end, path.name))
    return sorted(coverage, key=lambda item: item.start_utc)


def coverage_status(
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    coverage: Iterable[ExecutionCoverage],
) -> str:
    for item in coverage:
        if item.start_utc <= start_utc and end_utc <= item.end_utc:
            return item.status
    return "UNVERIFIED"


def audit_orders(executions: pd.DataFrame) -> pd.DataFrame:
    trades = executions[executions["exectype"].eq("Trade")].copy()
    trades = trades[trades["orderid"].notna() & trades["orderid"].ne(ZERO_UUID)]
    trades = trades.sort_values(
        [
            "orderid",
            "event_time_utc",
            "processed_time_utc",
            "cumqty",
            "execid",
        ],
        kind="stable",
    )

    rows: list[dict[str, object]] = []
    for order_id, group in trades.groupby("orderid", sort=False):
        cumqty = group["cumqty"].astype("int64")
        lastqty = group["lastqty"].astype("int64")
        increments = cumqty.diff().fillna(cumqty.iloc[0]).astype("int64")
        rows.append(
            {
                "order_id": order_id,
                "symbol": group["symbol"].iloc[0],
                "side": group["side"].iloc[0],
                "first_event_time_utc": group["event_time_utc"].iloc[0],
                "last_event_time_utc": group["event_time_utc"].iloc[-1],
                "fill_row_count": len(group),
                "sum_last_qty": int(lastqty.sum()),
                "first_cum_qty": int(cumqty.iloc[0]),
                "max_cum_qty": int(cumqty.max()),
                "final_leaves_qty": int(group["leavesqty"].iloc[-1]),
                "final_order_status": group["ordstatus"].iloc[-1],
                "starts_at_first_fill": bool(cumqty.iloc[0] == lastqty.iloc[0]),
                "cumqty_monotonic": bool(cumqty.is_monotonic_increasing),
                "increments_match_lastqty": bool(increments.eq(lastqty).all()),
                "sum_matches_max_cumqty": bool(lastqty.sum() == cumqty.max()),
                "symbol_consistent": bool(group["symbol"].nunique(dropna=False) == 1),
                "side_consistent": bool(group["side"].nunique(dropna=False) == 1),
                "orderqty_consistent": bool(
                    group["orderqty"].nunique(dropna=False) == 1
                ),
            }
        )
    return pd.DataFrame(rows)


def funding_anchor_candidates(executions: pd.DataFrame) -> pd.DataFrame:
    funding = executions[
        executions["exectype"].eq("Funding")
        & executions["foreignnotional"].notna()
        & executions["foreignnotional"].ne(0)
    ].copy()
    funding["position_qty"] = (
        -np.sign(funding["foreignnotional"]) * funding["lastqty"]
    ).astype("int64")
    funding["anchor_source"] = "FUNDING_INFERRED"
    return funding[
        ["symbol", "event_time_utc", "execid", "position_qty", "anchor_source"]
    ].sort_values(["symbol", "event_time_utc", "execid"], kind="stable")


def validate_funding_anchors(
    executions: pd.DataFrame, anchors: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validations: list[dict[str, object]] = []
    for symbol, symbol_anchors in anchors.groupby("symbol", sort=True):
        symbol_events = executions[executions["symbol"].eq(symbol)]
        trades = symbol_events[symbol_events["exectype"].eq("Trade")]
        previous = None
        for anchor in symbol_anchors.itertuples(index=False):
            if previous is not None:
                between = trades[
                    (trades["event_time_utc"] > previous.event_time_utc)
                    & (trades["event_time_utc"] <= anchor.event_time_utc)
                ]
                trade_delta = int(between["trade_signed_qty"].sum())
                expected = int(previous.position_qty) + trade_delta
                error = int(anchor.position_qty) - expected
                validations.append(
                    {
                        "symbol": symbol,
                        "previous_anchor_time_utc": previous.event_time_utc,
                        "anchor_time_utc": anchor.event_time_utc,
                        "previous_position_qty": int(previous.position_qty),
                        "trade_delta": trade_delta,
                        "expected_position_qty": expected,
                        "observed_position_qty": int(anchor.position_qty),
                        "position_error": error,
                    }
                )
            previous = anchor

    detail = pd.DataFrame(validations)
    if detail.empty:
        return detail, pd.DataFrame(
            columns=["symbol", "interval_count", "exact_count", "exact_rate", "max_abs_error"]
        )
    summary = (
        detail.assign(exact=detail["position_error"].eq(0))
        .groupby("symbol", as_index=False)
        .agg(
            interval_count=("position_error", "size"),
            exact_count=("exact", "sum"),
            max_abs_error=("position_error", lambda values: int(values.abs().max())),
        )
    )
    summary["exact_count"] = summary["exact_count"].astype("int64")
    summary["exact_rate"] = summary["exact_count"] / summary["interval_count"]
    return detail, summary


def accepted_position_anchors(
    executions: pd.DataFrame,
    minimum_funding_intervals: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    funding = funding_anchor_candidates(executions)
    detail, summary = validate_funding_anchors(executions, funding)
    accepted_symbols = set(
        summary.loc[
            summary["interval_count"].ge(minimum_funding_intervals)
            & summary["exact_rate"].eq(1.0),
            "symbol",
        ]
    )
    accepted_funding = funding[funding["symbol"].isin(accepted_symbols)].copy()
    accepted_funding["anchor_source"] = "FUNDING_INFERRED_VALIDATED"

    settlements = executions[executions["exectype"].eq("Settlement")].copy()
    settlement_anchors = pd.DataFrame(
        {
            "symbol": settlements["symbol"],
            "event_time_utc": settlements["event_time_utc"],
            "execid": settlements["execid"],
            "position_qty": 0,
            "anchor_source": "SETTLEMENT_ZERO",
        }
    )
    anchors = pd.concat([accepted_funding, settlement_anchors], ignore_index=True)
    anchors = anchors.sort_values(
        ["symbol", "event_time_utc", "anchor_source", "execid"], kind="stable"
    ).reset_index(drop=True)
    return anchors, detail, summary


def positions_before_decisions(
    decisions: pd.DataFrame,
    executions: pd.DataFrame,
    anchors: pd.DataFrame,
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for symbol, symbol_decisions in decisions.groupby("symbol", sort=False):
        current = symbol_decisions.sort_values("decision_time_utc").copy()
        symbol_anchors = anchors[anchors["symbol"].eq(symbol)].sort_values(
            "event_time_utc"
        )
        symbol_trades = executions[
            executions["symbol"].eq(symbol) & executions["exectype"].eq("Trade")
        ].sort_values("event_time_utc")
        if symbol_anchors.empty:
            current["position_before_qty"] = pd.Series(
                pd.NA, index=current.index, dtype="Int64"
            )
            current["position_status"] = "UNKNOWN"
            current["position_anchor_time_utc"] = pd.NaT
            current["position_anchor_source"] = pd.NA
            current["position_asof_time_utc"] = pd.NaT
            outputs.append(current)
            continue

        decision_ns = current["decision_time_utc"].astype("int64").to_numpy()
        anchor_ns = symbol_anchors["event_time_utc"].astype("int64").to_numpy()
        anchor_indices = np.searchsorted(anchor_ns, decision_ns, side="left") - 1
        known = anchor_indices >= 0

        trade_ns = symbol_trades["event_time_utc"].astype("int64").to_numpy()
        trade_qty = symbol_trades["trade_signed_qty"].to_numpy(dtype="int64")
        cumulative = np.concatenate(([0], np.cumsum(trade_qty, dtype="int64")))
        decision_trade_indices = np.searchsorted(trade_ns, decision_ns, side="left")

        positions = np.full(len(current), np.nan)
        anchor_times: list[pd.Timestamp | pd.NaTType] = []
        anchor_sources: list[str | pd._libs.missing.NAType] = []
        position_asof_times: list[pd.Timestamp | pd.NaTType] = []
        for index, anchor_index in enumerate(anchor_indices):
            if anchor_index < 0:
                anchor_times.append(pd.NaT)
                anchor_sources.append(pd.NA)
                position_asof_times.append(pd.NaT)
                continue
            anchor = symbol_anchors.iloc[anchor_index]
            anchor_time_ns = int(anchor["event_time_utc"].value)
            after_anchor = np.searchsorted(trade_ns, anchor_time_ns, side="right")
            delta = cumulative[decision_trade_indices[index]] - cumulative[after_anchor]
            positions[index] = int(anchor["position_qty"]) + int(delta)
            anchor_times.append(anchor["event_time_utc"])
            anchor_sources.append(str(anchor["anchor_source"]))
            contributing_index = decision_trade_indices[index] - 1
            if contributing_index >= after_anchor:
                position_asof_times.append(
                    symbol_trades.iloc[contributing_index]["event_time_utc"]
                )
            else:
                position_asof_times.append(anchor["event_time_utc"])

        current["position_before_qty"] = pd.array(positions, dtype="Int64")
        current["position_status"] = np.where(known, "KNOWN", "UNKNOWN")
        current["position_anchor_time_utc"] = anchor_times
        current["position_anchor_source"] = anchor_sources
        current["position_asof_time_utc"] = position_asof_times
        outputs.append(current)
    return pd.concat(outputs, ignore_index=True)
