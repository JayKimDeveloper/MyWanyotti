#!/usr/bin/env python3
"""Build leakage-safe decision rows and observed behavior labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wonyotti_pipeline.dataset import (  # noqa: E402
    apply_eligibility_policy,
    build_target_labels,
    decision_grid,
    leakage_report,
    load_instrument_snapshot,
)
from wonyotti_pipeline.executions import (  # noqa: E402
    accepted_position_anchors,
    audit_orders,
    coverage_from_filenames,
    execution_paths,
    load_executions,
    positions_before_decisions,
)
from wonyotti_pipeline.market import (  # noqa: E402
    build_market_features,
    compare_private_public_trades,
    discover_daily_files,
    load_public_quotes,
    load_public_trades,
)


def utc_timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--market-dir", type=Path, default=Path("data/raw/market/bitmex")
    )
    parser.add_argument(
        "--instrument-json",
        type=Path,
        default=Path(
            "data/raw/market/bitmex/instrument/instruments_2026-10-08.json"
        ),
    )
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--interval", default="5min")
    parser.add_argument("--lookback", default="5min")
    parser.add_argument("--max-quote-age", default="30s")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/training")
    )
    parser.add_argument(
        "--audit-dir", type=Path, default=Path("data/audit/training_pipeline")
    )
    return parser.parse_args()


def count_values(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts().items()}


def count_exclusion_reasons(series: pd.Series) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in series:
        for reason in filter(None, str(value).split("|")):
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    start = utc_timestamp(args.start)
    end = utc_timestamp(args.end)
    interval = pd.Timedelta(args.interval)
    lookback = pd.Timedelta(args.lookback)
    max_quote_age = pd.Timedelta(args.max_quote_age)
    if end <= start:
        raise ValueError("end must be after start")
    if interval <= pd.Timedelta(0):
        raise ValueError("interval must be positive")
    if end - start < interval:
        raise ValueError("requested range must contain at least one complete interval")

    paths = execution_paths(args.raw_dir)
    executions = load_executions(paths, symbols=set(args.symbols))
    order_audit = audit_orders(executions)
    order_integrity = (
        order_audit["starts_at_first_fill"]
        & order_audit["cumqty_monotonic"]
        & order_audit["increments_match_lastqty"]
        & order_audit["sum_matches_max_cumqty"]
        & order_audit["symbol_consistent"]
        & order_audit["side_consistent"]
    )
    invalid_order_ids = set(
        order_audit.loc[~order_integrity, "order_id"].astype(str)
    )
    anchors, anchor_detail, anchor_summary = accepted_position_anchors(executions)
    coverage = coverage_from_filenames(paths)

    decisions = decision_grid(args.symbols, start, end, interval)
    decisions = positions_before_decisions(decisions, executions, anchors)

    market_start = start - lookback
    trade_paths, missing_trade_dates = discover_daily_files(
        args.market_dir, "trade", market_start, end
    )
    quote_paths, missing_quote_dates = discover_daily_files(
        args.market_dir, "quote", market_start, end
    )
    public_trades = load_public_trades(
        trade_paths, set(args.symbols), market_start, end
    )
    public_quotes = load_public_quotes(
        quote_paths, set(args.symbols), market_start, end
    )
    featured = build_market_features(
        decisions, public_trades, public_quotes, lookback, max_quote_age
    )
    labels = build_target_labels(
        featured, executions, interval, coverage, invalid_order_ids
    )
    dataset = featured.merge(
        labels,
        on=["symbol", "decision_time_utc"],
        how="left",
        validate="one_to_one",
    )

    settlement_currency = (
        executions.groupby("symbol")["settlcurrency"]
        .agg(lambda values: "|".join(sorted(set(values.dropna().astype(str)))))
        .rename("settlement_currency_observed")
        .reset_index()
    )
    dataset = dataset.merge(settlement_currency, on="symbol", how="left")
    dataset["position_unit"] = "CONTRACTS"
    dataset["quantity_conversion_status"] = "RAW_CONTRACTS_ONLY"

    metadata = load_instrument_snapshot(args.instrument_json)
    dataset = dataset.merge(metadata, on="symbol", how="left", validate="many_to_one")
    dataset["instrument_metadata_status"] = dataset[
        "instrument_metadata_status"
    ].fillna("MISSING")
    dataset = apply_eligibility_policy(dataset)

    reconciliation, reconciliation_sample = compare_private_public_trades(
        executions, public_trades, start, end
    )
    leak_checks = leakage_report(dataset)
    report = {
        "exchange": "BitMEX",
        "synthetic_data_used": False,
        "data_contract_version": "v0.3",
        "source_timezone_assumption": "UTC_LOCALIZE_NAIVE_BITMEX_EXPORT",
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "interval": str(interval),
        "lookback": str(lookback),
        "max_quote_age": str(max_quote_age),
        "symbols": args.symbols,
        "row_count": int(len(dataset)),
        "eligible_row_count": int(dataset["training_eligible"].sum()),
        "label_counts": count_values(dataset["target_label"]),
        "position_status_counts": count_values(dataset["position_status"]),
        "market_feature_status_counts": count_values(
            dataset["market_feature_status"]
        ),
        "exclusion_reason_counts": count_exclusion_reasons(
            dataset.loc[~dataset["training_eligible"], "exclusion_reasons"]
        ),
        "exclusion_combination_counts": count_values(
            dataset.loc[~dataset["training_eligible"], "exclusion_reasons"]
        ),
        "missing_market_trade_dates": missing_trade_dates,
        "missing_market_quote_dates": missing_quote_dates,
        "accepted_position_anchor_count": int(len(anchors)),
        "audited_order_count": int(len(order_audit)),
        "order_integrity_failed_order_count": len(invalid_order_ids),
        "target_order_integrity_failure_row_count": int(
            dataset["target_order_integrity_failed"].sum()
        ),
        "funding_anchor_validation": anchor_summary.to_dict(orient="records"),
        "private_public_trade_reconciliation": reconciliation,
        "instrument_snapshot": metadata[
            metadata["symbol"].isin(args.symbols)
        ].to_dict(orient="records"),
        "quantity_unit": "CONTRACTS",
        "quantity_conversion_status": "RAW_CONTRACTS_ONLY",
        "leakage_checks": leak_checks,
        "no_fill_semantics": (
            "no recorded Trade in an interval covered by an explicit execution "
            "export filename range"
        ),
        "no_fill_intent_claimed": False,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(args.output_dir / "training_candidates.parquet", index=False)
    dataset.loc[dataset["training_eligible"]].to_parquet(
        args.output_dir / "training_eligible.parquet", index=False
    )
    reconciliation_sample.to_parquet(
        args.audit_dir / "private_public_trade_sample.parquet", index=False
    )
    order_audit.to_parquet(args.audit_dir / "order_audit.parquet", index=False)
    anchor_detail.to_parquet(
        args.audit_dir / "funding_anchor_validation.parquet", index=False
    )
    (args.audit_dir / "training_dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
