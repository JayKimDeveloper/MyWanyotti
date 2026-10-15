#!/usr/bin/env python3
"""Audit real fills, partial orders, and empirical position anchors."""

from __future__ import annotations

import argparse
import json
import sys
from math import gcd
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wonyotti_pipeline.executions import (  # noqa: E402
    accepted_position_anchors,
    audit_orders,
    execution_paths,
    load_executions,
)
from wonyotti_pipeline.dataset import load_instrument_snapshot  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/audit/training_pipeline")
    )
    parser.add_argument(
        "--instrument-json",
        type=Path,
        default=Path(
            "data/raw/market/bitmex/instrument/instruments_2026-10-08.json"
        ),
    )
    return parser.parse_args()


def quantity_metadata_audit(
    executions: pd.DataFrame, instrument_json: Path
) -> pd.DataFrame:
    trades = executions[executions["exectype"].eq("Trade")]
    rows: list[dict[str, object]] = []
    for symbol, group in trades.groupby("symbol", sort=True):
        quantities = group["lastqty"].astype("int64")
        quantity_gcd = 0
        for value in quantities.unique():
            quantity_gcd = gcd(quantity_gcd, int(value))
        rows.append(
            {
                "symbol": symbol,
                "trade_fill_count": len(group),
                "observed_min_fill_qty": int(quantities.min()),
                "observed_max_fill_qty": int(quantities.max()),
                "observed_fill_qty_gcd": quantity_gcd,
                "observed_settlement_currencies": "|".join(
                    sorted(set(group["settlcurrency"].dropna().astype(str)))
                ),
                "position_unit": "CONTRACTS",
            }
        )
    audit = pd.DataFrame(rows)
    metadata = load_instrument_snapshot(instrument_json)
    audit = audit.merge(metadata, on="symbol", how="left", validate="one_to_one")
    non_multiple_counts: list[int | None] = []
    for row in audit.itertuples(index=False):
        lot_size = row.metadata_lot_size_snapshot
        if pd.isna(lot_size) or float(lot_size) <= 0:
            non_multiple_counts.append(None)
            continue
        quantities = trades.loc[trades["symbol"].eq(row.symbol), "lastqty"]
        non_multiple_counts.append(int(quantities.mod(float(lot_size)).ne(0).sum()))
    audit["snapshot_lot_size_non_multiple_fill_count"] = pd.array(
        non_multiple_counts, dtype="Int64"
    )
    audit["historical_conversion_allowed"] = False
    return audit


def main() -> None:
    args = parse_args()
    paths = execution_paths(args.raw_dir)
    executions = load_executions(paths)
    orders = audit_orders(executions)
    anchors, anchor_detail, anchor_summary = accepted_position_anchors(executions)
    quantity_audit = quantity_metadata_audit(executions, args.instrument_json)

    event_counts = executions["exectype"].value_counts().sort_index().to_dict()
    order_summary = {
        "order_count": int(len(orders)),
        "multi_fill_order_count": int(orders["fill_row_count"].gt(1).sum()),
        "max_fill_rows_per_order": int(orders["fill_row_count"].max()),
        "starts_after_missing_fill_count": int((~orders["starts_at_first_fill"]).sum()),
        "non_monotonic_cumqty_count": int((~orders["cumqty_monotonic"]).sum()),
        "increment_mismatch_count": int((~orders["increments_match_lastqty"]).sum()),
        "sum_max_mismatch_count": int((~orders["sum_matches_max_cumqty"]).sum()),
        "inconsistent_symbol_count": int((~orders["symbol_consistent"]).sum()),
        "inconsistent_side_count": int((~orders["side_consistent"]).sum()),
        "inconsistent_orderqty_count": int((~orders["orderqty_consistent"]).sum()),
    }
    accepted_summary = anchor_summary.assign(
        accepted=lambda frame: frame["symbol"].isin(set(anchors["symbol"]))
    )
    report = {
        "source_timezone_assumption": "UTC_LOCALIZE_NAIVE_BITMEX_EXPORT",
        "event_counts": {key: int(value) for key, value in event_counts.items()},
        "order_summary": order_summary,
        "funding_anchor_summary": accepted_summary.to_dict(orient="records"),
        "accepted_anchor_count": int(len(anchors)),
        "instrument_count": int(len(quantity_audit)),
        "snapshot_lot_size_incompatible_symbol_count": int(
            quantity_audit["snapshot_lot_size_non_multiple_fill_count"].gt(0).sum()
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    orders.to_parquet(args.output_dir / "order_audit.parquet", index=False)
    anchors.to_parquet(args.output_dir / "position_anchors.parquet", index=False)
    anchor_detail.to_parquet(
        args.output_dir / "funding_anchor_validation.parquet", index=False
    )
    quantity_audit.to_parquet(
        args.output_dir / "instrument_quantity_audit.parquet", index=False
    )
    (args.output_dir / "execution_structure.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
