#!/usr/bin/env python3
"""Inventory account CSVs without exposing identifiers or trusting truncated times."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wonyotti_pipeline.candles import atomic_json
from wonyotti_pipeline.executions import parse_utc
from wonyotti_pipeline.training import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/audit/candle_v1/account_inventory.json"))
    args = parser.parse_args()
    reports = []
    for path in sorted(args.raw_dir.glob("aoa-*.csv")):
        fields = pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns.tolist()
        wallet = "wallet" in path.name
        time_column = "date" if wallet else "transacttime"
        names = [name for name in (time_column, "symbol", "exectype", "transacttype", "amount", "currency") if name in fields]
        symbols, events, currencies = Counter(), Counter(), Counter()
        rows = missing_types = populated_amounts = 0
        earliest = latest = None
        for chunk in pd.read_csv(path, encoding="utf-8-sig", usecols=names, chunksize=100000, low_memory=False):
            rows += len(chunk)
            times = parse_utc(chunk[time_column].dropna())
            if len(times):
                earliest = min(earliest, times.min()) if earliest is not None else times.min()
                latest = max(latest, times.max()) if latest is not None else times.max()
            if "symbol" in chunk:
                symbols.update(chunk["symbol"].value_counts().to_dict())
            event_column = "transacttype" if wallet else "exectype"
            events.update(chunk[event_column].value_counts().to_dict())
            missing_types += int(chunk[event_column].isna().sum())
            currencies.update(chunk["currency"].value_counts().to_dict())
            if "amount" in chunk:
                populated_amounts += int(chunk["amount"].notna().sum())
        report = {"path": str(path.resolve()), "sha256": file_sha256(path), "rows": rows,
                  "fields": fields, "symbol_counts": dict(symbols), "event_counts": dict(events),
                  "currency_counts": dict(currencies), "date_range_field": time_column,
                  "start_utc": str(earliest), "end_utc": str(latest), "missing_event_type_rows": missing_types,
                  "populated_amount_rows": populated_amounts if wallet else None,
                  "wallet_timestamp_status": "TRUNCATED_TIME_STRINGS; DATE_ONLY_VALID" if wallet else None}
        reports.append(report)
        print(json.dumps({k: report[k] for k in ("path", "rows", "start_utc", "end_utc", "event_counts")}), flush=True)
    atomic_json(args.output, reports)


if __name__ == "__main__":
    main()
