#!/usr/bin/env python3
"""Fetch only hourly XBTUSD candles, with validated daily checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wonyotti_pipeline.candles import HOUR, RUNTIME_CONTRACT, atomic_json, atomic_parquet, validate_candles
from wonyotti_pipeline.training import file_sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-03-03")
    parser.add_argument("--end", default="2022-01-01", help="Exclusive candle-start boundary in UTC.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/market/bitmex/candle_v1"))
    args = parser.parse_args()
    start, end = pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC")
    if start != start.normalize() or end != end.normalize() or end <= start:
        parser.error("start/end must be ascending UTC dates")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "download_checkpoint.json"
    contract = {"start_utc": start.isoformat(), "end_utc": end.isoformat(), "runtime_contract": RUNTIME_CONTRACT}
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {"contract": contract, "days": {}}
    if checkpoint["contract"] != contract:
        raise ValueError("download checkpoint range/contract mismatch; use another output directory")
    cursor = start
    try:
        while cursor < end:
            key = cursor.strftime("%Y%m%d")
            path = args.output_dir / f"{key}.parquet"
            if key in checkpoint["days"]:
                if not path.exists() or file_sha256(path) != checkpoint["days"][key]["sha256"]:
                    raise ValueError(f"checkpoint candle file missing or changed: {path}")
                cursor += pd.Timedelta(days=1)
                continue
            page_end = min(cursor + pd.Timedelta(days=40), end)
            # Stop the page before an already completed date during a partial resume.
            while page_end > cursor + pd.Timedelta(days=1) and any(
                d.strftime("%Y%m%d") in checkpoint["days"] for d in pd.date_range(cursor, page_end - pd.Timedelta(days=1), freq="1D")
            ):
                page_end -= pd.Timedelta(days=1)
            query = urlencode({"binSize": "1h", "partial": "false", "symbol": "XBTUSD",
                               "count": 1000, "reverse": "false", "startTime": (cursor + HOUR).isoformat(),
                               "endTime": page_end.isoformat()})
            url = RUNTIME_CONTRACT["source"] + "?" + query
            response_path = args.output_dir / "response.json.tmp"
            result = None
            for attempt in range(4):
                result = subprocess.run(["curl", "--fail", "--silent", "--show-error", "--max-time", "45", url, "-o", str(response_path)], capture_output=True, text=True)
                if result.returncode == 0:
                    break
                if attempt < 3:
                    time.sleep(2 ** attempt * 2)
            if result.returncode:
                raise RuntimeError(f"candle request failed at {cursor.isoformat()}: {result.stderr.strip()}")
            frame = validate_candles(json.loads(response_path.read_text()), cursor, page_end)
            for day, daily in frame.groupby(frame["candle_start_utc"].dt.normalize()):
                key = day.strftime("%Y%m%d")
                path = args.output_dir / f"{key}.parquet"
                atomic_parquet(path, daily.reset_index(drop=True))
                checkpoint["days"][key] = {"rows": len(daily), "sha256": file_sha256(path)}
            checkpoint.update(status="IN_PROGRESS", completed_through_utc=page_end.isoformat(), last_request_url=url)
            atomic_json(checkpoint_path, checkpoint)
            print(json.dumps({"completed_through_utc": page_end.isoformat(), "days": len(checkpoint["days"])}), flush=True)
            cursor = page_end
            time.sleep(1)
        checkpoint["status"] = "COMPLETE"
        atomic_json(checkpoint_path, checkpoint)
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        atomic_json(args.output_dir / "blocked_report.json", {"status": "BLOCKED", "error": str(error), "next_candle_date_utc": cursor.isoformat(),
                   "resume_command": f".venv/bin/python scripts/fetch_candles.py --start {args.start} --end {args.end} --output-dir {args.output_dir}"})
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
