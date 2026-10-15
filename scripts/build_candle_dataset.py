#!/usr/bin/env python3
"""Build candle_v1 hourly behavior rows using resumable daily partitions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wonyotti_pipeline.candles import (
    CONTRACT_VERSION, HOUR, RUNTIME_CONTRACT, atomic_json, atomic_parquet,
    build_candle_features, reconstruct_account, build_validated_labels,
)
from wonyotti_pipeline.dataset import decision_grid
from wonyotti_pipeline.executions import execution_paths
from wonyotti_pipeline.account_evidence import VALIDITY_POLICY, load_account_evidence
from wonyotti_pipeline.training import file_sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-03-05")
    parser.add_argument("--end", default="2022-01-01")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--candle-dir", type=Path, default=Path("data/raw/market/bitmex/candle_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/training/candle_v1_evidence_v1"))
    parser.add_argument("--audit-dir", type=Path, default=Path("data/audit/candle_v1_evidence_v1"))
    parser.add_argument("--evidence", type=Path, help="Explicit source-bound account evidence JSON; absent means unknown")
    args = parser.parse_args()
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    try:
        start, end = pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC")
        if start != start.normalize() or end != end.normalize() or end <= start:
            raise ValueError("start/end must be ascending UTC dates")
        download_path = args.candle_dir / "download_checkpoint.json"
        download = json.loads(download_path.read_text())
        account_policy_keys = {"position_anchor_policy", "no_fill_semantics", "account_validity_policy"}
        market_contract = lambda contract: {k: v for k, v in contract.items() if k not in account_policy_keys}
        if (download.get("status") != "COMPLETE"
                or market_contract(download["contract"]["runtime_contract"]) != market_contract(RUNTIME_CONTRACT)):
            raise ValueError("candle download is incomplete or has an incompatible contract; resume fetch_candles.py")
        account_hashes = {str(p.resolve()): file_sha256(p) for p in execution_paths(args.raw_dir)}
        evidence = load_account_evidence(args.evidence, account_hashes)
        candles = []
        source_hashes = {}
        for day in pd.date_range((start - pd.Timedelta(hours=26)).normalize(), end - pd.Timedelta(days=1), freq="1D"):
            key = day.strftime("%Y%m%d")
            path = args.candle_dir / f"{key}.parquet"
            if key not in download["days"] or not path.exists():
                raise ValueError(f"missing completed candle date: {path}")
            digest = file_sha256(path)
            if digest != download["days"][key]["sha256"]:
                raise ValueError(f"candle checksum mismatch: {path}")
            source_hashes[key] = digest
            candles.append(pd.read_parquet(path))
        candle_frame = pd.concat(candles, ignore_index=True)
        grid = decision_grid(["XBTUSD"], start, end + HOUR, HOUR)
        executions, positions, invalid_orders, position_report = reconstruct_account(args.raw_dir, grid, args.audit_dir, evidence)
        positions = positions[positions["decision_time_utc"] < end].reset_index(drop=True)
        fingerprint = {"validity_policy": VALIDITY_POLICY, "evidence": evidence.records, "contract": RUNTIME_CONTRACT, "start": start.isoformat(), "end": end.isoformat(),
                       "account_files": position_report["source_files"], "candle_files": source_hashes,
                       "implementation_sha256": file_sha256(Path(__file__).resolve()),
                       "feature_code_sha256": file_sha256(Path(__file__).resolve().parents[1] / "src/wonyotti_pipeline/candles.py"),
                       "evidence_code_sha256": file_sha256(Path(__file__).resolve().parents[1] / "src/wonyotti_pipeline/account_evidence.py"),
                       "shared_code_sha256": {name: file_sha256(Path(__file__).resolve().parents[1] / "src/wonyotti_pipeline" / name)
                                              for name in ("executions.py", "dataset.py")}}
        checkpoint_path = args.output_dir / "build_checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {"fingerprint": fingerprint, "days": {}}
        if checkpoint["fingerprint"] != fingerprint:
            raise ValueError("dataset checkpoint inputs/code changed; use a new output directory")
        execution_times = executions["event_time_utc"].astype("int64").to_numpy()
        for day, daily in positions.groupby(positions["decision_time_utc"].dt.normalize()):
            key = day.strftime("%Y%m%d")
            path = args.output_dir / "days" / f"{key}.parquet"
            if key in checkpoint["days"]:
                if not path.exists() or file_sha256(path) != checkpoint["days"][key]:
                    raise ValueError(f"dataset checkpoint partition changed: {path}")
                continue
            featured = build_candle_features(daily, candle_frame[(candle_frame["candle_end_utc"] >= day - pd.Timedelta(hours=26)) & (candle_frame["candle_end_utc"] < day + pd.Timedelta(days=1))])
            left = np.searchsorted(execution_times, day.value, side="left")
            right = np.searchsorted(execution_times, (day + pd.Timedelta(days=1)).value, side="left")
            dataset = build_validated_labels(featured, executions.iloc[left:right], evidence, invalid_orders)
            atomic_parquet(path, dataset)
            checkpoint["days"][key] = file_sha256(path)
            atomic_json(checkpoint_path, checkpoint)
            if len(checkpoint["days"]) % 100 == 0:
                print(json.dumps({"days_built": len(checkpoint["days"]), "through": key}), flush=True)
        dataset = pd.concat([pd.read_parquet(args.output_dir / "days" / f"{key}.parquet") for key in sorted(checkpoint["days"])], ignore_index=True)
        violations = {
            "duplicate_decision_keys": int(dataset.duplicated(["symbol", "decision_time_utc"]).sum()),
            "future_candle_rows": int(dataset["candle_end_utc"].ge(dataset["decision_time_utc"]).sum()),
            "future_position_rows": int(dataset["position_asof_time_utc"].ge(dataset["decision_time_utc"]).sum()),
            "future_anchor_dependencies": int(dataset["position_anchor_available_at_utc"].ge(dataset["decision_time_utc"]).sum()),
            "unverified_no_fill": int((dataset.target_label.eq("NO_FILL") & dataset.target_coverage_status.ne("VERIFIED")).sum()),
        }
        if any(violations.values()):
            raise ValueError(f"dataset leakage checks failed: {violations}")
        exclusions = Counter(reason for value in dataset["exclusion_reasons"] for reason in value.split("|") if reason)
        eligible = dataset[dataset["training_eligible"]]
        ranges = []
        for _, group in eligible.groupby(eligible.decision_time_utc.diff().ne(HOUR).cumsum()):
            ranges.append({"start_utc": group.decision_time_utc.min().isoformat(),
                           "end_exclusive_utc": group.target_end_time_utc.max().isoformat(), "decision_windows": len(group),
                           "coverage_evidence_ids": sorted(set("|".join(group.coverage_evidence_ids).split("|")) - {""}),
                           "timestamp_evidence_ids": sorted(set("|".join(group.timestamp_evidence_ids).split("|")) - {""}),
                           "position_evidence_ids": sorted(set(group.position_evidence_id.dropna()))})
        missing = [kind for kind in ("coverage", "timestamps") if not evidence.scoped(kind, "XBTUSD")]
        if not position_report["accepted_causal_anchors"]:
            missing.append("causally admissible position snapshot or covered funding prefix")
        report = {"validity_policy": VALIDITY_POLICY, "eligibility_status": "READY" if len(eligible) else "BLOCKED_MISSING_EVIDENCE_OR_VALID_ROWS",
                  "coverage_evidence": evidence.records,
                  "eligible_ranges": ranges, "remaining_missing_inputs": missing,
                  "data_contract_version": CONTRACT_VERSION, "synthetic_data_used": False,
                  "runtime_contract": RUNTIME_CONTRACT, "start_utc": start.isoformat(), "end_utc": end.isoformat(),
                  "candidate_rows": len(dataset), "eligible_rows": len(eligible),
                  "candidate_label_counts": dataset["target_label"].value_counts().to_dict(),
                  "eligible_label_counts": eligible["target_label"].value_counts().to_dict(),
                  "exclusion_reason_counts": dict(exclusions), "leakage_checks": violations,
                  "position_report": position_report, "source_candle_days": len(source_hashes),
                  "source_fingerprint": fingerprint}
        atomic_parquet(args.output_dir / "training_candidates.parquet", dataset)
        atomic_parquet(args.output_dir / "training_eligible.parquet", eligible)
        atomic_json(args.audit_dir / "training_dataset_report.json", report)
        atomic_json(args.audit_dir / "eligibility_report.json", {
            key: report[key] for key in ("validity_policy", "eligibility_status", "start_utc", "end_utc", "candidate_rows", "eligible_rows",
                                        "candidate_label_counts", "eligible_label_counts", "eligible_ranges",
                                        "exclusion_reason_counts", "coverage_evidence", "remaining_missing_inputs", "leakage_checks")})
        checkpoint["status"] = "COMPLETE"
        atomic_json(checkpoint_path, checkpoint)
        print(json.dumps({k: v for k, v in report.items() if k not in {"source_fingerprint", "position_report"}}, indent=2))
        return 0
    except (OSError, ValueError) as error:
        atomic_json(args.audit_dir / "dataset_blocked_report.json", {"status": "BLOCKED", "error": str(error),
                    "resume_command": " .venv/bin/python scripts/build_candle_dataset.py".strip()})
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
