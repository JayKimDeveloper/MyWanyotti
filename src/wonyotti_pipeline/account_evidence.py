"""Explicit, source-bound account evidence for candle dataset eligibility."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .executions import ExecutionCoverage
from .training import file_sha256

VALIDITY_POLICY = "candle_v1_evidence_v1"
EVIDENCE_KINDS = ("coverage", "timestamps", "positions")


def utc(value: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError("evidence timestamps must include an explicit UTC offset")
    return result.tz_convert("UTC")


class AccountEvidence:
    def __init__(self, records: dict | None = None, account: str = "aoa"):
        self.records = records or {kind: [] for kind in EVIDENCE_KINDS}
        self.account = account
        self._intervals = {}

    def scoped(self, kind: str, symbol: str) -> list[dict]:
        return [r for r in self.records[kind] if r["status"] == "VERIFIED"
                and r["account"] == self.account and r["symbol"] == symbol]

    def intervals(self, kind: str, symbol: str) -> list[ExecutionCoverage]:
        key = (kind, symbol, self.account)
        if key in self._intervals:
            return self._intervals[key]
        merged = []
        for r in sorted(self.scoped(kind, symbol), key=lambda r: utc(r["start_utc"])):
            start, end = utc(r["start_utc"]), utc(r["end_utc"])
            if merged and start <= merged[-1].end_utc:
                previous = merged.pop()
                merged.append(ExecutionCoverage(previous.start_utc, max(end, previous.end_utc),
                                                previous.source_file + "|" + r["id"], "VERIFIED"))
            else:
                merged.append(ExecutionCoverage(start, end, r["id"], "VERIFIED"))
        self._intervals[key] = merged
        return merged

    def supporting_ids(self, kind: str, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
        if not any(r.start_utc <= start and end <= r.end_utc for r in self.intervals(kind, symbol)):
            return []
        return sorted(r["id"] for r in self.scoped(kind, symbol)
                      if utc(r["start_utc"]) < end and utc(r["end_utc"]) > start)

    def verified(self, kind: str, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        return bool(self.supporting_ids(kind, symbol, start, end))

    def continuous(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        # Include the anchor instant, even when the decision has the same timestamp.
        end = max(end, start + pd.Timedelta(1, unit="ns"))
        return all(any(r.start_utc <= start and end <= r.end_utc
                       for r in self.intervals(kind, symbol)) for kind in ("coverage", "timestamps"))


def load_account_evidence(path: Path | None, source_hashes: dict[str, str], account: str = "aoa") -> AccountEvidence:
    if path is None:
        return AccountEvidence(account=account)
    source_hashes = {str(Path(name).resolve()): digest for name, digest in source_hashes.items()}
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != VALIDITY_POLICY:
        raise ValueError("account evidence schema mismatch")
    ids = set()
    for kind in EVIDENCE_KINDS:
        if not isinstance(payload.get(kind), list):
            raise ValueError(f"evidence {kind} must be a list")
        for record in payload[kind]:
            if not isinstance(record, dict) or not {"id", "account", "symbol", "start_utc", "end_utc", "status"}.issubset(record):
                raise ValueError("evidence record missing scope fields")
            if record.get("status") not in {"VERIFIED", "UNKNOWN"}:
                raise ValueError("evidence status must be VERIFIED or UNKNOWN")
            if record["id"] in ids:
                raise ValueError("duplicate evidence ID")
            ids.add(record["id"])
            start, end = utc(record["start_utc"]), utc(record["end_utc"])
            if start >= end or not record.get("account") or not record.get("symbol"):
                raise ValueError("invalid evidence scope")
            if record["status"] == "UNKNOWN":
                continue
            required = {"artifact", "artifact_sha256", "source_files", "basis"}
            if kind == "positions":
                required |= {"event_time_utc", "available_at_utc", "position_qty"}
            if not required.issubset(record):
                raise ValueError("verified evidence record missing provenance fields")
            artifact = (path.parent / record["artifact"]).resolve()
            if file_sha256(artifact) != record["artifact_sha256"]:
                raise ValueError(f"evidence artifact checksum mismatch: {record['id']}")
            bindings = record.get("source_files", {})
            if not bindings or any(source_hashes.get(str(Path(name).resolve())) != digest
                                   for name, digest in bindings.items()):
                raise ValueError(f"evidence source binding mismatch: {record['id']}")
            if not record.get("basis"):
                raise ValueError("verified evidence requires an explicit basis")
            if kind == "timestamps" and (record.get("timezone") != "UTC"
                    or record.get("fields") != ["transacttime", "timestamp"]):
                raise ValueError("timestamp evidence must establish UTC for both execution clocks")
            if kind == "positions":
                at, available = utc(record["event_time_utc"]), utc(record["available_at_utc"])
                if not start <= at < end or available < at or type(record.get("position_qty")) is not int:
                    raise ValueError("invalid position snapshot or availability time")
    return AccountEvidence({kind: payload[kind] for kind in EVIDENCE_KINDS}, account)
