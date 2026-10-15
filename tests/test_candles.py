from __future__ import annotations

import sys
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wonyotti_pipeline.candles import (
    FEATURE_COLUMNS, HOUR, build_candle_features, causal_position_anchors,
    deduplicate_executions, feature_schema, prepare_model_features,
    reconstruct_account, validate_candles, build_validated_labels,
)
from wonyotti_pipeline.dataset import MODEL_FEATURE_COLUMNS, build_target_labels, decision_grid
from wonyotti_pipeline.executions import EXECUTION_COLUMNS, ExecutionCoverage, ZERO_UUID
from wonyotti_pipeline.account_evidence import AccountEvidence, VALIDITY_POLICY, load_account_evidence
from wonyotti_pipeline.training import file_sha256, class_coverage_issues, REQUIRED_ACTION_LABELS


def verified_evidence(intervals=((0, 24),), snapshots=()):
    records = {kind: [] for kind in ("coverage", "timestamps", "positions")}
    for kind in ("coverage", "timestamps"):
        records[kind] = [{"id": f"{kind}-{i}", "status": "VERIFIED", "account": "aoa", "symbol": "XBTUSD",
                          "start_utc": timestamp(start).isoformat(), "end_utc": timestamp(end).isoformat()}
                         for i, (start, end) in enumerate(intervals)]
    records["positions"] = [{"id": f"snapshot-{i}", "status": "VERIFIED", "account": "aoa", "symbol": "XBTUSD",
                             "start_utc": timestamp(hour).isoformat(), "end_utc": timestamp(hour + 1).isoformat(),
                             "event_time_utc": timestamp(hour).isoformat(), "available_at_utc": timestamp(available).isoformat(),
                             "position_qty": qty} for i, (hour, available, qty) in enumerate(snapshots)]
    return AccountEvidence(records)


def timestamp(hour: int) -> pd.Timestamp:
    return pd.Timestamp("2020-01-01", tz="UTC") + hour * HOUR


class CandleTests(unittest.TestCase):
    def test_download_resume_skips_verified_days_and_blocks_changed_files(self):
        script = Path(__file__).resolve().parents[1] / "scripts/fetch_candles.py"
        spec = importlib.util.spec_from_file_location("fetch_candles", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            def response(command, **kwargs):
                rows = [{"timestamp": timestamp(h).isoformat(), "symbol": "XBTUSD", "open": 100,
                         "high": 101, "low": 99, "close": 100, "volume": 10, "foreignNotional": 10}
                        for h in range(1, 49)]
                Path(command[-1]).write_text(json.dumps(rows))
                return SimpleNamespace(returncode=0, stderr="")
            arguments = ["fetch_candles.py", "--start", "2020-01-01", "--end", "2020-01-03", "--output-dir", directory]
            with mock.patch.object(sys, "argv", arguments), mock.patch.object(module.subprocess, "run", side_effect=response) as requests, mock.patch.object(module.time, "sleep"):
                self.assertEqual(module.main(), 0)
                self.assertEqual(module.main(), 0)
                requests.assert_called_once()
                Path(directory, "20200101.parquet").write_bytes(b"changed")
                with mock.patch.object(sys, "stderr"):
                    self.assertEqual(module.main(), 2)
                requests.assert_called_once()

    def test_candle_end_boundaries_units_and_missing_data(self):
        records = [{"timestamp": timestamp(h).isoformat(), "symbol": "XBTUSD", "open": 100,
                    "high": 102, "low": 99, "close": 101, "volume": 50, "foreignNotional": 50} for h in (1, 2)]
        frame = validate_candles(records, timestamp(0), timestamp(2))
        self.assertEqual(frame.iloc[0].candle_start_utc, timestamp(0))
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            validate_candles(records[:1], timestamp(0), timestamp(2))
        records[0]["foreignNotional"] = 51
        with self.assertRaisesRegex(ValueError, "notional"):
            validate_candles(records, timestamp(0), timestamp(2))

    def test_deduplication_keeps_distinct_partial_fills_and_blocks_conflicts(self):
        rows = pd.DataFrame([{name: None for name in EXECUTION_COLUMNS} for _ in range(3)])
        rows["execid"] = ["a", "a", "b"]
        rows["lastqty"] = [2, 2, 3]
        unique, removed = deduplicate_executions(rows)
        self.assertEqual((len(unique), removed), (2, 1))
        rows.loc[1, "lastqty"] = 9
        with self.assertRaisesRegex(ValueError, "conflicting"):
            deduplicate_executions(rows)

    def test_features_ignore_boundary_future_candles_and_require_contiguity(self):
        candles = pd.DataFrame({"candle_end_utc": [timestamp(h) for h in range(31)],
                                "close": np.arange(100.0, 131.0), "volume": 10.0,
                                "high": np.arange(101.0, 132.0), "low": np.arange(99.0, 130.0)})
        decisions = pd.DataFrame({"decision_time_utc": [timestamp(27)], "position_before_qty": [2],
                                  "entry_asof_time_utc": [timestamp(25)],
                                  "holding_hours": [1.0], "entry_price": [100.0]})
        before = build_candle_features(decisions, candles)
        candles.loc[candles.candle_end_utc >= timestamp(27), "close"] = 999999
        after = build_candle_features(decisions, candles)
        pd.testing.assert_frame_equal(prepare_model_features(before), prepare_model_features(after))
        self.assertEqual(before.iloc[0].candle_end_utc, timestamp(26))
        self.assertAlmostEqual(before.iloc[0].unrealized_return, 1 - 100 / 126)
        decisions["entry_asof_time_utc"] = timestamp(26) + pd.Timedelta(minutes=30)
        invalid_mark = build_candle_features(decisions, candles)
        self.assertTrue(pd.isna(invalid_mark.iloc[0].unrealized_return))
        broken = build_candle_features(decisions, candles[candles.candle_end_utc != timestamp(10)])
        self.assertEqual(broken.iloc[0].market_feature_status, "CANDLE_WINDOW_INCOMPLETE")
        self.assertEqual(len(MODEL_FEATURE_COLUMNS), 20)
        self.assertEqual(feature_schema()["columns"], FEATURE_COLUMNS)

    def test_causal_anchors_closes_reversals_and_unknown_entry(self):
        rows = []
        for hour in (0, 8, 16):
            rows.append(dict.fromkeys(EXECUTION_COLUMNS) | {
                "execid": f"f{hour}", "symbol": "XBTUSD", "exectype": "Funding",
                "lastqty": 10, "foreignnotional": -10,
                "transacttime": timestamp(hour).isoformat(), "timestamp": timestamp(hour).isoformat(),
            })
        for hour, side, quantity, price in ((17, "Sell", 4, 100), (18, "Buy", 4, 100),
                                            (19, "Sell", 10, 100), (20, "Sell", 8, 100), (21, "Buy", 12, 120), (22, "Buy", 4, 150)):
            rows.append(dict.fromkeys(EXECUTION_COLUMNS) | {
                "execid": f"t{hour}", "orderid": f"o{hour}", "symbol": "XBTUSD", "exectype": "Trade",
                "side": side, "lastqty": quantity, "cumqty": quantity, "orderqty": quantity,
                "leavesqty": 0, "ordstatus": "Filled", "lastpx": price,
                "transacttime": timestamp(hour).isoformat(), "timestamp": timestamp(hour).isoformat(),
            })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(rows).to_csv(root / "aoa-execution-2020-01-01-2020-01-01.csv", index=False)
            grid = decision_grid(["XBTUSD"], timestamp(0), timestamp(24), HOUR)
            executions, positions, invalid, report = reconstruct_account(root, grid, root / "audit", verified_evidence())
            self.assertEqual(report["accepted_causal_anchors"], 1)
            self.assertTrue(pd.isna(positions.iloc[16].position_before_qty))
            self.assertEqual(positions.iloc[17].position_before_qty, 10)
            self.assertTrue(pd.isna(positions.iloc[18].entry_price))
            self.assertEqual(positions.iloc[20].position_before_qty, 0)
            self.assertEqual(positions.iloc[21].position_before_qty, -8)
            self.assertEqual(positions.iloc[22].position_before_qty, 4)
            self.assertEqual(positions.iloc[22].entry_price, 120)
            self.assertEqual(positions.iloc[22].holding_hours, 1)
            self.assertAlmostEqual(positions.iloc[23].entry_price, 8 / (4 / 120 + 4 / 150))
            self.assertEqual(positions.iloc[23].holding_hours, 2)
            truncated = executions[executions.event_time_utc < timestamp(16)]
            anchors, _ = causal_position_anchors(truncated, verified_evidence())
            self.assertTrue(anchors.empty)
            labels = build_target_labels(positions, executions, HOUR, [ExecutionCoverage(timestamp(0), timestamp(24), "test")], invalid)
            self.assertEqual(labels.iloc[19].target_label, "LONG_DECREASE")
            self.assertEqual(labels.iloc[21].target_label, "MIXED")
            uncovered = build_target_labels(positions.iloc[[23]], executions, HOUR, [])
            self.assertEqual(uncovered.iloc[0].target_label, "MISSING")


    def test_evidence_is_unknown_without_manifest_and_is_source_bound(self):
        self.assertFalse(AccountEvidence().continuous("XBTUSD", timestamp(0), timestamp(1)))
        evidence = verified_evidence(((0, 2), (2, 4)))
        self.assertTrue(evidence.continuous("XBTUSD", timestamp(1), timestamp(4)))
        self.assertFalse(evidence.continuous("ETHUSD", timestamp(1), timestamp(2)))
        self.assertFalse(evidence.continuous("XBTUSD", timestamp(1), timestamp(5)))
        evidence.account = "another-account"
        self.assertFalse(evidence.continuous("XBTUSD", timestamp(1), timestamp(2)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, artifact = root / "account.csv", root / "manifest.txt"
            source.write_text("account data")
            artifact.write_text("synthetic test evidence only")
            records = verified_evidence().records
            for kind in ("coverage", "timestamps"):
                for record in records[kind]:
                    record.update(artifact="manifest.txt", artifact_sha256=file_sha256(artifact),
                                  source_files={str(source): file_sha256(source)}, basis="engineering fixture",
                                  timezone="UTC", fields=["transacttime", "timestamp"])
            path = root / "evidence.json"
            path.write_text(json.dumps({"schema_version": VALIDITY_POLICY, **records}))
            self.assertTrue(load_account_evidence(path, {str(source): file_sha256(source)}).continuous("XBTUSD", timestamp(1), timestamp(2)))
            with self.assertRaisesRegex(ValueError, "source binding"):
                load_account_evidence(path, {str(source): "changed"})
            artifact.write_text("changed")
            with self.assertRaisesRegex(ValueError, "artifact checksum"):
                load_account_evidence(path, {str(source): file_sha256(source)})

    def test_no_fill_requires_coverage_and_fill_does_not_prove_position(self):
        decisions = pd.DataFrame({"symbol": ["XBTUSD"] * 3, "decision_time_utc": [timestamp(h) for h in (0, 1, 2)],
                                  "position_before_qty": pd.array([0, None, 0], dtype="Int64"),
                                  "position_status": ["KNOWN", "UNKNOWN", "KNOWN"], "market_feature_status": "OK"})
        fill = pd.DataFrame([{name: None for name in EXECUTION_COLUMNS} | {
            "symbol": "XBTUSD", "execid": "unique", "orderid": ZERO_UUID, "exectype": "Trade", "side": "Buy",
            "lastqty": 1, "lastpx": 100, "cumqty": 1, "ordstatus": "Filled",
            "event_time_utc": timestamp(1), "processed_time_utc": timestamp(1), "trade_signed_qty": 1}])
        unknown = build_validated_labels(decisions, fill, AccountEvidence(), set())
        self.assertEqual(unknown.target_label.tolist(), ["MISSING", "POSITION_UNKNOWN", "MISSING"])
        self.assertFalse(unknown.training_eligible.any())
        covered = build_validated_labels(decisions, fill, verified_evidence(((0, 2),)), set())
        self.assertEqual(covered.target_label.tolist(), ["NO_FILL", "POSITION_UNKNOWN", "MISSING"])
        self.assertEqual(covered.training_eligible.tolist(), [True, False, False])
        self.assertEqual(covered.iloc[1].target_fill_count, 1)
        self.assertEqual(covered.iloc[1].target_order_ids, "")
        self.assertEqual(covered.iloc[1].target_unidentified_order_fill_count, 1)
        decisions.loc[1, ["position_before_qty", "position_status"]] = [0, "KNOWN"]
        covered = build_validated_labels(decisions, fill, verified_evidence(((0, 2),)), set())
        self.assertTrue(covered.iloc[1].training_eligible)
        self.assertEqual(covered.iloc[1].target_label, "LONG_INCREASE")
        no_timezone = verified_evidence(((0, 2),))
        no_timezone.records["timestamps"] = []
        withheld = build_validated_labels(decisions, fill, no_timezone, set())
        self.assertFalse(withheld.training_eligible.any())
        self.assertIn("TIMESTAMP_PROVENANCE_UNKNOWN", withheld.iloc[0].exclusion_reasons)

    def test_gaps_invalidate_state_and_delayed_snapshot_restores_it(self):
        rows = []
        for hour in (0, 8, 16, 24, 32, 40):
            rows.append(dict.fromkeys(EXECUTION_COLUMNS) | {
                "execid": f"f{hour}", "symbol": "XBTUSD", "exectype": "Funding",
                "lastqty": 10, "foreignnotional": -10,
                "transacttime": timestamp(hour).isoformat(), "timestamp": timestamp(hour).isoformat()})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(rows).to_csv(root / "aoa-execution-2020-01-01-2020-01-03.csv", index=False)
            grid = decision_grid(["XBTUSD"], timestamp(0), timestamp(44), HOUR)
            evidence = verified_evidence(((0, 18), (24, 44)), ((24, 26, 10),))
            executions, positions, _, report = reconstruct_account(root, grid, root / "audit", evidence)
            self.assertTrue(pd.isna(positions.iloc[16].position_before_qty))
            self.assertEqual(positions.iloc[17].position_before_qty, 10)
            self.assertTrue(positions.iloc[19:27].position_before_qty.isna().all())
            self.assertEqual(positions.iloc[27].position_before_qty, 10)
            anchors, detail = causal_position_anchors(executions, evidence)
            self.assertEqual(anchors.loc[anchors.anchor_source.str.startswith("FUNDING"), "event_time_utc"].tolist(), [timestamp(16), timestamp(40)])
            truncated, _ = causal_position_anchors(executions[executions.event_time_utc < timestamp(40)], evidence)
            self.assertNotIn(timestamp(24), truncated.loc[truncated.anchor_source.str.startswith("FUNDING"), "event_time_utc"].tolist())
            unknown = reconstruct_account(root, grid, root / "unknown")[1]
            self.assertTrue(unknown.position_before_qty.isna().all())
            self.assertEqual(report["accepted_causal_anchors"], 3)

    def test_later_funding_records_never_qualify_earlier_online_anchors(self):
        rows = pd.DataFrame({"symbol": ["XBTUSD"] * 4, "execid": ["a", "b", "c", "d"],
                             "exectype": "Funding", "lastqty": 10, "foreignnotional": -10,
                             "event_time_utc": [timestamp(h) for h in (0, 8, 16, 24)],
                             "processed_time_utc": [timestamp(h) for h in (0, 8, 20, 24)], "trade_signed_qty": 0})
        evidence = verified_evidence(((0, 32),))
        anchors, _ = causal_position_anchors(rows, evidence)
        self.assertEqual(anchors.event_time_utc.tolist(), [timestamp(16), timestamp(24)])
        self.assertEqual(anchors.iloc[0].available_at_utc, timestamp(20))
        prefix, _ = causal_position_anchors(rows.iloc[:3], evidence)
        pd.testing.assert_frame_equal(anchors.iloc[:1].reset_index(drop=True), prefix.reset_index(drop=True))


    def test_legacy_training_report_is_blocked_before_model_construction(self):
        script = Path(__file__).resolve().parents[1] / "scripts/train_candle_catboost.py"
        spec = importlib.util.spec_from_file_location("train_candle_guard", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "legacy.json"
            report.write_text(json.dumps({"data_contract_version": "candle_v1", "synthetic_data_used": False}))
            with mock.patch.object(sys, "argv", ["train", "--dataset-report", str(report), "--output-dir", str(root / "model")]), \
                    mock.patch.object(module, "CatBoostClassifier") as model, mock.patch.object(sys, "stderr"):
                self.assertEqual(module.main(), 2)
                model.assert_not_called()
            self.assertIn("verified account evidence policy", json.loads((root / "model/preflight_report.json").read_text())["error"])
        issues = class_coverage_issues({"train": pd.DataFrame({"target_label": ["NO_FILL"]})}, REQUIRED_ACTION_LABELS, 1)
        self.assertEqual({r["label"] for r in issues}, set(REQUIRED_ACTION_LABELS) - {"NO_FILL"})


    def test_new_snapshot_restores_state_after_older_unavailable_anchor(self):
        rows = []
        for hour in (0, 8, 16):
            rows.append(dict.fromkeys(EXECUTION_COLUMNS) | {
                "execid": f"f{hour}", "symbol": "XBTUSD", "exectype": "Funding",
                "lastqty": 10, "foreignnotional": -10,
                "transacttime": timestamp(hour).isoformat(),
                "timestamp": timestamp(100 if hour == 16 else hour).isoformat()})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(rows).to_csv(root / "aoa-execution-2020-01-01-2020-01-03.csv", index=False)
            grid = decision_grid(["XBTUSD"], timestamp(0), timestamp(28), HOUR)
            evidence = verified_evidence(((0, 28),), ((24, 24, 10),))
            positions = reconstruct_account(root, grid, root / "audit", evidence)[1]
            self.assertTrue(pd.isna(positions.iloc[23].position_before_qty))
            self.assertEqual(positions.iloc[25].position_before_qty, 10)
            self.assertEqual(positions.iloc[25].position_anchor_source, "VERIFIED_POSITION_SNAPSHOT")

    def test_ambiguous_and_mixed_windows_are_excluded_with_verified_evidence(self):
        decisions = pd.DataFrame({"symbol": ["XBTUSD"], "decision_time_utc": [timestamp(1)],
                                  "position_before_qty": [1], "position_status": ["KNOWN"], "market_feature_status": "OK"})
        rows = []
        for side in ("Buy", "Sell"):
            rows.append(dict.fromkeys(EXECUTION_COLUMNS) | {
                "symbol": "XBTUSD", "execid": side, "orderid": side, "exectype": "Trade", "side": side,
                "lastqty": 1, "lastpx": 100, "cumqty": 1, "ordstatus": "Filled",
                "event_time_utc": timestamp(1), "processed_time_utc": timestamp(1),
                "trade_signed_qty": 1 if side == "Buy" else -1})
        result = build_validated_labels(decisions, pd.DataFrame(rows), verified_evidence(), set())
        self.assertFalse(result.iloc[0].training_eligible)
        self.assertIn("TARGET_SEQUENCE_AMBIGUOUS", result.iloc[0].exclusion_reasons)
        self.assertIn("MIXED_ACTION", result.iloc[0].exclusion_reasons)


    def test_unreliable_wallet_records_do_not_block_account_positions(self):
        rows = []
        for hour in (0, 8, 16):
            rows.append(dict.fromkeys(EXECUTION_COLUMNS) | {
                "execid": f"f{hour}", "symbol": "XBTUSD", "exectype": "Funding",
                "lastqty": 10, "foreignnotional": -10,
                "transacttime": timestamp(hour).isoformat(), "timestamp": timestamp(hour).isoformat()})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(rows).to_csv(root / "aoa-execution-2020-01-01-2020-01-01.csv", index=False)
            grid = decision_grid(["XBTUSD"], timestamp(0), timestamp(24), HOUR)
            evidence = verified_evidence()
            before = reconstruct_account(root, grid, root / "without_wallet", evidence)
            (root / "aoa-wallet-2020-01-01-2020-01-01.csv").write_text(
                "date,timestamp,transacttime\n2020-01-01,57:26.3,unparseable\n")
            after = reconstruct_account(root, grid, root / "with_wallet", evidence)
            pd.testing.assert_frame_equal(before[1], after[1])
            self.assertEqual(before[3]["source_files"], after[3]["source_files"])
            self.assertEqual(after[1].iloc[17].position_before_qty, 10)


if __name__ == "__main__":
    unittest.main()
