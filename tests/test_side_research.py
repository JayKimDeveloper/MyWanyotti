"""Synthetic fixtures test research engineering only, never model performance."""
import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wonyotti_pipeline.candles import build_candle_features, load_candle_model
from wonyotti_pipeline.side_research import (
    FEATURE_COLUMNS, HOUR, RUNTIME_CONTRACT, build_dataset,
    prepare_model_features, research_splits, feature_schema, load_research_model,
)


def timestamp(hour):
    return pd.Timestamp("2020-01-01T00:00:00Z") + hour * HOUR


def candles():
    return pd.DataFrame({"candle_end_utc": [timestamp(h) for h in range(40)],
                         "close": np.arange(100., 140.), "high": np.arange(101., 141.),
                         "low": np.arange(99., 139.), "volume": 10.})


class SideResearchTests(unittest.TestCase):
    def test_versioned_schema_and_loader_preserve_v1_and_reject_wrong_version(self):
        version = "candle_side_research_v2_365d"
        schema = feature_schema(version)
        self.assertEqual(schema["columns"], feature_schema()["columns"])
        self.assertEqual(schema["features"], feature_schema()["features"])
        self.assertEqual(feature_schema()["data_contract_version"], "candle_side_research_v1")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "runtime_contract.json").write_text(json.dumps(dict(RUNTIME_CONTRACT, data_contract_version=version)))
            (path / "feature_schema.json").write_text(json.dumps(feature_schema()))
            with self.assertRaisesRegex(ValueError, "runtime contract mismatch"):
                load_research_model(path)
            with self.assertRaisesRegex(ValueError, "feature schema mismatch"):
                load_research_model(path, version)

    def test_labels_include_start_exclude_end_and_ignore_funding(self):
        decisions = pd.DataFrame({"symbol": "XBTUSD", "decision_time_utc": [timestamp(h) for h in (27, 28, 29, 30)]})
        events = pd.DataFrame({"symbol": "XBTUSD", "event_time_utc": [timestamp(h) for h in (27, 27, 28, 29, 30)],
                               "exectype": ["Trade", "Trade", "Trade", "Funding", "Trade"],
                               "side": ["Buy", "Sell", "Buy", None, "Sell"],
                               "orderid": ["a", "b", "c", None, "d"], "lastqty": 1, "lastpx": 100})
        rows = build_dataset(decisions, candles(), events)
        self.assertEqual(rows.target_label.tolist(), ["MIXED", "BUY_ONLY", "NO_RECORDED_FILL", "SELL_ONLY"])
        self.assertEqual(rows.target_fill_count.tolist(), [2, 1, 0, 1])
        self.assertTrue(rows.research_eligible.all())
        self.assertFalse(any("position" in c for c in FEATURE_COLUMNS))
        self.assertTrue(RUNTIME_CONTRACT["research_only"])
        self.assertFalse(RUNTIME_CONTRACT["compatible_with_five_action_predictor"])

    def test_shared_features_ignore_future_and_match_existing_market_columns(self):
        decisions = pd.DataFrame({"symbol": ["XBTUSD"], "decision_time_utc": [timestamp(27)]})
        events = pd.DataFrame(columns=["symbol", "event_time_utc", "exectype", "side", "orderid", "lastqty", "lastpx"])
        events.event_time_utc = pd.to_datetime(events.event_time_utc, utc=True)
        source = candles()
        before = build_dataset(decisions, source, events)
        source.loc[source.candle_end_utc >= timestamp(27), ["close", "volume", "high", "low"]] = 9999
        after = build_dataset(decisions, source, events)
        pd.testing.assert_frame_equal(prepare_model_features(before), prepare_model_features(after))
        existing = decisions.assign(position_before_qty=pd.array([pd.NA], dtype="Int64"), entry_price=np.nan,
                                    entry_asof_time_utc=pd.NaT, holding_hours=np.nan)
        existing.entry_asof_time_utc = pd.to_datetime(existing.entry_asof_time_utc, utc=True)
        shared = build_candle_features(existing, candles())
        pd.testing.assert_frame_equal(prepare_model_features(before), prepare_model_features(shared))
        self.assertEqual(before.iloc[0].feature_start_time_utc, timestamp(1))
        self.assertEqual(before.iloc[0].candle_end_utc, timestamp(26))
        self.assertEqual(prepare_model_features(before).dtypes.unique().tolist(), [np.dtype('float64')])
        with self.assertRaises(ValueError):
            prepare_model_features(before.drop(columns=FEATURE_COLUMNS[0]))
        source = candles()
        gap = build_dataset(decisions, source[source.candle_end_utc != timestamp(10)], events)
        self.assertFalse(gap.iloc[0].research_eligible)

    def test_feature_and_label_dependencies_do_not_overlap_across_splits(self):
        frame = pd.DataFrame({"decision_time_utc": [timestamp(h) for h in range(120)],
                              "target_end_time_utc": [timestamp(h + 1) for h in range(120)],
                              "dependency_end_time_utc": [timestamp(h + 1) for h in range(120)],
                              "feature_start_time_utc": [timestamp(h - 26) for h in range(120)],
                              "target_order_ids": "", "target_label": "NO_RECORDED_FILL"})
        splits, report = research_splits(frame, timestamp(40), timestamp(80))
        self.assertTrue(report["leakage_checks"]["passed"])
        self.assertEqual(report["feature_dependency_purged_counts"], {"validation": 26, "test": 26})
        self.assertEqual(report["retained_counts"], {"train": 40, "validation": 14, "test": 14})
        frame.loc[0, "target_order_ids"] = "shared"
        frame.loc[106, "target_order_ids"] = "shared"
        _, report = research_splits(frame, timestamp(40), timestamp(80))
        self.assertEqual(report["order_purged_row_count"], 2)


if __name__ == "__main__":
    unittest.main()
