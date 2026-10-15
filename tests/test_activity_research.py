"""Synthetic engineering checks only; actual metrics use saved observations."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wonyotti_pipeline.activity_research import (
    CONTRACT, binary_labels, activity_curve, choose_activity_threshold,
    activity_metrics, load_activity_model,
)
from wonyotti_pipeline.side_research import RUNTIME_CONTRACT


class ActivityResearchTests(unittest.TestCase):
    def test_target_mapping_preserves_unknown_label_rejection(self):
        mapped = binary_labels(pd.Series(["BUY_ONLY", "SELL_ONLY", "MIXED", "NO_RECORDED_FILL"]))
        self.assertEqual(mapped.tolist(), ["ACTIVE_RECORDED"] * 3 + ["NO_RECORDED_FILL"])
        with self.assertRaises(ValueError):
            binary_labels(pd.Series([None]))
        self.assertTrue(CONTRACT["research_only"])
        self.assertFalse(CONTRACT["compatible_with_five_action_predictor"])
        self.assertEqual(RUNTIME_CONTRACT["data_contract_version"], "candle_side_research_v1")

    def test_average_precision_groups_ties_and_constant_baseline(self):
        self.assertAlmostEqual(activity_curve([1, 0, 1], [.9, .8, .7])[1], 5 / 6)
        self.assertAlmostEqual(activity_curve([1, 0, 1], [.5, .5, .5])[1], 2 / 3)
        self.assertIsNone(activity_curve([0, 0], [.2, .3])[1])
        with self.assertRaises(ValueError):
            activity_curve([1], [np.nan])

    def test_threshold_selection_and_boundary_inclusion(self):
        threshold = choose_activity_threshold([1, 0, 1], [.9, .8, .7])
        self.assertEqual(threshold, .7)
        metrics = activity_metrics([1, 0, 1], [.9, .8, .7], threshold)
        self.assertEqual(metrics["flagged_count"], 3)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertAlmostEqual(metrics["binary_f1"], .8)
        with self.assertRaises(ValueError):
            choose_activity_threshold([0, 0], [.1, .2])

    def test_equal_f1_prefers_highest_threshold(self):
        # With three positives, flagging 1, 5, or 9 rows all gives F1=0.5.
        self.assertEqual(choose_activity_threshold([1, 0, 0, 0, 1, 0, 0, 0, 1],
                         [.9, .8, .7, .6, .5, .4, .3, .2, .1]), .9)

    def test_binary_loader_rejects_multiclass_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "runtime_contract.json").write_text(json.dumps(RUNTIME_CONTRACT))
            with self.assertRaisesRegex(ValueError, "runtime contract mismatch"):
                load_activity_model(path)


if __name__ == "__main__":
    unittest.main()
