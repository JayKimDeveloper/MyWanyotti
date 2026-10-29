"""Focused synthetic checks for saved-score bootstrap engineering only."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bootstrap_activity_holdout import monday_blocks, measured, paired_bootstrap, METRICS, THRESHOLDS


class ActivityBootstrapTests(unittest.TestCase):
    def test_utc_monday_boundary_and_partial_blocks(self):
        times = pd.Series(pd.to_datetime(["2019-03-10T23:00:00Z", "2019-03-11T00:00:00Z", "2019-03-11T01:00:00Z"]))
        starts, blocks = monday_blocks(times)
        self.assertEqual(starts, [pd.Timestamp("2019-03-04T00:00:00Z"), pd.Timestamp("2019-03-11T00:00:00Z")])
        self.assertEqual([rows.tolist() for rows in blocks], [[0], [1, 2]])
        with self.assertRaises(ValueError):
            monday_blocks(times.dt.tz_localize(None))

    def test_threshold_equality_keeps_greater_or_equal_operator(self):
        for threshold in THRESHOLDS.values():
            metric = measured([True, False], [threshold, np.nextafter(threshold, -np.inf)], threshold)
            self.assertEqual(metric["precision"], 1)
            self.assertEqual(metric["recall"], 1)
            self.assertEqual(metric["binary_f1"], 1)
            self.assertEqual(metric["false_positives_per_1000"], 0)

    def test_undefined_metrics_are_not_imputed(self):
        none = measured([False], [.1], .5)
        self.assertTrue(all(none[key] is None for key in METRICS[:-1]))
        self.assertEqual(none["false_positives_per_1000"], 0)
        false_alarm = measured([False], [.9], .5)
        self.assertIsNone(false_alarm["average_precision"])
        self.assertIsNone(false_alarm["recall"])
        self.assertEqual(false_alarm["precision"], 0)
        self.assertEqual(false_alarm["binary_f1"], 0)
        self.assertEqual(false_alarm["false_positives_per_1000"], 1000)

    def test_paired_sampling_and_seed_reproducibility(self):
        truth = np.array([False, True, True, False])
        scores = {"v3_unweighted": np.array([.1, .5, .1, .5]), "v2_activity": np.array([.2, .8, .2, .8])}
        blocks = [np.array([0, 1]), np.array([2, 3])]
        result = paired_bootstrap(truth, scores, blocks, 20, 42)
        self.assertEqual(result, paired_bootstrap(truth, scores, blocks, 20, 42))
        for metric in result["metrics"].values():
            self.assertEqual(metric["percentile_95_interval"], [0, 0])
            self.assertEqual(metric["valid_replications"], 20)

    def test_invalid_pair_counts_are_reported(self):
        scores = {"v3_unweighted": np.array([.1]), "v2_activity": np.array([.1])}
        result = paired_bootstrap(np.array([False]), scores, [np.array([0])], 3, 42)
        self.assertEqual(result["metrics"]["average_precision"]["invalid_replications"], 3)
        self.assertIsNone(result["metrics"]["average_precision"]["percentile_95_interval"])
        self.assertEqual(result["metrics"]["false_positives_per_1000"]["valid_replications"], 3)


if __name__ == "__main__":
    unittest.main()
