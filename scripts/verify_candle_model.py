#!/usr/bin/env python3
"""Verify candle_v1 probability consistency in a newly started process."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wonyotti_pipeline.candles import atomic_json, load_candle_model, prepare_model_features


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("models/catboost/candle_v1"))
    args = parser.parse_args()
    try:
        probe = json.loads((args.model_dir / "reload_probe.json").read_text())
        model, labels = load_candle_model(args.model_dir)
        if labels != probe["class_order"]:
            raise ValueError("reload probe class order mismatch")
        actual = model.predict_proba(prepare_model_features(pd.DataFrame([probe["features"]])))[0]
        expected = np.array([probe["pre_save_probabilities"][label] for label in labels])
        passed = bool(np.allclose(actual, expected, atol=1e-12, rtol=0))
        report = {"status": "PASS" if passed else "FAIL", "new_process_model_load": True,
                  "shared_feature_preparation": "wonyotti_pipeline.candles.prepare_model_features",
                  "class_order": labels, "max_absolute_difference": float(np.abs(actual - expected).max()),
                  "atol": 1e-12, "rtol": 0}
        atomic_json(args.model_dir / "reload_verification.json", report)
        print(json.dumps(report, indent=2))
        return 0 if passed else 2
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
