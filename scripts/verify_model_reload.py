#!/usr/bin/env python3
"""Compare a pre-save probability probe with a newly loaded model process."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wonyotti_pipeline.inference import (  # noqa: E402
    CatBoostActionPredictor,
    InferenceContractError,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--rtol", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        probe_path = args.model_dir / "reload_probe.json"
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        predictor = CatBoostActionPredictor(args.model_dir)
        expected_classes = [str(value) for value in probe["class_order"]]
        if expected_classes != predictor.class_labels:
            raise InferenceContractError(
                "reload probe class order differs from loaded model.classes_"
            )
        expected = np.array(
            [probe["pre_save_probabilities"][label] for label in expected_classes],
            dtype="float64",
        )
        actual = predictor.predict_probabilities(probe["features"])
        differences = np.abs(expected - actual)
        passed = bool(np.allclose(expected, actual, atol=args.atol, rtol=args.rtol))
        report = {
            "status": "PASS" if passed else "FAIL",
            "new_process_model_load": True,
            "class_order_matches": True,
            "atol": args.atol,
            "rtol": args.rtol,
            "max_absolute_difference": float(differences.max()),
            "class_order": predictor.class_labels,
            "pre_save_probabilities": {
                label: float(expected[index])
                for index, label in enumerate(expected_classes)
            },
            "reloaded_probabilities": {
                label: float(actual[index])
                for index, label in enumerate(expected_classes)
            },
            "model_load_seconds": predictor.load_seconds,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if passed else 2
    except (OSError, ValueError, KeyError, json.JSONDecodeError, InferenceContractError) as error:
        print(
            json.dumps(
                {"status": "ERROR", "error": str(error)}, ensure_ascii=False
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
