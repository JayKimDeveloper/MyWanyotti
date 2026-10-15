#!/usr/bin/env python3
"""Benchmark warmed single-row CatBoost inference without model reloads."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wonyotti_pipeline.inference import (  # noqa: E402
    CatBoostActionPredictor,
    InferenceContractError,
)
from wonyotti_pipeline.training import max_rss_bytes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup < 1 or args.iterations < 1:
        print(
            json.dumps(
                {"status": "ERROR", "error": "warmup and iterations must be positive"}
            ),
            file=sys.stderr,
        )
        return 2
    try:
        record = json.loads(args.input_json.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise InferenceContractError("input JSON must contain one feature object")
        predictor = CatBoostActionPredictor(args.model_dir)
        for _ in range(args.warmup):
            predictor.predict_probabilities(record)

        durations_ns = np.empty(args.iterations, dtype="int64")
        for index in range(args.iterations):
            started_at = time.perf_counter_ns()
            predictor.predict_probabilities(record)
            durations_ns[index] = time.perf_counter_ns() - started_at
        durations_us = durations_ns / 1000.0
        report = {
            "status": "OK",
            "measurement_scope": (
                "schema/type validation + shared feature transformation + "
                "single-row predict_proba; model loading excluded"
            ),
            "model_loaded_once": True,
            "warmup_calls": args.warmup,
            "measured_calls": args.iterations,
            "latency_unit": "microseconds",
            "latency": {
                "median": float(np.percentile(durations_us, 50)),
                "p95": float(np.percentile(durations_us, 95)),
                "p99": float(np.percentile(durations_us, 99)),
                "minimum": float(durations_us.min()),
                "maximum": float(durations_us.max()),
            },
            "model_load_seconds_excluded": predictor.load_seconds,
            "process_max_rss_bytes": max_rss_bytes(),
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, json.JSONDecodeError, InferenceContractError) as error:
        print(
            json.dumps(
                {"status": "ERROR", "error": str(error)}, ensure_ascii=False
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
