#!/usr/bin/env python3
"""Load a CatBoost model once and emit action probabilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wonyotti_pipeline.inference import (  # noqa: E402
    CatBoostActionPredictor,
    InferenceContractError,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-json", type=Path)
    source.add_argument("--stdin-jsonl", action="store_true")
    return parser.parse_args()


def read_record(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise InferenceContractError("input JSON must contain one feature object")
    return value


def main() -> int:
    args = parse_args()
    try:
        predictor = CatBoostActionPredictor(args.model_dir)
        if args.input_json is not None:
            result = predictor.predict(read_record(args.input_json))
            result["model_load_seconds"] = predictor.load_seconds
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        for line_number, line in enumerate(sys.stdin, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise InferenceContractError(
                    f"stdin line {line_number} must contain a feature object"
                )
            result = predictor.predict(value)
            result["request_line"] = line_number
            print(json.dumps(result, ensure_ascii=False), flush=True)
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
