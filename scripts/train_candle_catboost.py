#!/usr/bin/env python3
"""Fit the separate candle_v1 CPU model on chronological, purged splits."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wonyotti_pipeline.candles import (
    CONTRACT_VERSION, FEATURE_COLUMNS, RUNTIME_CONTRACT, atomic_json,
    atomic_parquet, feature_record, feature_schema, prepare_model_features,
)
from wonyotti_pipeline.account_evidence import VALIDITY_POLICY
from wonyotti_pipeline.training import (
    REQUIRED_ACTION_LABELS, chronological_split, class_coverage_issues,
    class_distribution, classification_metrics, choose_validation_threshold,
    data_version, file_sha256, majority_label, max_rss_bytes, split_leakage_checks,
    threshold_predictions,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/training/candle_v1_evidence_v1/training_eligible.parquet"))
    parser.add_argument("--dataset-report", type=Path, default=Path("data/audit/candle_v1_evidence_v1/training_dataset_report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/catboost/candle_v1"))
    parser.add_argument("--train-end", default="2021-01-01T00:00:00Z")
    parser.add_argument("--validation-end", default="2021-07-01T00:00:00Z")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--thread-count", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if (args.output_dir / "model.cbm").exists():
        parser.error("model already exists; verify it or use a new versioned output directory")
    try:
        report = json.loads(args.dataset_report.read_text())
        if report.get("data_contract_version") != CONTRACT_VERSION or report.get("synthetic_data_used") is not False:
            raise ValueError("actual candle_v1 dataset report required")
        if report.get("validity_policy") != VALIDITY_POLICY:
            raise ValueError("verified account evidence policy required; legacy candle datasets are not eligible")
        if report.get("runtime_contract") != RUNTIME_CONTRACT:
            raise ValueError("dataset runtime contract mismatch")
        frame = pd.read_parquet(args.input).sort_values("decision_time_utc").reset_index(drop=True)
        required = {"symbol", "decision_time_utc", "target_end_time_utc", "target_order_ids", "target_label",
                    "target_first_event_time_utc", "target_last_event_time_utc", "candle_end_utc", "position_asof_time_utc", "training_eligible", "validity_policy", "target_coverage_status",
                    "timestamp_provenance_status", "position_anchor_available_at_utc", "exclusion_reasons", *FEATURE_COLUMNS}
        if not required.issubset(frame):
            raise ValueError(f"missing input columns: {sorted(required - set(frame))}")
        if frame.empty or not frame["training_eligible"].eq(True).all() or not frame["symbol"].eq("XBTUSD").all():
            raise ValueError("nonempty eligible XBTUSD input required")
        if len(frame) != report["eligible_rows"] or frame["target_label"].value_counts().to_dict() != report["eligible_label_counts"]:
            raise ValueError("input counts do not match dataset report")
        if (not frame.validity_policy.eq(VALIDITY_POLICY).all()
                or not frame.target_coverage_status.eq("VERIFIED").all()
                or not frame.timestamp_provenance_status.eq("VERIFIED").all()
                or not frame.exclusion_reasons.eq("").all()
                or frame.position_anchor_available_at_utc.isna().any()
                or frame.position_anchor_available_at_utc.ge(frame.decision_time_utc).any()):
            raise ValueError("input violates account evidence or anchor availability policy")
        prepare_model_features(frame)
        checks = {
            "duplicate_keys": int(frame.duplicated(["symbol", "decision_time_utc"]).sum()),
            "future_candles": int(frame["candle_end_utc"].ge(frame["decision_time_utc"]).sum()),
            "future_positions": int(frame["position_asof_time_utc"].ge(frame["decision_time_utc"]).sum()),
            "early_targets": int(frame["target_first_event_time_utc"].lt(frame["decision_time_utc"]).sum()),
            "late_targets": int(frame["target_last_event_time_utc"].ge(frame["target_end_time_utc"]).sum()),
            "invalid_target_intervals": int((frame["target_end_time_utc"] - frame["decision_time_utc"]).ne(pd.Timedelta(hours=1)).sum()),
            "null_time_rows": int(frame[["decision_time_utc", "target_end_time_utc", "candle_end_utc", "position_asof_time_utc"]].isna().any(axis=1).sum()),
            "unknown_labels": sorted(set(frame["target_label"]) - set(REQUIRED_ACTION_LABELS)),
        }
        if any(checks.values()):
            raise ValueError(f"input contract/leakage checks failed: {checks}")
        train_end, validation_end = pd.Timestamp(args.train_end), pd.Timestamp(args.validation_end)
        if train_end.tzinfo is None or validation_end.tzinfo is None or train_end >= validation_end:
            raise ValueError("ascending timezone-aware split boundaries required")
        splits, split_report = chronological_split(frame, train_end, validation_end)
        embargo_counts = {}
        for name, boundary in (("validation", train_end), ("test", validation_end)):
            original = len(splits[name])
            splits[name] = splits[name][splits[name]["decision_time_utc"] >= boundary + pd.Timedelta(hours=26)].reset_index(drop=True)
            embargo_counts[name] = original - len(splits[name])
        split_checks = split_leakage_checks(splits, train_end, validation_end)
        coverage = class_coverage_issues({name: splits[name] for name in ("train", "validation")}, REQUIRED_ACTION_LABELS, 1)
        split_report.update(embargo="26h after each boundary: 25 candle hours plus 1h delay", embargo_counts=embargo_counts,
                            retained_counts={name: len(d) for name, d in splits.items()}, leakage_checks=split_checks,
                            selection_class_distributions={name: class_distribution(splits[name]) for name in ("train", "validation")},
                            selection_class_coverage_issues=coverage)
        atomic_json(args.output_dir / "data_split.json", split_report)
        if coverage or not split_checks["passed"] or any(d.empty for d in splits.values()):
            raise ValueError(f"split preflight failed: coverage={coverage}; leakage={split_checks}")
        params = {"task_type": "CPU", "iterations": args.iterations, "depth": args.depth,
                  "thread_count": args.thread_count, "random_seed": 42, "loss_function": "MultiClass",
                  "eval_metric": "MultiClass", "auto_class_weights": "Balanced", "allow_writing_files": False}
        train, validation, test = (splits[name] for name in ("train", "validation", "test"))
        model = CatBoostClassifier(**params)
        fit_started = time.perf_counter()
        model.fit(prepare_model_features(train), train["target_label"],
                  eval_set=(prepare_model_features(validation), validation["target_label"]),
                  use_best_model=True, early_stopping_rounds=30, verbose=50)
        fit_seconds = time.perf_counter() - fit_started
        labels = [str(label) for label in model.classes_]
        if set(labels) != set(REQUIRED_ACTION_LABELS):
            raise ValueError("model class coverage mismatch")
        baseline = majority_label(train["target_label"])
        probabilities = model.predict_proba(prepare_model_features(validation))
        threshold, candidates = choose_validation_threshold(probabilities, validation["target_label"], labels, baseline, 0.05)
        config = {"data_contract_version": CONTRACT_VERSION, "requested_params": params,
                  "resolved_params": model.get_all_params(), "early_stopping_rounds": 30,
                  "best_iteration": int(model.get_best_iteration()), "class_labels": labels,
                  "selected_threshold": threshold, "majority_baseline_label": baseline,
                  "threshold_candidates": candidates, "test_used_for_selection": False}
        evaluation = {"status": "COMPLETE", "synthetic_data_used": False,
                      "performance_scope": "historical account behavior imitation; no trading/PnL claim",
                      "class_order": labels, "model": {}, "majority_baseline": {},
                      "argmax_model": {}, "fit_seconds": fit_seconds, "catboost_version": catboost.__version__}
        predictions = []
        for name, data in (("validation", validation), ("test", test)):
            probs = probabilities if name == "validation" else model.predict_proba(prepare_model_features(data))
            actions = threshold_predictions(probs, labels, threshold, baseline)
            evaluation["model"][name] = classification_metrics(data["target_label"], actions, labels)
            evaluation["argmax_model"][name] = classification_metrics(data["target_label"], [labels[i] for i in probs.argmax(axis=1)], labels)
            evaluation["majority_baseline"][name] = classification_metrics(data["target_label"], [baseline] * len(data), labels)
            output = data[["symbol", "decision_time_utc", "target_label", "target_order_ids"]].copy()
            output["split"], output["predicted_action"] = name, actions
            for i, label in enumerate(labels):
                output[f"probability_{label}"] = probs[:, i]
            predictions.append(output)
        split_report["class_distributions"] = {name: class_distribution(data) for name, data in splits.items()}
        probe = {"generated_before_model_save": True, "source_split": "validation", "features": feature_record(validation.iloc[0]),
                 "class_order": labels, "pre_save_probabilities": {label: float(probabilities[0, i]) for i, label in enumerate(labels)}}
        atomic_json(args.output_dir / "reload_probe.json", probe)
        temporary = args.output_dir / "model.cbm.tmp"
        model.save_model(temporary)
        temporary.replace(args.output_dir / "model.cbm")
        atomic_json(args.output_dir / "feature_schema.json", feature_schema())
        atomic_json(args.output_dir / "class_labels.json", {"source": "model.classes_", "class_labels": labels})
        atomic_json(args.output_dir / "training_config.json", config)
        atomic_json(args.output_dir / "runtime_contract.json", RUNTIME_CONTRACT)
        version = data_version(args.input, frame)
        version.update(data_contract_version=CONTRACT_VERSION, dataset_report_sha256=file_sha256(args.dataset_report), dataset_report=report)
        atomic_json(args.output_dir / "data_version.json", version)
        atomic_json(args.output_dir / "data_split.json", split_report)
        atomic_parquet(args.output_dir / "evaluation_predictions.parquet", pd.concat(predictions, ignore_index=True))
        evaluation.update(total_seconds=time.perf_counter() - started, max_rss_bytes=max_rss_bytes())
        atomic_json(args.output_dir / "evaluation_report.json", evaluation)
        atomic_json(args.output_dir / "preflight_report.json", {"status": "PASS", "input_checks": checks, "split_checks": split_checks})
        print(json.dumps({"model_dir": str(args.output_dir), "split_rows": split_report["retained_counts"],
                          "validation_macro_f1": evaluation["model"]["validation"]["macro_f1"],
                          "test_macro_f1": evaluation["model"]["test"]["macro_f1"]}, indent=2))
        return 0
    except (OSError, ValueError) as error:
        atomic_json(args.output_dir / "preflight_report.json", {"status": "BLOCKED", "error": str(error),
                    "resume_command": ".venv/bin/python scripts/train_candle_catboost.py"})
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
