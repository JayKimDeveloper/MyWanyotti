#!/usr/bin/env python3
"""Train and evaluate CatBoost on a leakage-safe chronological split."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wonyotti_pipeline.training import (  # noqa: E402
    REQUIRED_ACTION_LABELS,
    TrainingDataError,
    choose_split_boundaries,
    choose_validation_threshold,
    chronological_split,
    class_coverage_issues,
    class_distribution,
    classification_metrics,
    data_version,
    feature_schema,
    load_training_data,
    majority_label,
    max_rss_bytes,
    prepare_model_features,
    serialize_feature_row,
    split_leakage_checks,
    threshold_predictions,
    validate_input_contract,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/catboost/first_experiment")
    )
    parser.add_argument("--dataset-report", type=Path)
    parser.add_argument("--train-end")
    parser.add_argument("--validation-end")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--minimum-class-count", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--thread-count", type=int, default=-1)
    parser.add_argument(
        "--auto-class-weights",
        choices=["Balanced", "SqrtBalanced", "None"],
        default="Balanced",
    )
    parser.add_argument("--threshold-grid-step", type=float, default=0.05)
    parser.add_argument("--verbose", type=int, default=50)
    return parser.parse_args()


def utc_timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def split_period(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "row_count": 0,
            "decision_start_utc": None,
            "decision_end_utc": None,
            "target_end_utc": None,
        }
    return {
        "row_count": int(len(frame)),
        "decision_start_utc": frame["decision_time_utc"].min().isoformat(),
        "decision_end_utc": frame["decision_time_utc"].max().isoformat(),
        "target_end_utc": frame["target_end_time_utc"].max().isoformat(),
    }


def model_predictions(
    model: CatBoostClassifier,
    features: pd.DataFrame,
    class_labels: list[str],
    threshold: float,
    fallback_label: str,
) -> tuple[np.ndarray, list[str]]:
    probabilities = model.predict_proba(features)
    predictions = threshold_predictions(
        probabilities, class_labels, threshold, fallback_label
    )
    return probabilities, predictions


def prediction_frame(
    source: pd.DataFrame,
    split_name: str,
    probabilities: np.ndarray,
    predictions: list[str],
    class_labels: list[str],
) -> pd.DataFrame:
    output = source[
        ["symbol", "decision_time_utc", "target_end_time_utc", "target_label"]
    ].copy()
    output.insert(0, "split", split_name)
    output["prediction"] = predictions
    output["prediction_confidence"] = probabilities.max(axis=1)
    for index, label in enumerate(class_labels):
        output[f"probability__{label}"] = probabilities[:, index]
    return output


def blocked_report(
    output_dir: Path,
    error: str,
    started_at: float,
    details: dict[str, object] | None = None,
) -> int:
    report = {
        "status": "BLOCKED_INSUFFICIENT_OR_INVALID_DATA",
        "error": error,
        "details": details or {},
        "data_provenance": "UNVERIFIED_OR_REPORTED_IN_DETAILS",
        "model_trained": False,
        "performance_claimed": False,
        "elapsed_seconds": time.perf_counter() - started_at,
        "process_max_rss_bytes": max_rss_bytes(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "preflight_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), file=sys.stderr)
    return 2


def main() -> int:
    args = parse_args()
    started_at = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.minimum_class_count < 1:
        return blocked_report(
            args.output_dir,
            "minimum_class_count must be at least one",
            started_at,
        )

    try:
        frame, input_counts = load_training_data(args.input)
        if frame.empty:
            raise TrainingDataError("no training_eligible rows remain")
        input_checks = validate_input_contract(frame)
        train_end, validation_end, boundary_source = choose_split_boundaries(
            frame,
            utc_timestamp(args.train_end),
            utc_timestamp(args.validation_end),
            args.train_ratio,
            args.validation_ratio,
        )
        splits, split_report = chronological_split(
            frame, train_end, validation_end
        )
        split_checks = split_leakage_checks(splits, train_end, validation_end)
        selection_splits = {
            name: splits[name] for name in ("train", "validation")
        }
        distributions = {
            name: class_distribution(split)
            for name, split in selection_splits.items()
        }
        coverage_issues = class_coverage_issues(
            selection_splits, REQUIRED_ACTION_LABELS, args.minimum_class_count
        )
    except (OSError, ValueError, TrainingDataError) as error:
        return blocked_report(args.output_dir, str(error), started_at)

    split_metadata = {
        **input_counts,
        **split_report,
        "boundary_source": boundary_source,
        "train_ratio_requested": args.train_ratio,
        "validation_ratio_requested": args.validation_ratio,
        "periods": {name: split_period(split) for name, split in splits.items()},
        "selection_class_distributions": distributions,
        "selection_class_coverage_issues": coverage_issues,
        "minimum_class_count": args.minimum_class_count,
        "split_leakage_checks": split_checks,
    }
    version = data_version(args.input, frame)
    version.update(input_counts)
    provenance = {
        "dataset_report_provided": args.dataset_report is not None,
        "synthetic_data_status": "UNVERIFIED",
        "data_contract_status": "UNVERIFIED",
    }
    source_report: dict[str, object] | None = None
    if args.dataset_report is not None:
        if not args.dataset_report.exists():
            return blocked_report(
                args.output_dir,
                f"dataset report not found: {args.dataset_report}",
                started_at,
                {"data_split": split_metadata},
            )
        source_report = json.loads(
            args.dataset_report.read_text(encoding="utf-8")
        )
        synthetic_flag = source_report.get("synthetic_data_used")
        contract_version = source_report.get("data_contract_version")
        if synthetic_flag is False:
            provenance["synthetic_data_status"] = "VERIFIED_NON_SYNTHETIC"
        elif synthetic_flag is True:
            provenance["synthetic_data_status"] = "DECLARED_SYNTHETIC"
        provenance["data_contract_version"] = contract_version
        provenance["data_contract_status"] = (
            "MATCH" if contract_version == "v0.3" else "MISMATCH"
        )
        version["dataset_report_path"] = str(args.dataset_report.resolve())
        version["dataset_report"] = source_report
    version["provenance"] = provenance

    runtime_fields = ("exchange", "symbols", "interval", "lookback", "max_quote_age")
    runtime_contract = {
        "data_contract_version": "v0.3",
        "status": "UNVERIFIED_MISSING_DATASET_REPORT",
        "exchange": None,
        "symbols": [],
        "decision_interval": None,
        "lookback": None,
        "max_quote_age": None,
        "market_feature_function": "wonyotti_pipeline.market.build_market_features",
        "model_feature_function": "wonyotti_pipeline.training.prepare_model_features",
        "strict_asof_rule": "market_time_utc < decision_time_utc",
    }
    if source_report is not None:
        missing_runtime_fields = [
            field for field in runtime_fields if source_report.get(field) is None
        ]
        runtime_contract.update(
            {
                "status": (
                    "VERIFIED"
                    if not missing_runtime_fields
                    and provenance["synthetic_data_status"] == "VERIFIED_NON_SYNTHETIC"
                    and provenance["data_contract_status"] == "MATCH"
                    else "UNVERIFIED_OR_INCOMPLETE"
                ),
                "exchange": source_report.get("exchange"),
                "symbols": source_report.get("symbols", []),
                "decision_interval": source_report.get("interval"),
                "lookback": source_report.get("lookback"),
                "max_quote_age": source_report.get("max_quote_age"),
                "missing_fields": missing_runtime_fields,
            }
        )

    schema = feature_schema(frame)
    requested_config = {
        "task_type": "CPU",
        "iterations": args.iterations,
        "depth": args.depth,
        "random_seed": args.random_seed,
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "early_stopping_rounds": args.early_stopping_rounds,
        "thread_count": args.thread_count,
        "auto_class_weights": args.auto_class_weights,
        "threshold_policy": (
            "validation macro-F1 grid search; low confidence falls back to the "
            "training majority label"
        ),
        "threshold_grid_step": args.threshold_grid_step,
        "feature_selection": "FIXED_CONTRACT_ALLOWLIST",
        "test_used_for_selection": False,
    }
    write_json(args.output_dir / "feature_schema.json", schema)
    write_json(args.output_dir / "training_config.json", requested_config)
    write_json(args.output_dir / "data_split.json", split_metadata)
    write_json(args.output_dir / "data_version.json", version)
    write_json(args.output_dir / "runtime_contract.json", runtime_contract)

    blocking_coverage_issues = coverage_issues
    preflight = {
        "status": "READY" if not blocking_coverage_issues else "BLOCKED",
        "input_contract_checks": input_checks,
        "split_leakage_checks": split_checks,
        "class_coverage_issues": coverage_issues,
        "blocking_class_coverage_issues": blocking_coverage_issues,
        "required_action_labels": REQUIRED_ACTION_LABELS,
        "data_provenance": provenance,
    }
    write_json(args.output_dir / "preflight_report.json", preflight)
    if not input_checks["passed"]:
        return blocked_report(
            args.output_dir,
            "input contract or future-label leakage checks failed",
            started_at,
            preflight,
        )
    if not split_checks["passed"]:
        return blocked_report(
            args.output_dir,
            "split leakage checks failed",
            started_at,
            preflight,
        )
    if blocking_coverage_issues:
        return blocked_report(
            args.output_dir,
            "required action classes are missing from train or validation",
            started_at,
            preflight,
        )
    if provenance["synthetic_data_status"] == "DECLARED_SYNTHETIC":
        return blocked_report(
            args.output_dir,
            "dataset report declares synthetic data; production evaluation is blocked",
            started_at,
            preflight,
        )
    if provenance["data_contract_status"] == "MISMATCH":
        return blocked_report(
            args.output_dir,
            "dataset report does not match required data contract v0.3",
            started_at,
            preflight,
        )

    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]
    train_features = prepare_model_features(train)
    validation_features = prepare_model_features(validation)
    train_labels = train["target_label"].astype(str)
    validation_labels = validation["target_label"].astype(str)
    majority = majority_label(train_labels)
    class_weights = (
        None if args.auto_class_weights == "None" else args.auto_class_weights
    )

    model_params: dict[str, object] = {
        "iterations": args.iterations,
        "depth": args.depth,
        "random_seed": args.random_seed,
        "task_type": "CPU",
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "thread_count": args.thread_count,
        "allow_writing_files": False,
        "verbose": args.verbose,
    }
    if class_weights is not None:
        model_params["auto_class_weights"] = class_weights
    model = CatBoostClassifier(
        **model_params
    )
    rss_before_fit = max_rss_bytes()
    fit_started_at = time.perf_counter()
    model.fit(
        train_features,
        train_labels,
        cat_features=["symbol"],
        eval_set=(validation_features, validation_labels),
        use_best_model=True,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    fit_seconds = time.perf_counter() - fit_started_at

    class_labels = [str(label) for label in model.classes_]
    missing_model_classes = sorted(set(REQUIRED_ACTION_LABELS) - set(class_labels))
    if missing_model_classes:
        return blocked_report(
            args.output_dir,
            f"model.classes_ is missing required labels: {missing_model_classes}",
            started_at,
            {"model_classes": class_labels},
        )

    selection_started_at = time.perf_counter()
    validation_probabilities = model.predict_proba(validation_features)
    threshold, threshold_candidates = choose_validation_threshold(
        validation_probabilities,
        validation_labels,
        class_labels,
        majority,
        args.threshold_grid_step,
    )
    validation_predictions = threshold_predictions(
        validation_probabilities, class_labels, threshold, majority
    )
    selection_seconds = time.perf_counter() - selection_started_at

    frozen_selection = {
        "class_labels_source": "model.classes_",
        "class_labels": class_labels,
        "majority_baseline_label": majority,
        "selected_threshold": threshold,
        "threshold_candidates": threshold_candidates,
        "best_iteration": int(model.get_best_iteration()),
        "tree_count": int(model.tree_count_),
    }
    requested_config["frozen_selection"] = frozen_selection
    requested_config["resolved_catboost_params"] = model.get_all_params()
    write_json(args.output_dir / "training_config.json", requested_config)

    test_distribution = class_distribution(test)
    test_coverage_issues = class_coverage_issues(
        {"test": test}, REQUIRED_ACTION_LABELS, args.minimum_class_count
    )
    split_metadata["test_class_distribution_after_selection"] = test_distribution
    split_metadata["test_class_coverage_issues_after_selection"] = (
        test_coverage_issues
    )
    write_json(args.output_dir / "data_split.json", split_metadata)

    test_started_at = time.perf_counter()
    test_features = prepare_model_features(test)
    test_probabilities, test_predictions = model_predictions(
        model, test_features, class_labels, threshold, majority
    )
    test_seconds = time.perf_counter() - test_started_at

    baseline_validation = [majority] * len(validation)
    baseline_test = [majority] * len(test)
    evaluation = {
        "status": (
            "COMPLETE_WITH_TEST_CLASS_COVERAGE_WARNING"
            if test_coverage_issues
            else "COMPLETE"
        ),
        "data_provenance": provenance,
        "performance_is_from_verified_actual_input": (
            provenance["synthetic_data_status"] == "VERIFIED_NON_SYNTHETIC"
        ),
        "model": {
            "validation": classification_metrics(
                validation_labels, validation_predictions, class_labels
            ),
            "test": classification_metrics(
                test["target_label"], test_predictions, class_labels
            ),
        },
        "majority_baseline": {
            "label": majority,
            "validation": classification_metrics(
                validation_labels, baseline_validation, class_labels
            ),
            "test": classification_metrics(
                test["target_label"], baseline_test, class_labels
            ),
        },
        "selection": frozen_selection,
        "test_class_coverage_issues": test_coverage_issues,
        "timing": {
            "fit_seconds": fit_seconds,
            "validation_selection_seconds": selection_seconds,
            "test_evaluation_seconds": test_seconds,
            "total_seconds": time.perf_counter() - started_at,
        },
        "memory": {
            "measurement": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
            "max_rss_before_fit_bytes": rss_before_fit,
            "process_max_rss_bytes": max_rss_bytes(),
        },
        "catboost_version": catboost.__version__,
    }

    reload_probe = {
        "generated_before_model_save": True,
        "source_split": "validation",
        "features": serialize_feature_row(validation.iloc[0]),
        "class_order": class_labels,
        "pre_save_probabilities": {
            label: float(validation_probabilities[0, index])
            for index, label in enumerate(class_labels)
        },
    }
    write_json(args.output_dir / "reload_probe.json", reload_probe)
    model.save_model(args.output_dir / "model.cbm")
    write_json(
        args.output_dir / "class_labels.json",
        {"source": "model.classes_", "class_labels": class_labels},
    )
    write_json(args.output_dir / "evaluation_report.json", evaluation)
    validation_output = prediction_frame(
        validation,
        "validation",
        validation_probabilities,
        validation_predictions,
        class_labels,
    )
    test_output = prediction_frame(
        test, "test", test_probabilities, test_predictions, class_labels
    )
    pd.concat([validation_output, test_output], ignore_index=True).to_parquet(
        args.output_dir / "evaluation_predictions.parquet", index=False
    )
    print(json.dumps(evaluation, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
