#!/usr/bin/env python3
"""Build or resume fixed, versioned candle-side research experiments."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wonyotti_pipeline.candles import atomic_json, atomic_parquet, deduplicate_executions
from wonyotti_pipeline.dataset import decision_grid
from wonyotti_pipeline.executions import execution_paths, load_executions
from wonyotti_pipeline.side_research import (
    VERSION, FEATURE_COLUMNS, CLASSES, START, END, TRAIN_END, VALIDATION_END,
    ASSUMPTIONS, RUNTIME_CONTRACT, HOUR, build_dataset, feature_schema,
    prepare_model_features, research_splits, load_research_model,
)
from wonyotti_pipeline.training import classification_metrics, class_distribution, file_sha256, majority_label


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, choices=(90, 365), default=90)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train-prepared", action="store_true")
    args = parser.parse_args()
    if args.prepare_only and args.train_prepared:
        parser.error("prepare-only and train-prepared are mutually exclusive")
    version = VERSION if args.days == 90 else "candle_side_research_v2_365d"
    end = START + pd.Timedelta(days=args.days)
    train_end = START + pd.Timedelta(days=args.days * 3 // 5)
    validation_end = START + pd.Timedelta(days=args.days * 4 // 5)
    runtime_contract = dict(RUNTIME_CONTRACT, data_contract_version=version)
    if args.days == 365 and validation_end <= END:
        raise ValueError("new final test must be strictly later than the inspected 90-day interval")
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "models/catboost" / version
    audit_dir = root / "data/audit" / version
    dataset_dir = root / "data/processed/training" / version
    if model_dir.exists():
        raise ValueError(f"research model directory already exists; preserve it: {model_dir}")
    audit_dir.mkdir(parents=True, exist_ok=True)
    if not args.train_prepared:
        cache = root / "data/raw/market/bitmex/candle_v1"
        checkpoint = json.loads((cache / "download_checkpoint.json").read_text())
        candle_frames, candle_hashes, missing = [], {}, []
        for day in pd.date_range((START - 26 * HOUR).normalize(), end - pd.Timedelta(days=1), freq="1D"):
            key = day.strftime("%Y%m%d")
            path = cache / f"{key}.parquet"
            if not path.exists() or key not in checkpoint["days"]:
                missing.append(key)
                continue
            digest = file_sha256(path)
            if digest != checkpoint["days"][key]["sha256"]:
                raise ValueError(f"candle checksum mismatch: {path}")
            frame = pd.read_parquet(path)
            expected = pd.date_range(day + HOUR, day + pd.Timedelta(days=1), freq=HOUR).as_unit("ns")
            if (frame.candle_end_utc.tolist() != expected.tolist() or not frame.symbol.eq("XBTUSD").all()
                    or not frame.candle_start_utc.eq(frame.candle_end_utc - HOUR).all()):
                raise ValueError(f"invalid cached candle boundaries/symbol: {path}")
            candle_hashes[str(path)] = digest
            candle_frames.append(frame)
        if missing:
            raise ValueError(f"missing hourly candle cache dates: {missing}; no data fabricated")
        candles = pd.concat(candle_frames, ignore_index=True)
        paths = execution_paths(root / "data/raw")
        # Choose the bounded source export by its range; completeness remains an explicit assumption.
        required_sources = {"aoa-execution-2018-03-01-2018-12-31.csv"}
        if args.days == 365:
            required_sources.add("aoa-execution-2019-01-01-2020-12-31.csv")
        paths = [p for p in paths if p.name in required_sources]
        if {p.name for p in paths} != required_sources:
            raise ValueError(f"missing selected execution CSVs: {required_sources - {p.name for p in paths}}")
        executions, duplicates = deduplicate_executions(load_executions(paths, symbols={"XBTUSD"}, end_utc=end))
        executions = executions[executions.event_time_utc.ge(START)].reset_index(drop=True)
        grid = decision_grid(["XBTUSD"], START, end, HOUR)
        dataset = build_dataset(grid, candles, executions)
        eligible = dataset[dataset.research_eligible].copy()
        checks = {
            "duplicate_keys": int(dataset.duplicated(["symbol", "decision_time_utc"]).sum()),
            "future_candles": int(dataset.candle_end_utc.ge(dataset.decision_time_utc).sum()),
            "early_targets": int(dataset.target_first_event_time_utc.lt(dataset.decision_time_utc).sum()),
            "late_targets": int(dataset.target_last_event_time_utc.ge(dataset.target_end_time_utc).sum()),
            "position_features_selected": int(any("position" in c or "holding" in c or "unrealized" in c for c in FEATURE_COLUMNS)),
        }
        if any(checks.values()):
            raise ValueError(f"research window checks failed: {checks}")
        prepare_model_features(eligible)
        atomic_parquet(dataset_dir / "training_candidates.parquet", dataset)
        atomic_parquet(dataset_dir / "training_eligible.parquet", eligible)
        provenance = {"research_only": True, "data_contract_version": version, "assumptions": ASSUMPTIONS,
                      "interval_start_utc": START.isoformat(), "interval_end_exclusive_utc": end.isoformat(),
                      "interval_selection": f"Fixed {args.days}-day interval beginning 2018-03-07; extension requested after inspecting v1 test",
                      "split_selection": "Fixed calendar 60/20/20 percent; integer-day boundaries; 26-hour dependency purge",
                      "execution_files": {str(p): file_sha256(p) for p in paths}, "candle_files": candle_hashes,
                      "dataset_sha256": file_sha256(dataset_dir / "training_eligible.parquet"),
                      "implementation_sha256": {name: file_sha256(root / name) for name in (
                          "scripts/train_side_research.py", "src/wonyotti_pipeline/side_research.py",
                          "src/wonyotti_pipeline/candles.py", "src/wonyotti_pipeline/executions.py",
                          "src/wonyotti_pipeline/dataset.py", "src/wonyotti_pipeline/training.py")},
                      "unknown_evidence_config_sha256": file_sha256(root / "configs/account_evidence.json"),
                      "market_downloads": 0, "synthetic_data_used": False}
        atomic_json(audit_dir / "assumption_provenance.json", provenance)
        exclusions = Counter(reason for reasons in dataset.exclusion_reasons for reason in reasons.split("|") if reason)
        dataset_report = {"data_contract_version": version, "research_only": True, "candidate_rows": len(dataset),
                          "eligible_rows": len(eligible), "candidate_class_distribution": class_distribution(dataset, CLASSES),
                          "eligible_class_distribution": class_distribution(eligible, CLASSES), "exclusions": dict(exclusions),
                          "retained_trade_fills": int(dataset.target_fill_count.sum()), "duplicates_removed": duplicates,
                          "zero_uuid_or_missing_order_fills": int(dataset.target_unidentified_order_fill_count.sum()),
                          "candle_cache_days": len(candle_hashes), "checks": checks, "runtime_contract": runtime_contract}
        atomic_json(audit_dir / "dataset_report.json", dataset_report)
    else:
        dataset = pd.read_parquet(dataset_dir / "training_candidates.parquet")
        eligible = pd.read_parquet(dataset_dir / "training_eligible.parquet")
        dataset_report = json.loads((audit_dir / "dataset_report.json").read_text())
        provenance = json.loads((audit_dir / "assumption_provenance.json").read_text())
        checks = dataset_report["checks"]
        if (provenance["dataset_sha256"] != file_sha256(dataset_dir / "training_eligible.parquet")
                or provenance["interval_end_exclusive_utc"] != end.isoformat()):
            raise ValueError("prepared dataset checksum/interval mismatch")
        for name, digest in {**provenance["execution_files"], **provenance["candle_files"],
                             **provenance["implementation_sha256"]}.items():
            if file_sha256(Path(name) if Path(name).is_absolute() else root / name) != digest:
                raise ValueError(f"prepared source/implementation changed: {name}")
    splits, split_report = research_splits(eligible, train_end, validation_end)
    if not split_report["leakage_checks"]["passed"] or any(frame.empty for frame in splits.values()):
        raise ValueError(f"split preflight failed: {split_report}")
    train, validation, test = (splits[name] for name in ("train", "validation", "test"))
    monthly = eligible.groupby(eligible.decision_time_utc.dt.strftime("%Y-%m")).target_label.value_counts().unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)
    preflight = {"version": version, "interval_start_utc": START.isoformat(),
                 "interval_end_exclusive_utc": end.isoformat(), "monthly_class_counts": monthly.to_dict("index"),
                 "split_class_counts": {name: class_distribution(data, CLASSES) for name, data in splits.items()},
                 "train_end_utc": train_end.isoformat(), "validation_end_utc": validation_end.isoformat(),
                 "dataset_checks": checks, "split_report": split_report,
                 "previous_confusion_matrix_split": "v1 test: 2018-05-19T02:00:00Z to 2018-06-05T00:00:00Z",
                 "previous_test_role": "development history; contained in new training interval"}
    atomic_json(audit_dir / "preflight_report.json", preflight)
    print(json.dumps(preflight, indent=2, default=str), flush=True)
    if args.prepare_only:
        return 0
    model_dir.mkdir(parents=True)
    atomic_json(model_dir / "assumption_provenance.json", provenance)
    supported = set(train.target_label)
    if len(supported) < 2:
        raise ValueError("fewer than two actual training classes; no synthetic examples permitted")
    baseline = majority_label(train.target_label)
    validation_supported = validation[validation.target_label.isin(supported)]
    params = {"task_type": "CPU", "iterations": 300, "depth": 4, "thread_count": 4,
              "learning_rate": 0.05, "random_seed": 42, "loss_function": "MultiClass",
              "eval_metric": "MultiClass", "auto_class_weights": "Balanced", "allow_writing_files": False}
    model = CatBoostClassifier(**params)
    fit_kwargs = {"verbose": 50}
    if not validation_supported.empty:
        fit_kwargs.update(eval_set=(prepare_model_features(validation_supported), validation_supported.target_label),
                          use_best_model=True, early_stopping_rounds=30)
    model.fit(prepare_model_features(train), train.target_label, **fit_kwargs)
    labels = [str(label) for label in model.classes_]
    validation_probabilities = model.predict_proba(prepare_model_features(validation))
    config = {"research_only": True, "requested_params": params, "resolved_params": model.get_all_params(),
              "early_stopping_rounds": 30 if not validation_supported.empty else None,
              "best_iteration": int(model.get_best_iteration()), "model_selection": "validation MultiClass loss only; one fixed configuration; argmax",
              "test_used_for_selection": False, "baseline_label_from_train_only": baseline,
              "validation_rows_unsupported_for_early_stopping": len(validation) - len(validation_supported),
              "unsupported_training_classes": sorted(set(CLASSES) - supported)}
    probe_frame = validation.iloc[:5]
    before = model.predict_proba(prepare_model_features(probe_frame))
    probe = {"research_only": True, "source_split": "validation", "class_order": labels,
             "features": prepare_model_features(probe_frame).to_dict("records"), "pre_save_probabilities": before.tolist()}
    temporary = model_dir / "model.cbm.tmp"
    model.save_model(temporary)
    temporary.replace(model_dir / "model.cbm")
    atomic_json(model_dir / "feature_schema.json", feature_schema(version))
    atomic_json(model_dir / "runtime_contract.json", runtime_contract)
    atomic_json(model_dir / "class_labels.json", {"class_labels": labels, "evaluation_class_order": CLASSES,
                                               "unsupported_training_classes": config["unsupported_training_classes"]})
    atomic_json(model_dir / "training_config.json", config)
    reloaded, loaded_labels = load_research_model(model_dir, version)
    restored = reloaded.predict_proba(prepare_model_features(pd.DataFrame(probe["features"])))
    np.testing.assert_allclose(before, restored, rtol=0, atol=1e-12)
    # Verify every validation prediction before reading test outputs.
    reload_validation = reloaded.predict_proba(prepare_model_features(validation))
    np.testing.assert_allclose(validation_probabilities, reload_validation, rtol=0, atol=1e-12)
    probe.update(reload_passed=True, max_abs_difference=float(np.max(np.abs(before - restored))),
                 validation_max_abs_difference=float(np.max(np.abs(validation_probabilities - reload_validation))))
    atomic_json(model_dir / "reload_probe.json", probe)
    evaluation = {"research_only": True, "synthetic_data_used": False, "conditional_on": ASSUMPTIONS,
                  "performance_scope": "Observed execution side labels only; no trading or profitability claim",
                  "evaluation_class_order": CLASSES, "macro_f1_policy": "All four declared classes; undefined precision/recall/F1 represented as zero with support/count shown",
                  "model": {}, "majority_baseline": {}}
    predictions = []
    for name, data in (("validation", validation), ("test", test)):
        probabilities = reload_validation if name == "validation" else reloaded.predict_proba(prepare_model_features(data))
        actions = [loaded_labels[i] for i in probabilities.argmax(axis=1)]
        evaluation["model"][name] = classification_metrics(data.target_label, actions, CLASSES)
        evaluation["majority_baseline"][name] = classification_metrics(data.target_label, [baseline] * len(data), CLASSES)
        evaluation.setdefault("unsupported_in_model", {})[name] = sorted(set(data.target_label) - supported)
        prediction = data[["symbol", "decision_time_utc", "target_label"]].copy()
        prediction["split"], prediction["predicted_label"] = name, actions
        prediction["baseline_label"] = baseline
        for index, label in enumerate(loaded_labels):
            prediction[f"probability_{label}"] = probabilities[:, index]
        predictions.append(prediction)
    split_report["class_distributions"] = {name: class_distribution(data, CLASSES) for name, data in splits.items()}
    atomic_json(model_dir / "data_split.json", split_report)
    atomic_json(model_dir / "evaluation_report.json", evaluation)
    atomic_parquet(model_dir / "evaluation_predictions.parquet", pd.concat(predictions, ignore_index=True))
    atomic_json(audit_dir / "validation_report.json", {"status": "PASS", "dataset_checks": checks,
                "split_checks": split_report["leakage_checks"], "reload_consistency": probe})
    print(json.dumps({"model_dir": str(model_dir), "candidate_rows": len(dataset), "eligible_rows": len(eligible),
                      "split_rows": split_report["retained_counts"], "class_counts": dataset_report["eligible_class_distribution"]["counts"],
                      "validation_macro_f1": evaluation["model"]["validation"]["macro_f1"],
                      "test_macro_f1": evaluation["model"]["test"]["macro_f1"],
                      "baseline_test_macro_f1": evaluation["majority_baseline"]["test"]["macro_f1"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        atomic_json(Path("data/audit") / version / "blocked_report.json", {"research_only": True, "error": str(error)})
        raise
