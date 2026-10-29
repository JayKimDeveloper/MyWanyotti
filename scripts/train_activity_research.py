#!/usr/bin/env python3
"""Freeze one binary model before building and evaluating its later holdout."""
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
from wonyotti_pipeline.activity_research import (
    VERSION, CLASSES, CONTRACT, binary_labels, choose_activity_threshold,
    activity_metrics, load_activity_model,
)
from wonyotti_pipeline.candles import atomic_json, atomic_parquet, deduplicate_executions, HOUR
from wonyotti_pipeline.dataset import decision_grid
from wonyotti_pipeline.executions import load_executions
from wonyotti_pipeline.side_research import (
    ASSUMPTIONS, FEATURE_COLUMNS, build_dataset, feature_schema,
    prepare_model_features, research_splits, load_research_model,
)
from wonyotti_pipeline.training import file_sha256

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "models/catboost/candle_side_research_v2_365d"
MODEL = ROOT / "models/catboost" / VERSION
AUDIT = ROOT / "data/audit" / VERSION
DATASET = ROOT / "data/processed/training" / VERSION
HOLDOUT_START = pd.Timestamp("2019-03-07T00:00:00Z")
HOLDOUT_END = pd.Timestamp("2019-06-05T00:00:00Z")
V2_THRESHOLD = 0.7264083490499316


def row_digest(frame: pd.DataFrame) -> str:
    import hashlib
    return hashlib.sha256(pd.util.hash_pandas_object(
        frame[["symbol", "decision_time_utc", "target_label"]], index=False).to_numpy().tobytes()).hexdigest()


def existing_splits():
    path = ROOT / "data/processed/training/candle_side_research_v2_365d/training_eligible.parquet"
    metadata = json.loads((V2 / "data_split.json").read_text())
    splits, report = research_splits(pd.read_parquet(path), pd.Timestamp(metadata["train_end_utc"]),
                                     pd.Timestamp(metadata["validation_end_utc"]))
    assert report["leakage_checks"]["passed"]
    assert report["retained_counts"] == metadata["retained_counts"]
    saved_validation = pd.read_parquet(V2 / "evaluation_predictions.parquet")
    saved_validation = saved_validation[saved_validation.split.eq("validation")].reset_index(drop=True)
    pd.testing.assert_frame_equal(splits["validation"][["symbol", "decision_time_utc", "target_label"]],
                                  saved_validation[["symbol", "decision_time_utc", "target_label"]])
    return path, splits, report, saved_validation


def inspect_history() -> dict:
    inspected, overlaps = {}, []
    for path in sorted((ROOT / "models").rglob("data_split.json")):
        if MODEL in path.parents:
            continue
        metadata = json.loads(path.read_text())
        inspected[str(path)] = file_sha256(path)
        prediction = path.parent / "evaluation_predictions.parquet"
        if prediction.exists():
            frame = pd.read_parquet(prediction, columns=["split", "decision_time_utc"])
            for name, subset in frame.groupby("split"):
                count = int((subset.decision_time_utc.ge(HOLDOUT_START) & subset.decision_time_utc.lt(HOLDOUT_END)).sum())
                if count:
                    overlaps.append({"path": str(prediction), "split": name, "rows": count})
            inspected[str(prediction)] = file_sha256(prediction)
        elif metadata.get("periods"):
            for name in ("validation", "test"):
                period = metadata["periods"].get(name, {})
                if period and pd.Timestamp(period["decision_start_utc"]) < HOLDOUT_END and pd.Timestamp(period["target_end_utc"]) > HOLDOUT_START:
                    overlaps.append({"path": str(path), "split": name, "period": period})
    return {"inspected_sources": inspected, "prior_evaluation_overlaps": overlaps,
            "claim_scope": "No prior validation/test rows in inspected saved experiment metadata" if not overlaps else "Previously evaluated interval; exploratory only",
            "known_prior_training_use": "Older candle_v1 training range ends 2021-01-01 and includes the proposed interval; not globally untouched"}


def freeze() -> None:
    if MODEL.exists():
        raise ValueError(f"preserve existing model directory: {MODEL}")
    history = inspect_history()
    source, splits, report, saved_validation = existing_splits()
    train, validation = splits["train"], splits["validation"]
    train_labels, validation_labels = binary_labels(train.target_label), binary_labels(validation.target_label)
    if set(train_labels) != set(CLASSES):
        raise ValueError("both actual binary training classes are required")
    protected = [p for folder in (ROOT / "models", ROOT / "data/processed/training", ROOT / "data/audit")
                 for p in folder.rglob("*") if p.is_file()]
    protected += list((ROOT / "data/raw").glob("aoa-*.csv")) + list((ROOT / "configs").glob("*.json"))
    protected += list((ROOT / "docs").glob("candle*research*.md"))
    hashes = {str(p): file_sha256(p) for p in protected}
    params = dict(json.loads((V2 / "training_config.json").read_text())["requested_params"])
    params.pop("auto_class_weights", None)
    params.update(loss_function="Logloss", eval_metric="Logloss")
    assert not any(k in params for k in ("class_weights", "auto_class_weights", "scale_pos_weight"))
    model = CatBoostClassifier(**params)
    model.fit(prepare_model_features(train), train_labels,
              eval_set=(prepare_model_features(validation), validation_labels),
              early_stopping_rounds=30, use_best_model=True, verbose=50)
    resolved = model.get_all_params()
    assert resolved.get("auto_class_weights", "None") == "None"
    assert not resolved.get("class_weights") or all(w == 1 for w in resolved["class_weights"])
    labels = list(model.classes_)
    before = model.predict_proba(prepare_model_features(validation))
    scores = before[:, labels.index("ACTIVE_RECORDED")]
    threshold = choose_activity_threshold(validation_labels.eq("ACTIVE_RECORDED"), scores)
    MODEL.mkdir(parents=True)
    model.save_model(MODEL / "model.cbm")
    atomic_json(MODEL / "feature_schema.json", feature_schema(VERSION))
    atomic_json(MODEL / "runtime_contract.json", CONTRACT)
    atomic_json(MODEL / "class_labels.json", {"class_labels": labels, "evaluation_class_order": CLASSES})
    atomic_json(MODEL / "activity_threshold.json", {"threshold": threshold, "selected_on": "v2 validation rows only",
                "criterion": "maximum positive-class binary F1 over unique validation scores; ties prefer greatest threshold"})
    atomic_json(MODEL / "training_config.json", {"requested_params": params, "resolved_params": resolved,
                "best_iteration": model.get_best_iteration(), "early_stopping_rounds": 30,
                "selection_criterion": "validation unweighted binary Logloss", "resampling": False,
                "test_used_for_selection": False, "research_only": True})
    loaded, loaded_labels, loaded_threshold = load_activity_model(MODEL)
    after = loaded.predict_proba(prepare_model_features(validation))
    np.testing.assert_allclose(before, after, rtol=0, atol=1e-12)
    assert loaded_threshold == threshold and loaded_labels == labels
    atomic_json(MODEL / "reload_probe.json", {"source_split": "validation", "features": prepare_model_features(validation.iloc[:5]).to_dict("records"),
                "class_order": labels, "pre_save_probabilities": before[:5].tolist(),
                "reload_passed": True, "validation_rows": len(validation), "max_abs_difference": float(np.max(np.abs(before - after)))})
    provenance = {"research_only": True, "assumptions": ASSUMPTIONS, "v2_dataset": str(source),
                  "v2_dataset_sha256": file_sha256(source), "split_report": report,
                  "exact_v2_row_identity": {name: {"rows": len(splits[name]), "row_sha256": row_digest(splits[name]),
                       "start_utc": splits[name].decision_time_utc.min().isoformat(),
                       "end_exclusive_utc": splits[name].target_end_time_utc.max().isoformat(),
                       "binary_counts": binary_labels(splits[name].target_label).value_counts().to_dict()}
                       for name in ("train", "validation")}, "history": history}
    atomic_json(MODEL / "split_provenance.json", provenance)
    atomic_json(MODEL / "assumption_provenance.json", {"research_only": True, "assumptions": ASSUMPTIONS,
                "label_interval_end_exclusive_utc": HOLDOUT_END.isoformat(), "production_evidence_modified": False})
    validation_scores = {"v3_unweighted": scores,
                         "v2_activity": 1 - saved_validation.probability_NO_RECORDED_FILL.to_numpy()}
    val_report = {name: activity_metrics(validation_labels.eq("ACTIVE_RECORDED"), score,
                   threshold if name == "v3_unweighted" else V2_THRESHOLD) for name, score in validation_scores.items()}
    atomic_json(MODEL / "validation_evaluation.json", val_report)
    prediction = validation[["symbol", "decision_time_utc", "target_label"]].copy()
    prediction["binary_target"] = validation_labels
    for name, score in validation_scores.items():
        prediction[name + "_score"] = score
    atomic_parquet(MODEL / "validation_predictions.parquet", prediction)
    protocol = {"research_only": True, "frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "holdout_start_utc": HOLDOUT_START.isoformat(), "holdout_end_exclusive_utc": HOLDOUT_END.isoformat(),
                "holdout_feature_start_boundary": HOLDOUT_START.isoformat(), "holdout_initial_dependency_purge_hours": 26,
                "cross_order_policy": "exclude holdout rows sharing an identified order with v2 train/validation; do not change original rows",
                "v3_threshold": threshold, "v2_threshold": V2_THRESHOLD,
                "constant_score": float(train_labels.eq("ACTIVE_RECORDED").mean()), "constant_flag_threshold": 0.5,
                "always_no_recorded_fill": {"score": 0.0, "flagged": False},
                "metrics": "noninterpolated AP grouping ties; ACTIVE_RECORDED precision/recall/F1; flagged/false-positive counts; monthly same-row comparisons",
                "ap_no_positive_policy": "null; unsupported monthly positive class shown with support zero",
                "calibration_claim": False, "no_holdout_tuning": True,
                "confounding": "weighting and learning objective both change; effects cannot be attributed independently",
                "history": history,
                "implementation_hashes": {str(ROOT / name): file_sha256(ROOT / name) for name in
                    ("scripts/train_activity_research.py", "src/wonyotti_pipeline/activity_research.py", "src/wonyotti_pipeline/side_research.py", "src/wonyotti_pipeline/candles.py")},
                "frozen_artifact_hashes": {str(p): file_sha256(p) for p in MODEL.iterdir() if p.is_file()},
                "protected_artifact_hashes": hashes}
    atomic_json(MODEL / "frozen_protocol.json", protocol)
    print(json.dumps({"phase": "FROZEN_BEFORE_HOLDOUT", "model_dir": str(MODEL), "threshold": threshold,
                      "best_iteration": model.get_best_iteration(), "validation": val_report}, indent=2))


def evaluate() -> None:
    if (MODEL / "holdout_evaluation.json").exists():
        raise ValueError("preserve existing holdout evaluation; do not re-evaluate for tuning")
    protocol = json.loads((MODEL / "frozen_protocol.json").read_text())
    for name, digest in {**protocol["frozen_artifact_hashes"], **protocol["implementation_hashes"], **protocol["protected_artifact_hashes"]}.items():
        if file_sha256(Path(name)) != digest:
            raise ValueError(f"frozen or protected source changed: {name}")
    _, splits, _, _ = existing_splits()
    model, labels, threshold = load_activity_model(MODEL)
    assert threshold == protocol["v3_threshold"]
    cache = ROOT / "data/raw/market/bitmex/candle_v1"
    checkpoint = json.loads((cache / "download_checkpoint.json").read_text())
    frames, candle_hashes, missing = [], {}, []
    for day in pd.date_range((HOLDOUT_START - 26 * HOUR).normalize(), HOLDOUT_END - pd.Timedelta(days=1), freq="D"):
        key = f"{day:%Y%m%d}"
        path = cache / f"{key}.parquet"
        if not path.exists() or key not in checkpoint["days"]:
            missing.append(key)
            continue
        if file_sha256(path) != checkpoint["days"][key]["sha256"]:
            raise ValueError(f"candle checksum mismatch: {path}")
        frame = pd.read_parquet(path)
        expected = pd.date_range(day + HOUR, day + pd.Timedelta(days=1), freq=HOUR).as_unit("ns")
        assert frame.candle_end_utc.tolist() == expected.tolist()
        assert frame.symbol.eq("XBTUSD").all() and frame.candle_start_utc.eq(frame.candle_end_utc - HOUR).all()
        candle_hashes[str(path)] = file_sha256(path)
        frames.append(frame)
    if missing:
        raise ValueError(f"missing XBTUSD candle dates: {missing}; model remains frozen; retrieve only these dates and resume --phase evaluate")
    executions_path = ROOT / "data/raw/aoa-execution-2019-01-01-2020-12-31.csv"
    executions, duplicates = deduplicate_executions(load_executions([executions_path], symbols={"XBTUSD"}, end_utc=HOLDOUT_END))
    executions = executions[executions.event_time_utc.ge(HOLDOUT_START)].reset_index(drop=True)
    candidate = build_dataset(decision_grid(["XBTUSD"], HOLDOUT_START, HOLDOUT_END, HOUR),
                              pd.concat(frames, ignore_index=True), executions)
    original_order_ids = set()
    for name in ("train", "validation"):
        for value in splits[name].target_order_ids:
            original_order_ids.update(value.split("|") if value else [])
    initial_purge = candidate.feature_start_time_utc.lt(HOLDOUT_START)
    shared_orders = candidate.target_order_ids.map(lambda value: bool(original_order_ids.intersection(value.split("|") if value else [])))
    candidate.loc[initial_purge, "exclusion_reasons"] += "|HOLDOUT_DEPENDENCY_PURGE"
    candidate.loc[shared_orders, "exclusion_reasons"] += "|SHARED_TRAIN_VALIDATION_ORDER"
    candidate["research_eligible"] &= ~initial_purge & ~shared_orders
    holdout = candidate[candidate.research_eligible].reset_index(drop=True)
    if holdout.empty:
        raise ValueError("no actual eligible holdout rows; no synthetic results")
    assert holdout.feature_start_time_utc.min() >= HOLDOUT_START
    assert max(splits[name].dependency_end_time_utc.max() for name in ("train", "validation")) <= holdout.feature_start_time_utc.min()
    assert holdout.candle_end_utc.lt(holdout.decision_time_utc).all()
    assert not holdout.duplicated(["symbol", "decision_time_utc"]).any()
    assert not holdout.target_first_event_time_utc.lt(holdout.decision_time_utc).any()
    assert not holdout.target_last_event_time_utc.ge(holdout.target_end_time_utc).any()
    features = prepare_model_features(holdout)
    assert features.columns.tolist() == FEATURE_COLUMNS and features.dtypes.eq("float64").all()
    v3_prob = model.predict_proba(features)
    v2, v2_labels = load_research_model(V2, "candle_side_research_v2_365d")
    scores = {"v3_unweighted": v3_prob[:, labels.index("ACTIVE_RECORDED")],
              "v2_activity": 1 - v2.predict_proba(features)[:, v2_labels.index("NO_RECORDED_FILL")],
              "constant_score": np.full(len(holdout), protocol["constant_score"]),
              "always_no_recorded_fill": np.zeros(len(holdout))}
    thresholds = {"v3_unweighted": threshold, "v2_activity": V2_THRESHOLD,
                  "constant_score": 0.5, "always_no_recorded_fill": 0.5}
    actual = binary_labels(holdout.target_label).eq("ACTIVE_RECORDED").to_numpy()
    report = {"research_only": True, "assumptions": ASSUMPTIONS, "frozen_protocol_sha256": file_sha256(MODEL / "frozen_protocol.json"),
              "history": protocol["history"], "rows": len(holdout), "models": {}, "monthly": {},
              "start_utc": holdout.decision_time_utc.min().isoformat(), "end_exclusive_utc": holdout.target_end_time_utc.max().isoformat()}
    months = holdout.decision_time_utc.dt.strftime("%Y-%m").to_numpy()
    for name, score in scores.items():
        report["models"][name] = activity_metrics(actual, score, thresholds[name])
        report["monthly"][name] = {month: activity_metrics(actual[months == month], score[months == month], thresholds[name]) for month in sorted(set(months))}
    prediction = holdout[["symbol", "decision_time_utc", "target_label"]].copy()
    prediction["binary_target"] = binary_labels(holdout.target_label)
    for name, score in scores.items():
        prediction[name + "_score"] = score
        prediction[name + "_flagged"] = score >= thresholds[name]
    atomic_parquet(DATASET / "holdout_candidates.parquet", candidate)
    atomic_parquet(DATASET / "holdout_eligible.parquet", holdout)
    atomic_parquet(MODEL / "holdout_predictions.parquet", prediction)
    atomic_json(MODEL / "holdout_evaluation.json", report)
    exclusions = Counter(reason for value in candidate.exclusion_reasons for reason in value.split("|") if reason)
    atomic_json(AUDIT / "holdout_dataset_report.json", {"candidate_rows": len(candidate), "eligible_rows": len(holdout),
                "candidate_fills": int(candidate.target_fill_count.sum()), "eligible_fills": int(holdout.target_fill_count.sum()),
                "exclusions": dict(exclusions), "duplicates_removed": duplicates, "cache_days": len(frames),
                "candle_files": candle_hashes, "market_downloads": 0, "execution_file_sha256": file_sha256(executions_path)})
    probe = json.loads((MODEL / "reload_probe.json").read_text())
    check_model, check_labels, check_threshold = load_activity_model(MODEL)
    np.testing.assert_allclose(check_model.predict_proba(prepare_model_features(pd.DataFrame(probe["features"]))), probe["pre_save_probabilities"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(check_model.predict_proba(features), v3_prob, rtol=0, atol=1e-12)
    assert check_threshold == threshold and check_labels == labels
    for name, digest in protocol["protected_artifact_hashes"].items():
        assert file_sha256(Path(name)) == digest, name
    atomic_json(AUDIT / "validation_report.json", {"status": "PASS", "exact_v2_train_validation_rows": True,
                "completed_candles": True, "dependency_separation": True, "shared_orders": int(shared_orders.sum()),
                "feature_order_and_float64_consistency": True, "validation_reload_max_abs_difference": probe["max_abs_difference"],
                "holdout_reload_max_abs_difference": 0.0, "protected_files_unchanged": len(protocol["protected_artifact_hashes"]),
                "frozen_before_holdout": True, "model_sha256": file_sha256(MODEL / "model.cbm")})
    print(json.dumps({"model_dir": str(MODEL), "holdout_rows": len(holdout), "exclusions": dict(exclusions), "comparison": report["models"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("freeze", "evaluate"))
    args = parser.parse_args()
    try:
        freeze() if args.phase == "freeze" else evaluate()
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
