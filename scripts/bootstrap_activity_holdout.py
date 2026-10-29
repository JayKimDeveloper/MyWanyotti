#!/usr/bin/env python3
"""One fixed paired calendar-week bootstrap of saved research scores."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wonyotti_pipeline.activity_research import activity_metrics, binary_labels
from wonyotti_pipeline.training import file_sha256

REPLICATIONS = 2000
SEED = 42
THRESHOLDS = {"v3_unweighted": 0.22920271033221618, "v2_activity": 0.7264083490499316}
METRICS = ["average_precision", "binary_f1", "precision", "recall", "false_positives_per_1000"]


def monday_blocks(times: pd.Series) -> tuple[list[pd.Timestamp], list[np.ndarray]]:
    if times.dt.tz is None or str(times.dt.tz) != "UTC":
        raise ValueError("calendar blocks require explicit UTC timestamps")
    weeks = times.dt.normalize() - pd.to_timedelta(times.dt.dayofweek, unit="D")
    starts = sorted(weeks.unique())
    return starts, [np.flatnonzero(weeks.eq(start).to_numpy()) for start in starts]


def measured(actual, scores, threshold: float) -> dict:
    actual = np.asarray(actual, dtype=bool)
    result = activity_metrics(actual, scores, threshold)
    result["false_positives_per_1000"] = 1000 * result["false_positives"] / len(actual)
    # Bootstrap undefined denominators are excluded, rather than replaced by zero.
    if not result["flagged_count"]:
        result["precision"] = None
    if not result["active_count"]:
        result["recall"] = None
    if not result["active_count"] and not result["flagged_count"]:
        result["binary_f1"] = None
    return {key: result[key] for key in METRICS}


def paired_bootstrap(actual, scores: dict, blocks: list[np.ndarray], replications: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in METRICS}
    undefined = {name: {key: 0 for key in METRICS} for name in scores}
    row_counts = []
    for _ in range(replications):
        # The same selected blocks and row indices are used for both models.
        selected = rng.integers(0, len(blocks), size=len(blocks))
        rows = np.concatenate([blocks[i] for i in selected])
        row_counts.append(len(rows))
        results = {name: measured(actual[rows], values[rows], THRESHOLDS[name]) for name, values in scores.items()}
        for key in METRICS:
            for name in scores:
                undefined[name][key] += int(results[name][key] is None)
            left, right = results["v3_unweighted"][key], results["v2_activity"][key]
            if left is not None and right is not None:
                samples[key].append(left - right)
    return {"metrics": {key: {"valid_replications": len(values), "invalid_replications": replications - len(values),
                    "percentile_95_interval": np.quantile(values, [.025, .975], method="linear").tolist() if values else None}
                    for key, values in samples.items()}, "undefined_by_model": undefined,
            "sampled_row_counts": {"min": min(row_counts), "max": max(row_counts), "mean": float(np.mean(row_counts))}}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    version = "candle_activity_research_v3_unweighted"
    model_dir = root / "models/catboost" / version
    json_path = root / "data/audit" / version / "paired_weekly_bootstrap_2000_seed42.json"
    markdown_path = root / "docs/candle_activity_research_v3_weekly_bootstrap.md"
    if json_path.exists() or markdown_path.exists():
        raise ValueError("preserve existing uncertainty reports; fixed analysis must not overwrite them")
    prediction_path = model_dir / "holdout_predictions.parquet"
    reference_path = root / "data/processed/training" / version / "holdout_eligible.parquet"
    source_paths = [prediction_path, reference_path, model_dir / "holdout_evaluation.json", model_dir / "frozen_protocol.json"]
    before = {str(path): file_sha256(path) for path in source_paths}
    frame = pd.read_parquet(prediction_path)
    expected_columns = ["symbol", "decision_time_utc", "target_label", "binary_target"] + [name + suffix for name in THRESHOLDS for suffix in ("_score", "_flagged")]
    if not set(expected_columns).issubset(frame.columns):
        raise ValueError(f"missing saved labels/scores: {set(expected_columns) - set(frame.columns)}")
    frame = frame.sort_values(["decision_time_utc", "symbol"]).reset_index(drop=True)
    assert len(frame) == 2134 and not frame.duplicated(["symbol", "decision_time_utc"]).any()
    assert frame.symbol.eq("XBTUSD").all()
    assert frame.decision_time_utc.diff().dropna().eq(pd.Timedelta(hours=1)).all()
    reference = pd.read_parquet(reference_path, columns=["symbol", "decision_time_utc", "target_label"])
    reference = reference.sort_values(["decision_time_utc", "symbol"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(frame[["symbol", "decision_time_utc", "target_label"]], reference)
    pd.testing.assert_series_equal(binary_labels(frame.target_label), frame.binary_target, check_names=False)
    protocol = json.loads((model_dir / "frozen_protocol.json").read_text())
    assert protocol["v3_threshold"] == THRESHOLDS["v3_unweighted"]
    assert protocol["v2_threshold"] == THRESHOLDS["v2_activity"]
    actual = frame.binary_target.eq("ACTIVE_RECORDED").to_numpy()
    scores = {name: frame[name + "_score"].to_numpy() for name in THRESHOLDS}
    for name, values in scores.items():
        assert np.array_equal(values >= THRESHOLDS[name], frame[name + "_flagged"].to_numpy())
    original = json.loads((model_dir / "holdout_evaluation.json").read_text())
    points = {name: measured(actual, values, THRESHOLDS[name]) for name, values in scores.items()}
    for name, point in points.items():
        for key in METRICS:
            saved = original["models"][name]
            expected = saved[key] if key != "false_positives_per_1000" else saved["false_positives"] * 1000 / saved["row_count"]
            np.testing.assert_allclose(point[key], expected, rtol=0, atol=1e-12)
    starts, blocks = monday_blocks(frame.decision_time_utc)
    assert np.array_equal(np.concatenate(blocks), np.arange(len(frame)))
    result = paired_bootstrap(actual, scores, blocks, REPLICATIONS, SEED)
    differences = {key: points["v3_unweighted"][key] - points["v2_activity"][key] for key in METRICS}
    monthly = frame.assign(month=frame.decision_time_utc.dt.strftime("%Y-%m"), active=actual).groupby("month").active.agg(["size", "sum", "mean"])
    report = {"research_only": True, "exploratory": True, "assumptions": original["assumptions"],
              "rows": len(frame), "active_support": int(actual.sum()), "paired_alignment_checks": "PASS",
              "original_point_estimates_recalculated": "PASS", "comparison_operator": ">=", "thresholds": THRESHOLDS,
              "replications": REPLICATIONS, "seed": SEED, "rng": "numpy.default_rng PCG64",
              "method": "sample K of K observed UTC Monday calendar-week blocks uniformly with replacement; retain all rows in each block, including partial boundary weeks; identical row sample for both models",
              "interval_method": "exploratory 2.5th/97.5th percentile of paired V3-minus-V2 differences; linear quantiles; no multiplicity correction",
              "point_estimates": points, "point_differences_v3_minus_v2": differences,
              "blocks": [{"week_start_utc": start.isoformat(), "rows": len(rows),
                          "observed_start_utc": frame.iloc[rows[0]].decision_time_utc.isoformat(),
                          "observed_end_exclusive_utc": (frame.iloc[rows[-1]].decision_time_utc + pd.Timedelta(hours=1)).isoformat(),
                          "partial": len(rows) != 168} for start, rows in zip(starts, blocks)],
              "bootstrap": result, "monthly_activity": monthly.to_dict("index"),
              "undefined_policy": "AP and recall undefined with no positives; precision undefined with no flags; F1 undefined only when TP/FP/FN denominator is zero. Exclude a replication for a metric if either model is undefined. FP per 1000 always defined for nonempty samples. Original saved zero conventions are preserved in the source reports, not overwritten.",
              "limitations": ["Only 14 observed blocks in a roughly 90-day interval, including 70- and 48-hour partial weeks",
                  "Uniformly resampling weeks treats blocks as exchangeable despite changing monthly activity prevalence",
                  "Retains within-week dependence, but does not preserve dependence across week boundaries or temporal trends",
                  "Variable-size bootstrap samples arise from partial weeks; metrics use each replication's actual number of windows",
                  "Conditional on frozen fitted models and thresholds; excludes model-training and threshold-selection uncertainty",
                  "Holdout has already been inspected; exploratory intervals, not confirmatory evidence",
                  "Objective and weighting changed together; their separate effects cannot be identified",
                  "Unverified UTC/completeness assumptions; scores are not established as calibrated probabilities",
                  "Intervals containing zero do not prove equivalence; no profitability inference"],
              "source_sha256": before, "analysis_sha256": file_sha256(Path(__file__)),
              "source_files_unchanged": all(file_sha256(Path(path)) == digest for path, digest in before.items())}
    assert report["source_files_unchanged"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Paired weekly-block uncertainty: saved v2/v3 holdout", "",
             "Exploratory analysis of exactly 2,134 aligned XBTUSD hourly windows and labels (544 ACTIVE_RECORDED). Point metrics match the original saved comparison. No training, inference, data rebuilding, downloading, threshold tuning, or evidence reinvestigation.", "",
             "UTC and supplied-record completeness remain unverified research assumptions. NO_RECORDED_FILL does not prove inactivity; scores are not established as calibrated. Frozen operators remain score >= threshold: V3 = 0.22920271033221618; V2 = 0.7264083490499316.", "",
             "2,000 paired replications, seed 42, numpy.default_rng PCG64. In each replication, sample 14 observed UTC calendar weeks beginning Monday uniformly with replacement, keeping complete observed row blocks. Exactly the same sampled weeks/rows are used for both models. Include both partial boundary weeks. Percentile intervals use linear 2.5%/97.5% quantiles of paired differences, without multiplicity correction.", "",
             "| Metric | V2 point | V3 point | V3 − V2 | Exploratory 95% interval | Valid replications |",
             "|---|---:|---:|---:|---|---:|"]
    for key in METRICS:
        item = result["metrics"][key]
        lo, hi = item["percentile_95_interval"]
        lines.append(f'| {key} | {points["v2_activity"][key]:.6f} | {points["v3_unweighted"][key]:.6f} | {differences[key]:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {item["valid_replications"]}/2000 |')
    lines += ["", "Higher is better for AP, F1, precision and recall; lower is better for false positives per 1,000. F1 is ACTIVE_RECORDED positive-class F1; AP is noninterpolated and groups tied scores. FP counts remain V3=500, V2=526; rates use evaluated windows rather than negative-window counts.", "",
              "## Calendar blocks", "", "| UTC Monday week start | Windows | Observed interval (exclusive end) | Partial |", "|---|---:|---|---|"]
    for item in report["blocks"]:
        lines.append(f'| {item["week_start_utc"]} | {item["rows"]} | [{item["observed_start_utc"]}, {item["observed_end_exclusive_utc"]}) | {item["partial"]} |')
    lines += ["", f'Bootstrap window counts: min {result["sampled_row_counts"]["min"]}, max {result["sampled_row_counts"]["max"]}, mean {result["sampled_row_counts"]["mean"]:.2f}. Unequal block sizes cause variable sample sizes; all rates and metrics use each sampled row count.', "",
              "## Undefined metrics and prevalence", "", report["undefined_policy"], "",
              "Undefined counts by model and metric: " + json.dumps(result["undefined_by_model"]) + ". Valid/invalid paired counts are saved per metric in JSON.", "",
              "| Observed month | Windows | Active support | Active prevalence |", "|---|---:|---:|---:|"]
    for month, item in monthly.iterrows():
        lines.append(f'| {month} | {int(item["size"])} | {int(item["sum"])} | {item["mean"]:.2%} |')
    lines += ["", "March and June are partial. The March-to-May prevalence change makes a stationary/exchangeable-week approximation questionable. The short interval and only 14 blocks limit tail precision and generalization.", "",
              "## Limits and conclusion", ""] + ["- " + item for item in report["limitations"]]
    lines += ["", "The evidence does not establish a consistent preference for either model: small point differences and paired intervals must be read alongside metric tradeoffs and time variation. A zero-containing interval is not proof of equivalence.", "",
              f'JSON report: [paired_weekly_bootstrap_2000_seed42.json](../data/audit/{version}/paired_weekly_bootstrap_2000_seed42.json). Source artifacts were hash checked unchanged. Focused checks cover calendar-week boundaries, partial blocks, threshold equality, undefined denominators, pairing, and reproducibility.']
    markdown_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({"json_report": str(json_path), "markdown_report": str(markdown_path),
                      "block_sizes": [len(rows) for rows in blocks], "differences": differences, "bootstrap": result}, indent=2))


if __name__ == "__main__":
    main()
