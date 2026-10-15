#!/usr/bin/env python3
"""Run a held-out chronological backtest with explicit execution assumptions."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wonyotti_pipeline.backtest import (  # noqa: E402
    BacktestConfigError,
    evaluate_trader_positions,
    load_backtest_config,
    reconcile_market_features,
    run_bot_backtest,
)
from wonyotti_pipeline.inference import (  # noqa: E402
    CatBoostActionPredictor,
    InferenceContractError,
)
from wonyotti_pipeline.market import (  # noqa: E402
    build_market_features,
    discover_daily_files,
    load_public_quotes,
    load_public_trades,
)
from wonyotti_pipeline.training import (  # noqa: E402
    TrainingDataError,
    file_sha256,
    load_training_data,
    max_rss_bytes,
    validate_input_contract,
    write_json,
)


class BacktestPreflightError(ValueError):
    """Raised when model, data, or held-out split provenance is unsafe."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--market-dir", type=Path, default=Path("data/raw/market/bitmex")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/backtest.example.toml")
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/audit/backtests/latest")
    )
    parser.add_argument(
        "--acknowledge-research-assumptions",
        action="store_true",
        help="Required when the config is not historically verified.",
    )
    parser.add_argument(
        "--allow-unverified-model",
        action="store_true",
        help="Engineering smoke tests only; resulting metrics are not performance.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise BacktestPreflightError(f"required artifact not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BacktestPreflightError(f"JSON artifact must contain an object: {path}")
    return value


def utc_timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def blocked_report(
    output_dir: Path,
    error: str,
    started_at: float,
    details: dict[str, object] | None = None,
) -> int:
    report = {
        "status": "BLOCKED",
        "error": error,
        "details": details or {},
        "backtest_completed": False,
        "performance_claimed": False,
        "elapsed_seconds": time.perf_counter() - started_at,
        "process_max_rss_bytes": max_rss_bytes(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "preflight_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), file=sys.stderr)
    return 2


def check_model_and_data_provenance(
    model_dir: Path,
    dataset: Path,
    allow_unverified: bool,
) -> dict[str, object]:
    required = [
        "model.cbm",
        "feature_schema.json",
        "class_labels.json",
        "training_config.json",
        "data_split.json",
        "data_version.json",
        "evaluation_report.json",
        "evaluation_predictions.parquet",
    ]
    missing = [str(model_dir / name) for name in required if not (model_dir / name).exists()]
    if missing:
        raise BacktestPreflightError("required model artifacts not found: " + ", ".join(missing))

    version = read_json(model_dir / "data_version.json")
    split = read_json(model_dir / "data_split.json")
    evaluation = read_json(model_dir / "evaluation_report.json")
    provenance = version.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    checks = {
        "synthetic_data_status": provenance.get("synthetic_data_status"),
        "data_contract_status": provenance.get("data_contract_status"),
        "data_contract_version": version.get("data_contract_version"),
        "evaluation_uses_verified_actual_input": evaluation.get(
            "performance_is_from_verified_actual_input"
        ),
        "split_leakage_checks_passed": (
            split.get("split_leakage_checks", {}).get("passed")
            if isinstance(split.get("split_leakage_checks"), dict)
            else False
        ),
        "cross_split_order_count_after_purge": (
            split.get("split_leakage_checks", {}).get(
                "cross_split_order_count_after_purge"
            )
            if isinstance(split.get("split_leakage_checks"), dict)
            else None
        ),
        "dataset_sha256_expected": version.get("sha256"),
        "dataset_sha256_actual": file_sha256(dataset),
    }
    checks["dataset_sha256_matches"] = (
        checks["dataset_sha256_expected"] == checks["dataset_sha256_actual"]
    )
    checks["verified_actual_model_and_data"] = all(
        [
            checks["synthetic_data_status"] == "VERIFIED_NON_SYNTHETIC",
            checks["data_contract_status"] == "MATCH",
            checks["data_contract_version"] == "v0.3",
            checks["evaluation_uses_verified_actual_input"] is True,
            checks["split_leakage_checks_passed"] is True,
            checks["cross_split_order_count_after_purge"] == 0,
            checks["dataset_sha256_matches"] is True,
        ]
    )
    if not checks["verified_actual_model_and_data"] and not allow_unverified:
        raise BacktestPreflightError(
            "model/data provenance is not verified for performance reporting; "
            "--allow-unverified-model is restricted to engineering smoke tests"
        )
    return checks


def select_heldout_test_rows(
    frame: pd.DataFrame,
    model_dir: Path,
    symbol: str,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    predictions = pd.read_parquet(model_dir / "evaluation_predictions.parquet")
    required = {
        "split",
        "symbol",
        "decision_time_utc",
        "target_end_time_utc",
        "target_label",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise BacktestPreflightError(
            "evaluation_predictions.parquet is missing: " + ", ".join(missing)
        )
    predictions["decision_time_utc"] = pd.to_datetime(
        predictions["decision_time_utc"], utc=True
    )
    predictions["target_end_time_utc"] = pd.to_datetime(
        predictions["target_end_time_utc"], utc=True
    )
    test_keys = predictions.loc[
        predictions["split"].eq("test") & predictions["symbol"].eq(symbol),
        ["symbol", "decision_time_utc", "target_end_time_utc", "target_label"],
    ].copy()
    if test_keys.empty:
        raise BacktestPreflightError(f"no held-out test rows for symbol {symbol}")
    if test_keys.duplicated(["symbol", "decision_time_utc"]).any():
        raise BacktestPreflightError("held-out test keys contain duplicates")

    selected = frame.merge(
        test_keys,
        on=["symbol", "decision_time_utc"],
        how="inner",
        suffixes=("", "_artifact"),
        validate="one_to_one",
    )
    if len(selected) != len(test_keys):
        raise BacktestPreflightError(
            "dataset does not contain every held-out test key from model artifacts"
        )
    target_end_matches = selected["target_end_time_utc"].eq(
        selected["target_end_time_utc_artifact"]
    )
    target_label_matches = selected["target_label"].astype(str).eq(
        selected["target_label_artifact"].astype(str)
    )
    if not target_end_matches.all() or not target_label_matches.all():
        raise BacktestPreflightError(
            "held-out target interval or label differs from evaluation artifacts"
        )
    selected = selected.drop(
        columns=["target_end_time_utc_artifact", "target_label_artifact"]
    )
    before_narrowing = len(selected)
    if start is not None:
        selected = selected.loc[selected["decision_time_utc"].ge(start)]
    if end is not None:
        selected = selected.loc[selected["target_end_time_utc"].le(end)]
    selected = selected.sort_values("decision_time_utc").reset_index(drop=True)
    if selected.empty:
        raise BacktestPreflightError("no complete held-out test rows remain")
    return selected, {
        "scope": "HELDOUT_TEST_ONLY",
        "heldout_test_row_count": int(len(test_keys)),
        "selected_row_count": int(len(selected)),
        "row_count_before_optional_time_filter": int(before_narrowing),
        "decision_start_utc": selected["decision_time_utc"].min().isoformat(),
        "decision_end_utc": selected["decision_time_utc"].max().isoformat(),
        "target_end_utc": selected["target_end_time_utc"].max().isoformat(),
    }


def main() -> int:
    args = parse_args()
    started_at = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        config = load_backtest_config(args.config, args.symbol)
        if (
            config.assumption_status != "HISTORICALLY_VERIFIED"
            and not args.acknowledge_research_assumptions
        ):
            raise BacktestPreflightError(
                "backtest config is not historically verified; pass "
                "--acknowledge-research-assumptions to run a research scenario"
            )
        if not args.dataset.exists():
            raise BacktestPreflightError(f"dataset not found: {args.dataset}")
        provenance = check_model_and_data_provenance(
            args.model_dir, args.dataset, args.allow_unverified_model
        )
        predictor = CatBoostActionPredictor(args.model_dir)
        frame, input_counts = load_training_data(args.dataset)
        input_contract = validate_input_contract(frame)
        if not input_contract["passed"]:
            raise BacktestPreflightError("training data contract checks failed")
        rows, scope = select_heldout_test_rows(
            frame,
            args.model_dir,
            args.symbol,
            utc_timestamp(args.start),
            utc_timestamp(args.end),
        )

        market_start = rows["decision_time_utc"].min() - config.lookback
        market_end = rows["target_end_time_utc"].max()
        trade_paths, missing_trade_dates = discover_daily_files(
            args.market_dir, "trade", market_start, market_end
        )
        quote_paths, missing_quote_dates = discover_daily_files(
            args.market_dir, "quote", market_start, market_end
        )
        if missing_trade_dates or missing_quote_dates:
            raise BacktestPreflightError(
                "required public market dates are missing: "
                f"trade={missing_trade_dates}, quote={missing_quote_dates}"
            )
        public_trades = load_public_trades(
            trade_paths, {args.symbol}, market_start, market_end
        )
        public_quotes = load_public_quotes(
            quote_paths, {args.symbol}, market_start, market_end
        )
        if public_trades.empty or public_quotes.empty:
            raise BacktestPreflightError("public trade or quote rows are empty")

        rebuilt = build_market_features(
            rows,
            public_trades,
            public_quotes,
            config.lookback,
            config.max_quote_age,
        )
        reconciliation = reconcile_market_features(
            rows, rebuilt, config.reconciliation_atol
        )
        reconciliation["market_feature_at_or_after_decision_count"] = int(
            rebuilt["market_feature_max_time_utc"]
            .ge(rebuilt["decision_time_utc"])
            .fillna(False)
            .sum()
        )
        reconciliation["market_feature_status_counts"] = {
            str(key): int(value)
            for key, value in rebuilt["market_feature_status"].value_counts().items()
        }
        reconciliation["passed"] = bool(
            reconciliation["passed"]
            and reconciliation["market_feature_at_or_after_decision_count"] == 0
            and rebuilt["market_feature_status"].eq("OK").all()
        )
        if not reconciliation["passed"]:
            raise BacktestPreflightError(
                "recomputed market features do not match the model dataset"
            )

        trader_metrics, trader_predictions = evaluate_trader_positions(
            rebuilt, predictor
        )
        sensitivity: list[dict[str, object]] = []
        base_outputs: tuple[
            dict[str, object],
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
        ] | None = None
        for multiplier in config.cost_sensitivity_multipliers:
            outputs = run_bot_backtest(
                rebuilt,
                public_trades,
                public_quotes,
                predictor,
                config,
                multiplier,
            )
            sensitivity.append(outputs[0])
            if multiplier == 1.0:
                base_outputs = outputs
        if base_outputs is None:
            raise BacktestPreflightError("cost sensitivity has no 1.0 baseline")
        bot_summary, equity_curve, trades, decisions, funding = base_outputs

        shutil.copyfile(args.config, args.output_dir / "backtest_config.toml")
        write_json(args.output_dir / "backtest_config_resolved.json", asdict(config))
        write_json(args.output_dir / "feature_reconciliation.json", reconciliation)
        write_json(
            args.output_dir / "trader_position_evaluation.json", trader_metrics
        )
        write_json(
            args.output_dir / "bot_backtest_summary.json", bot_summary
        )
        write_json(
            args.output_dir / "cost_sensitivity.json",
            {"runs": sensitivity, "random_seed_reused_per_run": config.random_seed},
        )
        trader_predictions.to_parquet(
            args.output_dir / "trader_predictions.parquet", index=False
        )
        equity_curve.to_parquet(
            args.output_dir / "bot_equity_curve.parquet", index=False
        )
        trades.to_parquet(args.output_dir / "bot_trades.parquet", index=False)
        decisions.to_parquet(args.output_dir / "bot_decisions.parquet", index=False)
        funding.to_parquet(args.output_dir / "funding_ledger.parquet", index=False)

        verified = bool(provenance["verified_actual_model_and_data"])
        assumptions_verified = config.assumption_status == "HISTORICALLY_VERIFIED"
        bot_performance_reporting_allowed = verified and assumptions_verified
        report = {
            "status": (
                "COMPLETE"
                if bot_performance_reporting_allowed
                else (
                    "COMPLETE_RESEARCH_SCENARIO"
                    if verified
                    else "ENGINEERING_SMOKE_ONLY"
                )
            ),
            "performance_claimed": bot_performance_reporting_allowed,
            "trader_behavior_score_reporting_allowed": verified,
            "bot_performance_reporting_allowed": (
                bot_performance_reporting_allowed
            ),
            "results_are_backtest_scenario_estimates": True,
            "unverified_results_must_not_be_reported_as_performance": (
                not bot_performance_reporting_allowed
            ),
            "exchange": "BitMEX",
            "symbol": args.symbol,
            "scope": scope,
            "model_loaded_once": True,
            "model_load_seconds": predictor.load_seconds,
            "feature_calculation": {
                "market": "wonyotti_pipeline.market.build_market_features",
                "model_transform": "wonyotti_pipeline.training.prepare_model_features",
                "strict_asof": "market event timestamp < decision_time_utc",
                "reconciliation": reconciliation,
            },
            "position_reports": {
                "trader_actual_position_behavior_score": trader_metrics,
                "bot_own_position_backtest": bot_summary,
            },
            "execution": {
                "fill_price_source": "PUBLIC_QUOTE_AFTER_DECISION_PLUS_LATENCY",
                "original_trader_fill_price_used": False,
                "cost_sensitivity_run_count": len(sensitivity),
            },
            "assumption_status": config.assumption_status,
            "research_assumptions_acknowledged": (
                args.acknowledge_research_assumptions
            ),
            "provenance": provenance,
            "input_counts": input_counts,
            "input_contract": input_contract,
            "market": {
                "trade_files": [str(path) for path in trade_paths],
                "quote_files": [str(path) for path in quote_paths],
                "trade_row_count": int(len(public_trades)),
                "quote_row_count": int(len(public_quotes)),
                "start_utc": market_start.isoformat(),
                "end_utc": market_end.isoformat(),
            },
            "artifacts": {
                "config_sha256": file_sha256(args.config),
                "model_sha256": file_sha256(args.model_dir / "model.cbm"),
                "dataset_sha256": file_sha256(args.dataset),
            },
            "elapsed_seconds": time.perf_counter() - started_at,
            "process_max_rss_bytes": max_rss_bytes(),
        }
        write_json(args.output_dir / "backtest_report.json", report)
        write_json(
            args.output_dir / "preflight_report.json",
            {
                "status": "PASS",
                "trader_behavior_score_reporting_allowed": verified,
                "bot_performance_reporting_allowed": (
                    bot_performance_reporting_allowed
                ),
                "provenance": provenance,
                "scope": scope,
                "feature_reconciliation_passed": reconciliation["passed"],
            },
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0
    except (
        OSError,
        ValueError,
        BacktestConfigError,
        BacktestPreflightError,
        InferenceContractError,
        TrainingDataError,
    ) as error:
        return blocked_report(args.output_dir, str(error), started_at)


if __name__ == "__main__":
    raise SystemExit(main())
