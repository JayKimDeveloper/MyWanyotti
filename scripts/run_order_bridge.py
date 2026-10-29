#!/usr/bin/env python3
"""Build a descriptive order view and replay one policy entirely offline."""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wonyotti_pipeline.backtest import funding_times
from wonyotti_pipeline.candles import HOUR, atomic_json, atomic_parquet
from wonyotti_pipeline.executions import parse_utc
from wonyotti_pipeline.order_bridge import ACTIVITY_THRESHOLD, order_view, replay
from wonyotti_pipeline.paper import load_paper_config
from wonyotti_pipeline.side_research import load_research_model, prepare_model_features, research_splits
from wonyotti_pipeline.training import file_sha256

V2 = ROOT / "models/catboost/candle_side_research_v2_365d"
V3 = ROOT / "models/catboost/candle_activity_research_v3_unweighted"
CACHE = ROOT / "data/raw/market/bitmex/candle_v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/audit/order_bridge_v1")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError("preserve existing bridge output; choose a new output directory")
    config_path = ROOT / "configs/order_bridge.toml"
    config = load_paper_config(config_path, "XBTUSD")
    document = tomllib.loads(config_path.read_text())
    if document["policy"]["activity_threshold"] != ACTIVITY_THRESHOLD:
        raise ValueError("V3 threshold must remain unchanged")
    if document["policy"]["execution"] != "NEXT_HOURLY_CLOSE_AFTER_ARRIVAL" or document["policy"]["funding"] != "EXPORTED_FUNDING_COMMISSION_IF_COMPLETE_ELSE_MISSING":
        raise ValueError("unsupported frozen policy")
    if config.decision_interval != HOUR:
        raise ValueError("hourly policy required")
    protected = {str(path.relative_to(ROOT)): file_sha256(path)
                 for directory in ("data", "models", "configs")
                 for path in sorted((ROOT / directory).rglob("*")) if path.is_file()}
    implementation = {str(path.relative_to(ROOT)): file_sha256(path) for path in
                      [Path(__file__), ROOT / "src/wonyotti_pipeline/order_bridge.py",
                       ROOT / "src/wonyotti_pipeline/backtest.py", ROOT / "src/wonyotti_pipeline/paper.py"]}
    output.mkdir(parents=True)
    (output / "policy_config.toml").write_text(config_path.read_text())
    protocol = {"frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "policy": document, "protected_artifact_hashes": protected,
                "implementation_hashes": implementation, "parameter_search": False,
                "initial_state": "Synthetic bot: flat, 1 XBT; not teacher account equity or positions",
                "execution": "Next hourly close strictly after arrival; 250ms arrival latency; 3600s fill delay; full simulated taker fill, adverse 2bps plus tick rounding",
                "funding": "Use exported Funding commission as historical signed rate only with complete 8h UTC coverage; otherwise exclude all funding",
                "same_timestamp_priority": "funding, pending fill, mark/risk, decision",
                "terminal_rule": "One additional reduce-only decision at the next hour; execute its close one hour later",
                "unused_quote_settings": ["max_fill_wait_ms", "base_fill_probability", "max_top_of_book_participation", "allow_partial_fills", "random_seed"],
                "interval_status": "ALREADY_INSPECTED_EXPLORATORY_INTERVAL"}
    atomic_json(output / "frozen_protocol.json", protocol)

    # Descriptive profiling uses the original purged training and validation keys only.
    metadata = json.loads((V2 / "data_split.json").read_text())
    dataset = pd.read_parquet(ROOT / "data/processed/training/candle_side_research_v2_365d/training_eligible.parquet")
    splits, split_report = research_splits(dataset, pd.Timestamp(metadata["train_end_utc"]), pd.Timestamp(metadata["validation_end_utc"]))
    if not split_report["leakage_checks"]["passed"] or split_report["retained_counts"] != metadata["retained_counts"]:
        raise ValueError("existing split identity/count checks failed")
    split_keys = {time: name for name in ("train", "validation") for time in splits[name].decision_time_utc}
    raw_2018 = ROOT / "data/raw/aoa-execution-2018-03-01-2018-12-31.csv"
    parts = []
    for chunk in pd.read_csv(raw_2018, chunksize=100000, low_memory=False, dtype={"triggered": "string"}):
        chunk = chunk[chunk.symbol.eq("XBTUSD")].copy()
        chunk["event_time_utc"] = parse_utc(chunk.transacttime)
        chunk["processed_time_utc"] = parse_utc(chunk.timestamp)
        chunk["profile_split"] = chunk.event_time_utc.dt.floor("h").map(split_keys)
        parts.append(chunk[chunk.profile_split.notna()])
    orders, events, unidentified = order_view(pd.concat(parts, ignore_index=True))
    atomic_parquet(output / "orders.parquet", orders)
    atomic_parquet(output / "order_events.parquet", events)
    atomic_parquet(output / "unidentified_fills.parquet", unidentified)
    coverage_source = ROOT / "data/audit/account_20261012/csv_profile.json"
    coverage = json.loads(coverage_source.read_text())
    file_profile = next(item for item in coverage["files"] if Path(item["path"]).name == raw_2018.name)
    profile = {"scope": "Descriptive original purged train/validation windows; no policy tuning",
               "windows": {name: len(splits[name]) for name in ("train", "validation")},
               "existing_field_coverage_source": str(coverage_source.relative_to(ROOT)),
               "existing_field_coverage_whole_2018_file": file_profile["column_profiles"],
               "existing_execType_whole_2018_file": file_profile["categorical_profiles"]["exectype"],
               "selected_xbtusd_execType_counts": events.groupby("profile_split").exectype.value_counts().to_dict(),
               "selected_event_count": len(events), "valid_order_count": len(orders),
               "unidentified_fills": len(unidentified),
               "zero_uuid_fills": int(unidentified.order_identity_status.eq("ZERO_UUID").sum()),
               "multi_fill_orders": int(orders.fill_row_count.gt(1).sum()),
               "missing_new_events": int(orders.missing_new_event.sum()),
               "missing_terminal_events": int(orders.missing_terminal_event.sum()),
               "ambiguous_orders": int(orders.ambiguous_ordering.sum()),
               "quantity_quantiles_contracts": orders.max_cum_qty.quantile([0, .25, .5, .75, 1]).to_dict(),
               "quantity_measure": "Maximum observed cumQty; observed orderQty values retained, never summed",
               "liquidity_counts": events[events.exectype.eq("Trade")].lastliquidityind.value_counts().to_dict(),
               "commission_units": "commission is recorded rate; execcomm retains source XBt integer amounts",
               "positions": "Not reconstructed; starting positions and account equity remain UNKNOWN",
               "ordering": "Timestamp ties flagged even when cumQty allows deterministic display order"}
    profile["selected_xbtusd_execType_counts"] = {
        f"{name}/{event}": count for (name, event), count in profile["selected_xbtusd_execType_counts"].items()}
    atomic_json(output / "order_profile.json", profile)

    holdout = pd.read_parquet(ROOT / "data/processed/training/candle_activity_research_v3_unweighted/holdout_eligible.parquet")
    saved = pd.read_parquet(V3 / "holdout_predictions.parquet")
    keys = ["symbol", "decision_time_utc"]
    pd.testing.assert_frame_equal(holdout[keys], saved[keys])
    if len(saved) != 2134 or not holdout.candle_end_utc.lt(holdout.decision_time_utc).all():
        raise ValueError("expected existing 2,134-window causal holdout")
    predictions = saved[keys + ["v3_unweighted_score"]].copy()
    model, labels = load_research_model(V2, "candle_side_research_v2_365d")
    probabilities = model.predict_proba(prepare_model_features(holdout))
    # Saved V2 activity scores check same model/feature compatibility, not model quality.
    np.testing.assert_allclose(1 - probabilities[:, labels.index("NO_RECORDED_FILL")], saved.v2_activity_score, rtol=0, atol=1e-12)
    predictions["v2_argmax"] = np.asarray(labels)[probabilities.argmax(axis=1)]
    for i, label in enumerate(labels):
        predictions["v2_probability__" + label] = probabilities[:, i]
    atomic_parquet(output / "policy_predictions.parquet", predictions)
    start = predictions.decision_time_utc.min()
    end = predictions.decision_time_utc.max() + 2 * HOUR
    checkpoint = json.loads((CACHE / "download_checkpoint.json").read_text())
    frames, candle_hashes = [], {}
    for day in pd.date_range((start - HOUR).normalize(), end.normalize(), freq="D"):
        key = f"{day:%Y%m%d}"
        path = CACHE / f"{key}.parquet"
        digest = file_sha256(path)
        if digest != checkpoint["days"][key]["sha256"]:
            raise ValueError(f"cache checksum mismatch: {path}")
        frame = pd.read_parquet(path)
        expected = pd.date_range(day + HOUR, day + pd.Timedelta(days=1), freq=HOUR)
        if frame.candle_end_utc.tolist() != expected.tolist():
            raise ValueError(f"cache boundaries mismatch: {path}")
        frames.append(frame)
        candle_hashes[str(path.relative_to(ROOT))] = digest

    raw_2019 = ROOT / "data/raw/aoa-execution-2019-01-01-2020-12-31.csv"
    observed = []
    for chunk in pd.read_csv(raw_2019, chunksize=100000, usecols=["symbol", "exectype", "transacttime", "commission"]):
        chunk = chunk[chunk.symbol.eq("XBTUSD") & chunk.exectype.eq("Funding")].copy()
        chunk["funding_time_utc"] = parse_utc(chunk.transacttime)
        observed.append(chunk[chunk.funding_time_utc.between(start, end)])
    rates = pd.concat(observed).rename(columns={"commission": "funding_rate"})[["funding_time_utc", "funding_rate"]]
    expected_funding = pd.DatetimeIndex(funding_times(start, end + pd.Timedelta(nanoseconds=1), config.backtest.funding_interval, config.backtest.funding_offset_hour_utc))
    complete = (not rates.funding_time_utc.duplicated().any() and np.isfinite(rates.funding_rate).all()
                and sorted(rates.funding_time_utc.tolist()) == expected_funding.tolist())
    funding_status = "RECORDED_HISTORICAL_RATES_NOT_INDEPENDENTLY_VERIFIED" if complete else "MISSING_RESULTS_EXCLUDE_FUNDING"
    atomic_parquet(output / "observed_funding_rates.parquet", rates)
    summary, decisions, fills, equity, payments = replay(predictions, pd.concat(frames, ignore_index=True),
        rates if complete else rates.iloc[:0], config)
    for name, frame in (("decisions", decisions), ("fills", fills), ("equity_curve", equity), ("funding_ledger", payments)):
        atomic_parquet(output / f"{name}.parquet", frame)
    changed = [name for name, digest in protected.items() if file_sha256(ROOT / name) != digest]
    if changed:
        raise ValueError(f"protected sources changed: {changed}")
    if file_sha256(output / "policy_config.toml") != protected["configs/order_bridge.toml"]:
        raise ValueError("frozen config changed")
    checks = {"causal_execution": bool(fills.fill_time_utc.gt(fills.arrival_time_utc).all()),
              "no_additions_or_reversals": bool(((fills.position_before_qty.eq(0) & fills.position_after_qty.abs().eq(100)) |
                  (fills.position_before_qty.abs().eq(100) & fills.position_after_qty.eq(0))).all()),
              "exposure_limit": bool(equity.position_qty.abs().le(config.max_abs_position_contracts).all()),
              "terminal_flat": bool(equity.position_qty.iloc[-1] == 0),
              "protected_files_unchanged": len(protected), "saved_v2_score_max_difference": float(np.max(np.abs(
                  1 - probabilities[:, labels.index("NO_RECORDED_FILL")] - saved.v2_activity_score))),
              "config_frozen_before_replay": True, "downloads": 0, "real_orders": 0}
    report = {"implementation_checks": checks, "order_profile_summary": {key: profile[key] for key in
        ["windows", "valid_order_count", "unidentified_fills", "missing_new_events", "missing_terminal_events", "ambiguous_orders"]},
        "exploratory_trading_results": summary, "interval_status": "ALREADY_INSPECTED_EXPLORATORY_INTERVAL",
        "decision_start_utc": start.isoformat(), "last_holdout_decision_utc": predictions.decision_time_utc.max().isoformat(),
        "terminal_exit_time_utc": end.isoformat(), "funding_status": funding_status,
        "expected_funding_events": len(expected_funding), "observed_funding_events": len(rates),
        "economic_assumptions": protocol["execution"], "candle_files": candle_hashes,
        "limitations": ["New bot policy, not a reconstruction of teacher positions",
            "Initial 1 XBT is a hypothetical bot bankroll, never inferred from notional",
            "Historical fee tier, lot/tick specifications, spreads, capacity and liquidation not verified",
            "Full deterministic fills at hourly closes; no resting limits or teacher maker rebates",
            "Drawdown sampled at hourly boundaries; intrahour drawdown is unmeasured",
            "Funding rates are account-export observations; missing account events can imply missing rates",
            "UTC and completeness assumptions remain unverified; UNKNOWN evidence claims unchanged",
            "No retraining, parameter search, bootstrap, or repeated evidence/model audit"]}
    atomic_json(output / "report.json", report)
    text = f"""# Order-aware research bridge v1

Implementation: {len(saved):,} existing V3 windows; exact saved keys; V2 activity-score compatibility maximum difference {checks['saved_v2_score_max_difference']}; {len(protected):,} protected files unchanged. Configuration frozen at {protocol['frozen_at_utc']} before policy replay. No downloads, training or real orders. Descriptive profiling uses {len(splits['train']):,} train and {len(splits['validation']):,} validation windows only. Full selected raw events and separate unidentified fills preserve quantities, partial fills, instructions, liquidity, status and commissions. orderQty is never summed. Missing lifecycle events and ordering ties remain explicit.

Policy: unchanged V3 threshold {ACTIVITY_THRESHOLD}; when flat and flagged, original V2 four-class argmax requests +100 for BUY_ONLY or -100 for SELL_ONLY. MIXED/NO_RECORDED_FILL request no entry. Next-hour decision closes the bot position regardless of scores. No additions, same-decision reversals or overlapping orders. Existing maximum position {config.max_abs_position_contracts:,} contracts and session-loss limit {config.max_session_loss_settlement} XBT apply with REDUCE_ONLY behavior. A final close decision extends execution to {end.isoformat()}.

Economic assumptions: hypothetical flat bot with {config.backtest.initial_equity} XBT; USD prices, contract quantities, inverse XBT settlement. PnL = Q*(1/entry-1/exit); fee = abs(Q)/fill_price*{config.backtest.taker_rate}. {protocol['execution']}. Candle timestamps are bucket ends, starts are one hour earlier; candle open is the previous close and is not used as a post-decision fill. Only cached closes enter execution and marks. Slippage is already embedded in PnL and is not subtracted twice. Full fills assume capacity; spreads, queue position, intrahour prices and liquidation are unmeasured. Funding: {funding_status}; {len(rates)}/{len(expected_funding)} expected events. Funding uses only recorded Funding commission rates and market-candle marks, never teacher positions, cashflows or fill prices. Same-time funding precedes fills.

Exploratory results on an **already-inspected interval**, {start.isoformat()} through the last holdout decision {predictions.decision_time_utc.max().isoformat()}: net PnL {summary['net_pnl_xbt']:.10f} XBT; return {summary['total_return']:.6%}; hourly sampled drawdown {summary['max_hourly_sampled_drawdown']:.6%}; {summary['round_trips']} round trips; turnover {summary['turnover_contracts']:,} contracts. Fees {summary['fees_xbt']:.10f} XBT; embedded slippage cost {summary['slippage_cost_xbt']:.10f} XBT; funding payment {summary['funding_payment_xbt']:.10f} XBT. Holding seconds: {json.dumps(summary['holding_seconds'])}. Skipped decisions: {json.dumps(summary['skipped_decisions_by_reason'])}. Flat baseline: 1 XBT, zero return, costs and drawdown. These are simulated results under explicit assumptions, not actual trading or evidence of generalization.

No teacher starting positions or account equity are inferred. Lifecycle observations may be clipped by profile windows; deterministic display ordering does not resolve tied timestamps. Historical specifications, funding field interpretation, UTC and account completeness are not independently verified. Production evidence and UNKNOWN claims remain unchanged. Focused engineering tests are reported separately in validation.json.
"""
    (output / "report.md").write_text(text)
    print(json.dumps({"output_dir": str(output), "checks": checks, "funding_status": funding_status, "results": summary}, indent=2))


if __name__ == "__main__":
    main()
