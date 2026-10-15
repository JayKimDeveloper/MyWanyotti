from __future__ import annotations

import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wonyotti_pipeline.dataset import (  # noqa: E402
    MODEL_FEATURE_COLUMNS,
    _atomic_actions,
    build_target_labels,
    decision_grid,
)
from wonyotti_pipeline.backtest import (  # noqa: E402
    Portfolio,
    _fill_candidate,
    load_backtest_config,
    run_bot_backtest,
    signed_order_for_action,
)
from wonyotti_pipeline.executions import (  # noqa: E402
    ExecutionCoverage,
    audit_orders,
    positions_before_decisions,
)
from wonyotti_pipeline.market import build_market_features  # noqa: E402
from wonyotti_pipeline.inference import (  # noqa: E402
    InferenceContractError,
    validate_feature_record,
    validate_feature_schema,
)
from wonyotti_pipeline.live_market import (  # noqa: E402
    BitmexPublicWebSocket,
    PublicMarketEvent,
    PublicMarketError,
    parse_bitmex_message,
)
from wonyotti_pipeline.paper import (  # noqa: E402
    PaperStore,
    PaperTradingEngine,
    apply_action_constraints,
    load_paper_config,
    validate_runtime_contract,
)
from wonyotti_pipeline.training import (  # noqa: E402
    FEATURE_DEFINITIONS,
    chronological_split,
    classification_metrics,
    feature_schema,
    split_leakage_checks,
    threshold_predictions,
)


def timestamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).as_unit("ns")


class PipelineTests(unittest.TestCase):
    def test_public_websocket_connect_timeout_returns_without_hanging(self) -> None:
        source = BitmexPublicWebSocket(
            "wss://example.invalid/realtime", "XBTUSD", 0.01, 0.01
        )

        def slow_connect(*args: object, **kwargs: object) -> object:
            time.sleep(0.5)
            raise OSError("late failure")

        started = time.perf_counter()
        with mock.patch(
            "wonyotti_pipeline.live_market.websocket.create_connection",
            side_effect=slow_connect,
        ):
            with self.assertRaises(PublicMarketError):
                source.connect()
        self.assertLess(time.perf_counter() - started, 0.2)

    def test_public_market_parser_and_store_deduplicate_events(self) -> None:
        received = timestamp("2018-03-05T12:00:00.100Z")
        payload = """{
          "table": "trade",
          "action": "insert",
          "data": [{
            "timestamp": "2018-03-05T12:00:00.000Z",
            "symbol": "XBTUSD",
            "side": "Buy",
            "size": 100,
            "price": 10000,
            "trdMatchID": "match-1"
          }]
        }"""
        first = parse_bitmex_message(payload, received)
        second = parse_bitmex_message(payload, received)
        self.assertEqual(first[0].event_key, second[0].event_key)
        self.assertEqual(first[0].event_key, "trade:XBTUSD:match-1")

        with tempfile.TemporaryDirectory() as directory:
            store = PaperStore(Path(directory) / "paper.sqlite3")
            self.assertEqual(store.record_market_events(first)["trade"], 1)
            self.assertEqual(store.record_market_events(second)["trade"], 0)
            trades, _ = store.market_frames(
                "XBTUSD",
                timestamp("2018-03-05T11:59:00Z"),
                timestamp("2018-03-05T12:01:00Z"),
            )
            self.assertEqual(len(trades), 1)
            store.close()

    def test_runtime_contract_and_risk_constraints(self) -> None:
        config = load_paper_config(
            PROJECT_ROOT / "configs/paper_trading.example.toml", "XBTUSD"
        )
        contract = {
            "data_contract_version": "v0.3",
            "status": "VERIFIED",
            "exchange": "BitMEX",
            "symbols": ["XBTUSD"],
            "decision_interval": "5min",
            "lookback": "5min",
            "max_quote_age": "30s",
            "strict_asof_rule": "market_time_utc < decision_time_utc",
        }
        self.assertTrue(all(validate_runtime_contract(contract, config).values()))
        self.assertEqual(
            apply_action_constraints(
                "LONG_INCREASE", 0, 1000, 10000, True, 0.0, 0.05, False
            ),
            (0, "NO_ORDER", "STALE_MARKET_REDUCE_ONLY"),
        )
        self.assertEqual(
            apply_action_constraints(
                "LONG_DECREASE", 1000, 1000, 10000, True, 0.1, 0.05, False
            ),
            (-1000, "LONG_DECREASE", "ALLOWED"),
        )
        self.assertEqual(
            apply_action_constraints(
                "LONG_INCREASE", 10000, 1000, 10000, False, 0.0, 0.05, False
            )[2],
            "MAX_POSITION_LIMIT",
        )
        self.assertEqual(
            apply_action_constraints(
                "SHORT_INCREASE", 0, 1000, 10000, False, 0.05, 0.05, False
            )[2],
            "LOSS_LIMIT_REDUCE_ONLY",
        )

    def test_paper_engine_recovers_state_and_prevents_duplicate_fill(self) -> None:
        base_config = load_paper_config(
            PROJECT_ROOT / "configs/paper_trading.example.toml", "XBTUSD"
        )

        class FakePredictor:
            class_labels = [
                "LONG_INCREASE",
                "LONG_DECREASE",
                "SHORT_INCREASE",
                "SHORT_DECREASE",
                "NO_FILL",
            ]
            loaded_at_utc = "2018-03-05T11:00:00+00:00"

            def __init__(self) -> None:
                self.positions: list[int] = []

            def predict(self, record: dict[str, object]) -> dict[str, object]:
                self.positions.append(int(record["position_before_qty"]))
                return {
                    "policy_action": "LONG_INCREASE",
                    "argmax_action": "LONG_INCREASE",
                    "probabilities": {
                        label: float(label == "LONG_INCREASE")
                        for label in self.class_labels
                    },
                }

        def trade_event(value: str, price: float, key: str) -> PublicMarketEvent:
            market_time = timestamp(value)
            return PublicMarketEvent(
                "trade",
                key,
                "XBTUSD",
                market_time,
                market_time + pd.Timedelta(milliseconds=10),
                {
                    "market_side": "Buy",
                    "market_size": 1000.0,
                    "market_price": price,
                    "trade_match_id": key,
                },
            )

        def quote_event(value: str, key: str) -> PublicMarketEvent:
            market_time = timestamp(value)
            return PublicMarketEvent(
                "quote",
                key,
                "XBTUSD",
                market_time,
                market_time + pd.Timedelta(milliseconds=10),
                {
                    "bid_size": 20000.0,
                    "bid_price": 100.0,
                    "ask_price": 101.0,
                    "ask_size": 20000.0,
                },
            )

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "paper.sqlite3"
            backtest = replace(
                base_config.backtest,
                base_fill_probability=1.0,
                slippage_bps=0.0,
                taker_rate=0.0,
            )
            config = replace(base_config, backtest=backtest, sqlite_path=database)
            predictor = FakePredictor()
            store = PaperStore(database)
            engine = PaperTradingEngine(
                config, predictor, store, "model-v1", "config-v1"
            )
            engine.on_connected(timestamp("2018-03-05T11:55:00Z"))
            engine.state["next_decision_time_utc"] = timestamp(
                "2018-03-05T12:00:00Z"
            ).isoformat()
            engine.on_market_events(
                [
                    trade_event("2018-03-05T11:55:01Z", 99.0, "trade-1"),
                    trade_event("2018-03-05T11:59:59Z", 100.0, "trade-2"),
                    quote_event("2018-03-05T11:59:59.900Z", "quote-1"),
                ]
            )
            stored_trades, stored_quotes = store.market_frames(
                "XBTUSD",
                timestamp("2018-03-05T11:55:00Z"),
                timestamp("2018-03-05T12:00:00Z"),
            )
            self.assertEqual(len(stored_trades), 2)
            self.assertEqual(len(stored_quotes), 1)
            self.assertEqual(
                engine.run_due_decisions(timestamp("2018-03-05T12:00:02Z")), 1
            )
            self.assertIsNotNone(
                engine.state["pending_order"], store.decision_records()
            )
            fill_quote = quote_event("2018-03-05T12:00:02.300Z", "quote-fill")
            self.assertEqual(engine.on_market_events([fill_quote])["quote"], 1)
            self.assertEqual(engine.on_market_events([fill_quote])["quote"], 0)
            self.assertEqual(engine.state["position_qty"], 1000)
            records = store.decision_records()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["status"], "COMPLETED")
            self.assertEqual(records[0]["virtual_fill"]["fill_status"], "FILLED")
            self.assertGreaterEqual(
                pd.Timestamp(records[0]["virtual_fill"]["fill_time_utc"]),
                pd.Timestamp(records[0]["evaluated_at_utc"])
                + config.backtest.latency,
            )
            store.close()

            recovered_store = PaperStore(database)
            recovered = PaperTradingEngine(
                config,
                FakePredictor(),
                recovered_store,
                "model-v1",
                "config-v1",
            )
            self.assertEqual(recovered.state["position_qty"], 1000)
            recovered.state["next_decision_time_utc"] = timestamp(
                "2018-03-05T12:00:00Z"
            ).isoformat()
            recovered.run_due_decisions(timestamp("2018-03-05T12:00:03Z"))
            self.assertEqual(len(recovered_store.decision_records()), 1)
            recovered_store.close()

    def test_stale_market_prediction_is_logged_but_entry_is_blocked(self) -> None:
        base_config = load_paper_config(
            PROJECT_ROOT / "configs/paper_trading.example.toml", "XBTUSD"
        )

        class FakePredictor:
            class_labels = ["LONG_INCREASE", "NO_FILL"]
            loaded_at_utc = "2018-03-05T11:00:00+00:00"

            def predict(self, record: dict[str, object]) -> dict[str, object]:
                return {
                    "policy_action": "LONG_INCREASE",
                    "argmax_action": "LONG_INCREASE",
                    "probabilities": {"LONG_INCREASE": 0.9, "NO_FILL": 0.1},
                }

        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                base_config, sqlite_path=Path(directory) / "stale.sqlite3"
            )
            store = PaperStore(config.sqlite_path)
            engine = PaperTradingEngine(
                config, FakePredictor(), store, "model-v1", "config-v1"
            )
            engine.on_connected(timestamp("2018-03-05T11:55:00Z"))
            engine.state["next_decision_time_utc"] = timestamp(
                "2018-03-05T12:00:00Z"
            ).isoformat()
            events = [
                PublicMarketEvent(
                    "trade",
                    "trade-stale",
                    "XBTUSD",
                    timestamp("2018-03-05T11:59:59Z"),
                    timestamp("2018-03-05T11:59:59.010Z"),
                    {
                        "market_side": "Buy",
                        "market_size": 1000.0,
                        "market_price": 100.0,
                        "trade_match_id": "trade-stale",
                    },
                ),
                PublicMarketEvent(
                    "quote",
                    "quote-stale",
                    "XBTUSD",
                    timestamp("2018-03-05T11:59:00Z"),
                    timestamp("2018-03-05T11:59:00.010Z"),
                    {
                        "bid_size": 20000.0,
                        "bid_price": 99.0,
                        "ask_price": 101.0,
                        "ask_size": 20000.0,
                    },
                ),
            ]
            engine.on_market_events(events)
            engine.run_due_decisions(timestamp("2018-03-05T12:00:02Z"))
            record = store.decision_records()[0]
            self.assertEqual(record["input_status"], "STALE_QUOTE")
            self.assertEqual(record["model_action"], "LONG_INCREASE")
            self.assertEqual(record["final_action"], "NO_ORDER")
            self.assertEqual(record["action_reason"], "STALE_MARKET_REDUCE_ONLY")
            self.assertEqual(record["probabilities"]["LONG_INCREASE"], 0.9)
            store.close()

    def test_inverse_portfolio_pnl_and_funding_signs(self) -> None:
        config = load_backtest_config(
            PROJECT_ROOT / "configs/backtest.example.toml", "XBTUSD"
        )
        portfolio = Portfolio.initialize(config.instrument, 1.0, 0, 10_000.0)
        portfolio.apply_fill(100, 10_000.0, 0.0)
        expected_gain = 100 * (1 / 10_000.0 - 1 / 11_000.0)
        self.assertAlmostEqual(portfolio.equity(11_000.0), 1.0 + expected_gain)
        self.assertLess(portfolio.apply_funding(11_000.0, 0.0001), 0.0)

        short = Portfolio.initialize(config.instrument, 1.0, -100, 10_000.0)
        self.assertGreater(short.apply_funding(10_000.0, 0.0001), 0.0)

    def test_action_interpretation_uses_current_bot_position(self) -> None:
        self.assertEqual(signed_order_for_action("LONG_INCREASE", 0, 100), (100, "ORDER"))
        self.assertEqual(signed_order_for_action("LONG_DECREASE", 60, 100), (-60, "ORDER"))
        self.assertEqual(
            signed_order_for_action("LONG_DECREASE", -60, 100),
            (0, "INVALID_FOR_POSITION"),
        )
        self.assertEqual(signed_order_for_action("SHORT_INCREASE", 0, 100), (-100, "ORDER"))
        self.assertEqual(signed_order_for_action("SHORT_DECREASE", -60, 100), (60, "ORDER"))
        self.assertEqual(signed_order_for_action("NO_FILL", 0, 100), (0, "NO_ACTION"))

    def test_fill_uses_post_decision_public_quote_and_adverse_slippage(self) -> None:
        config = replace(
            load_backtest_config(
                PROJECT_ROOT / "configs/backtest.example.toml", "XBTUSD"
            ),
            base_fill_probability=1.0,
        )
        decision = timestamp("2018-03-05T12:00:00Z")
        quotes = pd.DataFrame(
            {
                "market_time_utc": [
                    decision - pd.Timedelta(milliseconds=1),
                    decision + pd.Timedelta(milliseconds=300),
                ],
                "bid_price": [99.0, 100.0],
                "ask_price": [100.0, 101.0],
                "bid_size": [20_000.0, 20_000.0],
                "ask_size": [20_000.0, 20_000.0],
            }
        )
        fill = _fill_candidate(
            quotes,
            decision,
            decision + pd.Timedelta(minutes=5),
            1_000,
            config,
            1.0,
            np.random.default_rng(42),
        )
        self.assertEqual(
            fill["quote_time_utc"], decision + pd.Timedelta(milliseconds=300)
        )
        self.assertEqual(fill["reference_price"], 101.0)
        self.assertGreater(fill["fill_price"], fill["reference_price"])
        self.assertEqual(fill["price_source"], "PUBLIC_QUOTE_AFTER_DECISION")

    def test_backtest_feeds_bot_position_back_into_shared_features(self) -> None:
        config = replace(
            load_backtest_config(
                PROJECT_ROOT / "configs/backtest.example.toml", "XBTUSD"
            ),
            base_fill_probability=1.0,
            funding_mode="disabled",
            taker_rate=0.0,
            slippage_bps=0.0,
        )
        decisions = [
            timestamp("2018-03-05T12:00:00Z"),
            timestamp("2018-03-05T12:05:00Z"),
        ]
        rows: list[dict[str, object]] = []
        for decision in decisions:
            record: dict[str, object] = {}
            for name in MODEL_FEATURE_COLUMNS:
                input_type = FEATURE_DEFINITIONS[name]["input_type"]
                if name == "symbol":
                    record[name] = "XBTUSD"
                elif name == "decision_time_utc":
                    record[name] = decision
                elif input_type == "integer":
                    record[name] = 0
                else:
                    record[name] = 1.0
            record["target_end_time_utc"] = decision + pd.Timedelta(minutes=5)
            rows.append(record)
        feature_rows = pd.DataFrame(rows)
        public_trades = pd.DataFrame(
            {
                "symbol": ["XBTUSD", "XBTUSD", "XBTUSD"],
                "market_time_utc": [
                    decisions[0] - pd.Timedelta(seconds=1),
                    decisions[1] - pd.Timedelta(seconds=1),
                    decisions[1] + pd.Timedelta(minutes=4, seconds=59),
                ],
                "market_price": [100.0, 101.0, 102.0],
            }
        )
        public_quotes = pd.DataFrame(
            {
                "symbol": ["XBTUSD", "XBTUSD"],
                "market_time_utc": [
                    decisions[0] + pd.Timedelta(seconds=1),
                    decisions[1] + pd.Timedelta(seconds=1),
                ],
                "bid_price": [99.0, 100.0],
                "ask_price": [100.0, 101.0],
                "bid_size": [20_000.0, 20_000.0],
                "ask_size": [20_000.0, 20_000.0],
            }
        )

        class FakePredictor:
            class_labels = ["LONG_INCREASE", "LONG_DECREASE"]

            def __init__(self) -> None:
                self.positions: list[int] = []

            def predict(self, record: dict[str, object]) -> dict[str, object]:
                self.positions.append(int(record["position_before_qty"]))
                action = "LONG_INCREASE" if len(self.positions) == 1 else "LONG_DECREASE"
                return {
                    "policy_action": action,
                    "argmax_action": action,
                    "confidence": 1.0,
                    "probabilities": {
                        "LONG_INCREASE": float(action == "LONG_INCREASE"),
                        "LONG_DECREASE": float(action == "LONG_DECREASE"),
                    },
                }

        predictor = FakePredictor()
        summary, _, trades, decisions_frame, _ = run_bot_backtest(
            feature_rows,
            public_trades,
            public_quotes,
            predictor,  # type: ignore[arg-type]
            config,
            1.0,
        )
        self.assertEqual(predictor.positions, [0, 1_000])
        self.assertEqual(decisions_frame.bot_position_after_qty.tolist(), [1_000, 0])
        self.assertEqual(len(trades), 2)
        self.assertGreater(summary["time_exposure_fraction"], 0.0)

    def test_inference_schema_checks_columns_types_and_units(self) -> None:
        record: dict[str, object] = {}
        for name in MODEL_FEATURE_COLUMNS:
            input_type = FEATURE_DEFINITIONS[name]["input_type"]
            if input_type == "string":
                record[name] = "XBTUSD"
            elif input_type == "timezone_aware_datetime":
                record[name] = "2018-03-05T12:05:00Z"
            elif input_type == "integer":
                record[name] = 1
            else:
                record[name] = 1.0
        raw = pd.DataFrame([record])
        raw["decision_time_utc"] = pd.to_datetime(raw["decision_time_utc"], utc=True)
        schema = feature_schema(raw)

        validate_feature_schema(schema)
        prepared = validate_feature_record(record)
        self.assertEqual(prepared.columns.tolist(), MODEL_FEATURE_COLUMNS)
        self.assertEqual(schema["fields"][2]["unit"], "CONTRACTS")

        invalid = dict(record)
        invalid["decision_time_utc"] = "2018-03-05T12:05:00"
        with self.assertRaises(InferenceContractError):
            validate_feature_record(invalid)

    def test_chronological_split_purges_boundaries_and_cross_split_orders(self) -> None:
        frame = pd.DataFrame(
            {
                "decision_time_utc": [
                    timestamp("2018-03-05T00:00:00Z"),
                    timestamp("2018-03-05T00:05:00Z"),
                    timestamp("2018-03-05T00:10:00Z"),
                    timestamp("2018-03-05T00:15:00Z"),
                    timestamp("2018-03-05T00:20:00Z"),
                ],
                "target_end_time_utc": [
                    timestamp("2018-03-05T00:05:00Z"),
                    timestamp("2018-03-05T00:11:00Z"),
                    timestamp("2018-03-05T00:15:00Z"),
                    timestamp("2018-03-05T00:20:00Z"),
                    timestamp("2018-03-05T00:25:00Z"),
                ],
                "target_order_ids": ["shared", "", "shared", "validation", "test"],
                "target_label": ["NO_FILL"] * 5,
            }
        )
        train_end = timestamp("2018-03-05T00:10:00Z")
        validation_end = timestamp("2018-03-05T00:20:00Z")
        splits, report = chronological_split(frame, train_end, validation_end)
        checks = split_leakage_checks(splits, train_end, validation_end)

        self.assertEqual(report["boundary_crossing_row_count"], 1)
        self.assertEqual(report["cross_split_order_count"], 1)
        self.assertEqual(report["order_purged_row_count"], 2)
        self.assertTrue(checks["passed"])
        self.assertEqual(len(splits["validation"]), 1)
        self.assertEqual(len(splits["test"]), 1)

    def test_metrics_and_threshold_fallback_use_declared_class_order(self) -> None:
        class_labels = ["NO_FILL", "LONG_INCREASE"]
        probabilities = np.array([[0.6, 0.4], [0.2, 0.8]])
        predictions = threshold_predictions(
            probabilities, class_labels, 0.7, "NO_FILL"
        )
        metrics = classification_metrics(
            ["NO_FILL", "LONG_INCREASE"], predictions, class_labels
        )

        self.assertEqual(predictions, ["NO_FILL", "LONG_INCREASE"])
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(
            metrics["confusion_matrix"]["row_order_actual"], class_labels
        )

    def test_model_feature_allowlist_excludes_outcome_fields(self) -> None:
        forbidden = ("target_", "next_", "pnl", "profit")
        self.assertFalse(
            any(
                token in column.lower()
                for column in MODEL_FEATURE_COLUMNS
                for token in forbidden
            )
        )

    def test_decision_grid_keeps_only_complete_target_intervals(self) -> None:
        grid = decision_grid(
            ["XBTUSD"],
            timestamp("2018-03-05T12:00:00Z"),
            timestamp("2018-03-05T12:12:00Z"),
            pd.Timedelta("5min"),
        )
        self.assertEqual(
            grid.decision_time_utc.tolist(),
            [
                timestamp("2018-03-05T12:00:00Z"),
                timestamp("2018-03-05T12:05:00Z"),
            ],
        )

    def test_atomic_actions_cover_increase_decrease_and_flip(self) -> None:
        self.assertEqual(_atomic_actions(0, 10), (["LONG_INCREASE"], 10))
        self.assertEqual(_atomic_actions(10, -4), (["LONG_DECREASE"], 6))
        self.assertEqual(_atomic_actions(0, -10), (["SHORT_INCREASE"], -10))
        self.assertEqual(_atomic_actions(-10, 4), (["SHORT_DECREASE"], -6))
        self.assertEqual(
            _atomic_actions(-5, 8),
            (["SHORT_DECREASE", "LONG_INCREASE"], 3),
        )

    def test_order_audit_recognizes_complete_partial_fills(self) -> None:
        rows = pd.DataFrame(
            {
                "exectype": ["Trade", "Trade"],
                "orderid": ["order-1", "order-1"],
                "symbol": ["XBTUSD", "XBTUSD"],
                "side": ["Buy", "Buy"],
                "lastqty": [3, 7],
                "cumqty": [3, 10],
                "leavesqty": [7, 0],
                "orderqty": [10, 10],
                "ordstatus": ["PartiallyFilled", "Filled"],
                "event_time_utc": [
                    timestamp("2018-03-05T00:00:01Z"),
                    timestamp("2018-03-05T00:00:02Z"),
                ],
                "processed_time_utc": [
                    timestamp("2018-03-05T00:00:01Z"),
                    timestamp("2018-03-05T00:00:02Z"),
                ],
                "execid": ["exec-1", "exec-2"],
            }
        )
        audit = audit_orders(rows).iloc[0]
        self.assertEqual(audit.fill_row_count, 2)
        self.assertTrue(audit.increments_match_lastqty)
        self.assertTrue(audit.sum_matches_max_cumqty)

    def test_position_and_market_features_exclude_decision_boundary(self) -> None:
        decision_time = timestamp("2018-03-05T12:05:00Z")
        decisions = pd.DataFrame(
            {"symbol": ["XBTUSD"], "decision_time_utc": [decision_time]}
        )
        executions = pd.DataFrame(
            {
                "symbol": ["XBTUSD", "XBTUSD"],
                "exectype": ["Trade", "Trade"],
                "event_time_utc": [
                    timestamp("2018-03-05T12:04:00Z"),
                    decision_time,
                ],
                "trade_signed_qty": [10, 5],
            }
        )
        anchors = pd.DataFrame(
            {
                "symbol": ["XBTUSD"],
                "event_time_utc": [timestamp("2018-03-05T12:00:00Z")],
                "execid": ["anchor"],
                "position_qty": [0],
                "anchor_source": ["TEST"],
            }
        )
        positioned = positions_before_decisions(decisions, executions, anchors)
        self.assertEqual(positioned.position_before_qty.iloc[0], 10)

        market_trades = pd.DataFrame(
            {
                "symbol": ["XBTUSD", "XBTUSD"],
                "market_time_utc": [
                    timestamp("2018-03-05T12:04:00Z"),
                    decision_time,
                ],
                "market_size": [10, 999],
                "market_price": [100.0, 999.0],
                "market_side": ["Buy", "Buy"],
            }
        )
        quotes = pd.DataFrame(
            {
                "symbol": ["XBTUSD", "XBTUSD"],
                "market_time_utc": [
                    timestamp("2018-03-05T12:04:30Z"),
                    decision_time,
                ],
                "bid_price": [99.0, 998.0],
                "ask_price": [101.0, 1000.0],
                "bid_size": [10.0, 10.0],
                "ask_size": [10.0, 10.0],
            }
        )
        featured = build_market_features(
            decisions,
            market_trades,
            quotes,
            pd.Timedelta("5min"),
            pd.Timedelta("1min"),
        )
        self.assertEqual(featured.market_trade_count.iloc[0], 1)
        self.assertEqual(featured.market_last_price.iloc[0], 100.0)
        self.assertEqual(featured.bid_price.iloc[0], 99.0)

    def test_target_includes_decision_boundary_and_marks_mixed(self) -> None:
        decision_time = timestamp("2018-03-05T12:05:00Z")
        decisions = pd.DataFrame(
            {
                "symbol": ["XBTUSD"],
                "decision_time_utc": [decision_time],
                "position_before_qty": pd.array([0], dtype="Int64"),
            }
        )
        executions = pd.DataFrame(
            {
                "symbol": ["XBTUSD", "XBTUSD"],
                "exectype": ["Trade", "Trade"],
                "event_time_utc": [
                    decision_time,
                    timestamp("2018-03-05T12:06:00Z"),
                ],
                "processed_time_utc": [
                    decision_time,
                    timestamp("2018-03-05T12:06:00Z"),
                ],
                "orderid": ["buy", "sell"],
                "cumqty": [10, 15],
                "execid": ["exec-buy", "exec-sell"],
                "side": ["Buy", "Sell"],
                "lastqty": [10, 15],
                "trade_signed_qty": [10, -15],
                "ordstatus": ["Filled", "Filled"],
            }
        )
        coverage = [
            ExecutionCoverage(
                timestamp("2018-03-05T00:00:00Z"),
                timestamp("2018-03-06T00:00:00Z"),
                "test.csv",
            )
        ]
        labels = build_target_labels(
            decisions,
            executions,
            pd.Timedelta("5min"),
            coverage,
            invalid_order_ids={"buy"},
        )
        self.assertEqual(labels.target_label.iloc[0], "MIXED")
        self.assertEqual(labels.target_fill_count.iloc[0], 2)
        self.assertEqual(labels.target_order_ids.iloc[0], "buy|sell")
        self.assertTrue(labels.target_order_integrity_failed.iloc[0])


if __name__ == "__main__":
    unittest.main()
