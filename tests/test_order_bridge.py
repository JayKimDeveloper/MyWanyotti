"""Synthetic bridge fixtures verify engineering, never trading performance."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wonyotti_pipeline.backtest import Portfolio
from wonyotti_pipeline.candles import HOUR
from wonyotti_pipeline.executions import ZERO_UUID
from wonyotti_pipeline.order_bridge import ACTIVITY_THRESHOLD, order_view, policy_order, replay
from wonyotti_pipeline.paper import load_paper_config


class OrderBridgeTests(unittest.TestCase):
    def setUp(self):
        self.config = load_paper_config(ROOT / "configs/order_bridge.toml", "XBTUSD")
        self.start = pd.Timestamp("2020-01-01T00:00:00Z")

    def fixtures(self):
        times = pd.date_range(self.start, periods=4, freq=HOUR)
        predictions = pd.DataFrame({"symbol": "XBTUSD", "decision_time_utc": times[:2],
                                    "v3_unweighted_score": 1., "v2_argmax": ["BUY_ONLY", "SELL_ONLY"]})
        candles = pd.DataFrame({"symbol": "XBTUSD", "candle_start_utc": times - HOUR,
                                "candle_end_utc": times, "close": [10000., 10000., 11000., 9000.]})
        funding = pd.DataFrame(columns=["funding_time_utc", "funding_rate"])
        return predictions, candles, funding

    def test_policy_threshold_four_classes_close_and_limits(self):
        self.assertEqual(policy_order(0, ACTIVITY_THRESHOLD, "BUY_ONLY", self.config, 1)[0], 100)
        self.assertEqual(policy_order(0, ACTIVITY_THRESHOLD, "SELL_ONLY", self.config, 1)[0], -100)
        self.assertEqual(policy_order(0, ACTIVITY_THRESHOLD - 1e-9, "BUY_ONLY", self.config, 1)[0], 0)
        for side in ("MIXED", "NO_RECORDED_FILL"):
            self.assertEqual(policy_order(0, 1, side, self.config, 1)[0], 0)
        for position in (-100, 100):
            self.assertEqual(policy_order(position, 0, "MIXED", self.config, .8)[0], -position)
        self.assertEqual(policy_order(0, 1, "BUY_ONLY", self.config, .94)[1], "LOSS_LIMIT_REDUCE_ONLY")
        self.assertEqual(policy_order(0, 1, "BUY_ONLY", self.config, 1, True)[1], "PENDING_ORDER_EXISTS")
        self.assertEqual(policy_order(0, 1, "BUY_ONLY", replace(self.config, max_abs_position_contracts=50), 1)[1], "MAX_POSITION_LIMIT")

    def test_causal_execution_no_same_decision_reversal_and_inverse_costs(self):
        predictions, candles, funding = self.fixtures()
        summary, decisions, fills, curve, _ = replay(predictions, candles, funding, self.config)
        self.assertEqual(fills.signed_requested_qty.tolist(), [100, -100])
        self.assertEqual(fills.fill_time_utc.tolist(), [self.start + HOUR, self.start + 2 * HOUR])
        self.assertTrue(fills.fill_time_utc.gt(fills.arrival_time_utc).all())
        self.assertEqual(fills.position_after_qty.tolist(), [100, 0])
        self.assertEqual(fills.holding_seconds.dropna().tolist(), [3600.])
        entry, exit = fills.fill_price
        expected_fee = (100 / entry + 100 / exit) * self.config.backtest.taker_rate
        self.assertAlmostEqual(summary["fees_xbt"], expected_fee)
        self.assertAlmostEqual(summary["net_pnl_xbt"], 100 * (1 / entry - 1 / exit) - expected_fee)
        self.assertGreater(summary["slippage_cost_xbt"], 0)
        self.assertEqual(curve.position_qty.iloc[-1], 0)
        self.assertTrue(decisions.terminal_close_decision.iloc[-1])
        # A later market price cannot change an earlier fill or decision.
        candles.loc[3, "close"] = 50000
        changed = replay(predictions, candles, funding, self.config)
        pd.testing.assert_frame_equal(fills, changed[2])
        pd.testing.assert_frame_equal(decisions, changed[1])

    def test_short_inverse_pnl_fees_and_funding_sign(self):
        portfolio = Portfolio.initialize(self.config.backtest.instrument, 1, 0, 10000)
        portfolio.apply_fill(-100, 10000, portfolio.fee(100, 10000, .00075))
        self.assertAlmostEqual(portfolio.apply_funding(10000, .001), .00001)
        portfolio.apply_fill(100, 9000, portfolio.fee(100, 9000, .00075))
        expected = 1 - 100 * (1 / 10000 - 1 / 9000) - .00075 * (100 / 10000 + 100 / 9000) + .00001
        self.assertAlmostEqual(portfolio.equity(9000), expected)

    def test_funding_before_tied_fill_and_terminal_exit(self):
        predictions, candles, funding = self.fixtures()
        predictions.v2_argmax = "SELL_ONLY"
        funding = pd.DataFrame({"funding_time_utc": [self.start + HOUR, self.start + 2 * HOUR], "funding_rate": .001})
        summary, _, fills, _, payments = replay(predictions, candles, funding, self.config)
        self.assertEqual(payments.position_qty.tolist(), [0, -100])
        self.assertAlmostEqual(summary["funding_payment_xbt"], 100 / 11000 * .001)
        self.assertEqual(fills.position_after_qty.iloc[-1], 0)
        # A final holdout entry closes on the added terminal decision, not prematurely.
        predictions.loc[0, "v3_unweighted_score"] = 0
        result = replay(predictions, candles, funding, self.config)
        self.assertEqual(result[2].fill_time_utc.tolist(), [self.start + 2 * HOUR, self.start + 3 * HOUR])
        self.assertTrue(result[1].terminal_close_decision.iloc[-1])

    def test_missing_candles_duplicate_decisions_and_invalid_probabilities_rejected(self):
        predictions, candles, funding = self.fixtures()
        with self.assertRaisesRegex(ValueError, "coverage"):
            replay(predictions, candles.iloc[:2], funding, self.config)
        with self.assertRaisesRegex(ValueError, "unique contiguous"):
            replay(pd.concat([predictions, predictions.iloc[:1]]), candles, funding, self.config)
        predictions.loc[0, "v3_unweighted_score"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid saved"):
            replay(predictions, candles, funding, self.config)

    def test_replay_loss_limit_blocks_new_entry_but_allows_exit(self):
        predictions, candles, funding = self.fixtures()
        predictions = pd.concat([predictions, predictions.iloc[:1].assign(
            decision_time_utc=self.start + 2 * HOUR)], ignore_index=True)
        candles.close = 10000.
        extra = candles.iloc[-1:].assign(candle_start_utc=self.start + 3 * HOUR,
                                        candle_end_utc=self.start + 4 * HOUR)
        config = replace(self.config, max_session_loss_settlement=.000001)
        summary, decisions, fills, _, _ = replay(predictions, pd.concat([candles, extra]), funding, config)
        self.assertEqual(fills.signed_requested_qty.tolist(), [100, -100])
        self.assertEqual(decisions.iloc[2].reason, "LOSS_LIMIT_REDUCE_ONLY")
        self.assertEqual(decisions.iloc[2].signed_requested_qty, 0)
        self.assertEqual(summary["round_trips"], 1)

    def test_order_identity_partial_fills_commissions_and_lifecycle(self):
        order = "11111111-1111-1111-1111-111111111111"
        events = pd.DataFrame({"account": ["a", "a", "b", "a"], "symbol": "XBTUSD",
            "orderid": [order, order, order, ZERO_UUID], "execid": ["1", "2", "3", "4"],
            "event_time_utc": self.start, "processed_time_utc": self.start, "exectype": "Trade",
            "side": "Buy", "lastqty": [40, 60, 10, 7], "cumqty": [40, 100, 10, 7],
            "leavesqty": [60, 0, 0, 0], "orderqty": [100, 100, 10, 7],
            "ordstatus": ["PartiallyFilled", "Filled", "Filled", "Filled"],
            "lastliquidityind": "RemovedLiquidity", "execinst": "ParticipateDoNotInitiate",
            "commission": .00075, "execcomm": [1, 2, 3, 4]})
        orders, retained, unidentified = order_view(events)
        self.assertEqual(len(orders), 2)
        a = orders[orders.account.eq("a")].iloc[0]
        self.assertEqual(a.observed_order_quantities, "[100]")
        self.assertEqual(a.sum_last_qty, 100)
        self.assertEqual(a.fill_row_count, 2)
        self.assertEqual(a.observed_execcomm_sum, 3)
        self.assertTrue(a.missing_new_event)
        self.assertFalse(a.missing_terminal_event)
        self.assertTrue(a.ambiguous_ordering)
        self.assertEqual(len(unidentified), 1)
        self.assertEqual(unidentified.iloc[0].order_identity_status, "ZERO_UUID")
        self.assertEqual(len(retained), 4)
        self.assertIn("commission", retained)


if __name__ == "__main__":
    unittest.main()
