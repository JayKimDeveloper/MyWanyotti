"""Offline order observations and one frozen, bot-owned hourly policy."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from .backtest import Portfolio
from .candles import HOUR
from .executions import ZERO_UUID, audit_orders
from .paper import PaperTradingConfig, apply_action_constraints
from .side_research import CLASSES

ACTIVITY_THRESHOLD = 0.22920271033221618


def order_view(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep events losslessly; never treat repeated orderQty as additive."""
    events = events.copy()
    valid = events.orderid.astype("string").str.fullmatch(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    ).fillna(False) & events.orderid.ne(ZERO_UUID)
    valid &= events.account.notna() & events.account.ne("")
    events["order_identity_status"] = np.where(valid, "VALID", "UNIDENTIFIED")
    events.loc[events.orderid.eq(ZERO_UUID), "order_identity_status"] = "ZERO_UUID"
    events["ambiguous_ordering"] = events.duplicated(
        ["account", "symbol", "event_time_utc", "processed_time_utc"], keep=False
    ) | events.processed_time_utc.lt(events.event_time_utc)
    events["duplicate_execid"] = events.duplicated(["account", "execid"], keep=False)
    ordered = events.sort_values(
        ["account", "symbol", "event_time_utc", "processed_time_utc", "cumqty", "execid"],
        kind="stable",
    )
    audits = []
    for (account, symbol), group in ordered[valid.reindex(ordered.index)].groupby(["account", "symbol"]):
        audit = audit_orders(group)
        if not audit.empty:
            audits.append(audit.assign(account=account))
    rows = []
    for (account, symbol, order_id), group in ordered[valid.reindex(ordered.index)].groupby(
        ["account", "symbol", "orderid"], sort=False
    ):
        trades = group[group.exectype.eq("Trade")]
        def observed(column):
            return json.dumps(group[column].dropna().drop_duplicates().tolist())
        rows.append({
            "account": account, "symbol": symbol, "order_id": order_id,
            "observed_event_count": len(group),
            "observed_order_quantities": observed("orderqty"),
            "observed_liquidity_indicators": observed("lastliquidityind"),
            "observed_execution_instructions": observed("execinst"),
            "observed_status_changes": json.dumps(group.ordstatus.fillna("MISSING").tolist()),
            "observed_commission_rates": observed("commission"),
            "observed_execcomm_sum": trades.execcomm.sum(min_count=1),
            "missing_new_event": not group.exectype.eq("New").any(),
            "missing_terminal_event": not group.ordstatus.isin(["Filled", "Canceled", "Rejected", "Expired"]).any(),
            "ambiguous_ordering": bool(group.ambiguous_ordering.any()),
            "duplicate_execid": bool(group.duplicate_execid.any()),
            "observation_scope": "TRAIN_VALIDATION_WINDOWS_ONLY_LIFECYCLE_MAY_BE_CLIPPED",
        })
    orders = pd.DataFrame(rows)
    if audits:
        orders = orders.merge(pd.concat(audits), on=["account", "symbol", "order_id"],
                              how="left", validate="one_to_one")
    unidentified = ordered[ordered.order_identity_status.ne("VALID") & ordered.exectype.eq("Trade")]
    return orders, ordered.reset_index(drop=True), unidentified.reset_index(drop=True)


def policy_order(position: int, score: float, side: str, config: PaperTradingConfig,
                 equity: float, pending: bool = False) -> tuple[int, str]:
    if side not in CLASSES or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("invalid saved prediction")
    if position:
        action = "LONG_DECREASE" if position > 0 else "SHORT_DECREASE"
        size = abs(position)
    elif score >= ACTIVITY_THRESHOLD and side in {"BUY_ONLY", "SELL_ONLY"}:
        action = "LONG_INCREASE" if side == "BUY_ONLY" else "SHORT_INCREASE"
        size = 100
    else:
        return 0, "NOT_FLAGGED" if score < ACTIVITY_THRESHOLD else side
    quantity, _, reason = apply_action_constraints(
        action, position, size, config.max_abs_position_contracts, False,
        config.backtest.initial_equity - equity, config.max_session_loss_settlement, pending,
    )
    if equity <= 0 and not position:
        return 0, "NONPOSITIVE_EQUITY_REDUCE_ONLY"
    return quantity, reason


def replay(predictions: pd.DataFrame, candles: pd.DataFrame, funding: pd.DataFrame,
           config: PaperTradingConfig) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fill at the next hourly close, after arrival, with funding before tied fills."""
    base = config.backtest
    if base.initial_position_contracts != 0 or base.contracts_per_action != 100:
        raise ValueError("policy requires a flat synthetic bot and 100-contract requests")
    if base.instrument.contract_type != "inverse" or base.instrument.settlement_currency != "XBT":
        raise ValueError("bridge accounting must use inverse XBT units")
    if not pd.Timedelta(0) < base.latency < HOUR or base.slippage_bps >= 10000:
        raise ValueError("latency must be within one hour and slippage below 100%")
    rows = predictions.sort_values("decision_time_utc").reset_index(drop=True)
    if rows.empty or not rows.symbol.eq("XBTUSD").all():
        raise ValueError("nonempty XBTUSD predictions required")
    times = pd.DatetimeIndex(rows.decision_time_utc)
    expected = pd.date_range(times[0], times[-1], freq=HOUR)
    if times.tz is None or times.tolist() != expected.tolist():
        raise ValueError("unique contiguous UTC hourly decisions required")
    if not np.isfinite(rows.v3_unweighted_score).all() or not rows.v3_unweighted_score.between(0, 1).all() or not rows.v2_argmax.isin(CLASSES).all():
        raise ValueError("invalid saved prediction")
    candles = candles.sort_values("candle_end_utc").set_index("candle_end_utc")
    horizon = times[-1] + 2 * HOUR
    timeline = pd.date_range(times[0], horizon, freq=HOUR)
    if candles.index.has_duplicates or not timeline.isin(candles.index).all():
        raise ValueError("complete candle coverage including terminal exit required")
    if not candles.symbol.eq("XBTUSD").all() or not np.isfinite(candles.close).all() or candles.close.le(0).any():
        raise ValueError("invalid execution candles")
    if not (candles.index - candles.candle_start_utc).eq(HOUR).all():
        raise ValueError("candle timestamps must denote one-hour bucket ends")
    if not funding.empty:
        if funding.funding_time_utc.duplicated().any() or not np.isfinite(funding.funding_rate).all():
            raise ValueError("invalid historical funding schedule")
        if not funding.funding_time_utc.isin(timeline).all():
            raise ValueError("funding events must lie on replay hourly boundaries")
    funding_rates = dict(zip(funding.funding_time_utc, funding.funding_rate))
    portfolio = Portfolio.initialize(base.instrument, base.initial_equity, 0, float(candles.loc[times[0], "close"]))
    pending = None
    entered_at = None
    decision_records, fills, curve, payments = [], [], [], []
    for time in timeline:
        reference = float(candles.loc[time, "close"])
        if time in funding_rates:
            quantity = portfolio.position
            payment = portfolio.apply_funding(reference, funding_rates[time])
            payments.append({"funding_time_utc": time, "position_qty": quantity,
                             "funding_rate": funding_rates[time], "mark_price": reference,
                             "funding_payment_xbt": payment})
        if pending is not None:
            if time != pending["fill_time_utc"] or time <= pending["arrival_time_utc"]:
                raise ValueError("noncausal or overlapping pending execution")
            quantity = pending["signed_requested_qty"]
            buy = quantity > 0
            slipped = reference * (1 + (1 if buy else -1) * base.slippage_bps / 10000)
            tick = base.instrument.price_tick_size
            price = (math.ceil(slipped / tick) if buy else math.floor(slipped / tick)) * tick
            if price <= 0:
                raise ValueError("invalid simulated price")
            before = portfolio.position
            fee = portfolio.fee(abs(quantity), price, base.taker_rate)
            slippage = portfolio.slippage_cost(quantity, reference, price)
            portfolio.apply_fill(quantity, price, fee)
            holding = (time - entered_at).total_seconds() if before else None
            entered_at = time if portfolio.position else None
            fills.append({**pending, "candle_start_utc": time - HOUR, "candle_end_utc": time,
                          "reference_price": reference, "fill_price": price,
                          "liquidity": "SIMULATED_TAKER", "price_source": "CACHED_HOURLY_CLOSE",
                          "fee_xbt": fee, "slippage_cost_xbt": slippage,
                          "position_before_qty": before, "position_after_qty": portfolio.position,
                          "holding_seconds": holding,
                          "execution_delay_seconds": (time - pending["decision_time_utc"]).total_seconds()})
            decision_records[pending["decision_index"]].update(
                fill_status="FILLED", position_after_fill=portfolio.position)
            pending = None
        equity = portfolio.equity(reference)
        curve.append({"timestamp_utc": time, "mark_price": reference,
                      "position_qty": portfolio.position, "cash_xbt": portfolio.cash,
                      "equity_xbt": equity, "flat_baseline_equity_xbt": base.initial_equity})
        if time > times[-1] + HOUR:
            continue
        terminal = time > times[-1]
        if terminal:
            score, side = 0.0, "NO_RECORDED_FILL"
        else:
            row = rows.iloc[int((time - times[0]) / HOUR)]
            score, side = float(row.v3_unweighted_score), row.v2_argmax
        quantity, reason = policy_order(portfolio.position, score, side, config, equity, pending is not None)
        record = {"decision_time_utc": time, "terminal_close_decision": terminal,
                  "bot_position_before_qty": portfolio.position, "v3_unweighted_score": score,
                  "v3_flagged": score >= ACTIVITY_THRESHOLD, "v2_argmax": side,
                  "signed_requested_qty": quantity, "reason": reason,
                  "fill_status": "PENDING" if quantity else "NO_ORDER", "position_after_fill": portfolio.position}
        decision_records.append(record)
        if quantity:
            pending = {"decision_time_utc": time, "arrival_time_utc": time + base.latency,
                       "fill_time_utc": time + HOUR, "signed_requested_qty": quantity,
                       "decision_index": len(decision_records) - 1}
    if portfolio.position or pending:
        raise ValueError("terminal exit did not complete")
    curve = pd.DataFrame(curve)
    curve["drawdown"] = curve.equity_xbt / curve.equity_xbt.cummax() - 1
    fills = pd.DataFrame(fills, columns=["decision_time_utc", "arrival_time_utc", "fill_time_utc",
        "signed_requested_qty", "decision_index", "candle_start_utc", "candle_end_utc", "reference_price",
        "fill_price", "liquidity", "price_source", "fee_xbt", "slippage_cost_xbt", "position_before_qty",
        "position_after_qty", "holding_seconds", "execution_delay_seconds"])
    decisions = pd.DataFrame(decision_records)
    payments = pd.DataFrame(payments, columns=["funding_time_utc", "position_qty", "funding_rate", "mark_price", "funding_payment_xbt"])
    final = float(curve.equity_xbt.iloc[-1])
    summary = {"holdout_windows": len(rows), "terminal_close_decisions": 1,
        "initial_equity_xbt": base.initial_equity, "final_equity_xbt": final,
        "net_pnl_xbt": final - base.initial_equity, "total_return": final / base.initial_equity - 1,
        "max_hourly_sampled_drawdown": float(-curve.drawdown.min()),
        "turnover_contracts": int(fills.signed_requested_qty.abs().sum()), "fill_count": len(fills),
        "round_trips": int(fills.position_after_qty.eq(0).sum()),
        "fees_xbt": float(fills.fee_xbt.sum()), "slippage_cost_xbt": float(fills.slippage_cost_xbt.sum()),
        "funding_payment_xbt": float(payments.funding_payment_xbt.sum()),
        "holding_seconds": fills.holding_seconds.dropna().describe().to_dict(),
        "skipped_decisions_by_reason": decisions.loc[decisions.signed_requested_qty.eq(0), "reason"].value_counts().to_dict(),
        "max_abs_position_contracts_observed": int(curve.position_qty.abs().max()),
        "flat_baseline": {"final_equity_xbt": base.initial_equity, "return": 0, "drawdown": 0, "costs_xbt": 0}}
    return summary, decisions, fills, curve, payments
