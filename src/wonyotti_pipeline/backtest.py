"""Chronological, position-aware backtesting with explicit execution costs."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import MODEL_FEATURE_COLUMNS
from .inference import CatBoostActionPredictor
from .training import classification_metrics, serialize_feature_row


class BacktestConfigError(ValueError):
    """Raised when a backtest assumption is missing or unsupported."""


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    contract_type: str
    quantity_unit: str
    price_currency: str
    settlement_currency: str
    lot_size_contracts: int
    price_tick_size: float
    contract_value_quote: float
    linear_multiplier: float
    pnl_formula: str


@dataclass(frozen=True)
class BacktestConfig:
    config_version: int
    assumption_status: str
    lookback: pd.Timedelta
    max_quote_age: pd.Timedelta
    reconciliation_atol: float
    initial_equity: float
    initial_position_contracts: int
    contracts_per_action: int
    invalid_action_policy: str
    latency: pd.Timedelta
    max_fill_wait: pd.Timedelta
    base_fill_probability: float
    slippage_bps: float
    max_top_of_book_participation: float
    allow_partial_fills: bool
    random_seed: int
    taker_rate: float
    funding_mode: str
    funding_rate: float
    funding_interval: pd.Timedelta
    funding_offset_hour_utc: int
    liquidation_model: str
    stop_on_nonpositive_equity: bool
    cost_sensitivity_multipliers: tuple[float, ...]
    instrument: InstrumentSpec


@dataclass
class Portfolio:
    instrument: InstrumentSpec
    cash: float
    position: int

    @classmethod
    def initialize(
        cls,
        instrument: InstrumentSpec,
        initial_equity: float,
        initial_position: int,
        initial_mark: float,
    ) -> "Portfolio":
        if instrument.contract_type == "inverse":
            cash = initial_equity + (
                initial_position * instrument.contract_value_quote / initial_mark
            )
        else:
            cash = initial_equity - (
                initial_position * instrument.linear_multiplier * initial_mark
            )
        return cls(instrument, float(cash), int(initial_position))

    def equity(self, mark_price: float) -> float:
        if self.instrument.contract_type == "inverse":
            return self.cash - (
                self.position * self.instrument.contract_value_quote / mark_price
            )
        return self.cash + (
            self.position * self.instrument.linear_multiplier * mark_price
        )

    def signed_notional_settlement(self, mark_price: float) -> float:
        if self.instrument.contract_type == "inverse":
            return (
                self.position * self.instrument.contract_value_quote / mark_price
            )
        return self.position * self.instrument.linear_multiplier * mark_price

    def absolute_notional_settlement(self, mark_price: float) -> float:
        return abs(self.signed_notional_settlement(mark_price))

    def fee(self, quantity: int, price: float, fee_rate: float) -> float:
        if self.instrument.contract_type == "inverse":
            notional = quantity * self.instrument.contract_value_quote / price
        else:
            notional = quantity * self.instrument.linear_multiplier * price
        return abs(notional) * fee_rate

    def slippage_cost(
        self, signed_quantity: int, reference_price: float, fill_price: float
    ) -> float:
        if self.instrument.contract_type == "inverse":
            actual = (
                signed_quantity
                * self.instrument.contract_value_quote
                / fill_price
            )
            reference = (
                signed_quantity
                * self.instrument.contract_value_quote
                / reference_price
            )
            return -(actual - reference)
        return (
            signed_quantity
            * self.instrument.linear_multiplier
            * (fill_price - reference_price)
        )

    def apply_fill(self, signed_quantity: int, fill_price: float, fee: float) -> None:
        if self.instrument.contract_type == "inverse":
            self.cash += (
                signed_quantity
                * self.instrument.contract_value_quote
                / fill_price
            )
        else:
            self.cash -= (
                signed_quantity
                * self.instrument.linear_multiplier
                * fill_price
            )
        self.cash -= fee
        self.position += signed_quantity

    def apply_funding(self, mark_price: float, funding_rate: float) -> float:
        payment = -self.signed_notional_settlement(mark_price) * funding_rate
        self.cash += payment
        return payment


def load_backtest_config(path: Path, symbol: str) -> BacktestConfig:
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    try:
        instrument_document = document["instruments"][symbol]
        instrument = InstrumentSpec(
            symbol=symbol,
            contract_type=str(instrument_document["contract_type"]),
            quantity_unit=str(instrument_document["quantity_unit"]),
            price_currency=str(instrument_document["price_currency"]),
            settlement_currency=str(instrument_document["settlement_currency"]),
            lot_size_contracts=int(instrument_document["lot_size_contracts"]),
            price_tick_size=float(instrument_document["price_tick_size"]),
            contract_value_quote=float(
                instrument_document["contract_value_quote"]
            ),
            linear_multiplier=float(instrument_document["linear_multiplier"]),
            pnl_formula=str(instrument_document["pnl_formula"]),
        )
        config = BacktestConfig(
            config_version=int(document["config_version"]),
            assumption_status=str(document["assumption_status"]),
            lookback=pd.Timedelta(document["features"]["lookback"]),
            max_quote_age=pd.Timedelta(document["features"]["max_quote_age"]),
            reconciliation_atol=float(
                document["features"]["reconciliation_atol"]
            ),
            initial_equity=float(document["portfolio"]["initial_equity"]),
            initial_position_contracts=int(
                document["portfolio"]["initial_position_contracts"]
            ),
            contracts_per_action=int(
                document["strategy"]["contracts_per_action"]
            ),
            invalid_action_policy=str(
                document["strategy"]["invalid_action_policy"]
            ),
            latency=pd.Timedelta(
                milliseconds=float(document["execution"]["latency_ms"])
            ),
            max_fill_wait=pd.Timedelta(
                milliseconds=float(document["execution"]["max_fill_wait_ms"])
            ),
            base_fill_probability=float(
                document["execution"]["base_fill_probability"]
            ),
            slippage_bps=float(document["execution"]["slippage_bps"]),
            max_top_of_book_participation=float(
                document["execution"]["max_top_of_book_participation"]
            ),
            allow_partial_fills=bool(
                document["execution"]["allow_partial_fills"]
            ),
            random_seed=int(document["execution"]["random_seed"]),
            taker_rate=float(document["fees"]["taker_rate"]),
            funding_mode=str(document["funding"]["mode"]),
            funding_rate=float(document["funding"]["rate"]),
            funding_interval=pd.Timedelta(
                hours=float(document["funding"]["interval_hours"])
            ),
            funding_offset_hour_utc=int(
                document["funding"]["offset_hour_utc"]
            ),
            liquidation_model=str(document["risk"]["liquidation_model"]),
            stop_on_nonpositive_equity=bool(
                document["risk"]["stop_on_nonpositive_equity"]
            ),
            cost_sensitivity_multipliers=tuple(
                float(value)
                for value in document["cost_sensitivity"]["multipliers"]
            ),
            instrument=instrument,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BacktestConfigError(f"invalid backtest config: {error}") from error
    validate_backtest_config(config)
    return config


def validate_backtest_config(config: BacktestConfig) -> None:
    if config.config_version != 1:
        raise BacktestConfigError("unsupported config_version")
    if config.instrument.contract_type not in {"inverse", "linear"}:
        raise BacktestConfigError("contract_type must be inverse or linear")
    if config.instrument.quantity_unit != "CONTRACTS":
        raise BacktestConfigError("only CONTRACTS quantity is supported")
    if config.instrument.lot_size_contracts <= 0:
        raise BacktestConfigError("lot_size_contracts must be positive")
    if config.instrument.price_tick_size <= 0:
        raise BacktestConfigError("price_tick_size must be positive")
    if config.instrument.contract_type == "inverse":
        if config.instrument.contract_value_quote <= 0:
            raise BacktestConfigError("inverse contract_value_quote must be positive")
    elif config.instrument.linear_multiplier <= 0:
        raise BacktestConfigError("linear_multiplier must be positive")
    numeric_values = {
        "initial_equity": config.initial_equity,
        "base_fill_probability": config.base_fill_probability,
        "slippage_bps": config.slippage_bps,
        "max_top_of_book_participation": config.max_top_of_book_participation,
        "taker_rate": config.taker_rate,
        "funding_rate": config.funding_rate,
        "price_tick_size": config.instrument.price_tick_size,
        "contract_value_quote": config.instrument.contract_value_quote,
        "linear_multiplier": config.instrument.linear_multiplier,
    }
    if any(not math.isfinite(value) for value in numeric_values.values()):
        raise BacktestConfigError("numeric assumptions must be finite")
    if config.initial_equity <= 0:
        raise BacktestConfigError("initial_equity must be positive")
    if config.contracts_per_action <= 0:
        raise BacktestConfigError("contracts_per_action must be positive")
    if config.contracts_per_action % config.instrument.lot_size_contracts:
        raise BacktestConfigError("contracts_per_action must be a whole lot")
    if config.initial_position_contracts % config.instrument.lot_size_contracts:
        raise BacktestConfigError("initial_position_contracts must be a whole lot")
    if config.invalid_action_policy != "NO_ORDER":
        raise BacktestConfigError("only invalid_action_policy=NO_ORDER is supported")
    if not 0 <= config.base_fill_probability <= 1:
        raise BacktestConfigError("base_fill_probability must be in [0, 1]")
    if config.latency < pd.Timedelta(0) or config.max_fill_wait < pd.Timedelta(0):
        raise BacktestConfigError("latency and max_fill_wait must be nonnegative")
    if not 0 < config.max_top_of_book_participation <= 1:
        raise BacktestConfigError("max_top_of_book_participation must be in (0, 1]")
    if config.slippage_bps < 0 or config.taker_rate < 0:
        raise BacktestConfigError("slippage and taker fee must be nonnegative")
    if config.funding_mode not in {"fixed", "disabled"}:
        raise BacktestConfigError("funding mode must be fixed or disabled")
    if config.funding_interval <= pd.Timedelta(0):
        raise BacktestConfigError("funding interval must be positive")
    if not 0 <= config.funding_offset_hour_utc <= 23:
        raise BacktestConfigError("funding_offset_hour_utc must be in [0, 23]")
    if config.liquidation_model != "disabled":
        raise BacktestConfigError("only liquidation_model=disabled is implemented")
    if not config.cost_sensitivity_multipliers:
        raise BacktestConfigError("cost sensitivity multipliers cannot be empty")
    if any(value < 0 for value in config.cost_sensitivity_multipliers):
        raise BacktestConfigError("cost sensitivity multipliers must be nonnegative")
    if any(
        not math.isfinite(value) for value in config.cost_sensitivity_multipliers
    ):
        raise BacktestConfigError("cost sensitivity multipliers must be finite")
    if 1.0 not in config.cost_sensitivity_multipliers:
        raise BacktestConfigError("cost sensitivity multipliers must include 1.0")


def signed_order_for_action(action: str, position: int, size: int) -> tuple[int, str]:
    if action == "NO_FILL":
        return 0, "NO_ACTION"
    if action == "LONG_INCREASE":
        return (size, "ORDER") if position >= 0 else (0, "INVALID_FOR_POSITION")
    if action == "LONG_DECREASE":
        return (-min(size, position), "ORDER") if position > 0 else (0, "INVALID_FOR_POSITION")
    if action == "SHORT_INCREASE":
        return (-size, "ORDER") if position <= 0 else (0, "INVALID_FOR_POSITION")
    if action == "SHORT_DECREASE":
        return (min(size, -position), "ORDER") if position < 0 else (0, "INVALID_FOR_POSITION")
    raise BacktestConfigError(f"unsupported action: {action}")


def latest_trade_price(
    trade_times_ns: np.ndarray,
    trade_prices: np.ndarray,
    timestamp: pd.Timestamp,
) -> float:
    if not len(trade_times_ns):
        raise BacktestConfigError("public trade data is empty")
    index = np.searchsorted(trade_times_ns, int(timestamp.value), side="left") - 1
    if index < 0:
        raise BacktestConfigError(f"no market trade before {timestamp.isoformat()}")
    price = float(trade_prices[index])
    if not math.isfinite(price) or price <= 0:
        raise BacktestConfigError(f"invalid market trade price before {timestamp.isoformat()}")
    return price


def funding_times(
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: pd.Timedelta,
    offset_hour_utc: int,
) -> list[pd.Timestamp]:
    anchor = start.normalize() + pd.Timedelta(hours=offset_hour_utc)
    while anchor < start:
        anchor += interval
    times: list[pd.Timestamp] = []
    while anchor < end:
        times.append(anchor)
        anchor += interval
    return times


def reconcile_market_features(
    stored: pd.DataFrame,
    rebuilt: pd.DataFrame,
    atol: float,
) -> dict[str, object]:
    feature_columns = [
        column
        for column in MODEL_FEATURE_COLUMNS
        if column not in {"symbol", "decision_time_utc", "position_before_qty"}
    ]
    merged = stored[["symbol", "decision_time_utc", *feature_columns]].merge(
        rebuilt[["symbol", "decision_time_utc", *feature_columns]],
        on=["symbol", "decision_time_utc"],
        suffixes=("_stored", "_rebuilt"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    report: dict[str, object] = {
        "row_count_stored": int(len(stored)),
        "row_count_rebuilt": int(len(rebuilt)),
        "join_mismatch_count": int(merged["_merge"].ne("both").sum()),
        "atol": atol,
        "features": {},
    }
    total_mismatches = int(report["join_mismatch_count"])
    joined = merged["_merge"].eq("both")
    for column in feature_columns:
        left = pd.to_numeric(merged[f"{column}_stored"], errors="coerce")
        right = pd.to_numeric(merged[f"{column}_rebuilt"], errors="coerce")
        finite = joined & left.notna() & right.notna()
        differences = (left - right).abs()
        close = pd.Series(
            np.isclose(left, right, rtol=0.0, atol=atol, equal_nan=False),
            index=merged.index,
        )
        mismatch = joined & ~(
            (left.isna() & right.isna())
            | (finite & close)
        )
        mismatch_count = int(mismatch.sum())
        total_mismatches += mismatch_count
        report["features"][column] = {
            "mismatch_count": mismatch_count,
            "max_absolute_difference": (
                float(differences[finite].max()) if finite.any() else None
            ),
        }
    report["total_mismatch_count"] = total_mismatches
    report["passed"] = total_mismatches == 0
    return report


def evaluate_trader_positions(
    rows: pd.DataFrame,
    predictor: CatBoostActionPredictor,
) -> tuple[dict[str, object], pd.DataFrame]:
    if rows.empty:
        raise BacktestConfigError("trader-position evaluation rows are empty")
    output_rows: list[dict[str, object]] = []
    for row in rows.itertuples(index=False):
        series = pd.Series(row._asdict())
        prediction = predictor.predict(serialize_feature_row(series))
        output_rows.append(
            {
                "symbol": row.symbol,
                "decision_time_utc": row.decision_time_utc,
                "target_label": row.target_label,
                "trader_position_before_qty": int(row.position_before_qty),
                "policy_prediction": prediction["policy_action"],
                "argmax_prediction": prediction["argmax_action"],
                "confidence": prediction["confidence"],
                **{
                    f"probability__{label}": probability
                    for label, probability in prediction["probabilities"].items()
                },
            }
        )
    predictions = pd.DataFrame(output_rows)
    return {
        "position_input": "ORIGINAL_TRADER_POSITION",
        "policy_metrics": classification_metrics(
            predictions["target_label"],
            predictions["policy_prediction"],
            predictor.class_labels,
        ),
        "argmax_metrics": classification_metrics(
            predictions["target_label"],
            predictions["argmax_prediction"],
            predictor.class_labels,
        ),
    }, predictions


def _fill_candidate(
    quotes: pd.DataFrame,
    decision_time: pd.Timestamp,
    target_end: pd.Timestamp,
    signed_quantity: int,
    config: BacktestConfig,
    cost_multiplier: float,
    random: np.random.Generator,
) -> dict[str, object]:
    arrival = decision_time + config.latency
    quote_times = quotes["market_time_utc"].astype("int64").to_numpy()
    index = np.searchsorted(quote_times, int(arrival.value), side="left")
    result: dict[str, object] = {
        "arrival_time_utc": arrival,
        "requested_qty": abs(signed_quantity),
        "signed_requested_qty": signed_quantity,
        "fill_status": "NO_QUOTE",
        "random_draw": None,
        "price_source": "PUBLIC_QUOTE_AFTER_DECISION",
    }
    if index >= len(quotes):
        return result
    quote = quotes.iloc[index]
    return simulate_quote_fill(
        quote,
        decision_time,
        target_end,
        signed_quantity,
        config,
        cost_multiplier,
        random,
    )


def simulate_quote_fill(
    quote: pd.Series,
    decision_time: pd.Timestamp,
    target_end: pd.Timestamp,
    signed_quantity: int,
    config: BacktestConfig,
    cost_multiplier: float,
    random: np.random.Generator,
) -> dict[str, object]:
    """Apply the shared paper-fill assumptions to one post-decision quote."""
    arrival = decision_time + config.latency
    deadline = min(target_end, arrival + config.max_fill_wait)
    quote_time = pd.Timestamp(quote["market_time_utc"])
    result: dict[str, object] = {
        "arrival_time_utc": arrival,
        "requested_qty": abs(signed_quantity),
        "signed_requested_qty": signed_quantity,
        "fill_status": "NO_QUOTE",
        "random_draw": None,
        "price_source": "PUBLIC_QUOTE_AFTER_DECISION",
    }
    if quote_time < arrival:
        result["fill_status"] = "QUOTE_BEFORE_ARRIVAL"
        return result
    if quote_time > deadline or quote_time >= target_end:
        result["fill_status"] = "QUOTE_AFTER_DEADLINE"
        return result

    draw = float(random.random())
    result["random_draw"] = draw
    if draw > config.base_fill_probability:
        result["fill_status"] = "PROBABILITY_REJECTED"
        return result

    is_buy = signed_quantity > 0
    reference_price = float(quote["ask_price"] if is_buy else quote["bid_price"])
    top_size = float(quote["ask_size"] if is_buy else quote["bid_size"])
    if not math.isfinite(reference_price) or reference_price <= 0:
        result["fill_status"] = "INVALID_TOP_BOOK_PRICE"
        return result
    if not math.isfinite(top_size) or top_size <= 0:
        result["fill_status"] = "INVALID_TOP_BOOK_SIZE"
        return result
    lot_size = config.instrument.lot_size_contracts
    raw_capacity = math.floor(top_size * config.max_top_of_book_participation)
    capacity = int(raw_capacity // lot_size * lot_size)
    if capacity <= 0:
        result["fill_status"] = "NO_TOP_BOOK_CAPACITY"
        return result
    requested = abs(signed_quantity)
    if capacity < requested and not config.allow_partial_fills:
        result["fill_status"] = "INSUFFICIENT_TOP_BOOK_CAPACITY"
        return result
    filled = min(requested, capacity)
    slippage_fraction = config.slippage_bps * cost_multiplier / 10_000.0
    unrounded_fill_price = reference_price * (
        1.0 + slippage_fraction if is_buy else 1.0 - slippage_fraction
    )
    if not math.isfinite(unrounded_fill_price) or unrounded_fill_price <= 0:
        result["fill_status"] = "INVALID_SLIPPAGE_PRICE"
        return result
    tick_size = config.instrument.price_tick_size
    if is_buy:
        fill_price = math.ceil(unrounded_fill_price / tick_size) * tick_size
    else:
        fill_price = math.floor(unrounded_fill_price / tick_size) * tick_size
    result.update(
        {
            "fill_status": "FILLED" if filled == requested else "PARTIAL",
            "quote_time_utc": quote_time,
            "fill_time_utc": quote_time,
            "reference_price": reference_price,
            "fill_price": fill_price,
            "unrounded_fill_price": unrounded_fill_price,
            "lot_size_contracts": lot_size,
            "price_tick_size": tick_size,
            "top_size": top_size,
            "capacity_qty": capacity,
            "filled_qty": filled,
            "signed_filled_qty": filled if is_buy else -filled,
            "fill_delay_ms": (
                quote_time - decision_time
            ).total_seconds()
            * 1000.0,
        }
    )
    return result


def _summary(
    config: BacktestConfig,
    cost_multiplier: float,
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    decisions: pd.DataFrame,
    funding: pd.DataFrame,
    exposure_seconds: dict[str, float],
    stopped_early: bool,
) -> dict[str, object]:
    final_equity = float(equity_curve["equity"].iloc[-1])
    initial_equity = config.initial_equity
    attempted = int(decisions["order_attempted"].sum())
    filled = int(decisions["filled_qty"].gt(0).sum())
    elapsed_seconds = exposure_seconds["elapsed"]
    return {
        "position_input": "BOT_OWN_POSITION",
        "cost_multiplier": cost_multiplier,
        "settlement_currency": config.instrument.settlement_currency,
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "total_return": final_equity / initial_equity - 1.0,
        "max_drawdown": float(-equity_curve["drawdown"].min()),
        "decision_count": int(len(decisions)),
        "order_attempt_count": attempted,
        "filled_trade_count": int(len(trades)),
        "filled_decision_count": filled,
        "fill_rate_per_attempt": filled / attempted if attempted else 0.0,
        "turnover_contracts": int(trades["filled_qty"].sum()) if len(trades) else 0,
        "elapsed_seconds": elapsed_seconds,
        "time_exposure_fraction": (
            exposure_seconds["exposed"] / elapsed_seconds if elapsed_seconds else 0.0
        ),
        "long_exposure_fraction": (
            exposure_seconds["long"] / elapsed_seconds if elapsed_seconds else 0.0
        ),
        "short_exposure_fraction": (
            exposure_seconds["short"] / elapsed_seconds if elapsed_seconds else 0.0
        ),
        "sample_average_absolute_exposure_ratio": float(
            equity_curve["absolute_exposure_ratio"].mean()
        ),
        "sample_maximum_absolute_exposure_ratio": float(
            equity_curve["absolute_exposure_ratio"].max()
        ),
        "total_fees": float(trades["fee"].sum()) if len(trades) else 0.0,
        "total_slippage_cost": (
            float(trades["slippage_cost"].sum()) if len(trades) else 0.0
        ),
        "net_funding": float(funding["funding_payment"].sum()) if len(funding) else 0.0,
        "net_total_cost": (
            (float(trades["fee"].sum()) if len(trades) else 0.0)
            + (float(trades["slippage_cost"].sum()) if len(trades) else 0.0)
            - (float(funding["funding_payment"].sum()) if len(funding) else 0.0)
        ),
        "stopped_on_nonpositive_equity": stopped_early,
        "liquidation_model": config.liquidation_model,
    }


def run_bot_backtest(
    feature_rows: pd.DataFrame,
    public_trades: pd.DataFrame,
    public_quotes: pd.DataFrame,
    predictor: CatBoostActionPredictor,
    config: BacktestConfig,
    cost_multiplier: float,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = feature_rows.sort_values("decision_time_utc").reset_index(drop=True)
    trades = public_trades.sort_values("market_time_utc").reset_index(drop=True)
    quotes = public_quotes.sort_values("market_time_utc").reset_index(drop=True)
    if not math.isfinite(cost_multiplier) or cost_multiplier < 0:
        raise BacktestConfigError("cost_multiplier must be finite and nonnegative")
    if rows.empty:
        raise BacktestConfigError("backtest rows are empty")
    if trades.empty or quotes.empty:
        raise BacktestConfigError("public trade and quote data must be nonempty")
    if rows["symbol"].nunique() != 1 or rows["symbol"].iloc[0] != config.instrument.symbol:
        raise BacktestConfigError("backtest rows must contain only the configured symbol")
    if rows["decision_time_utc"].duplicated().any():
        raise BacktestConfigError("duplicate decision times are not allowed")
    if rows["target_end_time_utc"].le(rows["decision_time_utc"]).any():
        raise BacktestConfigError("every target interval must have positive duration")
    previous_end = rows["target_end_time_utc"].shift()
    if rows.loc[previous_end.notna(), "decision_time_utc"].lt(
        previous_end.dropna()
    ).any():
        raise BacktestConfigError("overlapping target intervals are not supported")
    trade_times_ns = trades["market_time_utc"].astype("int64").to_numpy()
    trade_prices = trades["market_price"].to_numpy(dtype="float64")
    first_mark = latest_trade_price(
        trade_times_ns, trade_prices, rows["decision_time_utc"].iloc[0]
    )
    portfolio = Portfolio.initialize(
        config.instrument,
        config.initial_equity,
        config.initial_position_contracts,
        first_mark,
    )
    random = np.random.default_rng(config.random_seed)
    all_funding_times = (
        funding_times(
            rows["decision_time_utc"].min(),
            rows["target_end_time_utc"].max(),
            config.funding_interval,
            config.funding_offset_hour_utc,
        )
        if config.funding_mode == "fixed"
        else []
    )

    decision_records: list[dict[str, object]] = []
    trade_records: list[dict[str, object]] = []
    funding_records: list[dict[str, object]] = []
    equity_records: list[dict[str, object]] = [
        {
            "timestamp_utc": rows["decision_time_utc"].iloc[0],
            "mark_price": first_mark,
            "position_qty": portfolio.position,
            "cash": portfolio.cash,
            "equity": portfolio.equity(first_mark),
            "absolute_notional_settlement": portfolio.absolute_notional_settlement(
                first_mark
            ),
            "absolute_exposure_ratio": (
                portfolio.absolute_notional_settlement(first_mark)
                / config.initial_equity
            ),
        }
    ]
    stopped_early = False
    funding_index = 0
    exposure_cursor = rows["decision_time_utc"].iloc[0]
    exposure_seconds = {"elapsed": 0.0, "exposed": 0.0, "long": 0.0, "short": 0.0}

    def accrue_exposure(until: pd.Timestamp) -> None:
        nonlocal exposure_cursor
        seconds = (until - exposure_cursor).total_seconds()
        if seconds < 0:
            raise BacktestConfigError("backtest events are not chronological")
        exposure_seconds["elapsed"] += seconds
        if portfolio.position != 0:
            exposure_seconds["exposed"] += seconds
        if portfolio.position > 0:
            exposure_seconds["long"] += seconds
        elif portfolio.position < 0:
            exposure_seconds["short"] += seconds
        exposure_cursor = until

    def record_equity(timestamp: pd.Timestamp) -> tuple[float, float]:
        mark = latest_trade_price(trade_times_ns, trade_prices, timestamp)
        equity = portfolio.equity(mark)
        absolute_notional = portfolio.absolute_notional_settlement(mark)
        equity_records.append(
            {
                "timestamp_utc": timestamp,
                "mark_price": mark,
                "position_qty": portfolio.position,
                "cash": portfolio.cash,
                "equity": equity,
                "absolute_notional_settlement": absolute_notional,
                "absolute_exposure_ratio": (
                    absolute_notional / abs(equity) if equity else math.inf
                ),
            }
        )
        return mark, equity

    def apply_funding_event(event_time: pd.Timestamp) -> float:
        accrue_exposure(event_time)
        funding_mark = latest_trade_price(trade_times_ns, trade_prices, event_time)
        rate = config.funding_rate * cost_multiplier
        payment = portfolio.apply_funding(funding_mark, rate)
        funding_records.append(
            {
                "funding_time_utc": event_time,
                "position_qty": portfolio.position,
                "mark_price": funding_mark,
                "funding_rate": rate,
                "funding_payment": payment,
            }
        )
        record_equity(event_time)
        return payment

    for row in rows.itertuples(index=False):
        while (
            funding_index < len(all_funding_times)
            and all_funding_times[funding_index] < row.decision_time_utc
        ):
            apply_funding_event(all_funding_times[funding_index])
            funding_index += 1
        accrue_exposure(row.decision_time_utc)
        position_before = portfolio.position
        feature_series = pd.Series(row._asdict())
        feature_series["position_before_qty"] = position_before
        prediction = predictor.predict(serialize_feature_row(feature_series))
        signed_order, order_reason = signed_order_for_action(
            prediction["policy_action"],
            position_before,
            config.contracts_per_action,
        )
        fill = {
            "fill_status": order_reason,
            "filled_qty": 0,
            "signed_filled_qty": 0,
        }
        if signed_order:
            fill = _fill_candidate(
                quotes,
                row.decision_time_utc,
                row.target_end_time_utc,
                signed_order,
                config,
                cost_multiplier,
                random,
            )

        events: list[tuple[pd.Timestamp, int, str, object]] = []
        pending_funding_index = funding_index
        while (
            pending_funding_index < len(all_funding_times)
            and all_funding_times[pending_funding_index] < row.target_end_time_utc
        ):
            events.append(
                (
                    all_funding_times[pending_funding_index],
                    0,
                    "funding",
                    all_funding_times[pending_funding_index],
                )
            )
            pending_funding_index += 1
        if fill.get("filled_qty", 0):
            events.append((fill["fill_time_utc"], 1, "fill", fill))
        interval_funding = 0.0
        for event_time, _, event_kind, payload in sorted(events):
            if event_kind == "funding":
                payment = apply_funding_event(event_time)
                interval_funding += payment
                funding_index += 1
            else:
                fill_payload = payload
                accrue_exposure(event_time)
                signed_filled = int(fill_payload["signed_filled_qty"])
                fee = portfolio.fee(
                    abs(signed_filled),
                    float(fill_payload["fill_price"]),
                    config.taker_rate * cost_multiplier,
                )
                slippage_cost = portfolio.slippage_cost(
                    signed_filled,
                    float(fill_payload["reference_price"]),
                    float(fill_payload["fill_price"]),
                )
                fill_payload["fee"] = fee
                fill_payload["slippage_cost"] = slippage_cost
                fill_payload["position_before_qty"] = portfolio.position
                portfolio.apply_fill(
                    signed_filled, float(fill_payload["fill_price"]), fee
                )
                fill_payload["position_after_qty"] = portfolio.position
                fill_payload["decision_time_utc"] = row.decision_time_utc
                fill_payload["symbol"] = row.symbol
                fill_payload["predicted_action"] = prediction["policy_action"]
                trade_records.append(dict(fill_payload))
                record_equity(event_time)

        accrue_exposure(row.target_end_time_utc)
        end_mark, equity = record_equity(row.target_end_time_utc)
        absolute_notional = portfolio.absolute_notional_settlement(end_mark)
        exposure_ratio = absolute_notional / abs(equity) if equity else math.inf
        decision_records.append(
            {
                "symbol": row.symbol,
                "decision_time_utc": row.decision_time_utc,
                "target_end_time_utc": row.target_end_time_utc,
                "bot_position_before_qty": position_before,
                "predicted_action": prediction["policy_action"],
                "argmax_action": prediction["argmax_action"],
                "confidence": prediction["confidence"],
                "order_attempted": bool(signed_order),
                "signed_requested_qty": signed_order,
                "fill_status": fill["fill_status"],
                "filled_qty": int(fill.get("filled_qty", 0)),
                "signed_filled_qty": int(fill.get("signed_filled_qty", 0)),
                "bot_position_after_qty": portfolio.position,
                "interval_funding": interval_funding,
                "end_mark_price": end_mark,
                "end_equity": equity,
                **{
                    f"probability__{label}": probability
                    for label, probability in prediction["probabilities"].items()
                },
            }
        )
        if equity <= 0 and config.stop_on_nonpositive_equity:
            stopped_early = True
            break

    decisions_frame = pd.DataFrame(decision_records)
    trades_frame = pd.DataFrame(
        trade_records,
        columns=[
            "symbol",
            "decision_time_utc",
            "arrival_time_utc",
            "quote_time_utc",
            "fill_time_utc",
            "predicted_action",
            "fill_status",
            "price_source",
            "requested_qty",
            "signed_requested_qty",
            "filled_qty",
            "signed_filled_qty",
            "reference_price",
            "unrounded_fill_price",
            "fill_price",
            "top_size",
            "capacity_qty",
            "lot_size_contracts",
            "price_tick_size",
            "fill_delay_ms",
            "random_draw",
            "fee",
            "slippage_cost",
            "position_before_qty",
            "position_after_qty",
        ],
    )
    funding_frame = pd.DataFrame(
        funding_records,
        columns=[
            "funding_time_utc",
            "position_qty",
            "mark_price",
            "funding_rate",
            "funding_payment",
        ],
    )
    equity_curve = pd.DataFrame(equity_records)
    equity_curve["period_return"] = equity_curve["equity"].pct_change().fillna(0.0)
    running_peak = equity_curve["equity"].cummax()
    equity_curve["drawdown"] = equity_curve["equity"] / running_peak - 1.0
    equity_curve["is_exposed"] = equity_curve["position_qty"].ne(0)
    equity_curve["is_long"] = equity_curve["position_qty"].gt(0)
    equity_curve["is_short"] = equity_curve["position_qty"].lt(0)
    equity_curve["absolute_exposure_ratio"] = equity_curve[
        "absolute_exposure_ratio"
    ].fillna(0.0)
    summary = _summary(
        config,
        cost_multiplier,
        equity_curve,
        trades_frame,
        decisions_frame,
        funding_frame,
        exposure_seconds,
        stopped_early,
    )
    return summary, equity_curve, trades_frame, decisions_frame, funding_frame
