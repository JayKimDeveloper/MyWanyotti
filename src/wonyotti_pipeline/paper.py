"""Persistent, public-market-only paper trading runtime."""

from __future__ import annotations

import json
import math
import sqlite3
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from .backtest import (
    BacktestConfig,
    BacktestConfigError,
    Portfolio,
    load_backtest_config,
    signed_order_for_action,
    simulate_quote_fill,
)
from .dataset import MODEL_FEATURE_COLUMNS
from .live_market import PublicMarketEvent
from .market import build_market_features
from .training import serialize_feature_row


class PaperTradingError(RuntimeError):
    """Raised when persistent paper trading cannot safely continue."""


class ActionPredictor(Protocol):
    class_labels: list[str]
    loaded_at_utc: str

    def predict(self, record: dict[str, object]) -> dict[str, object]: ...


@dataclass(frozen=True)
class PaperTradingConfig:
    backtest: BacktestConfig
    exchange: str
    websocket_url: str
    decision_interval: pd.Timedelta
    decision_lag: pd.Timedelta
    connect_timeout_seconds: float
    receive_timeout_seconds: float
    ping_interval: pd.Timedelta
    reconnect_initial: pd.Timedelta
    reconnect_max: pd.Timedelta
    market_retention: pd.Timedelta
    sqlite_path: Path
    max_abs_position_contracts: int
    max_session_loss_settlement: float
    loss_limit_policy: str


def load_paper_config(path: Path, symbol: str) -> PaperTradingConfig:
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    backtest = load_backtest_config(path, symbol)
    try:
        config = PaperTradingConfig(
            backtest=backtest,
            exchange=str(document["market"]["exchange"]),
            websocket_url=str(document["market"]["websocket_url"]),
            decision_interval=pd.Timedelta(document["decision"]["interval"]),
            decision_lag=pd.Timedelta(document["decision"]["lag"]),
            connect_timeout_seconds=float(
                document["market"]["connect_timeout_seconds"]
            ),
            receive_timeout_seconds=float(
                document["market"]["receive_timeout_seconds"]
            ),
            ping_interval=pd.Timedelta(document["market"]["ping_interval"]),
            reconnect_initial=pd.Timedelta(
                document["market"]["reconnect_initial"]
            ),
            reconnect_max=pd.Timedelta(document["market"]["reconnect_max"]),
            market_retention=pd.Timedelta(document["state"]["market_retention"]),
            sqlite_path=Path(document["state"]["sqlite_path"]),
            max_abs_position_contracts=int(
                document["risk"]["max_abs_position_contracts"]
            ),
            max_session_loss_settlement=float(
                document["risk"]["max_session_loss_settlement"]
            ),
            loss_limit_policy=str(document["risk"]["loss_limit_policy"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BacktestConfigError(f"invalid paper trading config: {error}") from error
    validate_paper_config(config)
    return config


def validate_paper_config(config: PaperTradingConfig) -> None:
    if config.exchange != "BitMEX":
        raise BacktestConfigError("only exchange=BitMEX is supported")
    if not config.websocket_url.startswith("wss://"):
        raise BacktestConfigError("websocket_url must use wss://")
    if config.decision_interval <= pd.Timedelta(0):
        raise BacktestConfigError("decision interval must be positive")
    if config.decision_lag < pd.Timedelta(0):
        raise BacktestConfigError("decision lag must be nonnegative")
    for name, value in (
        ("connect_timeout_seconds", config.connect_timeout_seconds),
        ("receive_timeout_seconds", config.receive_timeout_seconds),
    ):
        if value <= 0 or not math.isfinite(value):
            raise BacktestConfigError(f"{name} must be finite and positive")
    if config.ping_interval <= pd.Timedelta(0):
        raise BacktestConfigError("ping_interval must be positive")
    if config.reconnect_initial <= pd.Timedelta(0):
        raise BacktestConfigError("reconnect_initial must be positive")
    if config.reconnect_max < config.reconnect_initial:
        raise BacktestConfigError("reconnect_max must be at least reconnect_initial")
    if config.market_retention < config.backtest.lookback + config.decision_interval:
        raise BacktestConfigError(
            "market_retention must cover lookback plus one decision interval"
        )
    if config.max_abs_position_contracts <= 0:
        raise BacktestConfigError("max_abs_position_contracts must be positive")
    lot_size = config.backtest.instrument.lot_size_contracts
    if config.max_abs_position_contracts % lot_size:
        raise BacktestConfigError("max_abs_position_contracts must be a whole lot")
    if abs(config.backtest.initial_position_contracts) > config.max_abs_position_contracts:
        raise BacktestConfigError("initial position exceeds max_abs_position_contracts")
    if (
        not math.isfinite(config.max_session_loss_settlement)
        or config.max_session_loss_settlement <= 0
    ):
        raise BacktestConfigError(
            "max_session_loss_settlement must be finite and positive"
        )
    if config.loss_limit_policy != "REDUCE_ONLY":
        raise BacktestConfigError("only loss_limit_policy=REDUCE_ONLY is supported")
    if config.backtest.funding_mode != "disabled":
        raise BacktestConfigError(
            "paper runtime requires funding.mode=disabled until live rates are ingested"
        )


def validate_runtime_contract(
    contract: dict[str, object],
    config: PaperTradingConfig,
    allow_unverified: bool = False,
) -> dict[str, object]:
    required = {
        "data_contract_version",
        "status",
        "exchange",
        "symbols",
        "decision_interval",
        "lookback",
        "max_quote_age",
        "strict_asof_rule",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise PaperTradingError(f"runtime_contract.json missing fields: {missing}")
    checks = {
        "data_contract_version": contract["data_contract_version"] == "v0.3",
        "verified": contract["status"] == "VERIFIED",
        "exchange": contract["exchange"] == config.exchange,
        "symbol": config.backtest.instrument.symbol in list(contract["symbols"]),
        "decision_interval": pd.Timedelta(contract["decision_interval"])
        == config.decision_interval,
        "lookback": pd.Timedelta(contract["lookback"])
        == config.backtest.lookback,
        "max_quote_age": pd.Timedelta(contract["max_quote_age"])
        == config.backtest.max_quote_age,
        "strict_asof": contract["strict_asof_rule"]
        == "market_time_utc < decision_time_utc",
    }
    blocking = [name for name, passed in checks.items() if not passed]
    if allow_unverified:
        blocking = [name for name in blocking if name != "verified"]
    if blocking:
        raise PaperTradingError(f"runtime contract mismatch: {blocking}")
    return checks


def possible_actions(position: int) -> list[str]:
    if position > 0:
        return ["LONG_INCREASE", "LONG_DECREASE", "NO_FILL"]
    if position < 0:
        return ["SHORT_INCREASE", "SHORT_DECREASE", "NO_FILL"]
    return ["LONG_INCREASE", "SHORT_INCREASE", "NO_FILL"]


def apply_action_constraints(
    action: str,
    position: int,
    action_size: int,
    max_abs_position: int,
    stale_market: bool,
    session_loss: float,
    max_session_loss: float,
    has_pending_order: bool,
) -> tuple[int, str, str]:
    signed_quantity, semantic_status = signed_order_for_action(
        action, position, action_size
    )
    if signed_quantity == 0:
        return 0, "NO_ORDER", semantic_status
    if has_pending_order:
        return 0, "NO_ORDER", "PENDING_ORDER_EXISTS"
    resulting_position = position + signed_quantity
    is_increase = abs(resulting_position) > abs(position)
    if stale_market and is_increase:
        return 0, "NO_ORDER", "STALE_MARKET_REDUCE_ONLY"
    if session_loss >= max_session_loss and is_increase:
        return 0, "NO_ORDER", "LOSS_LIMIT_REDUCE_ONLY"
    if abs(resulting_position) > max_abs_position:
        return 0, "NO_ORDER", "MAX_POSITION_LIMIT"
    return signed_quantity, action, "ALLOWED"


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _utc_ns(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.as_unit("ns")


class PaperStore:
    """SQLite event log and atomic runtime checkpoint."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS market_trades (
                event_key TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                market_time_utc TEXT NOT NULL,
                received_time_utc TEXT NOT NULL,
                market_side TEXT NOT NULL,
                market_size REAL NOT NULL,
                market_price REAL NOT NULL,
                trade_match_id TEXT
            );
            CREATE INDEX IF NOT EXISTS market_trades_time
                ON market_trades(symbol, market_time_utc);
            CREATE TABLE IF NOT EXISTS market_quotes (
                event_key TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                market_time_utc TEXT NOT NULL,
                received_time_utc TEXT NOT NULL,
                bid_size REAL NOT NULL,
                bid_price REAL NOT NULL,
                ask_price REAL NOT NULL,
                ask_size REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS market_quotes_time
                ON market_quotes(symbol, market_time_utc);
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                decision_time_utc TEXT NOT NULL,
                evaluated_at_utc TEXT NOT NULL,
                completed_at_utc TEXT,
                status TEXT NOT NULL,
                model_version TEXT NOT NULL,
                data_time_utc TEXT,
                model_action TEXT,
                final_action TEXT NOT NULL,
                position_after_qty INTEGER NOT NULL,
                equity_after REAL,
                total_latency_ms REAL,
                record_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS virtual_fills (
                fill_id TEXT PRIMARY KEY,
                decision_id TEXT NOT NULL UNIQUE,
                symbol TEXT NOT NULL,
                fill_time_utc TEXT NOT NULL,
                signed_filled_qty INTEGER NOT NULL,
                fill_price REAL NOT NULL,
                fee REAL NOT NULL,
                record_json TEXT NOT NULL,
                FOREIGN KEY(decision_id) REFERENCES decisions(decision_id)
            );
            CREATE TABLE IF NOT EXISTS runtime_state (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                model_version TEXT NOT NULL,
                config_sha256 TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL,
                state_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_time_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details_json TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def record_market_events(self, events: list[PublicMarketEvent]) -> dict[str, int]:
        inserted = {"trade": 0, "quote": 0}
        with self.connection:
            for event in events:
                if event.kind == "trade":
                    cursor = self.connection.execute(
                        """
                        INSERT OR IGNORE INTO market_trades VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event.event_key,
                            event.symbol,
                            event.market_time_utc.isoformat(),
                            event.received_time_utc.isoformat(),
                            event.values["market_side"],
                            event.values["market_size"],
                            event.values["market_price"],
                            event.values["trade_match_id"],
                        ),
                    )
                else:
                    cursor = self.connection.execute(
                        """
                        INSERT OR IGNORE INTO market_quotes VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event.event_key,
                            event.symbol,
                            event.market_time_utc.isoformat(),
                            event.received_time_utc.isoformat(),
                            event.values["bid_size"],
                            event.values["bid_price"],
                            event.values["ask_price"],
                            event.values["ask_size"],
                        ),
                    )
                inserted[event.kind] += int(cursor.rowcount == 1)
        return inserted

    def market_frames(
        self, symbol: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        trades = pd.read_sql_query(
            """
            SELECT symbol, market_time_utc, market_side, market_size,
                   market_price, trade_match_id
            FROM market_trades
            WHERE symbol = ? AND market_time_utc >= ? AND market_time_utc < ?
            ORDER BY market_time_utc, event_key
            """,
            self.connection,
            params=(symbol, start.isoformat(), end.isoformat()),
        )
        quotes = pd.read_sql_query(
            """
            SELECT symbol, market_time_utc, bid_size, bid_price, ask_price, ask_size
            FROM market_quotes
            WHERE symbol = ? AND market_time_utc >= ? AND market_time_utc < ?
            ORDER BY market_time_utc, event_key
            """,
            self.connection,
            params=(symbol, start.isoformat(), end.isoformat()),
        )
        trades["market_time_utc"] = pd.to_datetime(
            trades["market_time_utc"], utc=True
        ).dt.as_unit("ns")
        quotes["market_time_utc"] = pd.to_datetime(
            quotes["market_time_utc"], utc=True
        ).dt.as_unit("ns")
        return trades, quotes

    def first_quote_between(
        self, symbol: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series | None:
        frame = pd.read_sql_query(
            """
            SELECT symbol, market_time_utc, received_time_utc,
                   bid_size, bid_price, ask_price, ask_size
            FROM market_quotes
            WHERE symbol = ? AND market_time_utc >= ? AND market_time_utc <= ?
            ORDER BY market_time_utc, event_key LIMIT 1
            """,
            self.connection,
            params=(symbol, start.isoformat(), end.isoformat()),
        )
        if frame.empty:
            return None
        frame["market_time_utc"] = pd.to_datetime(
            frame["market_time_utc"], utc=True
        ).dt.as_unit("ns")
        frame["received_time_utc"] = pd.to_datetime(
            frame["received_time_utc"], utc=True
        ).dt.as_unit("ns")
        return frame.iloc[0]

    def latest_trade(self, symbol: str) -> tuple[pd.Timestamp, float] | None:
        row = self.connection.execute(
            """
            SELECT market_time_utc, market_price FROM market_trades
            WHERE symbol = ? ORDER BY market_time_utc DESC, event_key DESC LIMIT 1
            """,
            (symbol,),
        ).fetchone()
        if row is None:
            return None
        return _utc_ns(row[0]), float(row[1])

    def load_state(self) -> tuple[str, str, dict[str, object]] | None:
        row = self.connection.execute(
            "SELECT model_version, config_sha256, state_json FROM runtime_state WHERE singleton = 1"
        ).fetchone()
        if row is None:
            return None
        return str(row[0]), str(row[1]), json.loads(row[2])

    def _upsert_state(
        self,
        model_version: str,
        config_sha256: str,
        state: dict[str, object],
        updated_at: pd.Timestamp,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO runtime_state VALUES (1, ?, ?, ?, ?)
            ON CONFLICT(singleton) DO UPDATE SET
                model_version=excluded.model_version,
                config_sha256=excluded.config_sha256,
                updated_at_utc=excluded.updated_at_utc,
                state_json=excluded.state_json
            """,
            (model_version, config_sha256, updated_at.isoformat(), _json(state)),
        )

    def save_state(
        self,
        model_version: str,
        config_sha256: str,
        state: dict[str, object],
        updated_at: pd.Timestamp,
    ) -> None:
        with self.connection:
            self._upsert_state(model_version, config_sha256, state, updated_at)

    def decision_exists(self, decision_id: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM decisions WHERE decision_id = ?", (decision_id,)
        ).fetchone()
        return row is not None

    def record_decision(
        self,
        record: dict[str, object],
        model_version: str,
        config_sha256: str,
        state: dict[str, object],
        updated_at: pd.Timestamp,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["decision_id"],
                    record["symbol"],
                    record["decision_time_utc"],
                    record["evaluated_at_utc"],
                    record.get("completed_at_utc"),
                    record["status"],
                    record["model_version"],
                    record.get("data_time_utc"),
                    record.get("model_action"),
                    record["final_action"],
                    record["balance"]["position_qty"],
                    record["balance"].get("equity"),
                    record.get("total_latency_ms"),
                    _json(record),
                ),
            )
            self._upsert_state(model_version, config_sha256, state, updated_at)

    def load_decision(self, decision_id: str) -> dict[str, object]:
        row = self.connection.execute(
            "SELECT record_json FROM decisions WHERE decision_id = ?", (decision_id,)
        ).fetchone()
        if row is None:
            raise PaperTradingError(f"decision not found: {decision_id}")
        return json.loads(row[0])

    def complete_decision(
        self,
        record: dict[str, object],
        fill_record: dict[str, object] | None,
        model_version: str,
        config_sha256: str,
        state: dict[str, object],
        updated_at: pd.Timestamp,
    ) -> None:
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE decisions SET completed_at_utc=?, status=?, data_time_utc=?,
                    model_action=?, final_action=?, position_after_qty=?, equity_after=?,
                    total_latency_ms=?, record_json=? WHERE decision_id=?
                """,
                (
                    record.get("completed_at_utc"),
                    record["status"],
                    record.get("data_time_utc"),
                    record.get("model_action"),
                    record["final_action"],
                    record["balance"]["position_qty"],
                    record["balance"].get("equity"),
                    record.get("total_latency_ms"),
                    _json(record),
                    record["decision_id"],
                ),
            )
            if cursor.rowcount != 1:
                raise PaperTradingError(
                    f"cannot complete missing decision: {record['decision_id']}"
                )
            if fill_record is not None:
                self.connection.execute(
                    """
                    INSERT INTO virtual_fills VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fill_record["fill_id"],
                        fill_record["decision_id"],
                        fill_record["symbol"],
                        fill_record["fill_time_utc"],
                        fill_record["signed_filled_qty"],
                        fill_record["fill_price"],
                        fill_record["fee"],
                        _json(fill_record),
                    ),
                )
            self._upsert_state(model_version, config_sha256, state, updated_at)

    def record_runtime_event(
        self, event_time: pd.Timestamp, event_type: str, details: dict[str, object]
    ) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO runtime_events(event_time_utc, event_type, details_json) VALUES (?, ?, ?)",
                (event_time.isoformat(), event_type, _json(details)),
            )

    def prune_market_before(self, cutoff: pd.Timestamp) -> None:
        with self.connection:
            self.connection.execute(
                "DELETE FROM market_trades WHERE market_time_utc < ?",
                (cutoff.isoformat(),),
            )
            self.connection.execute(
                "DELETE FROM market_quotes WHERE market_time_utc < ?",
                (cutoff.isoformat(),),
            )

    def decision_records(self) -> list[dict[str, object]]:
        rows = self.connection.execute(
            "SELECT record_json FROM decisions ORDER BY decision_time_utc"
        ).fetchall()
        return [json.loads(row[0]) for row in rows]


class PaperTradingEngine:
    """Stateful model-to-paper-fill pipeline with no private exchange APIs."""

    def __init__(
        self,
        config: PaperTradingConfig,
        predictor: ActionPredictor,
        store: PaperStore,
        model_version: str,
        config_sha256: str,
    ) -> None:
        self.config = config
        self.predictor = predictor
        self.store = store
        self.model_version = model_version
        self.config_sha256 = config_sha256
        self.random = np.random.default_rng(config.backtest.random_seed)
        saved = store.load_state()
        if saved is None:
            cash = (
                config.backtest.initial_equity
                if config.backtest.initial_position_contracts == 0
                else None
            )
            self.state: dict[str, object] = {
                "cash": cash,
                "position_qty": config.backtest.initial_position_contracts,
                "latest_mark": None,
                "latest_mark_time_utc": None,
                "session_start_equity": config.backtest.initial_equity,
                "session_peak_equity": config.backtest.initial_equity,
                "next_decision_time_utc": None,
                "pending_order": None,
                "continuous_since_utc": None,
                "rng_state": self.random.bit_generator.state,
            }
        else:
            saved_model, saved_config, self.state = saved
            if saved_model != model_version or saved_config != config_sha256:
                raise PaperTradingError(
                    "persisted state model/config fingerprint does not match startup"
                )
            if self.state.get("rng_state") is not None:
                self.random.bit_generator.state = self.state["rng_state"]
        latest = store.latest_trade(config.backtest.instrument.symbol)
        if latest is not None:
            self.state["latest_mark_time_utc"] = latest[0].isoformat()
            self.state["latest_mark"] = latest[1]

    @property
    def symbol(self) -> str:
        return self.config.backtest.instrument.symbol

    def _now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz="UTC").as_unit("ns")

    def _state_for_storage(self) -> dict[str, object]:
        self.state["rng_state"] = self.random.bit_generator.state
        return self.state

    def _portfolio(self) -> Portfolio | None:
        cash = self.state.get("cash")
        if cash is None:
            return None
        return Portfolio(
            self.config.backtest.instrument,
            float(cash),
            int(self.state["position_qty"]),
        )

    def _save_portfolio(self, portfolio: Portfolio) -> None:
        self.state["cash"] = portfolio.cash
        self.state["position_qty"] = portfolio.position

    def _balance(self, mark: float | None = None) -> dict[str, object]:
        portfolio = self._portfolio()
        current_mark = mark if mark is not None else self.state.get("latest_mark")
        equity = None
        if portfolio is not None and current_mark is not None:
            equity = portfolio.equity(float(current_mark))
        return {
            "settlement_currency": self.config.backtest.instrument.settlement_currency,
            "cash": portfolio.cash if portfolio is not None else None,
            "equity": equity,
            "position_qty": int(self.state["position_qty"]),
            "mark_price": float(current_mark) if current_mark is not None else None,
            "session_start_equity": self.state["session_start_equity"],
            "session_peak_equity": self.state["session_peak_equity"],
        }

    def _candidate_boundary(self, now: pd.Timestamp) -> pd.Timestamp:
        interval_ns = int(self.config.decision_interval.value)
        floor_ns = int(now.value) // interval_ns * interval_ns
        floor = pd.Timestamp(floor_ns, unit="ns", tz="UTC")
        if now <= floor + self.config.decision_lag:
            return floor
        return floor + self.config.decision_interval

    def on_connected(self, connected_at: pd.Timestamp | None = None) -> None:
        now = connected_at if connected_at is not None else self._now()
        self.state["continuous_since_utc"] = now.isoformat()
        candidate = self._candidate_boundary(now)
        next_value = self.state.get("next_decision_time_utc")
        if next_value is None:
            self.state["next_decision_time_utc"] = candidate.isoformat()
        else:
            next_decision = _utc_ns(next_value)
            if next_decision < candidate:
                skipped = int(
                    (candidate - next_decision) // self.config.decision_interval
                )
                self.state["next_decision_time_utc"] = candidate.isoformat()
                self.store.record_runtime_event(
                    now,
                    "MISSED_DECISIONS_SKIPPED_ON_RECOVERY",
                    {"count": skipped, "rescheduled_to_utc": candidate.isoformat()},
                )
        self.recover_pending(now)
        self.store.save_state(
            self.model_version,
            self.config_sha256,
            self._state_for_storage(),
            now,
        )
        self.store.record_runtime_event(
            now,
            "PUBLIC_MARKET_CONNECTED",
            {"symbol": self.symbol, "continuous_since_utc": now.isoformat()},
        )

    def on_disconnected(
        self, reason: str, disconnected_at: pd.Timestamp | None = None
    ) -> None:
        now = disconnected_at if disconnected_at is not None else self._now()
        self.state["continuous_since_utc"] = None
        self.store.save_state(
            self.model_version,
            self.config_sha256,
            self._state_for_storage(),
            now,
        )
        self.store.record_runtime_event(
            now, "PUBLIC_MARKET_DISCONNECTED", {"reason": reason}
        )

    def recover_pending(self, now: pd.Timestamp | None = None) -> None:
        current = now if now is not None else self._now()
        pending = self.state.get("pending_order")
        if not isinstance(pending, dict):
            return
        arrival = _utc_ns(pending["arrival_time_utc"])
        deadline = _utc_ns(pending["deadline_utc"])
        quote = self.store.first_quote_between(self.symbol, arrival, deadline)
        if quote is not None:
            self._process_pending_quote(quote, pd.Timestamp(quote["received_time_utc"]))
        else:
            self.expire_pending(current, "RECOVERY_NO_QUOTE_BEFORE_DEADLINE")

    def on_market_events(self, events: list[PublicMarketEvent]) -> dict[str, int]:
        accepted = [event for event in events if event.symbol == self.symbol]
        inserted = self.store.record_market_events(accepted)
        for event in accepted:
            if event.kind == "trade":
                mark_time = self.state.get("latest_mark_time_utc")
                if mark_time is None or event.market_time_utc >= pd.Timestamp(mark_time):
                    self.state["latest_mark"] = float(event.values["market_price"])
                    self.state["latest_mark_time_utc"] = event.market_time_utc.isoformat()
                if self.state.get("cash") is None:
                    portfolio = Portfolio.initialize(
                        self.config.backtest.instrument,
                        self.config.backtest.initial_equity,
                        self.config.backtest.initial_position_contracts,
                        float(event.values["market_price"]),
                    )
                    self._save_portfolio(portfolio)
                    self.store.save_state(
                        self.model_version,
                        self.config_sha256,
                        self._state_for_storage(),
                        event.received_time_utc,
                    )
            elif self.state.get("pending_order") is not None:
                quote = pd.Series(
                    {"market_time_utc": event.market_time_utc, **event.values}
                )
                self._process_pending_quote(quote, event.received_time_utc)
        return inserted

    def run_due_decisions(
        self, now: pd.Timestamp | None = None, maximum: int = 10
    ) -> int:
        current = now if now is not None else self._now()
        self.expire_pending(current, "NO_QUOTE_BEFORE_DEADLINE")
        next_value = self.state.get("next_decision_time_utc")
        if next_value is None:
            return 0
        count = 0
        while count < maximum:
            decision_time = _utc_ns(self.state["next_decision_time_utc"])
            if current < decision_time + self.config.decision_lag:
                break
            next_decision = decision_time + self.config.decision_interval
            self.state["next_decision_time_utc"] = next_decision.isoformat()
            decision_id = f"{self.symbol}:{decision_time.isoformat()}"
            if self.store.decision_exists(decision_id):
                self.store.save_state(
                    self.model_version,
                    self.config_sha256,
                    self._state_for_storage(),
                    current,
                )
            else:
                self._evaluate_decision(decision_time, current)
            count += 1
        self.store.prune_market_before(current - self.config.market_retention)
        return count

    def _evaluate_decision(
        self, decision_time: pd.Timestamp, evaluated_at: pd.Timestamp
    ) -> None:
        started = time.perf_counter()
        start = decision_time - self.config.backtest.lookback
        trades, quotes = self.store.market_frames(self.symbol, start, decision_time)
        decision = pd.DataFrame(
            {"symbol": [self.symbol], "decision_time_utc": [decision_time]}
        )
        featured = build_market_features(
            decision,
            trades,
            quotes,
            self.config.backtest.lookback,
            self.config.backtest.max_quote_age,
        )
        row = featured.iloc[0].copy()
        position = int(self.state["position_qty"])
        row["position_before_qty"] = position
        feature_record = serialize_feature_row(row)
        continuous_value = self.state.get("continuous_since_utc")
        coverage_complete = (
            continuous_value is not None
            and _utc_ns(continuous_value) <= start
        )
        feature_status = str(row["market_feature_status"])
        stale_market = feature_status == "STALE_QUOTE"
        input_status = (
            feature_status if coverage_complete else "WARMUP_AFTER_RECONNECT"
        )
        probabilities = {label: None for label in self.predictor.class_labels}
        prediction: dict[str, object] | None = None
        prediction_error: str | None = None
        if coverage_complete and feature_status in {"OK", "STALE_QUOTE"}:
            try:
                prediction = self.predictor.predict(feature_record)
                probabilities = dict(prediction["probabilities"])
            except (TypeError, ValueError) as error:
                prediction_error = str(error)
                input_status = "MODEL_INPUT_REJECTED"

        mark = row.get("market_last_price")
        mark_price = float(mark) if pd.notna(mark) else None
        balance_before = self._balance(mark_price)
        equity = balance_before.get("equity")
        session_loss = (
            float(self.state["session_start_equity"]) - float(equity)
            if equity is not None
            else math.inf
        )
        model_action = str(prediction["policy_action"]) if prediction else None
        signed_quantity = 0
        final_action = "NO_ORDER"
        action_reason = input_status
        if model_action is not None:
            signed_quantity, final_action, action_reason = apply_action_constraints(
                model_action,
                position,
                self.config.backtest.contracts_per_action,
                self.config.max_abs_position_contracts,
                stale_market,
                session_loss,
                self.config.max_session_loss_settlement,
                self.state.get("pending_order") is not None,
            )

        decision_id = f"{self.symbol}:{decision_time.isoformat()}"
        compute_latency_ms = (time.perf_counter() - started) * 1000.0
        data_time = row.get("market_feature_max_time_utc")
        data_time_value = (
            pd.Timestamp(data_time).isoformat() if pd.notna(data_time) else None
        )
        record: dict[str, object] = {
            "decision_id": decision_id,
            "symbol": self.symbol,
            "decision_time_utc": decision_time.isoformat(),
            "evaluated_at_utc": evaluated_at.isoformat(),
            "completed_at_utc": None,
            "status": "PENDING_FILL" if signed_quantity else "COMPLETED",
            "model_version": self.model_version,
            "model_loaded_at_utc": self.predictor.loaded_at_utc,
            "data_time_utc": data_time_value,
            "input_status": input_status,
            "input_state": {
                "features": feature_record,
                "market_feature_status": feature_status,
                "coverage_complete": coverage_complete,
                "continuous_since_utc": continuous_value,
                "possible_actions": possible_actions(position),
                "prediction_error": prediction_error,
            },
            "probabilities": probabilities,
            "argmax_action": prediction.get("argmax_action") if prediction else None,
            "model_action": model_action,
            "final_action": final_action,
            "action_reason": action_reason,
            "signed_requested_qty": signed_quantity,
            "virtual_fill": None,
            "balance_before": balance_before,
            "balance": balance_before,
            "decision_compute_latency_ms": compute_latency_ms,
            "total_latency_ms": None,
        }
        if signed_quantity:
            execution_start = evaluated_at
            arrival = execution_start + self.config.backtest.latency
            target_end = decision_time + self.config.decision_interval
            deadline = min(
                target_end, arrival + self.config.backtest.max_fill_wait
            )
            self.state["pending_order"] = {
                "decision_id": decision_id,
                "decision_time_utc": decision_time.isoformat(),
                "execution_start_time_utc": execution_start.isoformat(),
                "target_end_time_utc": target_end.isoformat(),
                "arrival_time_utc": arrival.isoformat(),
                "deadline_utc": deadline.isoformat(),
                "signed_requested_qty": signed_quantity,
                "model_action": model_action,
            }
        else:
            record["completed_at_utc"] = evaluated_at.isoformat()
            record["virtual_fill"] = {
                "fill_status": "NOT_REQUESTED",
                "reason": action_reason,
            }
            record["total_latency_ms"] = max(
                0.0, (evaluated_at - decision_time).total_seconds() * 1000.0
            )
        self.store.record_decision(
            record,
            self.model_version,
            self.config_sha256,
            self._state_for_storage(),
            evaluated_at,
        )

    def _process_pending_quote(
        self, quote: pd.Series, received_at: pd.Timestamp
    ) -> None:
        pending = self.state.get("pending_order")
        if not isinstance(pending, dict):
            return
        decision_time = _utc_ns(pending["decision_time_utc"])
        execution_start = _utc_ns(pending["execution_start_time_utc"])
        target_end = _utc_ns(pending["target_end_time_utc"])
        arrival = _utc_ns(pending["arrival_time_utc"])
        deadline = _utc_ns(pending["deadline_utc"])
        quote_time = _utc_ns(quote["market_time_utc"])
        if quote_time < arrival:
            return
        if quote_time > deadline:
            self._complete_pending_without_fill(
                received_at, "QUOTE_AFTER_DEADLINE"
            )
            return
        fill = simulate_quote_fill(
            quote,
            execution_start,
            target_end,
            int(pending["signed_requested_qty"]),
            self.config.backtest,
            1.0,
            self.random,
        )
        fill_record: dict[str, object] | None = None
        if int(fill.get("filled_qty", 0)) > 0:
            portfolio = self._portfolio()
            if portfolio is None:
                raise PaperTradingError("portfolio is not initialized")
            signed_filled = int(fill["signed_filled_qty"])
            fee = portfolio.fee(
                abs(signed_filled),
                float(fill["fill_price"]),
                self.config.backtest.taker_rate,
            )
            fill["fee"] = fee
            fill["slippage_cost"] = portfolio.slippage_cost(
                signed_filled,
                float(fill["reference_price"]),
                float(fill["fill_price"]),
            )
            fill["position_before_qty"] = portfolio.position
            portfolio.apply_fill(signed_filled, float(fill["fill_price"]), fee)
            fill["position_after_qty"] = portfolio.position
            fill["received_time_utc"] = received_at
            fill["decision_to_fill_ms"] = (
                pd.Timestamp(fill["fill_time_utc"]) - decision_time
            ).total_seconds() * 1000.0
            fill["ingestion_latency_ms"] = max(
                0.0, (received_at - quote_time).total_seconds() * 1000.0
            )
            self._save_portfolio(portfolio)
            fill_record = {
                **fill,
                "fill_id": f"paper-fill:{pending['decision_id']}",
                "decision_id": pending["decision_id"],
                "symbol": self.symbol,
                "fill_time_utc": pd.Timestamp(fill["fill_time_utc"]).isoformat(),
            }
        self._finalize_pending(fill, fill_record, received_at)

    def _complete_pending_without_fill(
        self, completed_at: pd.Timestamp, fill_status: str
    ) -> None:
        pending = self.state.get("pending_order")
        if not isinstance(pending, dict):
            return
        self._finalize_pending(
            {
                "fill_status": fill_status,
                "arrival_time_utc": pending["arrival_time_utc"],
                "requested_qty": abs(int(pending["signed_requested_qty"])),
                "signed_requested_qty": pending["signed_requested_qty"],
                "filled_qty": 0,
                "signed_filled_qty": 0,
                "price_source": "PUBLIC_QUOTE_AFTER_DECISION",
            },
            None,
            completed_at,
        )

    def _finalize_pending(
        self,
        fill: dict[str, object],
        fill_record: dict[str, object] | None,
        completed_at: pd.Timestamp,
    ) -> None:
        pending = self.state.get("pending_order")
        if not isinstance(pending, dict):
            return
        record = self.store.load_decision(str(pending["decision_id"]))
        record["status"] = "COMPLETED"
        record["completed_at_utc"] = completed_at.isoformat()
        record["virtual_fill"] = fill_record if fill_record is not None else fill
        balance = self._balance()
        equity = balance.get("equity")
        if equity is not None:
            self.state["session_peak_equity"] = max(
                float(self.state["session_peak_equity"]), float(equity)
            )
            balance["session_peak_equity"] = self.state["session_peak_equity"]
        record["balance"] = balance
        decision_time = _utc_ns(record["decision_time_utc"])
        record["total_latency_ms"] = max(
            0.0, (completed_at - decision_time).total_seconds() * 1000.0
        )
        self.state["pending_order"] = None
        self.store.complete_decision(
            record,
            fill_record,
            self.model_version,
            self.config_sha256,
            self._state_for_storage(),
            completed_at,
        )

    def expire_pending(self, now: pd.Timestamp, reason: str) -> None:
        pending = self.state.get("pending_order")
        if not isinstance(pending, dict):
            return
        if now > _utc_ns(pending["deadline_utc"]):
            self._complete_pending_without_fill(now, reason)
