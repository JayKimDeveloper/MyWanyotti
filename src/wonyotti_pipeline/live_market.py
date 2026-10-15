"""Unauthenticated BitMEX public trade and quote WebSocket adapter."""

from __future__ import annotations

import hashlib
import json
import queue
import threading
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlencode

import pandas as pd
import websocket


class PublicMarketError(RuntimeError):
    """Raised when the public market stream is invalid or disconnected."""


@dataclass(frozen=True)
class PublicMarketEvent:
    kind: str
    event_key: str
    symbol: str
    market_time_utc: pd.Timestamp
    received_time_utc: pd.Timestamp
    values: dict[str, object]


def _utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise PublicMarketError("public market timestamp must include UTC offset")
    return timestamp.tz_convert("UTC").as_unit("ns")


def _hash_key(kind: str, values: list[object]) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return f"{kind}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _trade_event(row: dict[str, Any], received: pd.Timestamp) -> PublicMarketEvent:
    required = {"timestamp", "symbol", "side", "size", "price"}
    missing = sorted(required - set(row))
    if missing:
        raise PublicMarketError(f"trade event missing fields: {missing}")
    market_time = _utc_timestamp(row["timestamp"])
    symbol = str(row["symbol"])
    match_id = row.get("trdMatchID")
    event_key = (
        f"trade:{symbol}:{match_id}"
        if match_id
        else _hash_key(
            "trade",
            [
                symbol,
                market_time.isoformat(),
                row["side"],
                row["size"],
                row["price"],
            ],
        )
    )
    return PublicMarketEvent(
        kind="trade",
        event_key=event_key,
        symbol=symbol,
        market_time_utc=market_time,
        received_time_utc=received,
        values={
            "market_side": str(row["side"]),
            "market_size": float(row["size"]),
            "market_price": float(row["price"]),
            "trade_match_id": str(match_id) if match_id else None,
        },
    )


def _quote_event(row: dict[str, Any], received: pd.Timestamp) -> PublicMarketEvent:
    required = {
        "timestamp",
        "symbol",
        "bidSize",
        "bidPrice",
        "askPrice",
        "askSize",
    }
    missing = sorted(required - set(row))
    if missing:
        raise PublicMarketError(f"quote event missing fields: {missing}")
    market_time = _utc_timestamp(row["timestamp"])
    symbol = str(row["symbol"])
    values = {
        "bid_size": float(row["bidSize"]),
        "bid_price": float(row["bidPrice"]),
        "ask_price": float(row["askPrice"]),
        "ask_size": float(row["askSize"]),
    }
    event_key = _hash_key(
        "quote",
        [
            symbol,
            market_time.isoformat(),
            values["bid_size"],
            values["bid_price"],
            values["ask_price"],
            values["ask_size"],
        ],
    )
    return PublicMarketEvent(
        kind="quote",
        event_key=event_key,
        symbol=symbol,
        market_time_utc=market_time,
        received_time_utc=received,
        values=values,
    )


def parse_bitmex_message(
    payload: str | bytes,
    received_time_utc: pd.Timestamp | None = None,
) -> list[PublicMarketEvent]:
    """Parse public table messages; acknowledgements and info messages are ignored."""
    received = (
        received_time_utc
        if received_time_utc is not None
        else pd.Timestamp.now(tz="UTC").as_unit("ns")
    )
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    try:
        message = json.loads(payload)
    except json.JSONDecodeError as error:
        raise PublicMarketError(f"invalid WebSocket JSON: {error}") from error
    if not isinstance(message, dict):
        raise PublicMarketError("WebSocket message must be a JSON object")
    if message.get("error"):
        raise PublicMarketError(f"BitMEX WebSocket error: {message['error']}")
    table = message.get("table")
    if table not in {"trade", "quote"}:
        return []
    if message.get("action") not in {"partial", "insert"}:
        return []
    data = message.get("data")
    if not isinstance(data, list):
        raise PublicMarketError("public table message data must be a list")
    factory = _trade_event if table == "trade" else _quote_event
    try:
        return [factory(row, received) for row in data if isinstance(row, dict)]
    except (KeyError, TypeError, ValueError) as error:
        raise PublicMarketError(f"invalid {table} event: {error}") from error


class BitmexPublicWebSocket:
    """Blocking, unauthenticated public market connection."""

    def __init__(
        self,
        base_url: str,
        symbol: str,
        connect_timeout_seconds: float,
        receive_timeout_seconds: float,
    ) -> None:
        separator = "&" if "?" in base_url else "?"
        query = urlencode({"subscribe": f"trade:{symbol},quote:{symbol}"})
        self.url = f"{base_url}{separator}{query}"
        self.symbol = symbol
        self.connect_timeout_seconds = connect_timeout_seconds
        self.receive_timeout_seconds = receive_timeout_seconds
        self._connection: websocket.WebSocket | None = None

    def connect(self) -> None:
        self.close()
        result: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)
        abandoned = threading.Event()

        def open_connection() -> None:
            try:
                connection = websocket.create_connection(
                    self.url,
                    timeout=self.connect_timeout_seconds,
                    enable_multithread=True,
                )
                if abandoned.is_set():
                    connection.close()
                else:
                    result.put(("connection", connection))
            except (OSError, websocket.WebSocketException) as error:
                if not abandoned.is_set():
                    result.put(("error", error))

        threading.Thread(target=open_connection, daemon=True).start()
        try:
            kind, value = result.get(timeout=self.connect_timeout_seconds)
        except queue.Empty as error:
            abandoned.set()
            raise PublicMarketError(
                f"WebSocket connect timed out after {self.connect_timeout_seconds}s"
            ) from error
        if kind == "error":
            raise PublicMarketError(f"WebSocket connect failed: {value}")
        self._connection = cast(websocket.WebSocket, value)
        self._connection.settimeout(self.receive_timeout_seconds)

    def receive(self) -> list[PublicMarketEvent] | None:
        if self._connection is None:
            raise PublicMarketError("WebSocket is not connected")
        try:
            payload = self._connection.recv()
        except websocket.WebSocketTimeoutException:
            return None
        except websocket.WebSocketException as error:
            raise PublicMarketError(f"WebSocket receive failed: {error}") from error
        if payload in {None, "", b""}:
            raise PublicMarketError("WebSocket closed by peer")
        return parse_bitmex_message(payload)

    def ping(self) -> None:
        if self._connection is None:
            raise PublicMarketError("WebSocket is not connected")
        try:
            self._connection.ping()
        except websocket.WebSocketException as error:
            raise PublicMarketError(f"WebSocket ping failed: {error}") from error

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
