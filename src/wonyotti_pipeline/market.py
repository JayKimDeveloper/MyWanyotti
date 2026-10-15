"""BitMEX public trade and quote loading with backward-looking features."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .executions import parse_utc


TRADE_COLUMNS = ["timestamp", "symbol", "side", "size", "price", "trdMatchID"]
QUOTE_COLUMNS = ["timestamp", "symbol", "bidSize", "bidPrice", "askPrice", "askSize"]


def discover_daily_files(
    market_dir: Path,
    kind: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> tuple[list[Path], list[str]]:
    directory = market_dir / kind
    candidates = sorted([*directory.glob("*.csv"), *directory.glob("*.csv.gz")])
    by_date = {path.name[:8]: path for path in candidates if path.name[:8].isdigit()}
    required_dates = pd.date_range(
        start_utc.normalize(),
        (end_utc - pd.Timedelta(nanoseconds=1)).normalize(),
        freq="1D",
        tz="UTC",
    )
    keys = [timestamp.strftime("%Y%m%d") for timestamp in required_dates]
    return [by_date[key] for key in keys if key in by_date], [
        key for key in keys if key not in by_date
    ]


def _read_market_files(
    paths: Iterable[Path],
    columns: list[str],
    symbols: set[str],
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    chunk_size: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        for chunk in pd.read_csv(path, usecols=columns, chunksize=chunk_size):
            chunk = chunk[chunk["symbol"].isin(symbols)]
            if chunk.empty:
                continue
            chunk = chunk.copy()
            chunk["market_time_utc"] = parse_utc(chunk["timestamp"])
            chunk = chunk[
                (chunk["market_time_utc"] >= start_utc)
                & (chunk["market_time_utc"] < end_utc)
            ]
            if not chunk.empty:
                frames.append(chunk)
    if not frames:
        return pd.DataFrame(columns=[*columns, "market_time_utc"])
    return pd.concat(frames, ignore_index=True).sort_values(
        ["symbol", "market_time_utc"], kind="stable"
    )


def load_public_trades(
    paths: Iterable[Path],
    symbols: set[str],
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    chunk_size: int = 500_000,
) -> pd.DataFrame:
    trades = _read_market_files(
        paths, TRADE_COLUMNS, symbols, start_utc, end_utc, chunk_size
    )
    return trades.rename(
        columns={
            "side": "market_side",
            "size": "market_size",
            "price": "market_price",
            "trdMatchID": "trade_match_id",
        }
    )


def load_public_quotes(
    paths: Iterable[Path],
    symbols: set[str],
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    chunk_size: int = 500_000,
) -> pd.DataFrame:
    quotes = _read_market_files(
        paths, QUOTE_COLUMNS, symbols, start_utc, end_utc, chunk_size
    )
    return quotes.rename(
        columns={
            "bidSize": "bid_size",
            "bidPrice": "bid_price",
            "askPrice": "ask_price",
            "askSize": "ask_size",
        }
    )


def _window_sum(cumulative: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return cumulative[right] - cumulative[left]


def build_market_features(
    decisions: pd.DataFrame,
    trades: pd.DataFrame,
    quotes: pd.DataFrame,
    lookback: pd.Timedelta,
    max_quote_age: pd.Timedelta,
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    lookback_ns = int(lookback.value)
    for symbol, symbol_decisions in decisions.groupby("symbol", sort=False):
        current = symbol_decisions.sort_values("decision_time_utc").copy()
        decision_ns = current["decision_time_utc"].astype("int64").to_numpy()

        symbol_trades = trades[trades["symbol"].eq(symbol)].sort_values(
            "market_time_utc"
        )
        trade_ns = symbol_trades["market_time_utc"].astype("int64").to_numpy()
        sizes = symbol_trades["market_size"].to_numpy(dtype="float64")
        prices = symbol_trades["market_price"].to_numpy(dtype="float64")
        sides = symbol_trades["market_side"].astype("string").to_numpy()
        left = np.searchsorted(trade_ns, decision_ns - lookback_ns, side="left")
        right = np.searchsorted(trade_ns, decision_ns, side="left")
        counts = right - left

        cumulative_size = np.concatenate(([0.0], np.cumsum(sizes)))
        buy_sizes = np.where(sides == "Buy", sizes, 0.0)
        sell_sizes = np.where(sides == "Sell", sizes, 0.0)
        cumulative_buy = np.concatenate(([0.0], np.cumsum(buy_sizes)))
        cumulative_sell = np.concatenate(([0.0], np.cumsum(sell_sizes)))
        cumulative_notional = np.concatenate(([0.0], np.cumsum(sizes * prices)))

        volume = _window_sum(cumulative_size, left, right)
        buy_volume = _window_sum(cumulative_buy, left, right)
        sell_volume = _window_sum(cumulative_sell, left, right)
        notional = _window_sum(cumulative_notional, left, right)
        current["market_trade_count"] = counts
        current["market_volume"] = volume
        current["market_buy_volume"] = buy_volume
        current["market_sell_volume"] = sell_volume
        current["market_signed_volume"] = buy_volume - sell_volume
        current["market_vwap"] = np.divide(
            notional,
            volume,
            out=np.full(len(current), np.nan),
            where=volume > 0,
        )

        last_trade_indices = right - 1
        has_prior_trade = last_trade_indices >= 0
        last_prices = np.full(len(current), np.nan)
        last_trade_times = np.full(
            len(current), np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
        )
        last_prices[has_prior_trade] = prices[last_trade_indices[has_prior_trade]]
        if has_prior_trade.any():
            last_trade_times[has_prior_trade] = trade_ns[
                last_trade_indices[has_prior_trade]
            ].astype("datetime64[ns]")
        current["market_last_price"] = last_prices
        current["market_last_trade_time_utc"] = pd.to_datetime(
            last_trade_times, utc=True
        )

        returns = np.full(len(current), np.nan)
        enough_trades = counts >= 2
        returns[enough_trades] = (
            prices[last_trade_indices[enough_trades]] / prices[left[enough_trades]] - 1.0
        )
        current["market_return_lookback"] = returns

        symbol_quotes = quotes[quotes["symbol"].eq(symbol)].sort_values(
            "market_time_utc"
        )
        quote_ns = symbol_quotes["market_time_utc"].astype("int64").to_numpy()
        quote_indices = np.searchsorted(quote_ns, decision_ns, side="left") - 1
        has_quote = quote_indices >= 0
        for column in ("bid_price", "ask_price", "bid_size", "ask_size"):
            values = symbol_quotes[column].to_numpy(dtype="float64")
            selected = np.full(len(current), np.nan)
            selected[has_quote] = values[quote_indices[has_quote]]
            current[column] = selected

        quote_times = np.full(
            len(current), np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
        )
        if has_quote.any():
            quote_times[has_quote] = quote_ns[quote_indices[has_quote]].astype(
                "datetime64[ns]"
            )
        current["quote_time_utc"] = pd.to_datetime(quote_times, utc=True)
        current["quote_age_ms"] = (
            current["decision_time_utc"] - current["quote_time_utc"]
        ).dt.total_seconds() * 1000.0
        current["mid_price"] = (current["bid_price"] + current["ask_price"]) / 2.0
        current["spread"] = current["ask_price"] - current["bid_price"]
        current["spread_bps"] = current["spread"] / current["mid_price"] * 10_000.0
        depth = current["bid_size"] + current["ask_size"]
        current["top_book_imbalance"] = np.divide(
            current["bid_size"] - current["ask_size"],
            depth,
            out=np.full(len(current), np.nan),
            where=depth.gt(0),
        )
        current["market_feature_max_time_utc"] = current[
            ["market_last_trade_time_utc", "quote_time_utc"]
        ].max(axis=1)

        status = np.full(len(current), "OK", dtype=object)
        status[~has_quote] = "MISSING_QUOTE"
        status[counts == 0] = "NO_TRADE_IN_LOOKBACK"
        stale = current["quote_age_ms"].gt(max_quote_age.total_seconds() * 1000.0)
        status[stale.fillna(False).to_numpy()] = "STALE_QUOTE"
        current["market_feature_status"] = status
        outputs.append(current)
    return pd.concat(outputs, ignore_index=True)


def compare_private_public_trades(
    executions: pd.DataFrame,
    public_trades: pd.DataFrame,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> tuple[dict[str, int | float], pd.DataFrame]:
    private = executions[
        executions["exectype"].eq("Trade")
        & executions["event_time_utc"].ge(start_utc)
        & executions["event_time_utc"].lt(end_utc)
    ].copy()
    public = public_trades.drop_duplicates("trade_match_id").copy()
    compared = private.merge(
        public[
            [
                "trade_match_id",
                "symbol",
                "market_time_utc",
                "market_side",
                "market_size",
                "market_price",
            ]
        ],
        left_on="trdmatchid",
        right_on="trade_match_id",
        how="left",
        suffixes=("_private", "_public"),
    )
    matched = compared["trade_match_id"].notna()
    price_match = matched & np.isclose(
        compared["lastpx"], compared["market_price"], equal_nan=False
    )
    time_delta_ms = (
        compared["event_time_utc"] - compared["market_time_utc"]
    ).dt.total_seconds().abs() * 1000.0
    time_match = matched & time_delta_ms.eq(0)
    size_match = matched & compared["lastqty"].eq(compared["market_size"])
    symbol_match = matched & compared["symbol_private"].eq(compared["symbol_public"])
    expected_side = matched & (
        (
            compared["lastliquidityind"].eq("RemovedLiquidity")
            & compared["side"].eq(compared["market_side"])
        )
        | (
            compared["lastliquidityind"].eq("AddedLiquidity")
            & compared["side"].ne(compared["market_side"])
        )
    )
    report = {
        "private_fill_count": int(len(compared)),
        "matched_trade_match_id_count": int(matched.sum()),
        "match_rate": float(matched.mean()) if len(compared) else 0.0,
        "exact_utc_time_match_count": int(time_match.sum()),
        "max_utc_time_delta_ms": (
            float(time_delta_ms[matched].max()) if matched.any() else None
        ),
        "price_match_count": int(price_match.sum()),
        "size_match_count": int(size_match.sum()),
        "symbol_match_count": int(symbol_match.sum()),
        "liquidity_adjusted_side_match_count": int(expected_side.sum()),
    }
    sample_columns = [
        "execid",
        "trdmatchid",
        "event_time_utc",
        "market_time_utc",
        "symbol_private",
        "side",
        "lastliquidityind",
        "lastqty",
        "lastpx",
        "market_side",
        "market_size",
        "market_price",
    ]
    return report, compared.loc[matched, sample_columns].head(20)
