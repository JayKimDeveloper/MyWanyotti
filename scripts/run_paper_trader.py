#!/usr/bin/env python3
"""Run a persistent public-market paper trader without order API access."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wonyotti_pipeline.inference import (  # noqa: E402
    CatBoostActionPredictor,
    InferenceContractError,
)
from wonyotti_pipeline.live_market import (  # noqa: E402
    BitmexPublicWebSocket,
    PublicMarketError,
)
from wonyotti_pipeline.paper import (  # noqa: E402
    PaperStore,
    PaperTradingEngine,
    PaperTradingError,
    load_paper_config,
    validate_runtime_contract,
)
from wonyotti_pipeline.training import file_sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/paper_trading.example.toml")
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=0,
        help="Stop after this many decisions; zero runs until interrupted.",
    )
    parser.add_argument(
        "--acknowledge-research-assumptions", action="store_true"
    )
    parser.add_argument(
        "--allow-unverified-contract",
        action="store_true",
        help="Engineering smoke only; does not authorize performance claims.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise PaperTradingError(f"required artifact not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PaperTradingError(f"JSON artifact must contain an object: {path}")
    return value


def emit(event: str, **values: object) -> None:
    print(
        json.dumps(
            {
                "event": event,
                "logged_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                **values,
            },
            ensure_ascii=False,
            default=str,
        ),
        flush=True,
    )


def main() -> int:
    args = parse_args()
    if args.max_decisions < 0:
        emit("STARTUP_BLOCKED", error="max_decisions must be nonnegative")
        return 2
    store: PaperStore | None = None
    source: BitmexPublicWebSocket | None = None
    try:
        config = load_paper_config(args.config, args.symbol)
        if (
            config.backtest.assumption_status != "HISTORICALLY_VERIFIED"
            and not args.acknowledge_research_assumptions
        ):
            raise PaperTradingError(
                "paper config is not historically verified; pass "
                "--acknowledge-research-assumptions"
            )
        required_artifacts = [
            "model.cbm",
            "feature_schema.json",
            "class_labels.json",
            "training_config.json",
            "runtime_contract.json",
        ]
        missing_artifacts = [
            str(args.model_dir / name)
            for name in required_artifacts
            if not (args.model_dir / name).exists()
        ]
        if missing_artifacts:
            raise PaperTradingError(
                "required model artifacts not found: "
                + ", ".join(missing_artifacts)
            )
        contract = read_json(args.model_dir / "runtime_contract.json")
        contract_checks = validate_runtime_contract(
            contract, config, args.allow_unverified_contract
        )
        predictor = CatBoostActionPredictor(args.model_dir)
        model_version = file_sha256(args.model_dir / "model.cbm")
        config_sha256 = file_sha256(args.config)
        store = PaperStore(config.sqlite_path)
        engine = PaperTradingEngine(
            config,
            predictor,
            store,
            model_version,
            config_sha256,
        )
        source = BitmexPublicWebSocket(
            config.websocket_url,
            args.symbol,
            config.connect_timeout_seconds,
            config.receive_timeout_seconds,
        )
    except (OSError, ValueError, PaperTradingError, InferenceContractError) as error:
        emit(
            "STARTUP_BLOCKED",
            error=str(error),
            model_loaded=False,
            training_started=False,
            actual_order_api_enabled=False,
        )
        if store is not None:
            store.close()
        return 2

    emit(
        "STARTED",
        symbol=args.symbol,
        model_version=model_version,
        model_loaded_once=True,
        model_loaded_at_utc=predictor.loaded_at_utc,
        training_started=False,
        actual_order_api_enabled=False,
        runtime_contract_checks=contract_checks,
        sqlite_path=str(config.sqlite_path),
    )
    reconnect_seconds = config.reconnect_initial.total_seconds()
    completed_decisions = 0
    stopping = False
    try:
        while not stopping:
            try:
                source.connect()
                connected_at = pd.Timestamp.now(tz="UTC").as_unit("ns")
                engine.on_connected(connected_at)
                emit("CONNECTED", url=source.url, symbol=args.symbol)
                reconnect_seconds = config.reconnect_initial.total_seconds()
                last_ping = time.monotonic()
                while True:
                    events = source.receive()
                    if events:
                        engine.on_market_events(events)
                    decision_count = engine.run_due_decisions()
                    if decision_count:
                        completed_decisions += decision_count
                        emit(
                            "DECISIONS_RECORDED",
                            count=decision_count,
                            total=completed_decisions,
                        )
                    if (
                        args.max_decisions
                        and completed_decisions >= args.max_decisions
                    ):
                        stopping = True
                        break
                    if (
                        time.monotonic() - last_ping
                        >= config.ping_interval.total_seconds()
                    ):
                        source.ping()
                        last_ping = time.monotonic()
            except KeyboardInterrupt:
                stopping = True
            except PublicMarketError as error:
                now = pd.Timestamp.now(tz="UTC").as_unit("ns")
                engine.on_disconnected(str(error), now)
                emit(
                    "RECONNECT_WAIT",
                    error=str(error),
                    delay_seconds=reconnect_seconds,
                )
                source.close()
                time.sleep(reconnect_seconds)
                reconnect_seconds = min(
                    reconnect_seconds * 2,
                    config.reconnect_max.total_seconds(),
                )
    finally:
        engine.on_disconnected(
            "PROCESS_STOPPED", pd.Timestamp.now(tz="UTC").as_unit("ns")
        )
        source.close()
        if store is not None:
            store.close()
    emit("STOPPED", decisions_recorded=completed_decisions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
