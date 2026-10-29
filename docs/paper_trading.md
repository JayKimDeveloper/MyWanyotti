# Offline Paper Simulation

## Scope

`scripts/run_paper_trader.py` is designed to consume public exchange `trade` and `quote` streams. The model is loaded once at process start; training is a separate command. The simulator does not read private account state and does not submit, amend, or cancel exchange orders. It requires no exchange credentials.

The current model and data contracts must pass validation before a meaningful run. Network connectivity and live-stream parsing may be environment-dependent; a successful unit test is not proof of uninterrupted service operation.

## Example command

```bash
.venv/bin/python scripts/run_paper_trader.py \
  --model-dir models/catboost/full_v1 \
  --config configs/paper_trading.example.toml \
  --symbol XBTUSD \
  --acknowledge-research-assumptions
```

Use short, bounded runs when validating connectivity. Do not configure this command as a persistent background service without separate review.

## Runtime contract

The model's `runtime_contract.json` must match the configured exchange, symbols, decision interval, lookback, quote-age limit, and strict-as-of rule. Reuse the same market-feature and model-feature preparation functions as training. Feed the simulator's own position into each decision; never substitute a third party's reconstructed position.

## Decisions and simulated fills

At each UTC decision boundary, features include only events earlier than the boundary. After decision latency, the first eligible public quote may be used to simulate an execution, with participation, lot size, fill probability, slippage, and fees applied from configuration. Do not use historical account execution prices as simulated fills. Do not allow an action that requires an unsupported immediate position reversal.

## Data quality and risk controls

- After a disconnect/restart, require enough consecutive data to fill the lookback window before considering actions.
- Stale quotes may be logged, but must block position increases; any reduction-only behavior remains a simulation rule, not a live order.
- Apply maximum-position and session-loss limits and log every block reason.
- Disable funding unless a validated rate feed is available; disabled liquidation means the simulator is not a realistic leveraged-risk estimate.
- No private authenticated stream or order endpoint is permitted.

## Recovery and audit log

SQLite is the local source of truth. Use durable transactions and stable keys to deduplicate trades, quotes, decisions, and simulated fills. Persist position, cash, session baseline, next decision, pending simulated state, and random-generator state atomically. On restart, reconcile pending simulation state against saved public events. Refuse automatic recovery if model or configuration hashes differ. Reconnect with bounded backoff and reset warm-up state.

Decision records should include timestamps, feature cutoffs and values, model/config hashes, class order and probabilities, data-quality status, risk checks, simulated order/fill status, accounting before/after, and latency measurements.
