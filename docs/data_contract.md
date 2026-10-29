# Data Contract

Status: implementation draft v0.3. Keep source fields, validated derived rules, and unresolved metadata distinct.

## General rules

- Treat `data/raw/` as immutable. Write derived material only to audit/interim/processed locations.
- A trade fill is a source row with `execType == Trade`; Funding and Settlement are not trade labels.
- Preserve native quantities in `CONTRACTS`. Do not convert without point-in-time instrument specifications.
- Offset-free account timestamps are localized under an explicit timezone assumption; do not silently present that assumption as verified.
- A feature at decision time `t` may use only information strictly earlier than `t`.
- Labels describe observed trade behavior in the half-open interval `[t, t+h)`.
- Select model inputs only from the fixed `MODEL_FEATURE_COLUMNS` allowlist. Do not automatically use audit, target, or current-metadata columns.

## Execution source rows

Each row represents one exchange-style execution event.

| Field | Contract |
|---|---|
| `execid` | Unique, non-null row identifier |
| `orderid` | Nullable order grouping key; repeats are expected |
| `trdmatchid` | Public-trade matching key; repeats are possible |
| `symbol` | Instrument identifier |
| `exectype` | `Trade`, `Funding`, or `Settlement` |
| `transacttime`, `timestamp` | Preserve original timestamp strings |
| `side` | `Buy`, `Sell`, or null |
| `lastqty`, `orderqty`, `leavesqty`, `cumqty` | Native contract quantities |
| `lastpx` | Trade execution price |

Do not use `orderid` or `trdmatchid` as the row key. Do not infer a Funding side. A Buy/Sell fill alone does not identify a long/short entry or exit. Preserve all partial-fill rows and their original execution identifiers.

## Normalized events and order audit

| Derived field | Rule |
|---|---|
| `event_time_utc` | Localize `transacttime` to UTC only under the recorded source assumption |
| `processed_time_utc` | Localize `timestamp` under the same explicit assumption |
| `event_kind` | Uppercase `exectype` |
| `trade_signed_qty` | Trade Buy `+lastqty`, Trade Sell `-lastqty`, otherwise zero |
| `source_timezone_assumption` | Record the exact assumption used |

Audit order rows by order identifier, event time, processing time, cumulative quantity, and execution identifier. Check that the first cumulative quantity equals the first fill, cumulative quantity never decreases, cumulative differences equal fill quantities, fill sums equal the final cumulative quantity, and symbol/side remain consistent. An `orderqty` change may reflect an amendment and is not by itself grounds for exclusion. Failed fill-increment checks should exclude the affected order from training.

## Position evidence

Prefer an explicit, timestamped position snapshot. When none exists, keep unknown positions unknown except where a separately documented and validated anchor is available. Funding-derived candidate quantities are conditional inferences, not exchange-certified snapshots. Never assume an initial flat position without evidence.

For an accepted anchor at time `a`, a conditional position immediately before `t` is:

```text
position_before_qty = anchor.position_qty
                      + sum(trade_signed_qty where a < event_time < t)
```

If no accepted anchor exists, store a null quantity and `position_status=UNKNOWN`. Preserve the anchor time, last included execution time, and the assumption/provenance status.

## Instrument units and metadata

Keep position quantities in contracts. Record observed settlement currency separately. Current instrument metadata is not point-in-time history and must not be back-applied to old data without evidence. Preserve instrument fields such as listing/expiry, position/quote/settlement currencies, lot size, tick size, multiplier, inverse/quanto flags, and the metadata retrieval time.

## Market inputs

Prefer public data from the same exchange and symbol. A public trade record should retain timestamp, symbol, side, size, price, and match identifier. A quote record should retain timestamp, symbol, best bid/ask prices and sizes. Record missing dates and source coverage. Mark quotes stale when they exceed the configured age; distinguish missing quote, no trade in lookback, and incomplete source coverage.

## Training rows and features

The row key is `(symbol, decision_time_utc)`. With decision `t`, horizon `h`, and market lookback `L`, a candidate row contains the decision key, as-of market features, position plus evidence status, and observed behavior during `[t,t+h)`. Do not create an incomplete final interval.

The initial feature allowlist includes symbol, decision time, pre-decision quantity and public-market statistics: trade count/volume, buy/sell/signed volume, VWAP, last price, lookback return, best bid/ask and sizes, midpoint, spread, spread in basis points, top-book imbalance, and quote age. The maximum feature timestamp and position-as-of timestamp are audit fields, not model inputs. Future candles, future fills/prices, realized or unrealized PnL, and target columns are prohibited as features.

## Labels and exclusions

Apply each signed fill to the preceding candidate position to derive atomic increase/decrease behavior. If a fill crosses zero, split it into a closing leg and an opposite-direction increase. Preserve multiple behavior types as `MIXED`; do not collapse them to a single convenient label.

- `NO_FILL`: no Trade observed in a fully covered interval; this does not mean intentional inactivity.
- `MIXED`: multiple atomic behavior types; retain for audit and exclude from the default single-label training set.
- `POSITION_UNKNOWN`: a fill exists but prior position is unknown.
- `MISSING`: source coverage cannot be established.
- Simultaneous opposing fills with ambiguous ordering are marked ambiguous.
- Partial fills are preserved individually. Failed order-integrity checks, non-OK market state, or unknown position status are recorded as explicit exclusion reasons.

Store `training_eligible` and all exclusion reasons. Report both individual reasons and their combinations.

## Artifacts and required checks

Common derived artifacts include candidate and eligible Parquet tables, order audit, position-anchor audit, instrument-quantity audit, matched public-trade samples, and a dataset-quality report. Store timestamps as UTC-aware values and nullable positions as nullable integers.

Before model use, verify unique decision keys, strict as-of feature and position timestamps, target interval boundaries, order cumulative arithmetic, anchor consistency, source matching, absence of target/PnL leakage from the feature allowlist, and counts of missing/unknown/excluded data.

## Chronological evaluation and runtime

Split by UTC time, never randomly. Purge rows whose label intervals cross a split boundary and purge orders shared across splits. Fix feature allowlists and model settings using training data and validation only; select thresholds on validation. Do not open a final test period until the configuration is frozen.

At inference, validate the exact schema and class order, use the same feature-preparation function as training, and load the model once per process. In simulation, use only later public quotes after decision latency, feed the simulator's own position into subsequent decisions, deduplicate events, persist state transactionally, and reject restart recovery if model/configuration hashes differ. Never connect authenticated streams or order APIs.
