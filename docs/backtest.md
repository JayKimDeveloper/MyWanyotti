# Chronological Backtesting

## Status and evaluation boundary

The repository contains a backtesting implementation. A run is meaningful only when the input dataset, model provenance, instrument specification, and execution assumptions pass preflight. Synthetic or unverified models are suitable only for engineering smoke checks; their output must not be reported as trading performance.

By default, the runner evaluates only saved held-out test keys produced by training. It verifies dataset hashes and split metadata, recomputes market features, and checks that every feature timestamp precedes its decision timestamp. Optional start/end arguments may narrow, but must not expand, the frozen test interval.

Keep two outputs separate:

1. **Behavior classification:** predictions versus held-out labels. This does not establish intent or profitability.
2. **Simulated portfolio path:** the simulator's own position feeds subsequent decisions; fills are modeled from later public quotes and depend on execution assumptions.

`NO_FILL` means no observed fill in a label interval, not intentional inactivity. Invalid reductions are recorded without an order, and a single simulated order may not reverse a position.

## Execution assumptions

`configs/backtest.example.toml` externalizes contract type, quantity units, lot/tick sizes, settlement currency, contract value, initial capital and position, decision latency, fill delay, participation, partial fills, random seed, fees, funding, slippage, and liquidation handling.

A simulated order uses the first eligible public quote after configured latency and before the label interval ends. Buys reference the ask and sells the bid. Available size and lot constraints cap fills; adverse slippage is applied and rounded to the instrument tick. Historical account fills and future label prices are not execution prices.

Example values and current instrument snapshots are not verified historical specifications. Historical fee tiers, funding rates, queue position, market impact, and liquidation behavior remain uncertain. If liquidation is disabled, leveraged-strategy losses may be understated.

## Accounting convention

The inverse-contract example uses settlement-currency calculations:

```text
position_pnl = signed_contracts * contract_value_quote * (1 / entry_price - 1 / exit_price)
fee          = abs(filled_contracts) * contract_value_quote / fill_price * fee_rate
funding_flow = -signed_notional_settlement * funding_rate
```

Funding is applied at its configured timestamp to the position then open. Under this convention, a positive rate is paid by longs and received by shorts. Confirm point-in-time instrument rules before interpreting results historically. General exchange documentation does not validate example values for every past date.

## Outputs and interpretation

The runner records provenance, configuration, feature reconciliation, decisions, orders, simulated fills, funding flows, equity events, and cost sensitivity. Return is `final_equity / initial_equity - 1`; drawdown is measured from the running peak across recorded equity events. Exposure is time-weighted over the simulated interval.

Do not conflate a behavior-classification score with a portfolio return, or describe an assumption-dependent simulation as verified trader performance.
