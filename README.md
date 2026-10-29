# Wonyotti Research Toolkit

An offline research codebase for time-series modeling, market-data handling, and simulated execution. It is not an exchange-connected trading bot: no authenticated account API or order-submission endpoint is implemented.

## Scope and safety

- Raw exports, market archives, generated datasets, model binaries, audit outputs, and runtime databases are excluded from this source release.
- The repository contains code and documentation only. Supply any required data locally under the documented contracts.
- Historical behavior classification does not establish trading intent, a verified strategy reconstruction, or profitability.
- Backtests and paper simulations depend on explicit execution, contract, and cost assumptions. They are research tools, not performance claims.
- No exchange credentials are required. The optional paper runner uses public market streams and simulates fills locally.

## Project status

The codebase includes CSV inspection, market-data ingestion, feature construction, chronological model evaluation, a contract-aware backtest scaffold, and an offline paper simulator. Existing research has not established a robust net trading edge. Keep any final holdout untouched until a candidate and its evaluation protocol are frozen.

## Setup

Use Python 3.13 and install pinned dependencies in an isolated environment:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
```

Inspect the available CSV-inspection options:

```bash
.venv/bin/python scripts/inspect_csv.py --help
```

Commands that require source datasets or model artifacts will not run until those inputs are supplied locally. Do not commit source exports, model files, databases, credentials, or generated research outputs.

## Technical notes

- [Data contract](docs/data_contract.md)
- [Model training and evaluation](docs/model_training.md)
- [Backtesting assumptions](docs/backtest.md)
- [Offline paper simulation](docs/paper_trading.md)
- [Research plan](docs/plan.md)

## Reproducibility

Record data provenance and hashes, preserve chronological boundaries, and review each data-quality report before interpreting results. Never treat an assumed timezone, funding-derived quantity, or simulated fill as verified account state.
