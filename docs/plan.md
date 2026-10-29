# Research Plan

## Completed engineering areas

- Read-only source CSV inspection and schema profiling.
- Separation of trade, funding, and settlement events.
- Order partial-fill arithmetic checks and evidence-labeled position handling.
- Same-exchange public market feature construction with strict as-of checks.
- Candidate/eligible dataset generation and audit reports.
- Chronological training preflight, inference, backtest, and offline simulation scaffolds.
- Configuration-driven contract, cost, risk, and restart behavior.

These engineering components do not establish that a historical trading edge exists. Account-derived data and generated artifacts are not part of the public source release.

## Required before a valid strategy evaluation

1. Establish complete, reproducible market and funding coverage for the intended evaluation interval.
2. Obtain point-in-time instrument specifications, fee schedules, and defensible execution-cost assumptions.
3. Independently validate source timestamps, coverage, labels, and any conditional position anchors.
4. Create a frozen chronological development/validation protocol and protect a future holdout.
5. Require a meaningful predictive result before testing a model-driven portfolio policy.
6. Compare with non-ML baselines, include costs, and report uncertainty, drawdown, and regime sensitivity.

## Release and operation boundaries

- Do not include raw account exports, generated datasets, model binaries, audit ledgers, or runtime databases in source control.
- Do not infer intent, leverage, or complete position histories from fills alone.
- Do not enable a strategy or persistent service based solely on behavior-classification scores.
- Any future public-data paper simulation remains offline and order-free unless separately reviewed and authorized.
