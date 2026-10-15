# Project Instructions

- Use English for identifiers, comments, docstrings, project documentation,
  and development reports.
- Reuse existing structures and functions. Complete the requested scope
  without adding unrelated features or abstractions.
- Read only the source files and documentation relevant to the task.
- Inspect large CSV, Parquet, and log files programmatically. Report
  aggregates and small samples instead of dumping entire files.
- Preserve original account data. Maintain UTC timestamps, strict-as-of
  feature construction, and chronological splits with the required purge.
- Keep feature transformations, column order, data types, and units
  consistent between training and inference.
- Use synthetic data and models only for engineering tests. Never report
  their results as actual model performance or trading results.
- Run relevant checks when changing data leakage controls, PnL accounting,
  duplicate-order prevention, or restart and reconnection recovery.
- Report changes, actual validation results, and remaining issues concisely.
  Clearly distinguish verified results from assumptions.

## Required Claude plugin

- Claude coding sessions for this repository must load `ponytail@ponytail` in
  full mode before implementation. Preserve `.claude/settings.json` in approved
  baseline commits and derived worktrees. Verify activation in the exact working
  directory and inspect plugin/hook initialization; do not rely on an agent's
  self-report. If loading fails, stop before source edits and report the blocker.
- A new worktree also needs its own installed-plugin registration. If absent,
  preserve its local settings and run `~/.local/bin/claude plugin install
  ponytail@ponytail --scope local --json` from that worktree, then verify a fresh
  ACP session. Copied settings or another repository's install record are not
  sufficient proof of loading.
- Use project-scoped activation. Do not enable Ponytail globally or weaken
  authentication, permission, or data-preservation rules to make it load.
