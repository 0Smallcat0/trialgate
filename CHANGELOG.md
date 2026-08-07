# Changelog

## 0.4.0

Everything a caller could quietly get wrong is now either derived by the
library or refused. The API is smaller than 0.3.0's despite doing more.

### Added

- `python -m trialgate demo` (or `trialgate demo`): optimizes four parameters
  over 450 configurations of a pure random walk and puts the winner through
  the gates. The best configuration reaches an in-sample annualized Sharpe of
  **1.25** on data with no edge in it, and is rejected. Its source doubles as
  the integration example — swap the price series and the backtest, keep the
  rest. Always exits 0: the rejection is the demonstration succeeding.
- `evaluate_registry(path, *, candidate_trial_id, periods_per_year,
  effective_trials, ...)` — gates 2-4 in one call, over data that came only
  from the registry. The CSCV matrix, the candidate's return series, and the
  trial Sharpe spread the DSR deflates by are all read from registered trials
  rather than passed in.
- `append_trial(..., holdout=...)` enforces gate 5 at the only place that sees
  every run. A trial reaching into the locked period raises and is not
  recorded.
- Statistical calibration tests: DSR clears 0.95 on ~5% of true nulls (it is a
  probability, so it must), the expected-maximum-Sharpe approximation is
  checked against a direct simulation, and PBO sits near 0.5 on independent
  noise and at 0 on a persistent edge. These check the statistics against what
  they claim to be, not against a transcription of themselves.

### Changed

- `deflated_sharpe_ratio()` takes `trial_sharpes` (annualized, as the registry
  stores them) plus `periods_per_year`, and de-annualizes internally.
- `sharpe_ratio()` requires `periods_per_year`. The old default of 365 silently
  rescaled every equity-market Sharpe by ~20%.
- `append_trial()`'s `config_hash` is optional and defaults to a hash of
  `parameters`. The two were the same data twice in every realistic call.
- `effective_trials` is required with no default on `evaluate_registry()`.
  Correlated grid trials mean the honest count is below the registered N and
  nobody agrees how to find it, so the library refuses to guess. A guess
  dressed as a computation is the failure mode this package exists to prevent.

### Removed

- `non_annualized_sharpe_variance()`. It existed only because the registry
  stored annualized Sharpes and DSR wanted non-annualized ones — a patch over
  an inconsistency, now fixed at the source.

## 0.3.0

Gate 3's input is no longer supplied by the operator, and CSCV is ~1000x
faster. 0.2.0 made the trial count auditable; the CSCV performance matrix was
still whatever columns the caller chose to pass, so the same curation the
registry exists to prevent was available one level down.

### Added

- `append_trial(..., period_returns=[...])` stores a trial's per-period return
  series beside the registry, with its SHA-256 in the chained record.
- `performance_matrix(path, *, strategy_id=None)` assembles the T×N CSCV input
  from the registry, one column per registered trial. Raises if any selected
  trial has no stored returns, if the columns disagree on length or data span,
  or if a stored series no longer matches its recorded hash.

### Changed

- `probability_of_backtest_overfitting()` composes each split's column Sharpes
  from per-block sums instead of recomputing them over raw rows. Same result —
  pinned by an equivalence test against the literal definition — measured at
  2000 rows x 50 configs, `block_count=16`: **14 minutes to 0.78 seconds**.
  500 configs: an estimated 2.3 hours to 7.5 seconds.

### Removed

- `chain_head()` and the `trialgate head` subcommand. `verify` already prints
  the head, and the separate command existed for a workflow nobody runs.
- `load_trials(verify=False)`. A documented way to skip the integrity check is
  the hole this library exists to close.
- `GENESIS_HASH` from the public API, and `ChainReport.anchored_trials` /
  `.unanchored_trials`. Migration bookkeeping, not something a caller branches
  on. `sealed_trials` stays: it is the trust boundary.

## 0.2.0

The trial registry is now tamper-evident. Gate 1 previously relied on the
operator not editing a plain-text log; the honest trial count N — which every
statistical gate downstream depends on — was asserted rather than auditable.

### Added

- Hash-chained registry records. `append_trial()` writes `prev_hash`, the
  SHA-256 over every preceding line of the file as written on disk.
- `verify_chain(path) -> ChainReport` — audit a registry without raising, and
  report how many trials are sealed against edits.
- `chain_head(path) -> str` — the chain value over the whole registry. Publish
  it (a commit is enough) to make tail truncation detectable.
- `trialgate verify <registry>` and `trialgate head <registry>` command-line
  entry points. `verify` exits 1 on a broken chain, for CI and pre-commit use.
- `RegistryTamperError`, `ChainReport`, `GENESIS_HASH`, and the
  `TrialRecord.prev_hash` / `TrialRecord.entry_hash` fields.

### Changed

- `load_trials()` verifies the chain by default and raises
  `RegistryTamperError` when it does not match. Pass `verify=False` to inspect
  a registry you already know is broken; never to compute a verdict from one.
- `append_trial()` refuses to extend a registry whose chain is already broken,
  rather than laundering it.
- `config_hash_for()` accepts any `Mapping[str, object]` instead of requiring
  a `dict`.

### Compatibility

Registries written by 0.1.0 need no migration and keep loading. Their records
carry no `prev_hash` and cannot be verified individually, but they are hashed
into the chain, so the first append under 0.2.0 seals the whole legacy history
behind it. Rewriting old records to add hashes retroactively would itself be
the tampering this release exists to detect.

## 0.1.0

Initial release: append-only trial registry, CSCV/PBO, Deflated Sharpe Ratio,
single-use holdout lock, and the combined threshold gate report.
