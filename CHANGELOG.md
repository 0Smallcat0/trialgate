# Changelog

## 0.6.0

The release that covers the whole way from an optimizer run to money at risk.
Up to 0.5 the gates judged whatever trials someone remembered to register, and
the step after the backtest was a sentence ("paper trade for three months").
Now the trial count comes from the optimizer itself: one command registers
every epoch of a freqtrade hyperopt, and an Optuna callback registers every
trial as the study runs. Gate 6 is a computation: bands locked before the
first live day, with a STOP that exits 1. A new gate 9 asks whether the edge
survives paying for execution twice, since one basis point is enough to pass
gate 8. The statistics underneath were audited against their papers, and five
ways they could pass a strategy they should have rejected are fixed. The DSR
now reports when serially correlated returns make it too generous. The data
floor counts the days a strategy actually had a return, not the days it sat
in cash. On a real 20-epoch freqtrade run, the epoch freqtrade picked had won
5 trades out of 5 and reported a Sharpe of 39.7. It is now rejected by four
gates, each with its reason.

### Added

- **Gate 9, cost stress**, on by default in `evaluate_registry()`. The
  candidate's registered per-period costs are charged again (`cost_multiplier`,
  2× in total by default) and the DSR of what is left must clear `min_dsr`,
  deflated by the same trial spread as gate 4. The detail gives the multiple of
  those costs at which the mean return reaches zero. Backtests charge the fee
  their author typed and leave out spread and slippage (freqtrade's backtester
  fills every order at the requested price), and gate 8 is satisfied by any
  cost above zero. In the tests, a candidate netting 0.1% a day after 0.1% of
  costs passes gate 4 at a DSR of 0.98 and fails gate 9 at 0.12.
  - `append_trial(..., period_costs=[...])` stores what execution cost in each
    period, as a fraction of equity and already deducted from
    `period_returns`, hashed into the chained record like the returns.
    `load_period_costs()` reads it back and raises on an edited file. Costs
    must be finite, non-negative and line up with the returns.
  - `import-freqtrade` records each trade's fee: every fill's amount × price ×
    that side's fee, booked on the day the trade closed. On the real fixture,
    each trade's `profit_abs` plus that fee equals its price move to within
    5e-9, for all 37 trades. The Optuna callback reads
    `user_attrs["period_costs"]`.
  - `trialgate evaluate --cost-multiplier 3` or `--no-cost-stress`.
- **Trial capture from the optimizer** (`trialgate.capture`). The trial count
  no longer depends on remembering to register:
  - `import_freqtrade_results()` / `trialgate import-freqtrade` registers every
    epoch of a freqtrade hyperopt (`.fthypt`, a file or the results directory)
    with its parameters, fees and daily returns. Every calendar day of the
    backtest is a period; the returns compound back to freqtrade's
    `profit_total`. Re-importing registers only new epochs, and a holdout
    breach refuses the whole import before anything is written. Tested on a
    real freqtrade results file.
  - `OptunaRecorder`, an Optuna callback (no Optuna import), registers complete
    and pruned trials as the study runs; `import_optuna_study()` backfills.
- **Gate 6 is now a computation** (`trialgate.live`), replacing the procedural
  "≥ 3 months of paper trading". `lock_live_bands()` block-bootstraps the
  candidate's registered returns into bands for cumulative return, drawdown
  from peak and the longest losing run, calibrated jointly to one false-alarm
  rate over the whole horizon and measured on independent resamples. The lock
  cannot be re-chosen. `check_live()` returns STOP, PENDING or PASS and, given
  the registry, recomputes the bands to detect an edited lock. Each lock
  reports `power_if_no_edge`, how often the same returns with no edge would be
  stopped: about one time in ten over 90 days for a Sharpe-1 daily strategy.
- **Command line**: `import-freqtrade`, `evaluate` (exits 1 when not
  qualified), `live-lock` and `live-check` (exits 1 on STOP), so the path
  from hyperopt to a verdict to live monitoring needs no Python.
- **Serial correlation in the DSR and in annualization.** The PSR/DSR
  standard error and √p annualization assume independent periods, so
  positively autocorrelated returns overstated both. `deflated_sharpe_ratio()`
  now estimates the Sharpe estimator's long-run variance (Lo 2002, GMM case,
  AR(1)-prewhitened Bartlett HAC) and reports the DSR on the equivalent number
  of independent observations, with Ljung-Box diagnostics
  (`AutocorrelationDiagnostics`). `autocorrelation="report"` (default) keeps
  the IID value bit for bit; `"adjust"` judges the adjusted value; `"ignore"`
  skips the estimate. `evaluate_registry()` takes the same argument and, by
  default, attaches a warning to gate 4 when the correlation is significant.
  `observations_required_for_dsr()` scales its answer by the variance ratio
  under `"adjust"`. The adjustment is trialgate's extension: Bailey & López
  de Prado's DSR and PSR do not correct for serial correlation (the PSR
  paper, p. 4, leaves it unaddressed).
- **`lo_annualized_sharpe()`**: annual Sharpe by Lo's η(q) next to the √q
  figure. Uses the sample autocorrelations up to the Andrews (1991) bandwidth
  by default, or all q − 1 of Lo's formula with `max_lag=periods_per_year - 1`.
- The demo's gate 4 line now carries that warning: the best of 450
  configurations has lag-1 autocorrelation +0.35, an adjusted DSR of 0.574
  (0.603 unadjusted) and an η(q) annual Sharpe of 0.67 against 1.20 by √252.
  The verdict is unchanged.

- Citations in the docstrings and README now give equation and page numbers
  for Lo (2002) and the PSR/DSR papers.

### Changed

- **Breaking: a candidate registered without `period_costs` fails gate 9.**
  Its edge after a higher cost is unknown, and unknown is not cheap. Registries
  filled by `import-freqtrade` from this release on carry the costs. For older
  registries, re-import or re-register with `period_costs=`, or pass
  `cost_multiplier=None` (`--no-cost-stress`) and know what the verdict leaves
  out. The demo registers its costs; its best configuration fails gate 9 too
  (DSR 0.508 at 2×) and its mean return would reach zero only at 19.8× its
  costs. Here the edge was never there, so the costs are not the reason.
- **Gate 2 counts periods with a return, not the length of the series.** A
  period that returned exactly zero (flat, in cash) is not evidence, so a
  strategy in the market on 39 of 2000 days now has 39 observations. This
  changes verdicts: the demo's best configuration now fails gate 2 as well as
  gate 4.

### Fixed

- **PBO scored a constant-return column at a Sharpe of ~2e7.** The fast CSCV
  path computed variance as sum-of-squares minus squared-sum, which leaves a
  rounding residue (~1e-20) on a constant column instead of zero. A cash-like
  trial earning 0.169% every period then won every in-sample split and topped
  every out-of-sample rank: PBO came out **0.0** on a matrix whose literal
  definition gives **0.4**. Each column is now shifted by its first value
  before accumulating, which leaves the variance unchanged and the constant
  column at exactly zero.
- **NaN and infinity are refused** by `probability_of_backtest_overfitting()`
  and `deflated_sharpe_ratio()` (returns and `trial_sharpes`). A NaN from a
  backtest's first bar previously decided CSCV's argmax and rank silently, and
  crashed the DSR with an `AttributeError` from inside `statistics`.
- **`deflated_sharpe_ratio()` refuses `effective_trials > 1` with fewer than
  two `trial_sharpes`.** One Sharpe has no spread, so the expected maximum was
  set to 0 and the "deflated" ratio deflated by nothing.
- **`probability_of_backtest_overfitting()` refuses splits of one row**
  (`block_count=2` with fewer than 4 rows). Every Sharpe there was scored 0.0
  and PBO came out a passing 0.0.
- **`observations_required_for_dsr()` refuses `target_dsr` outside (0.5, 1).**
  The inversion squares the normal quantile, so a target of 0.05 reported the
  sample size for 0.95, and 1.0 failed inside `statistics`.

- **A column constant over a split but not equal to its first row** (an entry
  cost on bar one, then a flat yield) still left a rounding residue after the
  shift above and scored a Sharpe in the millions: PBO 0.0 where the literal
  definition gives 0.37. A split's spread inside the summation error bound is
  now treated as zero.
- **CSCV ranks out-of-sample ties against the chosen trial.** Ties were ranked
  in its favour, so duplicated columns (a grid parameter that changes nothing)
  lifted it past the median: on pure noise with 7 distinct trials, doubling
  each dropped PBO from about 0.6 to about 0.45. PBO now equals that of the
  distinct trials. The demo's 450-configuration grid has such ties; its PBO
  moves from 0.0417249 to 0.0418803.

### Docs

- The README's "What it will not do" now names two checks left out on
  purpose. Look-ahead is a bug for the engine to find (freqtrade ships
  `lookahead-analysis`); gate 6 catches what it does live. A cap on how much
  of the profit came from a few days measures volatility, not luck. On
  simulated strategies with a real edge (annual Sharpe 1.0, 1,000 days), the
  best five days are a median 28% of the profit, and removing the best 21
  erases it.

### Tests

- Gate 9: pass and fail around the same edge, the breakeven multiple against
  its arithmetic, missing and zero costs, the skip, monotonicity in the
  multiplier, refused multipliers, cost storage and tampering, and the
  freqtrade fee identity on every trade of the real fixture.
- Autocorrelation: the default DSR pinned bit for bit to the previous release
  on IID data; the adjusted Sharpe standard error against Monte Carlo and the
  AR(1) asymptotic variance for φ from −0.5 to 0.8; DSR calibration on
  autocorrelated nulls; η(q) against Lo's AR(1) closed form and simulated
  q-period sums; Ljung-Box and the HAC against numpy/scipy reimplementations.
  numpy and scipy are test-only `dev` extras.
- Cross-checks against Lo (2002): every cell of Table 2 (η(q) under AR(1),
  p. 41) to the printed two decimals, and the Sharpe columns of Table 4
  (p. 44), all 22 funds within the printed rounding. Table 4's SR(12) column
  needs 11 autocorrelations per fund and the table prints 3, so it is
  recorded as out of reach rather than fitted.
- Cross-checks against the worked example in Bailey & López de Prado (2014):
  SR_0 = 0.1132 and DSR = 0.9004 are reproduced; DSR against a reference
  written from raw sums; PBO against the paper's logit definition, and its
  invariance to column order and positive rescaling.

## 0.5.1

### Fixed

- **Gate 8 passed a cost that was declared and then marked as not charged.**
  `cost_assumptions={"round_trip_pct": "0.585", "deducted": "false"}` cleared
  the gate, because the only test was "some value parses above zero". That
  registry says in its own words that the returns are gross of cost, which is
  the case the gate exists for. A `charged`, `deducted` or `net_of_costs` key
  set to a false word (`false`, `no`, `0`, empty …) now fails the trial, and
  those flags no longer count as the positive price themselves (`{"charged":
  "1"}` alone declares no cost). Found against a real registry whose replay
  trials scored excess returns gross and said so.
- Gate 8's failure detail now reads "priced execution at zero or left it out
  of their returns", since there are two ways to get there.

### Docs

- The README, the gate docstring and the project links name the crypto signal
  system this library was extracted from as its host system, without a
  qualification status that goes stale.

## 0.5.0

Two gates for the case a backtest clears every statistic and is still
untradeable: the position size trips the operator's own stop before the edge
pays, or the returns were never charged for execution.

### Added

- **Gate 7, stop-line survivability** (`trialgate.risk`). Given a stop-loss
  policy, reports the probability that the strategy's own volatility breaches
  it within the horizon, and the volatility to size down to. Both moments come
  off the candidate's registered return series; the only operator input is the
  policy, which no library can derive without inventing someone's risk
  appetite. Activate it with `evaluate_registry(..., stop_loss_fraction=0.15)`
  — omitting it skips the gate rather than defaulting a stop line.

  The motivating number: a *genuine* annualized Sharpe of 1.0 run at 15%
  volatility breaches a −15% stop inside a year 8.1% of the time, and at
  Sharpe 0.5 that becomes 16.7%. Nothing is wrong with the signal in the
  second case, but the shutdown looks exactly like the signal being wrong.

  Computed from the exact first-passage identity for Brownian motion with
  drift, in log equity, measuring loss from starting equity rather than
  drawdown from a running peak. Those two differ by roughly 3× (23% against
  8.1% in the case above). The from-peak variant is deliberately absent: the
  cheap route through Lehoczky's expected first-passage time fed into a
  Poisson exceedance approximation was measured against simulation and
  overstated by 3× in the low-probability region a gate spends all its time
  in, so it was rejected rather than shipped with a caveat.

- **Gate 8, declared costs**, on by default in `evaluate_registry()`. Every
  trial in the registry must price execution above zero. A missing key and an
  explicit `"0"` get the same verdict, because only one of them looks like an
  oversight. Waive it with `require_declared_costs=False`.

- `observations_required_for_dsr()` returns a `DataRequirement`: how many
  observations the candidate would need to clear the DSR gate, and the
  shortfall against what it has. "Not enough data" is a true statement that
  ends a conversation; this turns it into a quantity someone can go and
  collect. When the candidate's Sharpe already sits below the expected maximum
  of N noise trials, `reachable` is False — no sample size fixes having been
  beaten by chance, and reporting a number there would be worse than
  reporting none.

- `GateCheck.detail`: the instruction a bare threshold cannot carry. Gate 7
  fails with a volatility to size to, gate 8 with the trials that were never
  charged, and the data floor with how many observations are missing. Defaults
  to empty, so pre-0.5.0 construction is unchanged.

### Changed

- **Breaking:** registries written before 0.5.0 that recorded
  `cost_assumptions={}` now fail gate 8. That is the intended verdict on a
  gross-of-cost backtest, not an upgrade accident. Re-run those trials with
  costs charged, or pass `require_declared_costs=False` and know what the
  number you are looking at excludes.
- The demo charges a 10 bps round trip on every position change and declares
  the same number to the registry, because a demo that priced execution at
  zero would be demonstrating the wrong thing. The best configuration's
  in-sample Sharpe drops from 1.25 to **1.20** and it is still rejected — but
  by DSR alone now (0.60), with PBO passing at 0.042. One gate clearing is not
  a strategy clearing, which is the reason there is more than one.

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
