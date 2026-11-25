# convergence

`tensor_optix.core.convergence` — the one convergence detector every training loop uses: `LoopController` (RL), the ML `Trainer`, `MultiAgentRunner` (one per seat) and the runtime `DeploymentObserver`.

```python
from tensor_optix import ConvergenceMonitor, ConvergenceRegime

monitor = ConvergenceMonitor()             # defaults below
report = monitor.observe(score, at=episode_id)
report.regime          # what the curve is doing
report.state           # LoopState: ACTIVE / COOLING / DORMANT
```

## What it answers

After every observation the monitor reports two separate things.

**The regime**: what the score curve is doing, judged against the best level the run has reached (its *peak*):

| Regime | Meaning |
|---|---|
| `WARMUP` | Too little history to read trends. |
| `IMPROVING` | New ground: above the peak. |
| `PLATEAU` | Flat, at the peak. |
| `REGRESSED_CLIMBING` | Below the peak and climbing back. Getting back to the peak is **not** improvement. |
| `REGRESSED_STALLED` | Below the peak and not climbing. |
| `COLLAPSING` | Falling away from the peak. |

**The state** (`LoopState`): how long it has been since real progress.

- **ACTIVE** — `IMPROVING` was seen within `cooling_patience`.
- **COOLING** — no new ground for `cooling_patience`.
- **DORMANT** — no new ground for `dormant_patience`.

Patience is measured in the caller's own units of `at` (episodes, windows or epochs), never in evaluation counts.

## How it reads the curve

Three timescales are read together, by default the last 8, 32 and 128 observations. For each it computes:

- the **level**: the mean, with its standard error;
- the **slope**: an OLS fit against the real positions `at`, with its standard error.

Irregular evaluation spacing is therefore handled by the regression itself.

**Noise** is estimated from the robust spread (MAD) of first differences. A change of slope, such as a rise flattening into a plateau, is a step in the mean of those differences. Steps are located by a CUSUM test and removed first, so a bending curve is not mistaken for noise. Lag-1 autocorrelation comes from the same differences; for AR(1) noise, corr(dₜ, dₜ₋₁) = −(1−ρ)/2. Every standard error is then inflated by √((1+ρ)/(1−ρ)).

**The peak** is the best medium-scale level, minus one standard error. A lucky window therefore cannot set an unbeatable bar. During an established rise (a significant medium slope plus a sustained `IMPROVING` streak), the medium mean lags the true level. In that case the peak follows the top of the rise. That fast-tracked top is provisional: after a full medium window of plateau it settles to the medium-scale estimate. A run that collapses first keeps the fast-tracked top, because that is the level it has to get back to.

**The rules**:

- **`IMPROVING`**, by any one of three paths:
  - *Fast:* the short level clears the peak by 2 SE, and the medium level is above the peak.
  - *Slow:* the medium level clears the peak, and the medium or long slope is significantly rising. The long window resolves slopes too gentle for the short one.
  - *Continuation:* inside an ongoing `IMPROVING` streak, the medium slope is still significant and the short level has not fallen below the peak.
- **`COLLAPSING`**: the medium slope is significantly falling and the short level is far below the peak.
- **`REGRESSED_*`**: both the medium and the short level are below the peak by more than their combined uncertainty. Climbing or stalled is decided by the short or medium slope.
- **Persistence**: a new regime must hold for `persist` consecutive observations before it is reported. This absorbs clusters of false alarms from overlapping windows.

**Warm-up** has separate mechanics. For the first `medium` observations, and for the first few after a restart or restore:

- **Improvement**: the mean of the last 4 scores beats a reference level by a significant margin. The reference is the level at the previous improvement during the first warm-up, and the peak after a restart or restore.
- **Collapse**: that mean and the newest score both fall below the peak by twice the margin.
- **Patience** doesn't run until a noise estimate exists (5 observations).

**DORMANT** is withheld in three cases:

- while the long-scale slope is still significantly rising (a slow learner is not converged);
- while a collapse is being confirmed or is in progress;
- right after a rollback, while the restored policy's scores can't be read yet.

## Constructor

```python
ConvergenceMonitor(
    timescales=(8, 32, 128),      # (short, medium, long) observations; first `medium` = warm-up
    cooling_patience=32,          # units of `at` without IMPROVING -> COOLING
    dormant_patience=160,         # units of `at` without IMPROVING -> DORMANT
    z=2.0,                        # significance (standard errors) of every test
    persist=3,                    # observations a new regime must hold
    min_delta=0.0,                # smallest absolute gain over the peak that counts
    min_at_before_dormant=0,      # DORMANT never before this position
    autocorrelation=True,         # correct SEs for AR(1) noise
)
```

## Methods

| Method | Use |
|---|---|
| `observe(score, at=None, probing=False)` | Record one score and return a `ConvergenceReport`. `at` must not decrease; it defaults to the observation count. With `probing=True` (a deliberate perturbation such as an SPSA probe) a collapse is never confirmed. |
| `restart(at=None)` | A new policy variant (spawn, topology change). The peak is kept, so the variant must still beat it. Trend history and patience start afresh. |
| `note_restored(at=None)` | The peak weights were restored (rollback). Trend history is cleared. The peak and the patience clock are kept, because restoring is not progress. |
| `reset_scale()` | The score scale changed. Forget everything, including the peak. |
| `defer_dormant()` | The owning loop refuses DORMANT for now. Step back to COOLING so DORMANT can be entered again later. |

Read-only views: `state`, `regime`, `peak`, `best_score`, `last_report`, `n_observations`.

## ConvergenceReport

| Field | Meaning |
|---|---|
| `regime`, `state` | as above |
| `scales` | `{"short" / "medium" / "long": ScaleReading(n, level, level_se, slope, slope_se)}`, for the windows that are full |
| `sigma`, `rho` | per-observation noise and its lag-1 autocorrelation |
| `peak`, `peak_at`, `best_score` | bias-corrected peak level, where it was last raised, highest raw score |
| `gap_to_peak` | (peak − short level) / sigma; positive means below the peak |
| `since_improvement` | units of `at` since the last `IMPROVING` |
| `below_peak_for` | units of `at` spent in `REGRESSED_*` / `COLLAPSING` |
| `eta_to_peak` | units of `at` to climb back at the current slope (`inf` unless `REGRESSED_CLIMBING`) |
| `improved`, `peak_raised` | this observation was `IMPROVING` / raised the peak |
| `regime_changed`, `entered_cooling`, `entered_dormant`, `collapse_confirmed` | transitions on this observation, so loops act **once, on entry** |
| `is_regressed` | property: regime is one of the two `REGRESSED_*` |

## What each loop does with it

| Transition | RL `LoopController` | ML `Trainer` | `MultiAgentRunner` (per seat) |
|---|---|---|---|
| `improved` | `optimizer.on_improvement`, RND η × 0.9 | `spsa.on_improvement` | `optimizer.on_improvement` |
| `entered_cooling` | `optimizer.on_cooling`, RND η × 1.5, `on_plateau` callback | `spsa.on_cooling`, `on_plateau` callback | `optimizer.on_cooling`, `on_plateau` |
| `collapse_confirmed` | `on_degradation`, plus rollback when enabled (on-policy only) | — | `on_degradation`, plus rollback when enabled (on-policy only) |
| `entered_dormant` | `optimizer.on_plateau`, `on_dormant`, stop unless a callback resumes | `on_dormant`, stop (`stop_reason="converged"`) | `optimizer.on_plateau`, `on_converged` |
| any regime or state change | `on_convergence(episode_id, report)` | `on_convergence(epoch, report)` | `on_convergence(agent_id, window_id, report)` |

## Behaviour under simulation

Noise σ = 5 around a level of 100, 20 seeds:

| Scenario | Result (median) |
|---|---|
| Flat plateau, false `IMPROVING` per 2,800 episodes | 3 |
| Flat plateau, episodes to DORMANT | 213 |
| Dip −30, then recovery to the same level, counted as `IMPROVING` | 0 |
| Dip −30, delay until reported as regressed / collapsing | 3 |
| Drop −15 that stays, delay to `REGRESSED_STALLED` | 31 |
| Collapse −30, delay to confirmed | 4 |
| Learning resumes at +0.5 / +0.2 / +0.05 per episode, delay to `IMPROVING` | 19 / 28 / 78 |
| Steady slow learner (+0.05 per episode), ever DORMANT | never |

Cost is about 0.2 ms per observation.
