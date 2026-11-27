# convergence

`tensor_optix.core.convergence` is the one convergence detector every training loop uses: `LoopController` (RL), the ML `Trainer`, `MultiAgentRunner` (one per seat) and the runtime `DeploymentObserver`.

```python
from tensor_optix import ConvergenceMonitor, ConvergenceRegime

monitor = ConvergenceMonitor()             # defaults below
report = monitor.observe(score, at=episode_id)
report.regime          # what the curve is doing
report.state           # LoopState: ACTIVE / COOLING / DORMANT
```

## What it answers

After every observation the monitor reports two separate things.

**The regime**: what the score curve is doing, judged against the best level the run has reached (its *peak*).

| Regime | Meaning |
|---|---|
| `WARMUP` | Too little history to read trends. |
| `IMPROVING` | New ground: above the peak. |
| `PLATEAU` | Flat, at the peak. |
| `REGRESSED_CLIMBING` | Below the peak and climbing back. Getting back to the peak is **not** improvement. |
| `REGRESSED_STALLED` | Below the peak and not climbing. |
| `COLLAPSING` | Falling away from the peak. |

**The state** (`LoopState`): how long it has been since real progress.

| State | Meaning |
|---|---|
| **ACTIVE** | `IMPROVING` within `cooling_patience` |
| **COOLING** | no new ground for `cooling_patience` |
| **DORMANT** | no new ground for `dormant_patience` |

Patience is measured in the caller's own units of `at` (episodes, scored windows or epochs), never in evaluation counts.

## How it reads the curve

**Three timescales.** By default the last 8, 32 and 128 observations are read together. For each window the monitor computes:

- the **level**: its mean, with the standard error of that mean;
- the **slope**: an OLS fit against the real positions `at`, with its standard error. Irregular evaluation spacing is therefore handled by the regression itself.

**Outliers.** Levels and slopes are computed on scores clipped to (running median of 5) ± `outlier_k` noise standard deviations, a Hampel filter.

- The noise estimate is a robust scale, so the means it is compared with must be robust too. Otherwise one spike, or a heavy-tailed noise draw, moves a short mean by several of its own standard errors.
- The two newest points have no future neighbours. Their reference is the trailing median advanced by the recent step, so a steady trend is not clipped.
- `best_score` always sees the raw values.

**Noise.**

- σ is estimated from the robust spread (MAD) of first differences.
- A change of slope (a rise flattening into a plateau) is a step in the mean of those differences. Steps are located by a CUSUM test with binary segmentation and removed first, so a bending curve is not mistaken for noise.
- A relative floor of 10⁻⁹ × the median |score| keeps a deterministic curve from making every 10⁻¹⁵ change "significant".

**Autocorrelation.**

- Lag-1 autocorrelation ρ comes from the same differences: for AR(1) noise, corr(dₜ, dₜ₋₁) = −(1−ρ)/2.
- One window's estimate is too noisy to set every standard error from. It is therefore pooled over the run with an EWMA (weight max(1/128, 1/count)), and only from windows with at least 32 differences.
- Before the pool exists, a single-window estimate is used from 12 differences on. Because it is clipped at 0, it is biased upward, which is the conservative side.
- With mostly identical differences (flat or piecewise-constant scores), no autocorrelation is estimated: there the formula would read ρ ≈ 1.
- Every standard error is inflated by √((1+ρ)/(1−ρ)).

**After `restart()` or `note_restored()`.** The previous segment's σ and ρ are reused until the new segment holds `medium` observations. Without this, an autocorrelated run is judged with standard errors about 3× too small right after a rollback, and "collapses" again.

**The peak**: the best medium-scale level minus one standard error.

- **During an established rise** (a significant medium slope and an `IMPROVING` streak of at least `short` observations), the medium mean lags the current level by half a window. The peak then follows the top of the rise: the lower of (short level − SE) and (the medium fit at the newest point − SE).
- **That top is provisional.** After a full medium window of plateau it settles to the medium-scale estimate. A run that regresses or collapses first keeps the fast-tracked top, since that is the level it has to get back to.
- **End of the first warm-up.** The peak becomes the larger of the medium-scale estimate and the warm-up peak deflated by two of its own standard errors. A run that already fell during warm-up does not forget how high it had been.
- **Residual bias.** A rare false `IMPROVING` on a long plateau raises the peak a little each time: measured at about +0.3σ after 10,000 observations of white noise. The margins of the "below the peak" test absorb this; it acts as hysteresis.

**Minimum gain**: `max(min_delta, rel_delta × max(|peak|, peak − lowest level seen))`. It is added to every "above the peak" and "below the peak" test. The span term keeps a relative threshold meaningful for a metric heading to zero, such as a loss.

**Regime rules (training phase).** "Clears by `z` SE" below means by the larger of `z` standard errors and the minimum gain.

| Regime | Condition |
|---|---|
| `IMPROVING`, fast | the short level clears the peak by `z` SE, the medium level is above the peak by 1 SE, and the medium slope is positive |
| `IMPROVING`, slow | the medium level clears the peak by `z` SE, and the medium or long slope is significantly rising (the long window resolves slopes too gentle for the short one) |
| `IMPROVING`, continuation | inside an ongoing `IMPROVING` streak, the medium slope is still significant and the short level is at or above the peak plus the minimum gain |
| `COLLAPSING` | the medium slope is significantly falling, and the short level is more than 2 × `z` SE below the peak |
| `REGRESSED_*` | both the medium and the short level are below the peak by more than their combined uncertainty; climbing or stalled is decided by the short or medium slope |

- **Persistence.** A new regime must hold for `persist` consecutive observations before it is reported.
- **Probing.** A collapse is never confirmed on an observation scored under a deliberate perturbation (`probing=True`, e.g. an SPSA probe or a self-play window against all-live opponents). Such an observation neither counts toward a pending collapse nor cancels it. The ML `Trainer` goes further: it does not feed probe epochs to the monitor at all (see the [ML core guide](../../guides/ml-core.md#hyperparameter-tuning)).

**Warm-up and the first readings after a restart or restore** use separate rules. The mean of the last 4 scores is compared with a reference:

- **`IMPROVING`** when it beats the reference by a significant margin. In the run's first warm-up the reference is the level at the previous improvement; after a restart or restore it is the run's peak, because climbing back to the peak is not new ground.
  - In the first warm-up the noise comes from very few differences, so the normal `z` is replaced by the Student-t quantile at the same level, with df = 0.37·(n − 2).
- **`COLLAPSING`** when that mean and the newest score both fall below the peak by twice the margin. In the first warm-up this needs at least 8 points.
- **No patience before noise.** The patience clock does not run until a noise estimate exists (5 observations).

**DORMANT.**

- **Never entered while the regime is `WARMUP`.** Patience counts in `at` units, so with sparse evaluation it can run out before trends are readable.
- **Sticky.** Once entered, only new ground (`IMPROVING`) or a confirmed `COLLAPSING` ends it, never a chance fluctuation of a trend test.
- **Withheld in these cases:**
  - While the longest readable trend is still significantly rising *and* would add more than the minimum gain over another `dormant_patience`.
  - While a collapse is being confirmed, or right after a rollback until the restored policy's scores can be read.
  - While `hold_dormant(True)` is in effect, or before `min_at_before_dormant`.
- **A run that collapsed and stays down** turns into `REGRESSED_STALLED` once the medium window has moved past the fall. Patience keeps running from the last real improvement, so such a run *is* eventually declared DORMANT. Loops restore the best weights at that point.
- **Slow learners.** "A slow learner is not converged" holds only above a detection floor. The 128-observation long window resolves slopes down to about 2σ·√(12/128³) ≈ 0.005σ per observation, roughly 0.8σ of gain per default `dormant_patience`. A learner slower than that is declared DORMANT with that much gain still to come.
  - To push the floor down, raise the long timescale. For example, `(8, 32, 256)` reduces the gain left at DORMANT by about a third, at the cost of about 40 more observations before DORMANT on a plateau.

**Known limits:**

- **Oscillating scores.** A policy that oscillates on the medium timescale (for example a period of about 50 observations) is reported as repeatedly `COLLAPSING`. Each fall is a real, significant fall.
- **Noise jumps.** σ lags a sudden jump in noise by up to one long window.
- **Discrete metrics near their ceiling.** For example, accuracy at 0.97 on 50 samples: σ is over-estimated by about 1.5×, which makes detection slower.

## Constructor

```python
ConvergenceMonitor(
    timescales=(8, 32, 128),      # (short, medium, long) observations; 3 <= short <= medium <= long, long >= 8
    cooling_patience=32,          # units of `at` without IMPROVING -> COOLING
    dormant_patience=160,         # units of `at` without IMPROVING -> DORMANT
    z=2.0,                        # significance (standard errors) of every test
    persist=3,                    # observations a new regime must hold
    min_delta=0.0,                # smallest absolute gain over the peak that counts
    min_at_before_dormant=0,      # DORMANT never before this position
    autocorrelation=True,         # correct SEs for AR(1) noise
    rel_delta=0.0,                # relative minimum gain (see above)
    outlier_k=3.0,                # Hampel clipping width in noise SDs; 0 disables
)
```

## Methods

| Method | Use |
|---|---|
| `observe(score, at=None, probing=False)` | Record one score and return a `ConvergenceReport`. `at` must not decrease within a segment; it defaults to the observation count. |
| `restart(at=None)` | A new policy variant (spawn, topology change, a repeated `run()`). The peak is kept, so the variant must still beat it. Trend history and patience start afresh; σ and ρ are carried over. |
| `note_restored(at=None)` | The peak weights were restored (rollback). Trend history is cleared. The peak and the patience clock are kept, because restoring is not progress. |
| `reset_scale()` | The score scale changed. Forget everything, including the peak. |
| `hold_dormant(hold)` | While held, the state stops at COOLING: DORMANT is neither entered nor reported. Releasing lets DORMANT be entered on the next observation. |
| `defer_dormant()` | One-off: step back from DORMANT to COOLING. |

Read-only views: `state`, `regime`, `peak`, `best_score`, `last_report`, `n_observations`.

## ConvergenceReport

| Field | Meaning |
|---|---|
| `regime`, `state` | as above |
| `scales` | `{"short" / "medium" / "long": ScaleReading(n, level, level_se, slope, slope_se)}` for the windows that are full |
| `sigma`, `rho` | per-observation noise (`None` before an estimate exists) and lag-1 autocorrelation |
| `peak`, `peak_at`, `best_score` | the peak level (see above), where it was last set, the highest raw score |
| `gap_to_peak` | (peak − short level) / sigma; positive means below the peak |
| `since_improvement` | units of `at` since the last `IMPROVING` |
| `below_peak_for` | units of `at` spent in `REGRESSED_*` / `COLLAPSING` |
| `eta_to_peak` | `REGRESSED_CLIMBING`: units of `at` to climb back at the current medium slope. `inf` when below the peak but not climbing. `None` when not below the peak. |
| `improved`, `peak_raised` | this observation was `IMPROVING` / raised the peak |
| `regime_changed`, `entered_cooling`, `entered_dormant`, `collapse_confirmed` | transitions on this observation, so loops act **once, on entry** |
| `is_regressed` | property: regime is one of the two `REGRESSED_*` |

## What each loop does with it

| Transition | RL `LoopController` | ML `Trainer` | `MultiAgentRunner` (per seat) |
|---|---|---|---|
| `improved` | `optimizer.on_improvement`, RND η × 0.9 | `tuner.on_improvement` | `optimizer.on_improvement` |
| `entered_cooling` | `optimizer.on_cooling`, RND η × 1.5, `on_plateau` | `on_plateau` only (the tuner is not widened) | `optimizer.on_cooling`, `on_plateau` |
| probe observations | `probing=True` (no collapse confirmation) | not fed to the monitor (`FitHistory.probe_epochs`) | `probing=True` |
| `collapse_confirmed` | `on_degradation`; rollback when enabled, on-policy only, and not again until the run has been back at the peak | — | `on_degradation`; rollback under the same rule |
| `entered_dormant` | `optimizer.on_plateau`, `on_dormant`; stop unless a callback resumes (state stays DORMANT) | `on_dormant`; stop when convergence is on (`stop_reason="converged"`) | `optimizer.on_plateau`, `on_converged` |
| any regime or state change | `on_convergence(episode_id, report)` (including the cold start) | `on_convergence(epoch, report)` | `on_convergence(agent_id, window_id, report)` |

The RL loop and the multi-agent runner hold DORMANT (`hold_dormant`) until `min_completed_episodes_before_dormant` / `min_completed_episodes_before_converged` episodes have finished.

## Behaviour under simulation

The figures below are medians over 16 seeds. Rows marked † are other scenarios from the statistical audit.

| Scenario | Result |
|---|---|
| Flat plateau (σ = 5), false `IMPROVING` observations per 2,800 | 3 |
| Flat plateau, episodes to DORMANT | 220 |
| Autocorrelated plateau (AR ρ = 0.6), false `REGRESSED` observations per 2,800 / false collapses per run | 14.5 / 0 |
| Dip −30, then recovery to the same level, counted as `IMPROVING` | 0 |
| Dip −30, delay until reported as regressed / collapsing | 4 |
| Drop −15 that stays, delay to `REGRESSED_STALLED` | 31 |
| Collapse −30, delay to confirmed | 3 |
| Learning resumes at +0.5 / +0.2 / +0.05 per episode, delay to `IMPROVING` | 19 / 27 / 69 |
| Steady learner at +0.05 per episode, ever DORMANT | never |
| † 1 % outlier episodes at −60, false collapses per run | 0 |
| † Student-t(3) noise, false collapses per run | 0.5 |
| False collapse right after a rollback, autocorrelated noise | 0 % (p90 0) |

Cost is about 0.35 ms per observation.
