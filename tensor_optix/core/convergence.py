"""
Convergence detection shared by every training loop.

One ``ConvergenceMonitor`` per run (or per multi-agent seat) reads the score
stream and answers, after every observation, *what the curve is doing* and
*how long it has been since real progress*. Loops act on the report; they do
not re-derive any of it.

Why this replaced the old single-slope check:

  * One 8-point slope cannot resolve slow learning, and a dip followed by a
    climb back looked exactly like improvement. Here three timescales are read
    together against a global peak, so returning to a level already reached is
    reported as ``REGRESSED_CLIMBING``, never as ``IMPROVING``.
  * Noise was estimated from residuals around one straight line, which a bent
    curve inflates. Here it comes from the spread of first differences (robust
    to curvature), corrected for autocorrelation.
  * Thresholds counted evaluations at a stretching interval. Here every
    position is in the caller's own units (episodes, windows or epochs), so
    patience means the same amount of training regardless of cadence.

Reading the report::

    report = monitor.observe(score, at=episode_id)
    report.regime      # ConvergenceRegime: what the curve is doing now
    report.state       # LoopState: ACTIVE / COOLING / DORMANT (patience)
    report.entered_cooling, report.entered_dormant, report.collapse_confirmed
    report.gap_to_peak, report.eta_to_peak   # cost of a regression
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from .types import LoopState


class ConvergenceRegime(Enum):
    """What the score curve is doing, judged against the run's own peak."""

    WARMUP = "warmup"                          # too little history to read trends
    IMPROVING = "improving"                    # new ground: above the best level reached
    PLATEAU = "plateau"                        # flat, at the peak
    REGRESSED_CLIMBING = "regressed_climbing"  # below the peak, climbing back
    REGRESSED_STALLED = "regressed_stalled"    # below the peak, not climbing
    COLLAPSING = "collapsing"                  # falling away from the peak


@dataclass(frozen=True)
class ScaleReading:
    """Level and slope over the most recent ``n`` observations."""

    n: int
    level: float        # mean score over the window
    level_se: float     # standard error of that mean
    slope: float        # OLS slope, score change per unit of `at`
    slope_se: float     # standard error of that slope


@dataclass(frozen=True)
class ConvergenceReport:
    """Everything a loop needs to act on one observation."""

    at: float
    score: float
    regime: ConvergenceRegime
    state: LoopState
    scales: Dict[str, ScaleReading] = field(default_factory=dict)
    sigma: Optional[float] = None          # per-observation noise estimate
    rho: float = 0.0                       # lag-1 autocorrelation of the noise
    peak: Optional[float] = None           # best level reached (bias-corrected)
    peak_at: Optional[float] = None        # position where the peak was last raised
    best_score: Optional[float] = None     # highest single raw score seen
    gap_to_peak: float = 0.0               # (peak - short level) / sigma; > 0 means below
    since_improvement: float = 0.0         # units of `at` since the last IMPROVING
    below_peak_for: float = 0.0            # units of `at` spent in REGRESSED_* / COLLAPSING
    eta_to_peak: Optional[float] = None    # REGRESSED_CLIMBING: units of `at` to climb back at the
                                           # current slope; inf when below the peak but not
                                           # climbing; None when not below the peak
    improved: bool = False                 # IMPROVING reported on this observation
    peak_raised: bool = False
    regime_changed: bool = False
    entered_cooling: bool = False
    entered_dormant: bool = False
    collapse_confirmed: bool = False       # entered COLLAPSING on this observation

    @property
    def is_regressed(self) -> bool:
        return self.regime in (
            ConvergenceRegime.REGRESSED_CLIMBING,
            ConvergenceRegime.REGRESSED_STALLED,
        )


_SCALE_NAMES = ("short", "medium", "long")


def _remove_slope_changes(d: np.ndarray, depth: int = 2, z: float = 3.5) -> np.ndarray:
    """
    Centre the first differences ``d`` on their local mean.

    The mean of the differences is the curve's slope, so a change of slope
    (a rise that flattens into a plateau) is a step in that mean. Steps that
    are significant under a CUSUM test are located by binary segmentation
    (up to ``depth`` levels) and each segment is centred on its own mean;
    without a significant step this is just ``d - mean(d)``, so stationary
    noise keeps an unbiased autocorrelation estimate.
    """
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    centred = d - d.mean() if n else d
    if n < 8 or depth <= 0:
        return centred
    # Spread of d estimated from ITS differences, which a step in the mean of
    # d does not inflate (for white noise Var(diff d) = 3 Var(d)).
    dd = np.diff(d)
    sd = 1.4826 * float(np.median(np.abs(dd - np.median(dd)))) / math.sqrt(3.0)
    if sd <= 0:
        sd = float(d.std())
    if sd <= 0:
        return centred
    k = np.arange(1, n, dtype=np.float64)
    stat = np.abs(np.cumsum(centred)[:-1]) / (sd * np.sqrt(k * (n - k) / n))
    j = int(np.argmax(stat))
    if stat[j] < z:
        return centred
    split = j + 1
    return np.concatenate([
        _remove_slope_changes(d[:split], depth - 1, z),
        _remove_slope_changes(d[split:], depth - 1, z),
    ])


def _t_quantile(z: float, df: float) -> float:
    """
    Student-t quantile matching the one-sided level of normal quantile ``z``
    (Hill's / Cornish-Fisher expansion; accurate to ~1% for df >= 1, and no
    scipy dependency).
    """
    df = max(float(df), 1.0)
    z3, z5 = z ** 3, z ** 5
    return z + (z3 + z) / (4 * df) + (5 * z5 + 16 * z3 + 3 * z) / (96 * df * df)


def _hampel(y: np.ndarray, sig: float, k: float = 3.0) -> np.ndarray:
    """
    Clip each point to (running median of 5) +- k*sigma before levels and
    slopes are taken. The noise estimate is a robust (MAD) scale, so the means
    it is compared with must be robust too: otherwise one spike or a heavy
    tail moves a short mean by several of its standard errors. The two newest
    points have no future neighbours; their reference is the trailing median
    advanced by the median recent step, so a steady trend is not clipped.
    """
    n = len(y)
    if n < 5 or sig <= 0:
        return y
    pad = np.concatenate([y[:2][::-1], y, y[-2:][::-1]])
    med = np.median(np.lib.stride_tricks.sliding_window_view(pad, 5), axis=1)
    tail = y[-5:]
    step = float(np.median(np.diff(tail)))
    centre = float(np.median(tail))          # sits at y[-3] for a monotone tail
    med[-2] = centre + step
    med[-1] = centre + 2.0 * step
    bound = k * sig + abs(step)
    return np.clip(y, med - bound, med + bound)



class ConvergenceMonitor:
    """
    Multi-timescale convergence detector with a global peak reference.

    Args:
        timescales:  (short, medium, long) window lengths, in observations.
                     The first ``medium`` observations are warm-up.
        cooling_patience: units of ``at`` without IMPROVING before COOLING.
        dormant_patience: units of ``at`` without IMPROVING before DORMANT.
                     DORMANT is never entered during warm-up, and is
                     withheld while the longest readable trend is still
                     significantly and materially rising (a slow learner is
                     not converged), while a collapse is pending or in
                     progress, and while hold_dormant(True) is in effect.
                     Once entered it is sticky: only IMPROVING or a
                     confirmed COLLAPSING ends it.
        z:           significance, in standard errors, for every trend and
                     level comparison.
        persist:     consecutive observations a new regime must hold before
                     it is reported (absorbs clusters of overlapping-window
                     false alarms). IMPROVING during warm-up is immediate.
        min_delta:   smallest absolute gain over the peak that counts as new
                     ground, on top of the statistical test. 0 by default.
        rel_delta:   the same, relative to |peak| (e.g. 1e-4 = 0.01 %). Keeps a
                     deterministic, ever-so-slowly improving metric (a smooth
                     validation loss) from counting as progress forever.
        outlier_k:   levels and slopes are computed on scores clipped to
                     (running median of 5) +- outlier_k noise SDs (a Hampel
                     filter; best_score still sees raw values), so one spike or
                     a heavy-tailed noise draw cannot set the peak or confirm a
                     collapse.
        min_at_before_dormant: DORMANT is never entered before this position.
        autocorrelation: correct standard errors for lag-1 autocorrelated
                     noise (on-policy RL scores usually are).
    """

    def __init__(
        self,
        timescales: Tuple[int, int, int] = (8, 32, 128),
        cooling_patience: float = 32,
        dormant_patience: float = 160,
        z: float = 2.0,
        persist: int = 3,
        min_delta: float = 0.0,
        min_at_before_dormant: float = 0,
        autocorrelation: bool = True,
        rel_delta: float = 0.0,
        outlier_k: float = 3.0,
    ):
        short, medium, long_ = (int(t) for t in timescales)
        if not (3 <= short <= medium <= long_) or long_ < 8:
            raise ValueError(
                "timescales must satisfy 3 <= short <= medium <= long and long >= 8 "
                f"(the noise estimate needs that much history), got {timescales}"
            )
        if dormant_patience < cooling_patience:
            raise ValueError("dormant_patience must be >= cooling_patience")
        self.timescales = (short, medium, long_)
        self.cooling_patience = float(cooling_patience)
        self.dormant_patience = float(dormant_patience)
        self.z = float(z)
        self.persist = max(1, int(persist))
        self.min_delta = float(min_delta)
        self.rel_delta = float(rel_delta)
        self.outlier_k = float(outlier_k)
        self.min_at_before_dormant = float(min_at_before_dormant)
        self.autocorrelation = bool(autocorrelation)
        self._dormant_hold = False
        self._full_reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _full_reset(self) -> None:
        self._at: Deque[float] = deque(maxlen=self.timescales[2])
        self._y: Deque[float] = deque(maxlen=self.timescales[2])
        self._n_total = 0                  # observations since the last full reset
        self._peak: Optional[float] = None
        self._peak_at: Optional[float] = None
        self._best_score: Optional[float] = None
        self._worst: Optional[float] = None
        self._sigma_memory: Optional[float] = None
        self._rho_memory: Optional[float] = None
        # Lag-1 difference correlation pooled over the whole run (EWMA): a
        # single window's estimate is too noisy to set every SE from.
        self._rd_pool: Optional[float] = None
        self._rd_n = 0
        self._yc: Optional[np.ndarray] = None   # Hampel-clipped window values
        self._first_warmup = True          # only the run's very first warm-up
        self._clear_segment()
        self._state = LoopState.ACTIVE
        self._last_improving_at: Optional[float] = None
        self._last_report: Optional[ConvergenceReport] = None

    def _clear_segment(self) -> None:
        """Forget the trend history but keep the peak (restart / restore)."""
        self._at.clear()
        self._y.clear()
        self._yc = None
        self._anchor: Optional[float] = None
        self._regime = ConvergenceRegime.WARMUP
        self._pending: Optional[ConvergenceRegime] = None
        self._pending_n = 0
        self._below_since: Optional[float] = None
        self._restored = False             # segment started by a rollback
        self._since_improving_obs = 0      # observations since the last IMPROVING
        self._improving_streak = 0         # consecutive IMPROVING observations
        # Peak set by fast tracking during a rise is provisional (see observe):
        # the medium-scale peak it replaced, and the best medium-scale level
        # seen while the plateau after the rise fills the medium window.
        self._provisional_base: Optional[float] = None
        self._settle_max: Optional[float] = None

    def reset_scale(self) -> None:
        """The score scale changed: nothing measured so far is comparable."""
        self._full_reset()

    def restart(self, at: Optional[float] = None) -> None:
        """
        A new policy variant (spawn, topology change) starts training.

        The peak is kept, so the variant must still beat the best level the run
        has reached; trend history and patience start afresh.
        """
        self._remember_sigma()
        self._clear_segment()
        self._first_warmup = False
        self._state = LoopState.ACTIVE
        self._last_improving_at = at

    def note_restored(self, at: Optional[float] = None) -> None:
        """
        The peak weights were restored (rollback). Scores taken before the
        restore describe a different policy, so trend history is cleared; the
        peak and patience clock are kept, since restoring is not progress.
        """
        self._remember_sigma()
        last_improving = self._last_improving_at
        state = self._state
        self._clear_segment()
        self._first_warmup = False
        self._restored = True
        self._state = state if state != LoopState.DORMANT else LoopState.COOLING
        self._last_improving_at = last_improving

    def _remember_sigma(self) -> None:
        # Only a real estimate (>= 5 observations) is carried into the next
        # segment; a 0.0 placeholder would bypass the "no noise yet" guards.
        if len(self._y) >= 5:
            self._sigma_memory, self._rho_memory = self._sigma_rho()

    def hold_dormant(self, hold: bool) -> None:
        """
        While held, the state stops at COOLING: DORMANT is neither entered nor
        reported (e.g. the owning loop has seen too few finished episodes to
        judge). Releasing the hold lets DORMANT be entered on the next
        observation if patience has run out.
        """
        self._dormant_hold = bool(hold)
        if self._dormant_hold and self._state == LoopState.DORMANT:
            self._state = LoopState.COOLING

    def defer_dormant(self) -> None:
        """
        One-off version of hold_dormant: step back from DORMANT to COOLING so
        DORMANT can be entered -- and reported -- again later.
        """
        if self._state == LoopState.DORMANT:
            self._state = LoopState.COOLING

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------

    @property
    def state(self) -> LoopState:
        return self._state

    @property
    def regime(self) -> ConvergenceRegime:
        return self._regime

    @property
    def peak(self) -> Optional[float]:
        return self._peak

    @property
    def best_score(self) -> Optional[float]:
        return self._best_score

    @property
    def last_report(self) -> Optional[ConvergenceReport]:
        return self._last_report

    @property
    def n_observations(self) -> int:
        return self._n_total

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _sigma_rho(self, update: bool = False) -> Tuple[float, float]:
        y = np.asarray(self._y, dtype=np.float64)
        if self._sigma_memory is not None and len(y) < self.timescales[1]:
            # A segment after restart() / note_restored(): a few points give a
            # poor, autocorrelation-free estimate, so the remembered (sigma,
            # rho) of the previous segment is used until the medium window is
            # full. (Without this an AR(1) run is judged with SEs ~3x too
            # small right after a rollback, and "collapses" again.)
            rho = self._rho_memory or 0.0
            if self._rd_pool is not None and self.autocorrelation:
                rho = float(np.clip(1.0 + 2.0 * self._rd_pool, 0.0, 0.95))
            return self._sigma_memory, rho
        if len(y) < 5:
            return 0.0, 0.0
        d = np.diff(y)
        # Differences centred on their local mean: a change of slope (a rise
        # flattening into a plateau) would otherwise widen their spread and
        # read as noise. See _remove_slope_changes.
        dc = _remove_slope_changes(d)
        mad = float(np.median(np.abs(dc - np.median(dc))))
        sig = 1.4826 * mad / math.sqrt(2.0)
        # Mostly-identical differences (flat or piecewise-constant scores):
        # their lag-1 correlation is ~0, which the AR(1) formula would read as
        # rho ~ 1 and inflate every SE several-fold. No autocorrelation there.
        degenerate = mad == 0.0
        if sig == 0.0:
            # Discrete-valued or piecewise-constant scores can have a zero MAD
            # while still varying; fall back to the mean absolute deviation
            # (still robust to a single step change, unlike the std).
            sig = 1.2533 * float(np.mean(np.abs(dc - np.median(dc)))) / math.sqrt(2.0)
        rho = 0.0
        if self.autocorrelation and not degenerate:
            # Lag-1 autocorrelation from the differences: for AR(1) noise
            # corr(d_t, d_t-1) = -(1 - rho) / 2. Differences are blind to a
            # linear trend; a change of slope shows up as a step in the mean
            # of d, which is located (binary segmentation, CUSUM test) and
            # removed before the correlation is taken. (Plain
            # residuals around one fitted line would read a rise followed by
            # a plateau as strongly autocorrelated noise.)
            # One window's estimate is noisy (p10-p90 0.41-0.81 at 127
            # differences for a true 0.6), so it is pooled over the run with
            # an EWMA, and never taken from fewer than 32 differences.
            denom = float((dc * dc).sum())
            if update and denom > 0 and len(d) >= max(32, self.timescales[1] - 1):
                r_d = float((dc[1:] * dc[:-1]).sum() / denom)
                self._rd_n += 1
                a = max(1.0 / 128.0, 1.0 / self._rd_n)
                self._rd_pool = r_d if self._rd_pool is None else (1 - a) * self._rd_pool + a * r_d
            if self._rd_pool is not None:
                rho = float(np.clip(1.0 + 2.0 * self._rd_pool, 0.0, 0.95))
            elif denom > 0 and len(d) >= 12:
                # Before the pool exists (first warm-up): a provisional
                # single-window estimate. It is noisy and, being clipped at 0,
                # biased upward -- i.e. conservative (wider SEs), which is the
                # safe side while there is little history.
                r_d = float((dc[1:] * dc[:-1]).sum() / denom)
                rho = float(np.clip(1.0 + 2.0 * r_d, 0.0, 0.95))
            # Differencing shrinks AR(1) noise by sqrt(1 - rho).
            sig = sig / math.sqrt(max(1e-6, 1.0 - rho))
        # Relative floor: a deterministic curve has (numerically) zero noise,
        # which would make every 1e-15 change "significant".
        scale = float(np.median(np.abs(y)))
        floor = 1e-9 * scale if scale > 0 else 1e-12
        return max(sig, floor), rho

    def _reading(self, k: int, sig: float, infl: float) -> ScaleReading:
        src = self._yc if self._yc is not None and len(self._yc) == len(self._y) else self._y
        y = np.asarray(src, dtype=np.float64)[-k:]
        t = np.asarray(self._at, dtype=np.float64)[-k:]
        tc = t - t.mean()
        sxx = float((tc * tc).sum())
        level = float(y.mean())
        if sxx <= 0:
            slope, slope_se = 0.0, float("inf")
        else:
            slope = float((tc * (y - level)).sum() / sxx)
            slope_se = sig * infl / math.sqrt(sxx)
        return ScaleReading(
            n=k, level=level, level_se=sig * infl / math.sqrt(k),
            slope=slope, slope_se=slope_se,
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(self, score: float, at: Optional[float] = None, probing: bool = False) -> ConvergenceReport:
        """
        Record one score and return the updated report.

        Args:
            score:   the evaluation score (higher is better).
            at:      position in training units (episode / window / epoch).
                     Must not decrease. Defaults to the observation count.
            probing: the score was taken under a deliberate perturbation
                     (e.g. an SPSA probe). A collapse is never confirmed on a
                     probing observation.
        """
        score = float(score)
        if not math.isfinite(score):
            raise ValueError(f"score must be finite, got {score!r}")
        if at is None:
            at = float(self._n_total)
        at = float(at)
        if self._at and at < self._at[-1]:
            raise ValueError(f"`at` must not decrease ({at} after {self._at[-1]})")

        if self._best_score is None or score > self._best_score:
            self._best_score = score
        stored = score
        self._at.append(at)
        self._y.append(stored)
        self._n_total += 1

        if self._last_improving_at is None:
            self._last_improving_at = at

        prev_regime = self._regime
        prev_state = self._state
        sig, rho = self._sigma_rho(update=True)
        infl = math.sqrt((1.0 + rho) / (1.0 - rho)) if rho > 0 else 1.0
        self._yc = (
            _hampel(np.asarray(self._y, dtype=np.float64), sig, self.outlier_k)
            if self.outlier_k > 0 else None
        )
        low = float(self._yc[-1]) if self._yc is not None and len(self._yc) else stored
        if self._worst is None or low < self._worst:
            self._worst = low
        short, medium, long_ = self.timescales
        n = len(self._y)

        scales: Dict[str, ScaleReading] = {}
        for name, k in zip(_SCALE_NAMES, self.timescales):
            if n >= k:
                scales[name] = self._reading(k, sig, infl)

        peak_raised = False
        early = (self._first_warmup and self._n_total < medium) or n < short
        if self._n_total == 1:
            # The very first score is the baseline: new ground by definition.
            self._anchor = score
            peak_raised = self._raise_peak(score, at)
            regime = ConvergenceRegime.IMPROVING
            self._pending, self._pending_n = None, 0
        elif early:
            raw = self._early_regime(sig, infl)
            if probing and raw == ConvergenceRegime.COLLAPSING:
                # A probe's drop is self-inflicted: it neither confirms nor
                # cancels a collapse that is being confirmed.
                regime = prev_regime
            elif raw == ConvergenceRegime.COLLAPSING:
                regime = self._persisted(raw, prev_regime)
            else:
                regime = raw
                self._pending, self._pending_n = None, 0
            if regime == ConvergenceRegime.IMPROVING:
                peak_raised = self._raise_peak(self._anchor, at)
        else:
            if self._first_warmup:
                self._first_warmup = False
                if "medium" in scales:
                    # End of the run's first warm-up. The warm-up peak came
                    # from 4-point means and is provisional: deflate it by its
                    # own uncertainty and keep the larger of that and the
                    # medium-scale estimate (a run that already fell during
                    # warm-up must not forget how high it had been).
                    med = scales["medium"]
                    warm_peak = self._peak
                    self._peak = None
                    self._raise_peak(med.level - med.level_se, at)
                    if warm_peak is not None:
                        # 4-point mean SE = sig/2; it was selected for being
                        # high, so deflate by z of them (2 SE), not 1.
                        self._raise_peak(warm_peak - self.z * sig * infl, at)
            raw = self._training_regime(scales, sig, infl, n)
            if probing and raw == ConvergenceRegime.COLLAPSING:
                # A probe's drop is self-inflicted: it neither confirms nor
                # cancels a collapse that is being confirmed.
                regime = prev_regime if prev_regime != ConvergenceRegime.WARMUP else ConvergenceRegime.PLATEAU
            else:
                regime = self._persisted(raw, prev_regime)
            m = scales.get("medium") or self._reading(n, sig, infl)
            candidate = m.level - m.level_se
            if regime == ConvergenceRegime.IMPROVING:
                sh = scales.get("short")
                if (
                    sh is not None
                    and m.slope > self.z * m.slope_se
                    and self._improving_streak >= short
                ):
                    # During an established rise (a significant medium slope
                    # and an IMPROVING streak of at least `short` observations,
                    # which a chance blip on a plateau does not sustain) the
                    # medium mean lags the current level by half a window. Two
                    # reads of the current level -- the short mean and the
                    # medium fit extrapolated to the newest point, each minus
                    # its SE -- must agree (the lower is taken), so a chance
                    # slope on a plateau cannot lift the peak on one noisy read.
                    fast_candidate = min(
                        sh.level - sh.level_se, self._fit_end_lower(m, sig, infl),
                    )
                    if fast_candidate > max(candidate, self._peak or -math.inf):
                        if self._provisional_base is None:
                            self._provisional_base = self._peak
                        candidate = fast_candidate
                peak_raised = self._raise_peak(candidate, at)
                if self._provisional_base is not None:
                    self._settle_max = None
            elif (
                regime == ConvergenceRegime.PLATEAU
                and self._since_improving_obs < medium
            ):
                # Catch-up: while a rise was being confirmed the medium mean
                # lagged the true level by half a window. For one window after
                # the last IMPROVING the peak may follow the medium level up,
                # so the top of the rise is remembered at its real height.
                if self._provisional_base is None:
                    peak_raised = self._raise_peak(candidate, at)
                else:
                    self._settle_max = candidate if self._settle_max is None else max(self._settle_max, candidate)
                    if self._since_improving_obs == medium - 1:
                        # Settle: a full medium window of plateau now follows
                        # the rise, so the medium-scale level measures its top
                        # better than the fast reads taken at the bend. (A run
                        # that collapsed or regressed instead never gets here
                        # and keeps the fast-tracked top.)
                        base = self._provisional_base
                        settled = self._settle_max if base is None else max(base, self._settle_max)
                        peak_raised = self._peak is None or settled > self._peak
                        self._peak = settled
                        self._peak_at = at
                        self._provisional_base = None
                        self._settle_max = None
            elif regime != ConvergenceRegime.PLATEAU:
                # Regressed / collapsing / unreadable: the provisional top is
                # what the run has to get back to; stop settling it.
                self._provisional_base = None
                self._settle_max = None
        if n >= short:
            self._restored = False

        self._regime = regime
        improved = regime == ConvergenceRegime.IMPROVING
        if improved:
            self._last_improving_at = at
            self._since_improving_obs = 0
            self._improving_streak += 1
        else:
            self._since_improving_obs += 1
            self._improving_streak = 0
            if n < 5 and self._sigma_memory is None:
                # No noise estimate yet, so no observation can be judged
                # either way: the patience clock does not run.
                self._last_improving_at = at

        below = regime in (
            ConvergenceRegime.REGRESSED_CLIMBING,
            ConvergenceRegime.REGRESSED_STALLED,
            ConvergenceRegime.COLLAPSING,
        )
        if below:
            if self._below_since is None:
                self._below_since = at
        else:
            self._below_since = None

        since = at - self._last_improving_at
        self._state = self._next_state(since, regime, scales, at, prev_state)

        level_s = scales["short"].level if "short" in scales else float(np.mean(self._y))
        gap = 0.0
        eta = None
        if self._peak is not None:
            gap = (self._peak - level_s) / sig if sig > 0 else 0.0
            if below:
                # Episodes to climb back at the current rate: only meaningful
                # while the run is actually (significantly) climbing.
                eta = float("inf")
                if regime == ConvergenceRegime.REGRESSED_CLIMBING:
                    ref = scales.get("medium") or scales.get("short")
                    slope = ref.slope if ref is not None else 0.0
                    if slope > 0:
                        eta = max(0.0, self._peak - level_s) / slope

        report = ConvergenceReport(
            at=at, score=score, regime=regime, state=self._state, scales=scales,
            sigma=sig if n >= 5 or self._sigma_memory is not None else None, rho=rho,
            peak=self._peak, peak_at=self._peak_at, best_score=self._best_score,
            gap_to_peak=gap, since_improvement=since,
            below_peak_for=(at - self._below_since) if self._below_since is not None else 0.0,
            eta_to_peak=eta, improved=improved, peak_raised=peak_raised,
            regime_changed=regime != prev_regime,
            entered_cooling=self._state == LoopState.COOLING and prev_state != LoopState.COOLING,
            entered_dormant=self._state == LoopState.DORMANT and prev_state != LoopState.DORMANT,
            collapse_confirmed=regime == ConvergenceRegime.COLLAPSING
            and prev_regime != ConvergenceRegime.COLLAPSING,
        )
        self._last_report = report
        return report

    # ------------------------------------------------------------------
    # Regime rules
    # ------------------------------------------------------------------

    def _early_regime(self, sig: float, infl: float) -> ConvergenceRegime:
        """
        Too little history for slopes: the first warm-up of a run, or the
        first few observations after a restart / restore. The mean of the last
        few scores is compared with a reference level:

          * IMPROVING when it beats the reference by a significant margin. In
            the first warm-up the reference is the level at the previous
            improvement; after a restart or restore it is the run's peak
            (climbing back to it is not new ground).
          * COLLAPSING when it -- and the newest score -- fall below the peak
            by twice that margin.
        """
        y = list(self._y)
        k = min(4, len(y))
        m = float(np.mean(y[-k:]))
        first_warmup = self._first_warmup
        if self._anchor is None:
            self._anchor = m
            if first_warmup:
                return ConvergenceRegime.WARMUP
        if len(y) < 5 and self._sigma_memory is None:
            return ConvergenceRegime.WARMUP     # no noise estimate yet
        # The noise estimate here comes from very few differences via MAD
        # (~37 % efficient), so the normal z is replaced by the Student-t
        # quantile at the matching level, df = 0.37 * (n - 2).
        # (After a restart / restore the remembered estimate is used instead,
        # so the normal z applies there.)
        z = _t_quantile(self.z, 0.37 * (len(y) - 2)) if first_warmup else self.z
        margin = max(z * sig * infl * math.sqrt(2.0 / k), self._min_gain())
        ref_up = self._anchor if first_warmup else (
            self._peak if self._peak is not None else self._anchor
        )
        if m > ref_up + margin or (margin == 0.0 and m > ref_up):
            self._anchor = m
            return ConvergenceRegime.IMPROVING
        ref_down = self._peak if self._peak is not None else self._anchor
        # In the run's first warm-up there is no earlier noise estimate to
        # lean on; a collapse needs at least 8 points (fewer give a noise
        # scale too unreliable to call one -- and autocorrelation is unknown).
        if (not first_warmup and len(y) >= 2) or len(y) >= 8:
            floor = ref_down - 2.0 * margin
            if m < floor and y[-1] < floor:
                return ConvergenceRegime.COLLAPSING
        return ConvergenceRegime.WARMUP

    def _training_regime(self, scales, sig: float, infl: float, n: int):
        """Joint reading of all available scales against the peak."""
        z = self.z
        s = scales.get("short")
        if s is None:
            return ConvergenceRegime.WARMUP
        m = scales.get("medium") or self._reading(n, sig, infl)
        lg = scales.get("long")
        peak = self._peak if self._peak is not None else m.level

        gain = self._min_gain()
        tol_s = max(z * s.level_se, gain)
        tol_m = max(z * m.level_se, gain)
        m_rising = m.slope > z * m.slope_se
        m_falling = m.slope < -z * m.slope_se
        s_rising = s.slope > z * s.slope_se
        l_rising = lg is not None and lg.slope > z * lg.slope_se

        # New ground. Fast path: the short level is clearly above the peak and
        # the medium trend is not falling. Slow path: the medium level itself
        # has cleared the peak and a longer trend is significantly rising
        # (slopes too gentle for the short window, resolved by the long one).
        # Continuation: inside an ongoing IMPROVING streak (the last one was a
        # few observations ago) the run keeps setting new ground as long as
        # the medium trend is still significantly rising and the short level
        # has not fallen back below the peak. A climb back after a dip never
        # qualifies: its peak was set long before, so there is no streak.
        fast = s.level > peak + tol_s and m.level > peak + m.level_se and m.slope > 0
        slow = m.level > peak + tol_m and (m_rising or l_rising)
        streak = self._since_improving_obs < self.persist
        continuing = streak and m_rising and s.level >= peak + gain
        if fast or slow or continuing:
            return ConvergenceRegime.IMPROVING

        if m_falling and s.level < peak - 2.0 * tol_s:
            return ConvergenceRegime.COLLAPSING

        # Below the peak: both the medium and the short level are under it by
        # more than the combined uncertainty of the level and of the peak
        # itself (the peak is a running maximum, so it sits a little above the
        # true best level even after its one-SE deflation). A single noisy
        # window cannot put the run "below peak".
        r2 = math.sqrt(2.0)
        below = (
            m.level < peak - max(z * r2 * m.level_se + m.level_se, gain)
            and s.level < peak - max(z * r2 * s.level_se, gain)
        )
        if below:
            if s_rising or m_rising:
                return ConvergenceRegime.REGRESSED_CLIMBING
            return ConvergenceRegime.REGRESSED_STALLED
        return ConvergenceRegime.PLATEAU

    def _min_gain(self) -> float:
        if self._peak is None or self.rel_delta <= 0:
            return self.min_delta
        # Relative to the size of the peak or, for a metric heading to zero
        # (a loss), to how far the run has come -- whichever is larger.
        span = self._peak - self._worst if self._worst is not None else 0.0
        return max(self.min_delta, self.rel_delta * max(abs(self._peak), span))

    def _fit_end_lower(self, m: ScaleReading, sig: float, infl: float) -> float:
        """Medium-window OLS fit evaluated at the newest point, minus its SE."""
        t = np.asarray(self._at, dtype=np.float64)[-m.n:]
        tc = t - t.mean()
        sxx = float((tc * tc).sum())
        dt = float(tc[-1])
        end = m.level + m.slope * dt
        se = sig * infl * math.sqrt(1.0 / m.n + (dt * dt / sxx if sxx > 0 else 0.0))
        return end - se

    def _persisted(self, raw: ConvergenceRegime, prev: ConvergenceRegime) -> ConvergenceRegime:
        if raw == prev:
            self._pending, self._pending_n = None, 0
            return raw
        if raw == self._pending:
            self._pending_n += 1
        else:
            self._pending, self._pending_n = raw, 1
        if self._pending_n >= self.persist or (
            prev == ConvergenceRegime.WARMUP and raw != ConvergenceRegime.COLLAPSING
        ):
            self._pending, self._pending_n = None, 0
            return raw
        return prev

    def _raise_peak(self, level: Optional[float], at: float) -> bool:
        if level is None:
            return False
        if self._peak is None or level > self._peak:
            self._peak = float(level)
            self._peak_at = at
            return True
        return False

    def _next_state(self, since: float, regime: ConvergenceRegime, scales, at: float,
                    prev_state: LoopState) -> LoopState:
        if regime == ConvergenceRegime.IMPROVING:
            return LoopState.ACTIVE
        if regime == ConvergenceRegime.COLLAPSING:
            # A collapsing run is not converged (and a converged one that
            # collapses is no longer converged).
            return LoopState.COOLING if prev_state == LoopState.DORMANT else prev_state
        if prev_state == LoopState.DORMANT and not self._dormant_hold:
            # DORMANT is sticky: only new ground or a confirmed collapse ends
            # it, never a chance fluctuation of a trend test.
            return LoopState.DORMANT
        if self._pending == ConvergenceRegime.COLLAPSING or (
            self._restored and regime == ConvergenceRegime.WARMUP
        ):
            # A collapse still being confirmed, or a just-restored policy whose
            # scores cannot be read yet: hold the patience state.
            return prev_state
        dormant_allowed = (
            regime != ConvergenceRegime.WARMUP     # trends must be readable
            and not self._dormant_hold
            and at >= self.min_at_before_dormant
        )
        if since >= self.dormant_patience and dormant_allowed:
            # A slow learner is not converged: the longest readable trend must
            # not still be significantly rising.
            # "Rising" must also be material: over another dormant-patience
            # period the trend has to add more than the minimum gain, or a
            # near-deterministic curve creeping by 1e-9 would never stop.
            guard = scales.get("long") or scales.get("medium")
            still_rising = (
                guard is not None
                and guard.slope > self.z * guard.slope_se
                and guard.slope * self.dormant_patience > self._min_gain()
            )
            if not still_rising:
                return LoopState.DORMANT
            return LoopState.COOLING
        if since >= self.cooling_patience:
            return LoopState.COOLING
        return LoopState.ACTIVE
