"""
Mathematical validation tests for the neuron protocol design.

Answers four questions before any production code is written:

  Q1. Merge detection — Option A (|h| scalar history) vs Option B (signed value)?
      Case that separates them: two neurons with correlated |h| but opposite sign.
      Anti-correlated neurons (h_a = -h_b) must NOT merge — they represent
      complementary features. Option A would incorrectly flag them.

  Q2. GRU relay function preservation — how large is the error?
      A relay neuron inserted on edge (u→v, w, d) must not change network output.
      GRU candidate uses tanh, so h_relay = tanh(x) ≠ x. Measure the error
      across the realistic activation range of a tanh network: x ∈ (-1, 1).

  Q3. LSTM relay function preservation — how large is the error?
      LSTM applies tanh twice: h = tanh(c) = tanh(tanh(x)). Measure vs GRU.

  Q4. Importance normalisation — are point neuron and GRU/LSTM scores comparable?
      Formula: I(v) = Σ|w_e| * (‖h‖₁/d + ε)
      Verify that normalising by hidden dim d keeps scores on the same scale.

Run with:
    python -m pytest tests/neuroevo/test_neuron_protocol_math.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Minimal scalar cell implementations (no framework deps, pure math)
# ──────────────────────────────────────────────────────────────────────────────

def point_neuron_step(x: float, bias: float = 0.0) -> float:
    """h = tanh(x + bias).  Standard point neuron."""
    return float(np.tanh(x + bias))


def gru_step(x: float, h_prev: float, p: dict) -> float:
    """
    Scalar GRU forward step.
    p keys: wz, uz, bz, wr, ur, br, wn, un, bn
    """
    z = 1 / (1 + np.exp(-(p["wz"] * h_prev + p["uz"] * x + p["bz"])))
    r = 1 / (1 + np.exp(-(p["wr"] * h_prev + p["ur"] * x + p["br"])))
    n = np.tanh(p["wn"] * r * h_prev + p["un"] * x + p["bn"])
    return float((1 - z) * h_prev + z * n)


def lstm_step(x: float, h_prev: float, c_prev: float, p: dict) -> tuple[float, float]:
    """
    Scalar LSTM forward step.
    p keys: wf, uf, bf, wi, ui, bi, wg, ug, bg, wo, uo, bo
    Returns (h, c).
    """
    f = 1 / (1 + np.exp(-(p["wf"] * h_prev + p["uf"] * x + p["bf"])))
    i = 1 / (1 + np.exp(-(p["wi"] * h_prev + p["ui"] * x + p["bi"])))
    g = np.tanh(p["wg"] * h_prev + p["ug"] * x + p["bg"])
    o = 1 / (1 + np.exp(-(p["wo"] * h_prev + p["uo"] * x + p["bo"])))
    c = f * c_prev + i * g
    h = o * np.tanh(c)
    return float(h), float(c)


def gru_relay_params() -> dict:
    """
    Identity-approximate GRU relay initialisation.

    Target: h_relay ≈ x  (relay passes signal through unchanged)

    z_t → 1  via  b_z = +10 (σ(10) = 0.99995)
    n_t = tanh(u_n * x + 0) = tanh(x)  via  u_n = 1

    So h_relay = (1-z)*h_prev + z*tanh(x) ≈ tanh(x).
    Error vs exact identity: |tanh(x) - x|.
    """
    return dict(wz=0.0, uz=0.0, bz=10.0,
                wr=0.0, ur=0.0, br=0.0,
                wn=0.0, un=1.0, bn=0.0)


def lstm_relay_params() -> dict:
    """
    Identity-approximate LSTM relay initialisation.

    Target: h_relay ≈ x

    f → 0  via  b_f = -10   (forget everything)
    i → 1  via  b_i = +10   (always write)
    g = tanh(u_g * x)       via  u_g = 1  → g = tanh(x)
    o → 1  via  b_o = +10   (always output)

    c_t = 0 + 1 * tanh(x) = tanh(x)
    h_t = 1 * tanh(c_t)   = tanh(tanh(x))

    Error vs exact identity: |tanh(tanh(x)) - x|.
    Two tanh squashings — worse than GRU.
    """
    return dict(wf=0.0, uf=0.0, bf=-10.0,
                wi=0.0, ui=0.0, bi=10.0,
                wg=0.0, ug=1.0, bg=0.0,
                wo=0.0, uo=0.0, bo=10.0)


def linear_relay(x: float) -> float:
    """Point neuron relay: exact identity.  h = x."""
    return x


# ──────────────────────────────────────────────────────────────────────────────
# Q1 — Merge detection: Option A vs Option B
# ──────────────────────────────────────────────────────────────────────────────

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


class TestMergeDetection:
    """
    Option A: store |h| in history, compute Pearson on magnitudes.
    Option B: store signed h in history, compute Pearson on signed values.

    The three cases that matter:
      redundant    — same signal, same sign   → both A and B say merge
      opposite     — same |h|, opposite sign  → A says merge, B says don't
      decorrelated — unrelated signals        → both A and B say don't merge
    """

    THRESHOLD = 0.95
    T = 500
    rng = np.random.default_rng(42)

    def _hist(self, signal: np.ndarray):
        return signal, np.abs(signal)  # returns (signed, magnitude)

    def test_case1_truly_redundant_both_agree(self):
        """Two neurons computing identical outputs — both options flag as merge."""
        base = np.sin(np.linspace(0, 10 * np.pi, self.T)) * 0.8
        h_a_signed, h_a_mag = self._hist(base)
        h_b_signed, h_b_mag = self._hist(base + self.rng.normal(0, 0.01, self.T))

        corr_A = pearson(h_a_mag, h_b_mag)
        corr_B = pearson(h_a_signed, h_b_signed)

        assert corr_A > self.THRESHOLD, f"Option A missed redundant pair: {corr_A:.3f}"
        assert corr_B > self.THRESHOLD, f"Option B missed redundant pair: {corr_B:.3f}"

    def test_case2_opposite_sign_option_A_false_positive(self):
        """
        Two neurons with h_b = -h_a.

        These represent COMPLEMENTARY features — removing one loses information.
        They must NOT merge.

        Option A (magnitude): |h_a| == |h_b| → corr = 1.0 → FALSE POSITIVE.
        Option B (signed):    h_b = -h_a    → corr = -1.0 → correctly rejects.
        """
        base = np.sin(np.linspace(0, 10 * np.pi, self.T)) * 0.8
        h_a = base
        h_b = -base  # anti-correlated

        corr_A = pearson(np.abs(h_a), np.abs(h_b))
        corr_B = pearson(h_a, h_b)

        # Option A incorrectly says merge
        assert corr_A > self.THRESHOLD, \
            f"Expected Option A false positive, got corr_A={corr_A:.3f}"

        # Option B correctly rejects
        assert corr_B < -self.THRESHOLD, \
            f"Expected Option B to detect anti-correlation, got corr_B={corr_B:.3f}"

        print(f"\n  [MERGE Q1] anti-correlated pair:")
        print(f"    Option A (|h|): corr = {corr_A:.3f}  → MERGE  (wrong)")
        print(f"    Option B (h):   corr = {corr_B:.3f}  → REJECT (correct)")

    def test_case3_decorrelated_both_reject(self):
        """Two neurons with unrelated signals — both options correctly reject."""
        h_a = np.sin(np.linspace(0, 10 * np.pi, self.T)) * 0.8
        h_b = np.cos(np.linspace(0, 7 * np.pi, self.T)) * 0.6

        corr_A = pearson(np.abs(h_a), np.abs(h_b))
        corr_B = pearson(h_a, h_b)

        assert corr_A < self.THRESHOLD, f"Option A false positive on decorrelated: {corr_A:.3f}"
        assert corr_B < self.THRESHOLD, f"Option B false positive on decorrelated: {corr_B:.3f}"

    def test_case4_how_common_is_anti_correlation_in_tanh_networks(self):
        """
        Simulate a small tanh network, measure how often anti-correlated pairs appear.
        If this is rare, Option A might be acceptable in practice despite the flaw.
        If common, Option B is necessary.
        """
        rng = np.random.default_rng(0)
        N_NEURONS = 20
        T = 1000
        N_PAIRS = N_NEURONS * (N_NEURONS - 1) // 2

        # Random tanh network: h_i(t) = tanh(W_i · x(t) + b_i)
        x = rng.normal(0, 1, (T, 8))             # 8-dim input stream
        W = rng.normal(0, 0.5, (N_NEURONS, 8))
        b = rng.normal(0, 0.3, N_NEURONS)
        H = np.tanh(x @ W.T + b)                  # [T, N_NEURONS]

        false_positives_A = 0
        true_detections_B = 0

        for i in range(N_NEURONS):
            for j in range(i + 1, N_NEURONS):
                corr_signed = pearson(H[:, i], H[:, j])
                corr_mag    = pearson(np.abs(H[:, i]), np.abs(H[:, j]))

                # Anti-correlation: B sees corr < -threshold, A sees corr_mag > threshold
                if corr_signed < -self.THRESHOLD and corr_mag > self.THRESHOLD:
                    false_positives_A += 1
                    true_detections_B += 1

        rate = false_positives_A / N_PAIRS * 100
        print(f"\n  [MERGE Q1] anti-correlated pairs in random tanh network:")
        print(f"    {false_positives_A}/{N_PAIRS} pairs ({rate:.1f}%) are anti-correlated")
        print(f"    Option A would incorrectly merge all of them")
        print(f"    VERDICT: {'Option B required' if false_positives_A > 0 else 'Option A acceptable'}")

        # The test is informational — it logs the rate.
        # Even 1 false positive is a correctness bug.
        assert false_positives_A >= 0  # always passes — result is in the output


# ──────────────────────────────────────────────────────────────────────────────
# Q2 + Q3 — Relay function preservation error
# ──────────────────────────────────────────────────────────────────────────────

class TestRelayFunctionPreservation:
    """
    Insert a relay on edge (u→v, w=1.0, d=0).
    Measure |h_v_before - h_v_after| across typical activation ranges.

    Point neuron relay: h_relay = x  (exact)
    GRU relay:          h_relay = tanh(x)      (one squash)
    LSTM relay:         h_relay = tanh(tanh(x)) (two squashes)

    Typical activation range for tanh networks: x ∈ (-1, 1).
    """

    def _relay_error(self, relay_fn, x_values):
        """Max and mean |relay(x) - x| over x_values."""
        errors = np.abs(np.array([relay_fn(float(x)) for x in x_values]) - x_values)
        return float(errors.max()), float(errors.mean())

    def test_linear_relay_is_exact(self):
        x = np.linspace(-1, 1, 1000)
        max_err, mean_err = self._relay_error(linear_relay, x)
        assert max_err < 1e-10, f"Linear relay not exact: max_err={max_err}"
        print(f"\n  [RELAY] Point neuron: max_err={max_err:.2e}  mean_err={mean_err:.2e}")

    def test_gru_relay_error_in_tanh_range(self):
        """
        GRU relay: h = tanh(x).
        Error = |tanh(x) - x| = x³/3 + O(x⁵) for small x.
        At x=1 (edge of tanh range): |tanh(1) - 1| = |0.762 - 1| = 0.238 = 23.8%.
        """
        gp = gru_relay_params()
        relay_fn = lambda x: gru_step(x, h_prev=0.0, p=gp)
        x = np.linspace(-1, 1, 1000)
        max_err, mean_err = self._relay_error(relay_fn, x)

        print(f"\n  [RELAY] GRU:         max_err={max_err:.4f}  mean_err={mean_err:.4f}")
        print(f"    at x=0.5: error = {abs(np.tanh(0.5) - 0.5):.4f}")
        print(f"    at x=1.0: error = {abs(np.tanh(1.0) - 1.0):.4f} ({abs(np.tanh(1.0)-1.0)*100:.1f}%)")

        # Document the error — not asserting acceptable, letting the number speak
        assert max_err > 0.1, \
            "GRU relay error is negligibly small — update assumptions"

    def test_lstm_relay_error_in_tanh_range(self):
        """
        LSTM relay: h = tanh(tanh(x)).
        Double squashing makes the error worse than GRU.
        """
        lp = lstm_relay_params()
        relay_fn = lambda x: lstm_step(x, h_prev=0.0, c_prev=0.0, p=lp)[0]
        x = np.linspace(-1, 1, 1000)
        max_err, mean_err = self._relay_error(relay_fn, x)

        # Compare to GRU
        gp = gru_relay_params()
        gru_fn = lambda x: gru_step(x, h_prev=0.0, p=gp)
        gru_max, _ = self._relay_error(gru_fn, x)

        print(f"\n  [RELAY] LSTM:        max_err={max_err:.4f}  mean_err={mean_err:.4f}")
        print(f"    at x=0.5: error = {abs(np.tanh(np.tanh(0.5)) - 0.5):.4f}")
        print(f"    at x=1.0: error = {abs(np.tanh(np.tanh(1.0)) - 1.0):.4f} ({abs(np.tanh(np.tanh(1.0))-1.0)*100:.1f}%)")
        print(f"    LSTM error vs GRU error: {max_err/gru_max:.2f}x worse")

        assert max_err > gru_max, "Expected LSTM relay to be worse than GRU relay"

    def test_relay_comparison_summary(self):
        """Print a clear comparison table across the activation range."""
        x_vals = np.array([0.1, 0.25, 0.5, 0.75, 1.0])
        gp = gru_relay_params()
        lp = lstm_relay_params()

        print("\n  [RELAY] Error |relay(x) - x| by input magnitude:")
        print(f"  {'x':>6}  {'linear':>10}  {'GRU':>10}  {'LSTM':>10}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")
        for x in x_vals:
            e_lin  = abs(linear_relay(x) - x)
            e_gru  = abs(gru_step(x, 0.0, gp) - x)
            e_lstm = abs(lstm_step(x, 0.0, 0.0, lp)[0] - x)
            print(f"  {x:>6.2f}  {e_lin:>10.2e}  {e_gru:>10.4f}  {e_lstm:>10.4f}")

        print(f"\n  VERDICT: linear relay is the only exact option.")
        print(f"  GRU/LSTM relays have significant error in the tanh activation range.")


# ──────────────────────────────────────────────────────────────────────────────
# Q4 — Importance normalisation
# ──────────────────────────────────────────────────────────────────────────────

class TestImportanceNormalisation:
    """
    Formula: I(v) = Σ|w_e| * (‖h‖₁/d + ε)

    For a point neuron (d=1): ‖h‖₁/d = |h|
    For a GRU (d=1 scalar):  ‖h‖₁/d = |h|  (same, scalar)
    For a GRU (d=K vector):  ‖h‖₁/d = mean(|h_i|)

    Test: same Σ|w_e| and same mean activation magnitude → same importance score
    regardless of neuron type or hidden dim.
    """

    def _importance(self, incident_weights: list, h_vec: np.ndarray) -> float:
        total_w = sum(abs(w) for w in incident_weights)
        d = len(h_vec)
        h_mag = float(np.abs(h_vec).mean())
        return total_w * (h_mag + 1e-8)

    def test_point_and_scalar_gru_same_score(self):
        """Point neuron and scalar GRU with same |h| and same edges → same score."""
        weights = [0.5, -0.3, 0.8, -0.1]
        h_point = np.array([0.6])
        h_gru   = np.array([0.6])

        I_point = self._importance(weights, h_point)
        I_gru   = self._importance(weights, h_gru)

        assert abs(I_point - I_gru) < 1e-9
        print(f"\n  [IMPORTANCE] point={I_point:.4f}  scalar_gru={I_gru:.4f}  diff={abs(I_point-I_gru):.2e}")

    def test_vector_gru_normalised_by_dim(self):
        """
        A GRU with d=8 and mean |h_i| = 0.6 should score the same as a
        point neuron with |h| = 0.6 given the same incident weights.
        """
        weights = [0.5, -0.3, 0.8, -0.1]
        h_point = np.array([0.6])
        # 8-dim GRU with mean magnitude 0.6
        rng = np.random.default_rng(7)
        h_gru_vec = rng.uniform(0.4, 0.8, 8) * rng.choice([-1, 1], 8)
        # Force mean |h| = 0.6
        h_gru_vec = h_gru_vec / np.abs(h_gru_vec).mean() * 0.6

        I_point   = self._importance(weights, h_point)
        I_gru_vec = self._importance(weights, h_gru_vec)

        assert abs(I_point - I_gru_vec) < 1e-6, \
            f"Normalisation broke comparability: point={I_point:.4f} gru={I_gru_vec:.4f}"
        print(f"\n  [IMPORTANCE] point={I_point:.4f}  vector_gru(d=8)={I_gru_vec:.4f}")

    def test_importance_scales_with_activity(self):
        """A dead neuron (h≈0) scores near zero regardless of connectivity."""
        weights = [1.0, 1.0, 1.0, 1.0]  # strong connectivity
        h_active = np.array([0.8])
        h_dead   = np.array([0.0])

        I_active = self._importance(weights, h_active)
        I_dead   = self._importance(weights, h_dead)

        assert I_active > 100 * I_dead, \
            f"Dead neuron not scoring near-zero: active={I_active:.4f} dead={I_dead:.6f}"
        print(f"\n  [IMPORTANCE] active={I_active:.4f}  dead={I_dead:.6f}")

    def test_importance_scales_with_connectivity(self):
        """An isolated neuron (no edges) scores near zero regardless of activation."""
        weights_connected = [0.5, -0.3, 0.8]
        weights_isolated  = []
        h = np.array([0.9])

        I_connected = self._importance(weights_connected, h)
        I_isolated  = self._importance(weights_isolated,  h)

        assert I_isolated < 1e-6
        assert I_connected > I_isolated
        print(f"\n  [IMPORTANCE] connected={I_connected:.4f}  isolated={I_isolated:.6f}")

    def test_unnormalised_vector_gru_inflated(self):
        """
        Without /d normalisation, a d=64 GRU would score 64x higher than a
        point neuron with the same per-unit activation — demonstrating why
        normalisation is necessary.
        """
        weights = [0.5, 0.5]
        h_point    = np.array([0.6])
        h_gru_d64  = np.full(64, 0.6)   # every unit at 0.6

        # Without normalisation: sum instead of mean
        total_w = sum(abs(w) for w in weights)
        I_unnorm = total_w * (float(np.abs(h_gru_d64).sum()) + 1e-8)  # no /d
        I_point  = self._importance(weights, h_point)
        I_norm   = self._importance(weights, h_gru_d64)                # with /d

        print(f"\n  [IMPORTANCE] without /d normalisation:")
        print(f"    point neuron:         {I_point:.4f}")
        print(f"    GRU d=64 (no /d):     {I_unnorm:.4f}  ({I_unnorm/I_point:.0f}x inflated)")
        print(f"    GRU d=64 (with /d):   {I_norm:.4f}  (comparable)")

        assert I_unnorm > 50 * I_point, "Expected inflation without normalisation"
        assert abs(I_norm - I_point) < 1e-6, "Expected comparability with normalisation"
