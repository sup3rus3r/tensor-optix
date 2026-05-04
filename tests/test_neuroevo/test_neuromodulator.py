"""Tests for NeuromodulatorSignal — regime-driven parameter modulation."""
import pytest
from unittest.mock import MagicMock
from tensor_optix.neuroevo.neuromodulator import NeuromodulatorSignal
from tensor_optix.core.types import EvalMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_metrics(scores: list) -> list:
    """Build a list of EvalMetrics from a list of primary scores."""
    return [EvalMetrics(primary_score=s, metrics={}, episode_id=i) for i, s in enumerate(scores)]


def make_detector(regime: str):
    """Mock RegimeDetector that always returns a fixed regime."""
    d = MagicMock()
    d.detect.return_value = regime
    return d


def make_hebbian_hook(lr: float = 1e-3):
    hook = MagicMock()
    hook.hebbian_lr = lr
    return hook


def make_agent(entropy_coef: float = 0.01):
    hp = MagicMock()
    hp.params = {"entropy_coef": entropy_coef}
    agent = MagicMock()
    agent.get_hyperparams.return_value = hp
    return agent, hp


def make_topology_controller(grow_gap: float = 0.15, prune_thresh: float = 1e-4):
    tc = MagicMock()
    tc.grow_gap_threshold = grow_gap
    tc.prune_neuron_threshold = prune_thresh
    return tc


# ---------------------------------------------------------------------------
# Construction — base values snapshotted
# ---------------------------------------------------------------------------

def test_base_hebbian_lr_snapshotted():
    hook = make_hebbian_hook(lr=5e-3)
    signal = NeuromodulatorSignal(detector=make_detector("ranging"), hebbian_hook=hook)
    assert signal._base_hebbian_lr == pytest.approx(5e-3)


def test_base_entropy_coef_snapshotted():
    agent, hp = make_agent(entropy_coef=0.02)
    signal = NeuromodulatorSignal(detector=make_detector("ranging"), agent=agent)
    assert signal._base_entropy_coef == pytest.approx(0.02)


def test_base_topology_snapshotted():
    tc = make_topology_controller(grow_gap=0.20, prune_thresh=2e-4)
    signal = NeuromodulatorSignal(detector=make_detector("ranging"), topology_controller=tc)
    assert signal._base_grow_gap == pytest.approx(0.20)
    assert signal._base_prune_threshold == pytest.approx(2e-4)


def test_default_regime_is_ranging():
    signal = NeuromodulatorSignal(detector=make_detector("ranging"))
    assert signal.current_regime == "ranging"


# ---------------------------------------------------------------------------
# step() returns detected regime
# ---------------------------------------------------------------------------

def test_step_returns_detected_regime():
    signal = NeuromodulatorSignal(detector=make_detector("trending"))
    regime = signal.step(make_metrics([1, 2, 3]))
    assert regime == "trending"


def test_step_increments_counter():
    signal = NeuromodulatorSignal(detector=make_detector("ranging"))
    signal.step(make_metrics([1, 2, 3]))
    signal.step(make_metrics([1, 2, 3]))
    assert signal.state["step"] == 2


# ---------------------------------------------------------------------------
# Hebbian modulation
# ---------------------------------------------------------------------------

def test_trending_lowers_hebbian_lr():
    hook = make_hebbian_hook(lr=1e-2)
    signal = NeuromodulatorSignal(detector=make_detector("trending"), hebbian_hook=hook)
    signal.step(make_metrics([1, 2, 3]))
    assert hook.hebbian_lr < 1e-2


def test_ranging_raises_hebbian_lr():
    hook = make_hebbian_hook(lr=1e-2)
    signal = NeuromodulatorSignal(detector=make_detector("ranging"), hebbian_hook=hook)
    signal.step(make_metrics([1, 2, 3]))
    assert hook.hebbian_lr > 1e-2


def test_volatile_lowers_hebbian_lr():
    hook = make_hebbian_hook(lr=1e-2)
    signal = NeuromodulatorSignal(detector=make_detector("volatile"), hebbian_hook=hook)
    signal.step(make_metrics([1, 2, 3]))
    assert hook.hebbian_lr < 1e-2


def test_hebbian_scale_relative_to_base():
    """Scale is always applied to base, not the previously modulated value."""
    hook = make_hebbian_hook(lr=1e-2)
    signal = NeuromodulatorSignal(detector=make_detector("ranging"), hebbian_hook=hook)
    signal.step(make_metrics([1, 2, 3]))
    lr_after_ranging = hook.hebbian_lr
    # Switch to trending — should use base * trending_scale, not ranging_result * scale
    signal.detector.detect.return_value = "trending"
    signal.step(make_metrics([1, 2, 3]))
    assert hook.hebbian_lr == pytest.approx(1e-2 * 0.5)


# ---------------------------------------------------------------------------
# Entropy modulation
# ---------------------------------------------------------------------------

def test_trending_lowers_entropy():
    agent, hp = make_agent(entropy_coef=0.01)
    signal = NeuromodulatorSignal(detector=make_detector("trending"), agent=agent)
    signal.step(make_metrics([1, 2, 3]))
    assert hp.params["entropy_coef"] < 0.01


def test_ranging_raises_entropy():
    agent, hp = make_agent(entropy_coef=0.01)
    signal = NeuromodulatorSignal(detector=make_detector("ranging"), agent=agent)
    signal.step(make_metrics([1, 2, 3]))
    assert hp.params["entropy_coef"] > 0.01


def test_volatile_raises_entropy():
    agent, hp = make_agent(entropy_coef=0.01)
    signal = NeuromodulatorSignal(detector=make_detector("volatile"), agent=agent)
    signal.step(make_metrics([1, 2, 3]))
    assert hp.params["entropy_coef"] > 0.01


# ---------------------------------------------------------------------------
# Topology modulation
# ---------------------------------------------------------------------------

def test_ranging_lowers_grow_gap_threshold():
    """Ranging → easier to grow → lower grow_gap_threshold."""
    tc = make_topology_controller(grow_gap=0.15)
    signal = NeuromodulatorSignal(detector=make_detector("ranging"), topology_controller=tc)
    signal.step(make_metrics([1, 2, 3]))
    assert tc.grow_gap_threshold < 0.15


def test_trending_raises_prune_threshold():
    """Trending → consolidate → prune more aggressively."""
    tc = make_topology_controller(prune_thresh=1e-4)
    signal = NeuromodulatorSignal(detector=make_detector("trending"), topology_controller=tc)
    signal.step(make_metrics([1, 2, 3]))
    assert tc.prune_neuron_threshold > 1e-4


def test_volatile_lowers_prune_threshold():
    """Volatile → don't prune unstable neurons."""
    tc = make_topology_controller(prune_thresh=1e-4)
    signal = NeuromodulatorSignal(detector=make_detector("volatile"), topology_controller=tc)
    signal.step(make_metrics([1, 2, 3]))
    assert tc.prune_neuron_threshold < 1e-4


# ---------------------------------------------------------------------------
# No targets attached — step() is a no-op but doesn't crash
# ---------------------------------------------------------------------------

def test_step_no_targets_does_not_crash():
    signal = NeuromodulatorSignal(detector=make_detector("volatile"))
    signal.step(make_metrics([1, 2, 3]))
    assert signal.current_regime == "volatile"


# ---------------------------------------------------------------------------
# Custom scale overrides
# ---------------------------------------------------------------------------

def test_custom_hebbian_lr_scale():
    hook = make_hebbian_hook(lr=1.0)
    custom = {"trending": 10.0, "ranging": 1.0, "volatile": 1.0}
    signal = NeuromodulatorSignal(
        detector=make_detector("trending"),
        hebbian_hook=hook,
        hebbian_lr_scale=custom,
    )
    signal.step(make_metrics([1, 2, 3]))
    assert hook.hebbian_lr == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# reset_to_base
# ---------------------------------------------------------------------------

def test_reset_to_base_restores_hebbian_lr():
    hook = make_hebbian_hook(lr=1e-2)
    signal = NeuromodulatorSignal(detector=make_detector("ranging"), hebbian_hook=hook)
    signal.step(make_metrics([1, 2, 3]))
    signal.reset_to_base()
    assert hook.hebbian_lr == pytest.approx(1e-2)


def test_reset_to_base_restores_entropy():
    agent, hp = make_agent(entropy_coef=0.01)
    signal = NeuromodulatorSignal(detector=make_detector("ranging"), agent=agent)
    signal.step(make_metrics([1, 2, 3]))
    signal.reset_to_base()
    assert hp.params["entropy_coef"] == pytest.approx(0.01)


def test_reset_to_base_restores_topology():
    tc = make_topology_controller(grow_gap=0.15, prune_thresh=1e-4)
    signal = NeuromodulatorSignal(detector=make_detector("ranging"), topology_controller=tc)
    signal.step(make_metrics([1, 2, 3]))
    signal.reset_to_base()
    assert tc.grow_gap_threshold == pytest.approx(0.15)
    assert tc.prune_neuron_threshold == pytest.approx(1e-4)


# ---------------------------------------------------------------------------
# state dict
# ---------------------------------------------------------------------------

def test_state_contains_regime_and_step():
    signal = NeuromodulatorSignal(detector=make_detector("trending"))
    signal.step(make_metrics([1, 2, 3]))
    s = signal.state
    assert s["regime"] == "trending"
    assert s["step"] == 1


def test_state_contains_active_values_when_targets_set():
    hook = make_hebbian_hook(lr=1e-3)
    agent, _ = make_agent(0.01)
    tc = make_topology_controller()
    signal = NeuromodulatorSignal(
        detector=make_detector("ranging"),
        hebbian_hook=hook,
        agent=agent,
        topology_controller=tc,
    )
    signal.step(make_metrics([1, 2, 3]))
    s = signal.state
    assert "hebbian_lr" in s
    assert "entropy_coef" in s
    assert "grow_gap_threshold" in s
    assert "prune_neuron_threshold" in s


# ---------------------------------------------------------------------------
# Top-level import
# ---------------------------------------------------------------------------

def test_neuromodulator_importable_from_top_level():
    from tensor_optix import NeuromodulatorSignal as NS
    assert NS is NeuromodulatorSignal
