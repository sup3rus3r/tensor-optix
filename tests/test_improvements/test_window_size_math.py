"""
Math validation tests for the optimal_window_size formula.

Formula:  window_size = clip(k * mean_episode_steps, min_steps, max_steps)

  k = 1.0  for SAC / TD3   (off-policy: 1 episode per window is enough)
  k = 4.0  for PPO          (on-policy: needs ~4x episode length for good GAE)
  min_steps = 512           (floor — prevents degenerate sub-batch windows)
  max_steps = 8192          (ceiling — prevents memory blow-up)

Validated values are cross-checked against Stable-Baselines3 defaults
(SB3 PPO n_steps=2048 for CartPole max=500 → k=4 gives 2000 ≈ 2048 ✓).
"""

import pytest


def ws(k: float, mean_ep: int, lo: int = 512, hi: int = 8192) -> int:
    return max(lo, min(hi, int(k * mean_ep)))


# ── PPO (k=4.0) ──────────────────────────────────────────────────────────────

class TestPPOWindowSize:
    def test_cartpole_matches_sb3_default(self):
        # SB3 PPO default n_steps=2048; CartPole max=500 → 4*500=2000 ≈ 2048
        assert ws(4.0, 500) == 2000

    def test_short_env_clamped_to_minimum(self):
        # 4 * 50 = 200 < 512 → clamped to 512
        assert ws(4.0, 50) == 512

    def test_humanoid_in_range(self):
        # HumanoidStandup-v5: max_episode_steps=1000 → 4*1000=4000
        assert ws(4.0, 1000) == 4000

    def test_long_env_clamped_to_maximum(self):
        # 4 * 3000 = 12000 > 8192 → clamped to 8192
        assert ws(4.0, 3000) == 8192

    def test_sb3_default_input_matches_sb3_output(self):
        # SB3 PPO uses n_steps=2048 by default; env with max=512 → 4*512=2048
        assert ws(4.0, 512) == 2048


# ── SAC / TD3 (k=1.0) ────────────────────────────────────────────────────────

class TestSACWindowSize:
    def test_humanoid_one_episode(self):
        assert ws(1.0, 1000) == 1000

    def test_biped_walker(self):
        # BipedalWalker-v3: max_episode_steps=1600
        assert ws(1.0, 1600) == 1600

    def test_short_env_clamped_to_minimum(self):
        # 1 * 100 = 100 < 512 → clamped to 512
        assert ws(1.0, 100) == 512

    def test_ant_v4(self):
        # Ant-v4: max_episode_steps=1000
        assert ws(1.0, 1000) == 1000


# ── Boundary / edge cases ─────────────────────────────────────────────────────

class TestWindowSizeBounds:
    def test_exactly_at_minimum(self):
        assert ws(1.0, 512) == 512

    def test_exactly_at_maximum(self):
        assert ws(1.0, 8192) == 8192

    def test_one_above_minimum(self):
        assert ws(1.0, 513) == 513

    def test_one_below_maximum(self):
        assert ws(1.0, 8191) == 8191

    def test_k_multiplier_scales_linearly(self):
        # Doubling k doubles the unclamped window
        k1 = ws(2.0, 1000)
        k2 = ws(4.0, 1000)
        assert k2 == 2 * k1

    def test_minimum_always_at_least_512(self):
        for ep in [1, 10, 50, 100, 200, 511]:
            assert ws(1.0, ep) == 512
            assert ws(4.0, ep) >= 512

    def test_maximum_never_exceeds_8192(self):
        for ep in [2049, 5000, 10000, 100000]:
            assert ws(1.0, ep) <= 8192
            assert ws(4.0, ep) <= 8192
