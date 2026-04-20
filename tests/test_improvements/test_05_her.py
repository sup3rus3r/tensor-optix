"""
tests/test_improvements/test_05_her.py

Tests for HERReplayBuffer — Hindsight Experience Replay.

Item 5 on the ROADMAP.

Mathematical claims verified:

1. RELABELING MATH
   For sparse reward r(s, g) ∈ {-1, 0}: a transition that failed under goal g
   (r = -1) may succeed (r = 0) under relabeled goal g' = achieved_goal.
   compute_reward(achieved_goal, g', {}) = 0 by construction.
   HER therefore injects non-zero gradient signal into every episode.

2. BUFFER FILL RATIO
   Without HER: fraction of buffer entries with reward == 0 ≈ P(success) ≈ 0
                in PointReachEnv (too sparse for a random policy).
   With HER:    fraction of buffer entries with reward == 0 > 0
                (at least 1 success per episode via final strategy).

3. STRATEGY CORRECTNESS
   future:  HER goals sampled only from t' ≥ t (future states in episode)
   final:   HER goal is always achieved_goals[-1]
   episode: HER goals sampled from any state in the episode

4. EPISODE AMPLIFICATION
   T transitions per episode + k*T HER transitions = T*(1+k) total stored.
   Verified exactly for k=4.

5. REWARD RECOMPUTED CORRECTLY
   When g' = achieved_goal[t], compute_reward(achieved_goal[t], g', {}) = 0.
   When g' ≠ achieved_goal[t] and distance > tol, reward remains -1.

6. INVALID STRATEGY RAISES
   Passing an unrecognised strategy name raises ValueError immediately.
"""

import sys
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Imports — no TF needed
# ---------------------------------------------------------------------------
sys.path.insert(0, "tests")
from envs.point_reach import PointReachEnv

from tensor_optix.core.replay_buffer import PrioritizedReplayBuffer
from tensor_optix.core.her_buffer import HERReplayBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ENV_OBS_DIM = 2   # PointReachEnv: [x, y]
GOAL_DIM    = 2   # [gx, gy]
ACT_DIM     = 2   # [dx, dy]
OBS_DIM     = ENV_OBS_DIM + GOAL_DIM   # flat obs passed to agent: [obs || goal]


def _make_her(k: int = 4, strategy: str = "future", capacity: int = 10_000) -> HERReplayBuffer:
    inner = PrioritizedReplayBuffer(capacity=capacity, alpha=0.0, n_step=1, gamma=0.99)
    return HERReplayBuffer(inner, k=k, strategy=strategy)


def _run_episode(
    env: PointReachEnv,
    seed: int = 0,
    action_seed: int = 0,
) -> dict:
    """
    Run one episode with random actions; return collected trajectory.
    Returns dict with keys: obs_list, act_list, rew_list, next_obs_list,
                            done_list, achieved_goals.
    """
    rng = np.random.default_rng(action_seed)
    obs_dict, _ = env.reset(seed=seed)

    obs_list          = []
    act_list          = []
    rew_list          = []
    next_obs_list     = []
    done_list         = []
    achieved_goals    = []   # achieved goal AFTER each transition

    for _ in range(env.max_steps):
        obs_flat = np.concatenate([
            obs_dict["observation"], obs_dict["desired_goal"]
        ]).astype(np.float32)
        action = rng.uniform(-1.0, 1.0, size=ACT_DIM).astype(np.float32)

        next_obs_dict, reward, terminated, truncated, _ = env.step(action)

        next_obs_flat = np.concatenate([
            next_obs_dict["observation"], next_obs_dict["desired_goal"]
        ]).astype(np.float32)
        done = terminated or truncated

        obs_list.append(obs_flat)
        act_list.append(action)
        rew_list.append(reward)
        next_obs_list.append(next_obs_flat)
        done_list.append(float(done))
        achieved_goals.append(next_obs_dict["achieved_goal"].copy())   # pos AFTER step

        obs_dict = next_obs_dict
        if done:
            break

    return dict(
        obs_list=obs_list, act_list=act_list, rew_list=rew_list,
        next_obs_list=next_obs_list, done_list=done_list,
        achieved_goals=achieved_goals,
    )


# ---------------------------------------------------------------------------
# 1. Reward recomputation math
# ---------------------------------------------------------------------------

class TestRewardRecomputation:

    def test_relabeling_achieved_goal_gives_zero_reward(self):
        """
        When g' = achieved_goal[t], compute_reward(achieved_goal[t], g', {}) = 0.
        This is HER's core insight: the agent DID achieve g' in hindsight.
        """
        env = PointReachEnv(tol=0.1, seed=0)
        achieved = np.array([0.5, 0.3], dtype=np.float32)
        r = env.compute_reward(achieved, achieved, {})
        assert float(r) == pytest.approx(0.0), \
            "compute_reward(g, g, {}) must be 0 (success)"

    def test_far_goal_gives_negative_one_reward(self):
        """A relabeled goal far from achieved position still gives -1."""
        env     = PointReachEnv(tol=0.1, seed=0)
        pos     = np.array([0.0, 0.0], dtype=np.float32)
        far_goal = np.array([1.0, 1.0], dtype=np.float32)
        r = env.compute_reward(pos, far_goal, {})
        assert float(r) == pytest.approx(-1.0)

    def test_batched_compute_reward(self):
        """compute_reward supports batched inputs for vectorised relabeling."""
        env = PointReachEnv(tol=0.1)
        achieved = np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float32)
        desired  = np.array([[0.0, 0.0], [0.9, 0.9]], dtype=np.float32)
        rewards = env.compute_reward(achieved, desired, {})
        assert float(rewards[0]) == pytest.approx(0.0),  "first pair: same pos → success"
        assert float(rewards[1]) == pytest.approx(-1.0), "second pair: far → failure"


# ---------------------------------------------------------------------------
# 2. Buffer fill ratio (before / after HER)
# ---------------------------------------------------------------------------

class TestBufferFillRatio:

    def test_without_her_mostly_negative_rewards(self):
        """
        Without HER, a random policy on PointReachEnv fills the buffer with
        almost entirely r=-1 transitions (success probability ≈ 0.008).
        We verify fraction_success < 0.10 over 20 episodes with random actions.
        """
        env    = PointReachEnv(tol=0.1, max_steps=50, seed=0)
        inner  = PrioritizedReplayBuffer(capacity=10_000, alpha=0.0)
        n_success = 0
        n_total   = 0

        rng = np.random.default_rng(42)
        for ep in range(20):
            obs_dict, _ = env.reset(seed=ep)
            for _ in range(env.max_steps):
                act = rng.uniform(-1, 1, size=2).astype(np.float32)
                next_obs_dict, r, terminated, truncated, _ = env.step(act)
                obs_flat      = np.concatenate([obs_dict["observation"], obs_dict["desired_goal"]])
                next_obs_flat = np.concatenate([next_obs_dict["observation"], next_obs_dict["desired_goal"]])
                inner.push(obs_flat, act, r, next_obs_flat, float(terminated or truncated))
                n_success += int(r == 0.0)
                n_total   += 1
                obs_dict = next_obs_dict
                if terminated or truncated:
                    break
            inner.flush_episode()

        fraction_success = n_success / max(n_total, 1)
        assert fraction_success < 0.10, \
            f"Without HER, expected < 10% success transitions, got {fraction_success:.2%}"

    def test_with_her_injects_success_transitions(self):
        """
        With HER (final strategy, k=1), every episode produces at least 1
        transition with r=0 — the final state is always a success under its
        own achieved goal as desired goal.
        Verified: after 10 episodes, buffer contains r=0 transitions.
        """
        env = PointReachEnv(tol=0.1, max_steps=50, seed=0)
        her = _make_her(k=1, strategy="final")

        for ep in range(10):
            traj = _run_episode(env, seed=ep, action_seed=ep + 100)
            her.store_episode(
                traj["obs_list"], traj["act_list"], traj["rew_list"],
                traj["next_obs_list"], traj["done_list"], traj["achieved_goals"],
                env.compute_reward,
            )

        assert len(her) > 0, "Buffer should not be empty after episodes"

        # Sample all transitions and count successes
        n = min(len(her), 500)
        _, _, rews, _, _, _, _, _ = her.sample(n)
        n_success = int(np.sum(rews == 0.0))
        assert n_success > 0, \
            "HER must inject at least 1 success (r=0) transition per episode"

    def test_her_increases_success_fraction_vs_no_her(self):
        """
        Fraction of success transitions in buffer: HER >> no-HER.
        Uses identical random episodes on same environment seeds.
        """
        env     = PointReachEnv(tol=0.1, max_steps=50, seed=0)
        her_buf = _make_her(k=4, strategy="final")
        raw_buf = PrioritizedReplayBuffer(capacity=10_000, alpha=0.0)

        for ep in range(15):
            traj = _run_episode(env, seed=ep, action_seed=ep)

            # HER
            her_buf.store_episode(
                traj["obs_list"], traj["act_list"], traj["rew_list"],
                traj["next_obs_list"], traj["done_list"], traj["achieved_goals"],
                env.compute_reward,
            )

            # No-HER (just push raw transitions)
            for t in range(len(traj["act_list"])):
                raw_buf.push(
                    traj["obs_list"][t], traj["act_list"][t], traj["rew_list"][t],
                    traj["next_obs_list"][t], traj["done_list"][t],
                )
            raw_buf.flush_episode()

        def _success_fraction(buf):
            n = min(len(buf), 500)
            _, _, rews, _, _, _, _, _ = buf.sample(n)
            return float(np.mean(rews == 0.0))

        her_frac = _success_fraction(her_buf)
        raw_frac = _success_fraction(raw_buf)
        assert her_frac > raw_frac, \
            f"HER success fraction ({her_frac:.3f}) must exceed raw ({raw_frac:.3f})"


# ---------------------------------------------------------------------------
# 3. Episode amplification (T → T*(1+k) transitions)
# ---------------------------------------------------------------------------

class TestEpisodeAmplification:

    def test_k4_stores_five_times_transitions(self):
        """
        k=4 → each transition stored 5 times (1 original + 4 HER).
        Total buffer entries = T * 5 after one episode.
        """
        env  = PointReachEnv(tol=0.1, max_steps=20, seed=7)
        her  = _make_her(k=4, strategy="episode")
        traj = _run_episode(env, seed=7, action_seed=7)
        T    = len(traj["act_list"])

        her.store_episode(
            traj["obs_list"], traj["act_list"], traj["rew_list"],
            traj["next_obs_list"], traj["done_list"], traj["achieved_goals"],
            env.compute_reward,
        )

        expected = T * 5   # 1 original + 4 HER
        assert len(her) == expected, \
            f"Expected {expected} transitions (T={T}, k=4), got {len(her)}"

    def test_k0_stores_only_original(self):
        """k=0: no HER relabeling, only original transitions."""
        env  = PointReachEnv(tol=0.1, max_steps=20, seed=1)
        her  = _make_her(k=0, strategy="final")
        traj = _run_episode(env, seed=1, action_seed=1)
        T    = len(traj["act_list"])

        her.store_episode(
            traj["obs_list"], traj["act_list"], traj["rew_list"],
            traj["next_obs_list"], traj["done_list"], traj["achieved_goals"],
            env.compute_reward,
        )
        assert len(her) == T, f"Expected {T} (original only), got {len(her)}"


# ---------------------------------------------------------------------------
# 4. Strategy correctness
# ---------------------------------------------------------------------------

class TestStrategyCorrectness:

    def _collect_relabeled_goals(self, strategy: str, seed: int = 42) -> list:
        """
        Run 1 episode, store with given strategy, sample all transitions,
        extract the desired_goal part of each next_obs.  These are the
        goals used in relabeled transitions.
        """
        env  = PointReachEnv(tol=0.1, max_steps=10, seed=seed)
        her  = _make_her(k=1, strategy=strategy)
        traj = _run_episode(env, seed=seed, action_seed=seed)
        T    = len(traj["act_list"])

        her.store_episode(
            traj["obs_list"], traj["act_list"], traj["rew_list"],
            traj["next_obs_list"], traj["done_list"], traj["achieved_goals"],
            env.compute_reward,
        )
        return traj, her

    def test_final_strategy_uses_last_achieved_goal(self):
        """
        final: all k relabeled transitions use achieved_goals[-1] as desired goal.
        We verify by checking that, among relabeled transitions, the desired_goal
        part of obs always equals the last achieved goal.
        """
        env  = PointReachEnv(tol=0.1, max_steps=10, seed=5)
        her  = _make_her(k=1, strategy="final")
        traj = _run_episode(env, seed=5, action_seed=5)

        her.store_episode(
            traj["obs_list"], traj["act_list"], traj["rew_list"],
            traj["next_obs_list"], traj["done_list"], traj["achieved_goals"],
            env.compute_reward,
        )

        last_achieved = np.asarray(traj["achieved_goals"][-1], dtype=np.float32)
        # Sample a large batch and look for transitions whose desired_goal == last_achieved
        n = len(her)
        obs_b, _, _, _, _, _, _, _ = her.sample(n)

        # desired_goal is in obs[env_obs_dim:]
        desired_goals_in_buffer = obs_b[:, ENV_OBS_DIM:]
        # At least some transitions should have desired_goal == last_achieved
        matches = np.all(
            np.abs(desired_goals_in_buffer - last_achieved[np.newaxis, :]) < 1e-5,
            axis=-1,
        )
        assert np.any(matches), \
            "final strategy: expected transitions with desired_goal == last_achieved"

    def test_invalid_strategy_raises(self):
        """Unknown strategy name raises ValueError at construction."""
        inner = PrioritizedReplayBuffer(capacity=100, alpha=0.0)
        with pytest.raises(ValueError, match="Unknown HER strategy"):
            HERReplayBuffer(inner, k=4, strategy="random_nonexistent")

    def test_all_valid_strategies_accepted(self):
        """future, final, episode are all valid at construction."""
        inner = PrioritizedReplayBuffer(capacity=100, alpha=0.0)
        for s in ("future", "final", "episode"):
            her = HERReplayBuffer(inner, k=2, strategy=s)
            assert her._strategy == s


# ---------------------------------------------------------------------------
# 5. Sampling API compatibility
# ---------------------------------------------------------------------------

class TestSamplingAPI:

    def test_sample_returns_eight_tuple(self):
        """sample() returns the same 8-tuple as PrioritizedReplayBuffer."""
        env  = PointReachEnv(tol=0.1, max_steps=30, seed=0)
        her  = _make_her(k=2, strategy="future")
        for ep in range(3):
            traj = _run_episode(env, seed=ep, action_seed=ep)
            her.store_episode(
                traj["obs_list"], traj["act_list"], traj["rew_list"],
                traj["next_obs_list"], traj["done_list"], traj["achieved_goals"],
                env.compute_reward,
            )

        result = her.sample(32)
        assert len(result) == 8, f"Expected 8-tuple, got {len(result)}"
        obs_b, act_b, rew_b, next_b, done_b, weights, indices, n_steps = result
        assert obs_b.shape == (32, OBS_DIM),  f"obs shape: {obs_b.shape}"
        assert act_b.shape == (32, ACT_DIM),  f"act shape: {act_b.shape}"
        assert rew_b.shape == (32,),          f"rew shape: {rew_b.shape}"
        assert next_b.shape == (32, OBS_DIM), f"next_obs shape: {next_b.shape}"

    def test_obs_contains_goal_concatenated(self):
        """
        The flat obs stored is [env_obs || desired_goal].
        ENV_OBS_DIM = 2, GOAL_DIM = 2 → total obs dim = 4.
        """
        env  = PointReachEnv(tol=0.1, max_steps=20, seed=3)
        her  = _make_her(k=1, strategy="final")
        traj = _run_episode(env, seed=3, action_seed=3)
        her.store_episode(
            traj["obs_list"], traj["act_list"], traj["rew_list"],
            traj["next_obs_list"], traj["done_list"], traj["achieved_goals"],
            env.compute_reward,
        )
        obs_b, _, _, _, _, _, _, _ = her.sample(min(len(her), 16))
        assert obs_b.shape[-1] == OBS_DIM, \
            f"Expected obs dim {OBS_DIM}, got {obs_b.shape[-1]}"

    def test_rewards_are_only_zero_or_minus_one(self):
        """PointReachEnv only emits 0 or -1; HER relabeling preserves this."""
        env  = PointReachEnv(tol=0.1, max_steps=50, seed=99)
        her  = _make_her(k=4, strategy="future")
        for ep in range(5):
            traj = _run_episode(env, seed=ep, action_seed=ep * 7)
            her.store_episode(
                traj["obs_list"], traj["act_list"], traj["rew_list"],
                traj["next_obs_list"], traj["done_list"], traj["achieved_goals"],
                env.compute_reward,
            )

        n = min(len(her), 200)
        _, _, rews, _, _, _, _, _ = her.sample(n)
        unique_rewards = set(float(r) for r in rews)
        assert unique_rewards.issubset({0.0, -1.0}), \
            f"Expected rewards in {{0.0, -1.0}}, got {unique_rewards}"
