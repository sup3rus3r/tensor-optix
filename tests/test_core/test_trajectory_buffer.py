import numpy as np
import pytest
from tensor_optix.core.trajectory_buffer import compute_gae, make_minibatches, resolve_gae_bootstrap


class TestComputeGAE:

    def test_single_step_no_discount(self):
        # G_0 = r_0 = 1.0, A_0 = G_0 - V_0 = 1.0 - 0.5 = 0.5
        adv, ret = compute_gae([1.0], [0.5], [True], gamma=1.0, gae_lambda=1.0)
        assert adv.shape == (1,)
        assert abs(adv[0] - 0.5) < 1e-6
        assert abs(ret[0] - 1.0) < 1e-6

    def test_returns_equal_advantages_plus_values(self):
        rewards = [1.0, 0.5, -0.5, 2.0]
        values  = [0.8, 0.6,  0.3, 0.1]
        dones   = [False, False, False, True]
        adv, ret = compute_gae(rewards, values, dones)
        np.testing.assert_allclose(ret, adv + np.array(values, dtype=np.float32), atol=1e-6)

    def test_episode_boundary_zeros_propagation(self):
        # Two-episode window: done at index 1
        rewards = [1.0, 1.0, 1.0, 1.0]
        values  = [0.5, 0.5, 0.5, 0.5]
        dones   = [False, True, False, True]
        adv, ret = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
        # A[1] should not include any future from step 2+ (episode boundary)
        # delta[1] = r[1] + 0*V[2] - V[1] = 1.0 - 0.5 = 0.5
        # gae[1] = delta[1] + 0 * last_gae  →  gae[1] = 0.5
        assert abs(adv[1] - 0.5) < 1e-5

    def test_gae_lambda_zero_equals_td_residuals(self):
        # λ=0: A_t = r_t + γ*V(t+1)*(1-d) - V(t)  (one-step TD)
        rewards = [1.0, 1.0, 1.0]
        values  = [0.5, 0.5, 0.5]
        dones   = [False, False, True]
        adv, _ = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.0)
        expected_0 = 1.0 + 0.99 * 0.5 - 0.5
        expected_1 = 1.0 + 0.99 * 0.5 - 0.5
        expected_2 = 1.0 + 0.0       - 0.5  # done → next_value = 0
        np.testing.assert_allclose(adv, [expected_0, expected_1, expected_2], atol=1e-5)

    def test_gae_lambda_one_equals_montecarlo(self):
        # λ=1, γ=1: A_t = G_t - V(t)  where G_t = sum of future rewards
        rewards = [1.0, 2.0, 3.0]
        values  = [0.0, 0.0, 0.0]
        dones   = [False, False, True]
        adv, ret = compute_gae(rewards, values, dones, gamma=1.0, gae_lambda=1.0)
        np.testing.assert_allclose(ret, [6.0, 5.0, 3.0], atol=1e-5)

    def test_output_dtype_is_float32(self):
        adv, ret = compute_gae([1.0], [0.5], [True])
        assert adv.dtype == np.float32
        assert ret.dtype == np.float32


class TestComputeGAEVectorized:
    """n_envs > 1: VectorBatchPipeline interleaves N envs' transitions as
    index t*n_envs + i == env i's timestep t. compute_gae must reset its
    recurrence at each env's own stride, never letting a neighbouring
    entry from a *different* env at the same timestep be treated as the
    next timestep of the same trajectory."""

    def test_default_n_envs_one_matches_flat_recurrence(self):
        rewards = [1.0, 0.5, -0.5, 2.0]
        values  = [0.8, 0.6,  0.3, 0.1]
        dones   = [False, False, False, True]
        adv1, ret1 = compute_gae(rewards, values, dones)
        adv2, ret2 = compute_gae(rewards, values, dones, n_envs=1)
        np.testing.assert_allclose(adv1, adv2)
        np.testing.assert_allclose(ret1, ret2)

    def test_zero_reward_env_not_contaminated_by_other_env_reward(self):
        # env0 never pays reward; env1 pays 1.0 every step. Interleaved:
        # [env0_t0, env1_t0, env0_t1, env1_t1, ...]. Before the fix, the
        # flat recurrence treated env1's reward as env0's "next step",
        # inflating env0's advantages even though it never earns anything.
        n_envs, window = 2, 8
        rewards, values, dones = [], [], []
        for _ in range(window):
            rewards += [0.0, 1.0]
            values  += [0.5, 0.5]
            dones   += [False, False]
        last_value = [0.3, 0.9]

        adv, _ = compute_gae(rewards, values, dones, 0.99, 0.95, last_value, n_envs=n_envs)
        env0_adv = adv[0::2]
        env1_adv = adv[1::2]

        assert np.all(np.abs(env0_adv) < 0.5), (
            f"zero-reward env0 advantages should stay near zero, got {env0_adv}"
        )
        assert np.all(env1_adv > 1.0), (
            f"reward=1.0 env1 advantages should be clearly positive, got {env1_adv}"
        )

    def test_matches_independent_per_env_ground_truth(self):
        """Interleaving N independently-computed single-env GAE sequences
        must equal compute_gae(..., n_envs=N) on the interleaved buffer."""
        rng = np.random.default_rng(1)
        n_envs, window = 3, 7

        per_env_rewards = [rng.standard_normal(window).tolist() for _ in range(n_envs)]
        per_env_values  = [rng.standard_normal(window).tolist() for _ in range(n_envs)]
        per_env_dones   = [[False] * (window - 1) + [bool(rng.integers(0, 2))] for _ in range(n_envs)]
        per_env_last_value = rng.standard_normal(n_envs).tolist()

        rewards, values, dones = [], [], []
        for t in range(window):
            for e in range(n_envs):
                rewards.append(per_env_rewards[e][t])
                values.append(per_env_values[e][t])
                dones.append(per_env_dones[e][t])

        adv, ret = compute_gae(rewards, values, dones, 0.99, 0.95, per_env_last_value, n_envs=n_envs)

        gt_adv = np.zeros(window * n_envs, dtype=np.float32)
        gt_ret = np.zeros(window * n_envs, dtype=np.float32)
        for e in range(n_envs):
            a, r = compute_gae(
                per_env_rewards[e], per_env_values[e], per_env_dones[e],
                0.99, 0.95, per_env_last_value[e],
            )
            for t in range(window):
                gt_adv[t * n_envs + e] = a[t]
                gt_ret[t * n_envs + e] = r[t]

        np.testing.assert_allclose(adv, gt_adv, atol=1e-5)
        np.testing.assert_allclose(ret, gt_ret, atol=1e-5)

    def test_per_env_mid_window_termination_resets_only_that_env(self):
        # env0 terminates at its 3rd step (index 4 in the interleaved buffer);
        # env1 never terminates. env0's advantage propagation must stop there
        # without affecting env1's stream.
        n_envs, window = 2, 4
        rewards = [1.0, 1.0,  1.0, 1.0,  1.0, 1.0,  1.0, 1.0]
        values  = [0.5, 0.5,  0.5, 0.5,  0.5, 0.5,  0.5, 0.5]
        dones   = [False, False, False, False, True, False, False, False]
        last_value = [0.0, 0.9]

        adv, _ = compute_gae(rewards, values, dones, 0.99, 0.95, last_value, n_envs=n_envs)
        # env0's advantage at its terminal step (index 4) must not propagate
        # any bootstrap beyond the terminal state: delta = r + 0*0 - V
        assert abs(adv[4] - (1.0 - 0.5)) < 1e-5


class TestResolveGAEBootstrap:

    def _episode(self, final_obs, dones):
        from types import SimpleNamespace
        return SimpleNamespace(final_obs=final_obs, dones=dones)

    def test_single_env_none_final_obs_gives_zero_bootstrap(self):
        ep = self._episode(final_obs=None, dones=[False, False, True])
        last_value, n_envs = resolve_gae_bootstrap(ep, critic_fn=lambda obs: 99.0)
        assert n_envs == 1
        assert last_value == 0.0

    def test_single_env_mid_episode_end_bootstraps_from_critic(self):
        ep = self._episode(final_obs="obs", dones=[False, False, False])
        last_value, n_envs = resolve_gae_bootstrap(ep, critic_fn=lambda obs: 42.0)
        assert n_envs == 1
        assert last_value == 42.0

    def test_vector_final_obs_list_gives_per_env_bootstrap(self):
        ep = self._episode(final_obs=["obs0", None, "obs2"], dones=[False] * 6)
        seen = []

        def critic_fn(obs):
            seen.append(obs)
            return {"obs0": 1.0, "obs2": 3.0}[obs]

        last_value, n_envs = resolve_gae_bootstrap(ep, critic_fn=critic_fn)
        assert n_envs == 3
        assert last_value == [1.0, 0.0, 3.0]
        assert seen == ["obs0", "obs2"]  # never calls the critic for a terminal env


class TestMakeMinibatches:

    def setup_method(self):
        np.random.seed(0)
        self.T = 100
        self.data = {
            "obs":     np.random.randn(self.T, 4).astype(np.float32),
            "actions": np.random.randint(0, 3, size=self.T).astype(np.int32),
        }

    def test_total_samples_covered(self):
        collected = 0
        for batch in make_minibatches(self.data, minibatch_size=32):
            collected += batch["obs"].shape[0]
        assert collected == self.T

    def test_all_keys_present(self):
        for batch in make_minibatches(self.data, minibatch_size=32):
            assert "obs" in batch
            assert "actions" in batch
            break

    def test_minibatch_size_respected(self):
        sizes = [b["obs"].shape[0] for b in make_minibatches(self.data, minibatch_size=32)]
        # All except possibly the last should be exactly 32
        for s in sizes[:-1]:
            assert s == 32

    def test_shuffle_false_is_sequential(self):
        batches = list(make_minibatches(self.data, minibatch_size=50, shuffle=False))
        np.testing.assert_array_equal(batches[0]["obs"], self.data["obs"][:50])
        np.testing.assert_array_equal(batches[1]["obs"], self.data["obs"][50:])

    def test_arrays_aligned_across_keys(self):
        for batch in make_minibatches(self.data, minibatch_size=25):
            assert batch["obs"].shape[0] == batch["actions"].shape[0]
