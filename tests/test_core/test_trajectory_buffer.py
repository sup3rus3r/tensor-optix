import numpy as np
import pytest
from tensor_optix.core.trajectory_buffer import compute_gae, make_minibatches


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
