import numpy as np
import pytest
from tensor_optix.core.normalizers import RunningMeanStd, ObsNormalizer, RewardNormalizer


class TestRunningMeanStd:

    def test_initial_mean_is_zero(self):
        rms = RunningMeanStd(shape=(4,))
        np.testing.assert_array_equal(rms.mean, np.zeros(4))

    def test_initial_var_is_one(self):
        rms = RunningMeanStd(shape=(4,))
        np.testing.assert_array_equal(rms.var, np.ones(4))

    def test_mean_converges_to_true_mean(self):
        rms = RunningMeanStd(shape=())
        data = np.random.normal(5.0, 1.0, size=(10_000,))
        for i in range(0, len(data), 100):
            rms.update(data[i:i+100])
        assert abs(rms.mean - 5.0) < 0.1

    def test_var_converges_to_true_var(self):
        rms = RunningMeanStd(shape=())
        data = np.random.normal(0.0, 2.0, size=(10_000,))
        for i in range(0, len(data), 100):
            rms.update(data[i:i+100])
        assert abs(rms.var - 4.0) < 0.2

    def test_normalize_zero_mean_unit_var(self):
        rms = RunningMeanStd(shape=())
        data = np.random.normal(10.0, 3.0, size=(5_000,))
        rms.update(data)
        normed = rms.normalize(data, clip=0)
        assert abs(normed.mean()) < 0.1
        assert abs(normed.std() - 1.0) < 0.1

    def test_normalize_clips(self):
        rms = RunningMeanStd(shape=())
        rms.update(np.zeros(100))
        normed = rms.normalize(np.array([1e9, -1e9]), clip=5.0)
        assert normed.max() <= 5.0
        assert normed.min() >= -5.0

    def test_vector_shape(self):
        rms = RunningMeanStd(shape=(3,))
        data = np.random.randn(1000, 3)
        rms.update(data)
        assert rms.mean.shape == (3,)
        assert rms.var.shape == (3,)

    def test_incremental_matches_batch(self):
        data = np.random.randn(500)
        rms_batch = RunningMeanStd(shape=())
        rms_batch.update(data)
        rms_inc = RunningMeanStd(shape=())
        for x in data:
            rms_inc.update(np.array([x]))
        assert abs(rms_batch.mean - rms_inc.mean) < 1e-6


class TestObsNormalizer:

    def test_update_and_normalize(self):
        norm = ObsNormalizer(obs_shape=(4,))
        data = np.random.normal(5.0, 2.0, size=(1000, 4)).astype(np.float32)
        norm.update(data)
        single = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32)
        normed = norm.normalize(single)
        assert normed.shape == (4,)
        np.testing.assert_allclose(normed, np.zeros(4), atol=0.2)

    def test_mean_var_accessible(self):
        norm = ObsNormalizer(obs_shape=(2,))
        norm.update(np.ones((100, 2)))
        assert norm.mean.shape == (2,)
        assert norm.var.shape == (2,)


class TestRewardNormalizer:

    def test_normalize_reduces_scale(self):
        norm = RewardNormalizer(gamma=0.99)
        rewards = [1.0] * 200
        for r in rewards:
            norm.step(r)
        normed = norm.normalize(rewards)
        assert np.abs(normed).max() < 5.0

    def test_reset_clears_running_return(self):
        norm = RewardNormalizer()
        for _ in range(100):
            norm.step(1.0)
        norm.reset()
        assert norm._running_return == 0.0
