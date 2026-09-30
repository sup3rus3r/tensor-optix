import numpy as np
import pytest

from tensor_optix.core.reward_shape import (
    episode_breadth,
    breadth_typicality,
    split_episode_returns,
)


class TestSplitEpisodeReturns:

    def test_splits_on_done(self):
        values = [1, 2, 3, 4, 5]
        dones = [False, True, False, False, True]
        segments = split_episode_returns(values, dones)
        assert segments == [[1, 2], [3, 4, 5]]

    def test_drops_trailing_incomplete_segment(self):
        values = [1, 2, 3]
        dones = [True, False, False]
        segments = split_episode_returns(values, dones)
        assert segments == [[1]]


class TestEpisodeBreadth:

    def test_concentrated_reward_has_low_breadth(self):
        # All reward earned on one step -> maximally concentrated.
        rewards = [0.0, 0.0, 0.0, 100.0]
        dones = [False, False, False, True]
        breadths = episode_breadth(rewards, dones)
        assert len(breadths) == 1
        assert breadths[0] < 0.3

    def test_uniform_reward_has_high_breadth(self):
        rewards = [25.0, 25.0, 25.0, 25.0]
        dones = [False, False, False, True]
        breadths = episode_breadth(rewards, dones)
        assert breadths[0] > 0.9

    def test_sign_agnostic_bonus_vs_cost_style(self):
        # Same concentration profile, opposite sign convention (cost-style:
        # a big one-time cost avoided looks like a big spike too).
        bonus_style = [0.0, 0.0, 0.0, 100.0]
        cost_style = [-1.0, -1.0, -1.0, -100.0]
        dones = [False, False, False, True]
        b1 = episode_breadth(bonus_style, dones)[0]
        b2 = episode_breadth(cost_style, dones)[0]
        assert abs(b1 - b2) < 0.05  # structurally identical concentration

    def test_mixed_sign_uniform_is_broad(self):
        rewards = [1.0, -1.0, 1.0, -1.0]
        dones = [False, False, False, True]
        breadth = episode_breadth(rewards, dones)[0]
        assert breadth > 0.9

    def test_empty_rewards(self):
        assert episode_breadth([], []) == []

    def test_multiple_episodes_in_window(self):
        rewards = [10.0, 10.0, 0.0, 0.0, 0.0, 50.0]
        dones = [False, True, False, False, False, True]
        breadths = episode_breadth(rewards, dones)
        assert len(breadths) == 2
        assert breadths[0] > breadths[1]  # first uniform, second concentrated

    def test_all_zero_reward_is_treated_as_uniform(self):
        rewards = [0.0, 0.0, 0.0]
        dones = [False, False, True]
        breadths = episode_breadth(rewards, dones)
        assert breadths[0] == 1.0


class TestBreadthTypicality:

    def test_no_history_defaults_to_typical(self):
        assert breadth_typicality(0.1, history=[]) == 1.0

    def test_insufficient_history_defaults_to_typical(self):
        history = [0.8, 0.9, 0.85]  # < min_samples default of 5
        assert breadth_typicality(0.1, history=history) == 1.0

    def test_typical_breadth_scores_fully_satisfied(self):
        history = [0.8, 0.82, 0.79, 0.81, 0.80, 0.78]
        assert breadth_typicality(0.80, history=history) >= 0.999

    def test_anomalously_narrow_breadth_is_penalized(self):
        history = [0.8, 0.82, 0.79, 0.81, 0.80, 0.78]
        score = breadth_typicality(0.05, history=history)
        assert score < 1.0

    def test_above_average_breadth_never_penalized(self):
        history = [0.3, 0.32, 0.29, 0.31, 0.30]
        assert breadth_typicality(0.9, history=history) == 1.0

    def test_sparse_reward_domain_never_flagged_once_normal(self):
        # Every legitimate win in this domain is a single terminal spike
        # (breadth always low, ~0.05) — that must not get crushed the way
        # an anomalous outlier would, once the run's own history establishes
        # low breadth as this domain's normal. A within-range reading should
        # stay close to fully satisfied, not near zero.
        history = [0.05, 0.06, 0.04, 0.05, 0.05, 0.06]
        assert breadth_typicality(0.05, history=history) > 0.7
        # A genuinely anomalous, far-more-concentrated reading should still
        # be discounted relative to this domain's own tight norm.
        assert breadth_typicality(0.001, history=history) < 0.7
