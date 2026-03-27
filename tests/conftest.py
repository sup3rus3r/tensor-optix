import pytest
import numpy as np
from tensor_optix.core.types import EpisodeData, EvalMetrics, HyperparamSet
from tensor_optix.core.base_agent import BaseAgent
from tensor_optix.core.base_evaluator import BaseEvaluator
from tensor_optix.core.base_pipeline import BasePipeline


class DummyAgent(BaseAgent):
    """Minimal agent for testing — no real learning."""

    def __init__(self, hyperparams: HyperparamSet = None):
        self._hyperparams = hyperparams or HyperparamSet(
            params={"learning_rate": 1e-3, "gamma": 0.99},
            episode_id=0,
        )
        self._weights = {"w": 1.0}
        self.learn_calls = 0

    def act(self, observation):
        return np.array([0])

    def learn(self, episode_data: EpisodeData) -> dict:
        self.learn_calls += 1
        return {"loss": 0.5}

    def get_hyperparams(self) -> HyperparamSet:
        return self._hyperparams.copy()

    def set_hyperparams(self, hyperparams: HyperparamSet) -> None:
        self._hyperparams = hyperparams.copy()

    def save_weights(self, path: str) -> None:
        import json, os
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "weights.json"), "w") as f:
            json.dump(self._weights, f)

    def load_weights(self, path: str) -> None:
        import json, os
        weights_file = os.path.join(path, "weights.json")
        if os.path.exists(weights_file):
            with open(weights_file) as f:
                self._weights = json.load(f)


class DummyEvaluator(BaseEvaluator):
    """Evaluator that returns total reward as primary score."""

    def score(self, episode_data: EpisodeData, train_diagnostics: dict) -> EvalMetrics:
        total = sum(episode_data.rewards)
        return EvalMetrics(
            primary_score=total,
            metrics={"total_reward": total},
            episode_id=episode_data.episode_id,
        )


class DummyPipeline(BasePipeline):
    """Pipeline that yields episodes with configurable rewards."""

    def __init__(self, rewards_sequence=None):
        # rewards_sequence: list of reward lists, cycles through them
        self._rewards_sequence = rewards_sequence or [[1.0, 1.0, 1.0]]
        self._idx = 0

    def setup(self) -> None:
        pass

    def episodes(self):
        while True:
            rewards = self._rewards_sequence[self._idx % len(self._rewards_sequence)]
            self._idx += 1
            n = len(rewards)
            yield EpisodeData(
                observations=np.zeros((n, 4)),
                actions=np.zeros(n, dtype=int),
                rewards=list(rewards),
                terminated=[False] * (n - 1) + [True],
                truncated=[False] * n,
                infos=[{}] * n,
                episode_id=self._idx,
            )

    def teardown(self) -> None:
        pass

    @property
    def is_live(self) -> bool:
        return False


@pytest.fixture
def dummy_agent():
    return DummyAgent()


@pytest.fixture
def dummy_evaluator():
    return DummyEvaluator()


@pytest.fixture
def dummy_pipeline():
    return DummyPipeline()


@pytest.fixture
def basic_hyperparams():
    return HyperparamSet(
        params={"learning_rate": 1e-3, "gamma": 0.99, "entropy_coeff": 0.01},
        episode_id=0,
    )


@pytest.fixture
def sample_episode():
    return EpisodeData(
        observations=np.random.rand(10, 4).astype(np.float32),
        actions=np.random.randint(0, 4, size=10),
        rewards=[1.0] * 9 + [0.0],
        terminated=[False] * 9 + [True],
        truncated=[False] * 10,
        infos=[{}] * 10,
        episode_id=0,
    )


@pytest.fixture
def sample_eval_metrics():
    return EvalMetrics(
        primary_score=10.0,
        metrics={"total_reward": 10.0, "mean_reward": 1.0},
        episode_id=0,
    )
