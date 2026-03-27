from .base_agent import BaseAgent
from .policy_manager import PolicyManager
from .types import EpisodeData, HyperparamSet


class EnsembleAgent(BaseAgent):
    """
    Wraps a PolicyManager to present multiple agents as a single BaseAgent.

    The pipeline and LoopController see one agent. Internally, act() delegates
    to PolicyManager.ensemble_action(), combining multiple policies.

    The primary_agent handles everything except act():
        learn(), get/set_hyperparams(), save/load_weights()

    Usage:
        pm = PolicyManager(registry)
        pm.add_agent(agent_a, weight=1.0)
        pm.add_agent(agent_b, weight=0.5)
        ensemble = EnsembleAgent(pm, primary_agent=agent_a)
        pipeline.set_agent(ensemble)
        rl_opt = RLOptimizer(agent=ensemble, pipeline=pipeline, ...)
    """

    def __init__(self, policy_manager: PolicyManager, primary_agent: BaseAgent):
        self._pm = policy_manager
        self._primary = primary_agent

    def act(self, observation) -> any:
        return self._pm.ensemble_action(observation)

    def learn(self, episode_data: EpisodeData) -> dict:
        return self._primary.learn(episode_data)

    def get_hyperparams(self) -> HyperparamSet:
        return self._primary.get_hyperparams()

    def set_hyperparams(self, hyperparams: HyperparamSet) -> None:
        self._primary.set_hyperparams(hyperparams)

    def save_weights(self, path: str) -> None:
        self._primary.save_weights(path)

    def load_weights(self, path: str) -> None:
        self._primary.load_weights(path)

    @property
    def policy_manager(self) -> PolicyManager:
        return self._pm
