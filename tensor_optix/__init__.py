from .core.types import (
    EpisodeData,
    EvalMetrics,
    HyperparamSet,
    PolicySnapshot,
    LoopState,
)
from .core.base_agent import BaseAgent
from .core.base_evaluator import BaseEvaluator
from .core.base_optimizer import BaseOptimizer
from .core.base_pipeline import BasePipeline
from .core.loop_controller import LoopCallback
from .core.policy_manager import PolicyManager, PolicyManagerCallback
from .core.ensemble_agent import EnsembleAgent
from .core.regime_detector import RegimeDetector
from .core.meta_controller import MetaController, MetaAction
from .pipeline.batch_pipeline import BatchPipeline
from .pipeline.live_pipeline import LivePipeline
from .optimizers.adaptive_optimizer import AdaptiveOptimizer
from .optimizers.backoff_optimizer import BackoffOptimizer
from .optimizers.momentum_optimizer import MomentumOptimizer
from .optimizers.pbt_optimizer import PBTOptimizer
from .optimizers.spsa_optimizer import SPSAOptimizer

__all__ = [
    "EpisodeData", "EvalMetrics", "HyperparamSet", "PolicySnapshot", "LoopState",
    "BaseAgent", "BaseEvaluator", "BaseOptimizer", "BasePipeline",
    "LoopCallback",
    "PolicyManager", "PolicyManagerCallback", "EnsembleAgent",
    "RegimeDetector", "MetaController", "MetaAction",
    "BatchPipeline", "LivePipeline",
    "AdaptiveOptimizer", "BackoffOptimizer", "MomentumOptimizer", "PBTOptimizer", "SPSAOptimizer",
]

from .core.normalizers import RunningMeanStd, ObsNormalizer, RewardNormalizer
from .core.trajectory_buffer import compute_gae, make_minibatches
from .pipeline.vector_pipeline import VectorBatchPipeline

__all__ += [
    "RunningMeanStd", "ObsNormalizer", "RewardNormalizer",
    "compute_gae", "make_minibatches",
    "VectorBatchPipeline",
]

from .orchestrator import TrialOrchestrator
from .exploration.rnd import RNDPipeline
from .core.replay_buffer import PrioritizedReplayBuffer
from .core.diagnostic_controller import DiagnosticController
from .callbacks import WandbCallback, TensorBoardCallback, RichDashboardCallback
from .factory import make_agent
from .core.her_buffer import HERReplayBuffer
from .config import TrainConfig, load_config, apply_overrides, config_to_dict, build_agent_from_config
__all__ += [
    "TrialOrchestrator", "RNDPipeline", "PrioritizedReplayBuffer", "DiagnosticController",
    "WandbCallback", "TensorBoardCallback", "RichDashboardCallback",
    "make_agent", "HERReplayBuffer",
    "TrainConfig", "load_config", "apply_overrides", "config_to_dict", "build_agent_from_config",
]

try:
    from .distributed import AsyncActorLearner, compute_vtrace_targets
    __all__ += ["AsyncActorLearner", "compute_vtrace_targets"]
except (ImportError, RuntimeError):
    pass

try:
    from .adapters.jax.flax_agent    import FlaxAgent
    from .adapters.jax.flax_evaluator import FlaxEvaluator
    from .algorithms.flax_ppo         import FlaxPPOAgent
    __all__ += ["FlaxAgent", "FlaxEvaluator", "FlaxPPOAgent"]
except (ImportError, RuntimeError):
    pass

try:
    from .optimizer import RLOptimizer
    from .adapters.tensorflow.tf_agent import TFAgent
    from .adapters.tensorflow.tf_evaluator import TFEvaluator
    from .algorithms.tf_ppo import TFPPOAgent
    from .algorithms.tf_dqn import TFDQNAgent
    from .algorithms.tf_sac import TFSACAgent
    from .algorithms.tf_ppo_continuous import TFGaussianPPOAgent
    from .algorithms.tf_td3 import TFTDDAgent
    __all__ += ["RLOptimizer", "TFAgent", "TFEvaluator",
                "TFPPOAgent", "TFDQNAgent", "TFSACAgent", "TFGaussianPPOAgent",
                "TFTDDAgent"]
except (ImportError, RuntimeError):
    pass

try:
    from .adapters.pytorch.torch_agent import TorchAgent
    from .adapters.pytorch.torch_evaluator import TorchEvaluator
    from .algorithms.torch_ppo import TorchPPOAgent
    from .algorithms.torch_dqn import TorchDQNAgent
    from .algorithms.torch_sac import TorchSACAgent
    from .algorithms.torch_ppo_continuous import TorchGaussianPPOAgent
    from .algorithms.torch_td3 import TorchTD3Agent
    from .algorithms.torch_recurrent_ppo import TorchRecurrentPPOAgent
    from .algorithms.torch_rainbow_dqn import TorchRainbowDQNAgent, RainbowQNetwork
    from .core.noisy_linear import NoisyLinear
    __all__ += ["TorchAgent", "TorchEvaluator",
                "TorchPPOAgent", "TorchDQNAgent", "TorchSACAgent", "TorchGaussianPPOAgent",
                "TorchTD3Agent", "TorchRecurrentPPOAgent",
                "TorchRainbowDQNAgent", "RainbowQNetwork", "NoisyLinear"]
except (ImportError, RuntimeError):
    pass

try:
    from .neuroevo import (
        NeuronGraph, Edge, Neuron,
        insert_neuron_on_edge, split_neuron, add_input_neuron, add_free_edge,
        prune_edge, prune_neuron, merge_neurons,
        neuron_importance, edge_importance, cosine_similarity_neurons,
        GraphAgent,
        TopologyController,
    )
    __all__ += [
        "NeuronGraph", "Edge", "Neuron",
        "insert_neuron_on_edge", "split_neuron", "add_input_neuron", "add_free_edge",
        "prune_edge", "prune_neuron", "merge_neurons",
        "neuron_importance", "edge_importance", "cosine_similarity_neurons",
        "GraphAgent",
        "TopologyController",
    ]
except (ImportError, RuntimeError):
    pass
