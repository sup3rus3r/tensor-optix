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
    "BackoffOptimizer", "MomentumOptimizer", "PBTOptimizer", "SPSAOptimizer",
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
from .callbacks import WandbCallback, TensorBoardCallback
__all__ += [
    "TrialOrchestrator", "RNDPipeline", "PrioritizedReplayBuffer", "DiagnosticController",
    "WandbCallback", "TensorBoardCallback",
]

try:
    from .optimizer import RLOptimizer
    from .adapters.tensorflow.tf_agent import TFAgent
    from .adapters.tensorflow.tf_evaluator import TFEvaluator
    from .algorithms.tf_ppo import TFPPOAgent
    from .algorithms.tf_dqn import TFDQNAgent
    from .algorithms.tf_sac import TFSACAgent
    from .algorithms.tf_ppo_continuous import TFGaussianPPOAgent
    __all__ += ["RLOptimizer", "TFAgent", "TFEvaluator",
                "TFPPOAgent", "TFDQNAgent", "TFSACAgent", "TFGaussianPPOAgent"]
except (ImportError, RuntimeError):
    pass

try:
    from .adapters.pytorch.torch_agent import TorchAgent
    from .adapters.pytorch.torch_evaluator import TorchEvaluator
    from .algorithms.torch_ppo import TorchPPOAgent
    from .algorithms.torch_dqn import TorchDQNAgent
    from .algorithms.torch_sac import TorchSACAgent
    from .algorithms.torch_ppo_continuous import TorchGaussianPPOAgent
    __all__ += ["TorchAgent", "TorchEvaluator",
                "TorchPPOAgent", "TorchDQNAgent", "TorchSACAgent", "TorchGaussianPPOAgent"]
except (ImportError, RuntimeError):
    pass
