"""
tensor_optix.config — YAML configuration loading for tensor-optix training runs.

A training run is parameterized by ~20 values.  Storing them in Python code:
  1. Is not serializable without executing code
  2. Makes diffs between two runs require diffing Python files
  3. Forces programmatic code modification for parameter sweeps

This module maps a declarative YAML file to the exact Python objects needed
to construct an agent, pipeline, and RLOptimizer.  The mapping is 1:1 and
lossless — every YAML key maps directly to a constructor kwarg, and every
kwarg has an explicit YAML key.  No hidden defaults, no implicit behavior.

Precedence order (standard for all CLI tools):

    defaults  <  config file  <  CLI overrides

CLI overrides use dotted-key notation:
    agent.learning_rate=1e-3
    optimizer.max_episodes=200
    seed=42

Schema
------
Required keys:
    env       — gymnasium env ID string (e.g. "CartPole-v1")
    algorithm — one of: TorchPPOAgent, TorchSACAgent, TorchTD3Agent,
                        TorchDQNAgent, TFPPOAgent, TFSACAgent, TFTDDAgent

Optional keys (with defaults):
    framework    — "torch" | "tf"  (default: inferred from algorithm name)
    deterministic — bool (default: false, affects SAC vs TD3 auto-select)
    pipeline     — "BatchPipeline" | "LivePipeline"  (default: "BatchPipeline")
    window_size  — int  (default: 2048, BatchPipeline only)
    seed         — int  (default: 0)
    hidden_sizes — list of ints  (default: [256, 256])

    agent:       — dict of hyperparameter key→value pairs (forwarded as
                   HyperparamSet.params to the agent)

    optimizer:   — dict of RLOptimizer constructor kwargs
                   (max_episodes, checkpoint_dir, rollback_on_degradation,
                    verbose, plateau_threshold, dormant_threshold, ...)

Example::

    # config.yaml
    env: CartPole-v1
    algorithm: TorchPPOAgent
    seed: 42

    agent:
      learning_rate: 3e-4
      clip_ratio: 0.2
      gamma: 0.99
      n_epochs: 10
      minibatch_size: 64

    optimizer:
      max_episodes: 300
      verbose: true
      checkpoint_dir: ./checkpoints
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Config dataclass — pure data, no side effects
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """
    Fully-specified training configuration.

    All fields that map to constructor kwargs are typed explicitly.
    Sub-dicts (``agent``, ``optimizer``) are passed through verbatim.
    """
    # Required
    env:       str
    algorithm: str

    # Framework / network
    framework:    str        = "torch"
    deterministic: bool      = False
    hidden_sizes: List[int]  = field(default_factory=lambda: [256, 256])

    # Pipeline
    pipeline:    str = "BatchPipeline"
    window_size: int = 2048

    # Reproducibility
    seed: int = 0

    # Sub-dicts forwarded verbatim to HyperparamSet and RLOptimizer
    agent:     Dict[str, Any] = field(default_factory=dict)
    optimizer: Dict[str, Any] = field(default_factory=dict)

    # Inferred at load time (not settable in YAML)
    _source_path: Optional[str] = field(default=None, repr=False, compare=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: str, overrides: Optional[List[str]] = None) -> TrainConfig:
    """
    Load a YAML config file and apply optional CLI override strings.

    Parameters
    ----------
    path:
        Path to a ``.yaml`` / ``.yml`` config file.
    overrides:
        List of ``"key=value"`` strings from the CLI.
        Dotted keys address nested dicts: ``"agent.learning_rate=1e-3"``.
        Values are YAML-parsed (so ``true``, ``1e-3``, ``[64,64]`` all work).

    Returns
    -------
    TrainConfig

    Raises
    ------
    FileNotFoundError  — path doesn't exist
    KeyError           — a required key is missing from the config
    ValueError         — an override string is malformed or the algorithm is
                         not recognised
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for YAML config support. "
            "Install it with: pip install pyyaml"
        )

    with open(path, "r") as f:
        raw: dict = yaml.safe_load(f) or {}

    # Apply CLI overrides on top of the YAML
    if overrides:
        raw = apply_overrides(raw, overrides)

    return _dict_to_config(raw, source_path=path)


def apply_overrides(config_dict: dict, overrides: List[str]) -> dict:
    """
    Apply a list of ``"dotted.key=value"`` overrides to a config dict.

    Returns a new dict (does not mutate the input).
    Values are YAML-parsed so numeric types, booleans, and lists work
    correctly (e.g. ``seed=42``, ``optimizer.verbose=true``).

    Examples
    --------
    >>> apply_overrides({"agent": {"lr": 1e-4}}, ["agent.lr=1e-3", "seed=42"])
    {'agent': {'lr': 0.001}, 'seed': 42}
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required: pip install pyyaml")

    result = copy.deepcopy(config_dict)
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid override {override!r}. "
                f"Expected format: 'key=value' or 'section.key=value'."
            )
        key_path, _, raw_val = override.partition("=")
        # Parse value using YAML so "true", "1e-3", "[64,64]" all deserialise.
        # PyYAML 1.1 does not parse bare scientific notation without a leading
        # digit ("1e-3" stays as string; "1.0e-3" is parsed as float).  Fall
        # back to Python float() for any string that looks numeric so that
        # "1e-3", "3e-4", "5e-4" etc. are always coerced to float.
        value = yaml.safe_load(raw_val)
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass  # keep as string (e.g. path names)
        _set_nested(result, key_path.split("."), value)

    return result


def config_to_dict(cfg: TrainConfig) -> dict:
    """
    Serialise a TrainConfig back to a plain dict (YAML-round-trippable).
    Excludes private fields (_source_path).
    """
    d = {
        "env":           cfg.env,
        "algorithm":     cfg.algorithm,
        "framework":     cfg.framework,
        "deterministic": cfg.deterministic,
        "hidden_sizes":  cfg.hidden_sizes,
        "pipeline":      cfg.pipeline,
        "window_size":   cfg.window_size,
        "seed":          cfg.seed,
        "agent":         copy.deepcopy(cfg.agent),
        "optimizer":     copy.deepcopy(cfg.optimizer),
    }
    return d


# ---------------------------------------------------------------------------
# Agent / pipeline / optimizer construction helpers
# ---------------------------------------------------------------------------

def build_agent_from_config(cfg: TrainConfig):
    """
    Build the agent and its HyperparamSet from a TrainConfig.

    Uses ``make_agent()`` for automatic network construction when the
    algorithm name matches the factory's supported types.
    Returns ``(agent, hyperparams)`` where hyperparams already contains
    all values from ``cfg.agent``.
    """
    import gymnasium as gym
    from tensor_optix.core.types import HyperparamSet
    from tensor_optix.factory import make_agent

    env = gym.make(cfg.env)
    hp  = HyperparamSet(params=dict(cfg.agent), episode_id=0)

    agent = make_agent(
        env,
        framework=cfg.framework,
        deterministic=cfg.deterministic,
        hidden_sizes=tuple(cfg.hidden_sizes),
        hyperparams=hp if cfg.agent else None,
    )
    env.close()
    return agent, hp


def build_pipeline_from_config(cfg: TrainConfig):
    """Build the pipeline from a TrainConfig."""
    import gymnasium as gym
    from tensor_optix.pipeline.batch_pipeline import BatchPipeline
    from tensor_optix.pipeline.live_pipeline import LivePipeline

    env = gym.make(cfg.env)
    if cfg.pipeline == "BatchPipeline":
        return BatchPipeline(env, window_size=cfg.window_size)
    elif cfg.pipeline == "LivePipeline":
        return LivePipeline(env)
    else:
        raise ValueError(
            f"Unknown pipeline {cfg.pipeline!r}. "
            f"Valid options: 'BatchPipeline', 'LivePipeline'."
        )


def build_optimizer_from_config(cfg: TrainConfig, agent, pipeline):
    """
    Build an RLOptimizer from a TrainConfig + already-constructed components.
    Only passes kwargs that are actually in cfg.optimizer — does not invent
    defaults beyond what RLOptimizer already provides.
    """
    from tensor_optix.optimizer import RLOptimizer

    # RLOptimizer's scalar kwargs (everything except agent, pipeline, evaluator,
    # callbacks, factories — those require Python objects, not YAML values).
    _VALID_OPTIMIZER_KWARGS = {
        "checkpoint_dir", "max_snapshots", "rollback_on_degradation",
        "improvement_margin", "max_episodes", "base_interval",
        "backoff_factor", "max_interval_episodes", "plateau_threshold",
        "dormant_threshold", "degradation_threshold", "min_degradation_drop",
        "noise_k", "score_window", "trend_window", "min_episodes_before_dormant",
        "min_episodes_before_degradation", "score_smoothing", "verbose",
        "verbose_log_file", "diag_loss_spike_factor", "diag_entropy_floor",
        "diag_target_kl", "diag_epsilon_patience", "diag_epsilon_reset_value",
        "diag_epsilon_score_threshold", "diag_min_episodes",
        "min_consecutive_degradations", "convergence_patience",
        "cv_threshold", "gap_threshold", "target_score",
    }

    opt_kwargs = {
        k: v for k, v in cfg.optimizer.items()
        if k in _VALID_OPTIMIZER_KWARGS
    }
    unknown = set(cfg.optimizer) - _VALID_OPTIMIZER_KWARGS
    if unknown:
        raise ValueError(
            f"Unknown optimizer keys in config: {sorted(unknown)}. "
            f"Valid keys: {sorted(_VALID_OPTIMIZER_KWARGS)}"
        )

    return RLOptimizer(agent=agent, pipeline=pipeline, **opt_kwargs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {"env", "algorithm"}

_KNOWN_ALGORITHMS = {
    "TorchPPOAgent", "TorchSACAgent", "TorchTD3Agent", "TorchDQNAgent",
    "TorchGaussianPPOAgent", "TorchRecurrentPPOAgent",
    "TFPPOAgent", "TFSACAgent", "TFTDDAgent", "TFDQNAgent",
    "TFGaussianPPOAgent",
}


def _dict_to_config(raw: dict, source_path: Optional[str] = None) -> TrainConfig:
    """Validate and convert a raw dict to a TrainConfig."""
    missing = _REQUIRED_KEYS - set(raw)
    if missing:
        raise KeyError(
            f"Config is missing required key(s): {sorted(missing)}. "
            f"Add them to your YAML file."
        )

    algo = raw["algorithm"]
    if algo not in _KNOWN_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm {algo!r}. "
            f"Known algorithms: {sorted(_KNOWN_ALGORITHMS)}"
        )

    # Infer framework from algorithm name if not given
    framework = raw.get("framework")
    if framework is None:
        framework = "tf" if algo.startswith("TF") else "torch"

    cfg = TrainConfig(
        env           = str(raw["env"]),
        algorithm     = algo,
        framework     = framework,
        deterministic = bool(raw.get("deterministic", False)),
        hidden_sizes  = list(raw.get("hidden_sizes", [256, 256])),
        pipeline      = str(raw.get("pipeline", "BatchPipeline")),
        window_size   = int(raw.get("window_size", 2048)),
        seed          = int(raw.get("seed", 0)),
        agent         = dict(raw.get("agent", {})),
        optimizer     = dict(raw.get("optimizer", {})),
        _source_path  = source_path,
    )
    return cfg


def _set_nested(d: dict, keys: List[str], value: Any) -> None:
    """Set d[keys[0]][keys[1]]...[keys[-1]] = value, creating dicts as needed."""
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value
