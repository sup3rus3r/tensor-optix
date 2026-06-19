# tensor_optix.config and the CLI

## TrainConfig

```python
@dataclass
class TrainConfig:
    """Fully-specified training configuration."""
    env:       str
    algorithm: str
    framework:    str        = "torch"
    deterministic: bool      = False
    hidden_sizes: List[int]  = field(default_factory=lambda: [256, 256])
    pipeline:    str = "BatchPipeline"
    window_size: int = 2048
    seed: int = 0
    agent:     Dict[str, Any] = field(default_factory=dict)
    optimizer: Dict[str, Any] = field(default_factory=dict)
```

Required keys: `env` (Gymnasium env ID), `algorithm` (one of `TorchPPOAgent`, `TorchSACAgent`, `TorchTD3Agent`, `TorchDQNAgent`, `TorchGaussianPPOAgent`, `TorchRecurrentPPOAgent`, `TFPPOAgent`, `TFSACAgent`, `TFTDDAgent`, `TFDQNAgent`, `TFGaussianPPOAgent`). `framework` is inferred from the algorithm name prefix when omitted.

## Functions

```python
def load_config(path: str, overrides: Optional[List[str]] = None) -> TrainConfig:
    """
    Load a YAML config file and apply optional CLI override strings.

    overrides: List of "key=value" strings. Dotted keys address nested dicts:
        "agent.learning_rate=1e-3". Values are YAML-parsed so true, 1e-3,
        [64,64] all work.

    Raises FileNotFoundError, KeyError (missing required key), or ValueError
    (malformed override / unrecognised algorithm).
    """

def apply_overrides(config_dict: dict, overrides: List[str]) -> dict:
    """
    Apply a list of "dotted.key=value" overrides to a config dict.
    Returns a new dict (does not mutate the input).

    >>> apply_overrides({"agent": {"lr": 1e-4}}, ["agent.lr=1e-3", "seed=42"])
    {'agent': {'lr': 0.001}, 'seed': 42}
    """

def config_to_dict(cfg: TrainConfig) -> dict:
    """Serialise a TrainConfig back to a plain dict (YAML-round-trippable)."""

def build_agent_from_config(cfg: TrainConfig):
    """
    Build the agent and its HyperparamSet from a TrainConfig.
    Uses make_agent() for automatic network construction.
    Returns (agent, hyperparams).
    """

def build_pipeline_from_config(cfg: TrainConfig):
    """Build BatchPipeline or LivePipeline from a TrainConfig."""

def build_optimizer_from_config(cfg: TrainConfig, agent, pipeline):
    """
    Build an RLOptimizer from a TrainConfig + already-constructed components.
    Only passes kwargs that are actually present in cfg.optimizer - raises
    ValueError on unrecognised keys rather than silently ignoring typos.
    """
```

Precedence order: `defaults < config file < CLI overrides`.

## CLI

Registered as the `tensor-optix` console script:

```bash
tensor-optix train config.yaml [key=value ...]
tensor-optix validate config.yaml [key=value ...]
```

`train` loads the config, seeds `random`/`numpy`/`torch` (and CUDA, if available) from `cfg.seed`, builds agent/pipeline/optimizer via the functions above, and calls `optimizer.run()`. Exits `1` with a printed error on a config or build failure rather than a raw traceback.

`validate` parses and validates the config (same error handling) without building or running anything - prints the resolved `env`, `algorithm`, `framework`, `seed`, and any `agent`/`optimizer` sub-dicts. Exits `0` on success, `1` on error - suitable for a CI gate.

See [Configure via YAML / CLI](../guides/cli-config.md) for the full schema and a worked example.
