# Configure via YAML / CLI

For reproducible runs and parameter sweeps without touching Python code, define a training run as a YAML file.

## Config schema

```yaml
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
```

Required keys: `env` (a Gymnasium env ID string) and `algorithm` (one of `TorchPPOAgent`, `TorchSACAgent`, `TorchTD3Agent`, `TorchDQNAgent`, `TorchGaussianPPOAgent`, `TorchRecurrentPPOAgent`, `TFPPOAgent`, `TFSACAgent`, `TFTDDAgent`, `TFDQNAgent`, `TFGaussianPPOAgent`).

Optional keys: `framework` (inferred from the algorithm name prefix if omitted), `deterministic`, `pipeline` (`"BatchPipeline"` or `"LivePipeline"`), `window_size`, `hidden_sizes`. The `agent` and `optimizer` sub-dicts are forwarded **verbatim** - every YAML key maps 1:1 to a constructor kwarg, with no hidden defaults.

## Running from the CLI

```bash
tensor-optix train config.yaml seed=42 agent.learning_rate=1e-3
tensor-optix validate config.yaml
```

CLI overrides use dotted-key notation and take precedence over the file. Precedence order: `defaults < config file < CLI overrides`.

`validate` parses and checks the config without running training - useful in CI to catch a typo'd algorithm name or missing required key before launching a job. It exits `0` on success, `1` on error.

## Loading config from Python

```python
from tensor_optix.config import load_config, build_agent_from_config, build_pipeline_from_config, build_optimizer_from_config

cfg = load_config("config.yaml", overrides=["seed=42", "agent.learning_rate=1e-3"])
agent, hyperparams = build_agent_from_config(cfg)
pipeline = build_pipeline_from_config(cfg)
optimizer = build_optimizer_from_config(cfg, agent, pipeline)
optimizer.run()
```

`apply_overrides(config_dict, overrides)` and `config_to_dict(cfg)` are also exported for programmatic config manipulation - e.g. writing a sweep script that generates many config dicts and calls `apply_overrides` per trial.

Override values are YAML-parsed so `true`, `1e-3`, and `[64,64]` all deserialize correctly. A `float()` fallback coerces strings that look numeric (e.g. PyYAML 1.1 doesn't auto-parse bare scientific notation like `1e-3` without special-casing it).

## Reference

Full schema and function signatures: [Config / CLI reference](../reference/config-cli.md).
