# Installation

tensor-optix requires Python ≥ 3.11.

```bash
# Core loop only, no algorithm implementations
pip install tensor-optix

# PyTorch algorithms
pip install tensor-optix[torch]

# TensorFlow algorithms
pip install tensor-optix[tensorflow]

# JAX/Flax
pip install tensor-optix[jax]

# Neuroevo subsystem (requires torch)
pip install tensor-optix[neuroevo]

# GPU (Linux/WSL2, CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install tensor-optix[torch]

# All frameworks and environment extras
pip install tensor-optix[all]
```

## Optional extras

| Extra | Adds |
|---|---|
| `[box2d]` | Box2D Gymnasium environments (LunarLander, BipedalWalker) |
| `[atari]` | Atari environments via `ale-py` |
| `[mujoco]` | MuJoCo environments |
| `[wandb]` | `WandbCallback` |
| `[tensorboard]` | `TensorBoardCallback` |
| `[onnx]` | `export_onnx()` on Torch agents |
| `[dev]` | pytest, ruff, mypy, black, twine - for contributing |

## Core dependencies

The base install pulls in:

- `gymnasium[box2d] >= 1.0.0`
- `numpy >= 1.24.0`
- `matplotlib >= 3.7.0`
- `optuna >= 3.0.0` (used by `TrialOrchestrator`)
- `pyyaml >= 6.0` (used by the YAML config loader)
- `torch >= 2.11.0`

The core loop, `PolicyManager`, and all ensemble/evolution logic have **no** framework dependency beyond this - TensorFlow and JAX are opt-in via their respective extras.

## CLI

Installing the package registers a `tensor-optix` command:

```bash
tensor-optix train config.yaml seed=42 agent.learning_rate=1e-3
tensor-optix validate config.yaml
```

See [Configure via YAML / CLI](../guides/cli-config.md).
