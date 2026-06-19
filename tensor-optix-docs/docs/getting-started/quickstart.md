# Quickstart

## Discrete action space (CartPole)

```python
import gymnasium as gym
from tensor_optix import make_agent, Optimizer

env = gym.make("CartPole-v1")
agent = make_agent(env)            # -> TorchPPOAgent (discrete default)
opt = Optimizer(agent, env)
opt.run()
```

`make_agent` inspects `env.action_space` and `env.observation_space` and returns a fully constructed agent with sensible default hyperparameters and network sizes. `Optimizer` wraps `RLOptimizer` with auto-computed `window_size`, automatic SPSA hyperparameter tuning (when the agent declares `default_param_bounds`), and neuroevo callback wiring.

## Continuous action space

```python
env = gym.make("LunarLanderContinuous-v3")
agent = make_agent(env)                      # -> TorchSACAgent (continuous default)
agent = make_agent(env, deterministic=True)  # -> TorchTD3Agent
opt = Optimizer(agent, env)
opt.run()
```

## Choosing an algorithm explicitly

```python
agent = make_agent("DQN", env)
agent = make_agent("RAINBOW", env)
agent = make_agent("SAC", env, framework="tf")
```

## Parallel environments

```python
opt = Optimizer(agent, lambda: gym.make("CartPole-v1"), n_envs=8)
opt.run()
```

When `n_envs > 1`, pass `env` as a zero-arg callable (or a list of callables) rather than a constructed environment - each subprocess/sync worker needs to build its own instance.

## What you get back

```python
opt.run()
opt.best_snapshot   # PolicySnapshot: best weights + EvalMetrics + HyperparamSet
```

When `run()` returns - whether because of convergence, a manual `stop()`, or `max_episodes` - the agent's weights are always the best known checkpoint, not the last one trained. See [Concepts](concepts.md) for why.

## Next steps

- [Concepts](concepts.md) - understand the loop states and why checkpointing is validation-driven.
- [Train an RL agent](../guides/train-rl-agent.md) - the full `RLOptimizer` constructor, validation pipelines, and callbacks.
- [Build a neuroevo agent](../guides/neuroevo.md) - policies that grow and prune their own topology.
