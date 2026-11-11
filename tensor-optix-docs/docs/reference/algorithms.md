# tensor_optix.algorithms - Algorithms

All agents implement `BaseAgent` and are interchangeable with `RLOptimizer`. Each is built via `make_agent()` with sensible defaults, or constructed manually for custom network architectures.

## PyTorch

| Agent | Algorithm | Action Space |
|---|---|---|
| `TorchPPOAgent` | PPO + GAE-λ | Discrete |
| `TorchGaussianPPOAgent` | PPO | Continuous |
| `TorchRecurrentPPOAgent` | PPO + GRU/LSTM hidden state | Discrete |
| `TorchDQNAgent` | DQN + PER + n-step returns | Discrete |
| `TorchRainbowDQNAgent` | Rainbow DQN (NoisyNet, distributional, PER, n-step, dueling, double) | Discrete |
| `TorchSACAgent` | SAC, twin Q-critics, automatic entropy tuning | Continuous |
| `TorchTD3Agent` | TD3, twin-delayed deterministic policy gradient | Continuous |

## TensorFlow

`TFPPOAgent`, `TFGaussianPPOAgent`, `TFDQNAgent`, `TFSACAgent`, `TFTDDAgent` (TD3) - same algorithmic coverage as the PyTorch agents for the standard (non-Rainbow, non-recurrent) cases.

## JAX / Flax

`FlaxPPOAgent` - PPO via `flax.nnx`.

## Base adapters

Each framework has a thin `*Agent`/`*Evaluator` pair that the algorithm-specific classes build on:

- **`TorchAgent`** / **`TorchEvaluator`** - base PyTorch agent (generic policy-gradient `learn()`, A2C advantage baseline when `values` are present, REINFORCE fallback otherwise) and the default evaluator (mean return of the episodes that completed in the window - split per env for `VectorBatchPipeline` windows - falling back to mean per-step reward when no episode completes in the window).
- **`TFAgent`** / **`TFEvaluator`** - same role for TensorFlow, using `tf.GradientTape`.
- **`FlaxAgent`** / **`FlaxEvaluator`** - same role for Flax NNX models. Weights are serialized via `flax.nnx.to_pure_dict(nnx.state(model))` (a plain nested dict of numpy arrays) rather than raw pickling, since JAX arrays aren't reliably pickle-safe across JAX/XLA versions.

Subclass the relevant base adapter and override `learn()` to wrap a custom architecture in algorithm-specific update logic (PPO clipping, SAC entropy tuning, DQN target updates, etc.) without reimplementing `act()`/`save_weights()`/`load_weights()` from scratch.

## Notable implementation details

- **`TorchRainbowDQNAgent`** combines all six Rainbow improvements: `NoisyLinear` layers (see [NoisyLinear](core/noisy_linear.md)) replace ε-greedy exploration, a distributional (categorical) value head, `PrioritizedReplayBuffer`, n-step returns, dueling architecture, and double-DQN target computation.
- **`TorchRecurrentPPOAgent`** carries GRU/LSTM hidden state across `act()` calls, caches the pre-step hidden state of every step, and trains with truncated BPTT (`bptt_len`), analogous to neuroevo's `RecurrentGraphAgent`. `BatchPipeline` calls its `reset_hidden()` at every episode boundary; when a window ends mid-episode the value target is bootstrapped from the critic on `final_obs` using the live hidden state. It holds a single hidden state, so use it with `BatchPipeline`, not `VectorBatchPipeline`.
- **`TorchTD3Agent`** implements the standard TD3 fixes: target policy smoothing (`target_noise`/`target_noise_clip`), delayed policy updates (`policy_delay`), and twin critics with the min-of-two target - its `_soft_update` does Polyak averaging `θ_target ← τ·θ_source + (1−τ)·θ_target`; critic targets are updated every gradient step, the actor target only on the delayed steps where the actor itself is updated. `act()` is deterministic by default; set the `exploration_noise` hyperparameter (e.g. `0.1`, as in the TD3 paper) to add clipped Gaussian noise to collected actions, and set it back to `0` for evaluation/deployment. The same applies to `TFTDDAgent`.
- **`export_onnx`** is implemented on all Torch agents (continuous agents apply `tanh` internally and export actions clipped to `(-1, 1)` - rescale at deployment) - see [Export to ONNX](../guides/onnx-export.md).
- **Parallel envs**: every built-in agent except `TorchRecurrentPPOAgent` accepts `VectorBatchPipeline` windows - PPO agents run GAE per env, off-policy agents build replay transitions per env (see [Pipelines](pipelines.md#vectorbatchpipeline)).
- **Replay transitions**: off-policy agents store `(s_t, a_t, r_t, s_{t+1}, done_t)` for consecutive steps inside a window; the last step of each env in a window has no in-window successor and is not stored. `done` is `terminated OR truncated`.
- **`is_on_policy`**: `True` for the PPO family (default), `False` for DQN/Rainbow/SAC/TD3 - see [BaseAgent](core/base_agent.md) for why this matters to rollback.

## Auto-selection

See [make_agent](factory.md) for the factory that builds any of these with default networks from just a Gymnasium environment.
