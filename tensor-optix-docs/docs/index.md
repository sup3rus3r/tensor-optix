# tensor-optix

tensor-optix is a training loop framework with statistical convergence control, online hyperparameter optimisation, and an optional neuroevolution subsystem for dynamic policy topology.

It runs your agent against a pipeline (a Gymnasium environment, a live data stream, or a PyTorch `Dataset`), maintains a separate validation signal, and manages four loop states - `ACTIVE`, `COOLING`, `DORMANT`, and watchdog shutdown - using statistical trend detection rather than a fixed patience counter or step budget. Hyperparameters are tuned online every episode via SPSA gradient estimates. Checkpointing and rollback are driven by the validation signal, never the noisy training score. On `DORMANT`, a `MetaController` decides whether to spawn a policy variant, prune the ensemble, or stop.

## Why

Most RL training scripts hard-code a step budget, a learning rate, and a hope that both were right. tensor-optix replaces that with:

- **Convergence detection** via a corrected t-test on the smoothed score slope plus lag-1 autocorrelation - not "stop after N episodes without improvement."
- **Online hyperparameter tuning** via SPSA (and three other optimizers, auto-routed by an `AdaptiveOptimizer`) - no separate sweep needed for most params.
- **Validation-only checkpointing** - the agent that gets restored at the end of training is the one that generalises, not the one that overfit the training window hardest.
- **An optional neuroevolution subsystem** - a policy whose topology (neurons, edges, recurrent connections) grows and prunes itself live during training, gated by three independent statistical signals.

## Where to go

- New to the library? Start at [Installation](getting-started/installation.md) and the [Quickstart](getting-started/quickstart.md).
- Want to understand the loop before using it? Read [Concepts](getting-started/concepts.md).
- Already know what you want to do? Jump to a [Guide](guides/train-rl-agent.md).
- Looking up a specific class or function? See the [API Reference](reference/index.md).

## Algorithms included

15 agents ship out of the box, all implementing the same six-method `BaseAgent` interface and interchangeable with `RLOptimizer`:

| Framework | Agents |
|---|---|
| PyTorch | PPO, Gaussian PPO, Recurrent PPO, DQN, Rainbow DQN, SAC, TD3 |
| TensorFlow | PPO, Gaussian PPO, DQN, SAC, TD3 |
| JAX / Flax | PPO |

Bring your own algorithm by implementing `BaseAgent` - see [Write a custom agent](guides/custom-agent.md).
