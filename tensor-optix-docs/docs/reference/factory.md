# tensor_optix.factory - make_agent

```
The action space type is a mathematical property that *determines* the valid
policy parameterisation.  This factory makes that mapping explicit and
enforces it:

    Discrete(n)     →  TorchPPOAgent / TFPPOAgent
                       (categorical π(a|s) = softmax(logits))
    Box(shape)      →  TorchSACAgent / TFSACAgent          [default]
                       TorchTD3Agent / TFTDDAgent           [deterministic=True]
    MultiDiscrete   →  NotImplementedError
    Dict / Tuple    →  NotImplementedError

When neuroevo=True the policy is a NeuronGraph wrapped in GraphAgent (PPO-based).
The algorithm argument in that case influences hyperparameter defaults only.
```

## make_agent

```python
def make_agent(
    algorithm_or_env,
    env=None,
    algorithm: Optional[str] = None,
    framework: str = "torch",
    deterministic: bool = False,
    hidden_sizes: Tuple[int, ...] = (256, 256),
    hyperparams: Optional[HyperparamSet] = None,
    device: str = "auto",
    # Neuroevo options
    neuroevo: bool = False,
    neuroevo_mode: str = "policy",   # "policy" | "feature_extractor"
    graph_in: Optional[int] = None,
    graph_hidden: int = 8,
    graph_out: Optional[int] = None,
    hebbian_lr: float = 1e-3,
    hebbian_decay: float = 1e-4,
    grow_cooldown: int = 20,
):
    """
    Inspect *env* and return a fully-constructed agent with default networks.

    Parameters
    ----------
    env:
        A Gymnasium environment (or any object with ``.observation_space``
        and ``.action_space`` attributes).
    framework:
        ``"torch"`` (default) or ``"tf"``.
    deterministic:
        Only relevant for continuous (Box) action spaces.  When ``True``
        returns TD3 (deterministic policy); when ``False`` returns SAC
        (stochastic policy with entropy regularisation).
    hidden_sizes:
        Hidden layer widths for all networks.  Default: ``(256, 256)``.
    hyperparams:
        Optional HyperparamSet override.  When ``None``, sensible
        defaults for the selected algorithm are used.
    device:
        ``"auto"``, ``"cpu"``, or ``"cuda"``.  Passed to the agent; ignored
        for the TF backend (TF manages device placement automatically).

    Returns
    -------
    BaseAgent - a fully constructed, ready-to-train agent instance.

    Raises
    ------
    NotImplementedError
        When the action space type has no supported algorithm, or when the
        observation space is not a flat 1-D Box.
    """
```

### Positional overloading

```python
agent = make_agent(env)            # algorithm inferred from action space
agent = make_agent("SAC", env)     # algorithm explicit
```

### Routing table

| Action space | `algorithm=` | Result |
|---|---|---|
| `Discrete(n)` | (none) | `TorchPPOAgent` / `TFPPOAgent` |
| `Discrete(n)` | `"DQN"` | `TorchDQNAgent` |
| `Discrete(n)` | `"RAINBOW"` | `TorchRainbowDQNAgent` |
| `Box(shape)`, 1-D | (none), `deterministic=False` | `TorchSACAgent` / `TFSACAgent` |
| `Box(shape)`, 1-D | `deterministic=True` or `"TD3"` | `TorchTD3Agent` / `TFTDDAgent` |
| `MultiDiscrete` | - | `NotImplementedError` - no single correct factored-policy parameterisation |
| `Dict` / `Tuple` | - | `NotImplementedError` - requires a custom policy architecture |
| `MultiBinary` | - | `NotImplementedError` - requires a multi-label Bernoulli policy |

Observation space **must** be a flat 1-D `Box` for all non-neuroevo paths - image or structured observations require building the agent manually.

### Neuroevo path

When `neuroevo=True`:

- `neuroevo_mode="policy"` (default) - builds a `NeuronGraph` with `graph_in` input neurons (default `min(obs_dim, 16)`), `graph_hidden` `TrainableGRUNeuron`s, and `graph_out` output neurons (default `act_dim + 1`, last is the value head), fully connected input→hidden→output with weight `0.1`, `dale_mode="softplus"`. Wraps it in `GraphAgent`. Attaches `agent._neuroevo_callbacks = [HebbianHook(...), TopologyController(...)]`, which `Optimizer`/`Optimizer`-style wiring picks up automatically.
- `neuroevo_mode="feature_extractor"` - builds a smaller graph (orthogonal projection `obs_dim → graph_in`, `TrainableGRUNeuron` "sensory" layer, `TrainableLSTMNeuron` "memory" layer, `tanh` output layer) whose output is concatenated with the raw observation before a `TorchSACAgent`'s actor/critic networks. Attaches `_neuroevo_callbacks = [TopologyController, HebbianHook, NeuromodulatorSignal]`.

### Default hyperparameters by algorithm

| Algorithm | Defaults |
|---|---|
| PPO | `learning_rate=3e-4, clip_ratio=0.2, entropy_coef=0.01, vf_coef=0.5, gamma=0.99, gae_lambda=0.95, n_epochs=10, minibatch_size=64, max_grad_norm=0.5` |
| DQN | `learning_rate=1e-3, gamma=0.99, epsilon=1.0→0.05 (decay 0.995), batch_size=64, target_update_freq=10, replay_capacity=10_000` |
| Rainbow | `learning_rate=6.25e-5, gamma=0.99, batch_size=32, target_update_freq=200, replay_capacity=100_000, per_alpha=0.5, per_beta=0.4, n_step=3, v_min=0, v_max=500, n_atoms=51` |
| SAC | `learning_rate=3e-4, gamma=0.99, tau=0.005, batch_size=256, updates_per_step=1, replay_capacity=1_000_000` |
| TD3 | `learning_rate=3e-4, gamma=0.99, tau=0.005, batch_size=256, policy_delay=2, target_noise=0.2, target_noise_clip=0.5` |

See [Algorithms](algorithms.md) for the agent classes themselves.
