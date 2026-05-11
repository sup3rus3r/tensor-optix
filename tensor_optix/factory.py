"""
tensor_optix.factory — make_agent() auto-selection factory.

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

Usage::

    import gymnasium as gym
    from tensor_optix import make_agent

    env   = gym.make("CartPole-v1")
    agent = make_agent(env)                                  # TorchPPOAgent
    agent = make_agent("PPO", env)                           # same
    agent = make_agent("PPO", env, neuroevo=True)            # GraphAgent + NeuronGraph

    env   = gym.make("LunarLanderContinuous-v3")
    agent = make_agent(env)                                  # TorchSACAgent
    agent = make_agent("SAC", env)                           # same
    agent = make_agent("SAC", env, neuroevo=True)            # GraphAgent + NeuronGraph
    agent = make_agent(env, deterministic=True)              # TorchTD3Agent
    agent = make_agent(env, framework="tf")                  # TFSACAgent

Only flat 1-D observation spaces (gym.spaces.Box with shape=(n,)) are
supported for non-neuroevo paths.  Image observations require custom
network architectures.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from tensor_optix.core.types import HyperparamSet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def make_agent(
    algorithm_or_env,
    env=None,
    framework: str = "torch",
    deterministic: bool = False,
    hidden_sizes: Tuple[int, ...] = (256, 256),
    hyperparams: Optional[HyperparamSet] = None,
    device: str = "auto",
    # Neuroevo options
    neuroevo: bool = False,
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
        Optional :class:`HyperparamSet` override.  When ``None``, sensible
        defaults for the selected algorithm are used.
    device:
        ``"auto"``, ``"cpu"``, or ``"cuda"``.  Passed to the agent; ignored
        for the TF backend (TF manages device placement automatically).

    Returns
    -------
    BaseAgent
        A fully constructed, ready-to-train agent instance.

    Raises
    ------
    NotImplementedError
        When the action space type has no supported algorithm, or when the
        observation space is not a flat 1-D Box.
    """
    # ------------------------------------------------------------------
    # Parse positional overloading: make_agent(env) or make_agent("SAC", env)
    # ------------------------------------------------------------------
    if isinstance(algorithm_or_env, str):
        algorithm = algorithm_or_env.upper()
        if env is None:
            raise ValueError(
                "When algorithm is passed as first argument, env must be provided "
                "as the second argument: make_agent('SAC', env)"
            )
    else:
        env = algorithm_or_env
        algorithm = None

    obs_space = env.observation_space
    act_space = env.action_space

    try:
        from gymnasium.spaces import Box, Discrete, Dict, MultiBinary, MultiDiscrete, Tuple as GymTuple
    except ImportError:
        raise ImportError(
            "gymnasium is required for make_agent(). "
            "Install it with: pip install gymnasium"
        )

    # ------------------------------------------------------------------
    # Validate observation space: must be a flat 1-D Box
    # ------------------------------------------------------------------
    if not isinstance(obs_space, Box):
        raise NotImplementedError(
            f"make_agent() only supports flat Box observation spaces. "
            f"Got: {type(obs_space).__name__}. "
            f"For image or structured observations, build the agent manually."
        )
    if len(obs_space.shape) != 1:
        raise NotImplementedError(
            f"make_agent() only supports 1-D (flat) observation spaces. "
            f"Got shape: {obs_space.shape}. "
            f"For image observations, build the agent manually."
        )

    obs_dim = int(obs_space.shape[0])

    # ------------------------------------------------------------------
    # Neuroevo path — wraps a NeuronGraph in GraphAgent
    # ------------------------------------------------------------------
    if neuroevo:
        if isinstance(act_space, Discrete):
            act_dim = int(act_space.n)
            continuous = False
        elif isinstance(act_space, Box):
            if len(act_space.shape) != 1:
                raise NotImplementedError(
                    f"neuroevo make_agent() only supports 1-D continuous action spaces. "
                    f"Got shape: {act_space.shape}."
                )
            act_dim = int(act_space.shape[0])
            continuous = True
        else:
            raise NotImplementedError(
                f"neuroevo make_agent() does not support {type(act_space).__name__} action spaces."
            )
        return _make_neuroevo_agent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            continuous=continuous,
            graph_in=graph_in,
            graph_hidden=graph_hidden,
            graph_out=graph_out,
            hebbian_lr=hebbian_lr,
            hebbian_decay=hebbian_decay,
            grow_cooldown=grow_cooldown,
            hyperparams=hyperparams,
            device=device,
        )

    # ------------------------------------------------------------------
    # Route by action space
    # ------------------------------------------------------------------
    if isinstance(act_space, Discrete):
        n_actions = int(act_space.n)
        return _make_ppo(obs_dim, n_actions, framework, hidden_sizes, hyperparams, device)

    elif isinstance(act_space, Box):
        if len(act_space.shape) != 1:
            raise NotImplementedError(
                f"make_agent() only supports 1-D continuous action spaces. "
                f"Got shape: {act_space.shape}."
            )
        act_dim = int(act_space.shape[0])
        if deterministic:
            return _make_td3(obs_dim, act_dim, framework, hidden_sizes, hyperparams, device)
        else:
            return _make_sac(obs_dim, act_dim, framework, hidden_sizes, hyperparams, device)

    elif isinstance(act_space, MultiDiscrete):
        raise NotImplementedError(
            f"make_agent() does not support MultiDiscrete action spaces. "
            f"MultiDiscrete requires a factored policy (e.g. multi-head softmax) "
            f"which has no single correct parameterisation. Build the agent manually."
        )

    elif isinstance(act_space, (Dict, GymTuple)):
        raise NotImplementedError(
            f"make_agent() does not support {type(act_space).__name__} action spaces. "
            f"Structured action spaces require a custom policy architecture. "
            f"Build the agent manually."
        )

    elif isinstance(act_space, MultiBinary):
        raise NotImplementedError(
            f"make_agent() does not support MultiBinary action spaces. "
            f"MultiBinary requires a multi-label Bernoulli policy. "
            f"Build the agent manually."
        )

    else:
        raise NotImplementedError(
            f"make_agent() received an unrecognised action space type: "
            f"{type(act_space).__name__}. Build the agent manually."
        )


# ---------------------------------------------------------------------------
# Algorithm builders
# ---------------------------------------------------------------------------

def _make_ppo(obs_dim, n_actions, framework, hidden_sizes, hyperparams, device):
    """Build a PPO agent for discrete action spaces."""
    hp = hyperparams or HyperparamSet(params={
        "learning_rate":  3e-4,
        "clip_ratio":     0.2,
        "entropy_coef":   0.01,
        "vf_coef":        0.5,
        "gamma":          0.99,
        "gae_lambda":     0.95,
        "n_epochs":       10,
        "minibatch_size": 64,
        "max_grad_norm":  0.5,
    }, episode_id=0)

    if framework == "tf":
        return _make_ppo_tf(obs_dim, n_actions, hidden_sizes, hp)
    else:
        return _make_ppo_torch(obs_dim, n_actions, hidden_sizes, hp, device)


def _make_ppo_torch(obs_dim, n_actions, hidden_sizes, hp, device):
    import torch.nn as nn
    from tensor_optix.algorithms.torch_ppo import TorchPPOAgent
    import torch

    def _mlp(in_dim, out_dim, activation=nn.Tanh):
        layers = []
        prev = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), activation()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    actor  = _mlp(obs_dim, n_actions)
    critic = _mlp(obs_dim, 1)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=float(hp.params.get("learning_rate", 3e-4)),
    )
    return TorchPPOAgent(
        actor=actor, critic=critic, optimizer=optimizer,
        hyperparams=hp, device=device,
    )


def _make_ppo_tf(obs_dim, n_actions, hidden_sizes, hp):
    import tensorflow as tf
    from tensor_optix.algorithms.tf_ppo import TFPPOAgent

    def _mlp(out_dim, activation="tanh"):
        layers = [tf.keras.layers.Input(shape=(obs_dim,))]
        for h in hidden_sizes:
            layers.append(tf.keras.layers.Dense(h, activation=activation))
        layers.append(tf.keras.layers.Dense(out_dim))
        return tf.keras.Sequential(layers)

    actor  = _mlp(n_actions)
    critic = _mlp(1)
    optimizer = tf.keras.optimizers.Adam(float(hp.params.get("learning_rate", 3e-4)))
    return TFPPOAgent(
        actor=actor, critic=critic, optimizer=optimizer, hyperparams=hp,
    )


def _make_sac(obs_dim, act_dim, framework, hidden_sizes, hyperparams, device):
    """Build a SAC agent for continuous action spaces."""
    hp = hyperparams or HyperparamSet(params={
        "learning_rate":    3e-4,
        "gamma":            0.99,
        "tau":              0.005,
        "batch_size":       256,
        "updates_per_step": 1,
        "replay_capacity":  1_000_000,
        "per_alpha":        0.0,
        "per_beta":         0.4,
        "n_step":           1,
    }, episode_id=0)

    if framework == "tf":
        return _make_sac_tf(obs_dim, act_dim, hidden_sizes, hp)
    else:
        return _make_sac_torch(obs_dim, act_dim, hidden_sizes, hp, device)


def _make_sac_torch(obs_dim, act_dim, hidden_sizes, hp, device):
    import torch
    import torch.nn as nn
    from tensor_optix.algorithms.torch_sac import TorchSACAgent

    def _mlp(in_dim, out_dim, activation=nn.ReLU):
        layers = []
        prev = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), activation()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    actor   = _mlp(obs_dim,           act_dim * 2)  # mean || log_std
    critic1 = _mlp(obs_dim + act_dim, 1)
    critic2 = _mlp(obs_dim + act_dim, 1)
    log_alpha = torch.zeros(1, requires_grad=True)
    lr = float(hp.params.get("learning_rate", 3e-4))

    return TorchSACAgent(
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        action_dim=act_dim,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=lr),
        critic_optimizer=torch.optim.Adam(
            list(critic1.parameters()) + list(critic2.parameters()), lr=lr
        ),
        alpha_optimizer=torch.optim.Adam([log_alpha], lr=lr),
        hyperparams=hp,
        device=device,
    )


def _make_sac_tf(obs_dim, act_dim, hidden_sizes, hp):
    import tensorflow as tf
    from tensor_optix.algorithms.tf_sac import TFSACAgent

    def _mlp(in_dim, out_dim, activation="relu"):
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(in_dim,)),
            *[tf.keras.layers.Dense(h, activation=activation) for h in hidden_sizes],
            tf.keras.layers.Dense(out_dim),
        ])

    actor   = _mlp(obs_dim,           act_dim * 2)
    critic1 = _mlp(obs_dim + act_dim, 1)
    critic2 = _mlp(obs_dim + act_dim, 1)
    lr = float(hp.params.get("learning_rate", 3e-4))

    return TFSACAgent(
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        action_dim=act_dim,
        actor_optimizer=tf.keras.optimizers.Adam(lr),
        critic_optimizer=tf.keras.optimizers.Adam(lr),
        alpha_optimizer=tf.keras.optimizers.Adam(lr),
        hyperparams=hp,
    )


def _make_td3(obs_dim, act_dim, framework, hidden_sizes, hyperparams, device):
    """Build a TD3 agent for continuous action spaces (deterministic policy)."""
    hp = hyperparams or HyperparamSet(params={
        "learning_rate":     3e-4,
        "gamma":             0.99,
        "tau":               0.005,
        "batch_size":        256,
        "updates_per_step":  1,
        "replay_capacity":   1_000_000,
        "policy_delay":      2,
        "target_noise":      0.2,
        "target_noise_clip": 0.5,
        "per_alpha":         0.0,
        "per_beta":          0.4,
    }, episode_id=0)

    if framework == "tf":
        return _make_td3_tf(obs_dim, act_dim, hidden_sizes, hp)
    else:
        return _make_td3_torch(obs_dim, act_dim, hidden_sizes, hp, device)


def _make_td3_torch(obs_dim, act_dim, hidden_sizes, hp, device):
    import torch
    import torch.nn as nn
    from tensor_optix.algorithms.torch_td3 import TorchTD3Agent

    def _mlp(in_dim, out_dim, activation=nn.ReLU, out_activation=None):
        layers = []
        prev = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), activation()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        if out_activation is not None:
            layers.append(out_activation())
        return nn.Sequential(*layers)

    actor   = _mlp(obs_dim,           act_dim, out_activation=nn.Tanh)
    critic1 = _mlp(obs_dim + act_dim, 1)
    critic2 = _mlp(obs_dim + act_dim, 1)
    lr = float(hp.params.get("learning_rate", 3e-4))

    return TorchTD3Agent(
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        action_dim=act_dim,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=lr),
        critic_optimizer=torch.optim.Adam(
            list(critic1.parameters()) + list(critic2.parameters()), lr=lr
        ),
        hyperparams=hp,
        device=device,
    )


def _make_td3_tf(obs_dim, act_dim, hidden_sizes, hp):
    import tensorflow as tf
    from tensor_optix.algorithms.tf_td3 import TFTDDAgent

    def _mlp(in_dim, out_dim, activation="relu", out_activation=None):
        layers = [
            tf.keras.layers.Input(shape=(in_dim,)),
            *[tf.keras.layers.Dense(h, activation=activation) for h in hidden_sizes],
            tf.keras.layers.Dense(out_dim, activation=out_activation),
        ]
        return tf.keras.Sequential(layers)

    actor   = _mlp(obs_dim,           act_dim, out_activation="tanh")
    critic1 = _mlp(obs_dim + act_dim, 1)
    critic2 = _mlp(obs_dim + act_dim, 1)
    lr = float(hp.params.get("learning_rate", 3e-4))

    return TFTDDAgent(
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        action_dim=act_dim,
        actor_optimizer=tf.keras.optimizers.Adam(lr),
        critic_optimizer=tf.keras.optimizers.Adam(lr),
        hyperparams=hp,
    )


# ---------------------------------------------------------------------------
# Neuroevo builder
# ---------------------------------------------------------------------------

def _make_neuroevo_agent(
    obs_dim, act_dim, continuous,
    graph_in, graph_hidden, graph_out,
    hebbian_lr, hebbian_decay, grow_cooldown,
    hyperparams, device,
):
    """Build a GraphAgent backed by a NeuronGraph with Hebbian + TopologyController."""
    from tensor_optix.neuroevo import (
        NeuronGraph, GraphAgent, HebbianHook, TopologyController,
    )
    from tensor_optix.neuroevo.graph.neuron import TrainableGRUNeuron
    import torch

    # Input projection: cap at 16 for large obs spaces
    n_in = graph_in if graph_in is not None else min(obs_dim, 16)
    # Output: act_dim action neurons + 1 value neuron
    n_out = graph_out if graph_out is not None else act_dim + 1

    graph = NeuronGraph(dale_mode="softplus")

    # Input neurons
    in_neurons = []
    for i in range(n_in):
        nid = graph.add_neuron(role="input", cell_type="excitatory", neuron_id=f"i{i}")
        in_neurons.append(nid)

    # Hidden GRU neurons
    hidden_neurons = []
    for j in range(graph_hidden):
        gru = TrainableGRUNeuron(neuron_id=f"h{j}", cell_type="excitatory")
        nid = graph.add_neuron(role="hidden", neuron=gru)
        hidden_neurons.append(nid)

    # Output neurons
    out_neurons = []
    for k in range(n_out):
        nid = graph.add_neuron(role="output", cell_type="excitatory", neuron_id=f"o{k}")
        out_neurons.append(nid)

    # Wire input → hidden → output
    for src in in_neurons:
        for dst in hidden_neurons:
            graph.add_edge(src, dst, weight=0.1)
    for src in hidden_neurons:
        for dst in out_neurons:
            graph.add_edge(src, dst, weight=0.1)

    _hp = hyperparams or HyperparamSet(params={
        "learning_rate":  3e-4,
        "clip_ratio":     0.2,
        "entropy_coef":   0.01,
        "vf_coef":        0.5,
        "gamma":          0.99,
        "gae_lambda":     0.95,
        "n_epochs":       4,
        "minibatch_size": 64,
        "max_grad_norm":  0.5,
    }, episode_id=0)

    agent = GraphAgent(
        graph=graph,
        obs_dim=n_in,
        n_actions=act_dim,
        continuous=continuous,
        hyperparams=_hp,
        device=device,
    )

    # Attach callbacks that can be used by Optimizer
    agent._neuroevo_callbacks = [
        HebbianHook(graph, lr=hebbian_lr, decay=hebbian_decay),
        TopologyController(graph, grow_cooldown=grow_cooldown),
    ]

    logger.info(
        "make_agent(neuroevo=True): obs_dim=%d → graph_in=%d, hidden=%d (GRU), out=%d, "
        "continuous=%s, dale_mode=softplus",
        obs_dim, n_in, graph_hidden, n_out, continuous,
    )
    return agent
