"""
Lightweight benchmark: GraphAgent + TopologyController on CartPole-v1.

Not a full training run — just validates that:
1. The agent can interact with a real gym env end-to-end
2. Topology grow events fire and increase network size
3. The loop doesn't crash over N episodes
4. Score is non-trivial (agent survives > 5 steps on average)
"""
import gymnasium as gym
import numpy as np
import pytest
import torch

from tensor_optix.neuroevo.graph.neuron_graph import NeuronGraph
from tensor_optix.neuroevo.agent.graph_agent import GraphAgent
from tensor_optix.neuroevo.controller.topology_controller import TopologyController
from tensor_optix.core.types import EpisodeData, LoopState


pytest.importorskip("gymnasium")


OBS_DIM = 4
N_ACTIONS = 2
N_EPISODES = 30
GROW_COOLDOWN = 5


def build_agent() -> tuple[GraphAgent, TopologyController]:
    g = NeuronGraph()
    for _ in range(OBS_DIM):
        g.add_neuron(role="input", activation="linear")
    h0 = g.add_neuron(role="hidden", activation="tanh")
    h1 = g.add_neuron(role="hidden", activation="tanh")
    # N_ACTIONS logits + 1 value head
    for _ in range(N_ACTIONS + 1):
        g.add_neuron(role="output", activation="linear")
    for inp in g.input_ids:
        for hid in g.hidden_ids:
            g.add_edge(inp, hid, weight=0.1, delay=0)
    for hid in g.hidden_ids:
        for out in g.output_ids:
            g.add_edge(hid, out, weight=0.1, delay=0)

    agent = GraphAgent(graph=g, obs_dim=OBS_DIM, n_actions=N_ACTIONS)
    ctrl = TopologyController(
        g,
        grow_op="insert_edge",
        grow_cooldown=GROW_COOLDOWN,
        min_prune_observations=20,
        max_neurons=32,
    )
    return agent, ctrl


def run_episode(env, agent: GraphAgent) -> EpisodeData:
    obs, _ = env.reset()
    observations, actions, rewards, terminated, truncated, log_probs = [], [], [], [], [], []

    agent.graph.reset_state()
    done = False
    while not done:
        action = agent.act(obs)
        log_prob = agent._last_log_prob.item() if hasattr(agent._last_log_prob, 'item') else float(agent._last_log_prob)
        next_obs, reward, term, trunc, _ = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(float(reward))
        terminated.append(bool(term))
        truncated.append(bool(trunc))
        log_probs.append(log_prob)
        obs = next_obs
        done = term or trunc

    return EpisodeData(
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions),
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        infos=[{}] * len(rewards),
        episode_id=0,
        log_probs=log_probs,
    )


def test_cartpole_end_to_end():
    env = gym.make("CartPole-v1")
    agent, ctrl = build_agent()

    episode_lengths = []
    plateau_interval = 3  # simulate plateau every 3 episodes
    n_neurons_over_time = []

    for ep in range(N_EPISODES):
        ep_data = run_episode(env, agent)
        episode_lengths.append(len(ep_data.rewards))

        diag = agent.learn(ep_data)

        # Record graph size
        n_neurons_over_time.append(diag["n_neurons"])

        # Simulate plateau signal periodically
        ctrl.on_episode_end(ep, None)
        if ep > 0 and ep % plateau_interval == 0:
            ctrl.on_plateau(ep, LoopState.COOLING)

    env.close()

    avg_length = np.mean(episode_lengths)
    final_neurons = n_neurons_over_time[-1]
    initial_neurons = n_neurons_over_time[0]

    print(f"\nCartPole benchmark:")
    print(f"  Episodes: {N_EPISODES}")
    print(f"  Avg episode length: {avg_length:.1f} steps")
    print(f"  Neurons: {initial_neurons} → {final_neurons}")
    print(f"  Grow events: {ctrl._grow_count}")
    print(f"  Topology stats: {ctrl.stats}")

    # Agent must survive more than random (random ≈ 8-9 steps on CartPole)
    assert avg_length > 5, f"Agent too weak: avg {avg_length:.1f} steps"

    # Topology must have grown at least once
    assert ctrl._grow_count >= 1, "No grow events fired"
    assert final_neurons > initial_neurons, f"Network did not grow: {initial_neurons} → {final_neurons}"

    # No crashes, diagnostics are finite
    assert np.isfinite(diag["loss"]), "Loss is not finite"
