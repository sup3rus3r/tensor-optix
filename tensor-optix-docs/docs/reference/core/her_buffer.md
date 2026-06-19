# tensor_optix.core.her_buffer

```
HERReplayBuffer - Hindsight Experience Replay for goal-conditioned RL.

Reference: Andrychowicz et al. 2017 - "Hindsight Experience Replay"
           https://arxiv.org/abs/1707.01495

Design
------
HER wraps any PrioritizedReplayBuffer and adds a relabeling layer:

    store_episode(obs, acts, rews, next_obs, dones, achieved_goals, compute_reward)

For each transition t in an episode, it stores:

    1. The original transition  (obs[t], a[t], r[t], obs[t+1], done[t])
    2. k relabeled transitions  (obs_g'[t], a[t], r_g'[t], obs_g'[t+1], done[t])
       where g' is sampled from the episode according to `strategy` and
       r_g' = compute_reward(achieved_goals[t], g', {}).

obs_g'[t] = concat([env_obs_part(obs[t]), g'])  - the desired goal component
             of the observation is replaced with g'.

Relabeling strategies
---------------------
future  (default, best empirically):
    g' ~ Uniform({achieved_goals[t'], t' ∈ [t, T-1]})
    Uses the achieved goal of a future state in the same episode.
    On average k * T additional transitions per episode.

final:
    g' = achieved_goals[T-1]  (the final achieved state of the episode)
    Minimal variance; always uses the episode's terminal state.

episode:
    g' ~ Uniform({achieved_goals[i], i ∈ [0, T-1]})
    Samples uniformly from all achieved goals in the episode.
```

## HERReplayBuffer

```python
class HERReplayBuffer:
    """
    Hindsight Experience Replay buffer.

    Wraps a PrioritizedReplayBuffer; adds episode-level relabeling before
    transitions are committed to storage.

    Parameters
    ----------
    inner:
        The underlying storage buffer.  HER adds relabeled transitions on top
        of the original transitions.
    k:
        Number of HER goals sampled per transition.  Default 4 (as in the
        original paper).  Total transitions per episode = T * (1 + k).
    strategy:
        Goal sampling strategy.  One of "future" (default), "final", "episode".
    """

    VALID_STRATEGIES = frozenset({"future", "final", "episode"})

    def __init__(self, inner: PrioritizedReplayBuffer, k: int = 4, strategy: str = "future"): ...

    def store_episode(
        self,
        obs_list, act_list, rew_list, next_obs_list, done_list,
        achieved_goals, compute_reward,
    ) -> None:
        """
        Store one full episode, with k HER-relabeled transitions per step.

        Parameters
        ----------
        obs_list:
            Observations at each step.  Shape [T, env_obs_dim + goal_dim].
            The last ``goal_dim`` elements are the desired goal (concatenated
            by the pipeline).
        act_list:      Actions.  Shape [T, act_dim].
        rew_list:       Original rewards from the environment.  Shape [T].
        next_obs_list:  Next observations.  Shape [T, env_obs_dim + goal_dim].
        done_list:      Done flags.  Shape [T].  1.0 = episode over.
        achieved_goals: Achieved goals **after** each transition.
            ``achieved_goals[t]`` = ``next_obs["achieved_goal"]`` at step t.
            Shape [T, goal_dim].
        compute_reward: ``env.compute_reward(achieved_goal, desired_goal, info) -> float``.
        """

    def sample(self, batch_size: int):
        """Sample a batch; delegates to the inner PrioritizedReplayBuffer."""

    def update_priorities(self, indices, errors) -> None: ...
    def __len__(self) -> int: ...
```

### Usage

```python
from tensor_optix.core.replay_buffer import PrioritizedReplayBuffer
from tensor_optix.core.her_buffer import HERReplayBuffer

inner = PrioritizedReplayBuffer(capacity=1_000_000, alpha=0.0)
her   = HERReplayBuffer(inner, k=4, strategy="future")

# At end of each episode:
her.store_episode(
    obs_list, act_list, rew_list, next_obs_list, done_list,
    achieved_goals,   # shape [T, goal_dim] - achieved goal AFTER each step
    compute_reward,   # env.compute_reward(achieved, desired, info) -> float
)

# Inside agent._update_step():
obs_b, act_b, rew_b, next_b, done_b, w, idx, n = her.sample(batch_size)
```

Agents do **not** call `push()` per step when using HER - the pipeline collects full episodes and calls `store_episode()` once per episode instead.
