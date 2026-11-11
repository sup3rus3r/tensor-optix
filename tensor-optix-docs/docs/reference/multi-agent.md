# tensor_optix.multi_agent

Pipelines and a minimal runner for [PettingZoo](https://pettingzoo.farama.org/) multi-agent environments. Each *seat* (PettingZoo agent id) is driven by its own ordinary `BaseAgent`; a window yields one `EpisodeData` per seat.

```bash
pip install tensor-optix[torch,pettingzoo]
```

```python
from tensor_optix.multi_agent import (
    SeatEnvView,
    ParallelMultiAgentPipeline,
    AECMultiAgentPipeline,
    MultiAgentRunner,
)
```

`MultiAgentEpisodeData` (in `tensor_optix.multi_agent.multi_agent_episode`) is the alias `Dict[Hashable, EpisodeData]` - the type both pipelines yield.

## SeatEnvView

```python
class SeatEnvView:
    """
    A single seat's view of a PettingZoo env: exposes .observation_space and
    .action_space for one agent id, so make_agent(view) can build that
    seat's agent exactly as it would for a Gymnasium env.

    Discrete observation spaces are presented as a one-hot Box(n,) -
    make_agent() needs a Box observation space.
    """

    def __init__(self, pz_env, agent_id): ...

    def encode_observation(self, raw_observation):
        """One-hot encode a Discrete observation; pass anything else through."""
```

Pass the views to a pipeline's `views=` argument so observations are encoded the same way the agent's network was sized.

## ParallelMultiAgentPipeline

```python
class ParallelMultiAgentPipeline:
    """
    Drives a PettingZoo ParallelEnv: every active seat acts each round.

    window_size: rounds collected per yielded window (each active seat
                 contributes one step per round).
    views:       optional {agent_id: SeatEnvView} for observation encoding.
    """

    def __init__(self, env, agents: Dict[Hashable, Any], window_size: int = 200, views=None): ...
    def set_agents(self, agents) -> None: ...
    def setup(self) -> None: ...
    def episodes(self) -> Generator[Dict[Hashable, EpisodeData], None, None]: ...
    def teardown(self) -> None: ...
```

When every seat is done the env is reset and the window keeps filling, mirroring `BatchPipeline`.

## AECMultiAgentPipeline

```python
class AECMultiAgentPipeline:
    """
    Drives a PettingZoo AECEnv (turn-based, one seat acts per env.step()).

    window_size: TOTAL env.step() calls per window, split across whichever
                 seats take turns during it - not window_size per seat.
    """

    def __init__(self, env, agents: Dict[Hashable, Any], window_size: int = 200, views=None): ...
```

**Reward alignment.** In the AEC API, `env.last()` returns the reward a seat accumulated *since its previous action*. The pipeline credits that reward - and the `terminated`/`truncated` flag observed on the seat's final `action=None` turn - back to the seat's previously recorded step. So within a seat's `EpisodeData`:

- `rewards[k]` is the outcome of `actions[k]`.
- A seat's last acted step of an episode carries the terminal reward with `terminated` (or `truncated`) set, so `dones` marks real episode boundaries for GAE / n-step returns.
- No un-acted-on step is ever appended, so agents' `act()`-populated rollout caches stay in sync with `rewards`.

**Known limitation.** The outcome of a seat's *last* action in a window only arrives on its next turn, which belongs to the next window. That one reward per seat per window boundary is not credited (the step keeps reward `0.0` and no done flag). `final_obs` is `None` for AEC windows - there is no per-seat post-window observation to bootstrap from.

A seat that gets no turn in a window yields an empty `EpisodeData`; skip `learn()` for it (`MultiAgentRunner` does).

## MultiAgentRunner

```python
class MultiAgentRunner:
    """
    Minimal training driver: for each window, call learn() on every seat's
    agent with that seat's EpisodeData (empty seats skipped).
    """

    def __init__(self, pipeline, agents: Dict[Hashable, Any]): ...
    def run(self, max_windows: Optional[int] = None) -> None: ...
```

`MultiAgentRunner` is deliberately small - it does not run the `LoopController` state machine (no checkpointing, hyperparameter tuning, or convergence detection). To get those for one seat, wrap that seat in your own `BasePipeline` and use `RLOptimizer`.

See [Multi-agent training](../guides/multi-agent.md) for a worked example.
