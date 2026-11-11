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
    Trains every seat and applies, per seat, the convergence and
    checkpoint machinery of the single-agent loop.
    """

    def __init__(
        self,
        pipeline,
        agents: Dict[Hashable, Any],
        *,
        evaluator=None,                    # BaseEvaluator, or {agent_id: BaseEvaluator}
        checkpoint_dir: Optional[str] = None,
        max_snapshots: int = 10,
        improvement_margin: float = 0.0,
        plateau_threshold: int = 5,
        dormant_threshold: int = 20,
        min_windows_before_dormant: int = 0,
        min_completed_episodes_before_converged: int = 10,
        noise_k: float = 2.0,
        score_window: int = 20,
        trend_window: int = 8,
        stop_when: str = "all",            # "all" | "any"
        stop_on_convergence: bool = True,
        restore_best: bool = True,
        callbacks: Optional[List[Any]] = None,
    ): ...

    def run(self, max_windows: Optional[int] = None) -> MultiAgentRunResult: ...
    def stop(self) -> None: ...
    def add_callback(self, callback) -> None: ...

    @property
    def states(self) -> Dict[Hashable, LoopState]: ...
    @property
    def best_scores(self) -> Dict[Hashable, Optional[float]]: ...
    @property
    def best_snapshots(self) -> Dict[Hashable, Optional[PolicySnapshot]]: ...
```

Each window, for every seat that took at least one step:

1. `agent.learn(episode_data)`.
2. The window is scored by `evaluator` (default: mean return of the episodes that completed in the window - the same default the single-agent loop uses).
3. **Checkpoint.** If the score beats the seat's best by more than `improvement_margin`, the seat is saved to its own `CheckpointRegistry` under `<checkpoint_dir>/<agent_id>`. With `checkpoint_dir=None` best scores are still tracked but nothing is written or restored.
4. **Convergence.** The score feeds the seat's own `BackoffScheduler`. A seat whose score trend stays flat for `plateau_threshold` consecutive windows is `COOLING`; after `dormant_threshold` it is `DORMANT`, i.e. converged. A later upward trend puts it back to `ACTIVE`. Every window is evaluated (no eval-interval backoff).

A seat with no turn in a window is skipped entirely - not learned, not scored, and not counted toward its plateau counter.

**Scores are finished-episode returns.** The pipelines track each seat's return across windows. A window in which a seat finished no episode still trains that seat but is not scored (no checkpoint, no plateau count). Until a seat's first episode finishes its windows are scored by mean per-step reward; when it does, that seat's best score and trend are reset because the scale changes. A seat cannot plateau or converge before `min_completed_episodes_before_converged` (default `10`) episodes have finished - set it to `0` for environments whose episodes never end. `SeatResult.completed_episodes` reports the count.

**When the run stops**

| Condition | `stop_reason` |
|---|---|
| every seat converged (`stop_when="all"`, default) or the first seat converged (`stop_when="any"`) | `"converged"` |
| `max_windows` reached | `"max_windows"` |
| `stop()` called (e.g. from a callback) | `"stopped"` |
| the pipeline's generator ended | `"pipeline_exhausted"` |

`stop_on_convergence=False` disables the first row. The default is `"all"` because in a competitive game one seat plateauing while its opponent still improves is not convergence.

**Best-weight restore.** When the run ends (including on an exception) and `restore_best=True`, every seat with a checkpoint is restored to its best snapshot. In competitive games each seat's best score was achieved against the opponents *as they were at that time*, so restored seats can come from different stages of training; pass `restore_best=False` to keep the final, jointly trained weights - the best snapshots remain on disk and in the result.

```python
@dataclass
class MultiAgentRunResult:
    windows_run: int
    stop_reason: str
    seats: Dict[Hashable, SeatResult]
    converged: bool                    # property: stop_reason == "converged"

@dataclass
class SeatResult:
    agent_id: Hashable
    state: LoopState
    converged: bool
    windows_scored: int
    completed_episodes: int
    best_score: Optional[float]
    last_score: Optional[float]
    best_snapshot: Optional[PolicySnapshot]
```

### MultiAgentCallback

```python
class MultiAgentCallback:
    def on_run_start(self) -> None: ...
    def on_window_end(self, window_id: int, metrics: Dict[Hashable, EvalMetrics]) -> None: ...
    def on_improvement(self, agent_id, snapshot: Optional[PolicySnapshot], score: float) -> None: ...
    def on_plateau(self, agent_id, window_id: int) -> None: ...
    def on_converged(self, agent_id, window_id: int) -> None: ...
    def on_run_stop(self, reason: str) -> None: ...
```

Subclassing is optional - any object defining some of these methods works. An exception raised by a callback is logged and does not stop training.

**Not provided** (single-agent `LoopController` features without an agreed multi-agent meaning): online hyperparameter tuning, degradation rollback during training, and policy spawning.

See [Multi-agent training](../guides/multi-agent.md) for a worked example.
