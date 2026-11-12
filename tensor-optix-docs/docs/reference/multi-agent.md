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
    SelfPlayConfig,
    SelfPlayPipeline,
    OpponentPool,
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
        optimizer="auto",                  # per-seat tuning: "auto" | None | callable(agent) | {agent_id: optimizer}
        self_play: Optional[SelfPlayConfig] = None,
        rollback_on_degradation: bool = False,
        degradation_threshold: float = 0.95,
        min_consecutive_degradations: int = 3,
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
    frozen_windows: int                # windows played by a pool snapshot (self-play)
    degradations: int                  # confirmed degradations
    rollbacks: int                     # of which restored the best checkpoint
    hyperparams: Optional[dict]        # the seat's hyperparameters at the end of the run
```

`MultiAgentRunResult.pool_sizes` reports the number of self-play snapshots held per seat.

### MultiAgentCallback

```python
class MultiAgentCallback:
    def on_run_start(self) -> None: ...
    def on_window_end(self, window_id: int, metrics: Dict[Hashable, EvalMetrics]) -> None: ...
    def on_improvement(self, agent_id, snapshot: Optional[PolicySnapshot], score: float) -> None: ...
    def on_plateau(self, agent_id, window_id: int) -> None: ...
    def on_converged(self, agent_id, window_id: int) -> None: ...
    def on_hyperparam_update(self, agent_id, old_params: dict, new_params: dict) -> None: ...
    def on_degradation(self, agent_id, window_id: int, score: float, rolled_back: bool) -> None: ...
    def on_run_stop(self, reason: str) -> None: ...
```

Subclassing is optional - any object defining some of these methods works. An exception raised by a callback is logged and does not stop training.

### Per-seat hyperparameter tuning

Every seat has its own optimizer, driven by that seat's own scores - the same online tuning `RLOptimizer` performs for a single agent.

| `optimizer=` | Result |
|---|---|
| `"auto"` (default) | an `SPSAOptimizer` over the agent's `default_param_bounds` / `default_log_params`, for each seat whose agent declares them; other seats are not tuned |
| `None` | no tuning |
| `callable(agent) -> BaseOptimizer` | called once per seat |
| `{agent_id: BaseOptimizer or None}` | explicit per seat |

A single optimizer instance cannot be shared between seats (optimizers hold per-run state); passing one with more than one seat raises `ValueError`.

After every *scored* window the seat's optimizer receives the seat's `EvalMetrics` history (last 200) and its suggestion is applied with `agent.set_hyperparams()`; `on_hyperparam_update` fires when anything changed. `optimizer.on_improvement()` is called when the seat's trend is improving and `optimizer.on_plateau()` when the seat converges. Windows that are not scored (no finished episode, no turn, or frozen for self-play) do not step the optimizer.

A seat's score also moves when the other seats change, so the tuning signal is noisier than in single-agent training.

### Rollback on degradation

Off by default; `rollback_on_degradation=True` requires `checkpoint_dir`.

- A scored window is **degraded** when the seat's score is below `best - max(|best| * (1 - degradation_threshold), noise floor)`, where the noise floor is the seat's own adaptive floor (so ordinary score noise does not qualify).
- After `min_consecutive_degradations` degraded windows in a row the degradation is **confirmed**: `on_degradation` fires, and for an on-policy agent the seat's best checkpoint is restored and training continues. Off-policy agents are reported (`rolled_back=False`) but never rolled back - their replay buffer would pull restored weights straight back.
- Degraded windows do not count toward the plateau counter, and a confirmed degradation resets it, so a collapsed seat is not declared converged at its collapsed level.
- Windows in which the seat's optimizer is probing never count.
- **With self-play enabled**, only windows in which at least one *other* seat was played by a pool snapshot count. In a competitive game a score also drops when the opponent improves, which is not a collapse; windows against a frozen opponent are the ones where the opposition did not change. Windows against all-live opponents neither count nor reset the streak.

Without self-play, use rollback for cooperative games. In a competitive game without a pool it would undo real learning whenever the opponent gets better.

## Self-play

Two agents that only ever play each other's current version tend to chase each other in circles. Training against a pool of past snapshots makes forgetting how to beat an old strategy cost reward. There are two ways to use it.

```python
@dataclass
class SelfPlayConfig:
    pool_size: int = 10               # snapshots kept per seat, oldest evicted first
    snapshot_every: int = 20          # save a snapshot every N windows
    opponent_prob: float = 0.5        # MultiAgentRunner: chance per window that a seat is frozen
    pool_dir: Optional[str] = None    # None = temporary directory, removed when the run ends
    opponent_factory: Optional[Callable[[Hashable], Any]] = None
    seed: Optional[int] = None        # seeds the pool's sampling
```

Snapshots are weights on disk. A frozen opponent is one *stand-in* agent per seat into which a sampled snapshot is loaded, so memory does not grow with `pool_size`. The stand-in is a `copy.deepcopy` of the live agent (framework module references are shared, not copied) - this works for the built-in Torch agents; pass `opponent_factory=lambda agent_id: <fresh agent>` for agent types that cannot be deep-copied. The stand-in's hyperparameters are synced from the live agent each time, and its rollout cache is discarded after every window.

### Every seat learns: `MultiAgentRunner(self_play=...)`

- Every `snapshot_every` windows each seat's current weights are added to **that seat's** pool.
- Each window, each seat whose pool is non-empty is frozen with probability `opponent_prob`: it is played by a uniformly sampled past snapshot of itself. At least one seat always stays live.
- A frozen seat does not learn, is not scored and does not step its optimizer for that window (`SeatResult.frozen_windows` counts these). Live seats learn and are scored whether their opponents were live or frozen.
- The pipeline's live agents are restored before `run()` returns, then best-weight restore proceeds as usual.

Pools are per seat, so this works for asymmetric games. It needs a pipeline with `set_agents()` (both built-in multi-agent pipelines).

When episodes are longer than a window, an episode's return can include steps that were played by a snapshot before the seat went live again.

### One learner: `SelfPlayPipeline`

```python
class SelfPlayPipeline(BasePipeline):
    """
    Turns a multi-agent pipeline into an ordinary single-agent pipeline:
    the learner plays learner_seat, every other seat is played by a past
    snapshot of the learner.
    """

    def __init__(self, pipeline, learner_seat, agent=None, config: Optional[SelfPlayConfig] = None): ...
    def set_agent(self, agent) -> None: ...

    @property
    def pool(self) -> Optional[OpponentPool]: ...
```

Because it is a `BasePipeline`, `RLOptimizer` / `LoopController` drive it with everything the single-agent loop has: every optimizer, rollback on degradation, policy spawning, checkpoint confirmation.

- The pool starts with the learner's initial weights and gains a snapshot every `snapshot_every` windows; each window every opponent seat is loaded with a uniformly sampled snapshot.
- The opponent seats must be playable by the learner's policy - same observation and action spaces (symmetric games). `opponent_prob` is not used here.
- Only the learner seat's `EpisodeData` is yielded; windows in which the learner took no step are skipped. With the AEC pipeline the learner gets roughly `window_size / n_seats` steps per window.

### OpponentPool

```python
class OpponentPool:
    def __init__(self, pool_size: int = 10, pool_dir: Optional[str] = None, seed: Optional[int] = None): ...
    def add(self, key, agent) -> str: ...                       # save agent's weights as the newest snapshot
    def sample(self, key) -> Optional[str]: ...                 # uniform over the pool
    def load_into(self, key, stand_in, live_agent=None) -> bool: ...
    def size(self, key) -> int: ...
    def sizes(self) -> Dict[Hashable, int]: ...
    def cleanup(self) -> None: ...                              # removes the directory if the pool created it
```

**Not provided** (a single-agent `LoopController` feature without an agreed multi-agent meaning): policy spawning in `MultiAgentRunner` - use `SelfPlayPipeline` with `RLOptimizer` when you need it for one learner. One policy shared across seats, league training and centralized critics are out of scope.

See [Multi-agent training](../guides/multi-agent.md) for a worked example.
