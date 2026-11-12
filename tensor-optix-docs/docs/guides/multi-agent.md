# Multi-agent training (PettingZoo)

`tensor_optix.multi_agent` drives PettingZoo environments with one ordinary tensor-optix agent per seat. Install the extra:

```bash
pip install tensor-optix[torch,pettingzoo]
```

## Simultaneous-move games (Parallel API)

```python
from pettingzoo.classic import rps_v2            # any ParallelEnv
from tensor_optix import make_agent
from tensor_optix.multi_agent import SeatEnvView, ParallelMultiAgentPipeline, MultiAgentRunner

env = rps_v2.parallel_env()
env.reset(seed=0)

views  = {aid: SeatEnvView(env, aid) for aid in env.possible_agents}
agents = {aid: make_agent(views[aid]) for aid in env.possible_agents}

pipeline = ParallelMultiAgentPipeline(env, agents, window_size=256, views=views)
MultiAgentRunner(pipeline, agents).run(max_windows=200)
```

`SeatEnvView` gives `make_agent()` the single-seat observation/action spaces it needs, and one-hot encodes `Discrete` observations so they can feed an MLP.

## Turn-based games (AEC API)

```python
from pettingzoo.classic import rps_v2
from tensor_optix import make_agent
from tensor_optix.multi_agent import SeatEnvView, AECMultiAgentPipeline, MultiAgentRunner

env = rps_v2.env()
env.reset(seed=0)

views  = {aid: SeatEnvView(env, aid) for aid in env.possible_agents}
agents = {aid: make_agent(views[aid]) for aid in env.possible_agents}

pipeline = AECMultiAgentPipeline(env, agents, window_size=400, views=views)
MultiAgentRunner(pipeline, agents).run(max_windows=500)
```

`window_size` here counts **total** `env.step()` calls across all seats. Rewards are credited to the action that earned them: in AEC games the result of a move (including the win/loss delivered on a seat's final turn) only becomes visible on that seat's *next* turn, and the pipeline writes it back onto the step that caused it - see [the reference](../reference/multi-agent.md#aecmultiagentpipeline) for the exact contract and its one window-boundary limitation.

**Action masks.** Board games such as `tictactoe_v3` or `connect_four_v3` return a dict observation with an `action_mask`. The built-in agents do not apply masks; wrap the env (or the agent's `act()`) so illegal moves are never selected.

## Convergence and checkpoints

`MultiAgentRunner` tracks every seat separately: it scores each seat's window, checkpoints a seat whenever it beats its own best score, and detects per-seat convergence with the same trend-based `BackoffScheduler` the single-agent loop uses.

```python
runner = MultiAgentRunner(
    pipeline, agents,
    checkpoint_dir="./checkpoints",   # one sub-directory per seat
    dormant_threshold=20,             # flat windows before a seat counts as converged
    stop_when="all",                  # stop when every seat has converged ("any": the first)
)
result = runner.run(max_windows=2000)

print(result.stop_reason)             # "converged" | "max_windows" | "stopped" | "pipeline_exhausted"
for aid, seat in result.seats.items():
    print(aid, seat.state.name, seat.best_score, seat.best_snapshot.weights_path)
```

When the run ends each seat is restored to its best checkpoint. In a competitive game a seat's best score was earned against its opponents as they were at that moment, so the restored seats may come from different points in training - pass `restore_best=False` to keep the final, jointly trained weights instead (the best snapshots stay on disk).

Without `checkpoint_dir` the runner still detects convergence and reports best scores, but saves and restores nothing. `MultiAgentRunner(pipeline, agents).run(max_windows=N)` therefore behaves as before, except that it can stop early once every seat has converged (`stop_on_convergence=False` turns that off).

**Long episodes.** A window is a fixed number of steps (`window_size`, counted across all seats for AEC). In games whose episodes run for thousands of steps most windows contain no finished episode; those windows train the agents but are not scored, and a seat is not considered converged until it has finished `min_completed_episodes_before_converged` episodes (default 10). Pick a `window_size` large enough to hold a few episodes where you can - the trend test needs scored windows to work with.

Hook into the run with a callback:

```python
class Log:
    def on_improvement(self, agent_id, snapshot, score):
        print(f"{agent_id}: new best {score:.2f}")
    def on_converged(self, agent_id, window_id):
        print(f"{agent_id} converged at window {window_id}")

runner.add_callback(Log())
```

## Hyperparameter tuning per seat

Each seat is tuned online by its own optimizer, exactly as `RLOptimizer` tunes a single agent. It is on by default: every seat whose agent declares `default_param_bounds` (all built-in agents do) gets its own SPSA optimizer. After each scored window the seat's hyperparameters may be adjusted.

```python
MultiAgentRunner(pipeline, agents)                                   # SPSA per seat (default)
MultiAgentRunner(pipeline, agents, optimizer=None)                   # no tuning
MultiAgentRunner(pipeline, agents,
                 optimizer=lambda agent: AdaptiveOptimizer(param_bounds=agent.default_param_bounds))
```

`result.seats[aid].hyperparams` holds each seat's final hyperparameters, and `on_hyperparam_update(agent_id, old, new)` fires on every change.

## Rollback on degradation

```python
runner = MultiAgentRunner(
    pipeline, agents,
    checkpoint_dir="./checkpoints",
    rollback_on_degradation=True,        # restore a seat's best weights if its score collapses
    min_consecutive_degradations=3,
)
```

When a seat's score stays clearly below its best for several scored windows in a row, the seat is restored to its best checkpoint and keeps training. `result.seats[aid].rollbacks` counts how often that happened.

A score also falls when an opponent gets better, so in a competitive game combine it with self-play: then only windows played against a frozen past opponent can trigger a rollback. Without self-play, enable it for cooperative games only.

## Self-play

In a competitive game, agents that only ever face each other's current version tend to go in circles. Self-play against a pool of past snapshots fixes that. Pick the form that matches what you are training.

**Train every seat** - each seat keeps its own policy and sometimes faces a past version of its opponent:

```python
from tensor_optix.multi_agent import SelfPlayConfig

runner = MultiAgentRunner(
    pipeline, agents,
    checkpoint_dir="./checkpoints",
    self_play=SelfPlayConfig(snapshot_every=20, pool_size=10, opponent_prob=0.5),
)
result = runner.run(max_windows=5000)
print(result.pool_sizes, {a: s.frozen_windows for a, s in result.seats.items()})
```

A seat that is played by a snapshot for a window does not learn or get scored for that window.

**Train one agent** for a symmetric game - the other seats become part of the environment, and the full single-agent loop drives the learner:

```python
from tensor_optix import RLOptimizer
from tensor_optix.multi_agent import SelfPlayPipeline, SelfPlayConfig

env = rps_v2.env()
env.reset(seed=0)
views = {aid: SeatEnvView(env, aid) for aid in env.possible_agents}
learner = make_agent(views["player_0"])

inner = AECMultiAgentPipeline(env, {}, window_size=2048, views=views)
pipeline = SelfPlayPipeline(inner, learner_seat="player_0",
                            config=SelfPlayConfig(snapshot_every=10))

RLOptimizer(agent=learner, pipeline=pipeline, checkpoint_dir="./checkpoints").run()
```

This form gets everything `RLOptimizer` offers (all optimizers, rollback on degradation, policy spawning). It requires that the opponent seats can be played by the learner's policy, i.e. the same observation and action spaces.

Self-play is for competitive or mixed games; in a purely cooperative game it is the wrong tool. The defaults (`snapshot_every=20`, `pool_size=10`, `opponent_prob=0.5`) are starting points, not tuned values.

## What you don't get

Policy spawning is not part of `MultiAgentRunner` (use `SelfPlayPipeline` + `RLOptimizer` when one learner is enough). One policy shared across seats, league training, and centralized critics are out of scope.
