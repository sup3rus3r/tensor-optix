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

Hook into the run with a callback:

```python
class Log:
    def on_improvement(self, agent_id, snapshot, score):
        print(f"{agent_id}: new best {score:.2f}")
    def on_converged(self, agent_id, window_id):
        print(f"{agent_id} converged at window {window_id}")

runner.add_callback(Log())
```

## What you don't get

Online hyperparameter tuning, degradation rollback during training, and policy spawning remain single-agent (`LoopController`) features. Self-play schedules, league training, and centralized critics are out of scope.
