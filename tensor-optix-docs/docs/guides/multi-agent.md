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

## What you get - and what you don't

`MultiAgentRunner` only alternates *collect → learn* for every seat. It does not provide checkpointing, online hyperparameter tuning, or convergence detection - those live in `LoopController`, which is single-agent. Self-play schedules, league training, and centralized critics are out of scope.
