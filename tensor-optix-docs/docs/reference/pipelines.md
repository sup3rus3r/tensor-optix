# tensor_optix.pipeline

Three concrete `BasePipeline` implementations. For PettingZoo environments see [multi_agent](multi-agent.md); for the intrinsic-reward wrapper see [Exploration (RND)](exploration.md).

## BatchPipeline

```python
class BatchPipeline(BasePipeline):
    """
    Continuous stepping pipeline against a Gymnasium-compatible environment.

    Does NOT reset the env per window. Steps continuously and yields fixed-size
    windows of experience. The env resets automatically only when terminated
    or truncated - training never stops between windows.

    window_size: number of steps per yielded EpisodeData window.
                 Default 200. This is the unit of training, not an env episode.

    Uses Gymnasium API:
        env.reset() -> (obs, info)
        env.step(action) -> (obs, reward, terminated, truncated, info)

    EpisodeData fields populated by this pipeline:
        episode_starts: list of step indices where a new episode begins within
                        the window. Index 0 is always included.
        final_obs:      the observation immediately after the last step. None when
                        the window ended at a terminal state. On-policy agents
                        use this to bootstrap V(s_T) correctly when the window
                        ends mid-episode.

    Warning - gym.Env method name collision:
        gymnasium's env wrapper exposes close(), step(), reset(), render(), and
        seed() as its own methods. If your env class defines an *attribute*
        with any of those names, it will shadow the wrapper's method and cause
        confusing AttributeErrors or silent misbehaviour. Rename any
        conflicting env attributes before passing the env to BatchPipeline.
    """

    def __init__(self, env, agent=None, window_size: int = 200): ...

    @property
    def steps_per_episode(self) -> int:
        """Env steps per yielded window (== window_size)."""
```

Physics-engine exceptions raised during `env.step()` (e.g. Box2D `AssertionError`) are caught and treated as episode termination rather than propagating.

When an episode ends mid-window and the agent defines `reset_hidden()` (e.g. `TorchRecurrentPPOAgent`), the pipeline calls it right after the env reset, so recurrent state never leaks from one episode into the next.

`steps_per_episode` is the step hint `TrialOrchestrator` / `ConcurrentTrialScheduler` use to convert a `trial_steps` budget into a number of windows.

## LivePipeline

```python
class LivePipeline(BasePipeline):
    """
    Streams data from a real-time source.

    Use case: live trading, real-world robotics, online environments.

    The user provides a data_source with a stream() generator that yields
    (observation, reward, terminated, truncated, info) tuples per step.

    The data source runs in a background thread with a bounded queue.
    The main loop consumes from the queue safely.

    On disconnect (StopIteration or exception from stream()):
    - If reconnect_on_disconnect=True: calls data_source.stream() again
    - Otherwise: signals end of pipeline

    Episode boundary is determined by episode_boundary_fn.
    Preset factories:
    - LivePipeline.every_n_steps(n)
    - LivePipeline.every_n_seconds(n)
    - LivePipeline.on_done_signal()   (default)
    """

    def __init__(
        self, data_source, agent=None,
        episode_boundary_fn=None, buffer_size: int = 1000,
        reconnect_on_disconnect=True,
    ): ...

    @staticmethod
    def every_n_steps(n: int): ...
    @staticmethod
    def every_n_seconds(n: float): ...
    @staticmethod
    def on_done_signal(): ...
```

Behaviour notes:

- **Backpressure.** The producer thread blocks while the bounded queue (`buffer_size` items) is full and resumes when the consumer catches up. A slow `learn()` therefore throttles the source instead of being mistaken for a disconnect - no samples are dropped and the stream is not restarted.
- **End of stream.** With `reconnect_on_disconnect=False`, when `stream()` is exhausted (or raises) the steps collected so far are yielded as a final, shorter `EpisodeData` and `episodes()` then ends.
- **Idle source.** If no item arrives for 30 s the current episode is closed with whatever was collected.
- **YAML configs.** `LivePipeline` needs a `data_source` object, so it cannot be built from a Gymnasium env id - see [Config / CLI](config-cli.md).

## VectorBatchPipeline

```python
class VectorBatchPipeline(BasePipeline):
    """
    Parallel environment pipeline using gymnasium.vector.

    Runs N environments simultaneously and collects rollouts from all of them
    in lockstep. The yielded EpisodeData contains observations and actions
    from ALL envs interleaved: shape [window_size * n_envs, ...].

    window_size: steps collected PER ENV per yielded EpisodeData.
                 Total steps per yield = window_size * n_envs.
    async_envs:  use AsyncVectorEnv (subprocess) instead of SyncVectorEnv.
                 Requires env_fns to be picklable. If any fn captures a
                 non-picklable object, setup() detects this and falls back
                 to SyncVectorEnv with a warning rather than letting
                 gymnasium raise a confusing TypeError deep in multiprocessing.

    Each env resets automatically when it terminates/truncates. Episode
    boundaries are tracked per-env and merged into the flat yielded arrays.

    Same gym.Env method name collision warning as BatchPipeline applies here.
    """

    def __init__(
        self, env_fns, agent=None, window_size: int = 256, async_envs: bool = False,
    ): ...

    def episodes(self):
        """
        Yield EpisodeData windows collected from all parallel envs.

        At each step, act() is called once PER ENV with each env's current obs
        (single obs, not batched). Results are concatenated across envs and
        steps into flat arrays of length window_size * n_envs.
        """

    @property
    def n_envs(self) -> int: ...

    @property
    def steps_per_episode(self) -> int:
        """Env steps per yielded window (window_size * n_envs)."""
```

Window layout and how the rest of the library consumes it:

- **Layout.** Flat arrays are interleaved as `[t * n_envs + env]`. `EpisodeData.metadata["n_envs"]` records the env count and `EpisodeData.final_obs` is a **list** with one entry per env (the observation after that env's last step, `None` if that step ended an episode).
- **Autoreset.** Vector envs are created with `autoreset_mode=SAME_STEP` (Gymnasium >= 1.1), so the observation returned by a terminal step is already the first observation of the next episode. Gymnasium's default `NEXT_STEP` mode would instead insert one action-ignored, zero-reward "reset" transition after every episode into the training data. On Gymnasium 1.0.x (no `AutoresetMode`) the library falls back to the vector env's default.
- **On-policy agents** (PPO family) pass `n_envs` to `compute_gae()`, which runs the recurrence per env and bootstraps each env from its own `final_obs`.
- **Off-policy agents** (DQN, Rainbow, SAC, TD3 - Torch and TF) pair each step with the next step *of the same env* when filling the replay buffer, and flush n-step accumulation per env.
- **Evaluators** (`TorchEvaluator` / `TFEvaluator`) and the composite-score breadth signal split episodes per env.
- **Not supported:** `TorchRecurrentPPOAgent` keeps a single hidden state and must be used with `BatchPipeline`.

See [Train an RL agent](../guides/train-rl-agent.md) for the `optimal_window_size()` helper that picks a sensible `window_size` automatically.

## Episode returns across windows

`BatchPipeline` and `VectorBatchPipeline` set `EpisodeData.metadata["episode_returns"]` to the full returns of the episodes that finished in the window (an empty list when none did), carrying partial returns from earlier windows. Evaluators read it through `tensor_optix.core.reward_shape.completed_episode_returns()`. A custom pipeline can do the same with `EpisodeReturnTracker`:

```python
from tensor_optix.core.reward_shape import EpisodeReturnTracker, EPISODE_RETURNS_KEY

tracker = EpisodeReturnTracker(n_envs=1)          # keep it for the pipeline's lifetime
returns = tracker.update(rewards, dones)          # per window
episode_data.metadata[EPISODE_RETURNS_KEY] = returns
```

Pipelines that do not publish it (`LivePipeline`, custom ones) are scored from the rewards inside each window, as before. `RNDPipeline` adds its intrinsic bonus to `rewards` after the wrapped pipeline has computed `episode_returns`, so scores stay extrinsic.
