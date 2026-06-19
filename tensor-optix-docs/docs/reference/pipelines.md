# tensor_optix.pipeline

Three concrete `BasePipeline` implementations.

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
```

Physics-engine exceptions raised during `env.step()` (e.g. Box2D `AssertionError`) are caught and treated as episode termination rather than propagating.

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
        episode_boundary_fn=None, reconnect_on_disconnect=True,
    ): ...

    @staticmethod
    def every_n_steps(n: int): ...
    @staticmethod
    def every_n_seconds(n: float): ...
    @staticmethod
    def on_done_signal(): ...
```

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
        self, env_fns, agent=None, window_size: int = 200, async_envs: bool = False,
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
```

See [Train an RL agent](../guides/train-rl-agent.md) for the `optimal_window_size()` helper that picks a sensible `window_size` automatically.
