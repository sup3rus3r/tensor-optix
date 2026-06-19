# tensor_optix.core.checkpoint_registry

## CheckpointRegistry

```python
class CheckpointRegistry:
    """
    Manages policy snapshots on disk.

    Responsibilities:
    - Save a new best snapshot when improvement is detected
    - Load the best snapshot (for watchdog rollback, if configured)
    - Maintain a manifest of all snapshots with metadata
    - Prune old snapshots beyond max_snapshots

    Directory structure:
        checkpoint_dir/
            manifest.json
            snapshot_<id>/
                weights/        ← agent.save_weights() writes here
                metadata.json   ← EvalMetrics + HyperparamSet
    """

    def __init__(self, checkpoint_dir: str, max_snapshots: int = 10): ...
```

### Methods

```python
def save(self, agent, eval_metrics: EvalMetrics, hyperparams: HyperparamSet) -> PolicySnapshot:
    """
    Save current agent weights + metadata as a new snapshot.
    Automatically prunes oldest snapshots beyond max_snapshots.
    Returns the created PolicySnapshot.
    """

def load_best(self, agent) -> Optional[PolicySnapshot]:
    """
    Restore agent weights from the best known snapshot.
    Returns the snapshot or None if no snapshots exist.
    """

def load_ensemble(self, agent, top_k: int = 3, score_band: float = 0.1) -> Optional[PolicySnapshot]:
    """
    Average the weights of the top-k checkpoints within a score band
    of the best, then apply the averaged weights to the agent.

    Only checkpoints whose score ≥ best_score × (1 - score_band) are
    included. This prevents averaging across checkpoints from very
    different training stages (e.g. pre/post collapse), which would
    produce a broken policy.

    Falls back to load_best() if fewer than 2 valid checkpoints exist
    or if the agent does not implement average_weights().

    Args:
        top_k:       Maximum number of checkpoints to average (default 3).
        score_band:  Maximum relative score drop allowed for inclusion
                     (default 0.1 = within 10% of best score).

    Returns the best snapshot (for metadata - the weights are the average).
    """

@property
def best(self) -> Optional[PolicySnapshot]: ...
```

`load_ensemble` is how Stochastic Weight Averaging (SWA) is exposed at the registry level - it requires the agent to implement `average_weights()` (see [BaseAgent](base_agent.md)).
