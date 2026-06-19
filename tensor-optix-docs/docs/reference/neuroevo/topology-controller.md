# TopologyController

```
Grow / prune / merge decisions are driven by three independent statistical
signals computed from the live training stream. The scheduler state is not
the grow trigger - topology changes only when the math says so.

Grow fires when ALL of:
  1. Improvement test   - slope β is NOT significantly positive (corrected t-test)
  2. Structure test     - lag-1 autocorrelation of residuals r1 > 2/√n (adaptive)
  3. Capacity test      - gradient utilisation > grow_grad_threshold

Prune (neuron) fires when:
  - avg importance < prune_neuron_threshold  (accumulated over min_prune_observations)
  - avg gradient magnitude < grad_eps        (neuron is truly dead, not just waiting)
  - neuron age > maturation_window           (prevents pruning freshly added neurons)

Prune (edge) fires when:
  - |w| < prune_edge_threshold for prune_edge_patience consecutive episodes

Merge fires when:
  - neuron_a.can_merge_with(neuron_b) - same type, verified
  - Pearson correlation of signed activation histories > merge_similarity_threshold
  - Both neurons have importance > prune_neuron_threshold (not dead - redundant)

Multi-region support:
  The controller operates on a Dict[str, NeuronGraph]. Signal buffers and
  cooldowns are per-region; score buffer and grow/prune/merge signal math are
  shared (eval score reflects the whole agent). Each region evolves
  independently - a saturated encoder region can grow without triggering
  growth in the decision region.
```

## TopologyController

```python
class TopologyController(LoopCallback):
    """
    Evolves one or more NeuronGraphs using statistical signals from the live
    training stream.

    Construct via:
        TopologyController.for_graph(graph, scheduler, **kwargs)
        TopologyController.for_brain(brain, scheduler, **kwargs)
    """

    def __init__(
        self, regions, scheduler=None, grow_op="insert_edge", grow_grad_threshold=0.70,
        grow_cooldown=20, backoff_reset_factor=0.5, grad_eps=1e-6, maturation_window=30,
        prune_edge_threshold=1e-3, prune_edge_patience=10, prune_neuron_threshold=1e-4,
        min_prune_observations=10, merge_similarity_threshold=0.95, merge_check_interval=50,
        act_history_len=100, min_score_buffer=30, max_neurons=256, min_hidden_neurons=1,
    ):
        """
        Parameters
        ----------
        regions:                Dict mapping region name → NeuronGraph.
        scheduler:               The BackoffScheduler inside LoopController.
        grow_op:                 'insert_edge' | 'split_neuron' | 'add_edge'
        grow_grad_threshold:     Minimum fraction of hidden neurons with |∇| >
                                  grad_eps required to allow grow.
        grow_cooldown:           Minimum episodes between successive GROW
                                  events (per region).
        backoff_reset_factor:    After GROW, multiply current backoff interval
                                  by this factor.
        grad_eps:                Gradient magnitude below which a neuron is
                                  considered inactive.
        maturation_window:       Episodes a neuron must exist before it is
                                  eligible for pruning.
        prune_edge_threshold:    Edges with |w| below this are prune candidates.
        prune_edge_patience:     Episodes an edge must stay below threshold
                                  before being pruned.
        prune_neuron_threshold:  Neurons with avg importance below this are pruned.
        min_prune_observations:  Minimum episodes of importance data required
                                  before neuron pruning.
        merge_similarity_threshold: Pearson correlation above which neurons merge.
        merge_check_interval:    Check for merge candidates every N episodes.
        act_history_len:         Per-neuron activation history window length
                                  for merge detection.
        min_score_buffer:        Minimum eval scores required before grow check runs.
        max_neurons:             Hard cap on total neurons per region.
        min_hidden_neurons:      Minimum hidden neurons kept during pruning per region.
        """

    @classmethod
    def for_graph(cls, graph, scheduler=None, **kwargs) -> "TopologyController":
        """Single-graph convenience constructor."""

    @classmethod
    def for_brain(cls, brain, scheduler=None, **kwargs) -> "TopologyController":
        """BrainNetwork convenience constructor - one region per brain region."""

    @property
    def graph(self) -> "NeuronGraph":
        """Returns the 'main' region graph. Raises if multi-region."""

    @property
    def stats(self) -> dict:
        """Grow/prune/merge counters."""
```

### Internal signal checks (for reference, not typically called directly)

```python
def _grow_signal_check(self, region_name: str) -> bool:
    """
    Returns True only when all three statistical conditions hold:
      1. Score trend is not significantly improving  (corrected t-test)
      2. Residual autocorrelation is significant     (adaptive threshold)
      3. Gradient utilisation is high                (region at capacity)
    """

def _compute_grad_util(self, hidden, grad_mag) -> float:
    """Fraction of hidden neurons with total |grad| > grad_eps."""

def _accumulate_neuron_importance(self, hidden, total_w, graph) -> None:
    """I(v) = Σ|w_e| * (‖h‖₁/d + ε) - delegates to neuron.importance()."""

def _episode_neuron_stats(self, graph, hidden):
    """
    Single O(N+E) pass → two dicts keyed by hidden neuron id:
      grad_mag:  |bias.grad| + Σ|edge.weight.grad| for incident edges
      total_w:   Σ|edge.weight| for incident edges
    """
```

See [Build a neuroevo agent](../../guides/neuroevo.md#the-topology-controller) for usage and [Topology operations](topology-ops.md) for the mutation primitives this class calls.
