# BrainNetwork

```
BrainNetwork - a container of named NeuronGraph regions with sparse,
learnable inter-region pathways.

Each region is an independent NeuronGraph that can be evolved by its own
TopologyController. Pathways are sparse sets of learnable edges that cross
region boundaries - they live in BrainNetwork itself, not inside either region.

Forward pass:
  1. Run each region in topological order (regions form a DAG via pathways;
     recurrent inter-region pathways use delay >= 1 from the source region's
     output history).
  2. Inject pathway signals into destination regions as additional pre-activation
     inputs, accumulated before that region's own forward pass.
  3. Return the activations of all output neurons across all regions (or a
     named subset if output_regions is specified).
```

## InterRegionEdge / Pathway

```python
@dataclass
class InterRegionEdge:
    """A single learnable edge crossing two regions."""
    edge_id: str
    src_region: str
    src_neuron: str
    dst_region: str
    dst_neuron: str
    weight: nn.Parameter
    delay: int  # timesteps; 0 = same-step (requires src before dst in region order)

@dataclass
class Pathway:
    """A named bundle of InterRegionEdges from one region to another."""
    pathway_id: str
    src_region: str
    dst_region: str
    edge_ids: List[str] = field(default_factory=list)
```

## BrainNetwork

```python
class BrainNetwork(nn.Module):
    """
    A collection of named NeuronGraph regions connected by sparse inter-region
    pathways.

    Parameters
    ----------
    name : str - Optional human-readable name (for logging / repr).
    output_regions : list[str] | None - If set, forward() only collects
        outputs from these regions. If None, all regions contribute.
    """

    def add_region(self, name: str, graph: NeuronGraph) -> None:
        """Register a NeuronGraph as a named region."""

    def get_region(self, name: str) -> NeuronGraph: ...

    @property
    def region_names(self) -> List[str]: ...
    @property
    def regions(self) -> Dict[str, NeuronGraph]: ...

    def add_pathway(self, src_region, dst_region, n_connections=4, delay=1,
                     weight_init=0.0, pathway_id=None) -> str:
        """
        Create a sparse pathway from src_region to dst_region.

        Randomly pairs n_connections output neurons from src with
        input/hidden neurons of dst. All weights initialised to weight_init
        (default 0.0, function-preserving). Returns the pathway_id.
        """

    def add_inter_region_edge(self, src_region, src_neuron, dst_region, dst_neuron,
                               weight=0.0, delay=1, edge_id=None) -> str:
        """Add a single hand-crafted inter-region edge. Returns edge_id."""

    def notify_neuron_pruned(self, region_name: str, neuron_id: str) -> None:
        """Remove all InterRegionEdges that reference a pruned neuron."""

    def forward(self, region_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Run one timestep of the entire brain network.

        Parameters
        ----------
        region_inputs : dict[str, Tensor]
            Mapping of region_name -> 1-D observation tensor. Only regions
            that receive external observations need an entry - regions
            driven purely by inter-region pathways can be omitted (they
            receive a zero observation vector).

        Returns
        -------
        Tensor - concatenated output activations from output_regions
        (or all regions if output_regions is None).
        """

    def reset_state(self) -> None:
        """Zero all region histories and neuron states. Call between episodes."""

    def enforce_dale(self) -> None:
        """Enforce Dale's Law on every region graph."""

    def summary(self) -> dict:
        """Per-region neuron/edge counts plus inter-region edge count."""
```

`_region_execution_order()` topologically sorts regions using feedforward (`delay=0`) inter-region edges; regions with only delayed incoming inter-region edges have no within-timestep dependency, since they read from history.

See [Build a neuroevo agent](../../guides/neuroevo.md#brainnetwork-composing-multiple-regions) for usage, and [TopologyController.for_brain](topology-controller.md) for multi-region evolution.
