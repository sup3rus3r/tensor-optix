"""
TopologyCallback — a public, sanctioned extension point for observing
NeuronGraph topology mutations (grow/prune/merge) performed by
TopologyController.

Before this existed, the only ways to observe a topology change were
(a) parsing logger.info/debug lines, or (b) polling TopologyController.stats
on an interval — no way to be notified exactly when a mutation happens.
This mirrors LoopCallback's existing design (discrete on_* hooks, same
per-callback exception isolation) so that external code — a GUI, a
custom logger, a research script — can react to topology events without
subclassing TopologyController or touching its private _do_grow/_check_*
methods.
"""
from typing import Optional


class TopologyCallback:
    """
    Hook into TopologyController at topology-mutation events.
    Subclass and override any methods you care about. All hooks are
    no-ops by default.

    Example:
        class TopologyLogger(TopologyCallback):
            def on_grow(self, **event):
                print(f"grew: {event['op']} -> {event['new_neuron_id']}")
    """

    def on_grow(
        self,
        region_name: str,
        episode_id: int,
        op: str,
        new_neuron_id: Optional[str],
        n_neurons: int,
        n_edges: int,
    ) -> None: ...

    def on_prune_edge(
        self,
        region_name: str,
        episode_id: int,
        edge_id: str,
        n_neurons: int,
        n_edges: int,
    ) -> None: ...

    def on_prune_neuron(
        self,
        region_name: str,
        episode_id: int,
        neuron_id: str,
        n_neurons: int,
        n_edges: int,
    ) -> None: ...

    def on_merge(
        self,
        region_name: str,
        episode_id: int,
        kept_neuron_id: str,
        removed_neuron_id: str,
        n_neurons: int,
        n_edges: int,
    ) -> None: ...
