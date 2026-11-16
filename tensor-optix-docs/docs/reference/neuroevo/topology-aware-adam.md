# TopologyAwareAdam

```
TopologyAwareAdam - Adam optimizer that resets momentum state for parameters
affected by topology changes (grow / prune / merge).

Without state reset, Adam's stale (m, v) estimates from before the structural
change distort the first several updates on the modified parameter, causing
transient instability. Calling notify_topology_change() after any grow/prune
operation discards only the affected parameters' state so the rest of the
network continues uninterrupted.
```

## TopologyAwareAdam

```python
class TopologyAwareAdam:
    """
    Thin wrapper around torch.optim.Adam with topology-aware state management.

    All standard Adam methods (step, zero_grad, state_dict, load_state_dict,
    add_param_group) delegate to the inner optimizer so this is a drop-in
    replacement.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, amsgrad=False): ...

    def notify_topology_change(self, params) -> None:
        """
        Reset Adam momentum state for the given parameters.

        Call after any grow, prune, or merge operation - pass the
        nn.Parameter objects whose weights have structurally changed (new
        edges/neurons, or surviving params that absorbed the weight of a
        pruned/merged neighbour).

        Parameters not in the optimizer's tracked groups are silently ignored.
        """

    def carry_over(self, old, old_packed=None, old_ids=None,
                   new_packed=None, new_ids=None) -> None:
        """
        Copy Adam state from `old` after a topology rebuild replaced some
        parameters with new objects.

        A parameter that is the same object in both optimizers keeps its state.
        The packed edge-weight vector is a new Parameter after every rebuild:
        pass the old and new vector with the edge id of each position
        (NeuronGraph._ff_edge_ids at the time of each rebuild) and positions
        are matched by edge id. A surviving edge keeps its momentum wherever it
        moved; a new edge starts from zero.
        """

    def step(self, closure=None): ...
    def zero_grad(self, set_to_none: bool = True) -> None: ...
    def state_dict(self): ...
    def load_state_dict(self, state_dict) -> None: ...
    def add_param_group(self, param_group) -> None: ...

    @property
    def param_groups(self): ...
    @property
    def state(self): ...
```

### Usage

```python
from tensor_optix.neuroevo import TopologyAwareAdam

optimizer = TopologyAwareAdam(graph.parameters(), lr=3e-4)
optimizer.notify_topology_change(new_params)  # call after any topology mutation
```

`GraphAgent` uses a `TopologyAwareAdam` and handles topology changes itself. Every grow, prune or merge rebuilds the graph's packed weight vector as a new `Parameter`, so the agent builds a new optimizer and calls `carry_over()` to move Adam's moments across by edge id. You do not need to call anything. Momentum is not reset for edges whose weight an operation changed in place (for example the outgoing edges halved by `split_neuron`), and `notify_topology_change()` cannot do that for single entries of the packed vector since it works per `Parameter`.

Adam keeps one step counter per `Parameter`, so a new edge in a long-running packed vector has a mature step count and empty moments. Its first few updates are a few times larger than `lr`, then settle.

`load_weights()` starts the optimizer from scratch, since the restored weights have no relation to the old momentum.
