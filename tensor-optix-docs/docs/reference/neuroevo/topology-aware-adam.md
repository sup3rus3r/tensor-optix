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

`TopologyController` does **not** call `notify_topology_change` for you automatically - if you're using `TopologyController` as a callback with a `GraphAgent`, the agent's own optimizer (constructed internally) would need to be a `TopologyAwareAdam` instance and wired to receive this notification after each mutation. For most users relying on `make_agent(..., neuroevo=True)`, this is an advanced customization point rather than something you need to manage directly.
