# Export to ONNX

```bash
pip install tensor-optix[onnx]
```

Every `BaseAgent` exposes `export_onnx(path)`. The default implementation raises `NotImplementedError`; it's overridden on all Torch agents (`TorchPPOAgent`, `TorchSACAgent`, `TorchTD3Agent`, `TorchDQNAgent`, `TorchRecurrentPPOAgent`).

```python
agent.export_onnx("policy.onnx")
```

Only the **actor** (policy) network is exported - `learn()`, `get_hyperparams()`, and `save_weights()` are training-time operations with no place in an inference graph. For example, `TorchTD3Agent.export_onnx`:

```
Export the deterministic continuous actor to ONNX.

Input  - "observation": (batch_size, obs_dim)    float32
Output - "action":      (batch_size, action_dim) float32

The actor applies tanh internally, so outputs are clipped to (-1, 1).
Rescale to your environment's action bounds at deployment if required.
```

Continuous-action agents that apply `tanh` internally (TD3, SAC) export actions in `(-1, 1)` - rescale to your environment's actual action bounds at deployment time. Discrete-action agents (PPO, DQN, Rainbow) export logits or Q-values, not sampled actions - apply `argmax` or your own sampling at inference time.

## Neuroevo agents

Neuroevo's `GraphAgent`/`RecurrentGraphAgent` do not currently implement `export_onnx` - the dynamic, mutating topology and Python-side neuron state (`_current`, history buffers) don't map cleanly onto a static ONNX graph. For neuroevo deployment, save/load the native checkpoint format (`agent.save_weights()` / `GraphAgent.from_checkpoint()`) instead - see [Build a neuroevo agent](neuroevo.md#save-and-load).
