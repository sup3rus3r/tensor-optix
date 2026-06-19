# API Reference

This section documents every public class and function in tensor-optix, organized by module. Docstrings are reproduced verbatim from source.

- **[core](core/types.md)** - the framework-agnostic abstractions (`BaseAgent`, `BasePipeline`, `BaseEvaluator`, `BaseOptimizer`, `LoopController`/`LoopCallback`) plus shared utilities (checkpointing, normalizers, replay buffers, regime detection, ensembles).
- **[Top-level API](optimizer.md)** - `RLOptimizer` and the simplified `Optimizer` wrapper.
- **[make_agent](factory.md)** - the auto-selection factory.
- **[Algorithms](algorithms.md)** - the 15 built-in agents across PyTorch, TensorFlow, and JAX/Flax.
- **[Pipelines](pipelines.md)** - `BatchPipeline`, `LivePipeline`, `VectorBatchPipeline`.
- **[Optimizers](optimizers.md)** - SPSA, Momentum, Backoff, PBT, Adaptive (hyperparameter tuning).
- **[Callbacks](callbacks.md)** - Rich dashboard, Weights & Biases, TensorBoard.
- **[Distributed](distributed.md)** - IMPALA-style async actor-learner with V-trace.
- **[Trial search](orchestrator.md)** - `TrialOrchestrator` (Optuna-based).
- **[Config / CLI](config-cli.md)** - YAML config loading and the `tensor-optix` CLI.
- **[ML mode](ml.md)** - training arbitrary `nn.Module`s with the same loop.
- **[Exploration](exploration.md)** - `RNDPipeline` (Random Network Distillation).
- **[Neuroevo](neuroevo/index.md)** - the dynamic-topology policy subsystem.
