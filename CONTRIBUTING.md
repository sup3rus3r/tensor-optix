# Contributing to tensor-optix

Thank you for your interest in contributing.

---

## Getting Started

```bash
git clone https://github.com/sup3rus3r/tensor-optix
cd tensor-optix
pip install -e ".[dev]"
```

---

## Development Rules

These rules exist to keep the library clean and portable. Please read them before submitting changes.

1. **No algorithm-specific code in `core/`.** PPO, DQN, SAC, etc. must never be referenced in `core/` or `loop_controller.py`.
2. **Gymnasium API only.** Use `(obs, info) = env.reset()` and `(obs, reward, terminated, truncated, info) = env.step()`. Never use the legacy `done` flag.
3. **`HyperparamSet.params` is opaque.** Core code must never read or hardcode specific key names.
4. **Separation of concerns.** The optimizer tunes hyperparameters. `PolicyManager` evolves models. Do not mix these.
5. **TensorFlow only.** No framework abstraction layer — TF-specific code belongs in `adapters/tensorflow/`.

---

## Running Tests

```bash
pytest tests/
```

All tests must pass before a PR is merged. Tests that require TensorFlow use `pytest.importorskip("tensorflow")` and are skipped automatically if TF is unavailable.

---

## Submitting Changes

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Add or update tests to cover your changes
4. Ensure `pytest tests/` passes
5. Open a pull request with a clear description of what changed and why

---

## Reporting Bugs

Use the [bug report issue template](.github/ISSUE_TEMPLATE/bug_report.md). Include a minimal reproducible example where possible.

---

## Suggesting Features

Use the [feature request issue template](.github/ISSUE_TEMPLATE/feature_request.md). Frame requests in terms of use cases, not implementation — explain what problem you're solving.
