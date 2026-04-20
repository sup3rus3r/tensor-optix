# tensor-optix — Improvement Roadmap

Living document. Every item is justified by mathematics or strict logical necessity.
Every functional change ships with a before/after test that proves measurable improvement.
Work proceeds top to bottom. Status updated as items complete.

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| `[ ]` | Not started |
| `[~]` | In progress |
| `[x]` | Complete |

---

## Testing Protocol

**Rule:** every item that changes functional behaviour must ship with a before/after test
that produces a measurable, reproducible difference. Observability-only changes (items 1, 10)
are exempt — they add no new signals, only surface existing ones.

**What a valid test looks like:**

1. **Fixed seed** — `np.random.seed(42)`, `torch.manual_seed(42)`, `gym.make(..., seed=42)`.
   A test that passes on one run and fails on another proves nothing.

2. **Minimal environment** — CartPole-v1 (discrete, 4D obs, ~200 steps to solve) or
   LunarLanderContinuous-v3 (continuous, 8D obs) depending on action space. No Atari, no MuJoCo
   — these introduce hardware dependencies and slow the test suite. If an item requires a
   goal-conditioned environment (HER), a custom `GoalEnv` is included in `tests/envs/`.

3. **Convergence metric, not final score** — measure _episodes to threshold_ rather than
   final score. Final scores are noisy; convergence speed is a cleaner signal. Threshold is
   set at 80% of the known solve score for the environment.

4. **Multiple seeds for statistical validity** — run 3 seeds minimum per condition.
   The improvement must hold on the majority of seeds, not just the median. A single lucky
   seed is not evidence.

5. **Ablation, not comparison** — where possible, the "before" is the "after" with one
   component removed (e.g., TD3 with target smoothing disabled vs. full TD3). This isolates
   the specific contribution being claimed rather than comparing two different algorithms
   that differ in many ways simultaneously.

6. **Tests live in `tests/test_improvements/`** — one file per roadmap item, named
   `test_<item_number>_<short_name>.py`. Each test imports the before and after variants
   and runs both under identical conditions.

---

## Release Plan

| Milestone | Version | Items |
|-----------|---------|-------|
| Current | `1.2.6` | — |
| Tier 1 complete | `1.3.0` | 1, 2, 3 |
| Tier 2 complete | `1.4.0` | 4, 5, 6 |
| Tier 3 complete | `1.5.0` | 7, 8, 9 |
| Tier 4 complete | `2.0.0` | 10, 11, 12 |

`pyproject.toml` version is bumped at each tier completion, immediately before PyPI release.

---

## Tier 1 — High impact, self-contained

### 1. `[x]` Built-in W&B / TensorBoard Callbacks

**What:** `WandbCallback` and `TensorBoardCallback` in `tensor_optix/callbacks/`,
each implementing the existing `LoopCallback` interface.

**Why this is correct:**

The `LoopCallback` contract already exposes every signal the loop produces:

```
on_episode_end(episode_id, eval_metrics)     → raw score, episode count
on_improvement(snapshot)                      → PolicySnapshot: eval_metrics + hyperparams
on_plateau(episode_id, state)                 → LoopState transition
on_dormant(episode_id)                        → convergence event
on_degradation(episode_id, eval_metrics)      → score drop detected
on_hyperparam_update(old_params, new_params)  → SPSA step output
```

**Information theory basis:** you cannot diagnose a training run without observing it.
The minimal sufficient statistic for diagnosing RL training is the tuple
`(score_t, entropy_t, ||∇||_t, EV_t)` at each eval step:

- `score_t` — primary signal, already in `EvalMetrics.primary_score`
- `entropy_t` — policy entropy. `entropy → 0` ⟹ policy is deterministic / collapsed.
  In `train_diagnostics["entropy"]` from `agent.learn()`
- `||∇||_t` — gradient norm. `||∇|| >> 1` ⟹ unstable landscape.
  In `train_diagnostics["grad_norm"]` for all built-in agents
- `EV_t` — explained variance: `EV = 1 − Var(R_t − V(s_t)) / Var(R_t)`.
  `EV < 0` ⟹ critic is actively harmful. In `train_diagnostics["explained_variance"]` for PPO

The generalization gap `train_score − val_score` is already in `EvalMetrics` when a val
pipeline is active — must be logged to detect overfitting before it compounds.

**SPSA diagnostics:** `on_hyperparam_update` gives `(old_params, new_params)`. The normalized
step magnitude `||x_new − x_old||₂` in `[0,1]^N` space measures SPSA aggression.
Large steps during improvement indicate `c` needs shrinking (already handled internally,
but worth surfacing).

**Implementation notes:**
- Both are pure `LoopCallback` subclasses — zero changes to core
- W&B: `wandb.log({"score": ..., "entropy": ..., "grad_norm": ...}, step=episode_id)`
- TensorBoard: `writer.add_scalar("score", ..., global_step=episode_id)`
- Hyperparams logged from `snapshot.hyperparams.params` on `on_improvement`
- `on_loop_stop` closes the writer / finalizes the W&B run

**New optional dependencies:**

```toml
[project.optional-dependencies]
wandb = ["wandb>=0.16"]
tensorboard = ["tensorboard>=2.14"]
```

**Test:** observability only — no before/after performance test required.
Functional test: instantiate both callbacks, run a short `RLOptimizer` loop (10 episodes),
assert that the expected `log()` / `add_scalar()` calls were made with the correct keys
and episode indices. Use `unittest.mock.patch` on the W&B and TensorBoard APIs.

---

### 2. `[ ]` TD3 — Twin Delayed DDPG

**What:** `TFTDDAgent` and `TorchTD3Agent` in `algorithms/`, following the same dual-adapter
pattern as SAC (`tf_td3.py` / `torch_td3.py`).

**Why this is correct:**

SAC and TD3 solve the same overestimation problem in Q-learning (Thrun & Schwartz 1993)
via different mechanisms. TD3 applies three structural fixes to the deterministic policy
gradient:

**Fix 1 — Twin critics (identical to SAC):**

```
Q_target(s, a) = r + γ · min(Q_φ1'(s', ã), Q_φ2'(s', ã))
```

Taking the minimum over two independently-initialized critics provably reduces overestimation
bias. Both critics are updated on every step.

**Fix 2 — Delayed policy updates (variance reduction):**

The policy gradient is:

```
∇_θ J(θ) = E_s[ ∇_a Q_φ1(s, a)|_{a=π_θ(s)} · ∇_θ π_θ(s) ]
```

The actor gradient flows through the critic. When the critic is noisy (early training),
the actor gradient is also noisy. Updating the policy every `d` critic steps (`d = 2`,
Fujimoto et al. 2018) reduces variance by allowing the critic to partially stabilize
before propagating error into the policy. This is variance reduction, not a heuristic.

**Fix 3 — Target policy smoothing (regularization):**

```
ã = clip(π_θ'(s') + clip(ε, −c, c),  a_low, a_high),    ε ~ N(0, σ)
```

Without smoothing, the critic can learn an exploitable spike: `Q(s, a*)` can be much larger
than `Q(s, a* ± δ)` because the replay buffer contains few transitions near `a*`. The
smoothing enforces that `Q` must be consistent in a neighbourhood around the target action,
eliminating spurious optima that the deterministic actor would otherwise exploit.

**Why TD3 fills a real gap:**

SAC maximizes `E[Σ γ^t (r_t + α H(π(·|s_t)))]`. The entropy bonus `α H` encourages
stochasticity. For environments where exploration is externally structured, or where a
deterministic policy is required at deployment, SAC's stochasticity is a liability.
TD3 provides a deterministic alternative. The two algorithms are mathematically
non-dominated: neither is uniformly better.

**Implementation leverage:** SAC already implements twin critics, target network polyak
updates, and the replay buffer. TD3 changes:

1. Replace the stochastic actor (`μ + σ·ε`) with a deterministic one: `a = π_θ(s) = tanh(f(s))`
2. Add target policy smoothing in `_compute_targets()`
3. Gate the actor update on `step % d == 0`
4. Remove `log_α` and the entropy temperature optimizer

---

**Test — `tests/test_improvements/test_02_td3.py`:**

**Claim:** each TD3 fix independently reduces Q-value overestimation and/or policy instability.

**Ablation design (3 seeds × 3 conditions × 150 episodes, LunarLanderContinuous-v3):**

| Condition | Twin critics | Delayed updates | Target smoothing |
|-----------|-------------|-----------------|-----------------|
| `td3_none` | ✗ | ✗ | ✗ (= DDPG) |
| `td3_twin` | ✓ | ✗ | ✗ |
| `td3_full` | ✓ | ✓ | ✓ |

**Assertions (must hold on ≥ 2/3 seeds):**

```python
# Q-value overestimation: predicted Q vs. Monte Carlo return
# Lower is better — positive means overestimation
assert overestimation(td3_twin) < overestimation(td3_none)
assert overestimation(td3_full) < overestimation(td3_twin)

# Convergence: episodes to reach score >= 150 (80% of 200 solve threshold)
assert episodes_to_threshold(td3_full) <= episodes_to_threshold(td3_none)
```

**Q-value overestimation measurement:**

```python
def overestimation(agent, env, n_episodes=20, seed=0):
    """
    Returns mean(Q_predicted(s,a) - G_actual(s,a)) over n_episodes.
    Positive = overestimation.
    """
    ...
```

---

### 3. `[ ]` Multi-step Returns for SAC

**What:** Wire the existing `n_step` parameter path (fully implemented in both DQN variants)
into `TFSACAgent` and `TorchSACAgent`.

**Why this is correct:**

The standard 1-step Bellman target for SAC:

```
Q_target(s_t, a_t) = r_t + γ · (min_j Q_φj'(s_{t+1}, ã) − α log π(ã | s_{t+1}))
```

The n-step Bellman target accumulates real returns for `n` steps before bootstrapping:

```
G^n_t = Σ_{k=0}^{n-1} γ^k r_{t+k}

Q_target(s_t, a_t) = G^n_t + γ^n · (min_j Q_φj'(s_{t+n}, ã) − α log π(ã | s_{t+n}))
```

**Bias-variance tradeoff:** 1-step TD has low variance (one reward sample) but high bias
(the bootstrap depends entirely on the current critic quality, which is poor early in training).
n-step TD accumulates `n` real reward samples, reducing the fraction of the target that
comes from the biased critic. In sparse reward environments, `r_t ≈ 0` for nearly every
step — the 1-step target is `γ · Q_target(s_{t+1}, ·) ≈ 0`, giving the critic near-zero
gradient signal. n-step returns break this by propagating real reward signal `n` steps back
through actual experience.

**Off-policy correctness note:** n-step returns are biased under off-policy sampling when
the replay buffer contains experience from old policies. This bias is already accepted in the
DQN implementation. For SAC, the bias introduced by n-step off-policy data is generally
outweighed by the variance reduction for `n ≤ 5` (Barth-Maron et al. 2018 use `n=5` with
SAC for robotics). The approximation is documented; it is not silent.

**Implementation effort:** the `n_step` logic in both DQN variants is self-contained.
Adding an `n_step_buffer` deque to the SAC `_store_transition()` path is a ~30-line change
per variant, identical in structure to the DQN path.

---

**Test — `tests/test_improvements/test_03_multistep_sac.py`:**

**Claim:** `n_step=3` SAC converges faster than `n_step=1` SAC in a sparse reward setting.

**Setup (3 seeds × 2 conditions × 300 episodes):**

A `SparseRewardWrapper` is applied to `LunarLanderContinuous-v3`. The wrapper maps all
per-step rewards to 0 except the terminal reward (landing bonus/penalty only). This creates
a genuinely sparse reward signal where the 1-step critic bootstraps from near-zero Q-values.

```python
class SparseRewardWrapper(gym.RewardWrapper):
    """Pass through only the terminal reward. All intermediate rewards → 0."""
    def reward(self, reward):
        return reward if self._is_terminal else 0.0
```

**Assertions (must hold on ≥ 2/3 seeds):**

```python
# Primary: convergence speed
assert episodes_to_threshold(sac_n3, threshold=100.0) \
     < episodes_to_threshold(sac_n1, threshold=100.0)

# Secondary: Q-value calibration at episode 100
# n=3 should be closer to true returns than n=1 (less bootstrap bias)
assert abs(q_error(sac_n3, ep=100)) < abs(q_error(sac_n1, ep=100))
```

---

## Tier 2 — High impact, moderate effort

### 4. `[ ]` Auto Algorithm Selection

**What:** `tensor_optix.make_agent(env, **kwargs)` factory that inspects `env.action_space`
and returns the correct algorithm. No guessing — the action space type is a mathematical
property that determines the valid policy parameterizations.

**Why this is a correctness guarantee, not a convenience feature:**

**Discrete action spaces** — `gym.spaces.Discrete(n)`:

The policy is a categorical distribution: `π(a|s) = softmax(logits(s))_a`.
Actions are indices in `{0, ..., n-1}`. SAC's squashed Gaussian outputs actions in
`(-1, 1)` — structurally incompatible with integer indices. Using SAC on a discrete
env is a type error, not a suboptimal choice.

**Continuous action spaces** — `gym.spaces.Box(low, high, shape)`:

The policy is a squashed Gaussian: `a = tanh(μ + σ·ε)`. PPO with softmax computes
`P(a = i)` for discrete index `i` — undefined on a continuous manifold. The gradient
of the categorical cross-entropy with respect to a continuous action is undefined.
Using PPO on a continuous env without modification is mathematically wrong.

**The selection function is a pure deterministic mapping:**

```
Discrete(n)          → PPO  (categorical, discrete cross-entropy loss)
Box(shape, ...)      → SAC  (default) | TD3 (if deterministic=True)
MultiDiscrete(...)   → NotImplementedError  (correct > convenient)
Dict(...)            → NotImplementedError
```

Never silently fall back to a wrong algorithm. Raise with an explicit message.

---

**Test — `tests/test_improvements/test_04_auto_select.py`:**

**Part A — Correctness test (no training required):**

```python
env_disc = gym.make("CartPole-v1")
env_cont = gym.make("LunarLanderContinuous-v3")

assert isinstance(make_agent(env_disc), (TFPPOAgent, TorchPPOAgent))
assert isinstance(make_agent(env_cont), (TFSACAgent, TorchSACAgent))

# Wrong algorithm on wrong env must produce worse results than correct selection
```

**Part B — Wrong algorithm penalty (3 seeds × 150 episodes):**

```python
# Claim: auto-selected algorithm matches or beats manually-correct selection
# (auto can't be worse than the right answer — it IS the right answer)
assert score(make_agent(env_disc)) >= score(manual_ppo_on_cartpole) * 0.95

# Claim: wrong algorithm on wrong env performs measurably worse
# SAC on CartPole (wrong) vs PPO on CartPole (correct)
assert score(ppo_on_cartpole) > score(sac_on_cartpole)
```

---

### 5. `[ ]` HER — Hindsight Experience Replay

**What:** `HERReplayBuffer` wrapper and `GoalConditionedPipeline` for goal-conditioned
environments implementing the Gymnasium `GoalEnv` interface.

**Why this is correct:**

In a sparse reward environment, `P(r > 0) ≈ 0` for a random or early policy.
The standard Bellman update:

```
Q(s, a, g) ← r(s, a, g) + γ · max Q(s', a', g) ≈ 0 + γ · 0 = 0
```

The critic receives zero gradient signal. Training stalls. This is not a hyperparameter
problem — it is a structural problem with the Bellman backup under sparse rewards.
No amount of hyperparameter tuning resolves it.

**HER's key insight (Andrychowicz et al. 2017):**

A trajectory `(s_0, a_0, ..., s_T)` that failed to reach goal `g` _did_ successfully
reach state `s_T`. We can retroactively relabel this trajectory as a success for
`g' = φ(s_T)` (the achieved goal), yielding `r(s_{T-1}, a_{T-1}, g') = 0` (success).
This creates dense training signal from failed trajectories without modifying the algorithm —
only the replay buffer sampling changes.

**Relabeling strategies:**

| Strategy | Definition | Property |
|----------|-----------|----------|
| `final` | `g' = φ(s_T)` | Simplest, minimum variance |
| `future` | `g' = φ(s_t')`, `t' ~ Uniform(t, T)` | Best empirically; `k=4` future goals per transition |
| `episode` | `g' ~ Uniform({φ(s_i)})` | All states in episode |

**Formal correctness:** HER relabeling is valid because the environment provides
`compute_reward(achieved_goal, desired_goal, info)`. Substituting `g' = φ(s_T)` and
calling `compute_reward` yields the correct reward for the relabeled transition.
No distributional assumption is violated — every stored tuple `(s, a, r, s', g')` is
a valid experience under the goal `g'`.

**Integration point:** `HERReplayBuffer` wraps `PrioritizedReplayBuffer`.
On each episode end, it samples `k` additional goals per transition using the chosen
strategy, calls `env.compute_reward()` for relabeled rewards, and stores both original
and relabeled transitions. `PrioritizedReplayBuffer` already handles the storage;
HER adds the relabeling layer above it.

---

**Test — `tests/test_improvements/test_05_her.py`:**

**Environment:** `PointReachEnv` (custom, included in `tests/envs/point_reach.py`).
A 2D point mass must reach a goal position. Reward: `0` on success (within tolerance),
`-1` otherwise. This is a minimal `GoalEnv` with no external dependencies.

```python
class PointReachEnv(gym.GoalEnv):
    """
    2D point mass. obs: [x, y]. goal: [gx, gy]. 
    Reward: 0 if ||pos - goal|| < 0.1, else -1.
    The agent almost never reaches the goal by chance.
    """
```

**Claim:** SAC+HER learns to reach goals within `N` episodes where SAC alone does not.

**Setup (3 seeds × 2 conditions × 500 episodes):**

```python
# Q-value signal: with sparse reward, SAC alone leaves Q ≈ -1/(1-γ) everywhere.
# HER creates non-trivial Q-value structure from episode 1.

# Primary assertion
assert success_rate(sac_with_her, ep=500) > success_rate(sac_alone, ep=500)

# Secondary: Q-value structure emerges earlier
# Q-values should become non-uniform (informative) within first 50 episodes with HER
assert q_value_variance(sac_with_her, ep=50) > q_value_variance(sac_alone, ep=50)
```

---

### 6. `[ ]` Recurrent Policies — LSTM/GRU PPO

**What:** `TorchRecurrentPPOAgent` and `TFRecurrentPPOAgent` with LSTM policy heads.

**Why this is correct:**

Standard PPO assumes the **Markov property**:
`P(s_{t+1} | s_0, ..., s_t, a_0, ..., a_t) = P(s_{t+1} | s_t, a_t)`.

In a POMDP, the agent observes `o_t` where `o_t ≠ s_t`. The sufficient statistic for
optimal action selection is the **belief state** `b_t = P(s_t | o_1:t, a_0:t-1)`.
Maintaining an exact belief is generally intractable. An LSTM approximates it:

```
h_t = LSTM(h_{t-1}, o_t)
π(a_t | o_1:t) ≈ π_θ(a_t | h_t)
```

For environments where history is irrelevant, the LSTM degenerates to a feedforward
(the gradient forces `h_t ≈ f(o_t)`) — so recurrent agents strictly subsume feedforward
agents. There is no environment where a recurrent policy is provably worse than a
feedforward one given sufficient capacity.

**Training correctness — episode boundaries in BPTT:**

Backpropagation through time requires that `h_0` at the start of each truncated sequence
is the hidden state that was actually active at that timestep during rollout, not
re-initialized to zero. The rollout buffer must store `(h_t, o_t, a_t, r_t, done_t)`.
`h_t` is reset to `0` at episode boundaries (not between truncated BPTT windows).

Truncated BPTT window `T_bptt`:

```
L_clip = (1/T_bptt) Σ_{t=1}^{T_bptt} min(r_t · Â_t, clip(r_t, 1−ε, 1+ε) · Â_t)
```

where `r_t = π_θ(a_t | h_t) / π_θ_old(a_t | h_t_old)`. The old hidden states `h_t_old`
are stored from rollout and **detached** before use in the ratio — they are not
recomputed during the update.

**Integration:** `EpisodeData` gains an optional `hidden_states: List[np.ndarray]` field.
Non-recurrent agents leave it `None`. The loop controller is unchanged.

---

**Test — `tests/test_improvements/test_06_recurrent_ppo.py`:**

**Environment:** `MaskedCartPole` (custom, included in `tests/envs/masked_cartpole.py`).
Standard CartPole-v1 with `cart_velocity` and `pole_velocity` observations zeroed out.
The agent observes only `[cart_position, pole_angle]` — a true POMDP since velocity is
required to balance.

```python
class MaskedCartPoleEnv(gym.Wrapper):
    """
    CartPole-v1 with velocity dims (indices 1, 3) set to 0.
    Feedforward policies cannot balance without velocity → low scores.
    LSTM can recover velocity from the history of positions → high scores.
    """
    def observation(self, obs):
        obs = obs.copy()
        obs[1] = 0.0   # cart_velocity
        obs[3] = 0.0   # pole_angular_velocity
        return obs
```

**Claim:** LSTM PPO achieves meaningfully higher score on MaskedCartPole than standard PPO.

**Setup (3 seeds × 2 conditions × 200 episodes):**

```python
# Standard PPO on MaskedCartPole: can't balance, expected score ~ 20-60
# LSTM PPO on MaskedCartPole: recovers velocity from history, expected score ~ 150+

assert mean_score(lstm_ppo, seeds=[0,1,2]) > mean_score(feedforward_ppo, seeds=[0,1,2]) * 1.5

# Also verify LSTM PPO is not worse on fully-observable CartPole
assert mean_score(lstm_ppo_full_obs) >= mean_score(ff_ppo_full_obs) * 0.90
```

---

## Tier 3 — Larger scope

### 7. `[ ]` CLI + YAML Configuration

**What:** A `tensor-optix train config.yaml` entrypoint and a YAML schema that maps
1:1 to `RLOptimizer` constructor kwargs. CLI overrides take precedence over the file.

**Why this is correct — reproducibility is a first-order requirement:**

A training run is parameterized by ~20 values. Storing them in Python code means:
1. The config is not serializable without executing code
2. Diffs between two runs require diffing Python files
3. Parameter sweeps require programmatic code modification

The mapping is exact and lossless — every YAML key maps directly to a kwarg.
No hidden defaults, no implicit behavior.

**Precedence order** (standard, logically necessary):

```
defaults < config file < CLI args
```

CLI overrides allow a sweep to vary one parameter without editing the config file.
This is not optional — it is the minimum interface for reproducible research.

**YAML schema (subset):**

```yaml
algorithm: TorchPPOAgent
env: CartPole-v1
pipeline: BatchPipeline
window_size: 2048

agent:
  learning_rate: 3e-4
  clip_ratio: 0.2
  gamma: 0.99
  gae_lambda: 0.95
  n_epochs: 10
  minibatch_size: 64

optimizer:
  max_episodes: 300
  rollback_on_degradation: true
  verbose: true
  checkpoint_dir: ./checkpoints
```

---

**Test — `tests/test_improvements/test_07_cli_yaml.py`:**

**Claim:** a run configured via YAML produces identical results to the same run configured
via Python, given identical seeds and hyperparameters. This is a correctness test, not a
performance test.

```python
def test_yaml_python_parity(tmp_path):
    seed = 42
    hp = {"learning_rate": 3e-4, "clip_ratio": 0.2, "gamma": 0.99, ...}

    # Run 1: Python config
    scores_python = run_via_python(hp, seed=seed, n_episodes=20)

    # Run 2: YAML config (written to tmp_path, then loaded)
    write_yaml(tmp_path / "config.yaml", hp, seed=seed, n_episodes=20)
    scores_yaml = run_via_cli(tmp_path / "config.yaml")

    np.testing.assert_allclose(scores_python, scores_yaml, rtol=1e-5)
```

---

### 8. `[ ]` ONNX Export

**What:** `agent.export_onnx(path)` on all built-in agents, exporting the actor network only.

**Why this is correct:**

At deployment, only `agent.act(obs)` is called. The actor network is:

```
f_actor: ℝ^{obs_dim} → ℝ^{n_actions}      (discrete: logits)
f_actor: ℝ^{obs_dim} → ℝ^{2·act_dim}      (continuous: [μ || log_σ])
```

`learn()`, `get_hyperparams()`, and `save_weights()` are training-time operations.
They must not be included in the exported model — they have Python runtime dependencies,
training-only state, and are undefined in an inference-only context.

ONNX is the standard interchange format. An ONNX model runs without a Python environment
(ONNX Runtime, TensorRT, OpenVINO, CoreML, C++ API).

**PyTorch export:**

```python
torch.onnx.export(
    actor.eval(),
    dummy_input,                           # shape: (1, obs_dim)
    path,
    input_names=["observation"],
    output_names=["logits"],
    dynamic_axes={"observation": {0: "batch_size"}},
    opset_version=17,
)
```

**Correctness note for continuous actors:** ONNX exports only the network forward pass —
`μ` and `log_σ` outputs. The sampling step `a = tanh(μ + exp(log_σ) · ε)` is not exported
because the deterministic policy `a = tanh(μ)` is the correct deployment default.
Both are documented.

**New optional dependency:**

```toml
[project.optional-dependencies]
onnx = ["onnx>=1.14", "onnxruntime>=1.16"]
```

---

**Test — `tests/test_improvements/test_08_onnx.py`:**

**Claim:** the exported ONNX model produces numerically identical outputs to the original
PyTorch/TF model on the same observations.

```python
def test_onnx_parity(tmp_path):
    agent = TorchPPOAgent(actor=build_actor(), ...)
    obs = np.random.randn(16, obs_dim).astype(np.float32)  # batch of 16

    # PyTorch output
    with torch.no_grad():
        logits_torch = agent._actor(torch.from_numpy(obs)).numpy()

    # ONNX output
    agent.export_onnx(tmp_path / "actor.onnx")
    session = onnxruntime.InferenceSession(str(tmp_path / "actor.onnx"))
    logits_onnx = session.run(["logits"], {"observation": obs})[0]

    np.testing.assert_allclose(logits_torch, logits_onnx, atol=1e-5)
```

---

### 9. `[ ]` Rainbow DQN — Noisy Nets + C51

**What:** Add the two remaining Rainbow components to both DQN variants.
PER and n-step returns are already implemented.

**Current Rainbow status:**

| Component | Status | Location |
|-----------|--------|----------|
| Double Q-learning | `[x]` | `_compute_targets()` uses `min(Q1, Q2)` target net |
| Prioritized replay (PER) | `[x]` | `PrioritizedReplayBuffer`, `per_alpha`/`per_beta` params |
| n-step returns | `[x]` | `n_step` param in both DQN variants |
| Dueling networks | `[x]` | user-composable via custom `q_network` architecture |
| Noisy nets | `[ ]` | see below |
| Distributional RL (C51) | `[ ]` | see below |

**Noisy Nets (Fortunato et al. 2017):**

Standard ε-greedy exploration is non-differentiable — the schedule is a hyperparameter
independent of what the network has learned. Noisy nets replace deterministic weights
with stochastic weights:

```
w = μ + σ ⊙ ε,    ε ~ N(0, I)
```

`μ` and `σ` are both learned parameters. The network learns _when_ to be uncertain:
`σ → 0` for well-understood states (exploit), `σ → large` for uncertain states (explore).
Exploration is in parameter space rather than action space. The gradient flows through
`σ`, making exploration adaptive to the learning signal. The `ε` term is sampled fresh
each forward pass during training; during evaluation `ε = 0` (use `μ` only).
This replaces `epsilon_decay` entirely — a fixed hyperparameter replaced by a learned one.

**C51 Distributional RL (Bellemare et al. 2017):**

Standard DQN learns `E[Z(s,a)]` where `Z(s,a)` is the random return.
C51 learns the full distribution `P(Z(s,a) = z_i)` over a fixed support
`{z_1, ..., z_N}`, `z_i = V_min + i · (V_max − V_min) / (N-1)`.

The distributional Bellman operator:

```
T̂Z(s,a) =^D R + γZ(s', a*)
```

is projected onto the support via linear interpolation:

```
p_i ← Σ_j [ 1 − |clip(T̂z_j, V_min, V_max) − z_i| / Δz ]_0^1 · p_j(s', a*)
```

Loss: `L = − Σ_i p_i(target) · log p_i(predicted)` (cross-entropy over support).

**Why distributional matters:** `E[Z]` is a sufficient statistic for `argmax_a E[Z]`
only when the Bellman operator is applied to expectations. The distributional operator
`T̂` preserves higher moments (variance, skewness) discarded by `E[·]`. The cross-entropy
loss over the full distribution provides richer gradient signal than scalar MSE —
it measures discrepancy across the entire return distribution, not just its mean.

---

**Test — `tests/test_improvements/test_09_rainbow.py`:**

**Part A — Noisy nets vs ε-greedy (ablation):**

**Claim:** Noisy nets achieve the same or better final score with less exploration
hyperparameter sensitivity (no `epsilon_decay` to tune).

**Setup (3 seeds × 2 conditions × 300 episodes, CartPole-v1):**

```python
# Noisy nets: epsilon_decay irrelevant (ε = 0 throughout)
# ε-greedy: epsilon_decay = 0.995 (standard)
assert episodes_to_threshold(dqn_noisy) <= episodes_to_threshold(dqn_epsilon_greedy) * 1.1

# Noisy nets must not require epsilon tuning: test with epsilon_decay=1.0 (no decay)
# ε-greedy with no decay should degrade; noisy nets should be unaffected
assert score(dqn_noisy, epsilon_decay=1.0) > score(dqn_greedy, epsilon_decay=1.0)
```

**Part B — C51 vs scalar DQN (Q-value calibration):**

**Claim:** C51 produces better-calibrated Q-value predictions (predicted mean closer
to true Monte Carlo return) after convergence.

```python
def q_calibration_error(agent, env, n_episodes=50, seed=0):
    """
    |mean(Z_predicted(s,a)) - G_actual(s,a)|, averaged over transitions.
    Lower = better calibrated.
    """
    ...

assert q_calibration_error(c51_agent) < q_calibration_error(scalar_dqn_agent)
```

---

## Tier 4 — Scale and research

### 10. `[ ]` Live Terminal Dashboard

**What:** `RichDashboardCallback` using the `rich` library, rendering a live-updating
panel of score curves, state, and SPSA diagnostics.

**Why:** Human-in-the-loop monitoring. The `verbose=True` path outputs lines;
a dashboard renders the same data as a chart. No new signals — purely a presentation
layer over existing `LoopCallback` hooks.

**Correctness requirement:** the dashboard must add zero latency to the training loop.
`rich.Live` updates in a background thread. The callback writes to a shared `deque`
and returns immediately. The rendering thread reads from the deque asynchronously.

**Test:** observability only — no before/after performance test required.
Latency test: mean callback execution time < 0.5ms per episode with dashboard active.

---

### 11. `[ ]` Distributed Training — Async Actor-Learner

**What:** IMPALA-style (Espeholt et al. 2018) async actor-learner architecture where
N actor processes collect experience and push to a central learner.

**Why:** `VectorBatchPipeline` parallelizes environments within a single process.
The learner is still single-process. For large networks on GPU, the learner is the
bottleneck — actors are idle during the `learn()` call. Async actors decouple
collection from learning:

```
Actors (CPU):      s_t → act() → (s_t, a_t, r_t, s_{t+1}) → queue
Learner (GPU):     dequeue batches → learn() → broadcast updated weights
```

**V-trace importance sampling correction (Espeholt et al. 2018):**

Async actors use stale weights `μ`. The policy ratio corrects for this:

```
ρ̄_s = min(ρ̄, π_θ(a_s|s_s) / π_μ(a_s|s_s))     ← IS weight, capped at ρ̄

δ_s = ρ̄_s (r_s + γ V(s_{s+1}) − V(s_s))         ← corrected TD error

v_s = V(s_s) + Σ_{t=s}^{s+n} (γ^{t-s} Π_{i=s}^{t-1} c̄_i) δ_t   ← V-trace target
```

where `c̄_s = min(c̄, ρ_s)` truncates the trace when the behaviour policy diverges
too far from the current policy, preventing IS weights from exploding.

---

**Test — `tests/test_improvements/test_11_distributed.py`:**

**Claim:** N async actors achieve N× sample throughput vs single actor, with no
degradation in policy quality (V-trace correction is working).

```python
# Throughput: steps per second should scale with actor count
assert steps_per_second(n_actors=4) >= steps_per_second(n_actors=1) * 2.5

# Quality: final score should be within 10% of synchronous baseline
assert score(async_4actors) >= score(sync_1actor) * 0.90
```

---

### 12. `[ ]` JAX/Flax Adapter

**What:** `FlaxAgent` and `FlaxEvaluator` in `adapters/jax/`.

**Why:** JAX's functional transform API (`jit`, `vmap`, `grad`) enables vectorized
gradient computation across entire rollout batches. `vmap` over the batch dimension
replaces explicit per-sample loops. XLA compilation eliminates Python overhead per step.
The `BaseAgent` interface is framework-agnostic by design — a JAX adapter requires no
changes to `LoopController`, `BackoffScheduler`, or any core component.

JAX's functional paradigm (pure functions, immutable state, `pytree` parameter trees)
requires a different weight serialization path than PyTorch `state_dict` or Keras
`save_weights`. This is solvable via `flax.serialization.to_bytes` / `from_bytes`.

---

**Test — `tests/test_improvements/test_12_jax.py`:**

**Claim:** `FlaxAgent` implementing PPO achieves the same score as `TorchPPOAgent`
on CartPole-v1 within the same episode budget, confirming correctness of the implementation.

```python
# Parity test: same env, same hyperparams, same seed
# JAX and PyTorch PPO should converge to within 10% of each other
assert abs(score(flax_ppo) - score(torch_ppo)) / score(torch_ppo) < 0.10
```

---

*Last updated: 2026-04-20*
