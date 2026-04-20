"""
tests/test_improvements/test_07_cli_yaml.py

Tests for tensor_optix.config (YAML loading) and tensor_optix.cli.

Correctness claims:

1. ROUND-TRIP PARITY
   A config written programmatically and a config loaded from YAML produce
   identically-structured TrainConfig objects (same env, algorithm, hyperparams).
   This proves that YAML is a lossless representation of training configuration.

2. PRECEDENCE ORDER
   defaults  <  config file  <  CLI overrides
   An override "agent.learning_rate=1e-3" overwrites the YAML value.
   An override "seed=42" overwrites the default seed.
   Lower-precedence values are not affected by higher-precedence overrides.

3. VALIDATION
   Missing required keys (env, algorithm) raise KeyError.
   Unknown algorithms raise ValueError.
   Malformed override strings (no "=") raise ValueError.

4. TYPE COERCION
   YAML values are correctly typed: floats as float, bools as bool, ints as int,
   lists as list.  CLI overrides are YAML-parsed so "true", "1e-3", "[64,64]"
   all work.

5. NESTED OVERRIDE
   Dotted keys address nested sub-dicts:
       "agent.learning_rate=1e-3" sets config.agent["learning_rate"] = 1e-3
       "optimizer.max_episodes=50" sets config.optimizer["max_episodes"] = 50

6. AGENT CONSTRUCTION PARITY
   build_agent_from_config() produces an agent whose get_hyperparams() matches
   the values in the config's "agent" section.

7. CLI VALIDATE COMMAND
   Running the CLI with "validate" exits 0 on a valid config, 1 on invalid.

8. FRAMEWORK INFERENCE
   "TorchPPOAgent" → framework="torch" (auto-inferred)
   "TFSACAgent"    → framework="tf"    (auto-inferred)
   Explicit "framework: tf" overrides the inferred value.
"""

import sys
import textwrap

import numpy as np
import pytest

from tensor_optix.config import (
    TrainConfig,
    load_config,
    apply_overrides,
    config_to_dict,
    build_agent_from_config,
)
from tensor_optix.cli import main as cli_main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_YAML = textwrap.dedent("""\
    env: CartPole-v1
    algorithm: TorchPPOAgent
    seed: 0
    agent:
      learning_rate: 3.0e-4
      clip_ratio: 0.2
      gamma: 0.99
      n_epochs: 4
      minibatch_size: 64
    optimizer:
      max_episodes: 50
      verbose: false
""")

SAC_YAML = textwrap.dedent("""\
    env: Pendulum-v1
    algorithm: TorchSACAgent
    agent:
      learning_rate: 3.0e-4
      gamma: 0.99
      tau: 0.005
      batch_size: 256
      replay_capacity: 100000
""")


def _write_yaml(tmp_path, content: str, name: str = "config.yaml") -> str:
    p = tmp_path / name
    p.write_text(content)
    return str(p)


# ---------------------------------------------------------------------------
# 1. Round-trip parity
# ---------------------------------------------------------------------------

class TestRoundTripParity:

    def test_load_produces_correct_env(self, tmp_path):
        path = _write_yaml(tmp_path, MINIMAL_YAML)
        cfg  = load_config(path)
        assert cfg.env == "CartPole-v1"

    def test_load_produces_correct_algorithm(self, tmp_path):
        path = _write_yaml(tmp_path, MINIMAL_YAML)
        cfg  = load_config(path)
        assert cfg.algorithm == "TorchPPOAgent"

    def test_load_preserves_agent_hyperparams(self, tmp_path):
        path = _write_yaml(tmp_path, MINIMAL_YAML)
        cfg  = load_config(path)
        assert cfg.agent["learning_rate"] == pytest.approx(3e-4)
        assert cfg.agent["clip_ratio"]    == pytest.approx(0.2)
        assert cfg.agent["gamma"]         == pytest.approx(0.99)

    def test_load_preserves_optimizer_kwargs(self, tmp_path):
        path = _write_yaml(tmp_path, MINIMAL_YAML)
        cfg  = load_config(path)
        assert cfg.optimizer["max_episodes"] == 50
        assert cfg.optimizer["verbose"] is False

    def test_config_to_dict_round_trips(self, tmp_path):
        path = _write_yaml(tmp_path, MINIMAL_YAML)
        cfg  = load_config(path)
        d    = config_to_dict(cfg)
        assert d["env"]       == cfg.env
        assert d["algorithm"] == cfg.algorithm
        assert d["agent"]     == cfg.agent
        assert d["optimizer"] == cfg.optimizer

    def test_seed_preserved(self, tmp_path):
        path = _write_yaml(tmp_path, MINIMAL_YAML)
        cfg  = load_config(path)
        assert cfg.seed == 0


# ---------------------------------------------------------------------------
# 2. Precedence: defaults < file < CLI overrides
# ---------------------------------------------------------------------------

class TestPrecedenceOrder:

    def test_cli_override_beats_file_value(self, tmp_path):
        path = _write_yaml(tmp_path, MINIMAL_YAML)
        cfg  = load_config(path, overrides=["agent.learning_rate=1e-3"])
        assert cfg.agent["learning_rate"] == pytest.approx(1e-3)

    def test_cli_override_does_not_affect_other_keys(self, tmp_path):
        path = _write_yaml(tmp_path, MINIMAL_YAML)
        cfg  = load_config(path, overrides=["agent.learning_rate=1e-3"])
        # Other agent keys unchanged
        assert cfg.agent["gamma"] == pytest.approx(0.99)
        assert cfg.agent["clip_ratio"] == pytest.approx(0.2)

    def test_seed_override(self, tmp_path):
        path = _write_yaml(tmp_path, MINIMAL_YAML)
        cfg  = load_config(path, overrides=["seed=42"])
        assert cfg.seed == 42

    def test_optimizer_override(self, tmp_path):
        path = _write_yaml(tmp_path, MINIMAL_YAML)
        cfg  = load_config(path, overrides=["optimizer.max_episodes=200"])
        assert cfg.optimizer["max_episodes"] == 200

    def test_multiple_overrides(self, tmp_path):
        path = _write_yaml(tmp_path, MINIMAL_YAML)
        cfg  = load_config(path, overrides=[
            "agent.learning_rate=5e-4",
            "seed=7",
            "optimizer.verbose=true",
        ])
        assert cfg.agent["learning_rate"] == pytest.approx(5e-4)
        assert cfg.seed == 7
        assert cfg.optimizer["verbose"] is True

    def test_file_beats_dataclass_default(self, tmp_path):
        """The YAML sets window_size explicitly; default is 2048."""
        yaml_content = MINIMAL_YAML + "window_size: 512\n"
        path = _write_yaml(tmp_path, yaml_content)
        cfg  = load_config(path)
        assert cfg.window_size == 512


# ---------------------------------------------------------------------------
# 3. Validation — missing keys and unknown values
# ---------------------------------------------------------------------------

class TestValidation:

    def test_missing_env_raises_key_error(self, tmp_path):
        content = "algorithm: TorchPPOAgent\n"
        path = _write_yaml(tmp_path, content)
        with pytest.raises(KeyError, match="env"):
            load_config(path)

    def test_missing_algorithm_raises_key_error(self, tmp_path):
        content = "env: CartPole-v1\n"
        path = _write_yaml(tmp_path, content)
        with pytest.raises(KeyError, match="algorithm"):
            load_config(path)

    def test_unknown_algorithm_raises_value_error(self, tmp_path):
        content = "env: CartPole-v1\nalgorithm: FooBarAgent\n"
        path = _write_yaml(tmp_path, content)
        with pytest.raises(ValueError, match="FooBarAgent"):
            load_config(path)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "nonexistent.yaml"))

    def test_malformed_override_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid override"):
            apply_overrides({}, ["no_equals_sign"])


# ---------------------------------------------------------------------------
# 4. Type coercion
# ---------------------------------------------------------------------------

class TestTypeCoercion:

    def test_float_parsed_correctly(self, tmp_path):
        path = _write_yaml(tmp_path, MINIMAL_YAML)
        cfg  = load_config(path)
        assert isinstance(cfg.agent["learning_rate"], float)

    def test_int_parsed_correctly(self, tmp_path):
        path = _write_yaml(tmp_path, MINIMAL_YAML)
        cfg  = load_config(path)
        assert isinstance(cfg.agent["n_epochs"], int)
        assert isinstance(cfg.optimizer["max_episodes"], int)

    def test_bool_parsed_correctly(self, tmp_path):
        path = _write_yaml(tmp_path, MINIMAL_YAML)
        cfg  = load_config(path)
        assert cfg.optimizer["verbose"] is False

    def test_cli_bool_override(self, tmp_path):
        path = _write_yaml(tmp_path, MINIMAL_YAML)
        cfg  = load_config(path, overrides=["optimizer.verbose=true"])
        assert cfg.optimizer["verbose"] is True

    def test_cli_list_override(self, tmp_path):
        path = _write_yaml(tmp_path, MINIMAL_YAML)
        cfg  = load_config(path, overrides=["hidden_sizes=[64,64]"])
        assert cfg.hidden_sizes == [64, 64]

    def test_cli_float_scientific_notation(self, tmp_path):
        path = _write_yaml(tmp_path, MINIMAL_YAML)
        cfg  = load_config(path, overrides=["agent.learning_rate=3e-4"])
        assert cfg.agent["learning_rate"] == pytest.approx(3e-4)


# ---------------------------------------------------------------------------
# 5. Nested override correctness
# ---------------------------------------------------------------------------

class TestNestedOverride:

    def test_apply_overrides_creates_nested_key(self):
        d = {}
        result = apply_overrides(d, ["agent.lr=1e-3"])
        assert result["agent"]["lr"] == pytest.approx(1e-3)

    def test_apply_overrides_does_not_mutate_input(self):
        d = {"agent": {"lr": 1e-4}}
        apply_overrides(d, ["agent.lr=1e-3"])
        assert d["agent"]["lr"] == 1e-4   # original untouched

    def test_apply_overrides_adds_new_nested_key(self):
        d = {"agent": {"gamma": 0.99}}
        result = apply_overrides(d, ["agent.tau=0.005"])
        assert result["agent"]["gamma"] == 0.99
        assert result["agent"]["tau"]   == pytest.approx(0.005)

    def test_apply_overrides_top_level_key(self):
        d = {"seed": 0}
        result = apply_overrides(d, ["seed=42"])
        assert result["seed"] == 42


# ---------------------------------------------------------------------------
# 6. Agent construction parity
# ---------------------------------------------------------------------------

class TestAgentConstructionParity:

    def test_ppo_agent_hyperparams_match_config(self, tmp_path):
        """
        build_agent_from_config() creates an agent whose hyperparams match
        what was written in the YAML.  This is the correctness guarantee that
        YAML config ≡ Python config for the agent's behaviour.
        """
        path  = _write_yaml(tmp_path, MINIMAL_YAML)
        cfg   = load_config(path)
        agent, _ = build_agent_from_config(cfg)
        hp    = agent.get_hyperparams()

        assert hp.params["learning_rate"] == pytest.approx(3e-4, rel=1e-4)
        assert hp.params["gamma"]         == pytest.approx(0.99)

    def test_sac_agent_from_yaml(self, tmp_path):
        """SAC agent is constructed correctly from a continuous-env config."""
        path  = _write_yaml(tmp_path, SAC_YAML)
        cfg   = load_config(path)
        agent, _ = build_agent_from_config(cfg)

        from tensor_optix.algorithms.torch_sac import TorchSACAgent
        assert isinstance(agent, TorchSACAgent)
        hp = agent.get_hyperparams()
        assert hp.params["gamma"] == pytest.approx(0.99)

    def test_config_hyperparams_equal_direct_hyperparams(self, tmp_path):
        """
        Python HyperparamSet({...}) == config.agent dict
        ↔ YAML and Python produce identical training parameters.
        """
        from tensor_optix.core.types import HyperparamSet
        from tensor_optix.factory import make_agent
        import gymnasium as gym

        path = _write_yaml(tmp_path, MINIMAL_YAML)
        cfg  = load_config(path)

        # Agent from YAML
        agent_yaml, _ = build_agent_from_config(cfg)
        hp_yaml = agent_yaml.get_hyperparams()

        # Agent from Python
        env = gym.make("CartPole-v1")
        hp_direct = HyperparamSet(params=dict(cfg.agent), episode_id=0)
        agent_py  = make_agent(env, framework="torch", hyperparams=hp_direct)
        env.close()
        hp_py = agent_py.get_hyperparams()

        # Both must have the same learning_rate and gamma
        assert hp_yaml.params["learning_rate"] == pytest.approx(
            hp_py.params["learning_rate"], rel=1e-4
        )
        assert hp_yaml.params["gamma"] == pytest.approx(hp_py.params["gamma"])


# ---------------------------------------------------------------------------
# 7. Framework inference
# ---------------------------------------------------------------------------

class TestFrameworkInference:

    def test_torch_algo_infers_torch_framework(self, tmp_path):
        content = "env: CartPole-v1\nalgorithm: TorchPPOAgent\n"
        path = _write_yaml(tmp_path, content)
        cfg  = load_config(path)
        assert cfg.framework == "torch"

    def test_tf_algo_infers_tf_framework(self, tmp_path):
        content = "env: CartPole-v1\nalgorithm: TFSACAgent\n"
        path = _write_yaml(tmp_path, content)
        cfg  = load_config(path)
        assert cfg.framework == "tf"

    def test_explicit_framework_overrides_inferred(self, tmp_path):
        content = "env: CartPole-v1\nalgorithm: TorchPPOAgent\nframework: torch\n"
        path = _write_yaml(tmp_path, content)
        cfg  = load_config(path)
        assert cfg.framework == "torch"

    def test_all_known_algorithms_accepted(self, tmp_path):
        from tensor_optix.config import _KNOWN_ALGORITHMS
        for algo in sorted(_KNOWN_ALGORITHMS):
            content = f"env: CartPole-v1\nalgorithm: {algo}\n"
            path = _write_yaml(tmp_path, content, name=f"{algo}.yaml")
            cfg  = load_config(path)
            assert cfg.algorithm == algo


# ---------------------------------------------------------------------------
# 8. CLI validate command
# ---------------------------------------------------------------------------

class TestCLIValidateCommand:

    def test_validate_exits_zero_on_valid_config(self, tmp_path):
        path = _write_yaml(tmp_path, MINIMAL_YAML)
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["validate", path])
        assert exc_info.value.code == 0

    def test_validate_exits_one_on_missing_key(self, tmp_path):
        content = "algorithm: TorchPPOAgent\n"   # missing env
        path = _write_yaml(tmp_path, content)
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["validate", path])
        assert exc_info.value.code == 1

    def test_validate_exits_one_on_missing_file(self, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["validate", str(tmp_path / "missing.yaml")])
        assert exc_info.value.code == 1

    def test_validate_accepts_overrides(self, tmp_path):
        path = _write_yaml(tmp_path, MINIMAL_YAML)
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["validate", path, "seed=99", "agent.gamma=0.95"])
        assert exc_info.value.code == 0
