"""
tensor_optix.cli — Command-line entry point for tensor-optix.

Installed as ``tensor-optix`` when the package is installed:

    tensor-optix train config.yaml [key=value ...]

Subcommands
-----------
train  Run a full training loop described by a YAML config file.
       Any key in the config can be overridden on the command line using
       dotted notation:

           tensor-optix train config.yaml seed=42 agent.learning_rate=1e-3

validate  Parse and validate a config file without running training.
          Exits 0 on success, 1 on error.  Useful in CI.

           tensor-optix validate config.yaml

Precedence:   defaults < config file < CLI overrides
"""

import argparse
import sys


def _cmd_train(args) -> int:
    """Execute the train subcommand."""
    from tensor_optix.config import (
        load_config,
        build_agent_from_config,
        build_pipeline_from_config,
        build_optimizer_from_config,
    )

    try:
        cfg = load_config(args.config, overrides=args.overrides)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"[tensor-optix] Config error: {exc}", file=sys.stderr)
        return 1

    # Optional: seed global RNGs for reproducibility
    _seed_all(cfg.seed)

    # Build components
    try:
        agent, _   = build_agent_from_config(cfg)
        pipeline   = build_pipeline_from_config(cfg)
        optimizer  = build_optimizer_from_config(cfg, agent, pipeline)
    except Exception as exc:
        print(f"[tensor-optix] Build error: {exc}", file=sys.stderr)
        return 1

    # Run
    optimizer.run()
    return 0


def _cmd_validate(args) -> int:
    """Execute the validate subcommand."""
    from tensor_optix.config import load_config
    try:
        cfg = load_config(args.config, overrides=args.overrides)
        print(f"[tensor-optix] Config valid:")
        print(f"  env={cfg.env!r}  algorithm={cfg.algorithm!r}  "
              f"framework={cfg.framework!r}  seed={cfg.seed}")
        if cfg.agent:
            print(f"  agent: {cfg.agent}")
        if cfg.optimizer:
            print(f"  optimizer: {cfg.optimizer}")
        return 0
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"[tensor-optix] Validation failed: {exc}", file=sys.stderr)
        return 1


def _seed_all(seed: int) -> None:
    """Seed numpy, random, and optionally torch/tf."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tensor-optix",
        description="Autonomous RL training loop — tensor-optix CLI",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # ---- train -----------------------------------------------------------
    train = sub.add_parser(
        "train",
        help="Run a training loop from a YAML config file",
        description=(
            "Load a YAML config, build the agent/pipeline/optimizer, and run. "
            "CLI overrides take precedence over the config file. "
            "Example: tensor-optix train cfg.yaml seed=42 agent.learning_rate=1e-3"
        ),
    )
    train.add_argument("config",    help="Path to YAML config file")
    train.add_argument(
        "overrides", nargs="*",
        metavar="key=value",
        help="Zero or more dotted-key overrides (e.g. agent.gamma=0.95)",
    )

    # ---- validate --------------------------------------------------------
    validate = sub.add_parser(
        "validate",
        help="Parse and validate a YAML config without running training",
    )
    validate.add_argument("config", help="Path to YAML config file")
    validate.add_argument("overrides", nargs="*", metavar="key=value")

    return parser


def main(argv=None) -> None:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    if args.command == "train":
        sys.exit(_cmd_train(args))
    elif args.command == "validate":
        sys.exit(_cmd_validate(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
