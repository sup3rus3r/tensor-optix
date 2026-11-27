"""
``report.build()`` for environments: an ``RLSolution`` with the recommended
algorithm (variant flags included) and the wrappers / add-ons from ACCEPTED
findings. ``fit(env)`` trains with tensor-optix's full loop (online tuning on,
convergence-based stopping) and returns the agent and the run.
"""
from __future__ import annotations

import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..build import PendingCriticalFindings, register_builder
from ..findings import DecisionState, Severity
from ..schema import DiscoveryReport
from . import env as E
from .trials import Variant, _recurrent_agent


@dataclass
class RLSolution:
    algorithm: str
    obs_normalize: Optional[Dict[str, Any]] = None     # {"mean": [...], "std": [...]}
    action_rescale: Optional[Dict[str, Any]] = None    # {"low": [...], "high": [...]}
    rnd: bool = False
    applied: List[str] = field(default_factory=list)
    evidence_mismatch: List[str] = field(default_factory=list)
    seed: int = 0

    def wrappers(self) -> Dict[str, Any]:
        w: Dict[str, Any] = {}
        if self.action_rescale:
            w["action_rescale"] = self.action_rescale
        if self.obs_normalize:
            w["obs_normalize"] = self.obs_normalize
        return w

    def make_env(self, env_or_factory: Any):
        factory, _, _ = E.env_factory(env_or_factory)
        return E.make_wrapped(factory, self.wrappers())

    def make_agent(self, env, **kwargs):
        if self.algorithm == "RECURRENT_PPO":
            return _recurrent_agent(env, int(kwargs.get("hidden", 128)))
        from ...factory import make_agent
        return make_agent(self.algorithm, env, **kwargs)

    def fit(self, env_or_factory: Any, *, max_episodes: Optional[int] = None,
            checkpoint_dir: Optional[str] = None, agent_kwargs: Optional[Dict[str, Any]] = None,
            **rl_kwargs) -> Tuple[Any, Any]:
        """Train to convergence (or ``max_episodes`` windows). Returns (agent, RLOptimizer)."""
        import numpy as np
        from ...optimizer import RLOptimizer
        from ...pipeline.batch_pipeline import BatchPipeline
        from ...utils.window_size import optimal_window_size
        env = self.make_env(env_or_factory)
        agent = self.make_agent(env, **(agent_kwargs or {}))
        window = rl_kwargs.pop("window_size", None) or optimal_window_size(
            self.make_env(env_or_factory), "PPO" if "PPO" in self.algorithm else self.algorithm)
        pipe = BatchPipeline(env=env, agent=agent, window_size=window)
        if self.rnd:
            from ...exploration.rnd import RNDPipeline
            pipe = RNDPipeline(pipe, obs_dim=int(np.prod(env.observation_space.shape)))
        ckpt = checkpoint_dir or tempfile.mkdtemp(prefix="optix_rl_solution_")
        run = RLOptimizer(agent=agent, pipeline=pipe, max_episodes=max_episodes, checkpoint_dir=ckpt, **rl_kwargs)
        run.run()
        return agent, run

    def describe(self) -> Dict[str, Any]:
        return {"algorithm": self.algorithm, "wrappers": sorted(self.wrappers()), "rnd": self.rnd,
                "applied_findings": list(self.applied), "evidence_mismatch": list(self.evidence_mismatch)}


def build_rl(report: DiscoveryReport, *, family: Optional[str] = None, acknowledge_pending: bool = False,
             **_: Any) -> RLSolution:
    pending_critical = [f.id for f in report.findings
                        if f.severity == Severity.CRITICAL and report.decision(f.id).state == DecisionState.PENDING]
    if pending_critical and not acknowledge_pending:
        raise PendingCriticalFindings(f"decide on these CRITICAL findings first: {pending_critical}")
    rec = report.recommendation
    if family is not None:
        v = Variant(family.upper())
    elif rec is not None and rec.family:
        c = rec.config
        v = Variant(c["algorithm"], obs_normalize=bool(c.get("obs_normalize")), rnd=bool(c.get("rnd")))
    else:
        raise ValueError("the report recommends no algorithm (it abstained); pass build(family=...) to choose one")
    sol = RLSolution(algorithm=v.algorithm, rnd=v.rnd, seed=report.seed)
    stats = None
    for f in report.findings:
        a = f.suggested_action
        if a is not None and a.kind == "wrap_obs_normalize" and a.params.get("mean") is not None:
            stats = {"mean": a.params["mean"], "std": a.params["std"]}
    if v.obs_normalize:
        sol.obs_normalize = stats
    applied = []
    for f in report.findings:
        a = f.suggested_action
        if a is None or report.decision(f.id).state != DecisionState.ACCEPTED:
            continue
        if a.kind == "wrap_action_rescale":
            sol.action_rescale = dict(a.params)
        elif a.kind == "wrap_obs_normalize":
            sol.obs_normalize = {"mean": a.params.get("mean"), "std": a.params.get("std")}
        elif a.kind == "add_rnd":
            sol.rnd = True
        elif a.kind == "use_recurrent":
            sol.algorithm = "RECURRENT_PPO"
        else:
            continue
        applied.append(f.id)
    sol.applied = applied
    in_trials = set(report.profile.sections.get("applied_findings") or [])
    with_actions = {f.id for f in report.findings if f.suggested_action is not None}
    sol.evidence_mismatch = sorted((in_trials ^ set(applied)) & with_actions)
    if sol.evidence_mismatch:
        warnings.warn("the report's trials and this build apply different findings "
                      f"({sol.evidence_mismatch}); its scores may not describe this setup.", stacklevel=3)
    return sol


register_builder("rl", build_rl)
