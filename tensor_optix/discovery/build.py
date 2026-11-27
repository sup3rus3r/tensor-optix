"""
Opt-in build step: turn a report's ACCEPTED findings into a ready-to-train
setup. Domain builders register here; the ML builder arrives with the ML
pipeline, the RL builder with RL discovery.
"""
from __future__ import annotations

from typing import Callable, Dict

_BUILDERS: Dict[str, Callable] = {}


class PendingCriticalFindings(RuntimeError):
    """``build()`` refuses while CRITICAL findings are neither accepted nor rejected."""


def register_builder(domain: str, fn: Callable) -> None:
    _BUILDERS[domain] = fn


def build_solution(report, *args, **kwargs):
    if report.domain == "ml" and "ml" not in _BUILDERS:
        from .ml import build as _ml_build  # noqa: F401  (registers the ML builder)
    fn = _BUILDERS.get(report.domain)
    if fn is None:
        raise NotImplementedError(
            f"no builder is available for domain {report.domain!r} in this version of "
            "tensor-optix discovery"
        )
    return fn(report, *args, **kwargs)
