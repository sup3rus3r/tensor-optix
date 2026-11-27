"""
tensor_optix.discovery -- understand a dataset or environment, find the model
family that fits it, and explain why, with honest uncertainty.

    from tensor_optix.discovery import discover
    report = discover(df, target="churned")
    report.to_html("discovery.html")

Discovery advises; it never changes anything by itself. Each finding can be
accepted or rejected by id, and ``report.build()`` applies only the accepted
ones.

Heavy pieces (pandas, torch, gymnasium) are imported only when the input
actually needs them.
"""
from .api import discover, discover_ml
from .budget import Budget
from .findings import (
    Action, Category, Decision, DecisionState, Evidence, Finding, Severity,
)
from .schema import (
    DISCOVERY_SCHEMA_VERSION, Comparison, DiscoveryReport, ExperimentResult, PlanStep,
    Profile, Recommendation,
)

__all__ = [
    "discover", "discover_ml", "Budget", "Action", "Category", "Decision", "DecisionState", "Evidence", "Finding", "Severity",
    "DISCOVERY_SCHEMA_VERSION", "Comparison", "DiscoveryReport", "ExperimentResult", "PlanStep",
    "Profile", "Recommendation",
]
