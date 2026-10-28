"""Phase 3 — baseline experiment selection.

Scope boundary: this package contains NO learning/ML model and NO
predictive uncertainty. StrategySelector.propose() makes simple, explicit,
evidence-based decisions over ledger history (ranking by observed score
confidence intervals) — it is not the Phase 4+ cross-task meta-model
("Nanonets"). task_features is deliberately rejected here; that belongs to
a later phase.

ASHA-style early stopping and cost-aware surrogate selection are also out
of scope for this package: both require a concurrent trial scheduler that
does not exist yet in this codebase (TrialOrchestrator runs trials
strictly sequentially), and a cost-aware surrogate is itself a predictive
model. Both are deferred, not stubbed.
"""

from .backfill import backfill_study_to_ledger
from .selector import StrategySelector, MethodStats, SelectionDecision
from .uncertainty import score_interval, ScoreInterval

__all__ = [
    "backfill_study_to_ledger",
    "StrategySelector",
    "MethodStats",
    "SelectionDecision",
    "score_interval",
    "ScoreInterval",
]
