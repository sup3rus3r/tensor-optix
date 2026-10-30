"""Live adaptive runtime (proposal sec 19) -- the recommend-only
observe -> detect -> propose -> experiment -> evaluate -> recommend loop,
composed entirely over the ledger, Nanonets, and strategy-discovery
layers built in earlier phases. Never auto-promotes: PromotionPolicy is
always constructed with approval_mode="human" inside this package, and
the deployed baseline only ever changes via the separate, explicitly
human-invoked tensor_optix.runtime.promote module.
"""

from .baseline import establish_baseline, get_baseline_weights_path, get_current_baseline_experiment_id, run_reevaluation
from .experiment_stage import SHORTLIST, evaluate_and_recommend, issue_recommendation, propose_and_run
from .live_runtime import CycleRecommendation, LiveAdaptiveRuntime, RuntimeConfig
from .observer import DeploymentObserver, ObservationResult
from .promote import promote_candidate_to_baseline

__all__ = [
    "establish_baseline", "get_baseline_weights_path", "get_current_baseline_experiment_id",
    "run_reevaluation", "SHORTLIST", "evaluate_and_recommend", "issue_recommendation",
    "propose_and_run", "CycleRecommendation", "LiveAdaptiveRuntime", "RuntimeConfig",
    "DeploymentObserver", "ObservationResult", "promote_candidate_to_baseline",
]
