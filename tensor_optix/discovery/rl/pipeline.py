"""
RL discovery: environment in, ``DiscoveryReport`` (domain "rl") out.

  1. probes (random policy): spaces, returns, reward sparsity / scale,
     observation scale, action bounds, determinism, partial observability
  2. stage A: every applicable base algorithm (Discrete: PPO, DQN; Box: SAC,
     TD3, PPO) trained for the same number of steps on the same seeds
  3. stage B: variants the probes motivate, on the leader -- observation
     normalisation, an RND exploration bonus, a recurrent policy -- each
     compared with its base on the same seeds; a variant is reported as a
     finding only when it is better by more than the ROPE
  4. refinement: more seeds for the leader and the variants it cannot be
     separated from, while the budget allows (up to ``MAX_SEEDS``)
  5. recommendation (same decision rule as for datasets, on per-seed mean
     evaluation returns paired by seed; ROPE = 5% of the gap between the random
     policy and the leader) and plan

Seeds are independent replicates, so comparisons use a plain paired t
interval over seeds (no resampling correction).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..budget import Budget, BudgetLedger
from ..findings import Action, Category, Evidence, Finding, Severity
from ..schema import Comparison, DiscoveryReport, ExperimentResult, PlanStep, Profile, Recommendation
from ..stats.compare import default_rope, holm, rope_verdict
from ..stats.intervals import t_interval
from . import env as E
from .probes import probe_env
from .trials import TrialResult, Variant, run_trial

BASE_SEEDS = 3
MAX_SEEDS = 5
EVAL_EPISODES = 10
SIMPLICITY = {"PPO": 1, "DQN": 2, "SAC": 2, "TD3": 3, "RAINBOW": 3, "RECURRENT_PPO": 4}
ON_POLICY = ("PPO", "RECURRENT_PPO")


def default_rl_budget() -> Budget:
    return Budget(time_s=900.0, env_steps=200_000)


def _rank(v: Variant) -> float:
    return SIMPLICITY.get(v.algorithm, 5) + 0.5 * (int(v.obs_normalize) + int(v.rnd))


def _window(algorithm: str, mean_len: Optional[float], steps: int) -> int:
    k = 4.0 if algorithm in ON_POLICY else 1.0
    w = int(np.clip(k * (mean_len or 200.0), 512, 8192))
    return int(max(64, min(w, steps // 4)))


def _experiment(v: Variant, trials: List[TrialResult], steps: int) -> ExperimentResult:
    trials = sorted(trials, key=lambda t: t.seed)
    scores = tuple(t.score for t in trials)
    ok = [s for s in scores if s is not None]
    interval = None
    if len(ok) >= 2:
        iv = t_interval(ok)
        interval = (iv.lo, iv.hi)
    errs = [t.error for t in trials if t.error]
    return ExperimentResult(
        family=v.key, config={**v.to_dict(), "steps": steps}, stage="trial", metric="return",
        higher_is_better=True, fold_ids=tuple(f"seed{t.seed}" for t in trials), scores=scores,
        interval=interval, mean=float(np.mean(ok)) if ok else None,
        fit_time_s=round(sum(t.seconds for t in trials), 3), stop_reasons=tuple(t.status for t in trials),
        status="ok" if ok else "failed", error="; ".join(errs[:3]) if errs else None)


def _paired(a: ExperimentResult, b: ExperimentResult, rope: float) -> Optional[Comparison]:
    bm = dict(zip(b.fold_ids, b.scores))
    d = [s - bm[f] for f, s in zip(a.fold_ids, a.scores)
         if s is not None and bm.get(f) is not None]
    if len(d) < 2:
        return None
    iv = t_interval(d)
    finite = math.isfinite(iv.lo) and math.isfinite(iv.hi)
    return Comparison(a=a.key, b=b.key, n_pairs=len(d), delta_mean=iv.mean,
                      delta_interval=(iv.lo, iv.hi) if finite else None, rope=rope,
                      verdict=rope_verdict(iv.lo, iv.hi, rope) if finite else "undetermined",
                      p_value=iv.p_two_sided, method="paired t over seeds")


def _vs_random(a: ExperimentResult, random_mean: float, rope: float) -> Optional[Comparison]:
    ok = [s - random_mean for s in a.scores if s is not None]
    if len(ok) < 2:
        return None
    iv = t_interval(ok)
    return Comparison(a=a.key, b="landmark:random_policy:{}", n_pairs=len(ok), delta_mean=iv.mean,
                      delta_interval=(iv.lo, iv.hi), rope=rope, verdict=rope_verdict(iv.lo, iv.hi, rope),
                      p_value=iv.p_two_sided, method="t over seeds vs the random policy's mean return")


class _Runner:
    def __init__(self, factory, wrappers, obs_stats, ledger, steps, mean_len, eval_episodes, verbose):
        self.factory, self.wrappers, self.obs_stats = factory, wrappers, obs_stats
        self.ledger, self.steps, self.mean_len = ledger, steps, mean_len
        self.eval_episodes, self.verbose = eval_episodes, verbose
        self.trials: Dict[str, List[TrialResult]] = {}
        self.variants: Dict[str, Variant] = {}
        self.per_trial_s: Dict[str, float] = {}
        self.skipped: List[Tuple[str, str]] = []

    def run(self, v: Variant, seeds: Sequence[int]) -> bool:
        self.variants[v.key] = v
        for s in seeds:
            if any(t.seed == s for t in self.trials.get(v.key, [])):
                continue
            pred = self.per_trial_s.get(v.key) or (max(self.per_trial_s.values()) if self.per_trial_s else 0.0)
            budget_steps = self.ledger.budget.env_steps
            if (self.ledger.remaining_s() < pred or self.ledger.exhausted()
                    or (budget_steps is not None and self.ledger.env_steps + self.steps > budget_steps)):
                self.skipped.append((v.key, f"seed {s}: budget"))
                self.ledger.record_cut("rl_trials", f"{v.key} seed {s}", "budget exhausted")
                return False
            if self.verbose:
                print(f"[discovery] trial {v.key} seed {s} ({self.steps} steps)", flush=True)
            r = run_trial(self.factory, v, wrappers=self.wrappers, seed=s, steps=self.steps,
                          window=_window(v.algorithm, self.mean_len, self.steps),
                          eval_episodes=self.eval_episodes, obs_stats=self.obs_stats)
            self.ledger.record_steps(self.steps)
            self.ledger.record_fit(self.steps)
            self.per_trial_s[v.key] = max(self.per_trial_s.get(v.key, 0.0), r.seconds)
            self.trials.setdefault(v.key, []).append(r)
        return True

    def result(self, key: str) -> ExperimentResult:
        return _experiment(self.variants[key], self.trials.get(key, []), self.steps)


def _profile_only_report(sections, findings, limitations, budget, ledger, seed, summary) -> DiscoveryReport:
    rec = Recommendation(family=None, verdict="abstain", metric="return",
                         rationale=tuple(limitations[:1]) or ("No algorithms could be compared.",))
    return DiscoveryReport(domain="rl", profile=Profile(domain="rl", n_total=summary.get("probe_steps"),
                                                        sections=sections),
                           findings=findings, recommendation=rec, limitations=limitations,
                           input_summary=summary, budget_requested=budget.to_dict(), budget_used=ledger.to_dict(),
                           seed=seed)


def discover_rl(env: Any, *, families: Optional[Sequence[str]] = None, budget: Optional[Budget] = None,
                decisions: Any = None, seed: int = 0, store: Any = None, verbose: bool = False,
                eval_episodes: int = EVAL_EPISODES) -> DiscoveryReport:
    from ..ml.pipeline import _Applier, _normalise_decisions
    budget = budget or default_rl_budget()
    ledger = BudgetLedger(budget)
    ap = _Applier(_normalise_decisions(decisions))
    limitations: List[str] = []
    factory, inst, note = E.env_factory(env)
    if note:
        limitations.append(note.capitalize() + ".")
    name = getattr(getattr(inst, "spec", None), "id", None) or type(inst).__name__
    summary: Dict[str, Any] = {"env": name, "metric": "return"}

    if E.is_multi_agent(inst):
        agents = list(getattr(inst, "possible_agents", []) or [])
        sec = {"spaces": {"agents": agents}}
        f = Finding(id="limitation.multi_agent", category=Category.LIMITATION, severity=Severity.ADVISORY,
                    statement=f"Multi-agent environment ({len(agents)} agents). Discovery profiles it but does not "
                              "yet compare multi-agent algorithms; see the multi-agent training guide.",
                    evidence=(Evidence(test="api_detection", details={"agents": agents[:20]}),),
                    confidence="high", confidence_basis="PettingZoo API")
        return _profile_only_report(sec, [f], ["Multi-agent algorithm comparison is not available yet."],
                                    budget, ledger, seed, summary)

    probe_steps = int(min(10_000, max(2_000, (budget.env_steps or 200_000) // 20)))
    _log(verbose, f"probing {name} with a random policy ({probe_steps} steps)")
    sections, findings, facts = probe_env(factory, inst, steps=probe_steps, seed=seed)
    ledger.record_steps(facts["rollout"]["steps"])
    summary["probe_steps"] = int(facts["rollout"]["steps"])
    if not facts["supported"] or not facts["flat_obs"]:
        if not facts["flat_obs"]:
            limitations.append("Only flat vector observations are compared in this version (image / dict "
                               "observations are profiled, not trialled).")
        return _profile_only_report(sections, findings, limitations or ["Unsupported spaces."], budget, ledger,
                                    seed, summary)

    rets = facts["rollout"]["returns"]
    random_mean = float(rets.mean()) if len(rets) else 0.0
    mean_len = float(facts["rollout"]["lengths"].mean()) if len(facts["rollout"]["lengths"]) else None
    wrappers: Dict[str, Any] = {}
    rescale = next((f for f in findings if f.id == "env.action_rescale"), None)
    if rescale is not None and ap.applies(rescale):
        wrappers["action_rescale"] = rescale.suggested_action.params

    discrete = facts["action"]["type"] == "Discrete"
    base = ["PPO", "DQN"] if discrete else ["SAC", "TD3", "PPO"]
    if families is not None:
        want = {f.upper() for f in families}
        base = [a for a in base if a in want]
        if not base:
            raise ValueError(f"none of {sorted(want)} applies to this action space; options: "
                             f"{['PPO', 'DQN'] if discrete else ['SAC', 'TD3', 'PPO']}")
    planned_variants = int(bool(facts.get("obs_badly_scaled"))) + int(bool(facts.get("sparse"))) + \
        int(bool(facts.get("pomdp") and discrete))
    remaining = (budget.env_steps or 200_000) - ledger.env_steps
    # budget split: every planned trial plus a reserve for refinement (up to MAX_SEEDS - BASE_SEEDS extra
    # seeds for the leader and one runner-up)
    reserve_trials = 2 * (MAX_SEEDS - BASE_SEEDS)
    steps = int(np.clip(remaining / ((len(base) + planned_variants) * BASE_SEEDS + reserve_trials), 2_000, 200_000))
    summary.update({"steps_per_trial": steps, "seeds": BASE_SEEDS, "eval_episodes": eval_episodes,
                    "random_return": random_mean})
    obs_stats = facts.get("obs_stats")
    runner = _Runner(factory, wrappers, obs_stats, ledger, steps, mean_len, eval_episodes, verbose)
    seeds = [seed * 1000 + i for i in range(BASE_SEEDS)]

    # ---- stage A: base algorithms
    for a in base:
        runner.run(Variant(a), seeds)
    results = {k: runner.result(k) for k in runner.trials}

    def leader() -> Optional[str]:
        ok = [k for k, r in results.items() if r.mean is not None]
        return max(ok, key=lambda k: results[k].mean) if ok else None

    lead = leader()
    rope = default_rope("return", scale=abs((results[lead].mean if lead else random_mean) - random_mean) or 1.0)

    # ---- stage B: variants the probes motivate
    variant_tests: List[Tuple[str, str, str]] = []           # (variant key, base key, finding kind)
    if lead is not None:
        lv = runner.variants[lead]
        if facts.get("obs_badly_scaled"):
            v = Variant(lv.algorithm, obs_normalize=True)
            runner.run(v, seeds)
            variant_tests.append((v.key, lead, "obs_normalize"))
        if facts.get("sparse"):
            v = Variant(lv.algorithm, rnd=True)
            runner.run(v, seeds)
            variant_tests.append((v.key, lead, "rnd"))
        if facts.get("pomdp") and discrete:
            v = Variant("RECURRENT_PPO")
            runner.run(v, seeds)
            ppo = Variant("PPO").key
            variant_tests.append((v.key, ppo if ppo in runner.trials else lead, "recurrent"))
    results = {k: runner.result(k) for k in runner.trials}
    lead = leader()

    # ---- refinement: more seeds while the leader cannot be separated from someone
    def undetermined() -> List[str]:
        if lead is None:
            return []
        return [k for k in results if k != lead and results[k].mean is not None and
                (c := _paired(results[k], results[lead], rope)) is not None and c.verdict == "undetermined"]

    def vs_random_undetermined() -> bool:
        if lead is None:
            return False
        r_rope = default_rope("return", scale=abs(results[lead].mean - random_mean) or 1.0)
        c = _vs_random(results[lead], random_mean, r_rope)
        return c is not None and c.verdict == "undetermined"

    extra = BASE_SEEDS
    while extra < MAX_SEEDS and (undetermined() or vs_random_undetermined()):
        s_new = [seed * 1000 + extra]
        keys = [lead] + sorted(undetermined(), key=lambda k: -results[k].mean)[:2]
        if not all(runner.run(runner.variants[k], s_new) for k in keys):
            break
        extra += 1
        results = {k: runner.result(k) for k in runner.trials}
        lead = leader()
    if lead is not None:
        rope = default_rope("return", scale=abs(results[lead].mean - random_mean) or 1.0)

    # ---- comparisons and recommendation
    comps: List[Comparison] = []
    rec, rf = _recommend(results, runner.variants, lead, random_mean, rope, comps)
    findings += rf
    # variant findings
    vt_comps = []
    for vk, bk, kind in variant_tests:
        if vk in results and bk in results:
            c = _paired(results[vk], results[bk], rope)
            vt_comps.append((vk, bk, kind, c))
    adj = holm([c.p_value if c else None for *_, c in vt_comps])
    variant_rows = []
    for (vk, bk, kind, c), q in zip(vt_comps, adj):
        ok = c is not None and c.verdict == "better" and q is not None and q < 0.05
        variant_rows.append({"variant": vk, "base": bk, "gain": c.delta_mean if c else None,
                             "interval": c.delta_interval if c else None, "verdict": c.verdict if c else None,
                             "p_holm": q, "recommended": ok})
        if c is not None:
            comps.append(c)
        if not ok:
            continue
        act = {"obs_normalize": Action("wrap_obs_normalize", {"mean": (obs_stats or {}).get("mean"),
                                                               "std": (obs_stats or {}).get("std")}),
               "rnd": Action("add_rnd", {}), "recurrent": Action("use_recurrent", {})}[kind]
        what = {"obs_normalize": "Normalising observations", "rnd": "An RND exploration bonus",
                "recurrent": "A recurrent (GRU) policy"}[kind]
        findings.append(Finding(
            id=f"rl.{kind}", category=Category.ENV, severity=Severity.ADVISORY,
            statement=(f"{what} improves {bk} by {c.delta_mean:+.2f} return (95% CI {c.delta_interval[0]:+.2f} "
                       f"to {c.delta_interval[1]:+.2f}) over {c.n_pairs} seeds."),
            evidence=(Evidence(test="paired_t_over_seeds", statistic=c.delta_mean, interval=c.delta_interval,
                               p_value=q, details={"variant": vk, "base": bk, "rope": rope}, seed=seed),),
            confidence="moderate", confidence_basis="paired seeds, Holm-adjusted across the variants tried",
            suggested_action=act, default_applied=False))

    for k, why in runner.skipped:
        fid = f"budget.skipped:variant={k.split(' ')[0]}"
        if not any(f.id == fid for f in findings):
            findings.append(Finding(
                id=fid, category=Category.BUDGET, severity=Severity.ADVISORY,
                statement=f"Some trials of {k} were not run ({why}); increase Budget(env_steps=..., time_s=...).",
                evidence=(Evidence(test="budget", details={"reason": why}),), confidence="high",
                confidence_basis="budget ledger"))
    limitations.append(f"Each trial trains for {steps:,} steps with the algorithm's default hyperparameters "
                       f"(networks {64}x{64}, no online tuning); longer training or tuning can change the "
                       "ranking, especially for off-policy methods that learn slowly at first.")
    plan = _plan(rec, list(ap.applied), wrappers)
    sections["trials"] = {"steps_per_trial": steps, "seeds_used": sorted({t.seed for ts in runner.trials.values()
                                                                          for t in ts}),
                          "eval_episodes": eval_episodes, "random_return": random_mean, "rope": rope,
                          "variants": variant_rows, "wrappers_in_trials": sorted(wrappers)}
    sections["applied_findings"] = list(ap.applied)
    seen, uniq = set(), []
    for f in findings:
        if f.id not in seen:
            seen.add(f.id)
            uniq.append(f)
    report = DiscoveryReport(
        domain="rl", profile=Profile(domain="rl", n_total=int(ledger.env_steps), sections=sections),
        findings=uniq, experiments=list(results.values()), comparisons=comps, recommendation=rec, plan=plan,
        limitations=limitations, input_summary=summary, budget_requested=budget.to_dict(),
        budget_used=ledger.to_dict(), seed=seed)
    for fid, d in ap.decisions.items():
        if fid in seen:
            report._decisions[fid] = d
    if store is not None:
        from ..ledger_bridge import record_report
        report.ledger_refs = record_report(report, store)
    _log(verbose, f"done: {rec.verdict} -> {rec.family}")
    return report


def _recommend(results: Dict[str, ExperimentResult], variants: Dict[str, Variant], lead: Optional[str],
               random_mean: float, rope: float, comps: List[Comparison]) -> Tuple[Recommendation, List[Finding]]:
    findings: List[Finding] = []
    if lead is None:
        return Recommendation(family=None, verdict="abstain", metric="return",
                              rationale=("Every trial failed; see the experiments' errors.",)), findings
    lr = results[lead]
    vr = _vs_random(lr, random_mean, rope)
    if vr is not None:
        comps.append(vr)
    if vr is None or vr.verdict != "better":
        findings.append(Finding(
            id="family.abstain", category=Category.FAMILY, severity=Severity.WARNING,
            statement=(f"No algorithm beat the random policy (return {random_mean:.2f}) by a margin that matters "
                       f"within the trial budget (best: {lead}, {lr.mean:.2f})."),
            evidence=(Evidence(test="t_over_seeds_vs_random", statistic=vr.delta_mean if vr else None,
                               interval=vr.delta_interval if vr else None, details={"rope": rope}),),
            confidence="low", confidence_basis="short trials; more steps may change this"))
        return Recommendation(family=None, verdict="abstain", metric="return",
                              rationale=(f"Best trial ({lead}, mean return {lr.mean:.2f}) is not clearly better "
                                         f"than random ({random_mean:.2f}) at {lr.config.get('steps')} steps.",
                                         "Increase Budget(env_steps=...) or check the reward signal."),
                              alternatives=(lead,)), findings
    others = {k: _paired(r, lr, rope) for k, r in results.items() if k != lead and r.mean is not None}
    others = {k: c for k, c in others.items() if c is not None}
    adj = holm([c.p_value for c in others.values()])
    from dataclasses import replace
    others = {k: replace(c, p_holm=q) for (k, c), q in zip(others.items(), adj)}
    comps.extend(others.values())
    equivalent = [k for k, c in others.items() if c.verdict == "equivalent"]
    und = [k for k, c in others.items() if c.verdict == "undetermined"]
    chosen = min([lead] + equivalent, key=lambda k: (_rank(variants[k]), k))
    rationale = []
    if chosen != lead:
        verdict = "tie_simplest_chosen"
        c = others[chosen]
        rationale.append(f"{lead} scored best ({lr.mean:.2f}), but {chosen} ({results[chosen].mean:.2f}) is "
                         f"equivalent within ±{rope:.2f} return over {c.n_pairs} seeds; the simpler one is chosen.")
    elif not und and not equivalent:
        verdict = "clear_winner"
        rationale.append(f"{lead} (mean return {lr.mean:.2f}) beats every other algorithm tried by more than "
                         f"±{rope:.2f} on the same seeds.")
    else:
        verdict = "undetermined"
        rationale.append(f"{lead} scored best ({lr.mean:.2f}) but cannot be separated from {', '.join(und)} "
                         "with the seeds and steps available.")
    rationale.append(f"It beats the random policy ({random_mean:.2f}) by {vr.delta_mean:+.2f} "
                     f"(95% CI {vr.delta_interval[0]:+.2f} to {vr.delta_interval[1]:+.2f}).")
    worse = [k for k, c in others.items() if c.verdict == "worse"]
    if worse:
        rationale.append(f"Ruled out: {', '.join(sorted(worse))}.")
    cr = results[chosen]
    return Recommendation(family=chosen, config=dict(cr.config), verdict=verdict, rationale=tuple(rationale),
                          expected_score=cr.mean, expected_interval=cr.interval, metric="return",
                          alternatives=tuple(sorted(set(und + equivalent + ([lead] if chosen != lead else []))
                                                    - {chosen})),
                          conditions=("Expected return is the mean over the trial seeds at the trial's step budget; "
                                      "it is not a converged score.",)), findings


def _plan(rec: Recommendation, applied: List[str], wrappers: Dict[str, Any]) -> List[PlanStep]:
    steps: List[PlanStep] = []
    if wrappers.get("action_rescale"):
        steps.append(PlanStep("environment", "Rescale actions from [-1, 1] to the environment's bounds.",
                              ("env.action_rescale",)))
    if rec.family:
        lines = ["# review and accept the findings you want applied, e.g.:"]
        lines += [f"report.accept({fid!r})" for fid in applied]
        lines += ["solution = report.build()", "agent, run = solution.fit(env)      # trains with tensor-optix"]
        steps.append(PlanStep("train", f"Train {rec.family} to convergence (tensor-optix stops it), with online "
                                       "tuning on; the trials ran without it.", code="\n".join(lines)))
        if rec.alternatives:
            steps.append(PlanStep("train", f"If the budget allows, also train {', '.join(rec.alternatives)} for "
                                           "longer: discovery could not separate them."))
    else:
        steps.append(PlanStep("environment", "Check the reward: discovery found no learning signal within the "
                                             "trial budget. Give more steps, or shape / densify the reward."))
    return steps


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[discovery] {msg}", flush=True)
