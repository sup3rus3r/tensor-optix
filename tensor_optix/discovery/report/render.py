"""
Render a discovery report dict to static HTML (and to plain text).

The renderer is a pure function of ``DiscoveryReport.to_dict()``: it never
sees live objects, so the HTML can only show what the report actually
contains, and the embedded JSON makes the page round-trip exactly.

Every value that came from the user's data (column names, category values,
finding ids, notes) is passed through ``html.escape``.
"""
from __future__ import annotations

import math
from html import escape
from typing import Any, Dict, List, Optional

from . import svg
from .io import embed_json
from .template import CSS, PAGE

_SEV_ORDER = {"critical": 3, "warning": 2, "advisory": 1, "info": 0}
_VERDICT_TEXT = {
    "clear_winner": "Clear winner",
    "tie_simplest_chosen": "Tie — simplest family chosen",
    "undetermined": "Too close to call",
    "abstain": "No learnable signal found",
}
_MAX_TABLE_ROWS = 200


def e(x: Any) -> str:
    return escape("" if x is None else str(x), quote=True)


def num(x: Any, digits: int = 4) -> str:
    if x is None or isinstance(x, bool) or not isinstance(x, (int, float)):
        return "n/a" if x is None else e(x)
    if isinstance(x, float) and not math.isfinite(x):
        return "n/a"
    if isinstance(x, int):
        return f"{x:,}"
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e6):
        return f"{x:.3g}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".") if ax < 1000 else f"{x:,.1f}"


def interval(iv: Optional[List[float]]) -> str:
    if not iv or len(iv) != 2:
        return "n/a"
    return f"[{num(iv[0])}, {num(iv[1])}]"


def pill(text: str, cls: str) -> str:
    return f'<span class="pill {e(cls)}">{e(text)}</span>'


def _kv(d: Dict[str, Any]) -> str:
    if not d:
        return '<p class="muted">none</p>'
    rows = []
    for k, v in d.items():
        rows.append(f"<dt>{e(k)}</dt><dd>{_value(v)}</dd>")
    return f'<dl class="kv">{"".join(rows)}</dl>'


def _value(v: Any) -> str:
    if isinstance(v, dict):
        if not v:
            return '<span class="muted">—</span>'
        return "; ".join(f"{e(k)}: {_value(x)}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        if not v:
            return '<span class="muted">—</span>'
        if all(not isinstance(x, (dict, list, tuple)) for x in v):
            shown = ", ".join(_value(x) for x in v[:20])
            return shown + (f" … (+{len(v) - 20})" if len(v) > 20 else "")
        return f"{len(v)} item(s)"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return num(v)
    return e(v)


def _table(rows: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> str:
    if not rows:
        return '<p class="muted">none</p>'
    if columns is None:
        columns = []
        for r in rows:
            for k in r:
                if k not in columns:
                    columns.append(k)
    head = "".join(f"<th>{e(c)}</th>" for c in columns)
    body = []
    for r in rows[:_MAX_TABLE_ROWS]:
        cells = []
        for c in columns:
            v = r.get(c)
            cls = ' class="num"' if isinstance(v, (int, float)) and not isinstance(v, bool) else ""
            cells.append(f"<td{cls}>{_value(v)}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    more = ""
    if len(rows) > _MAX_TABLE_ROWS:
        more = f'<p class="muted">… {len(rows) - _MAX_TABLE_ROWS} more row(s) in the JSON data.</p>'
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>{more}'


def _sorted_findings(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    return sorted(d.get("findings") or [], key=lambda f: (-_SEV_ORDER.get(f["severity"], 0), f["id"]))


def _decision(d: Dict[str, Any], fid: str) -> Dict[str, Any]:
    return (d.get("decisions") or {}).get(fid) or {"state": "pending"}


def _key_label(key: str) -> str:
    """Family name from an experiment key ``stage:family:config`` (any other string as is)."""
    parts = str(key).split(":", 2)
    return parts[1] if len(parts) >= 2 else str(key)


def _family_label(exp: Dict[str, Any]) -> str:
    cfg = exp.get("config") or {}
    tag = cfg.get("name") or ", ".join(f"{k}={v}" for k, v in cfg.items() if not isinstance(v, (dict, list)))
    return f"{exp['family']} ({tag})" if tag else exp["family"]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _header(d: Dict[str, Any]) -> str:
    domain = {"ml": "dataset", "rl": "environment"}.get(d.get("domain"), d.get("domain"))
    meta = (f"{e(domain)} · created {e(d.get('created_at'))} · tensor-optix {e(d.get('tensor_optix_version'))}"
            f" · report schema v{e(d.get('schema_version'))}")
    toc = ("<nav class='toc'><a href='#summary'>Summary</a><a href='#understanding'>Understanding</a>"
           "<a href='#risks'>Risks</a><a href='#families'>Families tried</a>"
           "<a href='#recommendation'>Recommendation</a><a href='#plan'>Plan</a>"
           "<a href='#limits'>Limitations &amp; reproducibility</a></nav>")
    return (f'<header class="top"><h1>Discovery report</h1><div class="meta">{meta}</div></header>{toc}')


def _summary(d: Dict[str, Any]) -> str:
    rec = d.get("recommendation") or {}
    verdict = rec.get("verdict", "undetermined")
    fam = rec.get("family")
    if fam:
        big = e(fam)
    elif verdict == "abstain":
        big = "No family recommended"
    else:
        big = "No single family recommended"
    score_line = ""
    if rec.get("expected_score") is not None:
        score_line = (f"<p>Expected {e(rec.get('metric'))} on held-out data: <b>{num(rec['expected_score'])}</b> "
                      f"<span class='muted'>95% CI {interval(rec.get('expected_interval'))}</span></p>")
    reasons = "".join(f"<li>{e(r)}</li>" for r in (rec.get("rationale") or [])[:3])
    reasons = f"<ul class='plain'>{reasons}</ul>" if reasons else ""
    alts = rec.get("alternatives") or []
    alt_line = f"<p class='muted'>Still in contention: {e(', '.join(alts))}</p>" if alts else ""

    risks = [f for f in _sorted_findings(d) if _SEV_ORDER.get(f["severity"], 0) >= 1][:3]
    if risks:
        items = "".join(f"<li>{pill(f['severity'], 'sev-' + f['severity'])} {e(f['statement'])}</li>" for f in risks)
        risk_html = f"<ul class='plain'>{items}</ul>"
    else:
        risk_html = "<p class='muted'>No risks above INFO were found.</p>"
    counts = {}
    for f in d.get("findings") or []:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1
    count_line = " · ".join(f"{n} {s}" for s, n in sorted(counts.items(), key=lambda kv: -_SEV_ORDER.get(kv[0], 0)))
    return (
        '<section id="summary"><div class="hero">'
        f'<div class="panel"><div class="muted">Recommended family</div><div class="big">{big}</div>'
        f'{pill(_VERDICT_TEXT.get(verdict, verdict), "v-" + verdict)}{score_line}{reasons}{alt_line}</div>'
        f'<div class="panel"><div class="muted">Top risks</div>{risk_html}'
        f'<p class="muted">{e(count_line or "no findings")}</p></div>'
        "</div></section>"
    )


def _understanding(d: Dict[str, Any]) -> str:
    p = d.get("profile") or {}
    top = {
        "rows / steps available": p.get("n_total"),
        "used for discovery": p.get("n_sampled"),
        "fingerprint": (p.get("fingerprint") or "")[:16] or None,
    }
    parts = ['<section id="understanding"><h2>Understanding the data</h2>',
             f'<div class="panel">{_kv({k: v for k, v in top.items() if v is not None})}']
    if p.get("sampling"):
        parts.append(f"<h3>How the sample was chosen</h3>{_kv(p['sampling'])}")
    parts.append("</div>")
    for name, section in (p.get("sections") or {}).items():
        title = name.replace("_", " ").capitalize()
        if isinstance(section, list) and section and all(isinstance(r, dict) for r in section):
            body = _table(section)
        elif isinstance(section, dict):
            body = _kv(section)
        else:
            body = f"<p>{_value(section)}</p>"
        parts.append(f'<div class="panel"><h3>{e(title)}</h3>{body}</div>')
    parts.append("</section>")
    return "".join(parts)


def _evidence_html(ev: Dict[str, Any]) -> str:
    bits = {
        "test": ev.get("test"), "statistic": ev.get("statistic"), "null value": ev.get("null_value"),
        "p-value": ev.get("p_value"), "interval": interval(ev.get("interval")) if ev.get("interval") else None,
        "effect size": ev.get("effect_size"), "n": ev.get("n"), "seed": ev.get("seed"),
    }
    kv = {k: v for k, v in bits.items() if v is not None}
    details = ev.get("details") or {}
    return _kv(kv) + (_kv(details) if details else "")


def _finding_card(d: Dict[str, Any], f: Dict[str, Any]) -> str:
    dec = _decision(d, f["id"])
    state = dec.get("state", "pending")
    act = f.get("suggested_action")
    act_html = ""
    if act:
        params = ", ".join(f"{k}={v!r}" for k, v in (act.get("params") or {}).items())
        act_html = f"<p>Suggested action: <code>{e(act['kind'])}({e(params)})</code>"
        if f.get("default_applied"):
            act_html += " <span class='muted'>(already applied in discovery's own experiments)</span>"
        act_html += "</p>"
    ev_html = "".join(f"<details><summary>Evidence: {e(ev.get('test'))}</summary>{_evidence_html(ev)}</details>"
                      for ev in f.get("evidence") or [])
    note = f" — {e(dec.get('note'))}" if dec.get("note") else ""
    return (
        f'<div class="panel finding sev-{e(f["severity"])}-b">'
        f'<div class="head">{pill(f["severity"], "sev-" + f["severity"])}'
        f'{pill(f["category"].replace("_", " "), "sev-info")}'
        f'{pill("decision: " + state, "d-" + state)}'
        f'<span class="muted">confidence: {e(f.get("confidence"))}</span></div>'
        f'<div class="stmt">{e(f["statement"])}</div>'
        f'<div class="fid">id: {e(f["id"])}{note}</div>'
        f'{act_html}'
        f'{"<p class=muted>" + e(f["confidence_basis"]) + "</p>" if f.get("confidence_basis") else ""}'
        f'{ev_html}'
        f'<details><summary>How to accept / reject</summary>'
        f'<pre>report.accept({e(repr(f["id"]))})\nreport.reject({e(repr(f["id"]))}, note="why")</pre></details>'
        "</div>"
    )


def _risks(d: Dict[str, Any]) -> str:
    fs = [f for f in _sorted_findings(d) if f.get("category") != "features"]
    parts = ['<section id="risks"><h2>Findings and risks</h2>']
    if not fs:
        parts.append('<p class="muted">No findings.</p></section>')
        return "".join(parts)
    parts.append("<p class='muted'>Nothing below is applied automatically. "
                 "Accept or reject each finding by id; <code>report.build()</code> applies only accepted ones.</p>")
    for f in fs:
        parts.append(_finding_card(d, f))
    parts.append("</section>")
    return "".join(parts)


def _features(d: Dict[str, Any]) -> str:
    sec = ((d.get("profile") or {}).get("sections") or {}).get("features")
    fs = [f for f in _sorted_findings(d) if f.get("category") == "features"]
    if not sec and not fs:
        return ""
    parts = ['<section id="features"><h2>Feature opportunities</h2>',
             "<p class='muted'>Kinds of features worth engineering. Each was triggered by a property of the data "
             "and kept only if adding it improved the model on the same folds. Not applied unless accepted.</p>"]
    for f in fs:
        parts.append(_finding_card(d, f))
    tested = (sec or {}).get("tested") or []
    if tested:
        rows = [{"family": t.get("family"), "triggered by": (t.get("trigger") or {}).get("test"),
                 "gain": t.get("gain"), "95% CI": interval(t.get("gain_interval")) if t.get("gain_interval") else None,
                 "p (Holm)": t.get("p_holm"), "folds improved": t.get("folds_improved"),
                 "verdict": t.get("verdict") or t.get("status"),
                 "recommended": "yes" if t.get("recommended") else "no"} for t in tested]
        parts.append(f"<h3>Families tested (model: {e((sec or {}).get('probe_family'))})</h3>")
        parts.append(_table(rows, list(rows[0].keys())))
    elif not fs:
        parts.append("<p class='muted'>No property of the data suggested a feature family to try.</p>")
    scal = (sec or {}).get("scaling") or {}
    if scal.get("tested"):
        rows = [{"scaler": t.get("method"), "columns": ", ".join((t.get("columns") or [])[:5]),
                 "gain vs standard": t.get("gain"),
                 "95% CI": interval(t.get("gain_interval")) if t.get("gain_interval") else None,
                 "p (Holm)": t.get("p_holm"), "verdict": t.get("verdict") or t.get("status"),
                 "recommended": "yes" if t.get("recommended") else "no"} for t in scal["tested"]]
        parts.append("<h3>Scalers tested</h3>")
        parts.append(_table(rows, list(rows[0].keys())))
    if scal.get("note"):
        parts.append(f"<p class='muted'>Scaling: {e(scal['note'])}.</p>")
    for note in (sec or {}).get("not_tried") or []:
        parts.append(f"<p class='muted'>Not tried: {e(', '.join(note.get('families') or []))} -- "
                     f"{e(note.get('reason'))}</p>")
    parts.append("</section>")
    return "".join(parts)


def _families(d: Dict[str, Any]) -> str:
    exps = d.get("experiments") or []
    rec = d.get("recommendation") or {}
    parts = ['<section id="families"><h2>Families tried</h2>']
    if not exps:
        parts.append('<p class="muted">No experiments were run.</p></section>')
        return "".join(parts)
    by_stage: Dict[str, List[Dict[str, Any]]] = {}
    for x in exps:
        by_stage.setdefault(x.get("stage", "trial"), []).append(x)
    rec_label = None
    for stage in ("trial", "confirm", "landmark"):
        group = by_stage.get(stage)
        if not group:
            continue
        hib = bool(group[0].get("higher_is_better", True))
        metric = group[0].get("metric", "")
        ok = [x for x in group if x.get("status") == "ok" and x.get("mean") is not None]
        rows = [(_family_label(x), x.get("mean"), *(x.get("interval") or [None, None])) for x in ok]
        rows.sort(key=lambda r: (r[1] if r[1] is not None else -math.inf), reverse=hib)
        if rec.get("family"):
            for x in ok:
                if x["family"] == rec["family"]:
                    rec_label = _family_label(x)
                    break
        title = {"trial": "Short trials", "confirm": "Confirmation on held-out data",
                 "landmark": "Landmark probes (cheap models that reveal the problem's shape)"}[stage]
        parts.append(f'<div class="panel"><h3>{e(title)}</h3>')
        if rows:
            parts.append(svg.forest_plot(rows, title=f"{metric} (mean and 95% CI)",
                                         higher_is_better=hib, highlight=rec_label))
        table_rows = []
        for x in group:
            table_rows.append({
                "family": _family_label(x), "metric": x.get("metric"), "mean": x.get("mean"),
                "95% CI": interval(x.get("interval")), "folds": len(x.get("scores") or []),
                "fit time (s)": x.get("fit_time_s"),
                "stop": ", ".join(sorted(set(x.get("stop_reasons") or []))) or None,
                "status": x.get("status"), "error": x.get("error"),
            })
        parts.append(_table(table_rows))
        parts.append("</div>")
    comps = d.get("comparisons") or []
    if comps:
        crow = [{
            "comparison": f"{_key_label(c['a'])} vs {_key_label(c['b'])}",
            "Δ (positive = first better)": c.get("delta_mean"),
            "95% CI": interval(c.get("delta_interval")), "ROPE ±": c.get("rope"),
            "verdict": c.get("verdict"), "p (Holm)": c.get("p_holm"), "method": c.get("method"),
        } for c in comps]
        parts.append(f'<div class="panel"><h3>Paired comparisons</h3>'
                     f'<p class="muted">Same folds for every family; differences inside the ROPE do not matter in practice.</p>'
                     f'{_table(crow)}</div>')
    parts.append("</section>")
    return "".join(parts)


def _recommendation(d: Dict[str, Any]) -> str:
    rec = d.get("recommendation")
    parts = ['<section id="recommendation"><h2>Recommendation and why</h2><div class="panel">']
    if not rec:
        parts.append('<p class="muted">No recommendation was produced.</p></div></section>')
        return "".join(parts)
    v = rec.get("verdict", "undetermined")
    parts.append(f"<p>{pill(_VERDICT_TEXT.get(v, v), 'v-' + v)} "
                 f"<b>{e(rec.get('family') or 'no single family')}</b></p>")
    if rec.get("config"):
        parts.append(f"<h3>Configuration</h3>{_kv(rec['config'])}")
    if rec.get("rationale"):
        parts.append("<h3>Why</h3><ul class='plain'>" + "".join(f"<li>{e(r)}</li>" for r in rec["rationale"]) + "</ul>")
    if rec.get("conditions"):
        parts.append("<h3>Conditions</h3><ul class='plain'>" + "".join(f"<li>{e(c)}</li>" for c in rec["conditions"]) + "</ul>")
    if rec.get("alternatives"):
        parts.append(f"<p class='muted'>Alternatives still in contention: {e(', '.join(rec['alternatives']))}</p>")
    parts.append("</div></section>")
    return "".join(parts)


def _plan(d: Dict[str, Any]) -> str:
    steps = d.get("plan") or []
    parts = ['<section id="plan"><h2>Suggested plan</h2>']
    if not steps:
        parts.append('<p class="muted">No plan.</p></section>')
        return "".join(parts)
    items = []
    for s in steps:
        refs = s.get("finding_refs") or []
        ref = f" <span class='muted'>(from {e(', '.join(refs))})</span>" if refs else ""
        code = f"<pre>{e(s['code'])}</pre>" if s.get("code") else ""
        items.append(f"<li><b>{e(s['kind'])}</b> — {e(s['description'])}{ref}{code}</li>")
    parts.append(f'<div class="panel"><ol class="plan">{"".join(items)}</ol></div></section>')
    return "".join(parts)


def _limits(d: Dict[str, Any]) -> str:
    lim = d.get("limitations") or []
    lim_html = ("<ul class='plain'>" + "".join(f"<li>{e(x)}</li>" for x in lim) + "</ul>") if lim \
        else "<p class='muted'>None recorded.</p>"
    repro = {"seed": d.get("seed"), "tensor-optix": d.get("tensor_optix_version"),
             "report schema": d.get("schema_version"),
             "data fingerprint": (d.get("profile") or {}).get("fingerprint")}
    repro.update(d.get("provenance") or {})
    return (
        '<section id="limits"><h2>Limitations, budget and reproducibility</h2>'
        f'<div class="panel"><h3>Limitations</h3>{lim_html}</div>'
        f'<div class="panel"><h3>Budget</h3><h3 class="muted">requested</h3>{_kv(d.get("budget_requested") or {})}'
        f'<h3 class="muted">used</h3>{_kv(d.get("budget_used") or {})}</div>'
        f'<div class="panel"><h3>Reproducibility</h3>{_kv({k: v for k, v in repro.items() if v is not None})}'
        "<p class='muted'>Load this report back in Python with "
        "<code>DiscoveryReport.from_html(path)</code>.</p></div></section>"
    )


def render_html(d: Dict[str, Any]) -> str:
    body = "".join([
        _header(d), _summary(d), _understanding(d), _risks(d), _features(d), _families(d),
        _recommendation(d), _plan(d), _limits(d),
        "<footer>Generated by tensor-optix discovery. Statistics are estimates from a sample of the data; "
        "see Limitations. Nothing in this report was applied to your data or models.</footer>",
    ])
    return PAGE.format(version=e(d.get("tensor_optix_version")), title="Discovery report",
                       css=CSS, body=body, data=embed_json(d))


def render_text(d: Dict[str, Any]) -> str:
    rec = d.get("recommendation") or {}
    lines = ["tensor-optix discovery report", "=" * 30]
    v = rec.get("verdict", "undetermined")
    lines.append(f"Recommendation: {rec.get('family') or 'none'}  [{_VERDICT_TEXT.get(v, v)}]")
    if rec.get("expected_score") is not None:
        lines.append(f"  expected {rec.get('metric')}: {num(rec['expected_score'])} "
                     f"(95% CI {interval(rec.get('expected_interval'))})")
    for r in rec.get("rationale") or []:
        lines.append(f"  - {r}")
    fs = _sorted_findings(d)
    lines.append("")
    lines.append(f"Findings ({len(fs)}):")
    for f in fs:
        state = _decision(d, f["id"]).get("state", "pending")
        lines.append(f"  [{f['severity'].upper():8s}] {f['statement']}")
        lines.append(f"             id={f['id']}  decision={state}")
    if d.get("limitations"):
        lines.append("")
        lines.append("Limitations:")
        lines.extend(f"  - {x}" for x in d["limitations"])
    return "\n".join(lines)
