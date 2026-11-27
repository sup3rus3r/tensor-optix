"""
Tiny inline-SVG chart primitives for the static report (no JavaScript, no
external libraries). Every label is escaped; missing / infinite values are
drawn as "n/a" rather than breaking the chart.

Colours come from CSS custom properties defined in the template, so charts
follow the page's light / dark theme.
"""
from __future__ import annotations

import math
from html import escape
from typing import Optional, Sequence, Tuple

_W = 640          # drawing width (viewBox units; scales to container)
_LABEL_W = 190
_ROW_H = 26


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _fmt(x: Optional[float]) -> str:
    if not _finite(x):
        return "n/a"
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e5):
        return f"{x:.3g}"
    return f"{x:.4f}".rstrip("0").rstrip(".") if ax < 10 else f"{x:.2f}"


def _scale(lo: float, hi: float, a: float, b: float):
    span = hi - lo if hi > lo else 1.0
    return lambda v: a + (v - lo) / span * (b - a)


def forest_plot(rows: Sequence[Tuple[str, Optional[float], Optional[float], Optional[float]]], *,
                reference: Optional[float] = None, rope: Optional[float] = None,
                title: str = "", higher_is_better: bool = True, highlight: Optional[str] = None) -> str:
    """
    Point estimates with interval bars, one row per item.

    ``reference`` +- ``rope`` is shaded (e.g. the leader's mean, or 0 for
    paired deltas) so "practically equivalent" is visible at a glance.
    """
    if not rows:
        return ""
    vals = []
    for _, m, lo, hi in rows:
        vals += [v for v in (m, lo, hi) if _finite(v)]
    if reference is not None and _finite(reference):
        vals.append(reference)
        if rope:
            vals += [reference - rope, reference + rope]
    if not vals:
        return ""
    vmin, vmax = min(vals), max(vals)
    pad = (vmax - vmin) * 0.08 or (abs(vmax) * 0.05 + 1e-9)
    vmin, vmax = vmin - pad, vmax + pad
    x0, x1 = _LABEL_W + 10, _W - 70
    sx = _scale(vmin, vmax, x0, x1)
    top = 28 if title else 8
    h = top + _ROW_H * len(rows) + 24
    out = [f'<svg class="chart" viewBox="0 0 {_W} {h}" role="img" aria-label="{escape(title or "forest plot")}">']
    if title:
        out.append(f'<text x="0" y="16" class="chart-title">{escape(title)}</text>')
    if reference is not None and _finite(reference):
        if rope:
            xa, xb = sx(reference - rope), sx(reference + rope)
            out.append(f'<rect x="{xa:.1f}" y="{top - 4}" width="{max(xb - xa, 0.5):.1f}" '
                       f'height="{_ROW_H * len(rows) + 4}" class="rope"/>')
        xr = sx(reference)
        out.append(f'<line x1="{xr:.1f}" y1="{top - 4}" x2="{xr:.1f}" y2="{top + _ROW_H * len(rows)}" class="ref"/>')
    for i, (label, m, lo, hi) in enumerate(rows):
        y = top + i * _ROW_H + _ROW_H / 2
        cls = "row-hi" if highlight is not None and label == highlight else ""
        out.append(f'<text x="{_LABEL_W}" y="{y + 4:.1f}" text-anchor="end" class="label {cls}">{escape(str(label))}</text>')
        if _finite(lo) and _finite(hi):
            out.append(f'<line x1="{sx(lo):.1f}" y1="{y:.1f}" x2="{sx(hi):.1f}" y2="{y:.1f}" class="ci"/>')
        if _finite(m):
            out.append(f'<circle cx="{sx(m):.1f}" cy="{y:.1f}" r="4.5" class="pt {cls}"/>')
            out.append(f'<text x="{_W - 4}" y="{y + 4:.1f}" text-anchor="end" class="val">{_fmt(m)}</text>')
        else:
            out.append(f'<text x="{x0}" y="{y + 4:.1f}" class="val">n/a</text>')
    ya = top + _ROW_H * len(rows) + 16
    out.append(f'<text x="{x0}" y="{ya}" class="axis">{_fmt(vmin)}</text>')
    out.append(f'<text x="{x1}" y="{ya}" text-anchor="end" class="axis">{_fmt(vmax)}</text>')
    arrow = "higher is better &#8594;" if higher_is_better else "&#8592; lower is better"
    out.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{ya}" text-anchor="middle" class="axis">{arrow}</text>')
    out.append("</svg>")
    return "".join(out)


def bar_chart(rows: Sequence[Tuple[str, Optional[float]]], *, threshold: Optional[float] = None,
              title: str = "", value_max: Optional[float] = None) -> str:
    """Horizontal bars (e.g. per-feature predictiveness), with an optional
    threshold line (e.g. the permutation-null 95th percentile)."""
    if not rows:
        return ""
    vals = [v for _, v in rows if _finite(v)]
    if threshold is not None and _finite(threshold):
        vals.append(threshold)
    vmax = value_max if value_max is not None else (max(vals) if vals else 1.0)
    vmin = min(0.0, min(vals) if vals else 0.0)
    if vmax <= vmin:
        vmax = vmin + 1.0
    x0, x1 = _LABEL_W + 10, _W - 70
    sx = _scale(vmin, vmax, x0, x1)
    top = 28 if title else 8
    h = top + _ROW_H * len(rows) + 10
    out = [f'<svg class="chart" viewBox="0 0 {_W} {h}" role="img" aria-label="{escape(title or "bar chart")}">']
    if title:
        out.append(f'<text x="0" y="16" class="chart-title">{escape(title)}</text>')
    for i, (label, v) in enumerate(rows):
        y = top + i * _ROW_H
        out.append(f'<text x="{_LABEL_W}" y="{y + _ROW_H / 2 + 4:.1f}" text-anchor="end" class="label">{escape(str(label))}</text>')
        if _finite(v):
            xa, xb = sorted((sx(0.0), sx(v)))
            out.append(f'<rect x="{xa:.1f}" y="{y + 5}" width="{max(xb - xa, 0.5):.1f}" height="{_ROW_H - 10}" class="bar"/>')
            out.append(f'<text x="{_W - 4}" y="{y + _ROW_H / 2 + 4:.1f}" text-anchor="end" class="val">{_fmt(v)}</text>')
    if threshold is not None and _finite(threshold):
        xt = sx(threshold)
        out.append(f'<line x1="{xt:.1f}" y1="{top}" x2="{xt:.1f}" y2="{top + _ROW_H * len(rows)}" class="ref"/>')
    out.append("</svg>")
    return "".join(out)


def line_chart(series: Sequence[Tuple[str, Sequence[float], Sequence[float]]], *,
               title: str = "", x_label: str = "", y_label: str = "") -> str:
    """Simple multi-series line chart (e.g. learning curves); series =
    (name, xs, ys). Up to 6 series get distinct theme colours."""
    pts = [(x, y) for _, xs, ys in series for x, y in zip(xs, ys) if _finite(x) and _finite(y)]
    if not pts:
        return ""
    xs_all, ys_all = [p[0] for p in pts], [p[1] for p in pts]
    xmin, xmax, ymin, ymax = min(xs_all), max(xs_all), min(ys_all), max(ys_all)
    if ymax == ymin:
        ymax = ymin + 1.0
    W, H, L, R, T, B = _W, 240, 56, 16, 28 if title else 10, 34
    sx, sy = _scale(xmin, xmax, L, W - R), _scale(ymin, ymax, H - B, T)
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{escape(title or "line chart")}">']
    if title:
        out.append(f'<text x="0" y="16" class="chart-title">{escape(title)}</text>')
    out.append(f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" class="axis-line"/>')
    out.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H - B}" class="axis-line"/>')
    for i, (name, xs, ys) in enumerate(series):
        p = [(sx(x), sy(y)) for x, y in zip(xs, ys) if _finite(x) and _finite(y)]
        if not p:
            continue
        d = " ".join(f"{a:.1f},{b:.1f}" for a, b in p)
        out.append(f'<polyline points="{d}" class="series s{i % 6}"/>')
        out.append(f'<text x="{p[-1][0]:.1f}" y="{p[-1][1] - 6:.1f}" text-anchor="end" class="val s{i % 6}t">{escape(str(name))}</text>')
    out.append(f'<text x="{L}" y="{H - 10}" class="axis">{_fmt(xmin)}</text>')
    out.append(f'<text x="{W - R}" y="{H - 10}" text-anchor="end" class="axis">{_fmt(xmax)}</text>')
    if x_label:
        out.append(f'<text x="{(L + W - R) / 2:.1f}" y="{H - 10}" text-anchor="middle" class="axis">{escape(x_label)}</text>')
    out.append(f'<text x="{L - 6}" y="{T + 10}" text-anchor="end" class="axis">{_fmt(ymax)}</text>')
    out.append(f'<text x="{L - 6}" y="{H - B}" text-anchor="end" class="axis">{_fmt(ymin)}</text>')
    if y_label:
        out.append(f'<text x="4" y="{(T + H - B) / 2:.1f}" class="axis">{escape(y_label)}</text>')
    out.append("</svg>")
    return "".join(out)
