"""
The static HTML template for discovery reports.

Kept as Python string constants (not a package-data file) so it always ships
with the wheel. Self-contained: inline CSS, no JavaScript, no external fonts
or scripts -- the page works offline and can be attached to an email.
Light / dark follow ``prefers-color-scheme``; print styles expand everything.
"""

CSS = r"""
:root {
  --bg: #fbfaf7; --panel: #ffffff; --ink: #1d2230; --muted: #5d6475; --line: #e3e1da;
  --accent: #2b59c3; --accent-soft: #e8eefb; --ok: #2f7d4f; --ok-soft: #e6f3eb;
  --warn: #a86400; --warn-soft: #fbf0dc; --crit: #b3261e; --crit-soft: #fbe5e3;
  --info: #5d6475; --info-soft: #eef0f3; --adv: #6a4bb5; --adv-soft: #efeafb;
  --code: #f3f2ee; --rope: rgba(43, 89, 195, 0.10);
  --s0: #2b59c3; --s1: #c2571a; --s2: #2f7d4f; --s3: #8a3ffc; --s4: #b3261e; --s5: #00838f;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #12141a; --panel: #1a1d25; --ink: #e8e9ee; --muted: #9aa1b2; --line: #2c303c;
    --accent: #7fa2ff; --accent-soft: #1f2a44; --ok: #6cc28f; --ok-soft: #173024;
    --warn: #f0b35a; --warn-soft: #3a2c14; --crit: #ff8a80; --crit-soft: #3d1c1a;
    --info: #9aa1b2; --info-soft: #242833; --adv: #b39dff; --adv-soft: #2a2342;
    --code: #222632; --rope: rgba(127, 162, 255, 0.14);
    --s0: #7fa2ff; --s1: #ff9e5e; --s2: #6cc28f; --s3: #c39bff; --s4: #ff8a80; --s5: #4dd0e1;
  }
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.wrap { max-width: 1040px; margin: 0 auto; padding: 28px 16px 64px; }
header.top { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px 24px;
  align-items: baseline; border-bottom: 1px solid var(--line); padding-bottom: 14px; margin-bottom: 22px; }
header.top h1 { font-size: 22px; margin: 0; letter-spacing: -0.01em; }
header.top .meta { color: var(--muted); font-size: 13px; }
h2 { font-size: 18px; margin: 34px 0 12px; letter-spacing: -0.005em; }
h3 { font-size: 15px; margin: 18px 0 8px; }
p { margin: 6px 0 10px; }
.muted { color: var(--muted); }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; margin: 12px 0; }
.hero { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr); gap: 16px; }
@media (max-width: 760px) { .hero { grid-template-columns: 1fr; } }
.hero .big { font-size: 24px; font-weight: 650; margin: 2px 0 4px; }
.kv { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 4px 14px; font-size: 14px; }
.kv dt { color: var(--muted); }
.kv dd { margin: 0; overflow-wrap: anywhere; }
.pill { display: inline-block; font-size: 12px; font-weight: 600; padding: 1px 9px; border-radius: 999px;
  border: 1px solid transparent; white-space: nowrap; }
.sev-critical { background: var(--crit-soft); color: var(--crit); border-color: var(--crit); }
.sev-warning { background: var(--warn-soft); color: var(--warn); border-color: var(--warn); }
.sev-advisory { background: var(--adv-soft); color: var(--adv); border-color: var(--adv); }
.sev-info { background: var(--info-soft); color: var(--info); border-color: var(--line); }
.v-clear_winner, .v-better { background: var(--ok-soft); color: var(--ok); border-color: var(--ok); }
.v-tie_simplest_chosen, .v-equivalent { background: var(--accent-soft); color: var(--accent); border-color: var(--accent); }
.v-undetermined { background: var(--warn-soft); color: var(--warn); border-color: var(--warn); }
.v-abstain, .v-worse { background: var(--crit-soft); color: var(--crit); border-color: var(--crit); }
.d-accepted { background: var(--ok-soft); color: var(--ok); }
.d-rejected { background: var(--info-soft); color: var(--muted); text-decoration: line-through; }
.d-pending { background: var(--info-soft); color: var(--muted); }
.finding { border-left: 4px solid var(--line); }
.finding.sev-critical-b { border-left-color: var(--crit); }
.finding.sev-warning-b { border-left-color: var(--warn); }
.finding.sev-advisory-b { border-left-color: var(--adv); }
.finding .head { display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: center; margin-bottom: 4px; }
.finding .stmt { font-weight: 560; }
.finding .fid { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; color: var(--muted); overflow-wrap: anywhere; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 600; font-size: 13px; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.table-wrap { overflow-x: auto; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; }
code { background: var(--code); padding: 1px 5px; border-radius: 4px; overflow-wrap: anywhere; }
pre { background: var(--code); padding: 10px 12px; border-radius: 8px; overflow-x: auto; margin: 8px 0; }
details { margin: 6px 0; }
details > summary { cursor: pointer; color: var(--accent); font-size: 13px; }
ul.plain { margin: 6px 0; padding-left: 20px; }
ol.plan { padding-left: 22px; }
ol.plan li { margin: 8px 0; }
svg.chart { width: 100%; height: auto; display: block; margin: 6px 0; }
svg.chart .label, svg.chart .val, svg.chart .axis { fill: var(--muted); font-size: 12px; }
svg.chart .label.row-hi { fill: var(--ink); font-weight: 650; }
svg.chart .chart-title { fill: var(--ink); font-size: 13px; font-weight: 600; }
svg.chart .ci { stroke: var(--muted); stroke-width: 2; }
svg.chart .pt { fill: var(--accent); }
svg.chart .pt.row-hi { fill: var(--ok); }
svg.chart .ref { stroke: var(--ink); stroke-width: 1; stroke-dasharray: 3 3; opacity: 0.6; }
svg.chart .rope { fill: var(--rope); }
svg.chart .bar { fill: var(--accent); opacity: 0.85; }
svg.chart .axis-line { stroke: var(--line); }
svg.chart .series { fill: none; stroke-width: 2; }
svg.chart .s0 { stroke: var(--s0); } svg.chart .s1 { stroke: var(--s1); } svg.chart .s2 { stroke: var(--s2); }
svg.chart .s3 { stroke: var(--s3); } svg.chart .s4 { stroke: var(--s4); } svg.chart .s5 { stroke: var(--s5); }
svg.chart .s0t { fill: var(--s0); } svg.chart .s1t { fill: var(--s1); } svg.chart .s2t { fill: var(--s2); }
svg.chart .s3t { fill: var(--s3); } svg.chart .s4t { fill: var(--s4); } svg.chart .s5t { fill: var(--s5); }
nav.toc { font-size: 13px; display: flex; flex-wrap: wrap; gap: 4px 14px; margin: 4px 0 6px; }
nav.toc a { color: var(--accent); text-decoration: none; }
footer { margin-top: 40px; color: var(--muted); font-size: 12px; border-top: 1px solid var(--line); padding-top: 12px; }
@media print {
  body { background: #fff; }
  .panel { break-inside: avoid; }
  details > *:not(summary) { display: block !important; }
  nav.toc { display: none; }
}
"""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="tensor-optix discovery {version}">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<div class="wrap">
{body}
</div>
{data}
</body>
</html>
"""
