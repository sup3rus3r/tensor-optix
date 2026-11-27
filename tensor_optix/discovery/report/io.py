"""Embed / extract the report JSON inside the HTML page (exact round-trip)."""
from __future__ import annotations

import json
import re
from typing import Any, Dict

SCRIPT_ID = "tensor-optix-discovery"
_PATTERN = re.compile(
    r'<script type="application/json" id="' + SCRIPT_ID + r'">(.*?)</script>', re.DOTALL,
)


def embed_json(d: Dict[str, Any]) -> str:
    """
    A ``<script type="application/json">`` block holding the report. ``</`` is
    escaped as ``<\\/`` (valid JSON, inert HTML) so no string inside the report
    -- e.g. a column named ``</script>`` -- can terminate the block early.
    """
    text = json.dumps(d, ensure_ascii=False, allow_nan=False, sort_keys=False)
    text = text.replace("</", "<\\/")
    return f'<script type="application/json" id="{SCRIPT_ID}">{text}</script>'


def extract_json(html: str) -> Dict[str, Any]:
    m = _PATTERN.search(html)
    if m is None:
        raise ValueError("no embedded tensor-optix discovery report found in this HTML")
    return json.loads(m.group(1))
