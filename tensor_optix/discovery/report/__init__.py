"""Static HTML rendering of discovery reports."""
from .io import embed_json, extract_json
from .render import render_html, render_text

__all__ = ["embed_json", "extract_json", "render_html", "render_text"]
