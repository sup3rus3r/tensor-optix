"""
tensor_optix.utils.console — encoding-safe console output.

Progress output uses box-drawing and arrow glyphs (━, →, ×, ✓ ...). On a
stream whose encoding cannot represent them — typically a Windows console or
a redirected/piped stdout using cp1252 — a plain print() raises
UnicodeEncodeError, which would abort a training run just for being verbose.
safe_print() degrades the unencodable characters to "?" instead.
"""
import sys


def safe_print(*args, sep: str = " ", end: str = "\n", file=None, flush: bool = False) -> None:
    """print() that never raises UnicodeEncodeError on a narrow-encoding stream."""
    stream = file if file is not None else sys.stdout
    if stream is None:  # e.g. pythonw / detached process with no stdout
        return
    text = sep.join(str(a) for a in args) + end
    try:
        stream.write(text)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "ascii"
        stream.write(text.encode(encoding, errors="replace").decode(encoding))
    if flush:
        try:
            stream.flush()
        except Exception:
            pass
