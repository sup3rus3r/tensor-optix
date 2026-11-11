"""Environment/provenance capture for the experiment ledger.

Every function here is defensive: a missing library, missing git binary, or
unreadable hostname must never prevent an experiment from being recorded.
"""

import socket
import subprocess
import sys
from typing import Any, Dict, Optional


def _git_revision() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _tensor_optix_version() -> Optional[str]:
    try:
        import importlib.metadata

        return importlib.metadata.version("tensor-optix")
    except Exception:
        pass
    try:
        import tensor_optix

        return getattr(tensor_optix, "__version__", None)
    except Exception:
        return None


def _library_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {"python": sys.version.split()[0]}
    # Read versions without importing: importing tensorflow/jax just to
    # stamp a ledger row costs seconds per experiment in a torch-only run.
    import importlib.metadata as _md
    dist_names = {
        "tensorflow": ("tensorflow", "tensorflow-cpu", "tensorflow-gpu", "tensorflow-macos"),
    }
    for module_name in ("numpy", "torch", "tensorflow", "jax", "gymnasium", "optuna"):
        version = None
        loaded = sys.modules.get(module_name)
        if loaded is not None:
            version = getattr(loaded, "__version__", None)
        if not version:
            for dist in dist_names.get(module_name, (module_name,)):
                try:
                    version = _md.version(dist)
                    break
                except Exception:
                    continue
        if version:
            versions[module_name] = str(version)
    return versions


def _hostname() -> Optional[str]:
    try:
        return socket.gethostname()
    except Exception:
        return None


def capture_environment() -> Dict[str, Any]:
    """Return a best-effort snapshot of provenance metadata for the current run.

    Never raises — every individual capture is independently guarded so a
    missing optional dependency or unavailable git binary degrades a single
    field to None/empty rather than failing the whole capture.
    """
    return {
        "tensor_optix_version": _tensor_optix_version(),
        "source_revision": _git_revision(),
        "library_versions": _library_versions(),
        "hostname": _hostname(),
    }
