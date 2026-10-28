"""Duplicate-work detection via sampled-parameter fingerprinting.

De-dupes on exact sampled param dict equality (all keys, exact values).
Near-duplicate / tolerance-based dedup is explicitly out of scope — that is
an asynchronous-selection-bias concern (proposal Sec 6.3), separate and
later than this scheduler's job.
"""

import hashlib
import json
from typing import Any, Dict


def fingerprint(params: Dict[str, Any]) -> str:
    """Stable hash over the sampled param dict, independent of key order."""
    encoded = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
