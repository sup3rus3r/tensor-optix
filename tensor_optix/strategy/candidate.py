"""Serializable, versioned identity of one grammar operation + its
concrete parameter values.

to_op() only ever reconstructs one of the dataclasses listed in
grammar.OP_REGISTRY, via keyword args matching that dataclass's declared
fields -- never eval()/exec(). This is the concrete mechanism that makes
"closed grammar, no arbitrary code" an enforceable code property rather
than a policy statement.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

from .grammar import OP_REGISTRY, StrategyOp

GRAMMAR_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StrategyCandidate:
    candidate_id: str
    op_name: str
    op_version: int
    params: Dict[str, Any] = field(default_factory=dict)
    grammar_schema_version: int = GRAMMAR_SCHEMA_VERSION

    @staticmethod
    def from_op(op: StrategyOp, op_version: int = 1) -> "StrategyCandidate":
        params = dict(asdict(op))
        candidate_id = _fingerprint(op.op_name, op_version, params)
        return StrategyCandidate(
            candidate_id=candidate_id, op_name=op.op_name, op_version=op_version,
            params=params, grammar_schema_version=GRAMMAR_SCHEMA_VERSION,
        )

    def to_op(self) -> StrategyOp:
        if self.op_name not in OP_REGISTRY:
            raise ValueError(f"Unknown op_name {self.op_name!r} -- not in OP_REGISTRY")
        cls = OP_REGISTRY[self.op_name]
        return cls(**self.params)


def _fingerprint(op_name: str, op_version: int, params: Dict[str, Any]) -> str:
    payload = json.dumps(
        {"op_name": op_name, "op_version": op_version, "params": params}, sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
