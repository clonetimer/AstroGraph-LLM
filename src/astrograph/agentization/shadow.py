from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from .protocol import CapabilityProtocolError, sha256_json

@dataclass(frozen=True)
class ShadowPlanResult:
    valid: bool
    proposed: tuple[str,...]
    reference: tuple[str,...]
    missing: tuple[str,...]
    unexpected: tuple[str,...]
    order_violations: tuple[str,...]
    plan_sha256: str

    def to_mapping(self): return {"valid":self.valid,"proposed":list(self.proposed),"reference":list(self.reference),"missing":list(self.missing),"unexpected":list(self.unexpected),"order_violations":list(self.order_violations),"plan_sha256":self.plan_sha256,"execution_authority":"REFERENCE_WORKFLOW_ONLY"}

def compare_plan(proposed: list[str], reference: list[str], allowed: set[str]) -> ShadowPlanResult:
    if not all(isinstance(x,str) and x for x in proposed): raise CapabilityProtocolError("INVALID_SHADOW_PLAN","proposed tools must be non-empty strings")
    unexpected=tuple(x for x in proposed if x not in allowed)
    missing=tuple(x for x in reference if x not in proposed)
    pos={x:i for i,x in enumerate(proposed)}; violations=[]
    for a,b in zip(reference,reference[1:]):
        if a in pos and b in pos and pos[a]>pos[b]: violations.append(f"{a}>{b}")
    digest=sha256_json({"proposed":proposed,"reference":reference})
    return ShadowPlanResult(not unexpected and not violations,tuple(proposed),tuple(reference),missing,unexpected,tuple(violations),digest)
