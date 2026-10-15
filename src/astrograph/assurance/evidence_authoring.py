
"""Governed rule-authoring receipts for human/agent-assisted evidence pipelines.

Code agents may transform, test, and propose rules.  They are never accepted as
scientific provenance and cannot promote a rule without pinned sources and an
explicit human review receipt.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from enum import Enum
import hashlib
import json
import re
from typing import Any, Mapping

_SHA = re.compile(r"^[0-9a-f]{64}$")

class AuthoringStage(str, Enum):
    DRAFT = "DRAFT"
    SOURCE_PINNED = "SOURCE_PINNED"
    EXECUTABLE = "EXECUTABLE"
    UNIT_VALIDATED = "UNIT_VALIDATED"
    DEVELOPMENT_VALIDATED = "DEVELOPMENT_VALIDATED"
    FROZEN = "FROZEN"

_ORDER = {v: i for i, v in enumerate(AuthoringStage)}

@dataclass(frozen=True)
class EvidenceAuthoringReceipt:
    schema: str
    evidence_id: str
    fault_id: str
    stage: AuthoringStage
    source_artifact_sha256s: tuple[str, ...] = ()
    source_independence_groups: tuple[str, ...] = ()
    executable_rule_sha256: str | None = None
    unit_test_receipt_sha256: str | None = None
    development_validation_receipt_sha256: str | None = None
    human_approval_receipt_sha256: str | None = None
    agent_transformation_receipt_sha256: str | None = None
    authoring_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema != "astrograph-evidence-authoring-receipt.v1":
            raise ValueError("unsupported authoring schema")
        if not self.evidence_id or not self.fault_id:
            raise ValueError("evidence_id and fault_id required")
        for name, value in asdict(self).items():
            if name.endswith("sha256") and value is not None and not _SHA.fullmatch(value):
                raise ValueError(f"{name} must be lowercase SHA-256")
            if name.endswith("sha256s"):
                for digest in value:
                    if not _SHA.fullmatch(digest):
                        raise ValueError(f"invalid digest in {name}")
        if _ORDER[self.stage] >= _ORDER[AuthoringStage.SOURCE_PINNED] and not self.source_artifact_sha256s:
            raise ValueError("source-pinned stage requires source artifacts")
        if _ORDER[self.stage] >= _ORDER[AuthoringStage.EXECUTABLE] and self.executable_rule_sha256 is None:
            raise ValueError("executable stage requires executable rule")
        if _ORDER[self.stage] >= _ORDER[AuthoringStage.UNIT_VALIDATED] and self.unit_test_receipt_sha256 is None:
            raise ValueError("unit-validated stage requires test receipt")
        if _ORDER[self.stage] >= _ORDER[AuthoringStage.DEVELOPMENT_VALIDATED] and self.development_validation_receipt_sha256 is None:
            raise ValueError("development-validated stage requires receipt")
        if self.stage is AuthoringStage.FROZEN and self.human_approval_receipt_sha256 is None:
            raise ValueError("frozen stage requires explicit human approval")
        expected = self.compute_sha256()
        if self.authoring_receipt_sha256 is not None and self.authoring_receipt_sha256 != expected:
            raise ValueError("authoring_receipt_sha256 mismatch")

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("authoring_receipt_sha256", None)
        data["stage"] = self.stage.value
        return data

    def compute_sha256(self) -> str:
        raw = json.dumps(self.payload(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

def promote(receipt: EvidenceAuthoringReceipt, target: AuthoringStage, **artifacts: Any) -> EvidenceAuthoringReceipt:
    if _ORDER[target] != _ORDER[receipt.stage] + 1:
        raise ValueError("authoring promotion must advance exactly one stage")
    candidate = replace(receipt, stage=target, authoring_receipt_sha256=None, **artifacts)
    return replace(candidate, authoring_receipt_sha256=candidate.compute_sha256())
