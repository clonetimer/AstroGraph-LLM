"""Non-filtering reference-use annotation for Evidence.

Application rounds showed that calibration qualification must control the claim
strength of Evidence without automatically deleting every research-use signal
from candidate generation.  This contract keeps those concerns separate.
"""
from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, ConfigDict, Field

from .trusted_reference import ReferenceQualification, ReferenceUseDecision


class EvidenceReferenceUseClass(str, Enum):
    FORMAL_SUPPORT = "FORMAL_SUPPORT"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    REJECTED = "REJECTED"


class EvidenceReferenceUse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-evidence-reference-use.v1"
    reference_id: str = Field(min_length=1)
    scope_id: str = Field(min_length=1)
    use_class: EvidenceReferenceUseClass
    candidate_generation_allowed: bool
    formal_support_allowed: bool
    reasons: tuple[str, ...] = ()
    claim_boundary: str = (
        "Reference qualification controls Evidence claim strength. RESEARCH_ONLY evidence may contribute to "
        "candidate generation but cannot independently support a formal calibrated claim; REJECTED reference "
        "evidence is not admitted."
    )


def evidence_reference_use(
    qualification: ReferenceQualification,
    *,
    scope_id: str,
) -> EvidenceReferenceUse:
    if qualification.decision is ReferenceUseDecision.QUALIFIED:
        use_class = EvidenceReferenceUseClass.FORMAL_SUPPORT
        candidate_allowed = True
        formal_allowed = True
    elif qualification.decision is ReferenceUseDecision.RESEARCH_ONLY:
        use_class = EvidenceReferenceUseClass.RESEARCH_ONLY
        candidate_allowed = True
        formal_allowed = False
    else:
        use_class = EvidenceReferenceUseClass.REJECTED
        candidate_allowed = False
        formal_allowed = False
    return EvidenceReferenceUse(
        reference_id=qualification.reference_id,
        scope_id=scope_id,
        use_class=use_class,
        candidate_generation_allowed=candidate_allowed,
        formal_support_allowed=formal_allowed,
        reasons=qualification.reasons,
    )
