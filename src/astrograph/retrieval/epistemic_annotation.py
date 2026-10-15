"""Fail-closed machine suggestions for document epistemic-span review."""
from __future__ import annotations

import hashlib
import re

from pydantic import BaseModel, ConfigDict, Field

from .document_governance import ClaimEpistemicStatus
from .structured_document import DocumentElement, DocumentElementKind


class EpistemicSpanCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1)
    element_id: str = Field(min_length=1)
    source_locator: str = Field(min_length=1)
    section_path: list[str] = Field(min_length=1)
    source_span_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    excerpt: str = Field(min_length=1, max_length=300)
    suggested_status: ClaimEpistemicStatus
    trigger_codes: list[str] = Field(min_length=1)
    human_decision: None = None
    runtime_evidence_eligible: bool = False


_PROPOSAL_PATTERNS = (
    ("PROPOSAL_MODAL", re.compile(r"拟|计划|建议|可采用|可以|可通过|需要|需对|后续|设计")),
    ("FUTURE_MODAL", re.compile(r"将(?:采用|实现|部署|构建|建立|完成)|未来")),
)
_ESTIMATE = re.compile(
    r"(?:\b\d+(?:\.\d+)?\s*(?:GB|MB|KB|W|ms|s|%)\b)|"
    r"(?:参数量|显存|内存|存储空间|功耗|耗时|推理速度|资源需求)",
    re.IGNORECASE,
)


def suggest_epistemic_span(element: DocumentElement) -> EpistemicSpanCandidate:
    text = element.text_representation.strip()
    triggers: list[str] = []
    if element.kind == DocumentElementKind.IMAGE:
        status = ClaimEpistemicStatus.ILLUSTRATIVE
        triggers.append("UNREVIEWED_IMAGE_DEFAULT")
    elif _ESTIMATE.search(text):
        status = ClaimEpistemicStatus.ENGINEERING_ESTIMATE
        triggers.append("RESOURCE_OR_NUMERIC_ESTIMATE")
    else:
        for code, pattern in _PROPOSAL_PATTERNS:
            if pattern.search(text):
                triggers.append(code)
        status = ClaimEpistemicStatus.DESIGN_PROPOSAL
        if not triggers:
            triggers.append("DOCUMENT_LEVEL_PROPOSAL_DEFAULT")
    identity = hashlib.sha256(
        f"{element.source_locator}\x1f{hashlib.sha256(text.encode('utf-8')).hexdigest()}".encode(
            "utf-8"
        )
    ).hexdigest()[:20]
    excerpt = re.sub(r"\s+", " ", text)[:300]
    return EpistemicSpanCandidate(
        candidate_id=f"epistemic-span:{identity}",
        element_id=element.element_id,
        source_locator=element.source_locator,
        section_path=element.section_path,
        source_span_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        excerpt=excerpt,
        suggested_status=status,
        trigger_codes=triggers,
    )


def suggest_epistemic_spans(
    elements: list[DocumentElement],
) -> list[EpistemicSpanCandidate]:
    return [suggest_epistemic_span(element) for element in elements]


__all__ = [
    "EpistemicSpanCandidate",
    "suggest_epistemic_span",
    "suggest_epistemic_spans",
]
