"""Hash-pinned reviewed document-claim enrichment for Mission RAG hits.

This module is intentionally contextual only.  It reuses the governed document
claim admission/mapping artifacts, but it does not build a fusion ranking and it
never grants a diagnostic vote to a retrieved hit.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from astrograph.fusion.document_claims import (
    DocumentClaimAdmissionTrace,
    DocumentClaimMappingBundle,
)
from astrograph.retrieval.document_claims import evaluate_document_claim_admissions


class ReviewedClaimRAGBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_trace_path: str
    claim_trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mapping_bundle_path: str
    mapping_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _read_pinned(path: Path, expected_sha256: str, label: str) -> str:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8 JSON") from exc


class ReviewedClaimRAGEnricher:
    """Exact source SHA + locator bridge from reviewed claims to retrieval hits."""

    def __init__(
        self,
        *,
        trace: DocumentClaimAdmissionTrace,
        mapping_bundle: DocumentClaimMappingBundle,
        trace_sha256: str,
        mapping_bundle_sha256: str,
    ) -> None:
        recomputed = evaluate_document_claim_admissions(trace.claims, trace.policy)
        if recomputed != trace.admissions:
            raise ValueError("stored document claim admissions differ from policy recomputation")
        admission_by_id = {item.claim_id: item for item in recomputed}
        claim_by_id = {item.claim_id: item for item in trace.claims}
        mapping_by_claim = {item.claim_id: item for item in mapping_bundle.mappings}
        if len(mapping_by_claim) != len(mapping_bundle.mappings):
            raise ValueError("each reviewed claim may have at most one fault mapping")

        index: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for claim_id, mapping in mapping_by_claim.items():
            claim = claim_by_id.get(claim_id)
            admission = admission_by_id.get(claim_id)
            if claim is None or admission is None:
                raise ValueError("reviewed claim mapping references unknown claim")
            # Context enrichment is allowed only for claims that survived the full
            # stored policy recomputation.  This is stricter than mere retrieval
            # relevance and prevents a reviewed-looking but ineligible claim from
            # acquiring SUPPORTS/CONTRADICTS semantics at runtime.
            if not admission.formal_fusion_eligible:
                continue
            if mapping.calibration_receipt_sha256 != admission.calibration_receipt_sha256:
                raise ValueError("reviewed mapping and admission calibration receipts differ")
            key = (claim.source_document_sha256, claim.source_locator)
            index.setdefault(key, []).append(
                {
                    "claim_id": claim.claim_id,
                    "mapping_id": mapping.mapping_id,
                    "target_fault_id": mapping.target_fault_id,
                    "polarity": mapping.polarity,
                    "source_document_id": claim.source_document_id,
                    "source_provenance_group_id": claim.source_provenance_group_id,
                    "policy_sha256": admission.policy_sha256,
                    "calibration_receipt_sha256": admission.calibration_receipt_sha256,
                    "source_independence_receipt_sha256": admission.source_independence_receipt_sha256,
                }
            )
        self._index = index
        self.trace_sha256 = trace_sha256
        self.mapping_bundle_sha256 = mapping_bundle_sha256

    @classmethod
    def from_binding(cls, binding: ReviewedClaimRAGBinding) -> "ReviewedClaimRAGEnricher":
        trace_path = Path(binding.claim_trace_path)
        mapping_path = Path(binding.mapping_bundle_path)
        if not trace_path.is_file() or not mapping_path.is_file():
            raise ValueError("reviewed claim RAG artifact is missing")
        trace = DocumentClaimAdmissionTrace.model_validate_json(
            _read_pinned(trace_path, binding.claim_trace_sha256, "reviewed claim trace")
        )
        mapping_bundle = DocumentClaimMappingBundle.model_validate_json(
            _read_pinned(mapping_path, binding.mapping_bundle_sha256, "reviewed claim mapping bundle")
        )
        return cls(
            trace=trace,
            mapping_bundle=mapping_bundle,
            trace_sha256=binding.claim_trace_sha256,
            mapping_bundle_sha256=binding.mapping_bundle_sha256,
        )

    @property
    def reviewed_mapping_count(self) -> int:
        return sum(len(items) for items in self._index.values())

    def enrich_hit(self, source: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of a retrieval hit with reviewed stance metadata if exact."""
        result = dict(source)
        source_sha = source.get("source_sha256")
        locator = source.get("source_locator")
        if not source_sha or not locator:
            result["reviewed_claim_match_status"] = "NO_EXACT_SOURCE_KEY"
            return result
        matches = list(self._index.get((str(source_sha), str(locator)), []))
        if not matches:
            result["reviewed_claim_match_status"] = "NO_REVIEWED_MAPPING"
            return result
        if len(matches) != 1:
            result["reviewed_claim_match_status"] = "AMBIGUOUS_REVIEWED_MAPPING"
            result["reviewed_claim_match_count"] = len(matches)
            return result
        match = matches[0]
        document_id = source.get("document_id")
        if document_id and str(document_id) != str(match["source_document_id"]):
            result["reviewed_claim_match_status"] = "DOCUMENT_ID_MISMATCH"
            return result
        result.update(
            {
                "reviewed_claim_match_status": "EXACT_REVIEWED_MAPPING",
                "reviewed_claim_polarity": match["polarity"],
                "reviewed_target_fault_id": match["target_fault_id"],
                "reviewed_claim_id": match["claim_id"],
                "reviewed_mapping_id": match["mapping_id"],
                "reviewed_provenance_group_id": match["source_provenance_group_id"],
                "reviewed_policy_sha256": match["policy_sha256"],
                "reviewed_claim_trace_sha256": self.trace_sha256,
                "reviewed_mapping_bundle_sha256": self.mapping_bundle_sha256,
                "diagnostic_vote_allowed": False,
            }
        )
        return result

    def enrich_hits(self, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.enrich_hit(item) for item in sources]


__all__ = ["ReviewedClaimRAGBinding", "ReviewedClaimRAGEnricher"]
