"""Adapt admitted document claims to the existing KG_FORWARD ranking contract."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.retrieval.document_governance import (
    ClaimTemporalScope, ClaimEpistemicStatus, DocumentEvidenceClaim,
    DocumentVisibilityClass, DocumentAuthority, DocumentReviewState,
)
from astrograph.retrieval.document_claims import evaluate_document_claim_admissions
from astrograph.retrieval.models import (
    DocumentClaimAdmission,
    DocumentClaimAdmissionPolicy,
)

from .models import (
    FaultRanking,
    KGFaultCoverageDeclaration,
    RankedFault,
    RankingSource,
    ScoreSemantics,
)


def _sha(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class DocumentClaimFaultMapping(BaseModel):
    """One exact, reviewed claim-to-fault mapping; no pattern matching is allowed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-document-claim-fault-mapping.v1"
    mapping_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    target_fault_id: str = Field(min_length=1)
    polarity: Literal["SUPPORTS", "CONTRADICTS"]
    calibrated_weight: float = Field(gt=0.0, le=1.0)
    review_status: Literal["HUMAN_REVIEWED"] = "HUMAN_REVIEWED"
    reviewer_id: str = Field(min_length=1)
    mapping_approval_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def mapping_sha256(self) -> str:
        return _sha(self.model_dump(mode="json"))


class DocumentClaimAdmissionTrace(BaseModel):
    """Frozen document claims, their asserted admissions, and the governing policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claims: list[DocumentEvidenceClaim]
    admissions: list[DocumentClaimAdmission]
    policy: DocumentClaimAdmissionPolicy


class DocumentClaimMappingBundle(BaseModel):
    """Frozen exact mappings; an empty reviewed set is a valid zero-vote bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-document-claim-mapping-bundle.v1"
    bundle_id: str = Field(min_length=1)
    mappings: list[DocumentClaimFaultMapping] = Field(default_factory=list)

    @model_validator(mode="after")
    def mappings_are_unique(self) -> "DocumentClaimMappingBundle":
        ids = [item.mapping_id for item in self.mappings]
        if len(ids) != len(set(ids)):
            raise ValueError("document claim mapping IDs must be unique")
        return self


class DocumentClaimOnlineShadowConfig(BaseModel):
    """Fail-closed online binding to frozen document-graph claim artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    execution_mode: Literal["SHADOW_ONLY"] = "SHADOW_ONLY"
    claim_trace_path: str | None = None
    claim_trace_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    mapping_bundle_path: str | None = None
    mapping_bundle_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_artifact_id: str | None = None

    @model_validator(mode="after")
    def frozen_assets_match_switch(self) -> "DocumentClaimOnlineShadowConfig":
        assets = (
            self.claim_trace_path,
            self.claim_trace_sha256,
            self.mapping_bundle_path,
            self.mapping_bundle_sha256,
            self.source_artifact_id,
        )
        if self.enabled and any(value is None for value in assets):
            raise ValueError("enabled document claim shadow requires all frozen artifact bindings")
        if not self.enabled and any(value is not None for value in assets):
            raise ValueError("disabled document claim shadow must not bind artifact paths")
        return self


class DocumentClaimOnlineShadowResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-document-claim-online-shadow.v1"
    status: Literal["DISABLED", "SHADOW_ZERO_VOTE", "SHADOW_RANKING_READY"]
    execution_mode: Literal["SHADOW_ONLY"] = "SHADOW_ONLY"
    ranking_build: "DocumentClaimRankingBuild | None" = None
    claim_trace_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    mapping_bundle_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    candidate_fusion_modified: Literal[False] = False
    reason_codes: list[str] = Field(default_factory=list)


def _read_frozen_json(path: Path, *, expected_sha256: str, label: str) -> str:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8 JSON") from exc


def build_online_document_claim_shadow(
    *,
    config: DocumentClaimOnlineShadowConfig,
    episode_id: str,
    registered_fault_ids: list[str],
    observation_time_s: float | None = None,
) -> DocumentClaimOnlineShadowResult:
    """Reload and revalidate frozen document claims without modifying online fusion."""

    if not config.enabled:
        return DocumentClaimOnlineShadowResult(
            status="DISABLED", reason_codes=["DOCUMENT_CLAIM_SHADOW_DISABLED"]
        )
    if observation_time_s is None or not math.isfinite(observation_time_s) or observation_time_s < 0:
        raise ValueError("enabled document claim shadow requires a trusted finite observation_time_s")
    trace_path = Path(config.claim_trace_path or "")
    mapping_path = Path(config.mapping_bundle_path or "")
    if not trace_path.is_file() or not mapping_path.is_file():
        raise ValueError("document claim shadow artifact is missing")
    trace_text = _read_frozen_json(
        trace_path,
        expected_sha256=config.claim_trace_sha256 or "",
        label="document claim trace",
    )
    mapping_text = _read_frozen_json(
        mapping_path,
        expected_sha256=config.mapping_bundle_sha256 or "",
        label="document claim mapping bundle",
    )
    actual_trace_sha = config.claim_trace_sha256 or ""
    actual_mapping_sha = config.mapping_bundle_sha256 or ""
    trace = DocumentClaimAdmissionTrace.model_validate_json(
        trace_text
    )
    mapping_bundle = DocumentClaimMappingBundle.model_validate_json(
        mapping_text
    )
    recomputed = evaluate_document_claim_admissions(
        trace.claims, trace.policy, observation_time_s=observation_time_s
    )
    if recomputed != trace.admissions:
        raise ValueError("stored document claim admissions differ from policy recomputation")
    eligible = {item.claim_id for item in recomputed if item.formal_fusion_eligible}
    claims_by_id = {item.claim_id: item for item in trace.claims}
    wrong_episode = sorted(
        claim_id
        for claim_id in eligible
        if f"episode:{episode_id}" not in claims_by_id[claim_id].applicability
    )
    if wrong_episode:
        raise ValueError(
            f"eligible document claims are not applicable to online episode: {wrong_episode}"
        )
    composite_sha = _sha(
        {"claim_trace_sha256": actual_trace_sha, "mapping_bundle_sha256": actual_mapping_sha}
    )
    build = build_document_claim_forward_ranking(
        episode_id=episode_id,
        observation_time_s=observation_time_s,
        claims=trace.claims,
        admissions=recomputed,
        mappings=mapping_bundle.mappings,
        registered_fault_ids=registered_fault_ids,
        source_artifact_id=config.source_artifact_id or mapping_bundle.bundle_id,
        source_artifact_sha256=composite_sha,
    )
    vote_count = sum(item.contributes_to_fusion for item in build.ranking.candidates)
    return DocumentClaimOnlineShadowResult(
        status="SHADOW_RANKING_READY" if vote_count else "SHADOW_ZERO_VOTE",
        ranking_build=build,
        claim_trace_sha256=actual_trace_sha,
        mapping_bundle_sha256=actual_mapping_sha,
        reason_codes=(
            ["REVIEWED_DOCUMENT_CLAIM_RANKING_AVAILABLE_SHADOW_ONLY"]
            if vote_count
            else ["NO_FORMALLY_ELIGIBLE_DOCUMENT_CLAIM_VOTES"]
        ),
    )


class DocumentClaimRankingBuild(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-document-claim-ranking-build.v1"
    ranking: FaultRanking
    consumed_claim_ids: list[str]
    consumed_mapping_ids: list[str]
    ignored_ineligible_claim_ids: list[str]
    observation_time_s: float | None = None
    active_graph_modified: Literal[False] = False
    claim_boundary: str = (
        "Only explicitly admitted and exactly mapped document claims contribute. This adapter "
        "does not infer fault mappings, publish graph knowledge, or establish calibration."
    )

    @model_validator(mode="after")
    def lists_are_disjoint(self) -> "DocumentClaimRankingBuild":
        if set(self.consumed_claim_ids) & set(self.ignored_ineligible_claim_ids):
            raise ValueError("consumed and ignored document claims must be disjoint")
        return self


def build_document_claim_forward_ranking(
    *,
    episode_id: str,
    observation_time_s: float | None = None,
    claims: list[DocumentEvidenceClaim],
    admissions: list[DocumentClaimAdmission],
    mappings: list[DocumentClaimFaultMapping],
    registered_fault_ids: list[str],
    source_artifact_id: str,
    source_artifact_sha256: str,
) -> DocumentClaimRankingBuild:
    """Build a lineage-complete KG_FORWARD ranking from exact admitted mappings."""

    registered = sorted(set(registered_fault_ids))
    if not registered or len(registered) != len(registered_fault_ids):
        raise ValueError("registered fault IDs must be nonempty and unique")
    claim_by_id = {item.claim_id: item for item in claims}
    admission_by_id = {item.claim_id: item for item in admissions}
    if len(claim_by_id) != len(claims) or set(admission_by_id) != set(claim_by_id):
        raise ValueError("claim admissions must cover unique claims exactly")
    if len(admission_by_id) != len(admissions):
        raise ValueError("claim admissions must be unique")
    mapping_by_claim = {item.claim_id: item for item in mappings}
    if len(mapping_by_claim) != len(mappings):
        raise ValueError("each document claim may map to at most one registered fault")
    unknown_mapping_claims = set(mapping_by_claim) - set(claim_by_id)
    if unknown_mapping_claims:
        raise ValueError("document mappings reference unknown claims")
    unknown_faults = {item.target_fault_id for item in mappings} - set(registered)
    if unknown_faults:
        raise ValueError(f"document mappings reference unregistered faults: {sorted(unknown_faults)}")

    eligible_ids = {
        item.claim_id for item in admissions if item.formal_fusion_eligible
    }
    # Independent defense: callers cannot bypass the admission evaluator by
    # passing a previously approved or forged admission at a later time.
    for claim_id in eligible_ids:
        claim = claim_by_id[claim_id]
        if f"episode:{episode_id}" not in claim.applicability:
            raise ValueError("eligible document claim is not bound to the requested episode")
        if (
            claim.source_authority != DocumentAuthority.APPROVED_SOURCE
            or claim.source_review_state != DocumentReviewState.HUMAN_REVIEWED
            or claim.epistemic_status != ClaimEpistemicStatus.SOURCE_FACT
            or not claim.publication_eligible
            or not claim.source_runtime_vote_allowed
            or not claim.runtime_evidence_eligible
            or claim.source_provenance_group_id is None
        ):
            raise ValueError("eligible document claim lacks approved runtime authority")
        if claim.source_visibility_class != DocumentVisibilityClass.OBSERVATION_CAUSAL:
            raise ValueError("document claim voting supports only governed causal observations")
        if claim.temporal_scope != ClaimTemporalScope.TIME_BOUNDED:
            raise ValueError("causal observation claim must be time bounded")
        if observation_time_s is None or not math.isfinite(observation_time_s) or observation_time_s < 0:
            raise ValueError("eligible document claims require a trusted finite observation_time_s")
        if (
            claim.valid_time_end_s is None or claim.observable_cutoff_time_s is None
            or not math.isfinite(claim.valid_time_end_s)
            or not math.isfinite(claim.observable_cutoff_time_s)
            or claim.valid_time_end_s > observation_time_s
            or claim.observable_cutoff_time_s > observation_time_s
        ):
            raise ValueError("eligible document claim contains future observation evidence")
    unmapped_eligible = eligible_ids - set(mapping_by_claim)
    if unmapped_eligible:
        raise ValueError(
            f"formally eligible document claims lack exact fault mappings: {sorted(unmapped_eligible)}"
        )
    positive: dict[str, float] = {fault_id: 0.0 for fault_id in registered}
    negative: dict[str, float] = {fault_id: 0.0 for fault_id in registered}
    claim_ids_by_fault: dict[str, list[str]] = {fault_id: [] for fault_id in registered}
    groups_by_fault: dict[str, list[str]] = {fault_id: [] for fault_id in registered}
    consumed_mappings: list[DocumentClaimFaultMapping] = []
    for claim_id in sorted(eligible_ids):
        claim = claim_by_id[claim_id]
        admission = admission_by_id[claim_id]
        mapping = mapping_by_claim[claim_id]
        if mapping.calibration_receipt_sha256 != admission.calibration_receipt_sha256:
            raise ValueError(
                "document claim mapping and admission reference different calibration receipts"
            )
        target = mapping.target_fault_id
        if mapping.polarity == "SUPPORTS":
            positive[target] += mapping.calibrated_weight
        else:
            negative[target] += mapping.calibrated_weight
        claim_ids_by_fault[target].append(claim_id)
        if claim.source_provenance_group_id is None:
            raise ValueError("eligible document claim lacks a provenance group")
        groups_by_fault[target].append(
            f"DOCUMENT_PROVENANCE:{claim.source_provenance_group_id}"
        )
        consumed_mappings.append(mapping)

    scores = {
        fault_id: max(0.0, positive[fault_id] - negative[fault_id])
        for fault_id in registered
    }
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    candidates: list[RankedFault] = []
    previous: float | None = None
    rank = 0
    for index, (fault_id, score) in enumerate(ordered, start=1):
        if previous is None or not math.isclose(score, previous, rel_tol=1e-12, abs_tol=1e-12):
            rank = index
            previous = score
        candidates.append(
            RankedFault(
                fault_id=fault_id,
                rank=rank,
                score=score,
                source=RankingSource.KG_FORWARD,
                evidence_fact_ids=[],
                source_evidence_ids=[
                    f"DOCUMENT_CLAIM:{claim_id}" for claim_id in claim_ids_by_fault[fault_id]
                ],
                correlation_group_ids=groups_by_fault[fault_id],
                lineage_complete=(score <= 0.0 or bool(claim_ids_by_fault[fault_id])),
                contributes_to_fusion=score > 0.0,
                metadata={
                    "document_claim_ids": claim_ids_by_fault[fault_id],
                    "positive_weight": positive[fault_id],
                    "contradicting_weight": negative[fault_id],
                    "admission_policy_sha256": sorted({
                        admission_by_id[claim_id].policy_sha256
                        for claim_id in claim_ids_by_fault[fault_id]
                    }),
                },
            )
        )

    supported_faults = sorted({
        item.target_fault_id
        for item in consumed_mappings
        if item.polarity == "SUPPORTS"
    })
    coverage = KGFaultCoverageDeclaration(
        coverage_basis="REVIEWED_CALIBRATED_DOCUMENT_CLAIM_MAPPINGS",
        source_artifact_id=source_artifact_id,
        source_artifact_sha256=source_artifact_sha256,
        supported_fault_ids=tuple(supported_faults),
        evidence_definition_ids_by_fault={
            fault_id: tuple(sorted(
                item.mapping_id for item in consumed_mappings
                if item.target_fault_id == fault_id and item.polarity == "SUPPORTS"
            ))
            for fault_id in supported_faults
        },
        claim_boundary=(
            "Executable coverage of reviewed, calibrated, exact document-claim mappings only; "
            "it does not imply coverage for unmapped faults or establish diagnostic performance."
        ),
    )
    mapping_hashes = [item.mapping_sha256 for item in sorted(
        consumed_mappings, key=lambda item: item.mapping_id
    )]
    ranking = FaultRanking(
        ranking_id="RANK-DOC-" + _sha({
            "episode_id": episode_id,
            "claim_ids": sorted(eligible_ids),
            "mapping_hashes": mapping_hashes,
            "source_artifact_sha256": source_artifact_sha256,
            "observation_time_s": observation_time_s if eligible_ids else None,
        })[:20].upper(),
        episode_id=episode_id,
        source=RankingSource.KG_FORWARD,
        score_semantics=ScoreSemantics.EVIDENCE_STRENGTH,
        candidates=candidates,
        producer_id="reviewed-document-claim-forward-ranking",
        producer_version="1.0.0",
        kg_fault_coverage=coverage,
        metadata={
            "document_claim_ranking": True,
            "formal_fusion_eligible_claim_count": len(eligible_ids),
            "observation_time_s": observation_time_s if eligible_ids else None,
            "mapping_sha256": mapping_hashes,
            "source_artifact_sha256": source_artifact_sha256,
        },
    )
    return DocumentClaimRankingBuild(
        ranking=ranking,
        consumed_claim_ids=sorted(eligible_ids),
        consumed_mapping_ids=sorted(item.mapping_id for item in consumed_mappings),
        ignored_ineligible_claim_ids=sorted(set(claim_by_id) - eligible_ids),
        observation_time_s=observation_time_s if eligible_ids else None,
    )


__all__ = [
    "DocumentClaimAdmissionTrace",
    "DocumentClaimFaultMapping",
    "DocumentClaimMappingBundle",
    "DocumentClaimOnlineShadowConfig",
    "DocumentClaimOnlineShadowResult",
    "DocumentClaimRankingBuild",
    "build_online_document_claim_shadow",
    "build_document_claim_forward_ranking",
]
