from __future__ import annotations

from enum import Enum
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from astrograph.data_driven import (
    BenchmarkClaimType,
    BenchmarkReleaseDecision,
    BenchmarkReleaseStatus,
)


class ReleaseClaimStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    BLOCKED = "BLOCKED"


class ReleaseClaimDisposition(str, Enum):
    CLAIMED = "CLAIMED"
    NOT_CLAIMED = "NOT_CLAIMED"


class ReleaseClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1)
    claim_type: BenchmarkClaimType
    disposition: ReleaseClaimDisposition = ReleaseClaimDisposition.CLAIMED
    statement: str = Field(min_length=1)
    required_model_ids: list[str] = Field(default_factory=list)
    source_artifacts: list[str] = Field(default_factory=list)

    @field_validator("required_model_ids", "source_artifacts")
    @classmethod
    def unique_items(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("release claim lists must not contain duplicates")
        return value


class ReleaseClaimManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-release-claim-manifest.v1"
    release_id: str = Field(min_length=1)
    claims: list[ReleaseClaim]

    @field_validator("claims")
    @classmethod
    def unique_claim_ids(cls, value: list[ReleaseClaim]) -> list[ReleaseClaim]:
        claim_ids = [item.claim_id for item in value]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("release claim ids must be unique")
        return value


class ReleaseClaimDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-release-claim-decision.v1"
    status: ReleaseClaimStatus
    claim_id: str
    claim_type: BenchmarkClaimType
    disposition: ReleaseClaimDisposition
    evaluated_model_ids: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    supporting_decision_statuses: list[str] = Field(default_factory=list)
    statement: str


class ReleaseClaimGateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-release-claim-gate-result.v1"
    status: ReleaseClaimStatus
    release_id: str
    decisions: list[ReleaseClaimDecision]
    blocking_reasons: list[str] = Field(default_factory=list)


class ReleaseClaimGate:
    def evaluate(
        self,
        manifest: ReleaseClaimManifest,
        benchmark_decisions: Iterable[BenchmarkReleaseDecision],
    ) -> ReleaseClaimGateResult:
        decisions_by_claim_type: dict[BenchmarkClaimType, list[BenchmarkReleaseDecision]] = {}
        for decision in benchmark_decisions:
            decisions_by_claim_type.setdefault(decision.claim_type, []).append(decision)

        release_decisions = [
            self._evaluate_claim(claim, decisions_by_claim_type.get(claim.claim_type, [])) for claim in manifest.claims
        ]
        blocking = [
            f"{decision.claim_id}:{reason}"
            for decision in release_decisions
            for reason in decision.blocking_reasons
        ]
        return ReleaseClaimGateResult(
            status=ReleaseClaimStatus.BLOCKED if blocking else ReleaseClaimStatus.ACCEPTED,
            release_id=manifest.release_id,
            decisions=release_decisions,
            blocking_reasons=blocking,
        )

    def _evaluate_claim(
        self,
        claim: ReleaseClaim,
        evidence_decisions: list[BenchmarkReleaseDecision],
    ) -> ReleaseClaimDecision:
        relevant = self._relevant_decisions(claim, evidence_decisions)
        evaluated_model_ids = sorted({item.report_model_id for item in relevant})
        supporting_statuses = [
            f"{item.report_model_id}:{item.claim_type.value}:{item.status.value}" for item in relevant
        ]
        blocking: list[str] = []

        if claim.disposition == ReleaseClaimDisposition.CLAIMED:
            if not relevant:
                blocking.append("NO_BENCHMARK_RELEASE_DECISION_FOR_CLAIM")
            blocked = [item for item in relevant if item.status != BenchmarkReleaseStatus.ACCEPTED]
            if blocked:
                blocking.append("CLAIM_NOT_SUPPORTED_BY_BENCHMARK_RELEASE_GATE")
                for item in blocked:
                    for reason in item.blocking_reasons:
                        blocking.append(f"{item.report_model_id}:{reason}")
            missing = sorted(set(claim.required_model_ids) - set(evaluated_model_ids))
            for model_id in missing:
                blocking.append(f"REQUIRED_MODEL_RELEASE_DECISION_MISSING:{model_id}")

        return ReleaseClaimDecision(
            status=ReleaseClaimStatus.BLOCKED if blocking else ReleaseClaimStatus.ACCEPTED,
            claim_id=claim.claim_id,
            claim_type=claim.claim_type,
            disposition=claim.disposition,
            evaluated_model_ids=evaluated_model_ids,
            blocking_reasons=list(dict.fromkeys(blocking)),
            supporting_decision_statuses=supporting_statuses,
            statement=claim.statement,
        )

    def _relevant_decisions(
        self,
        claim: ReleaseClaim,
        evidence_decisions: list[BenchmarkReleaseDecision],
    ) -> list[BenchmarkReleaseDecision]:
        if not claim.required_model_ids:
            return evidence_decisions
        required = set(claim.required_model_ids)
        return [item for item in evidence_decisions if item.report_model_id in required]
