"""Deterministic Step-6 lineage recovery and route-decision gate.

This module never reconstructs missing scientific artifacts from prose and never
promotes filename matches into provenance.  It chooses an evidence-feasible route
without reading outcome metrics.  Groups 4/9 are permanently post-hoc.
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ArtifactState(str, Enum):
    VERIFIED = "VERIFIED"
    MISSING = "MISSING"
    HASH_MISMATCH = "HASH_MISMATCH"
    UNATTESTED = "UNATTESTED"
    PLACEHOLDER = "PLACEHOLDER"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Step6Route(str, Enum):
    RESTORE_AUTHENTIC_CHAIN = "RESTORE_AUTHENTIC_CHAIN"
    CLEAN_ROOM_RETRAIN_NEW_LINEAGE = "CLEAN_ROOM_RETRAIN_NEW_LINEAGE"
    REFERENCE_BASELINE_NEW_LINEAGE = "REFERENCE_BASELINE_NEW_LINEAGE"
    MODEL_INDEPENDENT_TEMPORAL_ROUTE = "MODEL_INDEPENDENT_TEMPORAL_ROUTE"
    SCOPE_FREEZE = "SCOPE_FREEZE"


class LineageArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    artifact_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    path: str | None = None
    expected_sha256: str | None = None
    observed_sha256: str | None = None
    state: ArtifactState
    provenance_receipt_sha256: str | None = None
    notes: tuple[str, ...] = ()

    @field_validator("expected_sha256", "observed_sha256", "provenance_receipt_sha256")
    @classmethod
    def _digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.lower()
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("SHA-256 must be 64 lowercase hexadecimal characters")
        return value


class Step6AssessmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str = "astrograph-step6-assessment-input.v1"
    project_status: str
    selection_groups: tuple[int, ...] = (3, 8)
    permanently_post_hoc_groups: tuple[int, ...] = (4, 9)
    artifacts: tuple[LineageArtifact, ...]
    available_capabilities: tuple[str, ...] = ()
    accessed_groups: tuple[int, ...] = ()

    @model_validator(mode="after")
    def _split_isolation(self):
        if tuple(sorted(self.selection_groups)) != (3, 8):
            raise ValueError("Step-6 development selection groups must be exactly 3/8")
        if set(self.selection_groups) & set(self.permanently_post_hoc_groups):
            raise ValueError("development and post-hoc groups overlap")
        return self


class RouteEligibility(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    route: Step6Route
    eligible: bool
    missing_roles: tuple[str, ...] = ()
    rejected_roles: tuple[str, ...] = ()
    claim_boundary: tuple[str, ...]


class Step6RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str = "astrograph-step6-route-decision.v1"
    selected_route: Step6Route
    route_eligibility: tuple[RouteEligibility, ...]
    outcome_data_read: bool = False
    groups_used_for_decision: tuple[int, ...] = ()
    historical_metrics_reusable: bool
    authentic_migration_claim_allowed: bool
    required_next_actions: tuple[str, ...]
    prohibited_actions: tuple[str, ...]
    decision_sha256: str


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _role_states(inp: Step6AssessmentInput) -> dict[str, ArtifactState]:
    result: dict[str, ArtifactState] = {}
    for item in inp.artifacts:
        # A role is VERIFIED only if every artifact carrying that required role is verified.
        previous = result.get(item.role)
        if previous is None or previous == ArtifactState.VERIFIED:
            result[item.role] = item.state
    return result


def _eligibility(route: Step6Route, roles: Mapping[str, ArtifactState], caps: set[str]) -> RouteEligibility:
    requirements: dict[Step6Route, tuple[str, ...]] = {
        Step6Route.RESTORE_AUTHENTIC_CHAIN: (
            "original_model_package", "state_dict_identity", "feature_contract",
            "preprocessor_receipt", "training_receipt", "label_free_dev_traces",
            "temporal_v2_receipt", "development_split_receipt",
        ),
        Step6Route.CLEAN_ROOM_RETRAIN_NEW_LINEAGE: (
            "training_dataset_manifest", "training_labels", "feature_contract",
            "preprocessor_spec", "training_recipe", "development_split_receipt",
        ),
        Step6Route.REFERENCE_BASELINE_NEW_LINEAGE: (
            "training_dataset_manifest", "training_labels", "feature_contract",
            "preprocessor_spec", "development_split_receipt",
        ),
        Step6Route.MODEL_INDEPENDENT_TEMPORAL_ROUTE: (
            "telemetry_dataset_manifest", "development_split_receipt",
        ),
        Step6Route.SCOPE_FREEZE: (),
    }
    needed=requirements[route]
    missing=tuple(sorted(r for r in needed if roles.get(r) != ArtifactState.VERIFIED))
    rejected=tuple(sorted(r for r in needed if roles.get(r) in {ArtifactState.HASH_MISMATCH, ArtifactState.PLACEHOLDER, ArtifactState.UNATTESTED}))
    eligible=not missing
    if route is Step6Route.MODEL_INDEPENDENT_TEMPORAL_ROUTE:
        eligible = eligible and "label_free_temporal_score_generator" in caps
        if "label_free_temporal_score_generator" not in caps:
            missing=tuple(sorted(set(missing)|{"capability:label_free_temporal_score_generator"}))
    boundaries={
        Step6Route.RESTORE_AUTHENTIC_CHAIN:("May test migration/equivalence only after every original artifact is verified.",),
        Step6Route.CLEAN_ROOM_RETRAIN_NEW_LINEAGE:("Creates a new model lineage.","Historical model-equivalence and migration claims are prohibited.","Historical performance is context only."),
        Step6Route.REFERENCE_BASELINE_NEW_LINEAGE:("Creates a transparent reference lineage.","It is not a reconstruction of the historical P2 model.","Claims are limited to the new frozen recipe."),
        Step6Route.MODEL_INDEPENDENT_TEMPORAL_ROUTE:("Freezes ADCS fault-identification claims.","May evaluate label-free temporal localization/anomaly tasks only.","No historical classifier equivalence claim."),
        Step6Route.SCOPE_FREEZE:("Step 6 remains scientifically blocked.","Software release may proceed only with the affected research capability disabled and the blocker disclosed."),
    }
    return RouteEligibility(route=route,eligible=eligible,missing_roles=missing,rejected_roles=rejected,claim_boundary=boundaries[route])


def assess_step6_route(inp: Step6AssessmentInput) -> Step6RouteDecision:
    roles=_role_states(inp); caps=set(inp.available_capabilities)
    ordered=(
        Step6Route.RESTORE_AUTHENTIC_CHAIN,
        Step6Route.CLEAN_ROOM_RETRAIN_NEW_LINEAGE,
        Step6Route.REFERENCE_BASELINE_NEW_LINEAGE,
        Step6Route.MODEL_INDEPENDENT_TEMPORAL_ROUTE,
        Step6Route.SCOPE_FREEZE,
    )
    entries=tuple(_eligibility(r, roles, caps) for r in ordered)
    selected=next(e.route for e in entries if e.eligible)
    authentic=selected is Step6Route.RESTORE_AUTHENTIC_CHAIN
    next_actions={
        Step6Route.RESTORE_AUTHENTIC_CHAIN:("Verify state-dict and preprocessor identity.","Regenerate groups 3/8 label-free traces.","Freeze temporal-v2 receipt before any post-hoc replay."),
        Step6Route.CLEAN_ROOM_RETRAIN_NEW_LINEAGE:("Pre-register a new model identifier and training recipe.","Train without groups 3/8 or 4/9 leakage.","Generate new groups 3/8 traces and freeze a new temporal-v2 receipt.","Keep historical v1/v2 results in a separate lineage."),
        Step6Route.REFERENCE_BASELINE_NEW_LINEAGE:("Freeze a simple auditable reference-model recipe.","Train with group-disjoint data only.","Calibrate and select policies on groups 3/8.","Treat the result as a new baseline, not a recovered P2 model."),
        Step6Route.MODEL_INDEPENDENT_TEMPORAL_ROUTE:("Freeze ADCS identification and KG-gain claims.","Validate temporal localization/anomaly detection with label-free scores.","Use task-stratified external datasets without relabeling them as ADCS identification."),
        Step6Route.SCOPE_FREEZE:("Publish an explicit missing-lineage blocker receipt.","Disable the affected capability in the stable software profile.","Continue only task claims supported by independent evidence."),
    }
    prohibited=(
        "Do not infer a missing model or receipt from prose, filenames, metrics, or plots.",
        "Do not use groups 4/9 for route, model, threshold, fusion, rejection, or evidence selection.",
        "Do not call a clean-room or reference model a migration of the historical v1 package.",
        "Do not promote already accessed datasets to untouched confirmatory evidence.",
        "Do not generate placeholder digests for a sealed campaign.",
    )
    payload={
        "schema_version":"astrograph-step6-route-decision.v1",
        "selected_route":selected.value,
        "route_eligibility":[e.model_dump(mode="json") for e in entries],
        "outcome_data_read":False,
        "groups_used_for_decision":[],
        "historical_metrics_reusable":authentic,
        "authentic_migration_claim_allowed":authentic,
        "required_next_actions":list(next_actions[selected]),
        "prohibited_actions":list(prohibited),
    }
    return Step6RouteDecision(**payload, decision_sha256=_canonical_digest(payload))


def artifact_from_path(*, artifact_id: str, role: str, path: Path, expected_sha256: str | None = None, provenance_receipt_sha256: str | None = None, notes: Sequence[str] = ()) -> LineageArtifact:
    if not path.exists():
        return LineageArtifact(artifact_id=artifact_id,role=role,path=str(path),expected_sha256=expected_sha256,state=ArtifactState.MISSING,provenance_receipt_sha256=provenance_receipt_sha256,notes=tuple(notes))
    h=hashlib.sha256(path.read_bytes()).hexdigest()
    state=ArtifactState.VERIFIED if expected_sha256 is None or h==expected_sha256.lower() else ArtifactState.HASH_MISMATCH
    if state is ArtifactState.VERIFIED and provenance_receipt_sha256 is None and role in {"original_model_package","training_receipt","preprocessor_receipt","temporal_v2_receipt"}:
        state=ArtifactState.UNATTESTED
    return LineageArtifact(artifact_id=artifact_id,role=role,path=str(path),expected_sha256=expected_sha256,observed_sha256=h,state=state,provenance_receipt_sha256=provenance_receipt_sha256,notes=tuple(notes))
