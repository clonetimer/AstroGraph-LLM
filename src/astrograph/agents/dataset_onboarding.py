from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from astrograph.agents.base import ControlledAgent
from astrograph.onboarding import sha256_file
from astrograph.operators import BUILTIN_OPERATOR_REGISTRY
from astrograph.semantics import DEFAULT_UNIT_REGISTRY, is_prohibited_field


T = TypeVar("T", bound=BaseModel)


class DatasetOnboardingBackend(Protocol):
    model_id: str
    model_version: str

    def generate_structured(
        self,
        *,
        agent_id: str,
        output_model: type[T],
        payload: dict[str, Any],
    ) -> T: ...


class SourceColumnProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str = Field(min_length=1)
    declared_unit: str | None = None
    inferred_dtype: str | None = None
    non_null_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    numeric_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    prohibited_for_runtime_mapping: bool = False
    requires_source_role_review: bool = False


class DatasetSchemaProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-dataset-schema-profile.v1"
    dataset_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sampled_row_count: int = Field(ge=0)
    total_column_count: int = Field(ge=0)
    columns: list[SourceColumnProfile] = Field(min_length=1)
    values_disclosed_to_mapping_agent: bool = False

    @property
    def allowed_columns(self) -> list[SourceColumnProfile]:
        return [item for item in self.columns if not item.prohibited_for_runtime_mapping]


class SemanticTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    quantity_kind_id: str = Field(min_length=1)
    canonical_unit: str = Field(min_length=1)
    description: str = ""


class ChannelMappingCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_field: str = Field(min_length=1)
    target_quantity_kind_id: str = Field(min_length=1)
    component_instance: str = Field(min_length=1)
    source_unit: str = Field(min_length=1)
    canonical_unit: str = Field(min_length=1)
    transform_operator_id: str = Field(default="identity", min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    ambiguous: bool = False
    rationale: str = Field(min_length=1)
    source_references: list[str] = Field(min_length=1)

    @field_validator("source_references")
    @classmethod
    def unique_references(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class SchemaMappingProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-schema-mapping-proposal.v1"
    dataset_id: str = Field(min_length=1)
    source_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_semantic_schema_id: str = Field(min_length=1)
    candidates: list[ChannelMappingCandidate] = Field(min_length=1)
    unresolved_source_fields: list[str] = Field(default_factory=list)
    role_review_questions: list[str] = Field(default_factory=list, max_length=10)
    proposer_mode: Literal["DETERMINISTIC_DRAFT", "LLM"] = "DETERMINISTIC_DRAFT"
    proposer_id: str = Field(min_length=1)
    proposer_version: str = Field(min_length=1)
    executable_content_present: bool = False


class ContractFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    severity: Literal["BLOCKER", "WARNING"]
    code: str = Field(min_length=1)
    source_field: str | None = None
    detail: str = Field(min_length=1)


class MappingValidationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-mapping-validation-receipt.v1"
    status: Literal["BLOCKED", "READY_FOR_HUMAN_REVIEW"]
    proposal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    findings: list[ContractFinding] = Field(default_factory=list)
    reviewed_candidate_count: int = Field(ge=0)
    publication_allowed: bool = False
    requires_human_approval: bool = True
    claim_boundary: str = (
        "A READY receipt means only that the proposal passed deterministic contract checks. "
        "It is not published, executable, scientifically validated, or approved for diagnosis."
    )


class OnboardingTraceStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=1)
    agent_id: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ControlledOnboardingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-controlled-dataset-onboarding.v1"
    profile: DatasetSchemaProfile
    proposal: SchemaMappingProposal
    validation: MappingValidationReceipt
    trace: list[OnboardingTraceStep] = Field(min_length=3, max_length=3)
    files_written: list[str] = Field(default_factory=list, max_length=0)


def _model_sha256(value: BaseModel) -> str:
    payload = value.model_dump_json(exclude_none=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _semantic_targets(path: Path) -> tuple[str, str, list[SemanticTarget]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    targets = [
        SemanticTarget(
            quantity_kind_id=item["quantity_kind_id"],
            canonical_unit=item["canonical_unit"],
            description=item.get("description", ""),
        )
        for item in raw.get("quantity_kinds", [])
    ]
    if not targets:
        raise ValueError("semantic schema contains no quantity kinds")
    return str(raw["semantic_schema_id"]), sha256_file(path), targets


class DatasetProfilerAgent(ControlledAgent):
    agent_id = "dataset-profiler-agent"
    agent_version = "0.1.0"

    def profile(
        self,
        *,
        dataset_id: str,
        telemetry_path: str | Path,
        declared_units: dict[str, str] | None = None,
        max_rows: int = 10_000,
    ) -> DatasetSchemaProfile:
        path = Path(telemetry_path).resolve()
        if path.suffix.lower() != ".csv" or not path.is_file():
            raise ValueError("dataset profiler requires an existing CSV telemetry file")
        header = [str(item) for item in pd.read_csv(path, nrows=0).columns]
        if not header:
            raise ValueError("telemetry CSV has no columns")
        allowed = [field for field in header if not is_prohibited_field(field)]
        frame = pd.read_csv(path, usecols=allowed, nrows=max_rows, low_memory=False) if allowed else pd.DataFrame()
        units = declared_units or {}
        columns: list[SourceColumnProfile] = []
        for field in header:
            prohibited = is_prohibited_field(field)
            if prohibited:
                columns.append(SourceColumnProfile(
                    field=field,
                    prohibited_for_runtime_mapping=True,
                    requires_source_role_review=field.strip().casefold() == "mode",
                ))
                continue
            series = frame[field]
            numeric = pd.to_numeric(series, errors="coerce")
            columns.append(
                SourceColumnProfile(
                    field=field,
                    declared_unit=units.get(field),
                    inferred_dtype=str(series.dtype),
                    non_null_fraction=float(series.notna().mean()) if len(series) else 0.0,
                    numeric_fraction=float(numeric.notna().mean()) if len(series) else 0.0,
                )
            )
        schema_payload = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return DatasetSchemaProfile(
            dataset_id=dataset_id,
            source_path=str(path),
            source_sha256=sha256_file(path),
            schema_sha256=hashlib.sha256(schema_payload).hexdigest(),
            sampled_row_count=len(frame),
            total_column_count=len(header),
            columns=columns,
        )


class SchemaMappingProposalAgent(ControlledAgent):
    agent_id = "schema-mapping-proposal-agent"
    agent_version = "0.1.0"

    def __init__(self, *args: Any, backend: DatasetOnboardingBackend | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.backend = backend

    def propose(
        self,
        *,
        profile: DatasetSchemaProfile,
        semantic_schema_path: str | Path,
        draft_proposal: dict[str, Any] | None = None,
        approved_component_instances: list[str] | None = None,
        approved_source_references: list[str] | None = None,
    ) -> SchemaMappingProposal:
        schema_id, _, targets = _semantic_targets(Path(semantic_schema_path).resolve())
        if self.backend is None:
            if draft_proposal is None:
                raise ValueError("deterministic proposal mode requires draft_proposal")
            result = SchemaMappingProposal.model_validate(draft_proposal)
            return result.model_copy(
                update={
                    "proposer_mode": "DETERMINISTIC_DRAFT",
                    "proposer_id": self.agent_id,
                    "proposer_version": self.agent_version,
                }
            )
        declared_columns = [item for item in profile.allowed_columns if item.declared_unit is not None]
        payload = {
            "dataset_id": profile.dataset_id,
            "source_schema_sha256": profile.schema_sha256,
            "source_columns": [item.model_dump(mode="json") for item in declared_columns],
            "withheld_source_column_count": len(profile.allowed_columns) - len(declared_columns),
            "role_review_candidates": [
                {"field": item.field, "question": "Is this an independently recorded operating condition or an answer proxy?"}
                for item in profile.columns if item.requires_source_role_review
            ],
            "semantic_targets": [item.model_dump(mode="json") for item in targets],
            "target_semantic_schema_id": schema_id,
            "approved_component_instances": sorted(set(approved_component_instances or [])),
            "approved_source_references": sorted(set(approved_source_references or [])),
            "approved_transform_operators": [
                "identity",
                "unit_convert",
                *(item.operator_id for item in BUILTIN_OPERATOR_REGISTRY.list_specs()),
            ],
            "output_protocol": SchemaMappingProposal.model_json_schema(),
            "rules": [
                "Return candidates only; do not emit executable code or tool instructions.",
                "Do not invent source fields, units, component identities, or source references.",
                "Use only approved component instances and approved source references.",
                "Mark ambiguous matches explicitly and leave uncertain fields unresolved.",
                "For role_review_candidates, ask a source-provenance question; never map them or assert they are safe.",
            ],
        }
        generated = self.backend.generate_structured(
            agent_id=self.agent_id,
            output_model=SchemaMappingProposal,
            payload=payload,
        )
        result = SchemaMappingProposal.model_validate(generated)
        return result.model_copy(
            update={
                "proposer_mode": "LLM",
                "proposer_id": self.backend.model_id,
                "proposer_version": self.backend.model_version,
            }
        )


class MappingContractCriticAgent(ControlledAgent):
    agent_id = "mapping-contract-critic-agent"
    agent_version = "0.1.0"

    def review(
        self,
        *,
        profile: DatasetSchemaProfile,
        proposal: SchemaMappingProposal,
        semantic_schema_path: str | Path,
        minimum_confidence: float = 0.75,
        approved_component_instances: list[str] | None = None,
        approved_source_references: list[str] | None = None,
    ) -> MappingValidationReceipt:
        schema_path = Path(semantic_schema_path).resolve()
        schema_id, schema_sha, targets = _semantic_targets(schema_path)
        target_by_id = {item.quantity_kind_id: item for item in targets}
        allowed_profile = {item.field: item for item in profile.allowed_columns}
        allowed_fields = set(allowed_profile)
        approved_components = set(approved_component_instances or [])
        approved_references = set(approved_source_references or [])
        findings: list[ContractFinding] = []

        def add(code: str, detail: str, source_field: str | None = None, severity: str = "BLOCKER") -> None:
            findings.append(
                ContractFinding(severity=severity, code=code, source_field=source_field, detail=detail)
            )

        if proposal.dataset_id != profile.dataset_id:
            add("DATASET_ID_MISMATCH", "proposal dataset ID does not match the profiler receipt")
        if proposal.source_schema_sha256 != profile.schema_sha256:
            add("SOURCE_SCHEMA_HASH_MISMATCH", "proposal source schema hash does not match the profiler receipt")
        if proposal.target_semantic_schema_id != schema_id:
            add("TARGET_SCHEMA_ID_MISMATCH", "proposal target schema ID does not match the reviewed schema")
        if proposal.executable_content_present:
            add("EXECUTABLE_CONTENT_FORBIDDEN", "mapping proposals cannot contain executable content")

        seen: set[str] = set()
        approved_operators = {
            "identity",
            "unit_convert",
            *(item.operator_id for item in BUILTIN_OPERATOR_REGISTRY.list_specs()),
        }
        for candidate in proposal.candidates:
            field = candidate.source_field
            if is_prohibited_field(field):
                add("PROHIBITED_SOURCE_FIELD", "label/truth/event/answer-proxy fields cannot be mapped", field)
            elif field not in allowed_fields:
                add("UNKNOWN_SOURCE_FIELD", "source field was not observed by the profiler", field)
            elif allowed_profile[field].declared_unit is None:
                add(
                    "MISSING_DECLARED_SOURCE_UNIT",
                    "source unit must come from a traceable declaration and cannot be inferred by the model",
                    field,
                )
            elif (
                candidate.source_unit != allowed_profile[field].declared_unit
            ):
                add(
                    "SOURCE_DECLARED_UNIT_MISMATCH",
                    "candidate source unit differs from the profiler's declared unit",
                    field,
                )
            if field in seen:
                add("DUPLICATE_SOURCE_MAPPING", "source field has more than one mapping candidate", field)
            seen.add(field)
            target = target_by_id.get(candidate.target_quantity_kind_id)
            if target is None:
                add("UNKNOWN_SEMANTIC_TARGET", "target quantity kind is not registered", field)
            else:
                if candidate.canonical_unit != target.canonical_unit:
                    add("CANONICAL_UNIT_MISMATCH", "candidate canonical unit differs from target schema", field)
                try:
                    source_unit = DEFAULT_UNIT_REGISTRY.get(candidate.source_unit)
                    target_unit = DEFAULT_UNIT_REGISTRY.get(target.canonical_unit)
                    if source_unit.dimension != target_unit.dimension:
                        add("INCOMPATIBLE_UNIT_DIMENSION", "source and target units have incompatible dimensions", field)
                except ValueError as exc:
                    add("UNKNOWN_UNIT", str(exc), field)
            if candidate.transform_operator_id not in approved_operators:
                add("UNREGISTERED_OPERATOR", "transform operator is not in the deterministic allow-list", field)
            if candidate.component_instance not in approved_components:
                add(
                    "UNAPPROVED_COMPONENT_INSTANCE",
                    "component instance is not in the caller-supplied approved catalog",
                    field,
                )
            unapproved_references = sorted(set(candidate.source_references) - approved_references)
            if unapproved_references:
                add(
                    "UNAPPROVED_SOURCE_REFERENCE",
                    f"candidate cites unapproved references: {unapproved_references}",
                    field,
                )
            if candidate.confidence < minimum_confidence:
                add("LOW_CONFIDENCE", f"confidence is below {minimum_confidence:g}", field)
            if candidate.ambiguous:
                add("AMBIGUOUS_MAPPING", "candidate is explicitly marked ambiguous", field)

        unresolved = sorted(set(proposal.unresolved_source_fields))
        for field in unresolved:
            if field not in allowed_fields:
                add("UNKNOWN_UNRESOLVED_FIELD", "unresolved field was not observed by the profiler", field)
            else:
                add("UNRESOLVED_SOURCE_FIELD", "field requires human semantic resolution", field, "WARNING")

        for column in profile.columns:
            if column.requires_source_role_review:
                add(
                    "SOURCE_ROLE_REVIEW_REQUIRED",
                    "operating-condition versus answer-proxy role requires source provenance; field remains unmappable",
                    column.field,
                    "WARNING",
                )

        proposal_sha = _model_sha256(proposal)
        blocked = any(item.severity == "BLOCKER" for item in findings)
        return MappingValidationReceipt(
            status="BLOCKED" if blocked else "READY_FOR_HUMAN_REVIEW",
            proposal_sha256=proposal_sha,
            source_schema_sha256=profile.schema_sha256,
            semantic_schema_sha256=schema_sha,
            findings=findings,
            reviewed_candidate_count=len(proposal.candidates),
        )


class ControlledDatasetOnboardingWorkflow:
    workflow_id = "controlled-dataset-onboarding-workflow"
    workflow_version = "0.1.0"

    def __init__(
        self,
        *,
        profiler: DatasetProfilerAgent | None = None,
        proposer: SchemaMappingProposalAgent | None = None,
        critic: MappingContractCriticAgent | None = None,
    ) -> None:
        self.profiler = profiler or DatasetProfilerAgent()
        self.proposer = proposer or SchemaMappingProposalAgent()
        self.critic = critic or MappingContractCriticAgent()

    def run(
        self,
        *,
        dataset_id: str,
        telemetry_path: str | Path,
        semantic_schema_path: str | Path,
        declared_units: dict[str, str] | None = None,
        draft_proposal: dict[str, Any] | None = None,
        minimum_confidence: float = 0.75,
        approved_component_instances: list[str] | None = None,
        approved_source_references: list[str] | None = None,
    ) -> ControlledOnboardingResult:
        profile = self.profiler.profile(
            dataset_id=dataset_id,
            telemetry_path=telemetry_path,
            declared_units=declared_units,
        )
        proposal = self.proposer.propose(
            profile=profile,
            semantic_schema_path=semantic_schema_path,
            draft_proposal=draft_proposal,
            approved_component_instances=approved_component_instances,
            approved_source_references=approved_source_references,
        )
        validation = self.critic.review(
            profile=profile,
            proposal=proposal,
            semantic_schema_path=semantic_schema_path,
            minimum_confidence=minimum_confidence,
            approved_component_instances=approved_component_instances,
            approved_source_references=approved_source_references,
        )
        return ControlledOnboardingResult(
            profile=profile,
            proposal=proposal,
            validation=validation,
            trace=[
                OnboardingTraceStep(
                    sequence=1,
                    agent_id=self.profiler.agent_id,
                    outcome="PROFILED",
                    artifact_sha256=_model_sha256(profile),
                ),
                OnboardingTraceStep(
                    sequence=2,
                    agent_id=self.proposer.agent_id,
                    outcome="PROPOSED",
                    artifact_sha256=_model_sha256(proposal),
                ),
                OnboardingTraceStep(
                    sequence=3,
                    agent_id=self.critic.agent_id,
                    outcome=validation.status,
                    artifact_sha256=_model_sha256(validation),
                ),
            ],
        )


__all__ = [
    "ChannelMappingCandidate",
    "ContractFinding",
    "ControlledDatasetOnboardingWorkflow",
    "ControlledOnboardingResult",
    "DatasetOnboardingBackend",
    "DatasetProfilerAgent",
    "DatasetSchemaProfile",
    "MappingContractCriticAgent",
    "MappingValidationReceipt",
    "SchemaMappingProposal",
    "SchemaMappingProposalAgent",
    "SourceColumnProfile",
]
