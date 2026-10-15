from __future__ import annotations

from enum import Enum
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator


class KnowledgeNodeType(str, Enum):
    SUBSYSTEM = "Subsystem"
    COMPONENT_TYPE = "ComponentType"
    COMPONENT_INSTANCE = "ComponentInstance"
    FAULT_MODE = "FaultMode"
    ANOMALY = "Anomaly"
    TELEMETRY_CHANNEL = "TelemetryChannel"
    FEATURE_DEFINITION = "FeatureDefinition"
    FEATURE_RESULT = "FeatureResult"
    EVIDENCE_DEFINITION = "EvidenceDefinition"
    EVIDENCE_FACT = "EvidenceFact"
    EVIDENCE_SOURCE = "EvidenceSource"
    DIAGNOSIS_EPISODE = "DiagnosisEpisode"
    RESPONSE_WINDOW = "ResponseWindow"
    DIAGNOSTIC_HYPOTHESIS = "DiagnosticHypothesis"
    DECISION_RECORD = "DecisionRecord"
    MODEL_ARTIFACT = "ModelArtifact"


class KnowledgeRelationType(str, Enum):
    PART_OF = "PART_OF"
    AFFECTS = "AFFECTS"
    OBSERVED_AS = "OBSERVED_AS"
    MEASURED_ON = "MEASURED_ON"
    DERIVED_FROM = "DERIVED_FROM"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    PRECEDES = "PRECEDES"
    PROPAGATES_TO = "PROPAGATES_TO"
    GENERATED_BY = "GENERATED_BY"
    VALIDATED_BY = "VALIDATED_BY"
    HAS_RESPONSE_WINDOW = "HAS_RESPONSE_WINDOW"
    INVOLVES_COMPONENT = "INVOLVES_COMPONENT"
    HAS_FEATURE_RESULT = "HAS_FEATURE_RESULT"
    INSTANCE_OF = "INSTANCE_OF"
    USES_FEATURE = "USES_FEATURE"
    HAS_EVIDENCE_FACT = "HAS_EVIDENCE_FACT"
    HAS_HYPOTHESIS = "HAS_HYPOTHESIS"
    SUPPORTED_BY = "SUPPORTED_BY"
    CONTRADICTED_BY = "CONTRADICTED_BY"
    DECIDES_EPISODE = "DECIDES_EPISODE"
    ACCEPTS_HYPOTHESIS = "ACCEPTS_HYPOTHESIS"
    REQUIRES_EVIDENCE = "REQUIRES_EVIDENCE"
    EXCLUDES = "EXCLUDES"
    DOCUMENTED_IN = "DOCUMENTED_IN"
    USES_TELEMETRY_FIELD = "USES_TELEMETRY_FIELD"
    HAS_MEMBER_CANDIDATE = "HAS_MEMBER_CANDIDATE"


class AssertionStatus(str, Enum):
    EXTRACTED = "EXTRACTED"
    DERIVED = "DERIVED"
    INFERRED = "INFERRED"
    AMBIGUOUS = "AMBIGUOUS"


class ReviewStatus(str, Enum):
    UNREVIEWED = "UNREVIEWED"
    MACHINE_VALIDATED = "MACHINE_VALIDATED"
    HUMAN_REVIEWED = "HUMAN_REVIEWED"
    REJECTED = "REJECTED"


class KnowledgeSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-knowledge-schema.v2"
    node_types: list[KnowledgeNodeType] = Field(default_factory=lambda: list(KnowledgeNodeType))
    relation_types: list[KnowledgeRelationType] = Field(default_factory=lambda: list(KnowledgeRelationType))
    assertion_statuses: list[AssertionStatus] = Field(default_factory=lambda: list(AssertionStatus))
    review_statuses: list[ReviewStatus] = Field(default_factory=lambda: list(ReviewStatus))
    claim_boundary: str = (
        "Schema constrains knowledge representation and provenance. It does not convert associative "
        "evidence paths into causal effects."
    )

    @model_validator(mode="after")
    def unique_values(self) -> "KnowledgeSchema":
        for name in ("node_types", "relation_types", "assertion_statuses", "review_statuses"):
            values = [item.value for item in getattr(self, name)]
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")
        return self


DEFAULT_KNOWLEDGE_SCHEMA = KnowledgeSchema()


def validate_node_type(value: str | KnowledgeNodeType) -> KnowledgeNodeType:
    return value if isinstance(value, KnowledgeNodeType) else KnowledgeNodeType(value)


def validate_relation_type(value: str | KnowledgeRelationType) -> KnowledgeRelationType:
    return value if isinstance(value, KnowledgeRelationType) else KnowledgeRelationType(value)


def validate_schema_values(*, node_types: Iterable[str], relation_types: Iterable[str]) -> None:
    for value in node_types:
        validate_node_type(value)
    for value in relation_types:
        validate_relation_type(value)


__all__ = [
    "KnowledgeNodeType",
    "KnowledgeRelationType",
    "AssertionStatus",
    "ReviewStatus",
    "KnowledgeSchema",
    "DEFAULT_KNOWLEDGE_SCHEMA",
    "validate_node_type",
    "validate_relation_type",
    "validate_schema_values",
]
