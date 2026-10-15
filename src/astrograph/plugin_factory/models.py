from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FeatureImplementationMode(str, Enum):
    CONFIGURATION_ONLY = "CONFIGURATION_ONLY"
    OPERATOR_COMPOSITION = "OPERATOR_COMPOSITION"
    CUSTOM_CODE_REQUIRED = "CUSTOM_CODE_REQUIRED"
    BLOCKED = "BLOCKED"


class OperatorDataSourceKind(str, Enum):
    CURRENT_SIGNAL = "CURRENT_SIGNAL"
    BASELINE_SIGNAL = "BASELINE_SIGNAL"
    NODE_OUTPUT = "NODE_OUTPUT"
    TIME = "TIME"


class OperatorDataRef(BaseModel):
    """Typed reference used by a deterministic operator graph node."""

    model_config = ConfigDict(extra="forbid")

    source: OperatorDataSourceKind
    ref: str | None = None

    @model_validator(mode="after")
    def validate_ref(self) -> "OperatorDataRef":
        requires_ref = self.source in {
            OperatorDataSourceKind.CURRENT_SIGNAL,
            OperatorDataSourceKind.BASELINE_SIGNAL,
            OperatorDataSourceKind.NODE_OUTPUT,
        }
        if requires_ref and not self.ref:
            raise ValueError(f"{self.source.value} data reference requires ref")
        if self.source == OperatorDataSourceKind.TIME and self.ref is not None:
            raise ValueError("TIME data reference must not define ref")
        return self


class OperatorGraphNodeSpec(BaseModel):
    """One white-listed deterministic operator invocation in a feature DAG."""

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    operator_id: str = Field(min_length=1)
    current_inputs: list[OperatorDataRef] = Field(min_length=1)
    baseline_inputs: list[OperatorDataRef] = Field(default_factory=list)
    time_input: OperatorDataRef | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_time_input(self) -> "OperatorGraphNodeSpec":
        if self.time_input is not None and self.time_input.source != OperatorDataSourceKind.TIME:
            raise ValueError("time_input must use TIME source")
        return self


class OperatorGraphSpec(BaseModel):
    """Acyclic deterministic operator graph for one computed feature."""

    model_config = ConfigDict(extra="forbid")

    graph_version: str = "astrograph-operator-graph.v1"
    nodes: list[OperatorGraphNodeSpec] = Field(min_length=1)
    output_node_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_graph_shape(self) -> "OperatorGraphSpec":
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("operator graph node IDs must be unique")
        if self.output_node_id not in set(node_ids):
            raise ValueError("operator graph output_node_id is not registered")
        known = set(node_ids)
        for node in self.nodes:
            refs = [*node.current_inputs, *node.baseline_inputs]
            for ref in refs:
                if ref.source == OperatorDataSourceKind.NODE_OUTPUT and ref.ref not in known:
                    raise ValueError(
                        f"operator graph node {node.node_id} references unknown node {ref.ref}"
                    )
                if ref.source == OperatorDataSourceKind.NODE_OUTPUT and ref.ref == node.node_id:
                    raise ValueError(f"operator graph node {node.node_id} cannot reference itself")
        return self


class TelemetryBindingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    binding_id: str = Field(min_length=1)
    source_field: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    semantic_role: str = Field(min_length=1)
    required: bool = True
    description: str = ""


class FeatureServiceSpec(BaseModel):
    """Agent-produced, reviewable specification for one deterministic feature."""

    model_config = ConfigDict(extra="forbid")

    feature_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    output_unit: str = Field(min_length=1)
    input_bindings: dict[str, str] = Field(default_factory=dict)
    operator_id: str | None = None
    operator_pipeline: list[str] = Field(default_factory=list)
    operator_graph: OperatorGraphSpec | None = None
    operator_parameters: dict[str, Any] = Field(default_factory=dict)
    custom_formula: str | None = None
    source_refs: list[str] = Field(default_factory=list)
    claim_boundary: str = "Engineering feature definition; requires object-level validation."

    @model_validator(mode="after")
    def exactly_one_implementation(self) -> "FeatureServiceSpec":
        choices = (
            int(self.operator_id is not None)
            + int(bool(self.operator_pipeline))
            + int(self.operator_graph is not None)
            + int(self.custom_formula is not None)
        )
        if choices != 1:
            raise ValueError(
                "feature must define exactly one of operator_id, operator_pipeline, "
                "operator_graph, or custom_formula"
            )
        if self.custom_formula is not None and not self.source_refs:
            raise ValueError("custom_formula requires at least one source reference")
        return self


class EvidenceServiceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    target_fault_id: str = Field(min_length=1)
    feature_id: str = Field(min_length=1)
    role: str = Field(pattern=r"^(REQUIRED|SUPPORTING|COUNTER|EXCLUSION)$")
    comparison: str = Field(
        pattern=r"^(GREATER_THAN|GREATER_OR_EQUAL|LESS_THAN|LESS_OR_EQUAL|EQUAL|NOT_EQUAL)$"
    )
    threshold: float
    threshold_unit: str = Field(min_length=1)
    critical: bool = False
    source_refs: list[str] = Field(default_factory=list)
    claim_boundary: str = "Engineering evidence threshold; requires object-level validation."


class ObjectPluginBuildSpec(BaseModel):
    """Structured output contract consumed by the deterministic plugin factory."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-object-plugin-build-spec.v1"
    plugin_id: str = Field(min_length=3, pattern=r"^[a-z0-9][a-z0-9_.-]+$")
    plugin_version: str = Field(min_length=1)
    object_type: str = Field(min_length=1)
    subsystem: str = Field(min_length=1)
    dataset_profiles: list[str] = Field(default_factory=list)
    object_parameters: dict[str, Any] = Field(default_factory=dict)
    telemetry_bindings: list[TelemetryBindingSpec]
    features: list[FeatureServiceSpec]
    evidence: list[EvidenceServiceSpec]
    source_refs: list[str] = Field(default_factory=list)
    claim_boundary: str = "Generated engineering object plugin; review required before online registration."

    @model_validator(mode="after")
    def validate_references(self) -> "ObjectPluginBuildSpec":
        binding_ids = [item.binding_id for item in self.telemetry_bindings]
        if len(binding_ids) != len(set(binding_ids)):
            raise ValueError("telemetry binding IDs must be unique")
        feature_ids = [item.feature_id for item in self.features]
        if len(feature_ids) != len(set(feature_ids)):
            raise ValueError("feature IDs must be unique")
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence IDs must be unique")
        known_bindings = set(binding_ids)
        for feature in self.features:
            unknown = sorted(set(feature.input_bindings.values()) - known_bindings)
            if unknown:
                raise ValueError(f"feature {feature.feature_id} references unknown bindings: {unknown}")
            if feature.operator_graph is not None:
                for node in feature.operator_graph.nodes:
                    refs = [*node.current_inputs, *node.baseline_inputs]
                    graph_bindings = {
                        ref.ref
                        for ref in refs
                        if ref.source
                        in {
                            OperatorDataSourceKind.CURRENT_SIGNAL,
                            OperatorDataSourceKind.BASELINE_SIGNAL,
                        }
                    }
                    unknown_graph = sorted(graph_bindings - known_bindings)
                    if unknown_graph:
                        raise ValueError(
                            f"feature {feature.feature_id} graph references unknown bindings: "
                            f"{unknown_graph}"
                        )
        known_features = set(feature_ids)
        feature_units = {item.feature_id: item.output_unit for item in self.features}
        for evidence in self.evidence:
            if evidence.feature_id not in known_features:
                raise ValueError(
                    f"evidence {evidence.evidence_id} references unknown feature {evidence.feature_id}"
                )
            if evidence.threshold_unit != feature_units[evidence.feature_id]:
                raise ValueError(
                    f"evidence {evidence.evidence_id} threshold unit {evidence.threshold_unit!r} "
                    f"does not match feature unit {feature_units[evidence.feature_id]!r}"
                )
            if not evidence.source_refs:
                raise ValueError(f"evidence {evidence.evidence_id} requires a threshold source reference")
        return self


class FeatureGenerationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_id: str
    mode: FeatureImplementationMode
    resolved_operator_ids: list[str] = Field(default_factory=list)
    reason: str
    requires_human_review: bool = False


class PluginGenerationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-plugin-generation-plan.v1"
    plugin_id: str
    plugin_version: str
    decisions: list[FeatureGenerationDecision]
    required_operator_ids: list[str] = Field(default_factory=list)
    configuration_only_feature_count: int = Field(ge=0)
    composition_feature_count: int = Field(ge=0)
    custom_code_feature_count: int = Field(ge=0)
    blocked_feature_count: int = Field(ge=0)
    requires_human_review: bool = False
    online_registration_allowed: bool = False


class OperatorExecutionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    operator_id: str
    value: float | None
    quality_flags: list[str] = Field(default_factory=list)


class ComputedFeatureValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_id: str
    value: float | None
    unit: str
    operator_id: str
    implementation_mode: FeatureImplementationMode = FeatureImplementationMode.CONFIGURATION_ONLY
    quality_flags: list[str] = Field(default_factory=list)
    operator_trace: list[OperatorExecutionRecord] = Field(default_factory=list)
