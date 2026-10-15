"""Machine-readable governance specification for AstroGraph Agent-callable tools.

R30 unifies three historically separate surfaces without changing their runtime
semantics:

* mission/platform tools registered through :class:`ToolRegistry`;
* the controlled diagnostic-loop capabilities;
* host-bound deterministic DiagnosticProbe kinds.

The catalogue is descriptive/governance metadata.  It never grants a permission
or makes an otherwise unavailable tool callable.  Runtime allow-lists, task
snapshots, evidence admission and independent governors remain authoritative.
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.agents.mission.modes import allowed_tools, TASK_TOOLS, COMMON_TOOLS
from astrograph.agents.mission.service import CATALOG
from astrograph.integrations.platform_tools import DISPLAY, PERMISSIONS, PURPOSE_ZH, _builtin_tool
from astrograph.investigation.diagnostic_probes import DiagnosticProbeKind
from astrograph.orchestration.llm_diagnostic_loop import ControlledDiagnosticDeliberationService
from astrograph.tools.agentic import build_agentic_tool_registry


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AgentToolSurface(str, Enum):
    MISSION_PLATFORM = "MISSION_PLATFORM"
    CONTROLLED_DIAGNOSTIC_CAPABILITY = "CONTROLLED_DIAGNOSTIC_CAPABILITY"
    DIAGNOSTIC_PROBE = "DIAGNOSTIC_PROBE"
    LOW_LEVEL_AGENTIC_REGISTRY = "LOW_LEVEL_AGENTIC_REGISTRY"
    DYNAMIC_OAG_EVIDENCE_CHECK = "DYNAMIC_OAG_EVIDENCE_CHECK"
    TRUSTED_PLUGIN_EXTENSION = "TRUSTED_PLUGIN_EXTENSION"


class InvocationAuthority(str, Enum):
    HOST_ORCHESTRATED = "HOST_ORCHESTRATED"
    AGENT_SELECTS_TOOL_ID_SCHEMA_BOUNDED_INPUT = "AGENT_SELECTS_TOOL_ID_SCHEMA_BOUNDED_INPUT"
    AGENT_SELECTS_CAPABILITY_NO_FREEFORM_EXECUTION = "AGENT_SELECTS_CAPABILITY_NO_FREEFORM_EXECUTION"
    AGENT_SELECTS_HOST_BOUND_PROBE_ID_ONLY = "AGENT_SELECTS_HOST_BOUND_PROBE_ID_ONLY"
    INTERNAL_AGENT_SELECTS_PREBUILT_CHECK = "INTERNAL_AGENT_SELECTS_PREBUILT_CHECK"
    RUNTIME_PLUGIN_POLICY_BOUND = "RUNTIME_PLUGIN_POLICY_BOUND"


class EvidenceAuthority(str, Enum):
    NONE = "NONE"
    READ_FORMAL_EVIDENCE = "READ_FORMAL_EVIDENCE"
    BUILD_FORMAL_EVIDENCE_DETERMINISTICALLY = "BUILD_FORMAL_EVIDENCE_DETERMINISTICALLY"
    SHADOW_DERIVED_OBSERVATION_ONLY = "SHADOW_DERIVED_OBSERVATION_ONLY"
    OFFLINE_EVALUATION_ONLY = "OFFLINE_EVALUATION_ONLY"


class DiagnosisAuthority(str, Enum):
    NONE = "NONE"
    DETERMINISTIC_CORE_EXECUTION = "DETERMINISTIC_CORE_EXECUTION"
    CANDIDATE_OR_CONTEXT_ONLY = "CANDIDATE_OR_CONTEXT_ONLY"
    SHADOW_REVIEW_ONLY = "SHADOW_REVIEW_ONLY"


class TruthAccessPolicy(str, Enum):
    FORBIDDEN = "FORBIDDEN"
    OFFLINE_EVALUATION_ONLY = "OFFLINE_EVALUATION_ONLY"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class AgentToolGovernanceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_key: str = Field(min_length=1)
    surface: AgentToolSurface
    runtime_tool_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    agent_selectable: bool
    invocation_authority: InvocationAuthority
    deterministic: bool | None
    side_effect_level: str
    required_permissions: tuple[str, ...] = ()
    allowed_tasks: tuple[str, ...] = ()
    allowed_work_modes: tuple[str, ...] = ()
    argument_contract: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    evidence_authority: EvidenceAuthority = EvidenceAuthority.NONE
    diagnosis_authority: DiagnosisAuthority = DiagnosisAuthority.NONE
    truth_access_policy: TruthAccessPolicy = TruthAccessPolicy.FORBIDDEN
    core_prediction_depends_on_agent_call: bool = False
    formal_release_authority: bool = False
    arbitrary_python_allowed: bool = False
    arbitrary_file_path_allowed: bool = False
    arbitrary_channel_allowed: bool = False
    arbitrary_window_allowed: bool = False
    arbitrary_threshold_allowed: bool = False
    arbitrary_database_query_allowed: bool = False
    provenance_requirement: str = "HOST_BOUND_OR_REGISTRY_SNAPSHOT"
    failure_policy: str = "FAIL_CLOSED"
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def governance_invariants(self) -> "AgentToolGovernanceSpec":
        if self.formal_release_authority:
            raise ValueError("AGENT_TOOL_RELEASE_AUTHORITY_FORBIDDEN")
        if any((
            self.arbitrary_python_allowed,
            self.arbitrary_file_path_allowed,
            self.arbitrary_channel_allowed,
            self.arbitrary_window_allowed,
            self.arbitrary_threshold_allowed,
            self.arbitrary_database_query_allowed,
        )):
            raise ValueError("R30_AGENT_TOOL_ARBITRARY_AUTHORITY_FORBIDDEN")
        if self.surface == AgentToolSurface.DIAGNOSTIC_PROBE:
            if self.evidence_authority != EvidenceAuthority.SHADOW_DERIVED_OBSERVATION_ONLY:
                raise ValueError("DIAGNOSTIC_PROBE_MUST_REMAIN_SHADOW_ONLY")
            if self.diagnosis_authority != DiagnosisAuthority.SHADOW_REVIEW_ONLY:
                raise ValueError("DIAGNOSTIC_PROBE_CANNOT_HAVE_DIAGNOSIS_AUTHORITY")
            if self.core_prediction_depends_on_agent_call:
                raise ValueError("CORE_DIAGNOSIS_CANNOT_DEPEND_ON_PROBE_EXECUTION")
        return self


class AgentToolGovernanceCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-agent-tool-governance-catalog.v1"
    policy_version: str = "R31-v1"
    tools: tuple[AgentToolGovernanceSpec, ...]
    runtime_extension_policy: dict[str, Any]
    global_invariants: dict[str, Any]
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def verify_hash(self) -> "AgentToolGovernanceCatalog":
        payload = {
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "tools": [item.model_dump(mode="json") for item in self.tools],
            "runtime_extension_policy": self.runtime_extension_policy,
            "global_invariants": self.global_invariants,
        }
        if _digest(payload) != self.catalog_sha256:
            raise ValueError("AGENT_TOOL_GOVERNANCE_CATALOG_SHA256_MISMATCH")
        keys = [item.tool_key for item in self.tools]
        if len(keys) != len(set(keys)):
            raise ValueError("AGENT_TOOL_GOVERNANCE_DUPLICATE_TOOL_KEY")
        return self


_MISSION_CONTEXT_ONLY = {
    "model.capabilities", "graph.reason", "knowledge.graph.mechanism",
    "knowledge.graph.match_observations", "rag.retrieve", "literature.search",
    "document.convert", "document.clean", "graph.build", "exploration.propose",
    "output.structured", "model.infer_candidate",
}


def _mission_specs() -> list[AgentToolGovernanceSpec]:
    rows: list[AgentToolGovernanceSpec] = []
    for tool_id in sorted(CATALOG):
        name, category = DISPLAY.get(tool_id, (tool_id, "其他"))
        modes = tuple(mode for mode in ("normal", "thinking", "creative") if tool_id in allowed_tools("diagnose", mode))
        tasks = tuple(sorted(task for task, ids in TASK_TOOLS.items() if tool_id in ids or tool_id in COMMON_TOOLS))
        if tool_id == "dataset.replay":
            truth = TruthAccessPolicy.OFFLINE_EVALUATION_ONLY
            evidence = EvidenceAuthority.OFFLINE_EVALUATION_ONLY
            diagnosis = DiagnosisAuthority.NONE
        else:
            truth = TruthAccessPolicy.FORBIDDEN
            evidence = EvidenceAuthority.NONE
            diagnosis = (
                DiagnosisAuthority.DETERMINISTIC_CORE_EXECUTION
                if tool_id in {"diagnosis.fuse", "model.infer", "model.execute_compatible"}
                else DiagnosisAuthority.CANDIDATE_OR_CONTEXT_ONLY
                if tool_id in _MISSION_CONTEXT_ONLY
                else DiagnosisAuthority.NONE
            )
        builtin = _builtin_tool(tool_id)
        rows.append(AgentToolGovernanceSpec(
            tool_key=f"mission:{tool_id}",
            surface=AgentToolSurface.MISSION_PLATFORM,
            runtime_tool_id=tool_id,
            display_name=name,
            purpose=PURPOSE_ZH.get(tool_id, CATALOG[tool_id]),
            agent_selectable=True,
            invocation_authority=InvocationAuthority.AGENT_SELECTS_TOOL_ID_SCHEMA_BOUNDED_INPUT,
            deterministic=(tool_id not in {"graph.build", "literature.search"}),
            side_effect_level=(
                "EXTERNAL_IO" if tool_id == "literature.search"
                else "WRITE_ARTIFACT" if tool_id in {"graph.build", "exploration.propose", "output.structured"}
                else "READ_ONLY"
            ),
            required_permissions=tuple(item.value for item in PERMISSIONS.get(tool_id, ())),
            allowed_tasks=tasks,
            allowed_work_modes=modes,
            argument_contract=(
                "TASK_SNAPSHOT_SCHEMA_BOUND; HOST_VALIDATES_PATHS/DATA/PROVIDERS; MODEL_CANNOT_NAME_PYTHON_CALLABLE"
            ),
            input_schema=builtin["spec"]["input_schema"],
            output_schema=builtin["spec"]["output_schema"],
            evidence_authority=evidence,
            diagnosis_authority=diagnosis,
            truth_access_policy=truth,
            core_prediction_depends_on_agent_call=False,
            notes=(f"UI category={category}",),
        ))
    return rows


def _diagnostic_capability_specs() -> list[AgentToolGovernanceSpec]:
    rows: list[AgentToolGovernanceSpec] = []
    for cap in ControlledDiagnosticDeliberationService.capabilities():
        is_probe = cap.capability_id == "execute_diagnostic_probe"
        rows.append(AgentToolGovernanceSpec(
            tool_key=f"diagnostic-capability:{cap.capability_id}",
            surface=AgentToolSurface.CONTROLLED_DIAGNOSTIC_CAPABILITY,
            runtime_tool_id=cap.capability_id,
            display_name=cap.capability_id,
            purpose=cap.description,
            agent_selectable=True,
            invocation_authority=(
                InvocationAuthority.AGENT_SELECTS_HOST_BOUND_PROBE_ID_ONLY
                if is_probe else InvocationAuthority.AGENT_SELECTS_CAPABILITY_NO_FREEFORM_EXECUTION
            ),
            deterministic=True,
            side_effect_level="READ_ONLY",
            required_permissions=(cap.permission,),
            allowed_tasks=("diagnose",),
            allowed_work_modes=("normal", "thinking", "creative"),
            argument_contract=(
                "PROBE_ID_ONLY_FROM_HOST_CATALOG" if is_probe
                else "NO_FREEFORM_TOOL_ARGUMENTS; INPUT_ARTIFACTS_ARE_HOST_BOUND"
            ),
            input_schema=(
                {"type":"object","properties":{"probe_id":{"type":"string"}},"required":["probe_id"],"additionalProperties":False}
                if is_probe else {"type":"object","additionalProperties":False}
            ),
            output_schema={"type":"object"},
            evidence_authority=(
                EvidenceAuthority.SHADOW_DERIVED_OBSERVATION_ONLY if is_probe
                else EvidenceAuthority.READ_FORMAL_EVIDENCE
            ),
            diagnosis_authority=(DiagnosisAuthority.SHADOW_REVIEW_ONLY if is_probe else DiagnosisAuthority.CANDIDATE_OR_CONTEXT_ONLY),
            truth_access_policy=TruthAccessPolicy.FORBIDDEN,
            core_prediction_depends_on_agent_call=False,
        ))
    return rows


_PROBE_PURPOSE = {
    "EVIDENCE_THRESHOLD_MARGIN": "Recompute distance from an already-frozen evidence threshold.",
    "PRE_POST_CHANGE": "Review a host-bound before/after time-series change.",
    "COMMAND_RESPONSE_RESIDUAL": "Review host-bound commanded versus observed response residual.",
    "LAG_CORRELATION": "Review a host-bound lagged association over admitted arrays.",
    "OSCILLATION_SIGNATURE": "Review a host-bound oscillation signature.",
    "SATURATION_LIMIT_CHECK": "Review whether admitted values approach a frozen saturation limit.",
    "MINIMAL_DIAGNOSIS_REVIEW": "Review deterministic conflicts and bounded minimal diagnoses.",
    "SIMULATION_COUNTERFACTUAL_REVIEW": "Compare host-admitted observed and counterfactual simulation traces.",
    "DXC_EXPECTED_RESPONSE_REVIEW": "Mirror the frozen R27 expected-response ranking.",
    "DXC_FORWARD_SIMULATION_REVIEW": "Mirror the frozen R28 qualitative forward-model decision.",
    "DXC_REDUCED_ORDER_SIMULATION_REVIEW": "Mirror the frozen R29 numerical reduced-order simulation decision.",
    "DXC_WATERPUMP_REDUCED_ORDER_SIMULATION_REVIEW": "Mirror the frozen R31 WaterPump current/flow reduced-order corroboration decision.",
}


def _probe_specs() -> list[AgentToolGovernanceSpec]:
    return [
        AgentToolGovernanceSpec(
            tool_key=f"probe:{kind.value}",
            surface=AgentToolSurface.DIAGNOSTIC_PROBE,
            runtime_tool_id=kind.value,
            display_name=kind.value,
            purpose=_PROBE_PURPOSE[kind.value],
            agent_selectable=True,
            invocation_authority=InvocationAuthority.AGENT_SELECTS_HOST_BOUND_PROBE_ID_ONLY,
            deterministic=True,
            side_effect_level="READ_ONLY_SHADOW",
            required_permissions=("DETERMINISTIC_SHADOW_EXECUTE",),
            allowed_tasks=("diagnose",),
            allowed_work_modes=("normal", "thinking", "creative"),
            argument_contract="ALL_CHANNELS/WINDOWS/THRESHOLDS/MODELS/SOURCE_IDENTITY_HOST_BOUND_BEFORE_AGENT_SELECTION",
            input_schema={"type":"object","properties":{"probe_id":{"type":"string"}},"required":["probe_id"],"additionalProperties":False},
            output_schema={"type":"object","required":["probe_id","kind","status","formal_evidence_eligible","formal_evidence_generated"]},
            evidence_authority=EvidenceAuthority.SHADOW_DERIVED_OBSERVATION_ONLY,
            diagnosis_authority=DiagnosisAuthority.SHADOW_REVIEW_ONLY,
            truth_access_policy=TruthAccessPolicy.FORBIDDEN,
            core_prediction_depends_on_agent_call=False,
            provenance_requirement="CONTENT_BOUND_DIAGNOSTIC_PROBE_SPEC_AND_SOURCE_ARTIFACT_SHA256",
        )
        for kind in DiagnosticProbeKind
    ]


def _low_level_agentic_specs() -> list[AgentToolGovernanceSpec]:
    registry = build_agentic_tool_registry(include_thermal_star=True)
    rows: list[AgentToolGovernanceSpec] = []
    for spec in registry.list_specs():
        if spec.tool_id in {"d3_rw.evidence.build", "d3_thermal_star.evidence.build"}:
            evidence = EvidenceAuthority.BUILD_FORMAL_EVIDENCE_DETERMINISTICALLY
            diagnosis = DiagnosisAuthority.NONE
        elif spec.tool_id.startswith("diagnosis.invoke"):
            evidence = EvidenceAuthority.READ_FORMAL_EVIDENCE
            diagnosis = DiagnosisAuthority.DETERMINISTIC_CORE_EXECUTION
        elif spec.tool_id == "evidence.catalog.read":
            evidence = EvidenceAuthority.READ_FORMAL_EVIDENCE
            diagnosis = DiagnosisAuthority.NONE
        else:
            evidence = EvidenceAuthority.NONE
            diagnosis = DiagnosisAuthority.CANDIDATE_OR_CONTEXT_ONLY
        rows.append(AgentToolGovernanceSpec(
            tool_key=f"agentic-registry:{spec.tool_id}",
            surface=AgentToolSurface.LOW_LEVEL_AGENTIC_REGISTRY,
            runtime_tool_id=spec.tool_id,
            display_name=spec.tool_id,
            purpose=f"Registered deterministic workflow tool {spec.tool_id}.",
            agent_selectable=False,
            invocation_authority=InvocationAuthority.HOST_ORCHESTRATED,
            deterministic=spec.deterministic,
            side_effect_level=spec.side_effect_level.value,
            required_permissions=tuple(item.value for item in spec.required_permissions),
            argument_contract="TOOLREGISTRY_JSON_SCHEMA_AND_WORKFLOW_CONTEXT_PERMISSION_BOUND",
            input_schema=spec.input_schema,
            output_schema=spec.output_schema,
            evidence_authority=evidence,
            diagnosis_authority=diagnosis,
            truth_access_policy=TruthAccessPolicy.FORBIDDEN,
            provenance_requirement="SEALED_TOOLREGISTRY_SNAPSHOT_WITH_VERSION_PINS",
            notes=("Not automatically exposed as a MissionAgent optional tool.",),
        ))
    return rows


def build_agent_tool_governance_catalog() -> AgentToolGovernanceCatalog:
    tools = tuple(sorted(
        _mission_specs() + _diagnostic_capability_specs() + _probe_specs() + _low_level_agentic_specs(),
        key=lambda item: item.tool_key,
    ))
    runtime_extension_policy = {
        "trusted_plugins_supported": True,
        "plugin_review_status_required": "APPROVED",
        "plugin_execution": "ISOLATED_PROCESS_SHA256_PINNED",
        "hot_reload_scope": "FUTURE_REGISTRY_SNAPSHOTS_ONLY",
        "runtime_plugin_tools_must_conform_to": "ToolSpec + plugin manifest allow-listed tasks/work-modes",
        "dynamic_oag_evidence_checks": {
            "pattern": "telemetry.compute:<feature_definition_id> / registered EvidenceCheckTool",
            "selection": "INTERNAL_AGENT_SELECTS_PREBUILT_CHECK",
            "formal_evidence": "ONLY_THROUGH_DETERMINISTIC_EVIDENCE_COMPILER; TOOL_OUTPUT_ALONE_IS_NOT_RELEASE_AUTHORITY",
        },
    }
    global_invariants = {
        "unregistered_tool_execution": False,
        "arbitrary_python": False,
        "arbitrary_file_paths": False,
        "arbitrary_channels_windows_thresholds_for_probes": False,
        "arbitrary_database_queries": False,
        "agent_release_authority": False,
        "probe_formal_evidence_authority": False,
        "core_detection_must_not_depend_on_optional_agent_probe": True,
        "offline_truth_access_is_segregated_from_online_diagnosis": True,
        "tool_registry_snapshot_sealed_per_task": True,
    }
    payload = {
        "schema_version": "astrograph-agent-tool-governance-catalog.v1",
        "policy_version": "R31-v1",
        "tools": [item.model_dump(mode="json") for item in tools],
        "runtime_extension_policy": runtime_extension_policy,
        "global_invariants": global_invariants,
    }
    return AgentToolGovernanceCatalog(
        tools=tools,
        runtime_extension_policy=runtime_extension_policy,
        global_invariants=global_invariants,
        catalog_sha256=_digest(payload),
    )


__all__ = [
    "AgentToolGovernanceCatalog",
    "AgentToolGovernanceSpec",
    "AgentToolSurface",
    "DiagnosisAuthority",
    "EvidenceAuthority",
    "InvocationAuthority",
    "TruthAccessPolicy",
    "build_agent_tool_governance_catalog",
]
