from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from astrograph.agents.agentic import (
    DiagnosticReasonerAgent,
    EvidenceNavigatorAgent,
    InvestigationOrchestratorAgent,
    SafetyCriticAgent,
    StructuredReasoningBackend,
    VerificationScientistAgent,
)
from astrograph.evidence import EpisodeEvidenceBundle, EvidenceDefinition
from astrograph.evaluation import EvidencePerturbationSpec, apply_evidence_perturbations
from astrograph.investigation import AgentTrace, AgentTraceEvent, DecisionGovernor, VerificationResult
from astrograph.knowledge import EpisodeEvidenceGraph
from astrograph.orchestration.models import ExecutionResult, PermissionScope, TaskEnvelope, WorkflowContext
from astrograph.tools import ToolRegistry
from astrograph.tools.agentic import build_agentic_tool_registry


class AgenticInvestigationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    d3_delivery_path: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    protocol_root: str = Field(min_length=1)
    evidence_catalog_path: str = Field(min_length=1)
    shadow_mode: bool = True
    evidence_perturbations: list[EvidencePerturbationSpec] = Field(default_factory=list)


class AgenticInvestigationWorkflow:
    workflow_id = "agentic.investigation.workflow.v0.5.2"
    workflow_version = "0.5.2"

    def __init__(
        self,
        *,
        reasoning_backend: StructuredReasoningBackend | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.reasoning_backend = reasoning_backend
        self.tool_registry = tool_registry or build_agentic_tool_registry()
        self.orchestrator = InvestigationOrchestratorAgent(
            tool_registry=self.tool_registry,
            backend=reasoning_backend,
        )
        self.navigator = EvidenceNavigatorAgent(tool_registry=self.tool_registry)
        self.verification_scientist = VerificationScientistAgent(tool_registry=self.tool_registry)
        self.reasoner = DiagnosticReasonerAgent(
            tool_registry=self.tool_registry,
            backend=reasoning_backend,
        )
        self.critic = SafetyCriticAgent(tool_registry=self.tool_registry)
        self.governor = DecisionGovernor()

    def invoke(self, envelope: TaskEnvelope, context: WorkflowContext) -> ExecutionResult:
        required = {
            PermissionScope.READ_DATA,
            PermissionScope.READ_KNOWLEDGE,
            PermissionScope.RUN_DIAGNOSIS,
            PermissionScope.WRITE_AUDIT,
        }
        missing = sorted(required - set(context.permissions), key=lambda item: item.value)
        if missing:
            return ExecutionResult.failed(
                task_id=envelope.task_id,
                workflow_id=self.workflow_id,
                code="ASTROGRAPH_AGENTIC_PERMISSION_DENIED",
                message="Agentic investigation requires read-data, read-knowledge, diagnosis, and audit permissions.",
                details={"missing_permissions": [item.value for item in missing]},
            )
        if self.reasoning_backend is not None:
            if envelope.execution_policy.deterministic_only:
                return ExecutionResult.failed(
                    task_id=envelope.task_id,
                    workflow_id=self.workflow_id,
                    code="ASTROGRAPH_LLM_BLOCKED_BY_DETERMINISTIC_POLICY",
                    message="Structured reasoning backend requires deterministic_only=false.",
                )
            if not envelope.execution_policy.allow_llm:
                return ExecutionResult.failed(
                    task_id=envelope.task_id,
                    workflow_id=self.workflow_id,
                    code="ASTROGRAPH_LLM_NOT_ALLOWED",
                    message="Structured reasoning backend requires allow_llm=true.",
                )
        try:
            request = AgenticInvestigationInput.model_validate(envelope.input_payload)
            if not request.shadow_mode:
                raise ValueError("V0.5.2 Agentic investigation is available only in shadow_mode")
            trace = AgentTrace(
                trace_id=f"TRACE-{context.run_id}",
                workflow_id=self.workflow_id,
                run_id=context.run_id,
                model_id=(
                    getattr(self.reasoning_backend, "model_id", "deterministic-reference-agent")
                    if self.reasoning_backend is not None
                    else "deterministic-reference-agent"
                ),
                model_version=(
                    getattr(self.reasoning_backend, "model_version", self.workflow_version)
                    if self.reasoning_backend is not None
                    else self.workflow_version
                ),
            )

            semantics_result = self.navigator.invoke_tool(
                "semantics.validate",
                {"protocol_root": request.protocol_root},
                context,
            )
            trace.append(
                AgentTraceEvent(
                    sequence=1,
                    agent_id=self.navigator.agent_id,
                    action="validate_telemetry_semantics",
                    input_summary={"protocol_root": request.protocol_root},
                    output_summary=semantics_result["validation"],
                    tool_id="semantics.validate",
                    tool_version=str(semantics_result["registry_version"]),
                )
            )

            catalog_result = self.navigator.invoke_tool(
                "evidence.catalog.read",
                {"evidence_catalog_path": request.evidence_catalog_path},
                context,
            )
            candidate_fault_ids = list(catalog_result["registered_fault_ids"])
            evidence_definitions = [
                EvidenceDefinition.model_validate(item)
                for item in catalog_result["evidence_definitions"]
            ]
            trace.append(
                AgentTraceEvent(
                    sequence=2,
                    agent_id=self.navigator.agent_id,
                    action="read_executable_evidence_catalog",
                    input_summary={"evidence_catalog_path": request.evidence_catalog_path},
                    output_summary={
                        "catalog_version": catalog_result["catalog_version"],
                        "catalog_sha256": catalog_result["catalog_sha256"],
                        "registered_fault_ids": candidate_fault_ids,
                        "evidence_definition_count": len(evidence_definitions),
                    },
                    tool_id="evidence.catalog.read",
                    tool_version=str(catalog_result["catalog_version"]),
                )
            )

            plan = self.orchestrator.plan(
                task_id=envelope.task_id,
                run_id=request.run_id,
                candidate_fault_ids=candidate_fault_ids,
            )
            trace.append(
                AgentTraceEvent(
                    sequence=3,
                    agent_id=self.orchestrator.agent_id,
                    action="create_investigation_plan",
                    input_summary={"run_id": request.run_id, "candidate_fault_ids": candidate_fault_ids},
                    output_summary={"plan_id": plan.plan_id, "step_count": len(plan.steps)},
                )
            )

            graph_query_plan = self.navigator.navigate(
                candidate_fault_ids=candidate_fault_ids,
                definitions=evidence_definitions,
            )
            trace.append(
                AgentTraceEvent(
                    sequence=4,
                    agent_id=self.navigator.agent_id,
                    action="plan_evidence_navigation",
                    input_summary={"candidate_fault_ids": candidate_fault_ids},
                    output_summary={
                        "query_plan_id": graph_query_plan.query_plan_id,
                        "relation_types": graph_query_plan.allowed_relation_types,
                        "include_counter_evidence": graph_query_plan.include_counter_evidence,
                    },
                )
            )

            evidence_tool_result = self.verification_scientist.invoke_tool(
                "d3_rw.evidence.build",
                {
                    "d3_delivery_path": request.d3_delivery_path,
                    "run_id": request.run_id,
                    "protocol_root": request.protocol_root,
                    "evidence_catalog_path": request.evidence_catalog_path,
                },
                context,
            )
            bundle = EpisodeEvidenceBundle.model_validate(evidence_tool_result["bundle"])
            bundle, perturbation_audit = apply_evidence_perturbations(bundle, request.evidence_perturbations)
            verification_plan = self.verification_scientist.build_plan(bundle)
            verification_results = self._verification_results(bundle)
            trace.append(
                AgentTraceEvent(
                    sequence=5,
                    agent_id=self.verification_scientist.agent_id,
                    action="execute_registered_numeric_verification",
                    input_summary={
                        "verification_plan_id": verification_plan.verification_plan_id,
                        "task_count": len(verification_plan.tasks),
                    },
                    output_summary={
                        "episode_id": bundle.episode.episode_id,
                        "response_window_id": (
                            None if bundle.response_window is None else bundle.response_window.episode_id
                        ),
                        "analysis_wheel_index": bundle.analysis_wheel_index,
                        "selected_wheel_index": bundle.selected_wheel_index,
                        "wheel_selection_status": bundle.wheel_selection_status,
                        "feature_result_count": len(bundle.feature_results),
                        "evidence_fact_count": len(bundle.evidence_facts),
                        "evaluation_perturbation_count": len(perturbation_audit),
                    },
                    tool_id="d3_rw.evidence.build",
                    tool_version=str(evidence_tool_result["builder_version"]),
                )
            )

            hypothesis_set = self.reasoner.reason(bundle)
            trace.append(
                AgentTraceEvent(
                    sequence=6,
                    agent_id=self.reasoner.agent_id,
                    action="generate_competing_hypotheses",
                    input_summary={"evidence_fact_count": len(bundle.evidence_facts)},
                    output_summary={
                        "hypothesis_set_id": hypothesis_set.hypothesis_set_id,
                        "reasoning_backend": (
                            getattr(self.reasoning_backend, "last_call_metadata", {})
                            if self.reasoning_backend is not None
                            else {"mode": "deterministic-reference"}
                        ),
                        "hypotheses": [
                            {
                                "hypothesis_id": item.hypothesis_id,
                                "fault_id": item.target_fault_id,
                                "status": item.status.value,
                            }
                            for item in hypothesis_set.hypotheses
                        ],
                    },
                )
            )

            critique = self.critic.critique(hypothesis_set, bundle)
            trace.append(
                AgentTraceEvent(
                    sequence=7,
                    agent_id=self.critic.agent_id,
                    action="critique_hypotheses",
                    input_summary={"hypothesis_set_id": hypothesis_set.hypothesis_set_id},
                    output_summary={
                        "critique_id": critique.critique_id,
                        "blocking": critique.blocking,
                        "finding_codes": [item.code for item in critique.findings],
                    },
                )
            )

            evaluation_case = str(bundle.evaluation_metadata.get("truth_case_id", ""))
            core_result = self.tool_registry.invoke(
                "diagnosis.invoke_core",
                {
                    "wheel_evidence": evidence_tool_result["core_evidence"],
                    "nominal_case": evaluation_case == "nominal",
                },
                context,
            )
            trace.append(
                AgentTraceEvent(
                    sequence=8,
                    agent_id="diagnosis-core-v0.4.4",
                    action="run_deterministic_core_comparison",
                    input_summary={"wheel_evidence_count": len(evidence_tool_result["core_evidence"])},
                    output_summary=core_result,
                    tool_id="diagnosis.invoke_core",
                    tool_version="0.4.4",
                )
            )

            decision = self.governor.decide(
                hypothesis_set=hypothesis_set,
                critique=critique,
                deterministic_core_result=core_result,
                protocol_versions={
                    "astrograph": self.workflow_version,
                    "telemetry_semantics": str(semantics_result["registry_version"]),
                    "evidence_catalog": str(catalog_result["catalog_version"]),
                    "deterministic_core": "0.4.4",
                },
            )
            trace.append(
                AgentTraceEvent(
                    sequence=9,
                    agent_id=self.governor.governor_id,
                    action="govern_shadow_decision",
                    input_summary={
                        "hypothesis_set_id": hypothesis_set.hypothesis_set_id,
                        "critique_id": critique.critique_id,
                    },
                    output_summary={
                        "decision_id": decision.decision_id,
                        "disposition": decision.disposition.value,
                        "accepted_fault_id": decision.accepted_fault_id,
                        "comparison_status": decision.comparison_status,
                        "formal_release_allowed": decision.formal_release_allowed,
                    },
                )
            )

            graph_document = self.tool_registry.invoke(
                "evidence_graph.build",
                {
                    "bundle": bundle.model_dump(mode="json"),
                    "hypothesis_set": hypothesis_set.model_dump(mode="json"),
                    "decision_record": decision.model_dump(mode="json"),
                },
                context,
            )
            evidence_graph = EpisodeEvidenceGraph.model_validate(graph_document)
            trace.append(
                AgentTraceEvent(
                    sequence=10,
                    agent_id="episode-evidence-graph-builder",
                    action="build_dynamic_episode_evidence_graph",
                    input_summary={
                        "episode_id": bundle.episode.episode_id,
                        "hypothesis_set_id": hypothesis_set.hypothesis_set_id,
                        "decision_id": decision.decision_id,
                    },
                    output_summary={
                        "node_count": len(evidence_graph.nodes),
                        "edge_count": len(evidence_graph.edges),
                        "persistent_raw_telemetry": evidence_graph.persistent_raw_telemetry,
                    },
                    tool_id="evidence_graph.build",
                    tool_version="0.5.0",
                )
            )
            trace.termination_reason = (
                "shadow_decision_completed_with_human_review"
                if decision.requires_human_review
                else "shadow_decision_completed"
            )

            output = {
                "mode": "SHADOW",
                "official_result_source": "AstroGraph V0.4.4 deterministic Core",
                "formal_result_changed": False,
                "semantics_validation": semantics_result["validation"],
                "investigation_plan": plan.model_dump(mode="json"),
                "graph_query_plan": graph_query_plan.model_dump(mode="json"),
                "episode_evidence_bundle": bundle.model_dump(mode="json"),
                "evaluation_perturbations": [item.model_dump(mode="json") for item in perturbation_audit],
                "verification_plan": verification_plan.model_dump(mode="json"),
                "verification_results": [item.model_dump(mode="json") for item in verification_results],
                "hypothesis_set": hypothesis_set.model_dump(mode="json"),
                "critique_result": critique.model_dump(mode="json"),
                "decision_record": decision.model_dump(mode="json"),
                "episode_evidence_graph": evidence_graph.model_dump(mode="json"),
                "agent_trace": trace.model_dump(mode="json"),
            }
            warnings = [
                "V0.5.2 output is shadow-mode only and cannot replace the V0.4.4 official diagnosis.",
                "D3 data and thresholds are E0 synthetic engineering evidence, not flight-calibrated limits.",
                "Mechanism Episode and downstream response window are distinct temporal objects.",
            ]
            if perturbation_audit:
                warnings.append("Evaluation-only perturbations were applied after deterministic evidence construction.")
            if evaluation_case == "nominal":
                warnings.append(
                    "Nominal replay comparison uses evaluation metadata only for the V0.4.4 offline Core switch."
                )
            return ExecutionResult.succeeded(
                task_id=envelope.task_id,
                workflow_id=self.workflow_id,
                output_payload=output,
                tool_versions={spec.tool_id: spec.tool_version for spec in self.tool_registry.list_specs()},
                warnings=warnings,
                metadata={
                    "run_id": request.run_id,
                    "episode_id": bundle.episode.episode_id,
                    "response_window_id": (
                        None if bundle.response_window is None else bundle.response_window.episode_id
                    ),
                    "analysis_wheel_index": bundle.analysis_wheel_index,
                    "selected_wheel_index": bundle.selected_wheel_index,
                    "wheel_selection_status": bundle.wheel_selection_status,
                    "shadow_mode": True,
                },
            )
        except Exception as exc:
            return ExecutionResult.failed(
                task_id=envelope.task_id,
                workflow_id=self.workflow_id,
                code="ASTROGRAPH_AGENTIC_INVESTIGATION_FAILED",
                message=f"Agentic investigation failed: {type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _verification_results(bundle: EpisodeEvidenceBundle) -> list[VerificationResult]:
        results: list[VerificationResult] = []
        for feature_result in bundle.feature_results:
            results.append(
                VerificationResult(
                    verification_task_id=f"VT-FEATURE-{feature_result.feature_definition_id}",
                    status="SUCCEEDED" if feature_result.value is not None else "FAILED",
                    output_refs={"feature_result_id": feature_result.feature_result_id},
                    output_payload={
                        "value": feature_result.value,
                        "unit": feature_result.unit,
                        "quality_flags": feature_result.quality_flags,
                    },
                    tool_id="feature.compute",
                    tool_version=feature_result.algorithm_version,
                    data_hashes=[feature_result.input_data_sha256],
                    warnings=feature_result.quality_flags,
                )
            )
        for fact in bundle.evidence_facts:
            results.append(
                VerificationResult(
                    verification_task_id=f"VT-EVIDENCE-{fact.evidence_definition_id}",
                    status=(
                        "FAILED"
                        if fact.status.value in {"MISSING_DATA", "INVALID_DATA"}
                        else "SUCCEEDED"
                    ),
                    output_refs={"evidence_fact_id": fact.evidence_fact_id},
                    output_payload={
                        "status": fact.status.value,
                        "observed_value": fact.observed_value,
                        "threshold_value": fact.threshold_value,
                        "unit": fact.unit,
                    },
                    tool_id="evidence.evaluate",
                    tool_version="0.5.0",
                    data_hashes=[],
                )
            )
        return results
