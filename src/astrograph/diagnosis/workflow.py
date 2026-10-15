from __future__ import annotations

import uuid
from pathlib import Path

try:
    from langgraph.graph import END, START, StateGraph
except ModuleNotFoundError:  # pragma: no cover - exercised only when optional dependency is absent
    START = "__start__"
    END = "__end__"

    class _CompiledSequentialGraph:
        def __init__(self, nodes: dict, edges: list[tuple[str, str]]):
            self.nodes = nodes
            self.edges = edges

        def invoke(self, initial_state: dict) -> dict:
            state = dict(initial_state)
            current = START
            visited = 0
            while current != END:
                next_nodes = [dst for src, dst in self.edges if src == current]
                if not next_nodes:
                    raise RuntimeError(f"no outgoing edge from {current}")
                current = next_nodes[0]
                if current == END:
                    break
                update = self.nodes[current](state)
                if update:
                    state.update(update)
                visited += 1
                if visited > len(self.nodes) + 2:
                    raise RuntimeError("workflow graph appears cyclic; fallback only supports deterministic chains")
            return state

    class StateGraph:
        def __init__(self, state_type):
            self.state_type = state_type
            self._nodes = {}
            self._edges = []

        def add_node(self, name: str, func):
            self._nodes[name] = func

        def add_edge(self, src: str, dst: str):
            self._edges.append((src, dst))

        def compile(self):
            return _CompiledSequentialGraph(self._nodes, self._edges)

from astrograph.audit.store import AuditStore, NullAuditStore, sha256_json
from astrograph.config import DiagnosisPolicy
from astrograph.diagnosis.report import render_markdown
from astrograph.agents.report_agent import ReportAgent, ReportPayload
from astrograph.diagnosis.scoring import diagnose
from astrograph.evidence import EvidenceConstraintCatalog
from astrograph.knowledge.repository import KnowledgeRepository
from astrograph.schemas import (
    AnomalyEvidence,
    DiagnosticAuditRecord,
    DiagnosisResult,
    DiagnosisStatus,
    KnowledgeEdge,
    WorkflowInput,
    WorkflowState,
)


def build_workflow(
    repository: KnowledgeRepository,
    *,
    policy: DiagnosisPolicy | None = None,
    audit_store: AuditStore | None = None,
    constraints: EvidenceConstraintCatalog | None = None,
    report_agent: ReportAgent | None = None,
):
    policy = policy or DiagnosisPolicy()
    audit_store = audit_store or NullAuditStore()

    def retrieve_node(state: WorkflowState) -> dict:
        try:
            anomalies = [AnomalyEvidence.model_validate(item) for item in state.get("anomalies", [])]
            edges = repository.retrieve([item.anomaly_id for item in anomalies])
            return {"retrieved_edges": [edge.model_dump(mode="json") for edge in edges], "error": ""}
        except Exception as exc:  # repository errors must become SYSTEM_ERROR, never mock fallback
            return {"retrieved_edges": [], "error": f"knowledge_repository_error: {type(exc).__name__}: {exc}"}

    def diagnose_node(state: WorkflowState) -> dict:
        anomalies = [AnomalyEvidence.model_validate(item) for item in state.get("anomalies", [])]
        if state.get("error"):
            result = DiagnosisResult(
                status=DiagnosisStatus.SYSTEM_ERROR,
                reasons=[state["error"]],
                policy_version=policy.policy_version,
                knowledge_version=repository.knowledge_version,
                claim_boundary=policy.claim_boundary,
            )
        else:
            edges = [KnowledgeEdge.model_validate(item) for item in state.get("retrieved_edges", [])]
            condition_payload = state.get("operating_condition", {})
            condition_ids = set(condition_payload.get("condition_ids", [])) if isinstance(condition_payload, dict) else set()
            result = diagnose(
                anomalies,
                edges,
                policy=policy,
                knowledge_version=repository.knowledge_version,
                operating_condition_ids=condition_ids,
                constraints=constraints,
                temporal_evaluations=(state.get("metadata", {}) or {}).get("temporal_constraint_evaluations", {}),
            )
        return {"diagnosis": result.model_dump(mode="json")}

    def report_node(state: WorkflowState) -> dict:
        anomalies = [AnomalyEvidence.model_validate(item) for item in state.get("anomalies", [])]
        result = DiagnosisResult.model_validate(state["diagnosis"])
        if report_agent is not None:
            payload = ReportPayload(
                case_id=state["case_id"],
                request_id=state["request_id"],
                anomalies=anomalies,
                diagnosis=result,
                metadata=state.get("metadata", {}),
            )
            try:
                agent_result = report_agent.render(payload)
                return {
                    "report_markdown": str(agent_result.output["report_markdown"]),
                    "metadata": {
                        **(state.get("metadata", {}) or {}),
                        "report_agent": agent_result.model_dump(mode="json"),
                    },
                }
            except Exception as exc:
                return {
                    "report_markdown": render_markdown(
                        state["case_id"],
                        state["request_id"],
                        anomalies,
                        result,
                    ),
                    "metadata": {
                        **(state.get("metadata", {}) or {}),
                        "report_agent_warning": f"ReportAgent failed; deterministic report fallback used: {type(exc).__name__}: {exc}",
                    },
                }
        return {
            "report_markdown": render_markdown(
                state["case_id"],
                state["request_id"],
                anomalies,
                result,
            )
        }

    def audit_node(state: WorkflowState) -> dict:
        record = DiagnosticAuditRecord(
            audit_id=uuid.uuid4().hex,
            request_id=state["request_id"],
            case_id=state["case_id"],
            input_hash=sha256_json(
                {
                    "case_id": state["case_id"],
                    "request_id": state["request_id"],
                    "anomalies": state.get("anomalies", []),
                    "operating_condition": state.get("operating_condition", {}),
                }
            ),
            policy_hash=sha256_json(policy.model_dump(mode="json")),
            knowledge_hash=sha256_json(state.get("retrieved_edges", [])),
            report_hash=sha256_json(state.get("report_markdown", "")),
            knowledge_version=repository.knowledge_version,
            policy_version=policy.policy_version,
            anomalies=state.get("anomalies", []),
            retrieved_edges=state.get("retrieved_edges", []),
            diagnosis=state["diagnosis"],
            metadata=state.get("metadata", {}),
        )
        audit_store.append(record)
        return {"audit_record": record.model_dump(mode="json")}

    graph = StateGraph(WorkflowState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("diagnose", diagnose_node)
    graph.add_node("render", report_node)
    graph.add_node("audit", audit_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "diagnose")
    graph.add_edge("diagnose", "render")
    graph.add_edge("render", "audit")
    graph.add_edge("audit", END)
    return graph.compile()


def invoke_workflow(
    repository: KnowledgeRepository,
    request: WorkflowInput,
    *,
    policy: DiagnosisPolicy | None = None,
    audit_store: AuditStore | None = None,
    constraints: EvidenceConstraintCatalog | None = None,
    report_agent: ReportAgent | None = None,
) -> WorkflowState:
    app = build_workflow(repository, policy=policy, audit_store=audit_store, constraints=constraints, report_agent=report_agent)
    return app.invoke(
        {
            "case_id": request.case_id,
            "request_id": request.request_id,
            "anomalies": [item.model_dump(mode="json") for item in request.anomalies],
            "operating_condition": request.operating_condition,
            "metadata": request.metadata,
        }
    )
