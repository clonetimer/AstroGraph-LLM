from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from astrograph.agents.agentic import DiagnosticReasonerAgent
from astrograph.diagnosis.case_catalog import FaultCaseCatalog
from astrograph.evidence import (
    CONFIGURED_EVIDENCE_BUILDER_ID,
    LEGACY_WHOLE_SPACECRAFT_EVIDENCE_BUILDER_ID,
    EpisodeEvidenceBundle,
    RuntimeEvidenceBuildService,
)
from astrograph.investigation import DecisionRecord, HypothesisSet, HypothesisStatus, HypothesisType
from astrograph.knowledge import build_episode_evidence_graph
from astrograph.orchestration.models import PermissionScope, SideEffectLevel
from astrograph.compat.evidence import build_legacy_whole_spacecraft_evidence_builder
from astrograph.tools.agentic import build_agentic_tool_registry
from astrograph.tools.registry import ToolDefinition, ToolRegistry, ToolSpec

_DEFAULT_FAULT_CASE_CATALOG = FaultCaseCatalog.packaged_default()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class _OnlineRuntimeEvidenceBuildTool:
    def __init__(self) -> None:
        self.service = RuntimeEvidenceBuildService(
            legacy_builder_factory=build_legacy_whole_spacecraft_evidence_builder
        )

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.service.build(payload)


def _invoke_online_core(payload: dict[str, Any]) -> dict[str, Any]:
    """Independent deterministic reference computed from evidence, never from LLM hypotheses."""
    bundle = EpisodeEvidenceBundle.model_validate(payload["bundle"])
    reference = DiagnosticReasonerAgent.reference_hypotheses(bundle)
    supported = [
        item
        for item in reference.hypotheses
        if item.hypothesis_type == HypothesisType.REGISTERED_FAULT
        and item.status == HypothesisStatus.SUPPORTED
    ]
    if len(supported) == 1:
        result: dict[str, Any] = {
            "diagnosis_status": "DIAGNOSED",
            "predicted_case_id": _DEFAULT_FAULT_CASE_CATALOG.case_id_for_fault(
                supported[0].target_fault_id
            ),
            "supported_fault_ids": [supported[0].target_fault_id],
        }
    elif len(supported) > 1:
        result = {
            "diagnosis_status": "AMBIGUOUS",
            "predicted_case_id": None,
            "supported_fault_ids": [item.target_fault_id for item in supported],
        }
    elif int(bundle.analysis_metadata.get("generic_anomaly_field_count", 0)) > 0:
        result = {
            "diagnosis_status": "OUT_OF_SCOPE",
            "predicted_case_id": None,
            "supported_fault_ids": [],
        }
    else:
        result = {
            "diagnosis_status": "DIAGNOSED",
            "predicted_case_id": _DEFAULT_FAULT_CASE_CATALOG.normal_case_id,
            "supported_fault_ids": [],
        }
    result.update(
        {
            "core_version": "v0.5.9-online-independent-core-e0-v4",
            "reference_hypothesis_set": reference.model_dump(mode="json"),
            "reference_source": "deterministic evidence evaluation",
            "independence_level": "LLM_INDEPENDENT_SHARED_REFERENCE_IMPLEMENTATION",
        }
    )
    return result


def _build_online_evidence_graph(payload: dict[str, Any]) -> dict[str, Any]:
    graph = build_episode_evidence_graph(
        EpisodeEvidenceBundle.model_validate(payload["bundle"]),
        HypothesisSet.model_validate(payload["hypothesis_set"]),
        DecisionRecord.model_validate(payload["decision_record"]),
    )
    return graph.model_dump(mode="json")


def _persist_online_artifacts_transaction_body(payload: dict[str, Any], root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    for part in ("input", "diagnosis", "report", "audit", "runtime"):
        (root / part).mkdir(parents=True, exist_ok=True)

    documents: dict[str, tuple[Path, Any]] = {
        "simulation_event": (root / "input" / "simulation_event.json", payload["simulation_event"]),
        "diagnosis_execution_plan": (
            root / "diagnosis" / "diagnosis_execution_plan.json",
            payload["diagnosis_execution_plan"],
        ),
        "data_gate_report": (root / "input" / "data_gate_report.json", payload["data_gate_report"]),
        "episode_evidence_bundle": (
            root / "diagnosis" / "episode_evidence_bundle.json",
            payload["episode_evidence_bundle"],
        ),
        "verification_plan": (root / "diagnosis" / "verification_plan.json", payload["verification_plan"]),
        "hypothesis_set": (root / "diagnosis" / "hypothesis_set.json", payload["hypothesis_set"]),
        "research_hypotheses": (
            root / "diagnosis" / "research_hypotheses.json",
            {
                "schema_version": "astrograph-research-hypotheses.v1",
                "run_id": payload["simulation_event"]["run_id"],
                "hypotheses": [
                    item
                    for item in payload["hypothesis_set"].get("hypotheses", [])
                    if item.get("hypothesis_type") == "RESEARCH_HYPOTHESIS"
                ],
                "claim_boundary": (
                    "UNVERIFIED E0 research objects; never a formal composite diagnosis or control command."
                ),
            },
        ),
        "critique_result": (root / "diagnosis" / "critique_result.json", payload["critique_result"]),
        "deterministic_core_result": (
            root / "diagnosis" / "deterministic_core_result.json",
            payload["deterministic_core_result"],
        ),
        "decision_record": (root / "diagnosis" / "decision_record.json", payload["decision_record"]),
        "episode_evidence_graph": (
            root / "diagnosis" / "episode_evidence_graph.json",
            payload["episode_evidence_graph"],
        ),
        "structured_diagnosis_output": (
            root / "report" / "structured_diagnosis_output.json",
            payload["structured_diagnosis_output"],
        ),
        "agent_trace": (root / "audit" / "agent_trace.json", payload["agent_trace"]),
        "llm_runtime_snapshot": (
            root / "audit" / "llm_runtime_snapshot.json",
            payload.get("llm_runtime_snapshot", {
                "status": "NOT_PROVIDED",
                "runtime_snapshot": {
                    "schema_version": "astrograph-llm-runtime-snapshot.v1"
                },
            }),
        ),
    }
    if payload.get("candidate_fusion") is not None:
        documents["candidate_fusion"] = (
            root / "diagnosis" / "candidate_fusion.json",
            payload["candidate_fusion"],
        )
    if payload.get("document_claim_shadow") is not None:
        documents["document_claim_shadow"] = (
            root / "diagnosis" / "document_claim_shadow.json",
            payload["document_claim_shadow"],
        )
    if payload.get("soft_evidence") is not None:
        documents["soft_evidence"] = (
            root / "diagnosis" / "soft_evidence.json",
            payload["soft_evidence"],
        )
    if payload.get("soft_graph_shadow") is not None:
        documents["soft_graph_shadow"] = (
            root / "diagnosis" / "soft_graph_shadow.json",
            payload["soft_graph_shadow"],
        )
    if payload.get("pre_reasoning_retrieval") is not None:
        documents["pre_reasoning_retrieval"] = (
            root / "diagnosis" / "pre_reasoning_retrieval.json",
            payload["pre_reasoning_retrieval"],
        )
    if payload.get("paperseek_fallback_decision") is not None:
        documents["paperseek_fallback_decision"] = (
            root / "diagnosis" / "paperseek_fallback_decision.json",
            payload["paperseek_fallback_decision"],
        )
    if payload.get("llm_diagnostic_trajectory") is not None:
        documents["llm_diagnostic_trajectory"] = (
            root / "diagnosis" / "llm_diagnostic_trajectory.json",
            payload["llm_diagnostic_trajectory"],
        )
    if payload.get("dataset_profile_receipt") is not None:
        documents["dataset_profile_receipt"] = (
            root / "input" / "dataset_profile_receipt.json",
            payload["dataset_profile_receipt"],
        )
    if payload.get("component_spatiotemporal_fault_graph") is not None:
        documents["component_spatiotemporal_fault_graph"] = (
            root / "diagnosis" / "component_spatiotemporal_fault_graph.json",
            payload["component_spatiotemporal_fault_graph"],
        )
        documents["spatiotemporal_template_matches"] = (
            root / "diagnosis" / "spatiotemporal_template_matches.json",
            {
                "schema_version": "astrograph-spatiotemporal-template-match-set.v1",
                "matches": payload.get("spatiotemporal_template_matches", []),
                "claim_boundary": (
                    "Approximate template matches re-rank KG_FORWARD only and are not causal proof."
                ),
            },
        )
    if payload.get("dor_semantic_audit") is not None:
        documents["dor_semantic_audit"] = (
            root / "diagnosis" / "dor_semantic_audit.json",
            payload["dor_semantic_audit"],
        )
    if payload.get("dor_control_plane_audit") is not None:
        documents["dor_control_plane_audit"] = (
            root / "audit" / "dor_control_plane_audit.json",
            payload["dor_control_plane_audit"],
        )
    if payload.get("dor_multitask_audit") is not None:
        documents["dor_multitask_audit"] = (
            root / "audit" / "dor_multitask_audit.json",
            payload["dor_multitask_audit"],
        )
    if payload.get("dor_vertical_freeze_audit") is not None:
        documents["dor_vertical_freeze_audit"] = (
            root / "audit" / "dor_vertical_freeze_audit.json",
            payload["dor_vertical_freeze_audit"],
        )
    if payload.get("fec_evidence_audit") is not None:
        documents["fec_evidence_audit"] = (
            root / "audit" / "fec_model_evidence_audit.json",
            payload["fec_evidence_audit"],
        )
    if payload.get("isgn_routing_audit") is not None:
        documents["isgn_routing_audit"] = (
            root / "audit" / "isgn_mechanism_routing_audit.json",
            payload["isgn_routing_audit"],
        )
    if payload.get("isgn_observation_audit") is not None:
        documents["isgn_observation_audit"] = (
            root / "audit" / "isgn_observation_qualification.json",
            payload["isgn_observation_audit"],
        )
    if payload.get("diagnosis_evidence_bundle") is not None:
        documents["diagnosis_evidence_bundle"] = (
            root / "diagnosis" / "diagnosis_evidence_bundle.json",
            payload["diagnosis_evidence_bundle"],
        )
    if payload.get("diagnosis_policy_decision") is not None:
        documents["diagnosis_policy_decision"] = (
            root / "diagnosis" / "diagnosis_policy_decision.json",
            payload["diagnosis_policy_decision"],
        )
    runtime_stage_artifacts = payload.get("runtime_stage_artifacts") or {}
    if runtime_stage_artifacts:
        artifact_index = {
            "schema_version": "astrograph-runtime-stage-artifact-index.v1",
            "run_id": payload["simulation_event"]["run_id"],
            "artifacts": {},
        }
        for stage_id, artifact in sorted(runtime_stage_artifacts.items()):
            filename = stage_id.replace(".", "_") + ".json"
            relative_path = f"runtime/{filename}"
            key = "runtime_stage_artifact_" + stage_id.replace(".", "_")
            documents[key] = (root / relative_path, artifact)
            artifact_index["artifacts"][stage_id] = {
                "path": relative_path,
                "artifact_type": artifact.get("artifact_type"),
                "content_sha256": artifact.get("content_sha256"),
                "semantic_sha256": artifact.get("semantic_sha256"),
                "producer_stage_version": (artifact.get("provenance") or {}).get("producer_stage_version"),
            }
        documents["runtime_stage_artifact_index"] = (
            root / "runtime" / "stage_artifact_index.json", artifact_index
        )
    for path, document in documents.values():
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    decision = payload["decision_record"]
    structured = payload["structured_diagnosis_output"]
    validation = payload["hypothesis_set"].get("validation_metadata", {})
    research_items = [
        item
        for item in payload["hypothesis_set"].get("hypotheses", [])
        if item.get("hypothesis_type") == "RESEARCH_HYPOTHESIS"
    ]
    composite_members = sorted(
        {
            member
            for item in research_items
            for member in item.get("member_fault_ids", [])
        }
    )
    report = (
        "# AstroGraph 在线仿真诊断报告\n\n"
        f"- 运行ID：`{payload['simulation_event']['run_id']}`\n"
        f"- Reasoner模式：`{payload['reasoner_mode']}`\n"
        f"- 诊断计划：`{payload['diagnosis_execution_plan']['plan_id']}`\n"
        f"- 计划要求人工复核：`{payload['diagnosis_execution_plan']['requires_human_review']}`\n"
        f"- 模型：`{payload.get('model_id') or 'deterministic-reference-agent'}`\n"
        f"- 数据闸门：`{'PASS' if payload['data_gate_report']['passed'] else 'FAIL'}`\n"
        f"- 决策：`{decision['disposition']}`\n"
        f"- 接受故障：`{decision.get('accepted_fault_id') or '无'}`\n"
        f"- 是否人工复核：`{decision['requires_human_review']}`\n"
        f"- 对照状态：`{decision['comparison_status']}`\n"
        f"- 发布门状态：`{payload.get('diagnosis_policy_decision', {}).get('status', 'NOT_EVALUATED')}`\n"
        f"- DOR控制面资格：`{(payload.get('dor_control_plane_audit') or {}).get('status', 'NOT_CONFIGURED')}`\n"
        f"- DOR语义审计：`{(payload.get('dor_semantic_audit') or {}).get('status', 'NOT_CONFIGURED')}`\n"
        f"- DOR多任务一致性：`{(payload.get('dor_multitask_audit') or {}).get('status', 'NOT_CONFIGURED')}`\n"
        f"- DOR垂直验证/冻结门：`{(payload.get('dor_vertical_freeze_audit') or {}).get('status', 'NOT_CONFIGURED')}`\n"
        f"- FEC模型证据资格：`{(payload.get('fec_evidence_audit') or {}).get('status', 'NOT_CONFIGURED')}`\n"
        f"- ISGN机理路由资格：`{(payload.get('isgn_routing_audit') or {}).get('status', 'NOT_CONFIGURED')}`\n"
        f"- ISGN观测资格：`{(payload.get('isgn_observation_audit') or {}).get('status', 'NOT_CONFIGURED')}`\n"
        f"- LLM修复次数：`{validation.get('repair_attempt_count', 0)}`\n"
        f"- 提案协议：`{validation.get('proposal_protocol', 'deterministic')}`\n"
        f"- LLM提案数量：`{validation.get('llm_proposal_count', 0)}`\n"
        f"- 是否降级参考：`{validation.get('degraded_to_reference', False)}`\n"
        f"- 检索状态：`{structured['knowledge_and_retrieval_summary']['retrieval_status']}`\n"
        f"- 经验命中：`{structured['knowledge_and_retrieval_summary']['experience_hit_count']}`\n"
        f"- KG更新候选：`{len(structured['knowledge_and_retrieval_summary']['knowledge_update_candidates'])}`（待人工审核，未写入活动图谱）\n"
        f"- 文档图谱诊断影子：`{structured['knowledge_and_retrieval_summary']['document_graph_diagnosis_shadow']['status']}`"
        "（不修改BiRRF候选融合）\n"
        f"- 证据事实统计：`{structured['evidence_summary']['fact_status_counts']}`\n"
        f"- 软图影子建议：`{structured['uncertainty_and_soft_graph_summary']['suggested_action']}`"
        "（不参与BiRRF/Governor）\n"
        f"- Critic阻断：`{structured['safety_and_release_summary']['critique_blocking']}`\n"
        f"- 验证任务数量：`{structured['verification_summary']['task_count']}`\n"
        f"- 研究假设数量：`{len(research_items)}`\n"
        f"- 复合成员候选：`{', '.join(composite_members) if composite_members else '无'}`\n\n"
        "## 边界\n\n"
        "E0合成仿真影子结果；不代表飞行诊断、自动控制或正式复合故障能力。\n"
    )
    report_path = root / "report" / "summary.md"
    report_path.write_text(report, encoding="utf-8")
    documents["summary_report"] = (report_path, report)

    manifest = {
        "schema_version": "astrograph-online-artifact-manifest.v12",
        "commit_protocol": "directory-swap-transaction.v1",
        "transactional_commit": True,
        "generated_at": _now(),
        "run_id": payload["simulation_event"]["run_id"],
        "workflow_version": str(payload["workflow_version"]),
        "reasoner_mode": str(payload["reasoner_mode"]),
        "files": {
            key: {
                "path": str(path.relative_to(root)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for key, (path, _) in documents.items()
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path)}


def _persist_online_artifacts(payload: dict[str, Any]) -> dict[str, Any]:
    """Persist one online DAG result through a directory-level commit boundary.

    All files are materialized in a sibling staging directory first. The visible output
    directory is replaced only after the complete manifest has been written. On commit
    failure, an existing output directory is restored fail-closed.
    """

    final_root = Path(str(payload["artifact_output_dir"])).resolve()
    parent = final_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    staging_root = parent / f".{final_root.name}.astrograph-txn-{nonce}"
    backup_root = parent / f".{final_root.name}.astrograph-backup-{nonce}"
    had_existing = final_root.exists()
    try:
        result = _persist_online_artifacts_transaction_body(payload, staging_root)
        if had_existing:
            os.replace(final_root, backup_root)
        try:
            for attempt in range(5):
                try:
                    os.replace(staging_root, final_root)
                    break
                except PermissionError:
                    # Windows may briefly hold a newly written file handle while a
                    # scanner closes it. Retry only this atomic commit boundary;
                    # persistent denial still fails closed and restores the backup.
                    if os.name != "nt" or attempt == 4:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        except Exception:
            if had_existing and backup_root.exists() and not final_root.exists():
                os.replace(backup_root, final_root)
            raise
        if backup_root.exists():
            shutil.rmtree(backup_root)
        return {**result, "manifest_path": str(final_root / "manifest.json")}
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        if backup_root.exists() and not final_root.exists():
            os.replace(backup_root, final_root)
        raise


def build_online_tool_registry() -> ToolRegistry:
    registry = build_agentic_tool_registry(include_thermal_star=True)
    registry.register(
        ToolDefinition(
            spec=ToolSpec(
                tool_id="online.runtime_evidence.build",
                input_schema={
                    "type": "object",
                    "required": [
                        "simulation_event",
                        "protocol_root",
                        "runtime_evidence_builder_id",
                    ],
                },
                output_schema={"type": "object"},
                tool_version="0.6.6-explicit-runtime-evidence-profile-v1",
                deterministic=True,
                side_effect_level=SideEffectLevel.READ_ONLY,
                required_permissions=[PermissionScope.READ_DATA, PermissionScope.READ_KNOWLEDGE],
                timeout_s=180.0,
            ),
            handler=_OnlineRuntimeEvidenceBuildTool(),
        )
    )
    registry.register(
        ToolDefinition(
            spec=ToolSpec(
                tool_id="diagnosis.invoke_online_core",
                input_schema={"type": "object", "required": ["bundle"]},
                output_schema={"type": "object"},
                tool_version="0.5.9",
                deterministic=True,
                side_effect_level=SideEffectLevel.READ_ONLY,
                required_permissions=[PermissionScope.RUN_DIAGNOSIS],
            ),
            handler=_invoke_online_core,
        )
    )
    registry.register(
        ToolDefinition(
            spec=ToolSpec(
                tool_id="online.evidence_graph.build",
                input_schema={
                    "type": "object",
                    "required": ["bundle", "hypothesis_set", "decision_record"],
                },
                output_schema={"type": "object"},
                tool_version="0.5.9",
                deterministic=True,
                side_effect_level=SideEffectLevel.READ_ONLY,
                required_permissions=[PermissionScope.READ_DATA, PermissionScope.READ_KNOWLEDGE],
            ),
            handler=_build_online_evidence_graph,
        )
    )
    registry.register(
        ToolDefinition(
            spec=ToolSpec(
                tool_id="online.artifacts.persist",
                input_schema={"type": "object", "required": ["artifact_output_dir"]},
                output_schema={"type": "object"},
                tool_version="0.5.9",
                deterministic=True,
                side_effect_level=SideEffectLevel.WRITE_ARTIFACT,
                required_permissions=[PermissionScope.WRITE_ARTIFACTS, PermissionScope.WRITE_AUDIT],
            ),
            handler=_persist_online_artifacts,
        )
    )
    return registry
