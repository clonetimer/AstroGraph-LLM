"""Mission-Agent work-mode adapter for upload-to-report jobs.

The upload platform deliberately keeps its existing truth-isolated diagnosis kernels.
A selected Mission Agent work mode is nevertheless an executed server-side policy:
``thinking`` performs an additional, bounded evidence investigation after the diagnosis
has been frozen, and ``creative`` may additionally request falsifiable, review-only
hypotheses from the configured structured LLM backend.  Neither path can edit the
frozen candidate, thresholds, BiRRF input, Active KG, confirmed memory, or control
permissions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from astrograph.agents.mission.models import (
    AgentNarrative,
    AgentRequest,
    AgentSettings,
    ExplorationProposal,
    ModeBudget,
    Strict,
    WorkMode,
)
from astrograph.agents.mission.modes import allowed_tools, resolve_mode
from astrograph.integrations.platform_knowledge import (
    UploadedDocumentKnowledgeIndex,
    _validate_observation_numeric_grounding,
)


PLATFORM_AGENT_EXECUTION_SCHEMA_VERSION = "astrograph-platform-agent-execution.v1"


class PlatformInvestigationFinding(Strict):
    kind: Literal["CONTEXT", "LIMITATION", "DISCRIMINATING_OBSERVATION"]
    text: str = Field(min_length=1, max_length=1800)
    citation_ids: list[str] = Field(default_factory=list, max_length=16)


class PlatformInvestigation(Strict):
    status: Literal["EVIDENCE_GROUNDED_INVESTIGATION"] = "EVIDENCE_GROUNDED_INVESTIGATION"
    candidate_fault_id: str | None = None
    competing_fault_ids: list[str] = Field(default_factory=list, max_length=24)
    findings: list[PlatformInvestigationFinding] = Field(default_factory=list, max_length=16)
    required_observations: list[str] = Field(default_factory=list, max_length=16)
    stopping_reason: str = Field(max_length=1200)
    not_a_new_diagnosis: Literal[True] = True
    diagnostic_vote_allowed: Literal[False] = False


def platform_agent_settings(*, llm_available: bool) -> AgentSettings:
    """Return host-owned upload-platform budgets for the three canonical modes."""
    return AgentSettings(
        planner="llm_assisted" if llm_available else "deterministic",
        max_checks=3,
        max_optional_actions=2,
        max_llm_calls=4 if llm_available else 0,
        work_mode_budgets={
            "thinking": ModeBudget(
                max_checks=5,
                max_optional_actions=3,
                max_llm_calls=8 if llm_available else 0,
                max_proposals=3,
            ),
            "creative": ModeBudget(
                max_checks=6,
                max_optional_actions=4,
                max_llm_calls=10 if llm_available else 0,
                max_proposals=3,
            ),
        },
    )


def _canonical_sha(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _candidate_ids(snapshot: dict[str, Any]) -> list[str]:
    values = [str(x) for x in snapshot.get("candidate_fault_ids") or [] if str(x).strip()]
    primary = snapshot.get("candidate_fault_id")
    if primary and str(primary) not in values:
        values.insert(0, str(primary))
    return values[:24]


def _visible_evidence_ids(snapshot: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for item in snapshot.get("evidence") or []:
        if isinstance(item, dict) and item.get("evidence_id"):
            result.append(str(item["evidence_id"]))
    return result


def _investigation_query(snapshot: dict[str, Any]) -> str:
    candidates = _candidate_ids(snapshot)
    primary = snapshot.get("candidate_fault_id")
    evidence_fields = [
        str(item.get("field"))
        for item in snapshot.get("evidence") or []
        if isinstance(item, dict) and item.get("field")
    ]
    return " ".join(
        part
        for part in [
            "spacecraft telemetry fault differential diagnosis counterevidence distinguishing observations",
            str(primary or ""),
            " ".join(candidates[1:5]),
            " ".join(evidence_fields[:8]),
        ]
        if part
    )[:2400]


def _validate_narrative(narrative: AgentNarrative, *, snapshot: dict[str, Any], visible: set[str]) -> None:
    frozen = snapshot.get("candidate_fault_id")
    if narrative.candidate_fault_id not in {None, frozen}:
        raise ValueError("agent narrative attempted to replace the frozen candidate")
    cited: list[str] = list(narrative.excerpt_ids)
    evidence_ids = set(_visible_evidence_ids(snapshot))
    for paragraph in narrative.paragraphs:
        cited.extend(paragraph.citation_ids)
        paragraph_citations = set(paragraph.citation_ids)
        evidence_citations = paragraph_citations & evidence_ids
        if evidence_citations:
            _validate_observation_numeric_grounding(
                paragraph.text, evidence_citations, snapshot, context="agent narrative paragraph"
            )
    unknown = sorted(set(cited) - visible)
    if unknown:
        raise ValueError("agent narrative contains unknown citation ids: " + ", ".join(unknown[:8]))


def _validate_exploration(proposal: ExplorationProposal, *, visible: set[str], max_proposals: int) -> None:
    if len(proposal.hypotheses) > max_proposals:
        raise ValueError("exploration proposal exceeds host mode budget")
    unknown = sorted({cid for item in proposal.hypotheses for cid in item.citation_ids} - visible)
    if unknown:
        raise ValueError("exploration proposal contains unknown citation ids: " + ", ".join(unknown[:8]))
    if proposal.active_release_allowed or proposal.confirmed_memory_write_allowed:
        raise ValueError("exploration proposal attempted to cross a governance boundary")


def execute_platform_agent_mode(
    *,
    job_id: str,
    task: str,
    work_mode: WorkMode,
    diagnosis_snapshot: dict[str, Any],
    document_paths: list[Path],
    knowledge_service: Any,
    output_root: Path,
) -> dict[str, Any]:
    """Execute the selected work mode without changing a frozen diagnosis."""
    output_root.mkdir(parents=True, exist_ok=True)
    llm_backend = getattr(knowledge_service, "llm_backend", None)
    settings = platform_agent_settings(llm_available=llm_backend is not None)
    request = AgentRequest(
        task_id=(job_id.replace("JOB-", "platform-").lower())[:80],
        task="diagnose",
        work_mode=work_mode,
        goal=task or "Diagnose spacecraft faults using data and verified mechanism evidence.",
        artifacts={"frozen_diagnosis": "platform:frozen-diagnosis"},
        documents=[path.name for path in document_paths],
    )
    effective, mode_receipt = resolve_mode(settings, request, backend_available=llm_backend is not None)

    frozen_sha = _canonical_sha(diagnosis_snapshot)
    execution: dict[str, Any] = {
        "schema_version": PLATFORM_AGENT_EXECUTION_SCHEMA_VERSION,
        "engine": "MISSION_AGENT_WORK_MODE_ADAPTER",
        "work_mode": work_mode,
        "mode_receipt": mode_receipt,
        "frozen_diagnosis_sha256": frozen_sha,
        "diagnostic_candidate_modified": False,
        "diagnostic_vote_allowed": False,
        "active_kg_modified": False,
        "confirmed_memory_write_allowed": False,
        "control_actions_authorized": False,
        "allowed_tools": sorted(allowed_tools("diagnose", work_mode)),
        "executed_actions": [
            {
                "tool_id": "diagnosis.fuse",
                "status": "COMPLETED_BY_EXISTING_GOVERNED_DIAGNOSTIC_KERNEL",
                "diagnostic_candidate_modified": False,
            }
        ],
        "investigation": None,
        "exploration": None,
        "limitations": [],
    }

    if work_mode == "normal":
        execution["executed_actions"].append({
            "tool_id": "output.structured",
            "status": "COMPLETED",
            "note": "Normal mode preserves the bounded established diagnosis path.",
        })
        _write_execution(output_root, execution)
        return execution

    index = UploadedDocumentKnowledgeIndex(top_k=min(8, max(3, effective.max_checks)))
    chunks, receipts = index.build(document_paths)
    query = _investigation_query(diagnosis_snapshot)
    hits = index.retrieve(query, chunks)
    visible = set(_visible_evidence_ids(diagnosis_snapshot)) | {
        str(item.get("knowledge_id")) for item in hits if item.get("knowledge_id")
    }
    execution["executed_actions"].append({
        "tool_id": "rag.retrieve",
        "status": "COMPLETED" if hits else "NO_HITS",
        "query": query,
        "hit_count": len(hits),
        "document_receipts": receipts,
        "diagnostic_vote_allowed": False,
    })

    candidates = _candidate_ids(diagnosis_snapshot)
    deterministic_findings = [
        PlatformInvestigationFinding(
            kind="CONTEXT",
            text=str(item.get("quote") or "")[:1600],
            citation_ids=[str(item["knowledge_id"])],
        )
        for item in hits[: min(5, effective.max_checks)]
        if item.get("quote") and item.get("knowledge_id")
    ]
    if not deterministic_findings:
        deterministic_findings = [
            PlatformInvestigationFinding(
                kind="LIMITATION",
                text="当前任务没有检索到可用于竞争假设复核的上传文档片段；模式不会因此补造证据。",
                citation_ids=[],
            )
        ]
    required_observations = [
        f"补充能够区分 {candidates[0]} 与 {candidate} 的独立遥测或检修观测。"
        for candidate in candidates[1:4]
    ] if candidates else ["补充独立、可追溯的遥测或检修观测以缩小候选范围。"]
    investigation = PlatformInvestigation(
        candidate_fault_id=diagnosis_snapshot.get("candidate_fault_id"),
        competing_fault_ids=candidates,
        findings=deterministic_findings,
        required_observations=required_observations,
        stopping_reason=(
            "达到本模式的有界复核步骤；上传文档仅作上下文，未产生新的诊断投票。"
            if hits else "没有新增可检索证据，按思考模式停止条件结束。"
        ),
    ).model_dump(mode="json")

    if llm_backend is not None and effective.max_llm_calls > 0 and visible:
        try:
            narrative = llm_backend.generate_structured(
                agent_id="platform-agent-thinking",
                output_model=AgentNarrative,
                payload={
                    "instruction": (
                        "Investigate the frozen diagnosis only. Do not replace the candidate, create evidence, "
                        "or issue control commands. Cite only visible evidence_id/knowledge_id values. "
                        "Separate limitations from cited observations."
                    ),
                    "frozen_diagnosis": diagnosis_snapshot,
                    "retrieved_context": hits,
                    "visible_citation_ids": sorted(visible),
                    "mode_budget": mode_receipt.get("budgets", {}),
                },
            )
            _validate_narrative(narrative, snapshot=diagnosis_snapshot, visible=visible)
            investigation["llm_narrative"] = narrative.model_dump(mode="json")
            execution["executed_actions"].append({
                "tool_id": "output.structured",
                "status": "COMPLETED",
                "purpose": "EVIDENCE_GROUNDED_INVESTIGATION_NARRATIVE",
                "backend_receipt": getattr(llm_backend, "last_call_metadata", {}),
            })
        except Exception as exc:  # diagnosis must not fail because optional agent narration failed
            execution["limitations"].append(
                f"思考模式结构化叙述失败：{type(exc).__name__}: {exc}"
            )
    elif llm_backend is None:
        execution["limitations"].append("未配置结构化大模型；思考模式执行确定性检索与证据缺口整理。")

    execution["investigation"] = investigation

    if work_mode == "creative":
        max_proposals = int(mode_receipt.get("max_proposals") or 3)
        if llm_backend is not None and effective.max_llm_calls > 0 and visible:
            try:
                proposal = llm_backend.generate_structured(
                    agent_id="platform-agent-creative",
                    output_model=ExplorationProposal,
                    payload={
                        "instruction": (
                            "Propose only falsifiable, offline investigation hypotheses around the frozen diagnosis. "
                            "Every hypothesis must cite visible evidence. Hypotheses are UNVERIFIED, cannot vote in "
                            "diagnosis, cannot enter confirmed memory, cannot authorize release or control actions."
                        ),
                        "frozen_diagnosis": diagnosis_snapshot,
                        "investigation": investigation,
                        "visible_citation_ids": sorted(visible),
                        "maximum_hypotheses": max_proposals,
                    },
                )
                _validate_exploration(proposal, visible=visible, max_proposals=max_proposals)
                execution["exploration"] = {
                    "status": "UNVERIFIED_OFFLINE_PROPOSALS",
                    **proposal.model_dump(mode="json"),
                }
                execution["executed_actions"].append({
                    "tool_id": "exploration.propose",
                    "status": "COMPLETED",
                    "proposal_count": len(proposal.hypotheses),
                    "diagnostic_vote_allowed": False,
                    "backend_receipt": getattr(llm_backend, "last_call_metadata", {}),
                })
            except Exception as exc:
                execution["exploration"] = {
                    "status": "FAILED_OPTIONAL_EXPLORATION",
                    "hypotheses": [],
                    "diagnostic_vote_allowed": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
                execution["limitations"].append("创造模式探索提案失败；冻结诊断保持不变。")
        else:
            execution["exploration"] = {
                "status": "RESEARCH_AGENDA_ONLY_NO_LLM",
                "hypotheses": [],
                "required_observations": required_observations,
                "diagnostic_vote_allowed": False,
                "active_release_allowed": False,
                "confirmed_memory_write_allowed": False,
                "note": "未配置结构化大模型，因此不生成看似具体但无模型支持的探索假设。",
            }
            execution["executed_actions"].append({
                "tool_id": "exploration.propose",
                "status": "DEGRADED_NO_LLM",
                "proposal_count": 0,
                "diagnostic_vote_allowed": False,
            })

    _write_execution(output_root, execution)
    return execution


def _write_execution(output_root: Path, execution: dict[str, Any]) -> None:
    (output_root / "agent_execution.json").write_text(
        json.dumps(execution, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
