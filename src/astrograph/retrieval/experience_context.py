"""Governed bridge from reviewed experience assets into pre-generation retrieval.

Experience is explanatory context, never episode evidence.  The bridge is
read-only and emits knowledge-update *candidates* for the existing human
governance workflow; it cannot publish or mutate the active graph.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from astrograph.data_driven.experience import ExperienceRecord, ExperienceStatus
from astrograph.data_driven.memory_layers import MemoryEntry, MemoryLayer

from .models import GovernedTextChunk, RetrievalLane


class ExperienceReuseReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-experience-reuse-receipt.v1"
    eligible_fault_ids: list[str] = Field(default_factory=list)
    selected_experience_ids: list[str] = Field(default_factory=list)
    selected_memory_entry_ids: list[str] = Field(default_factory=list)
    rejected_counts: dict[str, int] = Field(default_factory=dict)
    source_sha256: dict[str, str] = Field(default_factory=dict)
    human_review_required: bool = True
    active_graph_modified: bool = False
    claim_boundary: str = (
        "Reviewed experience may support retrieval and explanation only; it is not episode "
        "evidence, cannot change BiRRF, and cannot update the active KG without review and publish."
    )


class ExperienceKnowledgeUpdateCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-experience-knowledge-candidate.v1"
    candidate_id: str = Field(min_length=1)
    experience_id: str = Field(min_length=1)
    fault_ids: list[str] = Field(min_length=1)
    proposed_statement: str = Field(min_length=1)
    source_refs: list[str] = Field(default_factory=list)
    review_status: str = "HUMAN_REVIEW_REQUIRED"
    target_governance_workflow: str = "JsonKnowledgeGovernanceStore"
    active_graph_modified: bool = False


class ExperienceKnowledgeCandidateInbox:
    """Append-only curation inbox upstream of ``JsonKnowledgeGovernanceStore``.

    Staging is deliberately separate from drafting a KnowledgeRelease: a human
    curator must translate and validate the statement against the graph schema
    before the existing draft/submit/review/publish workflow can begin.
    """

    schema_version = "astrograph-experience-knowledge-candidate-inbox.v1"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def records(self) -> list[ExperienceKnowledgeUpdateCandidate]:
        if not self.path.is_file():
            return []
        return [
            ExperienceKnowledgeUpdateCandidate.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def stage(self, candidate: ExperienceKnowledgeUpdateCandidate) -> bool:
        """Stage once; never create or publish an active KnowledgeRelease."""
        if any(item.candidate_id == candidate.candidate_id for item in self.records()):
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(candidate.model_dump_json() + "\n")
        return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_asset(path: str | Path | None, filename: str) -> Path | None:
    if path is None:
        return None
    candidate = Path(path).resolve()
    return candidate / filename if candidate.is_dir() else candidate


def _latest_experiences(path: Path) -> list[ExperienceRecord]:
    latest: dict[str, ExperienceRecord] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = ExperienceRecord.model_validate_json(line)
                latest[item.experience_id] = item
    return list(latest.values())


def build_experience_retrieval_assets(
    *,
    eligible_fault_ids: list[str],
    experience_ledger_path: str | Path | None = None,
    memory_store_path: str | Path | None = None,
) -> tuple[list[GovernedTextChunk], ExperienceReuseReceipt, list[ExperienceKnowledgeUpdateCandidate]]:
    """Select only explicitly reviewed, fault-bound experience and memory."""
    eligible = set(eligible_fault_ids)
    chunks: list[GovernedTextChunk] = []
    candidates: list[ExperienceKnowledgeUpdateCandidate] = []
    selected_experience: list[str] = []
    selected_memory: list[str] = []
    rejected = {"not_reuse_approved": 0, "fault_mismatch": 0, "unsafe_memory": 0}
    source_hashes: dict[str, str] = {}

    ledger = _resolve_asset(experience_ledger_path, "experience_ledger.jsonl")
    if ledger is not None and ledger.is_file():
        ledger_hash = _sha256(ledger)
        source_hashes[str(ledger)] = ledger_hash
        for item in _latest_experiences(ledger):
            if item.status is not ExperienceStatus.REUSE_APPROVED:
                rejected["not_reuse_approved"] += 1
                continue
            matched = sorted(set(item.fault_ids) & eligible)
            if not matched:
                rejected["fault_mismatch"] += 1
                continue
            text = "\n".join([
                f"Reviewed diagnostic experience: {item.experience_id}",
                f"Fault scope: {', '.join(matched)}",
                f"Original decision: {item.decision}",
                f"Observed outcome: {item.outcome}",
                f"Reusable lesson: {item.lesson}",
                "Boundary: historical experience is retrieval context, not current-episode evidence.",
            ])
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            chunks.append(GovernedTextChunk(
                chunk_id=f"experience:{item.experience_id}",
                source_id=f"experience:{item.experience_id}",
                source_uri=f"file://{ledger}#{item.experience_id}",
                source_sha256=ledger_hash,
                chunk_sha256=text_hash,
                knowledge_version=item.recorded_at,
                text=text,
                concept_ids=[f"fault:{fault_id}" for fault_id in matched],
                lane=RetrievalLane.EXPERIENCE,
            ))
            selected_experience.append(item.experience_id)
            candidates.append(ExperienceKnowledgeUpdateCandidate(
                candidate_id="EXPKG-" + hashlib.sha256(
                    f"{item.experience_id}:{item.lesson}:{','.join(matched)}".encode("utf-8")
                ).hexdigest()[:20].upper(),
                experience_id=item.experience_id,
                fault_ids=matched,
                proposed_statement=item.lesson,
                source_refs=sorted(set([*item.source_refs, f"file://{ledger}#{item.experience_id}"])),
            ))

    memory = _resolve_asset(memory_store_path, "memory_layers.jsonl")
    if memory is not None and memory.is_file():
        memory_hash = _sha256(memory)
        source_hashes[str(memory)] = memory_hash
        for line in memory.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = MemoryEntry.model_validate_json(line)
            matched = sorted(set(item.fault_ids) & eligible)
            if item.layer not in {MemoryLayer.EPISODIC, MemoryLayer.SEMANTIC} or item.is_negative or item.demoted:
                rejected["unsafe_memory"] += 1
                continue
            if not matched:
                rejected["fault_mismatch"] += 1
                continue
            text = "\n".join([
                f"Human-promoted {item.layer.value.lower()} memory: {item.entry_id}",
                f"Fault scope: {', '.join(matched)}",
                item.content,
                "Boundary: promoted memory is retrieval context, not current-episode evidence.",
            ])
            chunks.append(GovernedTextChunk(
                chunk_id=f"memory:{item.entry_id}",
                source_id=f"memory:{item.entry_id}",
                source_uri=f"file://{memory}#{item.entry_id}",
                source_sha256=memory_hash,
                chunk_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                knowledge_version=item.created_at,
                text=text,
                concept_ids=[f"fault:{fault_id}" for fault_id in matched],
                lane=RetrievalLane.EXPERIENCE,
            ))
            selected_memory.append(item.entry_id)

    receipt = ExperienceReuseReceipt(
        eligible_fault_ids=sorted(eligible),
        selected_experience_ids=sorted(selected_experience),
        selected_memory_entry_ids=sorted(selected_memory),
        rejected_counts=rejected,
        source_sha256=source_hashes,
    )
    return chunks, receipt, candidates


__all__ = [
    "ExperienceKnowledgeUpdateCandidate",
    "ExperienceKnowledgeCandidateInbox",
    "ExperienceReuseReceipt",
    "build_experience_retrieval_assets",
]
