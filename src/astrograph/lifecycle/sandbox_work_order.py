"""Hash-pinned sandbox work orders for governed improvement challengers."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from astrograph.agents.mission.storage import sha
from .knowledge_feedback import KnowledgeFeedbackStore
from .layer_evaluation import REQUIRED_CHECKS

Layer = Literal["planning", "memory", "tool", "knowledge", "model", "execution"]

ALLOWED_CHANGE_KINDS: dict[str, tuple[str, ...]] = {
    "planning": ("PLANNER_POLICY", "PLANNER_PROMPT", "PLAN_CONSTRAINTS"),
    "memory": ("MEMORY_CANDIDATE", "MEMORY_RETRIEVAL_POLICY"),
    "tool": ("TOOL_SELECTION_POLICY", "TOOL_ARGUMENT_POLICY"),
    "knowledge": ("KNOWLEDGE_PROPOSAL", "ONTOLOGY_MAPPING_CANDIDATE"),
    "model": ("MODEL_CHALLENGER_PACKAGE",),
    "execution": ("RUNTIME_POLICY", "TIMEOUT_RETRY_POLICY"),
}

MODEL_REQUIRED_CHECKS = ("TRIGGER_CASE_REPLAY", "FAILURE_REGRESSION", "TASK_METRIC_GATE", "SPLIT_AUDIT")


class BoundArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ref: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)


class ImprovementWorkOrder(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["astrograph-improvement-work-order.v1"] = "astrograph-improvement-work-order.v1"
    work_order_id: str = Field(min_length=1)
    improvement_candidate_id: str = Field(min_length=1)
    feedback_id: str = Field(min_length=1)
    target_layer: Layer
    change_kind: str = Field(min_length=1)
    proposed_change: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)
    baseline_artifacts: list[BoundArtifact] = Field(min_length=1)
    challenger_root: str = Field(min_length=1)
    required_evaluation_checks: list[str] = Field(min_length=1)
    candidate_status: Literal["APPROVED_FOR_OFFLINE_EVALUATION"] = "APPROVED_FOR_OFFLINE_EVALUATION"
    production_write_allowed: Literal[False] = False
    automatic_promotion_allowed: Literal[False] = False
    claim_boundary: str = (
        "This work order authorizes challenger preparation only inside the declared sandbox challenger root. "
        "It does not authorize production writes, permission changes, active-KG writes, confirmed-memory writes, or promotion."
    )

    @field_validator("evidence_refs")
    @classmethod
    def unique_evidence(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("evidence_refs must be unique")
        return value


class ChallengerArtifactSeal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["astrograph-challenger-artifact-seal.v1"] = "astrograph-challenger-artifact-seal.v1"
    seal_id: str = Field(min_length=1)
    work_order_id: str = Field(min_length=1)
    work_order_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    challenger_artifacts: list[BoundArtifact] = Field(min_length=1)
    changed_from_baseline: Literal[True] = True
    ready_for_offline_evaluation: Literal[True] = True
    production_modified: Literal[False] = False
    automatic_promotion_allowed: Literal[False] = False


def _file_sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def _bound(path: Path, ref: str) -> BoundArtifact:
    raw=path.expanduser()
    if raw.is_symlink():
        raise ValueError("bound artifact must be a regular non-symlink file")
    p=raw.resolve(strict=True)
    if not p.is_file():
        raise ValueError("bound artifact must be a regular non-symlink file")
    return BoundArtifact(ref=ref,path=str(p),sha256=_file_sha(p),bytes=p.stat().st_size)


def build_improvement_work_order(
    *,
    store: KnowledgeFeedbackStore,
    improvement_candidate_id: str,
    sandbox_root: str | Path,
    change_kind: str,
    baseline_artifacts: dict[str, str | Path],
) -> tuple[ImprovementWorkOrder, Path, str]:
    candidate=store.export_candidate(improvement_candidate_id)
    if candidate.get("status") != "APPROVED_FOR_OFFLINE_EVALUATION":
        raise PermissionError("improvement candidate is not approved for offline evaluation")
    layer=candidate.get("target_layer")
    if layer not in ALLOWED_CHANGE_KINDS:
        raise ValueError("unsupported improvement target layer")
    if change_kind not in ALLOWED_CHANGE_KINDS[layer]:
        raise ValueError("change kind is not allowed for improvement target layer")
    if not baseline_artifacts:
        raise ValueError("at least one baseline artifact is required")
    bound=[_bound(Path(path),ref) for ref,path in sorted(baseline_artifacts.items())]
    identity={"candidate_id":improvement_candidate_id,"feedback_id":candidate["feedback_id"],"layer":layer,
              "change_kind":change_kind,"baseline":[x.model_dump(mode="json") for x in bound]}
    work_order_id="WORK-"+sha(identity)[:20].upper()
    root=Path(sandbox_root).expanduser().resolve(); root.mkdir(parents=True,exist_ok=True)
    order_root=(root/work_order_id).resolve()
    if not order_root.is_relative_to(root):
        raise PermissionError("work order escaped sandbox root")
    challenger_root=order_root/"challenger"; challenger_root.mkdir(parents=True,exist_ok=True)
    required=list(MODEL_REQUIRED_CHECKS if layer=="model" else REQUIRED_CHECKS[layer])
    order=ImprovementWorkOrder(
        work_order_id=work_order_id,improvement_candidate_id=improvement_candidate_id,feedback_id=candidate["feedback_id"],
        target_layer=layer,change_kind=change_kind,proposed_change=candidate["proposed_change"],
        evidence_refs=list(candidate.get("evidence_refs") or []),baseline_artifacts=bound,
        challenger_root=str(challenger_root),required_evaluation_checks=required,
    )
    path=order_root/"work_order.json"; payload=order.model_dump_json(indent=2)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise ValueError("existing work order content does not match deterministic candidate binding")
    path.write_text(payload,encoding="utf-8")
    return order,path,_file_sha(path)


def seal_sandbox_challenger(
    *,
    work_order_path: str | Path,
    expected_work_order_sha256: str,
    challenger_artifacts: dict[str, str | Path],
) -> tuple[ChallengerArtifactSeal, Path, str]:
    order_path=Path(work_order_path).expanduser().resolve(strict=True)
    if _file_sha(order_path) != expected_work_order_sha256:
        raise ValueError("work order SHA-256 mismatch")
    order=ImprovementWorkOrder.model_validate_json(order_path.read_text(encoding="utf-8"))
    root=Path(order.challenger_root).expanduser().resolve(strict=True)
    if not challenger_artifacts:
        raise ValueError("at least one challenger artifact is required")
    bound=[]
    for ref,path in sorted(challenger_artifacts.items()):
        raw=Path(path).expanduser()
        if raw.is_symlink():
            raise ValueError("bound artifact must be a regular non-symlink file")
        p=raw.resolve(strict=True)
        if not p.is_relative_to(root):
            raise PermissionError("challenger artifact is outside work-order sandbox root")
        bound.append(_bound(raw,ref))
    baseline_hashes={x.sha256 for x in order.baseline_artifacts}
    if all(x.sha256 in baseline_hashes for x in bound):
        raise ValueError("challenger artifacts do not contain a content change from baseline")
    identity={"work_order_sha256":expected_work_order_sha256,"challengers":[x.model_dump(mode="json") for x in bound]}
    seal=ChallengerArtifactSeal(
        seal_id="SEAL-"+sha(identity)[:20].upper(),work_order_id=order.work_order_id,
        work_order_sha256=expected_work_order_sha256,challenger_artifacts=bound,
    )
    seal_path=order_path.parent/"challenger_seal.json"; payload=seal.model_dump_json(indent=2)
    if seal_path.exists() and seal_path.read_text(encoding="utf-8") != payload:
        raise ValueError("existing challenger seal differs from current sealed artifacts")
    seal_path.write_text(payload,encoding="utf-8")
    return seal,seal_path,_file_sha(seal_path)


__all__=["ALLOWED_CHANGE_KINDS","MODEL_REQUIRED_CHECKS","BoundArtifact","ImprovementWorkOrder","ChallengerArtifactSeal",
         "build_improvement_work_order","seal_sandbox_challenger"]
