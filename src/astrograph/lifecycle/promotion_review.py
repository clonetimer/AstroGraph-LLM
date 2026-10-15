"""Content-addressed promotion-review packets for governed evolution candidates.

This module deliberately stops before production promotion.  It binds the exact
human-review-eligible evaluator result to the sealed challenger lineage and the
evaluation-run receipt so that later review cannot silently substitute evidence.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.agents.mission.storage import sha
from .evolution_dispatcher import GovernedEvolutionReviewPacket

Layer = Literal["planning", "memory", "tool", "knowledge", "model", "execution"]


class PromotionReviewPacket(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-promotion-review-packet.v1"] = "astrograph-promotion-review-packet.v1"
    packet_id: str = Field(min_length=1)
    improvement_candidate_id: str = Field(min_length=1)
    feedback_id: str = Field(min_length=1)
    target_layer: Layer
    governed_review_packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_run_id: str = Field(min_length=1)
    evaluation_run_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    work_order_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    challenger_seal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    upgrade_proposal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    upgrade_proposal: dict[str, Any]
    review_state: Literal["PENDING_GOVERNED_PROMOTION_REVIEW"] = "PENDING_GOVERNED_PROMOTION_REVIEW"
    requires_human_gate: Literal[True] = True
    automatic_production_promotion: Literal[False] = False
    production_modified: Literal[False] = False
    claim_boundary: str = (
        "This packet is immutable evidence for governed promotion review only. "
        "It cannot deploy, activate, or write any production model, planner, memory, tool, knowledge, or runtime change."
    )

    @model_validator(mode="after")
    def packet_identity_matches(self):
        identity = {
            "improvement_candidate_id": self.improvement_candidate_id,
            "feedback_id": self.feedback_id,
            "target_layer": self.target_layer,
            "governed_review_packet_sha256": self.governed_review_packet_sha256,
            "evaluation_run_id": self.evaluation_run_id,
            "evaluation_run_receipt_sha256": self.evaluation_run_receipt_sha256,
            "work_order_sha256": self.work_order_sha256,
            "challenger_seal_sha256": self.challenger_seal_sha256,
            "evaluator_result_sha256": self.evaluator_result_sha256,
            "upgrade_proposal_sha256": self.upgrade_proposal_sha256,
        }
        expected = "PRPK-" + sha(identity)[:20].upper()
        if self.packet_id != expected:
            raise ValueError("promotion review packet_id does not match bound identity")
        if sha(self.upgrade_proposal) != self.upgrade_proposal_sha256:
            raise ValueError("promotion review upgrade proposal hash mismatch")
        return self


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_promotion_review_packet(
    *,
    review: GovernedEvolutionReviewPacket,
    expected_evaluation_run_receipt_sha256: str,
) -> PromotionReviewPacket:
    if review.review_status != "ELIGIBLE_FOR_HUMAN_REVIEW":
        raise PermissionError("only a human-review-eligible evaluator result can enter promotion review")
    if not review.upgrade_proposal:
        raise ValueError("promotion review requires an upgrade proposal")
    result = review.evaluator_result
    run_id = str(result.get("evaluation_run_id") or "")
    receipt_sha = str(result.get("evaluation_run_receipt_sha256") or "")
    work_order_sha = str(result.get("work_order_sha256") or "")
    seal_sha = str(result.get("challenger_seal_sha256") or "")
    if not run_id:
        raise ValueError("evaluator result is missing evaluation_run_id")
    if receipt_sha != expected_evaluation_run_receipt_sha256:
        raise ValueError("evaluator result evaluation-run receipt binding mismatch")
    for value, label in ((work_order_sha, "work order"), (seal_sha, "challenger seal"), (receipt_sha, "evaluation receipt")):
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError(f"evaluator result has invalid {label} SHA-256")
    review_sha = sha(review.model_dump(mode="json"))
    proposal_sha = sha(review.upgrade_proposal)
    identity = {
        "improvement_candidate_id": review.improvement_candidate_id,
        "feedback_id": review.feedback_id,
        "target_layer": review.target_layer,
        "governed_review_packet_sha256": review_sha,
        "evaluation_run_id": run_id,
        "evaluation_run_receipt_sha256": receipt_sha,
        "work_order_sha256": work_order_sha,
        "challenger_seal_sha256": seal_sha,
        "evaluator_result_sha256": review.evaluator_result_sha256,
        "upgrade_proposal_sha256": proposal_sha,
    }
    return PromotionReviewPacket(
        packet_id="PRPK-" + sha(identity)[:20].upper(),
        improvement_candidate_id=review.improvement_candidate_id,
        feedback_id=review.feedback_id,
        target_layer=review.target_layer,
        governed_review_packet_sha256=review_sha,
        evaluation_run_id=run_id,
        evaluation_run_receipt_sha256=receipt_sha,
        work_order_sha256=work_order_sha,
        challenger_seal_sha256=seal_sha,
        evaluator_result_sha256=review.evaluator_result_sha256,
        upgrade_proposal_sha256=proposal_sha,
        upgrade_proposal=review.upgrade_proposal,
    )


def write_promotion_review_packet(
    *, root: str | Path, packet: PromotionReviewPacket
) -> tuple[Path, str]:
    rootp = Path(root).expanduser().resolve()
    rootp.mkdir(parents=True, exist_ok=True)
    path = rootp / f"{packet.packet_id}.json"
    payload = packet.model_dump_json(indent=2)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise ValueError("existing promotion review packet differs for deterministic identity")
    path.write_text(payload, encoding="utf-8")
    return path, _file_sha(path)


def load_and_validate_promotion_review_packet(
    *,
    path: str | Path,
    expected_sha256: str,
    expected_improvement_candidate_id: str,
    expected_feedback_id: str,
    expected_target_layer: Layer,
    expected_governed_review_packet_sha256: str,
    expected_evaluation_run_receipt_sha256: str,
) -> PromotionReviewPacket:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise ValueError("promotion review packet must not be a symlink")
    resolved = raw.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("promotion review packet must be a regular file")
    if _file_sha(resolved) != expected_sha256:
        raise ValueError("promotion review packet SHA-256 mismatch")
    packet = PromotionReviewPacket.model_validate_json(resolved.read_text(encoding="utf-8"))
    expected = {
        "improvement_candidate_id": expected_improvement_candidate_id,
        "feedback_id": expected_feedback_id,
        "target_layer": expected_target_layer,
        "governed_review_packet_sha256": expected_governed_review_packet_sha256,
        "evaluation_run_receipt_sha256": expected_evaluation_run_receipt_sha256,
    }
    for field, value in expected.items():
        if getattr(packet, field) != value:
            raise ValueError(f"promotion review packet {field} binding mismatch")
    return packet


__all__ = [
    "PromotionReviewPacket",
    "build_promotion_review_packet",
    "write_promotion_review_packet",
    "load_and_validate_promotion_review_packet",
]
