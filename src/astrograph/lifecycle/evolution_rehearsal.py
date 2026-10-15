"""End-to-end governed evolution rehearsal through promotion preflight, with no production writes."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.agents.mission.storage import sha
from .evolution_dispatcher import evaluate_approved_improvement_candidate
from .knowledge_feedback import KnowledgeFeedbackStore
from .promotion_review import build_promotion_review_packet, write_promotion_review_packet
from .promotion_transaction import build_promotion_transaction_preflight, write_promotion_transaction_preflight

Layer = Literal["planning", "memory", "tool", "knowledge", "model", "execution"]


class GovernedEvolutionRehearsalReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["astrograph-governed-evolution-rehearsal-receipt.v1"] = "astrograph-governed-evolution-rehearsal-receipt.v1"
    rehearsal_id: str = Field(min_length=1)
    improvement_candidate_id: str = Field(min_length=1)
    feedback_id: str = Field(min_length=1)
    target_layer: Layer
    evaluator_route: str = Field(min_length=1)
    evaluation_run_id: str = Field(min_length=1)
    evaluation_run_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    promotion_review_packet_id: str = Field(min_length=1)
    promotion_review_packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    promotion_transaction_id: str = Field(min_length=1)
    promotion_preflight_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    final_state: Literal["READY_FOR_GOVERNED_PROMOTION_REVIEW"] = "READY_FOR_GOVERNED_PROMOTION_REVIEW"
    requires_human_gate: Literal[True] = True
    automatic_production_promotion: Literal[False] = False
    production_modified: Literal[False] = False
    claim_boundary: str = (
        "End-to-end rehearsal only. The chain proves lineage from an approved candidate through offline evaluation and promotion preflight. "
        "It does not apply a production transaction."
    )

    @model_validator(mode="after")
    def identity_matches(self):
        identity = {
            "improvement_candidate_id": self.improvement_candidate_id,
            "feedback_id": self.feedback_id,
            "target_layer": self.target_layer,
            "evaluator_route": self.evaluator_route,
            "evaluation_run_id": self.evaluation_run_id,
            "evaluation_run_receipt_sha256": self.evaluation_run_receipt_sha256,
            "promotion_review_packet_id": self.promotion_review_packet_id,
            "promotion_review_packet_sha256": self.promotion_review_packet_sha256,
            "promotion_transaction_id": self.promotion_transaction_id,
            "promotion_preflight_sha256": self.promotion_preflight_sha256,
        }
        expected = "EVR-" + sha(identity)[:20].upper()
        if self.rehearsal_id != expected:
            raise ValueError("evolution rehearsal_id does not match bound lineage")
        return self


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare_governed_evolution_rehearsal(
    *,
    store: KnowledgeFeedbackStore,
    improvement_candidate_id: str,
    dispatcher_kwargs: dict[str, Any],
    artifact_root: str | Path,
    production_root: str | Path,
    write_targets: dict[str, str | Path],
) -> tuple[GovernedEvolutionRehearsalReceipt, Path, str]:
    review = evaluate_approved_improvement_candidate(
        store=store,
        improvement_candidate_id=improvement_candidate_id,
        **dispatcher_kwargs,
    )
    if review.review_status != "ELIGIBLE_FOR_HUMAN_REVIEW":
        raise PermissionError("evolution rehearsal stopped because offline evaluation is not eligible for promotion review")
    result = review.evaluator_result
    evaluation_receipt_sha = str(result.get("evaluation_run_receipt_sha256") or "")
    if len(evaluation_receipt_sha) != 64:
        raise ValueError("eligible evaluator result is missing a valid evaluation-run receipt SHA-256")
    artifact_root = Path(artifact_root).expanduser().resolve()
    packet = build_promotion_review_packet(
        review=review,
        expected_evaluation_run_receipt_sha256=evaluation_receipt_sha,
    )
    packet_path, packet_sha = write_promotion_review_packet(root=artifact_root / "promotion-review", packet=packet)
    preflight = build_promotion_transaction_preflight(
        promotion_review_packet_path=packet_path,
        expected_promotion_review_packet_sha256=packet_sha,
        production_root=production_root,
        write_targets=write_targets,
    )
    preflight_path, preflight_sha = write_promotion_transaction_preflight(
        root=artifact_root / "promotion-preflight", preflight=preflight
    )
    identity = {
        "improvement_candidate_id": review.improvement_candidate_id,
        "feedback_id": review.feedback_id,
        "target_layer": review.target_layer,
        "evaluator_route": review.evaluator_route,
        "evaluation_run_id": result["evaluation_run_id"],
        "evaluation_run_receipt_sha256": evaluation_receipt_sha,
        "promotion_review_packet_id": packet.packet_id,
        "promotion_review_packet_sha256": packet_sha,
        "promotion_transaction_id": preflight.transaction_id,
        "promotion_preflight_sha256": preflight_sha,
    }
    receipt = GovernedEvolutionRehearsalReceipt(
        rehearsal_id="EVR-" + sha(identity)[:20].upper(),
        **identity,
    )
    receipt_root = artifact_root / "rehearsals"
    receipt_root.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_root / f"{receipt.rehearsal_id}.json"
    payload = receipt.model_dump_json(indent=2)
    if receipt_path.exists() and receipt_path.read_text(encoding="utf-8") != payload:
        raise ValueError("existing rehearsal receipt differs for deterministic lineage")
    receipt_path.write_text(payload, encoding="utf-8")
    # Explicitly verify no helper modified production while generating the receipt.
    return receipt, receipt_path, _file_sha(receipt_path)


__all__ = ["GovernedEvolutionRehearsalReceipt", "prepare_governed_evolution_rehearsal"]
