"""Promotion transaction preflight and rollback coverage without applying production writes."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.agents.mission.storage import sha
from .promotion_review import PromotionReviewPacket, load_and_validate_promotion_review_packet

Layer = Literal["planning", "memory", "tool", "knowledge", "model", "execution"]


class ProductionBaselineArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ref: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PromotionWriteTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ref: str = Field(min_length=1)
    path: str = Field(min_length=1)
    baseline_ref: str = Field(min_length=1)
    rollback_action: Literal["RESTORE_BASELINE"] = "RESTORE_BASELINE"


class PromotionTransactionPreflight(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["astrograph-promotion-transaction-preflight.v1"] = "astrograph-promotion-transaction-preflight.v1"
    transaction_id: str = Field(min_length=1)
    promotion_review_packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    promotion_review_packet_id: str = Field(min_length=1)
    improvement_candidate_id: str = Field(min_length=1)
    feedback_id: str = Field(min_length=1)
    target_layer: Layer
    production_root: str = Field(min_length=1)
    baseline_artifacts: tuple[ProductionBaselineArtifact, ...] = Field(min_length=1)
    write_targets: tuple[PromotionWriteTarget, ...] = Field(min_length=1)
    rollback_baseline_refs: tuple[str, ...]
    preflight_status: Literal["READY_FOR_GOVERNED_TRANSACTION_REVIEW"] = "READY_FOR_GOVERNED_TRANSACTION_REVIEW"
    writes_applied: Literal[False] = False
    production_modified: Literal[False] = False
    automatic_production_promotion: Literal[False] = False
    requires_human_gate: Literal[True] = True
    claim_boundary: str = (
        "Preflight only. The current production baseline and full rollback coverage are frozen before any governed promotion. "
        "No production file is modified by this artifact."
    )

    @model_validator(mode="after")
    def identity_and_rollback_coverage(self):
        refs = [x.ref for x in self.baseline_artifacts]
        if len(refs) != len(set(refs)):
            raise ValueError("baseline artifact refs must be unique")
        targets = [x.ref for x in self.write_targets]
        if len(targets) != len(set(targets)):
            raise ValueError("write target refs must be unique")
        baseline_set = set(refs)
        required = {x.baseline_ref for x in self.write_targets}
        if required != set(self.rollback_baseline_refs):
            raise ValueError("rollback coverage must exactly match write-target baseline refs")
        if not required.issubset(baseline_set):
            raise ValueError("write target references unknown baseline artifact")
        identity = {
            "promotion_review_packet_sha256": self.promotion_review_packet_sha256,
            "promotion_review_packet_id": self.promotion_review_packet_id,
            "improvement_candidate_id": self.improvement_candidate_id,
            "feedback_id": self.feedback_id,
            "target_layer": self.target_layer,
            "production_root": self.production_root,
            "baseline_artifacts": [x.model_dump(mode="json") for x in self.baseline_artifacts],
            "write_targets": [x.model_dump(mode="json") for x in self.write_targets],
            "rollback_baseline_refs": list(self.rollback_baseline_refs),
        }
        expected = "PTXN-" + sha(identity)[:20].upper()
        if self.transaction_id != expected:
            raise ValueError("promotion transaction_id does not match frozen preflight identity")
        return self


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _resolve_regular_under(path: str | Path, root: Path, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    resolved = raw.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file")
    if not resolved.is_relative_to(root):
        raise PermissionError(f"{label} escaped approved production root")
    return resolved


def build_promotion_transaction_preflight(
    *,
    promotion_review_packet_path: str | Path,
    expected_promotion_review_packet_sha256: str,
    production_root: str | Path,
    write_targets: dict[str, str | Path],
) -> PromotionTransactionPreflight:
    raw_packet = Path(promotion_review_packet_path).expanduser()
    if raw_packet.is_symlink():
        raise ValueError("promotion review packet must not be a symlink")
    packet_path = raw_packet.resolve(strict=True)
    if _file_sha(packet_path) != expected_promotion_review_packet_sha256:
        raise ValueError("promotion review packet SHA-256 mismatch")
    packet = PromotionReviewPacket.model_validate_json(packet_path.read_text(encoding="utf-8"))
    # Reuse the packet validator's exact identity, then bind the file SHA.
    load_and_validate_promotion_review_packet(
        path=packet_path,
        expected_sha256=expected_promotion_review_packet_sha256,
        expected_improvement_candidate_id=packet.improvement_candidate_id,
        expected_feedback_id=packet.feedback_id,
        expected_target_layer=packet.target_layer,
        expected_governed_review_packet_sha256=packet.governed_review_packet_sha256,
        expected_evaluation_run_receipt_sha256=packet.evaluation_run_receipt_sha256,
    )
    root = Path(production_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("production root must be a directory")
    if not write_targets:
        raise ValueError("promotion preflight requires at least one production write target")
    baselines = []
    writes = []
    for ref, path in sorted(write_targets.items()):
        current = _resolve_regular_under(path, root, f"production target {ref}")
        baseline_ref = f"baseline:{ref}"
        baselines.append(ProductionBaselineArtifact(ref=baseline_ref, path=str(current), sha256=_file_sha(current)))
        writes.append(PromotionWriteTarget(ref=ref, path=str(current), baseline_ref=baseline_ref))
    rollback_refs = tuple(x.ref for x in baselines)
    identity = {
        "promotion_review_packet_sha256": expected_promotion_review_packet_sha256,
        "promotion_review_packet_id": packet.packet_id,
        "improvement_candidate_id": packet.improvement_candidate_id,
        "feedback_id": packet.feedback_id,
        "target_layer": packet.target_layer,
        "production_root": str(root),
        "baseline_artifacts": [x.model_dump(mode="json") for x in baselines],
        "write_targets": [x.model_dump(mode="json") for x in writes],
        "rollback_baseline_refs": list(rollback_refs),
    }
    return PromotionTransactionPreflight(
        transaction_id="PTXN-" + sha(identity)[:20].upper(),
        promotion_review_packet_sha256=expected_promotion_review_packet_sha256,
        promotion_review_packet_id=packet.packet_id,
        improvement_candidate_id=packet.improvement_candidate_id,
        feedback_id=packet.feedback_id,
        target_layer=packet.target_layer,
        production_root=str(root),
        baseline_artifacts=tuple(baselines),
        write_targets=tuple(writes),
        rollback_baseline_refs=rollback_refs,
    )


def validate_promotion_transaction_preflight(preflight: PromotionTransactionPreflight) -> None:
    root = Path(preflight.production_root).resolve(strict=True)
    baseline_by_ref = {x.ref: x for x in preflight.baseline_artifacts}
    for target in preflight.write_targets:
        baseline = baseline_by_ref[target.baseline_ref]
        current = _resolve_regular_under(target.path, root, f"production target {target.ref}")
        if str(current) != baseline.path:
            raise ValueError("promotion target path no longer matches frozen baseline")
        if _file_sha(current) != baseline.sha256:
            raise ValueError("production baseline changed after promotion preflight")


def write_promotion_transaction_preflight(*, root: str | Path, preflight: PromotionTransactionPreflight) -> tuple[Path, str]:
    rootp = Path(root).expanduser().resolve()
    rootp.mkdir(parents=True, exist_ok=True)
    path = rootp / f"{preflight.transaction_id}.json"
    payload = preflight.model_dump_json(indent=2)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise ValueError("existing promotion transaction preflight differs for deterministic identity")
    path.write_text(payload, encoding="utf-8")
    return path, _file_sha(path)


def load_and_validate_promotion_transaction_preflight(*, path: str | Path, expected_sha256: str) -> PromotionTransactionPreflight:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise ValueError("promotion transaction preflight must not be a symlink")
    resolved = raw.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("promotion transaction preflight must be a regular file")
    if _file_sha(resolved) != expected_sha256:
        raise ValueError("promotion transaction preflight SHA-256 mismatch")
    preflight = PromotionTransactionPreflight.model_validate_json(resolved.read_text(encoding="utf-8"))
    validate_promotion_transaction_preflight(preflight)
    return preflight


__all__ = [
    "ProductionBaselineArtifact", "PromotionWriteTarget", "PromotionTransactionPreflight",
    "build_promotion_transaction_preflight", "validate_promotion_transaction_preflight",
    "write_promotion_transaction_preflight", "load_and_validate_promotion_transaction_preflight",
]
