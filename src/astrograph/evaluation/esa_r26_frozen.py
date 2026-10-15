"""Frozen R26 high-resolution ESA detector and future-holdout binding.

R25 selected one development configuration on the raw lightweight-channel grid.
R26 does not retune it.  This module content-binds that R25 reference/config,
produces truth-free prediction receipts, and can bind an R22-style first-contact
holdout admission to the R26 detector before truth is joined.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .esa_high_resolution import (
    ESAHighResolutionConfig,
    ESAHighResolutionReference,
    group_alarm_intervals,
    score_high_resolution_transients,
)
from .external_dataset_contract import digest, file_identity, seal, verify_seal

SOURCE_ROOT = Path(__file__).resolve().parents[3]
R25_RECEIPT = SOURCE_ROOT / "governance/evidence/research/r25_remediation/esa_high_resolution_development.json"
EXPECTED_R25_RECEIPT_SHA256 = "f3267ca7de06ed1a96b364991c07cd725a7dc23bf468ee9249f6ccae2959ab55"
EXPECTED_SELECTED_CONFIG = {
    "robust_z_threshold": 8.0,
    "minimum_coincident_channels": 2,
    "alarm_group_gap_s": 40.0,
}


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_r26_frozen_esa_detector_artifact() -> dict[str, Any]:
    receipt = json.loads(R25_RECEIPT.read_text(encoding="utf-8"))
    verify_seal(receipt)
    if receipt.get("receipt_sha256") != EXPECTED_R25_RECEIPT_SHA256:
        raise ValueError("R26_ESA_R25_DEVELOPMENT_RECEIPT_IDENTITY_DRIFT")
    selected = receipt.get("selected_candidate") or {}
    actual_config = {
        "robust_z_threshold": float(selected.get("robust_z_threshold")),
        "minimum_coincident_channels": int(selected.get("minimum_coincident_channels")),
        "alarm_group_gap_s": float(selected.get("alarm_group_gap_s")),
    }
    if actual_config != EXPECTED_SELECTED_CONFIG:
        raise ValueError("R26_ESA_SELECTED_CONFIG_DRIFT")
    reference = ESAHighResolutionReference.model_validate(receipt["reference"])
    source_files = []
    for relative in (
        "src/astrograph/evaluation/esa_high_resolution.py",
        "src/astrograph/evaluation/esa_r26_frozen.py",
    ):
        path = SOURCE_ROOT / relative
        source_files.append({"relative_path": relative, **file_identity(path)})
    return seal({
        "schema_version": "astrograph-r26-esa-frozen-detector-artifact.v1",
        "status": "FROZEN_FROM_R25_DEVELOPMENT_BEFORE_NEW_HOLDOUT",
        "source_r25_development_receipt_sha256": receipt["receipt_sha256"],
        "reference": reference.model_dump(mode="json"),
        "config": actual_config,
        "implementation_sources": source_files,
        "detector_identity_sha256": digest({
            "source_r25_development_receipt_sha256": receipt["receipt_sha256"],
            "reference_sha256": reference.reference_sha256,
            "config": actual_config,
            "implementation_sources": source_files,
        }),
        "score_semantics": "ROBUST_LAG_ONE_DELTA_Z_NOT_PROBABILITY",
        "formal_evidence_eligible": False,
        "qualification": "FROZEN_DEVELOPMENT_CANDIDATE_NOT_EXTERNAL_VALIDATION_NOT_FLIGHT_QUALIFIED",
        "claim_boundary": (
            "Configuration/reference are frozen from R25 development evidence. This artifact may be transported to a "
            "genuinely new admitted holdout but does not itself establish external validity."
        ),
    })


def verify_r26_frozen_esa_detector_artifact(artifact: Mapping[str, Any]) -> None:
    row = dict(artifact)
    verify_seal(row)
    if row.get("schema_version") != "astrograph-r26-esa-frozen-detector-artifact.v1":
        raise ValueError("R26_ESA_FROZEN_ARTIFACT_SCHEMA_INVALID")
    reference = ESAHighResolutionReference.model_validate(row["reference"])
    config = row.get("config") or {}
    if config != EXPECTED_SELECTED_CONFIG:
        raise ValueError("R26_ESA_FROZEN_CONFIG_DRIFT")
    for source in row.get("implementation_sources") or []:
        relative = str(source.get("relative_path") or "")
        path = SOURCE_ROOT / relative
        if not path.is_file() or path.is_symlink() or file_identity(path) != {"size": source.get("size"), "sha256": source.get("sha256")}:
            raise ValueError("R26_ESA_IMPLEMENTATION_SOURCE_DRIFT:" + relative)
    expected = digest({
        "source_r25_development_receipt_sha256": row["source_r25_development_receipt_sha256"],
        "reference_sha256": reference.reference_sha256,
        "config": config,
        "implementation_sources": row["implementation_sources"],
    })
    if expected != row.get("detector_identity_sha256"):
        raise ValueError("R26_ESA_DETECTOR_IDENTITY_DRIFT")


def build_r26_esa_holdout_binding(
    r22_intake_receipt: Mapping[str, Any],
    r22_group_receipt: Mapping[str, Any],
    detector_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    intake = dict(r22_intake_receipt)
    groups = dict(r22_group_receipt)
    verify_seal(intake)
    verify_seal(groups)
    verify_r26_frozen_esa_detector_artifact(detector_artifact)
    if intake.get("dataset_family") != "ESA" or groups.get("dataset_family") != "ESA":
        raise ValueError("R26_ESA_HOLDOUT_FAMILY_MISMATCH")
    if groups.get("intake_receipt_sha256") != intake.get("receipt_sha256"):
        raise ValueError("R26_ESA_HOLDOUT_INTAKE_GROUP_BINDING_DRIFT")
    if groups.get("source_artifact_sha256") != intake.get("source_artifact_sha256"):
        raise ValueError("R26_ESA_HOLDOUT_SOURCE_IDENTITY_DRIFT")
    return seal({
        "schema_version": "astrograph-r26-esa-new-holdout-binding.v1",
        "status": "R26_FROZEN_DETECTOR_BOUND_PRE_TRUTH",
        "dataset_id": intake["dataset_id"],
        "source_artifact_sha256": intake["source_artifact_sha256"],
        "r22_intake_receipt_sha256": intake["receipt_sha256"],
        "r22_group_receipt_sha256": groups["receipt_sha256"],
        "canonical_group_identity_sha256": groups["canonical_group_identity_sha256"],
        "group_count": groups["group_count"],
        "r26_detector_receipt_sha256": detector_artifact["receipt_sha256"],
        "r26_detector_identity_sha256": detector_artifact["detector_identity_sha256"],
        "truth_joined": False,
        "external_validation_claimed": False,
    })


def run_r26_frozen_esa_prediction(
    *,
    detector_artifact: Mapping[str, Any],
    raw_group_id: str,
    time_ns: Sequence[int],
    channel_values: Mapping[str, Sequence[float]],
    holdout_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from .data_exposure import guard_prediction_generation

    guard_prediction_generation()
    verify_r26_frozen_esa_detector_artifact(detector_artifact)
    if not str(raw_group_id).strip():
        raise ValueError("R26_ESA_RAW_GROUP_ID_REQUIRED")
    if holdout_binding is not None:
        binding = dict(holdout_binding)
        verify_seal(binding)
        if binding.get("schema_version") != "astrograph-r26-esa-new-holdout-binding.v1":
            raise ValueError("R26_ESA_HOLDOUT_BINDING_SCHEMA_INVALID")
        if binding.get("r26_detector_receipt_sha256") != detector_artifact.get("receipt_sha256"):
            raise ValueError("R26_ESA_HOLDOUT_DETECTOR_BINDING_DRIFT")
    reference = ESAHighResolutionReference.model_validate(detector_artifact["reference"])
    config = ESAHighResolutionConfig(**detector_artifact["config"])
    score = score_high_resolution_transients(
        reference=reference,
        time_ns=time_ns,
        channel_values=channel_values,
        config=config,
    )
    groups = group_alarm_intervals(score, gap_s=config.alarm_group_gap_s)
    times = np.asarray(time_ns, dtype=np.int64)
    values_identity = {
        channel: digest([float(value) for value in channel_values[channel]])
        for channel in sorted(channel_values)
    }
    return seal({
        "schema_version": "astrograph-r26-esa-frozen-prediction-receipt.v1",
        "status": "PREDICTION_FROZEN_PRE_TRUTH",
        "raw_group_id": str(raw_group_id),
        "r26_detector_receipt_sha256": detector_artifact["receipt_sha256"],
        "r26_detector_identity_sha256": detector_artifact["detector_identity_sha256"],
        "holdout_binding_receipt_sha256": None if holdout_binding is None else holdout_binding["receipt_sha256"],
        "time_ns_sha256": digest([int(value) for value in times]),
        "channel_value_sha256": values_identity,
        "sample_count": int(times.size),
        "score_sha256": score.score_sha256,
        "alarm_point_count": int(sum(score.alarm)),
        "alarm_groups": [list(item) for item in groups],
        "alarm_group_count": len(groups),
        "truth_joined": False,
        "score_semantics": score.score_semantics,
        "formal_evidence_generated": False,
    })


__all__ = [
    "EXPECTED_R25_RECEIPT_SHA256",
    "EXPECTED_SELECTED_CONFIG",
    "build_r26_esa_holdout_binding",
    "build_r26_frozen_esa_detector_artifact",
    "run_r26_frozen_esa_prediction",
    "verify_r26_frozen_esa_detector_artifact",
]
