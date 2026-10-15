from __future__ import annotations

import numpy as np
import pytest

from astrograph.evaluation.data_exposure import dataset_use
from astrograph.evaluation.esa_r26_frozen import (
    EXPECTED_SELECTED_CONFIG,
    build_r26_esa_holdout_binding,
    build_r26_frozen_esa_detector_artifact,
    run_r26_frozen_esa_prediction,
    verify_r26_frozen_esa_detector_artifact,
)
from astrograph.evaluation.external_dataset_contract import seal


def test_r26_esa_detector_is_exactly_frozen_from_r25_selected_candidate():
    artifact = build_r26_frozen_esa_detector_artifact()
    verify_r26_frozen_esa_detector_artifact(artifact)
    assert artifact["config"] == EXPECTED_SELECTED_CONFIG
    assert artifact["source_r25_development_receipt_sha256"] == "f3267ca7de06ed1a96b364991c07cd725a7dc23bf468ee9249f6ccae2959ab55"
    assert artifact["formal_evidence_eligible"] is False


def test_r26_esa_prediction_receipt_is_truth_free_and_content_bound():
    artifact = build_r26_frozen_esa_detector_artifact()
    times = np.arange(5, dtype=np.int64) * 1_000_000_000
    channels = {f"channel_{i}": np.zeros(5, dtype=float) for i in range(41, 47)}
    channels["channel_41"][2] = 1.0
    channels["channel_42"][2] = 1.0
    with dataset_use("SYNTHETIC_PROTOCOL", "DEVELOPMENT_SELECTION", "OPTIMIZE", group_ids=["r26-esa"]):
        receipt = run_r26_frozen_esa_prediction(
            detector_artifact=artifact,
            raw_group_id="g1",
            time_ns=times,
            channel_values=channels,
        )
    assert receipt["truth_joined"] is False
    assert receipt["alarm_point_count"] >= 1
    assert receipt["alarm_group_count"] >= 1
    assert receipt["formal_evidence_generated"] is False


def test_r26_esa_holdout_binding_reuses_first_contact_identity_without_granting_validation():
    artifact = build_r26_frozen_esa_detector_artifact()
    intake = seal({
        "schema_version": "astrograph-r22-new-holdout-intake-receipt.v1",
        "status": "INTAKE_FROZEN_PRE_TRUTH",
        "dataset_id": "ESA-NEW-1",
        "dataset_family": "ESA",
        "source_artifact_sha256": "a" * 64,
    })
    groups = seal({
        "schema_version": "astrograph-r22-new-holdout-group-admission.v1",
        "status": "ADMITTED_IDENTITY_NOVELTY_NOT_INDEPENDENCE_PROOF",
        "dataset_id": "ESA-NEW-1",
        "dataset_family": "ESA",
        "intake_receipt_sha256": intake["receipt_sha256"],
        "source_artifact_sha256": "a" * 64,
        "canonical_group_identity_sha256": "b" * 64,
        "group_count": 3,
    })
    binding = build_r26_esa_holdout_binding(intake, groups, artifact)
    assert binding["truth_joined"] is False
    assert binding["external_validation_claimed"] is False
    assert binding["r26_detector_identity_sha256"] == artifact["detector_identity_sha256"]


def test_r26_esa_artifact_tamper_fails_closed():
    artifact = build_r26_frozen_esa_detector_artifact()
    artifact = dict(artifact)
    artifact["config"] = dict(artifact["config"])
    artifact["config"]["robust_z_threshold"] = 6.0
    with pytest.raises(ValueError):
        verify_r26_frozen_esa_detector_artifact(artifact)
