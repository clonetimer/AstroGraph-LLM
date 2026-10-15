from __future__ import annotations

import copy

import pytest

from astrograph.governance.supplemental_provenance import (
    build_supplemental_input_provenance_receipt,
    verify_supplemental_input_provenance_receipt,
)


def _receipt():
    return build_supplemental_input_provenance_receipt(
        source_job_id="JOB-" + "A" * 24,
        source_report={"schema_version": "report.v1", "value": 1},
        original_evidence_bundle={"schema_version": "astrograph-episode-evidence-bundle.v1", "value": 2},
        contract={"contract_id": "CONTRACT:1", "required_unit": "A"},
        capsule={
            "contract_id": "CONTRACT:1",
            "resolved_source_field": "adcs.rw.motor_current_a_2",
            "unit": "A",
            "window_start_s": 0.0,
            "window_end_s": 10.0,
            "sample_times_s": [0.0, 5.0, 10.0],
            "current_values": [1.0, 2.0, 3.0],
            "baseline_values": None,
        },
        admission_receipt_sha256="1" * 64,
        parent_admission_receipt_sha256s=["2" * 64],
        actor="tester",
        admitted_at="2026-09-18T00:00:00+00:00",
    )


def test_r11_supplemental_provenance_receipt_is_content_bound():
    receipt = _receipt()
    assert receipt.formal_release_eligible is False
    assert receipt.provenance_id.startswith("SUPP-PROV-")
    verify_supplemental_input_provenance_receipt(receipt.model_dump(mode="json"))

    tampered = copy.deepcopy(receipt.model_dump(mode="json"))
    tampered["current_values_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="receipt_sha256 mismatch"):
        verify_supplemental_input_provenance_receipt(tampered)
