from __future__ import annotations

import importlib.util
import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/acceptance/run_candidate2_authorization_gate.py"
    spec = importlib.util.spec_from_file_location("candidate2_authorization_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate2_authorization_preserves_timer_attestation_and_passes() -> None:
    report = _module().evaluate(ROOT)
    assert report["technical_status"] == "PASS"
    assert report["status"] == "PASS"
    assert report["checks"]["all_bound_artifact_hashes_match"] is True
    assert report["checks"]["approval_present"] is True
    assert report["checks"]["source_attestation_hash_matches"] is True
    assert report["checks"]["source_attestation_binds_preapproval_request"] is True
    assert report["checks"]["approval_attestation_is_verbatim_source"] is True
    assert report["checks"]["approval_declarations_cover_frozen_request"] is True
    request = json.loads(
        (
            ROOT
            / "protocols/acceptance/p015_candidate2_authorization_request.json"
        ).read_text(encoding="utf-8")
    )
    assert request["approval"]["status"] == "APPROVED"
    assert request["candidate"]["new_single_use_validation_seed"] == 20290820
    assert request["denominators"]["total"] == 1680


def test_candidate2_protocol_is_frozen_before_implementation_or_data_access() -> None:
    protocol = json.loads(
        (
            ROOT
            / "configs/diagnosis/generic_head_calibration_protocol.v2.json"
        ).read_text(encoding="utf-8")
    )
    assert protocol["candidate_ordinal"] == 2
    assert protocol["maximum_candidates"] == 3
    assert protocol["known_mechanisms"] == ["CAPACITY_LOSS", "DRIFT", "LEAKAGE"]
    assert protocol["open_set_mechanisms"] == ["BIAS", "INTERMITTENT", "STUCK"]
    assert protocol["split"]["development_seed"] == 20260820
    assert protocol["split"]["calibration_seed"] == 20270820
    assert protocol["split"]["exhausted_validation_seed"] == 20280820
    assert protocol["split"]["technical_validation_seed"] == 20290820
    assert protocol["split"]["technical_validation_run_count"] == 1
    assert protocol["denominators"] == {
        "normal": 240,
        "known": 720,
        "open_set": 720,
        "total": 1680,
    }
    assert protocol["thresholds"]["detection_accuracy"] == 0.98
    assert protocol["thresholds"]["classification_accuracy"] == 0.90
    assert protocol["thresholds"]["localization_accuracy"] == 0.95
    assert protocol["thresholds"]["reasoning_correctness"] == 0.90
    assert protocol["thresholds"]["known_class_retention"] == 0.98
    assert protocol["thresholds"]["open_set_rejection"] == 0.98
    assert protocol["production_candidate"]["head_count"] == 1
    assert protocol["production_candidate"]["ensemble_allowed"] is False
    assert protocol["production_candidate"]["fallback_model_allowed"] is False
    assert protocol["production_candidate"]["second_implicit_production_model_allowed"] is False
    for item in protocol["authorization"].items():
        if item[0].endswith("_path"):
            digest_key = item[0].removesuffix("_path") + "_sha256"
            digest = hashlib.sha256((ROOT / item[1]).read_bytes()).hexdigest()
            assert digest == protocol["authorization"][digest_key]
