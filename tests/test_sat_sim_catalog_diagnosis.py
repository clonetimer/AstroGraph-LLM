from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from astrograph.diagnosis.sat_sim_catalog import (
    ALLOWED_FIELDS,
    CatalogDiagnosisStatus,
    NORMAL_CLASS_ID,
    SatSimCatalogDiagnosisEngine,
    save_catalog_artifact,
    train_catalog_artifact,
)


def _frame(backend: str, *, payload: float = 1.0, downlink: float = 1.0) -> pd.DataFrame:
    data: dict[str, list[float]] = {"time_s": [0.0, 1.0, 2.0, 3.0]}
    for field in ALLOWED_FIELDS[backend]:
        data[field] = [0.0, 0.0, 0.0, 0.0]
    if backend == "unified":
        data["payload.generated_bps"] = [payload] * 4
        data["payload.active"] = [1.0 if payload else 0.0] * 4
        data["comm.downlink_bps"] = [downlink] * 4
        data["comm.active"] = [1.0 if downlink else 0.0] * 4
    return pd.DataFrame(data)


def _artifact() -> dict:
    examples = {
        "unified": [
            (NORMAL_CLASS_ID, _frame("unified", payload=2_000_000.0, downlink=1_500_000.0)),
            (NORMAL_CLASS_ID, _frame("unified", payload=2_100_000.0, downlink=1_600_000.0)),
            ("payload_instrument_off", _frame("unified", payload=0.0, downlink=1_500_000.0)),
            ("payload_instrument_off", _frame("unified", payload=0.0, downlink=1_600_000.0)),
            ("comm_data_downlink_link_loss", _frame("unified", payload=2_000_000.0, downlink=0.0)),
            ("comm_data_downlink_link_loss", _frame("unified", payload=2_100_000.0, downlink=0.0)),
        ]
    }
    return train_catalog_artifact(training_frames=examples, model_version="test", random_state=7)


def test_single_trace_catalog_distinguishes_normal_and_effect() -> None:
    engine = SatSimCatalogDiagnosisEngine(_artifact())
    normal = engine.diagnose_frame(
        case_id="opaque-normal", backend="unified",
        telemetry=_frame("unified", payload=2_050_000.0, downlink=1_550_000.0),
    )
    effect = engine.diagnose_frame(
        case_id="opaque-effect", backend="unified",
        telemetry=_frame("unified", payload=0.0, downlink=1_550_000.0),
    )
    assert normal.status == CatalogDiagnosisStatus.NORMAL
    assert normal.conclusion_effect_id is None
    assert effect.status == CatalogDiagnosisStatus.DIAGNOSED
    assert effect.conclusion_effect_id == "payload_instrument_off"
    assert effect.truth_used is False
    assert effect.labels_used is False


def test_catalog_rejects_truth_or_label_columns() -> None:
    frame = _frame("unified", payload=0.0, downlink=1_500_000.0)
    frame["truth.effect_id"] = 1.0
    result = SatSimCatalogDiagnosisEngine(_artifact()).diagnose_frame(
        case_id="opaque", backend="unified", telemetry=frame
    )
    assert result.status == CatalogDiagnosisStatus.DATA_ERROR
    assert result.prohibited_columns_used == ["truth.effect_id"]


def test_case_id_does_not_change_catalog_prediction() -> None:
    engine = SatSimCatalogDiagnosisEngine(_artifact())
    frame = _frame("unified", payload=2_000_000.0, downlink=0.0)
    first = engine.diagnose_frame(case_id="CASE-001", backend="unified", telemetry=frame)
    second = engine.diagnose_frame(case_id="renamed-and-shuffled", backend="unified", telemetry=frame)
    assert first.conclusion_effect_id == second.conclusion_effect_id
    assert first.status == second.status
    assert first.candidates == second.candidates


def test_catalog_cli_writes_result(tmp_path: Path) -> None:
    from astrograph.cli import _diagnosis_sat_sim_catalog

    model = tmp_path / "model.joblib"
    manifest = tmp_path / "model_manifest.json"
    telemetry = tmp_path / "telemetry.csv"
    output = tmp_path / "diagnosis.json"
    save_catalog_artifact(_artifact(), model)
    manifest.write_text(json.dumps({
        "schema_version": "astrograph-sat-sim-catalog-release.v1",
        "model_file": model.name,
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    _frame("unified", payload=0.0, downlink=1_500_000.0).to_csv(telemetry, index=False)
    rc = _diagnosis_sat_sim_catalog(argparse.Namespace(
        case_id="opaque", backend="unified", model=manifest, telemetry=telemetry, output=output,
    ))
    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["conclusion_effect_id"] == "payload_instrument_off"
    assert payload["claim_boundary"].startswith("Known-catalog single-trace")


def test_compound_dynamic_resolver_handles_low_effect_vote() -> None:
    import numpy as np
    from astrograph.diagnosis.sat_sim_catalog import extract_catalog_features

    class StaticModel:
        def __init__(self, classes: list[str], votes: list[float]) -> None:
            self.classes_ = np.asarray(classes, dtype=str)
            self._votes = np.asarray(votes, dtype=float)
        def predict_proba(self, rows):
            return np.asarray([self._votes for _ in rows], dtype=float)

    frame = _frame("composite")
    frame["attitude_error_deg"] = [0.0, 0.5, 1.0, 2.0]
    frame["attitude_error_norm"] = [0.0, 3.0, 7.0, 10.0]
    frame["rate_error_norm_rad_s"] = [1.0, 1.0, 1.0, 1.0]
    vector, names, _ = extract_catalog_features(frame, backend="composite")
    classes = ["NORMAL", "adcs_rw_bearing_seizure", "adcs_rw_friction_and_sensor_noise"]
    artifact = {
        "schema_version": "astrograph-sat-sim-catalog-model.v1",
        "normal_class_id": NORMAL_CLASS_ID,
        "backend_models": {
            "composite": {
                "model": StaticModel(classes, [0.05, 0.50, 0.45]),
                "effect_model": StaticModel(classes[1:], [0.64, 0.36]),
                "feature_names": names,
                "medians": np.zeros_like(vector),
                "active_mask": np.ones_like(vector, dtype=bool),
                "effect_active_mask": np.ones_like(vector, dtype=bool),
                "class_profiles": {},
                "resolver_thresholds": {
                    "attitude_delta_over_rate_mean_max": 3.0,
                    "rate_mean_over_attitude_norm_delta_min": 0.08,
                },
                "minimum_observed_field_ratio": 0.5,
            }
        },
    }
    result = SatSimCatalogDiagnosisEngine(artifact).diagnose_frame(
        case_id="opaque-compound", backend="composite", telemetry=frame
    )
    assert result.conclusion_effect_id == "adcs_rw_friction_and_sensor_noise"
    assert result.resolution_basis == "COMPOUND_DYNAMIC_RESOLVER"
    assert result.model_vote_share == 0.45
    assert result.vote_margin < 0.0
