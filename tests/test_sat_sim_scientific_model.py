from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from astrograph.assurance import AdmissionError, canonical_sha256
from astrograph.data_driven import ModelInferenceRequest, ObservationScope
from astrograph.integrations.sat_sim_scientific_model import (
    build_registry_from_sat_sim_package,
    load_sat_sim_scientific_model_package,
)


def test_sat_sim_model_package_loads_and_runs_review_only(tmp_path: Path) -> None:
    order = ["f1", "f2"]
    estimator = LogisticRegression().fit(
        pd.DataFrame([[0, 0], [1, 1], [2, 2], [3, 3]], columns=order),
        ["NORMAL", "NORMAL", "ADCS_RW_JAM", "ADCS_RW_JAM"],
    )
    artifact = tmp_path / "model.joblib"
    joblib.dump({"model": estimator, "feature_columns": order}, artifact)
    import hashlib

    sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "astrograph-data-driven-model-manifest.v1",
        "model_id": "rw-test",
        "model_version": "0.1",
        "display_name": "RW test",
        "task_type": "MULTICLASS_FAULT_CLASSIFICATION",
        "runtime": "SCIKIT_LEARN",
        "output_semantics": "PROBABILITIES",
        "capabilities": ["FAULT_RANKING", "NORMAL_CLASS"],
        "supported_fault_ids": ["ADCS_RW_JAM"],
        "normal_class_id": "NORMAL",
        "feature_schema_ids": ["rw.features.v1"],
        "input_contract": {"feature_order": order},
        "training_dataset_ids": ["proxy"],
        "claim_boundary": "engineering screening only",
        "metadata": {"artifact_file": artifact.name, "artifact_sha256": sha, "formal_model_binding_enabled": False},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    package = load_sat_sim_scientific_model_package(manifest_path)
    registry = build_registry_from_sat_sim_package(package)
    payload = {"features": {"f1": 3, "f2": 3}}
    request = ModelInferenceRequest(
        episode_id="episode", model_id="rw-test", model_version="0.1",
        feature_schema_id="rw.features.v1", input_data_sha256=canonical_sha256(payload),
        registered_fault_ids=["ADCS_RW_JAM"], payload=payload,
        observation_scope=ObservationScope(
            scope_id="sat-sim-scientific-window-1", source_id="sat-sim-scientific-source",
            window_start_s=0.0, window_end_s=1.0,
        ),
    )
    prediction = registry.predict(request)
    assert registry.require_admission is True
    assert prediction.classes[0].class_id == "ADCS_RW_JAM"
    assert package.formal_model_binding_eligible is False
    assert prediction.calibrated_probabilities is False
    assert "engineering" in package.claim_boundary.lower()
    assert prediction.metadata["model_admission"]["feature_contract_id"]

    with pytest.raises(AdmissionError, match="unapproved model inputs"):
        registry.predict(
            request.model_copy(
                update={
                    "payload": {"features": {"f1": 3, "f2": 3, "case_id": 99}},
                    "input_data_sha256": canonical_sha256(
                        {"features": {"f1": 3, "f2": 3, "case_id": 99}}
                    ),
                }
            )
        )
