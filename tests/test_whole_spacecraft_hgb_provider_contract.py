from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from astrograph.assurance import canonical_sha256
from astrograph.data_driven import (
    DataDrivenModelManifest,
    DataDrivenTaskType,
    ModelCapability,
    ModelInferenceRequest,
    ModelOutputSemantics,
    ModelRuntime,
)
from astrograph.data_driven.whole_spacecraft_hgb import WholeSpacecraftHGBProvider


class _Classifier:
    classes_ = np.asarray(["NORMAL", "ADCS_RW_JAM"])

    def predict_proba(self, frame):
        assert list(frame.columns) == ["x", "y"]
        return np.asarray([[0.25, 0.75]])


class _Calibrator:
    def transform(self, probabilities):
        return probabilities


def _provider() -> WholeSpacecraftHGBProvider:
    manifest = DataDrivenModelManifest(
        model_id="hgb-contract-test",
        model_version="1",
        display_name="HGB contract test",
        task_type=DataDrivenTaskType.MULTICLASS_FAULT_CLASSIFICATION,
        runtime=ModelRuntime.SCIKIT_LEARN,
        output_semantics=ModelOutputSemantics.PROBABILITIES,
        capabilities=[ModelCapability.FAULT_RANKING, ModelCapability.NORMAL_CLASS],
        supported_fault_ids=["ADCS_RW_JAM"],
        feature_schema_ids=["hgb.features.v1"],
    )
    model = SimpleNamespace(
        manifest=manifest,
        feature_columns=["x", "y"],
        classifier=_Classifier(),
        calibrator=_Calibrator(),
        model_id=manifest.model_id,
        model_version=manifest.model_version,
        calibration_record=SimpleNamespace(calibration_id="cal-test"),
    )
    return WholeSpacecraftHGBProvider(model)


def _request(features: dict[str, float]) -> ModelInferenceRequest:
    payload = {"features": features}
    return ModelInferenceRequest(
        episode_id="episode-1",
        model_id="hgb-contract-test",
        model_version="1",
        feature_schema_id="hgb.features.v1",
        input_data_sha256=canonical_sha256(payload),
        registered_fault_ids=["ADCS_RW_JAM"],
        payload=payload,
    )


def test_hgb_provider_requires_exact_finite_feature_mapping() -> None:
    output = _provider().predict_raw(_request({"x": 1.0, "y": 2.0}))
    assert output.class_values["ADCS_RW_JAM"] == pytest.approx(0.75)

    with pytest.raises(ValueError, match="missing=.*y"):
        _provider().predict_raw(_request({"x": 1.0}))
    with pytest.raises(ValueError, match="unknown=.*z"):
        _provider().predict_raw(_request({"x": 1.0, "y": 2.0, "z": 3.0}))
    with pytest.raises(ValueError, match="finite"):
        _provider().predict_raw(_request({"x": 1.0, "y": float("nan")}))
