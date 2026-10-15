from __future__ import annotations

import numpy as np
import pytest

from astrograph.evaluation.data_exposure import dataset_use
from astrograph.evaluation.dxc_hybrid_model_based import (
    DXCHybridResidualConfig,
    DXCNumericSequence,
    extract_numeric_sequence,
    fit_hybrid_residual_model,
    hybrid_model_artifact,
    hybrid_model_from_artifact,
    predict_hybrid_model_based,
)
from astrograph.evaluation.nasa_adapt_adapter import parse_scenario, system_catalog


XML = b'''<root xmlns="urn:org:dx-competition:system" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<modeCatalog>
  <modeGroup><name>Breaker</name><mode xsi:type="mode"><name>Nominal</name></mode><mode xsi:type="faultMode"><name>FailedOpen</name></mode></modeGroup>
  <modeGroup><name>Sense</name><mode xsi:type="mode"><name>Nominal</name></mode><mode xsi:type="faultMode"><name>Offset</name></mode></modeGroup>
</modeCatalog>
<componentTypeCatalog>
  <componentType xsi:type="switch"><name>BreakerType</name><modesRef>Breaker</modesRef></componentType>
  <componentType xsi:type="sensor"><name>SensorType</name><engUnits>VDC</engUnits><modesRef>Sense</modesRef><sensorValue><dataType>real</dataType></sensorValue></componentType>
</componentTypeCatalog>
<systems><system><components>
  <component><name>CB1</name><componentType>BreakerType</componentType><command><name>CB1_OPEN</name><commandValue><dataType>bool</dataType></commandValue></command></component>
  <component><name>E1</name><componentType>SensorType</componentType></component>
  <component><name>E2</name><componentType>SensorType</componentType></component>
</components><connections>
  <connection><c1>CB1</c1><c2>E1</c2></connection><connection><c1>CB1</c1><c2>E2</c2></connection>
</connections></system></systems>
</root>'''


def _normal_sequence() -> DXCNumericSequence:
    time = np.arange(0.0, 61.0)
    return DXCNumericSequence(
        time_s=time,
        sensor_values=np.c_[0.1 * time, -0.05 * time],
        command_states=np.zeros((len(time), 1), dtype=float),
        sensor_ids=("E1", "E2"),
        command_ids=("CB1_OPEN",),
    )


def _fit_model():
    with dataset_use("SYNTHETIC_PROTOCOL", "MODEL_TRAIN", "FIT", group_ids=["dxc-synth-fit"]):
        return fit_hybrid_residual_model([_normal_sequence(), _normal_sequence()], alpha=10.0)


def test_command_state_is_preserved_as_model_input():
    catalog = system_catalog(XML)
    scenario = "\n".join([
        "sensors @0 {E1=0, E2=0};",
        "command @1000 CB1_OPEN = true;",
        "sensors @1000 {E1=0.1, E2=-0.05};",
        "sensors @2000 {E1=0.2, E2=-0.1};",
    ])
    records, _ = parse_scenario(scenario)
    sequence = extract_numeric_sequence(records, catalog)
    assert sequence.command_states[:, 0].tolist() == [0.0, 1.0, 1.0]


def test_hybrid_model_artifact_roundtrip_is_content_bound():
    model = _fit_model()
    artifact = hybrid_model_artifact(model)
    restored = hybrid_model_from_artifact(artifact)
    assert restored.model_sha256 == model.model_sha256
    tampered = dict(artifact)
    tampered["ridge_intercept"] = list(tampered["ridge_intercept"])
    tampered["ridge_intercept"][0] += 0.1
    with pytest.raises(ValueError, match="IDENTITY_DRIFT"):
        hybrid_model_from_artifact(tampered)


def test_hybrid_predictor_detects_post_startup_offset_and_retains_mbd_audit():
    catalog = system_catalog(XML)
    model = hybrid_model_from_artifact(hybrid_model_artifact(_fit_model()))
    lines = []
    for t in range(61):
        e1 = 0.1 * t + (8.0 if t >= 40 else 0.0)
        e2 = -0.05 * t
        lines.append(f"sensors @{t*1000} {{E1={e1}, E2={e2}}};")
    records, _ = parse_scenario("\n".join(lines))
    with dataset_use("SYNTHETIC_PROTOCOL", "DEVELOPMENT_SELECTION", "OPTIMIZE", group_ids=["dxc-synth-score"]):
        result = predict_hybrid_model_based(
            records,
            catalog,
            model,
            DXCHybridResidualConfig(
                startup_calibration_s=30.0,
                residual_z_threshold=6.0,
                persistence_decisions=1,
                top_residual_sensors=1,
                top_components=2,
            ),
        )
    assert result.first_alarm_time_s is not None
    assert result.first_alarm_time_s >= 40.0
    assert "E1" in result.final_component_ids
    assert result.minimal_diagnosis is not None
    assert result.score_semantics.endswith("NOT_PROBABILITY")
