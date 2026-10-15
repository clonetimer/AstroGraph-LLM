from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from astrograph.evaluation.data_exposure import dataset_use

from astrograph.data_driven.phm_v60 import (
    PHMDataset,
    PHMRecord,
    _route_class_probabilities,
    train_three_stage_adaptive,
)


@pytest.fixture(autouse=True)
def _synthetic_development_scope():
    with dataset_use("SYNTHETIC_PROTOCOL", "MODEL_TRAIN", "FIT"):
        yield


def _record(index: int, split: str, condition: str) -> PHMRecord:
    return PHMRecord(
        sample_id=f"{split}:{condition}:{index}",
        sample_index=index,
        subsystem="test",
        condition_id=condition,
        activation_time_s=1.0,
        event_start_s=1.0,
        fault_manifestation_time_s=1.0,
        fault_injection_phase="test",
        telemetry_path=Path(f"unused-{index}.csv"),
        split=split,
    )


def test_route_class_probabilities_respects_component_contract() -> None:
    class_prob = np.asarray([[0.9, 0.1], [0.8, 0.2]], dtype=float)
    component_prob = np.asarray([[0.1, 0.9], [0.9, 0.1]], dtype=float)
    class_labels = ["fault_a", "fault_b"]
    component_labels = ["component_a", "component_b"]
    mapping = {"fault_a": "component_a", "fault_b": "component_b"}

    raw = _route_class_probabilities(
        class_prob, component_prob, class_labels, component_labels, mapping, "raw"
    )
    hard = _route_class_probabilities(
        class_prob, component_prob, class_labels, component_labels, mapping, "hard"
    )

    assert raw.tolist() == ["fault_a", "fault_a"]
    assert hard.tolist() == ["fault_b", "fault_a"]


def test_adaptive_three_stage_is_validation_selected_and_test_scoped(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    rows: list[list[float]] = []
    records: list[PHMRecord] = []
    components: list[str] = []
    index = 0
    for split, count in (("train", 8), ("validation", 3), ("test", 3)):
        for condition, center, component in (
            ("nominal", 0.0, "nominal"),
            ("fault_a", 3.0, "component_a"),
            ("fault_b", -3.0, "component_b"),
        ):
            for _ in range(count):
                rows.append((rng.normal(center, 0.25, size=12)).tolist())
                records.append(_record(index, split, condition))
                components.append(component)
                index += 1
    dataset = PHMDataset(
        features=pd.DataFrame(rows, columns=[f"signal_{i}" for i in range(12)]),
        records=records,
        components=np.asarray(components, dtype=str),
    )
    result = train_three_stage_adaptive(
        dataset,
        {"fault_a": "component_a", "fault_b": "component_b"},
    )

    assert result["schema_version"] == "astrograph-phm-v60-three-stage-result.v2"
    assert result["strategy"] == "adaptive_classical_validation_selected"
    assert result["selection"]["selection_boundary"].startswith("Train fits candidates")
    assert result["selection"]["detection"]["candidate"]["id"]
    assert result["selection"]["classification"]["route"] in {
        "raw", "soft:0.25", "soft:0.5", "soft:1", "hard"
    }
    assert result["detection"]["sample_count"] == 9
    assert result["localization"]["sample_count"] == 6
    assert len(result["test_predictions"]) == 9

    import joblib
    for name, model in result["models"].items():
        target = tmp_path / f"{name}.joblib"
        joblib.dump(model, target)
        restored = joblib.load(target)
        assert restored.predict(dataset.features[result["feature_columns"][name]].iloc[:1]).shape == (1,)


def test_promoted_extra_trees_strategy_is_fixed_family_and_raw_route() -> None:
    from astrograph.data_driven.phm_v60 import train_three_stage_promoted_extra_trees

    rng = np.random.default_rng(314)
    rows: list[list[float]] = []
    records: list[PHMRecord] = []
    components: list[str] = []
    index = 0
    for split, count in (("train", 8), ("validation", 3), ("test", 3)):
        for condition, center, component in (
            ("nominal", 0.0, "nominal"),
            ("fault_a", 3.0, "component_a"),
            ("fault_b", -3.0, "component_b"),
        ):
            for _ in range(count):
                rows.append((rng.normal(center, 0.2, size=16)).tolist())
                records.append(_record(index, split, condition))
                components.append(component)
                index += 1
    dataset = PHMDataset(
        features=pd.DataFrame(rows, columns=[f"signal_{i}" for i in range(16)]),
        records=records,
        components=np.asarray(components, dtype=str),
    )
    result = train_three_stage_promoted_extra_trees(
        dataset,
        {"fault_a": "component_a", "fault_b": "component_b"},
    )

    assert result["strategy"] == "promoted_extra_trees_trace_screening_v1"
    assert result["selection"]["detection"]["candidate"]["id"] == "extra-trees-500"
    assert result["selection"]["localization"]["candidate"]["id"] == "extra-trees-500"
    assert result["selection"]["classification"]["candidate"]["id"] == "extra-trees-500"
    assert result["selection"]["classification"]["route"] == "raw"
    assert len(result["selection"]["candidate_results"]["detection"]) == 1
    assert len(result["selection"]["candidate_results"]["localization"]) == 1
    assert len(result["selection"]["candidate_results"]["classification"]) == 1


def test_promoted_bundle_round_trip_preserves_predictions(tmp_path: Path) -> None:
    import hashlib
    import json
    import joblib
    from astrograph.data_driven.phm_v60 import (
        PROMOTED_EXTRA_TREES_SPEC,
        load_three_stage_promoted_bundle,
        train_three_stage_promoted_extra_trees,
    )

    rng = np.random.default_rng(2718)
    rows: list[list[float]] = []
    records: list[PHMRecord] = []
    components: list[str] = []
    index = 0
    for split, count in (("train", 8), ("validation", 3), ("test", 3)):
        for condition, center, component in (
            ("nominal", 0.0, "nominal"),
            ("fault_a", 3.0, "component_a"),
            ("fault_b", -3.0, "component_b"),
        ):
            for _ in range(count):
                rows.append((rng.normal(center, 0.2, size=10)).tolist())
                records.append(_record(index, split, condition))
                components.append(component)
                index += 1
    dataset = PHMDataset(
        features=pd.DataFrame(rows, columns=[f"signal_{i}" for i in range(10)]),
        records=records,
        components=np.asarray(components, dtype=str),
    )
    trained = train_three_stage_promoted_extra_trees(
        dataset, {"fault_a": "component_a", "fault_b": "component_b"}
    )

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    model_entries = {}
    for head, model in trained["models"].items():
        path = bundle / f"{head}.joblib"
        joblib.dump(model, path)
        model_entries[head] = {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "schema_version": "astrograph-phm-v60-promoted-model-bundle.v1",
        "bundle_id": "test-bundle",
        "strategy": "promoted_extra_trees_trace_screening_v1",
        "model_spec": PROMOTED_EXTRA_TREES_SPEC,
        "feature_scope": "post_activation_trace_summary",
        "detection_threshold": trained["selection"]["detection"]["threshold"],
        "classification_route": "raw",
        "feature_columns": trained["feature_columns"],
        "models": model_entries,
        "claim_boundary": "test only",
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_three_stage_promoted_bundle(dataset, bundle)
    assert loaded["end_to_end_18_class"]["macro_f1"] == trained["end_to_end_18_class"]["macro_f1"]
    assert [x["predicted_class"] for x in loaded["test_predictions"]] == [
        x["predicted_class"] for x in trained["test_predictions"]
    ]
    assert loaded["model_bundle"]["loaded_frozen_artifacts"] is True
