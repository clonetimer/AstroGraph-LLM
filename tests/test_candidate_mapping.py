from __future__ import annotations

import json
from pathlib import Path

import pytest

from astrograph.data_driven.candidate_mapping import (
    NoMappedFaultMassError,
    map_runtime_fault_attribution,
)
from astrograph.data_driven.adapters import rebind_prediction_episode
from astrograph.data_driven.models import ClassProbability, DataDrivenPrediction


ROOT = Path(__file__).resolve().parents[1]


def test_mapping_retains_unsupported_mass_without_redistribution() -> None:
    result = map_runtime_fault_attribution(
        {"dataset_known": 0.3, "dataset_unknown": 0.7},
        {"dataset_known": "FAULT_A"},
        {"FAULT_A"},
    )
    assert result.mapped_attribution == {"FAULT_A": pytest.approx(0.3)}
    assert result.unsupported_probabilities == {"dataset_unknown": pytest.approx(0.7)}
    assert result.mapped_mass == pytest.approx(0.3)
    assert result.unsupported_mass == pytest.approx(0.7)


def test_zero_mapped_mass_abstains_instead_of_manufacturing_vote() -> None:
    with pytest.raises(NoMappedFaultMassError, match="no probability mass"):
        map_runtime_fault_attribution(
            {"dataset_unknown": 1.0}, {"dataset_known": "FAULT_A"}, {"FAULT_A"}
        )


def test_episode_rebind_recomputes_prediction_identity() -> None:
    prediction = DataDrivenPrediction(
        prediction_id="PRED-OLD",
        episode_id="EPISODE-OLD",
        model_id="model",
        model_version="1",
        model_manifest_id="model@1",
        classes=[
            ClassProbability(class_id="FAULT_A", probability=0.8, rank=1),
            ClassProbability(class_id="NORMAL", probability=0.2, rank=2),
        ],
        feature_schema_id="schema",
        input_data_sha256="a" * 64,
    )
    rebound = rebind_prediction_episode(prediction, "EPISODE-NEW")
    assert rebound.episode_id == "EPISODE-NEW"
    assert rebound.prediction_id != prediction.prediction_id
    assert rebind_prediction_episode(prediction, "EPISODE-NEW").prediction_id == rebound.prediction_id


@pytest.mark.parametrize("dataset", ["adcs_v5_2_2", "phm_v6_0"])
def test_candidate_universe_matches_runtime_evidence(dataset: str) -> None:
    config_root = ROOT / "configs/datasets" / dataset
    universe = json.loads((config_root / "candidate_universe.json").read_text(encoding="utf-8"))
    evidence = json.loads((config_root / "runtime_evidence.json").read_text(encoding="utf-8"))
    evidence_faults = sorted({item["target_fault_id"] for item in evidence["evidence_definitions"]})
    configured = sorted(universe.get("configured_rule_evidence_fault_ids", evidence_faults))
    assert configured == evidence_faults
    assert set(configured).issubset(universe["runtime_fault_ids"])
    assert universe["zero_mapped_mass_policy"] == "ABSTAIN_DATA_BRANCH"
