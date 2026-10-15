from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ops_sat_action_policy_comparator",
    ROOT / "scripts/experiments/run_ops_sat_action_policy_comparator.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _points(segment: str, values: list[float]) -> list[dict]:
    return [
        {
            "channel": "CADC0001",
            "timestamp": f"2026-01-01T00:00:{index:02d}+00:00",
            "value": value,
            "sampling": "1",
            "segment": segment,
        }
        for index, value in enumerate(values)
    ]


def test_action_contracts_fit_only_model_train_and_are_deterministic() -> None:
    raw = {
        "1": _points("1", [0, 0, 0, 0, 0, 0, 0, 0]),
        "2": _points("2", [0, 0, 0, 0, 4, 4, 4, 4]),
        "3": _points("3", [100, 100, 100, 100, -100, -100, -100, -100]),
    }
    truth = {
        "1": {"anomaly": False},
        "2": {"anomaly": True},
        "3": {"anomaly": False},
    }
    partitions = {"1": "MODEL_TRAIN", "2": "MODEL_TRAIN", "3": "DEVELOPMENT_SELECTION"}
    first = MODULE.fit_action_contracts(raw, truth, partitions)
    second = MODULE.fit_action_contracts(raw, truth, partitions)
    assert first == second
    assert first["training_group_count"] == 2
    assert first["evaluation_truth_used_for_fit"] is False
    assert first["selected_action"] in MODULE.ACTIVE_ACTIONS


def test_contract_prediction_and_stable_random_action_are_bounded() -> None:
    contract = {"threshold": 1.5, "anomalous_when": "GE"}
    assert MODULE.contract_prediction(contract, 2.0) == "ANOMALOUS"
    assert MODULE.contract_prediction(contract, 1.0) == "NOMINAL"
    assert MODULE.stable_random_action("17") == MODULE.stable_random_action("17")
    assert MODULE.stable_random_action("17") in MODULE.ACTIVE_ACTIONS


def test_deterministic_active_arm_exposes_one_non_voting_observation() -> None:
    action = "inspect_early_late_shift"
    contracts = {
        "contracts": {
            action: {"threshold": 0.5, "anomalous_when": "GE"},
        }
    }
    frozen = {
        "score": 0.2,
        "threshold": 0.5,
        "predicted_anomaly": False,
        "score_semantics": "test",
    }
    arm = MODULE.run_deterministic_arm(
        arm="TRAIN_INFORMATION_GAIN_POLICY",
        group_id="7",
        points=_points("7", [0, 0, 0, 0, 2, 2, 2, 2]),
        frozen=frozen,
        action=action,
        contracts=contracts,
    )
    assert arm["active_observation_exposed"] is True
    assert arm["observation"]["formal_evidence_generated"] is False
    assert len(arm["validated_candidate_states"]) == 1
    assert arm["usage"]["call_count"] == 0
