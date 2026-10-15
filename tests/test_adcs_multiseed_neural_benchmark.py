from __future__ import annotations

import pytest

from scripts.data_driven.run_adcs_multiseed_neural_benchmark import aggregate, parse_args


def test_formal_benchmark_rejects_short_training() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--epochs", "29"])


def test_formal_benchmark_requires_three_distinct_seeds() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--seeds", "1", "1", "2"])


def test_formal_benchmark_defaults_are_long_and_multiseed() -> None:
    args = parse_args([])
    assert args.epochs >= 30
    assert args.minimum_epochs >= 20
    assert len(set(args.seeds)) >= 3


def test_aggregate_records_external_nominal_seed_robustness() -> None:
    task = {"accuracy": 1.0, "macro_precision": 1.0, "macro_recall": 1.0, "macro_f1": 1.0}
    run = {
        "detection": task,
        "localization": task,
        "classification_predicted_component": task,
        "end_to_end_15_class": {"macro_f1": 1.0},
        "best_validation": {
            "fault_event_recall": 1.0,
            "nominal_trajectory_false_alarm_rate": 0.0,
            "false_alarm_episodes_per_hour": 0.0,
            "meets_nominal_false_alarm_constraint": True,
        },
        "independent_nominal_envelope": {"trajectory_false_alarm_rate": 0.0},
    }
    result = aggregate([run, run, run])
    assert result["all_seeds_zero_confirmation_false_alarms"] is True
    assert result["confirmation_nominal_trajectory_false_alarm_rate"]["maximum"] == 0.0
