from __future__ import annotations

import os
import csv
import hashlib
import json
from pathlib import Path

import pytest

from astrograph.integrations.m2n_challenger import (
    M2NChallengerAuditRequest,
    M2NChallengerAuditService,
    M2NChallengerStatus,
)


M2N = Path(os.environ.get("ASTROGRAPH_M2N_EXECUTABLE", "__optional_m2n_not_found__"))


def _write_fixture(
    tmp_path: Path,
    *,
    include_calibration: bool = True,
    crossing_group: bool = False,
    seeds: list[int] | None = None,
    include_context_minimums: bool = True,
) -> tuple[Path, str, Path, str]:
    roles = {
        "train": [("train-a-1", "A"), ("train-a-2", "A"), ("train-b-1", "B"), ("train-b-2", "B")],
        "validation": [("val-a", "A"), ("val-b", "B")],
        "calibration": [("cal-a-1", "A"), ("cal-a-2", "A"), ("cal-b-1", "B"), ("cal-b-2", "B")],
        "test": [("test-a", "A"), ("test-b", "B")],
    }
    if not include_calibration:
        roles.pop("calibration")
    rows: list[dict[str, object]] = []
    for split, groups in roles.items():
        for group, label in groups:
            for time_index in range(5):
                base = -1.0 if label == "A" else 1.0
                rows.append(
                    {
                        "trajectory_id": group,
                        "split": split,
                        "time_s": float(time_index),
                        "x1": base + 0.05 * time_index,
                        "x2": 0.5 * base - 0.02 * time_index,
                        "label": label,
                    }
                )
    if crossing_group:
        rows[-1]["trajectory_id"] = "train-a-1"
    data_path = tmp_path / "challenger.csv"
    with data_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    qualification: dict[str, object] = {"independent_context_column": "trajectory_id"}
    if include_context_minimums:
        qualification.update(
            {
                "min_train_independent_contexts": 4,
                "min_validation_independent_contexts": 2,
            }
        )
    spec = {
        "modeling_contract": {
            "objective": "CATEGORICAL_PREDICTION",
            "input_representation": "MULTIVARIATE_SEQUENCE",
            "mechanism_usage": "NONE",
            "prediction_unit": "INDEPENDENT_CONTEXT",
            "input_columns": ["x1", "x2"],
            "label_column": "label",
            "time_column": "time_s",
            "constraints": {
                "max_dt_relative_spread": 0.001,
                "temporal_training": {
                    "seeds": seeds or [101, 202, 303],
                    "epochs": 2,
                    "hidden_size": 4,
                    "lr": 0.02,
                },
            },
        },
        "qualification": qualification,
        "data": {
            "split_column": "split",
            "train_values": ["train"],
            "validation_values": ["validation"],
            "time_column": "time_s",
        },
        "selection": {"practical_ratio": 0.8, "restart_stability": 0.75},
    }
    spec_path = tmp_path / "challenger-spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return (
        spec_path,
        hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        data_path,
        hashlib.sha256(data_path.read_bytes()).hexdigest(),
    )


def _request(tmp_path: Path, **kwargs: object) -> M2NChallengerAuditRequest:
    tmp_path.mkdir(parents=True, exist_ok=True)
    spec, spec_sha, data, data_sha = _write_fixture(tmp_path, **kwargs)
    return M2NChallengerAuditRequest(
        spec_path=str(spec),
        expected_spec_sha256=spec_sha,
        data_path=str(data),
        expected_data_sha256=data_sha,
        output_dir=str(tmp_path / "output"),
        group_column="trajectory_id",
        split_column="split",
        label_column="label",
        executable_path=str(M2N),
        timeout_s=300.0,
    )


@pytest.mark.skipif(not M2N.is_file(), reason="optional external M2N runtime is not installed")
def test_offline_challenger_runs_four_roles_and_never_promotes(tmp_path: Path) -> None:
    audit = M2NChallengerAuditService().audit(_request(tmp_path))

    assert audit.status == M2NChallengerStatus.COMPLETED
    assert audit.split_group_counts == {
        "calibration": 4,
        "test": 2,
        "train": 4,
        "validation": 2,
    }
    assert audit.optimizer_seeds == [101, 202, 303]
    assert audit.simple_baseline_id == "summary_mlp"
    assert "summary_mlp" in audit.candidate_ids
    assert audit.selection_status == "AMBIGUOUS"
    assert audit.selected_candidate_id is None
    assert audit.calibration_context_count == 4
    assert audit.test_context_count == 2
    assert 0.0 <= float(audit.test_macro_f1) <= 1.0
    assert 0.0 <= float(audit.test_prediction_set_coverage) <= 1.0
    assert audit.promotion_allowed is False
    assert audit.candidate_bound_to_production is False
    assert audit.runtime is not None and audit.runtime.version == "2.16.1+timer.1"
    assert audit.output_manifest_sha256


def test_missing_independent_calibration_role_fails_before_m2n(tmp_path: Path) -> None:
    request = _request(tmp_path, include_calibration=False).model_copy(
        update={"executable_path": str(tmp_path / "missing-m2n")}
    )
    audit = M2NChallengerAuditService().audit(request)

    assert audit.status == M2NChallengerStatus.ERROR
    assert "non-empty train/validation/calibration/test" in audit.reason
    assert audit.runtime is None


def test_group_crossing_roles_fails_before_m2n(tmp_path: Path) -> None:
    request = _request(tmp_path, crossing_group=True).model_copy(
        update={"executable_path": str(tmp_path / "missing-m2n")}
    )
    audit = M2NChallengerAuditService().audit(request)

    assert audit.status == M2NChallengerStatus.ERROR
    assert "crosses split or label boundaries" in audit.reason
    assert audit.runtime is None


def test_fewer_than_three_optimizer_seeds_fails_before_m2n(tmp_path: Path) -> None:
    request = _request(tmp_path, seeds=[101, 202]).model_copy(
        update={"executable_path": str(tmp_path / "missing-m2n")}
    )
    audit = M2NChallengerAuditService().audit(request)

    assert audit.status == M2NChallengerStatus.ERROR
    assert "distinct declared optimizer seeds" in audit.reason
    assert audit.runtime is None


def test_missing_evidence_justified_context_minimums_fails_before_m2n(tmp_path: Path) -> None:
    request = _request(tmp_path, include_context_minimums=False).model_copy(
        update={"executable_path": str(tmp_path / "missing-m2n")}
    )
    audit = M2NChallengerAuditService().audit(request)

    assert audit.status == M2NChallengerStatus.ERROR
    assert "evidence-justified independent-context minimums" in audit.reason
    assert audit.runtime is None
