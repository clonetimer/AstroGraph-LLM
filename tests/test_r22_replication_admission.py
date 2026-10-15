import json
from pathlib import Path

import pytest

from astrograph.evaluation.external_dataset_contract import seal
from astrograph.evaluation.replication_r22 import (
    build_new_holdout_intake_receipt,
    admit_new_holdout_groups,
    verify_new_holdout_admission,
    load_replication_plan,
    OPS_FAMILY,
)


def _custom_plan(tmp_path: Path, known_hashes=(), member_hashes=()):
    base = load_replication_plan()
    payload = {k: v for k, v in base.items() if k != "receipt_sha256"}
    payload["known_source_artifact_sha256s"] = list(known_hashes)
    payload["known_member_sha256s_by_family"] = {**payload["known_member_sha256s_by_family"], OPS_FAMILY: list(member_hashes)}
    return seal(payload)


def test_plan_is_frozen_and_bound_to_current_executor_source():
    plan = load_replication_plan()
    assert plan["frozen_before_new_holdout"] is True
    assert plan["status"] == "READY_FOR_NEW_HOLDOUT_BLOCKED_DATASET"
    assert plan["families"][OPS_FAMILY]["retuning_allowed"] is False


def test_intake_requires_pretruth_attestation_and_novel_bytes(tmp_path):
    source = tmp_path / "new.bin"
    source.write_bytes(b"novel-r22")
    with pytest.raises(ValueError, match="PRE_TRUTH"):
        build_new_holdout_intake_receipt(source, dataset_id="NEW", dataset_family=OPS_FAMILY,
                                         operator_attests_pre_truth_inspection=False,
                                         plan=_custom_plan(tmp_path))
    intake = build_new_holdout_intake_receipt(source, dataset_id="NEW", dataset_family=OPS_FAMILY,
                                              operator_attests_pre_truth_inspection=True,
                                              plan=_custom_plan(tmp_path))
    assert intake["status"] == "INTAKE_FROZEN_PRE_TRUTH"
    blocked_plan = _custom_plan(tmp_path, known_hashes=[intake["source_artifact_sha256"]])
    with pytest.raises(ValueError, match="KNOWN_SOURCE_ARTIFACT"):
        build_new_holdout_intake_receipt(source, dataset_id="NEW2", dataset_family=OPS_FAMILY,
                                         operator_attests_pre_truth_inspection=True, plan=blocked_plan)


def test_group_admission_binds_task_without_claiming_statistical_independence(tmp_path):
    source = tmp_path / "new.bin"; source.write_bytes(b"new-holdout")
    plan = _custom_plan(tmp_path)
    intake = build_new_holdout_intake_receipt(source, dataset_id="OPS_NEW_001", dataset_family=OPS_FAMILY,
                                              operator_attests_pre_truth_inspection=True, plan=plan)
    task = plan["families"][OPS_FAMILY]
    groups = admit_new_holdout_groups(intake, raw_group_ids=["g1", "g2"],
        evaluation_task_id=task["evaluation_task_id"], label_semantics_sha256=task["label_semantics_sha256"],
        target_universe_sha256=task["target_universe_sha256"], plan=plan)
    assert groups["group_count"] == 2
    assert groups["statistical_independence_proven"] is False
    verify_new_holdout_admission(source, intake, groups)
    source.write_bytes(b"changed-holdout")
    with pytest.raises(ValueError, match="SOURCE_DRIFT"):
        verify_new_holdout_admission(source, intake, groups)
