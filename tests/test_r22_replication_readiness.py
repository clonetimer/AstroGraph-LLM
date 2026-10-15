from astrograph.evaluation.external_dataset_contract import seal
from astrograph.evaluation.replication_r22 import build_r22_readiness, load_replication_plan


def test_r22_is_ready_protocol_but_blocked_without_new_holdout():
    receipt = build_r22_readiness()
    assert receipt["status"] == "READY_FOR_NEW_HOLDOUT_BLOCKED_DATASET"
    assert set(receipt["blockers"]) == {"NEW_HOLDOUT_NOT_ADMITTED:OPS-SAT", "NEW_HOLDOUT_NOT_ADMITTED:ESA"}
    assert receipt["performance_result_generated"] is False
    assert receipt["new_external_validation_receipt_generated"] is False


def test_r22_track_specific_readiness_does_not_require_unrelated_family():
    plan = load_replication_plan()
    admission = seal({
        "schema_version": "astrograph-r22-new-holdout-group-admission.v1",
        "status": "ADMITTED_IDENTITY_NOVELTY_NOT_INDEPENDENCE_PROOF",
        "dataset_id": "OPS_NEW",
        "dataset_family": "OPS-SAT",
        "intake_receipt_sha256": "a" * 64,
        "source_artifact_sha256": "b" * 64,
        "evaluation_task_id": plan["families"]["OPS-SAT"]["evaluation_task_id"],
        "label_semantics_sha256": plan["families"]["OPS-SAT"]["label_semantics_sha256"],
        "target_universe_sha256": plan["families"]["OPS-SAT"]["target_universe_sha256"],
        "raw_group_ids": ["novel-1"],
        "raw_group_id_overlap_with_prior": [],
        "canonical_group_identity_sha256": "c" * 64,
        "group_count": 1,
        "statistical_independence_proven": False,
        "independence_claim_boundary": "identity only",
        "replication_plan_sha256": plan["receipt_sha256"],
    })
    receipt = build_r22_readiness(
        holdout_admissions={"OPS-SAT": admission},
        required_families=["OPS-SAT"],
        plan=plan,
    )
    assert receipt["status"] == "READY"
    assert receipt["required_families"] == ["OPS-SAT"]
    assert receipt["blockers"] == []
