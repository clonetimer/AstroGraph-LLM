from datetime import datetime, timedelta, timezone

from astrograph.evaluation.external_dataset_contract import seal
from astrograph.evaluation.replication_r22 import (
    load_replication_plan,
    ops_r20_frozen_baseline_prediction,
    ops_r21_frozen_remediation_prediction,
    evaluate_ops_replication,
)


def _points(n=12, channel="CADC0888", segment="novel-g"):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        {"channel": channel, "timestamp": (start + timedelta(seconds=5*i)).isoformat(),
         "value": 0.8 + 0.01*i, "sampling": "5", "segment": segment}
        for i in range(n)
    ]


def test_short_segment_remediation_scores_where_r20_baseline_blocks():
    plan = load_replication_plan()
    points = _points()
    baseline = ops_r20_frozen_baseline_prediction(points, plan)
    remediation = ops_r21_frozen_remediation_prediction(points, plan)
    assert baseline["status"] == "BLOCKED"
    assert remediation["status"] == "SCORED"
    assert remediation["kind"] == "short"
    assert remediation["threshold"] == 12


def test_ops_evaluation_keeps_coverage_gain_distinct_from_paired_scored_cases():
    groups = seal({"schema_version":"astrograph-r22-new-holdout-group-admission.v1",
                   "status":"ADMITTED_IDENTITY_NOVELTY_NOT_INDEPENDENCE_PROOF","dataset_id":"D",
                   "dataset_family":"OPS-SAT","intake_receipt_sha256":"a"*64,"source_artifact_sha256":"b"*64,
                   "evaluation_task_id":"OPS","label_semantics_sha256":"c"*64,"target_universe_sha256":"d"*64,
                   "raw_group_ids":["g1","g2"],"raw_group_id_overlap_with_prior":[],
                   "canonical_group_identity_sha256":"e"*64,"group_count":2,"statistical_independence_proven":False,
                   "independence_claim_boundary":"identity only","replication_plan_sha256":"f"*64})
    predictions = seal({"schema_version":"astrograph-r22-ops-replication-predictions.v1","status":"FROZEN_BEFORE_TRUTH_JOIN",
                        "dataset_id":"D","group_admission_sha256":groups["receipt_sha256"],"source_artifact_sha256":"b"*64,
                        "replication_plan_sha256":"f"*64,"baseline_implementation_sha256":"1"*64,
                        "remediation_implementation_sha256":"2"*64,
                        "predictions":{"baseline":{"g1":{"status":"BLOCKED"},"g2":{"status":"SCORED","predicted_anomaly":0}},
                                       "remediation":{"g1":{"status":"SCORED","predicted_anomaly":1},"g2":{"status":"SCORED","predicted_anomaly":1}}},
                        "truth_joined":False,"external_validation_receipt_generated":False})
    report = evaluate_ops_replication(predictions, {"g1":1,"g2":0}, groups)
    assert report["baseline"]["coverage"] == 0.5
    assert report["remediation"]["coverage"] == 1.0
    assert report["paired_descriptive"]["coverage_gain_correct"] == 1
    assert report["paired_descriptive"]["induced_error"] == 1
    assert report["inferential_p_value"] is None
