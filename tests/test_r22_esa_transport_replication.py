import numpy as np
import pytest

from astrograph.evaluation.external_dataset_contract import seal
from astrograph.evaluation.replication_r22 import (
    ESA_FAMILY,
    build_esa_preprocessed_input_receipt,
    esa_channel_payload_identity,
    evaluate_esa_transport_replication,
    generate_esa_transport_predictions,
    load_replication_plan,
    require_esa_no_superiority_contrast,
)


def _intake(plan):
    return seal({"schema_version":"astrograph-r22-new-holdout-intake-receipt.v1","status":"INTAKE_FROZEN_PRE_TRUTH",
        "dataset_id":"ESA_NEW","dataset_family":"ESA","source_artifact_kind":"FILE_BYTES","source_artifact_size_bytes":10,
        "source_artifact_sha256":"b"*64,"source_files":[],"operator_attests_pre_truth_inspection":True,
        "attestation_is_not_independent_proof":True,"known_source_collision_count":0,"known_member_collision_count":0,
        "r21_source_sha256":plan["r21_source_sha256"],"replication_plan_sha256":plan["receipt_sha256"]})


def _group_receipt(plan, intake_sha):
    spec = plan["families"][ESA_FAMILY]
    return seal({"schema_version":"astrograph-r22-new-holdout-group-admission.v1",
        "status":"ADMITTED_IDENTITY_NOVELTY_NOT_INDEPENDENCE_PROOF","dataset_id":"ESA_NEW","dataset_family":"ESA",
        "intake_receipt_sha256":intake_sha,"source_artifact_sha256":"b"*64,"evaluation_task_id":spec["evaluation_task_id"],
        "label_semantics_sha256":spec["label_semantics_sha256"],"target_universe_sha256":spec["target_universe_sha256"],
        "raw_group_ids":["event-new"],"raw_group_id_overlap_with_prior":[],"canonical_group_identity_sha256":"e"*64,"group_count":1,
        "statistical_independence_proven":False,"independence_claim_boundary":"identity only","replication_plan_sha256":plan["receipt_sha256"]})


def _payloads_and_artifacts(spec):
    times=np.array([1,2,3],dtype=np.int64)
    payloads={}
    for ch in spec["channels"]:
        ref=spec["reference"][ch]
        payloads[ch]={"time_ns":times,"value":np.array([ref["center"], ref["center"], ref["center"]+7*ref["scale"]]),
                      "observed":np.array([True,True,True]),"channel":ch}
    artifacts={ch:{"size":10,"sha256":f"{i+1:x}".rjust(64,"0"), **esa_channel_payload_identity(payloads[ch])}
               for i,ch in enumerate(spec["channels"])}
    return payloads, artifacts


def test_esa_selection_is_transport_only_and_superiority_claim_is_forbidden():
    plan = load_replication_plan()
    assert plan["families"][ESA_FAMILY]["comparison_semantics"] == "TRANSPORT_ONLY_IDENTICAL_BASELINE_AND_R21_SELECTION"
    with pytest.raises(ValueError, match="SUPERIORITY_CLAIM_FORBIDDEN"):
        require_esa_no_superiority_contrast({"superiority_contrast": True})


def test_esa_transport_uses_frozen_reference_and_threshold_without_truth_join():
    plan = load_replication_plan(); spec = plan["families"][ESA_FAMILY]
    intake = _intake(plan); groups = _group_receipt(plan, intake["receipt_sha256"])
    payloads, artifacts = _payloads_and_artifacts(spec)
    pre=build_esa_preprocessed_input_receipt(intake_receipt=intake,group_receipt=groups,channel_artifacts=artifacts,
        preprocessor_source_sha256="9"*64,transform=spec["required_preprocessing_transform"],evaluation_range_ns=[0,4],plan=plan)
    pred=generate_esa_transport_predictions(channel_payloads=payloads,group_receipt=groups,preprocessed_receipt=pre,plan=plan)
    assert pred["baseline_and_remediation_are_identical"] is True
    assert pred["superiority_contrast_allowed"] is False
    assert pred["threshold"] == 6
    assert pred["truth_joined"] is False
    assert pred["evaluation_range_ns"] == [0,4]


def test_esa_transport_truth_join_reports_event_metrics_without_superiority_delta():
    plan = load_replication_plan(); spec = plan["families"][ESA_FAMILY]
    intake = _intake(plan); groups = _group_receipt(plan, intake["receipt_sha256"])
    payloads, artifacts = _payloads_and_artifacts(spec)
    pre=build_esa_preprocessed_input_receipt(intake_receipt=intake,group_receipt=groups,channel_artifacts=artifacts,
        preprocessor_source_sha256="9"*64,transform=spec["required_preprocessing_transform"],evaluation_range_ns=[0,4],plan=plan)
    pred=generate_esa_transport_predictions(channel_payloads=payloads,group_receipt=groups,preprocessed_receipt=pre,plan=plan)
    report=evaluate_esa_transport_replication(pred,truth_events=[{"event_id":"event-new","category":"ANOMALY","start_ns":3,"end_ns":4}],
        full_range_ns=[0,4],group_receipt=groups,plan=plan)
    assert report["superiority_contrast_allowed"] is False
    assert report["baseline_vs_remediation_delta"] is None
    assert report["metrics"]["TP_events"] == 1
    assert report["retuning_performed"] is False


def test_esa_transport_rejects_payload_array_drift_from_preprocessed_receipt():
    plan = load_replication_plan(); spec = plan["families"][ESA_FAMILY]
    intake = _intake(plan); groups = _group_receipt(plan, intake["receipt_sha256"])
    payloads, artifacts = _payloads_and_artifacts(spec)
    pre=build_esa_preprocessed_input_receipt(intake_receipt=intake,group_receipt=groups,channel_artifacts=artifacts,
        preprocessor_source_sha256="9"*64,transform=spec["required_preprocessing_transform"],evaluation_range_ns=[0,4],plan=plan)
    payloads[spec["channels"][0]]["value"][0] += 1.0
    with pytest.raises(ValueError, match="ARRAY_IDENTITY_DRIFT"):
        generate_esa_transport_predictions(channel_payloads=payloads,group_receipt=groups,preprocessed_receipt=pre,plan=plan)
