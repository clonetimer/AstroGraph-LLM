import json

from astrograph.evaluation.data_exposure import contamination_audit, registry
from astrograph.evaluation.development_remediation import load_preregistration


def test_locked_group_in_development_receipt_invalidates_study():
    locked = registry()["datasets"]["OPS_SAT"]["group_ids_by_partition"]["EXTERNAL_VALIDATION"][0]
    receipt = {"dataset_id": "OPS_SAT", "partition": "MODEL_TRAIN", "purpose": "FIT", "group_ids": [locked]}
    assert contamination_audit([receipt], load_preregistration())["status"] == "INVALID_SEEN_EVALUATION_CONTAMINATION"


def test_new_external_prediction_flag_invalidates_study():
    receipt = {"dataset_id": "OPS_SAT", "partition": "MODEL_TRAIN", "purpose": "FIT",
               "external_predictions_generated": True}
    audit = contamination_audit([receipt], load_preregistration())
    assert any(v["reason"] == "R21_NEW_EXTERNAL_PREDICTION_OR_QUALIFICATION_FORBIDDEN" for v in audit["violations"])


def test_preregistration_change_invalidates_study():
    plan = json.loads(json.dumps(load_preregistration()))
    plan["OPS"]["maximum_trials"] = 99
    assert contamination_audit([], plan)["status"] == "INVALID_SEEN_EVALUATION_CONTAMINATION"
