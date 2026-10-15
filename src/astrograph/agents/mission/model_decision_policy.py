"""Fail-closed calibration/OOD policy for generic learned model outputs."""
from __future__ import annotations

from typing import Any


def assess_model_decision_eligibility(execution: dict[str, Any], *, max_ood_score: float = 0.5) -> dict[str, Any]:
    if execution.get("status") != "SUCCEEDED":
        return {"status":"NOT_EXECUTED","decision_eligible":False,"reason":"MODEL_NOT_EXECUTED"}
    prediction=execution.get("prediction",{}) or {}
    calibrated=bool(prediction.get("calibrated_probabilities",False))
    calibration_id=prediction.get("calibration_id")
    ood=prediction.get("out_of_distribution_score")
    if not calibrated or not calibration_id:
        return {"status":"ABSTAIN","decision_eligible":False,"reason":"UNCALIBRATED_MODEL_OUTPUT",
                "calibrated":False,"ood_assessed":ood is not None}
    if ood is None:
        return {"status":"ABSTAIN","decision_eligible":False,"reason":"OOD_NOT_ASSESSED",
                "calibrated":True,"calibration_id":calibration_id,"ood_assessed":False}
    if float(ood)>max_ood_score:
        return {"status":"UNKNOWN_OOD","decision_eligible":False,"reason":"OOD_SCORE_ABOVE_THRESHOLD",
                "calibrated":True,"calibration_id":calibration_id,"ood_assessed":True,
                "out_of_distribution_score":float(ood),"max_ood_score":max_ood_score}
    return {"status":"ELIGIBLE","decision_eligible":True,"reason":"CALIBRATED_AND_IN_DISTRIBUTION",
            "calibrated":True,"calibration_id":calibration_id,"ood_assessed":True,
            "out_of_distribution_score":float(ood),"max_ood_score":max_ood_score}


__all__=["assess_model_decision_eligibility"]
