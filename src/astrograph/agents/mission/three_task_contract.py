"""Normalized three-task result contract for Mission model execution."""
from __future__ import annotations

from typing import Any


def normalize_three_task_execution(execution: dict[str, Any]) -> dict[str, Any]:
    if execution.get("status") != "SUCCEEDED":
        return {
            "status": "NOT_EXECUTED",
            "detection": {"status": "NOT_EXECUTED"},
            "localization": {"status": "NOT_EXECUTED"},
            "classification": {"status": "NOT_EXECUTED"},
            "formal_diagnostic_vote_allowed": False,
        }
    heads = execution.get("heads", {}) or {}
    det = heads.get("detection", {}) or {}
    loc = heads.get("localization", {}) or {}
    cls = heads.get("classification", {}) or {}
    prediction = execution.get("prediction", {}) or {}
    calibrated = bool(prediction.get("calibrated_probabilities", False))
    ood = prediction.get("out_of_distribution_score")
    return {
        "status": "EXECUTED",
        "detection": {
            "status": "EXECUTED",
            "detected": bool(det.get("detected", False)),
            "anomaly_score": det.get("anomaly_probability"),
            "threshold": det.get("threshold"),
            "score_semantics": "CALIBRATED_PROBABILITY" if calibrated else "UNCALIBRATED_SOFTMAX",
        },
        "localization": {
            "status": "EXECUTED_AFFECTED_COMPONENT_ONLY",
            "predicted_component": loc.get("predicted_component"),
            "classes": list(loc.get("classes", [])),
            "scores": list(loc.get("probabilities", [])),
            "root_cause_supported": bool(loc.get("root_cause_supported", False)),
        },
        "classification": {
            "status": "EXECUTED_CALIBRATED" if calibrated else "EXECUTED_SHADOW_UNCALIBRATED",
            "predicted_fault": cls.get("predicted_fault"),
            "classes": list(cls.get("classes", [])),
            "scores": list(cls.get("probabilities", [])),
            "calibrated": calibrated,
        },
        "out_of_distribution_score": ood,
        # Stage 85 policy refines this; default remains fail closed.
        "formal_diagnostic_vote_allowed": False,
        "claim_boundary": (
            "Detection/localization/classification heads were executed through strict admission. "
            "Localization is affected-component output, not root-cause proof; uncalibrated classification is shadow evidence."
        ),
    }


__all__ = ["normalize_three_task_execution"]
