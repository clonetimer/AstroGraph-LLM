import pytest

from astrograph.evaluation.real_data_failure_analysis import esa_frozen_range_audit, esa_stagewise_trace, require_probability_semantics


def test_range_proof_never_calls_model_and_short_event_grid_gap_is_explicit():
    ranges = esa_frozen_range_audit({"c": [9.0, 10.0, 11.0]}, {"c": {"center": 10.0, "scale": 1.0}}, 6)
    trace = esa_stagewise_trace(decision_points=73641, observable_decision_points=73641, alarm_points=0,
                                true_events=[{"id": 107, "annotated_duration_s": 9.912,
                                              "decision_timestamps_inside_annotation": 0}], range_evidence=ranges)
    assert trace["mechanism"] == "RESPONSE_BELOW_THRESHOLD"
    assert trace["per_anomaly_event_trace"]["107"]["mechanism"] == "AGGREGATION_SUPPRESSION"
    assert trace["inference_rerun"] is False


def test_uncalibrated_score_cannot_be_used_as_probability():
    with pytest.raises(ValueError, match="UNCALIBRATED"):
        require_probability_semantics("UNCALIBRATED_ANOMALY_SCORE")
