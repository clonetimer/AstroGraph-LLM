from __future__ import annotations

from astrograph.evaluation.ops_sat_adapter import FEATURE_FIELDS
from scripts.experiments import run_ops_sat_feature_llm_comparator as comparator
from scripts.experiments.run_ops_sat_compact_bundle_comparator import CompactBundleRevision


class _FakeBackend:
    def __init__(self, revision: CompactBundleRevision) -> None:
        self.revision = revision
        self.call_history = [{
            "output_budget": {
                "server_usage": {"prompt_tokens": 11, "completion_tokens": 2},
            },
        }]
        self.payload = None

    def generate_structured(self, **kwargs):
        self.payload = kwargs["payload"]
        return self.revision


def _feature_row() -> dict:
    row = {field: float(index + 1) for index, field in enumerate(FEATURE_FIELDS)}
    row.update({"segment": "39", "channel": "CADC0872", "sampling": "1"})
    return row


def test_feature_llm_arm_binds_calibration_and_requires_active_citation(monkeypatch) -> None:
    backend = _FakeBackend(CompactBundleRevision(
        decision="NOMINAL", citation_scope="ACTIVE_BUNDLE_ONLY",
    ))
    monkeypatch.setattr(comparator, "_backend", lambda *args, **kwargs: backend)
    arm = comparator.run_feature_llm_arm(
        feature_row=_feature_row(),
        feature_source_sha256="a" * 64,
        a13_case={"length_free_policy": {
            "out_of_fold_probability": 0.1,
            "fold_selected_threshold": 0.5,
            "fold_selected_candidate_id": "ALL_LENGTH_FREE_FEATURES",
            "predicted_state": "NOMINAL",
        }},
        frozen_prediction={
            "score": 0.2,
            "threshold": 0.7,
            "predicted_anomaly": False,
            "score_semantics": "frozen-test-score",
        },
        a13_report_sha256="b" * 64,
        url="http://unused.invalid/v1",
        model="test-model",
    )
    assert arm["candidate_set_changed"] is True
    assert arm["active_observation_cited"] is True
    assert arm["active_observation_attributable"] is True
    assert arm["validated_candidate_states"] == ["NOMINAL"]
    assert arm["observation"]["calibration"]["a13_report_sha256"] == "b" * 64
    assert "offline_truth" not in backend.payload
    assert "anomaly" not in backend.payload
    assert "offline_truth" not in backend.payload["active_derived_feature_observation"]
    assert "anomaly" not in backend.payload["active_derived_feature_observation"]
    assert set(backend.payload["active_derived_feature_observation"]["values"]) == set(
        comparator.measure_ops_sat_length_free_feature_bundle(
            _feature_row(), source_sha256="a" * 64,
        )["values"]
    )


def test_a13_paired_summary_uses_only_selected_rows() -> None:
    rows = [{"group_id": "healthy", "offline_truth": "NOMINAL"}]
    a13 = {
        "healthy": {"length_free_policy": {"predicted_state": "NOMINAL"}},
        "unselected": {"length_free_policy": {"predicted_state": "ANOMALOUS"}},
    }
    summary = comparator._a13_paired_summary(rows, a13)
    assert summary["case_count"] == 1
    assert summary["accuracy"] == 1.0
    assert summary["healthy_false_alarm_count"] == 0
    assert summary["balanced_accuracy"] is None
