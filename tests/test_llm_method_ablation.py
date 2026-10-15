from __future__ import annotations

import json
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from tests.delivery_evidence import require_delivery_evidence

from astrograph.evaluation import (
    LLMMethodAblationRow, annotate_net_diagnostic_effect, summarize_llm_method_ablation,
)


def _row(case: str, arm: str, truth: str, prediction: str | None):
    return LLMMethodAblationRow(
        case_id=case,
        dataset_id="dataset",
        group_id=case,
        arm_id=arm,
        true_fault_id=truth,
        predicted_fault_id=prediction,
        abstained=prediction is None,
        latency_s=1.0,
        valid_tool_call_count=2 if arm == "full" else 0,
        observation_conditioned_replan_count=1 if arm == "full" else 0,
        substance_status="SUBSTANTIVE_METHOD_PATH" if arm == "full" else "NOT_EXECUTED",
    )


def test_ablation_summary_enforces_same_cases_and_reports_agent_metrics():
    rows = [
        _row("c1", "algorithm", "F1", "F1"),
        _row("c2", "algorithm", "F2", None),
        _row("c1", "full", "F1", "F1"),
        _row("c2", "full", "F2", "F2"),
    ]
    report = summarize_llm_method_ablation(
        rows, expected_arm_ids=["algorithm", "full"]
    )
    assert report["arm_metrics"]["algorithm"]["macro_f1"] == 0.5
    assert report["arm_metrics"]["full"]["macro_f1"] == 1.0
    assert report["arm_metrics"]["full"]["valid_tool_call_rate"] == 1.0
    assert report["arm_metrics"]["full"]["substantive_case_count"] == 2


def test_ablation_v2_reports_candidate_probe_and_net_gain_metrics():
    rows = [
        _row("c1", "A0_ALGORITHM_ONLY", "F1", None),
        _row("c2", "A0_ALGORITHM_ONLY", "F2", "F2"),
        LLMMethodAblationRow(
            case_id="c1", dataset_id="dataset", group_id="c1", arm_id="A2_CONTROLLED_FULL",
            true_fault_id="F1", predicted_fault_id="F1", abstained=False, latency_s=1.0,
            true_fault_in_candidate_set=True, probe_execution_count=1, new_observation_count=1,
            new_raw_observation_count=0, formal_evidence_generated_count=0, resolved_gap_count=0,
            diagnostic_contribution_status="DERIVED_OBSERVATION_ONLY",
        ),
        LLMMethodAblationRow(
            case_id="c2", dataset_id="dataset", group_id="c2", arm_id="A2_CONTROLLED_FULL",
            true_fault_id="F2", predicted_fault_id=None, abstained=True, latency_s=1.0,
            true_fault_in_candidate_set=False, probe_execution_count=0,
            diagnostic_contribution_status="NO_NEW_INFORMATION",
        ),
    ]
    rows = annotate_net_diagnostic_effect(rows)
    report = summarize_llm_method_ablation(
        rows, expected_arm_ids=["A0_ALGORITHM_ONLY", "A2_CONTROLLED_FULL"]
    )
    metrics = report["arm_metrics"]["A2_CONTROLLED_FULL"]
    assert report["schema_version"] == "astrograph-llm-method-ablation-report.v2"
    assert metrics["true_fault_in_candidate_set_rate"] == 0.5
    assert metrics["probe_execution_count"] == 1
    assert metrics["new_observation_count"] == 1
    assert metrics["new_raw_observation_count"] == 0
    assert metrics["corrected_case_count"] == 1
    assert metrics["induced_error_count"] == 1
    assert metrics["net_diagnostic_gain"] == 0
    assert metrics["diagnostic_contribution_status_counts"] == {
        "DERIVED_OBSERVATION_ONLY": 1, "NO_NEW_INFORMATION": 1
    }


def test_net_gain_annotation_requires_paired_baseline_and_never_changes_ground_truth():
    rows = [
        _row("c1", "A0_ALGORITHM_ONLY", "F1", "F1"),
        _row("c1", "challenger", "F1", None),
    ]
    annotated = annotate_net_diagnostic_effect(rows)
    assert annotated[0].corrected_case is False
    assert annotated[0].induced_error is False
    assert annotated[1].corrected_case is False
    assert annotated[1].induced_error is True
    assert annotated[1].true_fault_id == "F1"


def test_acceptance_campaign_template_declares_active_probe_ablation_without_fake_thresholds():
    root = Path(__file__).resolve().parents[1]
    campaign = json.loads((root / "configs/evaluation/llm_method_acceptance.v1.json").read_text(encoding="utf-8"))
    assert campaign["qualification_status"] == "TEMPLATE_NOT_PREREGISTERED"
    assert campaign["arm_order"][-1] == "A7_NO_ACTIVE_PROBE"
    assert campaign["promotion_thresholds"] is None
    assert campaign["case_manifest"] is None
    assert "net_diagnostic_gain" in campaign["required_metrics"]


def test_a7_runner_removes_only_active_probe_capabilities(tmp_path: Path):
    from scripts.experiments.run_llm_method_ablation import _arm_overrides
    override = _arm_overrides("A7_NO_ACTIVE_PROBE", tmp_path / "experience")
    capabilities = set(override["llm_diagnostic_capability_ids"] )
    assert "inspect_probe_catalog" not in capabilities
    assert "execute_diagnostic_probe" not in capabilities
    assert "inspect_evidence_status" in capabilities
    assert "inspect_deterministic_core" in capabilities


def test_active_telemetry_pilot_keeps_frozen_campaign_unchanged(tmp_path: Path):
    from scripts.experiments.run_llm_method_ablation import (
        ACTIVE_TELEMETRY_ARM_ORDER, ARM_ORDER, _arm_overrides,
    )
    assert ARM_ORDER[-1] == "A7_NO_ACTIVE_PROBE"
    assert ACTIVE_TELEMETRY_ARM_ORDER == [
        "A0_ALGORITHM_ONLY", "A1_LATE_LLM", "A2_CONTROLLED_FULL", "A8_ACTIVE_TELEMETRY"
    ]
    baseline = _arm_overrides("A2_CONTROLLED_FULL", tmp_path / "experience")
    active = _arm_overrides("A8_ACTIVE_TELEMETRY", tmp_path / "experience")
    assert active == {**baseline, "llm_active_telemetry_observation_enabled": True}


def test_fixed_probe_comparator_uses_same_runtime_overrides_and_no_truth(tmp_path: Path):
    from scripts.experiments.fixed_probe_action_policy import FixedProbeActionBackend
    from scripts.experiments.run_llm_method_ablation import _arm_overrides
    from astrograph.orchestration.llm_diagnostic_loop import DiagnosticActionProposal

    a8 = _arm_overrides("A8_ACTIVE_TELEMETRY", tmp_path, active_exploratory=True)
    a9 = _arm_overrides("A9_FIXED_PROBE_POLICY", tmp_path, active_exploratory=True)
    assert a9 == a8

    class Delegate:
        model_id = "fake-llm"
        calls = 0

        def generate_structured(self, *, agent_id, output_model, payload):
            self.calls += 1
            return output_model.model_validate({"answer": "llm-final"})

    class FinalOutput(BaseModel):
        answer: str

    delegate = Delegate()
    backend = FixedProbeActionBackend(delegate)
    assert backend.model_id == "fake-llm"
    base = {
        "allowed_capabilities": [
            {"capability_id": item} for item in (
                "inspect_candidate_fusion", "execute_diagnostic_probe",
                "inspect_deterministic_core", "synthesize_hypotheses",
            )
        ],
        "probe_index": [
            {"probe_id": "derived", "creates_new_raw_observation": False},
            {"probe_id": "raw", "creates_new_raw_observation": True},
        ],
    }
    first = backend.generate_structured(
        agent_id="diagnostic", output_model=DiagnosticActionProposal,
        payload={**base, "already_executed_capability_ids": [], "observation_index": []},
    )
    assert first.capability_id == "inspect_candidate_fusion"
    second = backend.generate_structured(
        agent_id="diagnostic", output_model=DiagnosticActionProposal,
        payload={**base, "already_executed_capability_ids": [first.capability_id],
                 "observation_index": [{"observation_id": "OBS-1"}]},
    )
    assert second.capability_id == "execute_diagnostic_probe"
    assert second.probe_id == "raw"
    assert second.cited_observation_ids == ["OBS-1"]
    third = backend.generate_structured(
        agent_id="diagnostic", output_model=DiagnosticActionProposal,
        payload={**base, "already_executed_capability_ids": [first.capability_id, second.capability_id],
                 "observation_index": [{"observation_id": "OBS-2"}]},
    )
    assert third.capability_id == "inspect_deterministic_core"
    final_action = backend.generate_structured(
        agent_id="diagnostic", output_model=DiagnosticActionProposal,
        payload={**base, "allowed_capabilities": [{"capability_id": "synthesize_hypotheses"}],
                 "already_executed_capability_ids": [first.capability_id, second.capability_id, third.capability_id],
                 "observation_index": [{"observation_id": "OBS-3"}]},
    )
    assert final_action.capability_id == "synthesize_hypotheses"
    assert backend.fixed_action_count == 4
    assert backend.generate_structured(
        agent_id="reasoner", output_model=FinalOutput, payload={},
    ).answer == "llm-final"
    assert delegate.calls == 1


def test_active_telemetry_pair_is_compared_directly_with_equal_budget_control():
    rows = [
        _row("c1", "A0_ALGORITHM_ONLY", "F1", None),
        _row("c1", "A2_CONTROLLED_FULL", "F1", None),
        _row("c1", "A8_ACTIVE_TELEMETRY", "F1", "F1"),
    ]
    report = summarize_llm_method_ablation(
        annotate_net_diagnostic_effect(rows),
        expected_arm_ids=["A0_ALGORITHM_ONLY", "A2_CONTROLLED_FULL", "A8_ACTIVE_TELEMETRY"],
    )
    pair = report["paired_comparisons"]["A2_CONTROLLED_FULL__vs__A8_ACTIVE_TELEMETRY"]
    assert pair["corrected_case_count"] == 1
    assert pair["induced_error_count"] == 0
    assert pair["effect_identifiability"] == "NO_ACTIVE_TELEMETRY_EXPOSURE"


def test_nominal_control_denominator_includes_review_and_abstention():
    from scripts.experiments.run_llm_method_ablation import (
        _nominal_control_metrics, _select_nominal_controls,
    )
    controls = _select_nominal_controls([
        SimpleNamespace(sample_id="b", split="test", diagnostic_class="nominal"),
        SimpleNamespace(sample_id="a", split="test", diagnostic_class="nominal"),
        SimpleNamespace(sample_id="z", split="train", diagnostic_class="nominal"),
    ], "diagnostic_class")
    assert [item.sample_id for item in controls] == ["a", "b"]
    indexed = _select_nominal_controls([
        SimpleNamespace(sample_id="mc04", sample_index=4, split="test", diagnostic_class="nominal"),
        SimpleNamespace(sample_id="mc09", sample_index=9, split="test", diagnostic_class="nominal"),
    ], "diagnostic_class", sample_index=9, limit=1)
    assert [item.sample_id for item in indexed] == ["mc09"]
    rows = [
        {"case_id": "a", "arm_id": "A2", "accepted_fault_id": None, "requires_human_review": True},
        {"case_id": "b", "arm_id": "A2", "accepted_fault_id": None, "requires_human_review": False},
        {"case_id": "a", "arm_id": "A8", "accepted_fault_id": "F1", "requires_human_review": False,
         "shadow_revision_available": True, "shadow_candidate_set_changed": True,
         "active_telemetry_attributable": True},
        {"case_id": "b", "arm_id": "A8", "accepted_fault_id": None, "requires_human_review": False},
    ]
    metrics = _nominal_control_metrics(rows, arm_order=["A2", "A8"])
    assert metrics["A2"]["case_count"] == 2
    assert metrics["A2"]["human_review_count"] == 1
    assert metrics["A2"]["formal_false_alarm_rate"] == 0.0
    assert metrics["A8"]["formal_false_alarm_rate"] == 0.5
    assert metrics["A8"]["shadow_revision_available_case_count"] == 1
    assert metrics["A8"]["shadow_candidate_set_changed_case_count"] == 1
    assert metrics["A8"]["active_telemetry_attributable_case_count"] == 1


def test_exploratory_a2_a8_share_shadow_schema_and_step_budget(tmp_path):
    from scripts.experiments.run_llm_method_ablation import _arm_overrides
    a2 = _arm_overrides("A2_CONTROLLED_FULL", tmp_path, active_exploratory=True)
    a8 = _arm_overrides("A8_ACTIVE_TELEMETRY", tmp_path, active_exploratory=True)
    assert a2["llm_diagnostic_loop_max_steps"] == a8["llm_diagnostic_loop_max_steps"] == 4
    assert a2["llm_shadow_candidate_revision_enabled"] is True
    assert a8["llm_shadow_candidate_revision_enabled"] is True
    assert "llm_active_telemetry_observation_enabled" not in a2
    assert a8["llm_active_telemetry_observation_enabled"] is True
    assert "llm_shadow_candidate_revision_enabled" not in _arm_overrides(
        "A2_CONTROLLED_FULL", tmp_path
    )


def test_exploratory_fault_sample_selection_is_stable_without_changing_frozen_case_rule():
    from scripts.experiments.run_llm_method_ablation import _select_pilot_fault_sample
    case = {"case_id": "pilot"}
    samples = [SimpleNamespace(sample_id="z"), SimpleNamespace(sample_id="a")]
    assert _select_pilot_fault_sample(samples, case=case, exploratory=True).sample_id == "a"
    with pytest.raises(ValueError, match="frozen case must resolve"):
        _select_pilot_fault_sample(samples, case=case, exploratory=False)


def test_lmstudio_exploration_records_adapted_provider_without_changing_vllm_profile():
    from scripts.experiments import run_llm_method_ablation as runner
    profile = Path(__file__).resolve().parents[1] / "configs/evaluation/llm_method_confirmatory_inference.v1.json"
    vllm = runner._inference_snapshot(
        url="http://127.0.0.1:8000/v1", model="Qwen3.5-9B",
        inference_profile=profile, provider="vllm",
    )
    studio = runner._inference_snapshot(
        url="http://127.0.0.1:1234/v1", model="qwen/qwen3.5-9b",
        inference_profile=profile, provider="lmstudio",
    )
    assert vllm["provider"] == "vllm"
    assert "profile_adaptation" not in vllm
    assert studio["provider"] == "lmstudio"
    assert studio["profile_adaptation"] == "EXPLORATORY_PROVIDER_OVERRIDE_NOT_CONFIRMATORY"
    assert studio["inference_config_sha256"] != vllm["inference_config_sha256"]
    backend = runner._backend(
        "http://127.0.0.1:1234/v1", "qwen/qwen3.5-9b", profile, provider="lmstudio",
    )
    assert backend.config.provider == "lmstudio"
    assert backend._strategies()[0] == "json_schema"


def test_shadow_candidate_report_separates_exposure_from_revision():
    base = [
        _row("c1", "A0_ALGORITHM_ONLY", "F1", None),
        _row("c1", "A2_CONTROLLED_FULL", "F1", None),
        LLMMethodAblationRow(
            case_id="c1", dataset_id="dataset", group_id="c1", arm_id="A8_ACTIVE_TELEMETRY",
            true_fault_id="F1", predicted_fault_id=None, abstained=True, latency_s=2.0,
            new_raw_observation_count=1, shadow_revision_available=True,
            shadow_candidate_set_changed=True, active_telemetry_attributable=True,
            shadow_abstain_recommended=False, true_fault_in_shadow_candidate_set=True,
        ),
    ]
    report = summarize_llm_method_ablation(
        annotate_net_diagnostic_effect(base),
        expected_arm_ids=["A0_ALGORITHM_ONLY", "A2_CONTROLLED_FULL", "A8_ACTIVE_TELEMETRY"],
    )
    metrics = report["arm_metrics"]["A8_ACTIVE_TELEMETRY"]
    assert metrics["active_telemetry_attributable_case_count"] == 1
    assert metrics["true_fault_in_shadow_candidate_set_rate"] == 1.0
    pair = report["paired_comparisons"]["A2_CONTROLLED_FULL__vs__A8_ACTIVE_TELEMETRY"]
    assert pair["active_telemetry_exposed_case_count"] == 1
    assert pair["shadow_candidate_recall_delta"] is None


def test_active_experiment_preflight_checks_datasets_and_served_model(tmp_path, monkeypatch):
    from scripts.experiments import run_llm_method_ablation as runner
    profile = tmp_path / "inference.json"
    profile.write_text("{}", encoding="utf-8")
    def load_rows(dataset):
        class_attr = "diagnostic_class" if dataset == "adcs" else "condition_id"
        nominal = SimpleNamespace(sample_id=f"{dataset}-healthy", split="test", telemetry_path=profile)
        fault = SimpleNamespace(sample_id=f"{dataset}-fault", split="test", telemetry_path=profile)
        setattr(nominal, class_attr, "nominal")
        setattr(fault, class_attr, next(
            case["dataset_class"] for case in runner.CASES if case["dataset"] == dataset
        ))
        return [nominal, fault], "profile", dataset, class_attr
    monkeypatch.setattr(runner, "_load_rows", load_rows)
    monkeypatch.setattr(runner, "_sample_pair", lambda fault, rows, dataset: (fault, rows[0]))

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda *_args, **_kwargs: Response(
        b'{"data":[{"id":"fake-model"}]}'
    ))
    receipt = runner._preflight_active_experiment(
        vllm_url="http://127.0.0.1:8000/v1", vllm_model="fake-model", inference_profile=profile,
    )
    assert receipt["datasets"]["adcs"]["test_nominal_count"] == 1
    assert receipt["model_id"] == "fake-model"
    with pytest.raises(ValueError, match="healthy test controls"):
        runner._preflight_active_experiment(
            vllm_url="http://127.0.0.1:8000/v1", vllm_model="fake-model",
            inference_profile=profile, sample_index=9,
        )
    with pytest.raises(ValueError, match="not served"):
        runner._preflight_active_experiment(
            vllm_url="http://127.0.0.1:8000/v1", vllm_model="missing", inference_profile=profile,
        )


def test_experiment_refuses_to_overwrite_existing_output(tmp_path, monkeypatch):
    from scripts.experiments import run_llm_method_ablation as runner
    output = tmp_path / "existing-run"
    output.mkdir()
    sentinel = output / "report.json"
    sentinel.write_text("owner data", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "run_llm_method_ablation.py", "--output-dir", str(output),
        "--active-telemetry-exploratory",
    ])
    with pytest.raises(SystemExit) as raised:
        runner.main()
    assert raised.value.code == 2
    assert sentinel.read_text(encoding="utf-8") == "owner data"


def test_ablation_rejects_case_denominator_changes():
    rows = [
        _row("c1", "algorithm", "F1", "F1"),
        _row("c2", "full", "F2", "F2"),
    ]
    with pytest.raises(ValueError, match="same case IDs"):
        summarize_llm_method_ablation(rows, expected_arm_ids=["algorithm", "full"])


def test_live_pilot_preserves_failed_attempt_and_disclosed_correction():
    project_root = Path(__file__).resolve().parents[1]
    original_path = require_delivery_evidence(
        project_root, "governance/evidence/research/llm_method_ablation_20260903/report.json"
    )
    require_delivery_evidence(
        project_root, "governance/evidence/research/llm_method_ablation_20260903/report_after_constraint_fix.json"
    )
    require_delivery_evidence(
        project_root, "governance/evidence/research/llm_method_ablation_20260903/posthoc_correction_protocol.json"
    )
    root = original_path.parent
    original = json.loads(original_path.read_text(encoding="utf-8"))
    corrected = json.loads(
        (root / "report_after_constraint_fix.json").read_text(encoding="utf-8")
    )
    protocol = json.loads(
        (root / "posthoc_correction_protocol.json").read_text(encoding="utf-8")
    )

    assert original["arm_metrics"]["A2_CONTROLLED_FULL"]["substantive_case_count"] == 0
    assert corrected["case_ids"] == original["case_ids"]
    assert corrected["arm_order"] == original["arm_order"]
    assert corrected["arm_metrics"]["A2_CONTROLLED_FULL"]["substantive_case_count"] == 2
    assert corrected["arm_metrics"]["A2_CONTROLLED_FULL"]["accuracy"] == 1.0
    assert corrected["arm_metrics"]["A0_ALGORITHM_ONLY"]["accuracy"] == 1.0
    assert corrected["arm_metrics"]["A6_NO_OBSERVATION_FEEDBACK"]["substantive_case_count"] == 0
    assert protocol["status"] == "FROZEN_AFTER_FAILED_PILOT_BEFORE_CORRECTION_EXECUTION"


def test_probe_specific_pair_is_reported_separately_from_full_stack_baseline():
    rows = [
        LLMMethodAblationRow(
            case_id="c1", dataset_id="dataset", group_id="g1", arm_id="A7_NO_ACTIVE_PROBE",
            true_fault_id="F1", predicted_fault_id=None, abstained=True, latency_s=1.0,
        ),
        LLMMethodAblationRow(
            case_id="c1", dataset_id="dataset", group_id="g1", arm_id="A2_CONTROLLED_FULL",
            true_fault_id="F1", predicted_fault_id="F1", abstained=False, latency_s=1.0,
            probe_execution_count=1,
        ),
    ]
    report = summarize_llm_method_ablation(
        rows, expected_arm_ids=["A7_NO_ACTIVE_PROBE", "A2_CONTROLLED_FULL"]
    )
    paired = report["paired_comparisons"]["A7_NO_ACTIVE_PROBE__vs__A2_CONTROLLED_FULL"]
    assert paired["corrected_case_count"] == 1
    assert paired["probe_exposed_group_count"] == 1
    assert paired["inferential_unit"] == "group_id"
