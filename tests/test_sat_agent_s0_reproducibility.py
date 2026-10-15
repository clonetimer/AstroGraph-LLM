from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]
TASK_MATRIX = (
    ROOT
    / "governance"
    / "evidence"
    / "0.7.1-extension"
    / "p018_sat_agent_task_suitability_matrix.v1.json"
)
SPLIT_PREREGISTRATION = (
    ROOT
    / "protocols"
    / "acceptance"
    / "p018_sat_agent_s1_grouped_split_preregistration.json"
)


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENV_AUDIT = _load(
    "audit_sat_agent_source_environment",
    "scripts/datasets/audit_sat_agent_source_environment.py",
)
REPLAY = _load(
    "replay_sat_agent_test_only_smoke",
    "scripts/datasets/replay_sat_agent_test_only_smoke.py",
)
LINEAGE = _load(
    "audit_sat_agent_formal_scope_lineage",
    "scripts/datasets/audit_sat_agent_formal_scope_lineage.py",
)
PROFILE_GAPS = _load(
    "audit_sat_agent_semantic_profile_gaps",
    "scripts/datasets/audit_sat_agent_semantic_profile_gaps.py",
)


def test_tree_record_is_content_bound_and_does_not_emit_paths(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    first = ENV_AUDIT._tree_record(tmp_path, ("src",))
    assert first["file_count"] == 1
    assert first["individual_paths_emitted"] is False
    assert "a.py" not in json.dumps(first)

    (tmp_path / "src" / "a.py").write_text("x = 2\n", encoding="utf-8")
    second = ENV_AUDIT._tree_record(tmp_path, ("src",))
    assert first["manifest_sha256"] != second["manifest_sha256"]


def test_tree_record_refuses_a_member_outside_injected_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        ENV_AUDIT._tree_record(root, ("../outside.py",))


def test_all_replay_cases_cover_seven_archetypes_with_truth_blind_allowlists() -> None:
    assert len(REPLAY.REPLAY_CASES) == 7
    assert len({item["component_archetype_id"] for item in REPLAY.REPLAY_CASES}) == 7
    for item in REPLAY.REPLAY_CASES:
        REPLAY._validate_allowlist(item["public_allowlist"])
        assert item["public_allowlist"][0] == "time_s"


def test_public_projection_is_deterministic_and_rejects_bad_time() -> None:
    fields = ("time_s", "physical.signal")
    rows = [
        {"time_s": 0.0, "physical.signal": 1.0, "label.fault": "ignored"},
        {"time_s": 1.0, "physical.signal": 2.0, "label.fault": "ignored"},
    ]
    first = REPLAY._project_public_rows(rows, fields)
    second = REPLAY._project_public_rows(rows, fields)
    assert REPLAY._canonical_sha256(first) == REPLAY._canonical_sha256(second)
    with pytest.raises(ValueError, match="strictly increasing"):
        REPLAY._project_public_rows(list(reversed(rows)), fields)


def test_replay_refuses_seed_and_never_requests_dataset_write() -> None:
    calls = []

    def instantiate(_: str):
        return {
            "capability_id": "component.fake.v1",
            "simulation": {"seed": 1},
        }

    def run_task(*args, **kwargs):  # pragma: no cover - must not be reached
        calls.append((args, kwargs))
        return SimpleNamespace(summary={"status": "complete"}, trace_rows=[])

    case = {
        "template_id": "test-only",
        "capability_id": "component.fake.v1",
        "public_allowlist": ("time_s", "physical.signal"),
    }
    with pytest.raises(ValueError, match="contains a seed"):
        REPLAY._run_once(case, instantiate, lambda value: value, run_task)
    assert calls == []


def test_capability_binding_supports_current_nested_sat_agent_contract() -> None:
    assert REPLAY._capability_id({"capability_id": "top"}) == "top"
    assert REPLAY._capability_id({"model": {"capability_id": "nested"}}) == "nested"
    assert REPLAY._capability_id({"model": {}}) is None


def test_replay_requests_in_memory_execution_only() -> None:
    calls = []

    def instantiate(_: str):
        return {"capability_id": "component.fake.v1", "simulation": {}}

    def run_task(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            summary={"status": "complete"},
            trace_rows=[
                {"time_s": 0.0, "physical.signal": 1.0},
                {"time_s": 1.0, "physical.signal": 2.0},
            ],
        )

    case = {
        "template_id": "test-only",
        "capability_id": "component.fake.v1",
        "public_allowlist": ("time_s", "physical.signal"),
    }
    digest, count, _ = REPLAY._run_once(
        case, instantiate, lambda value: value, run_task
    )
    assert len(digest) == 64
    assert count == 2
    assert calls == [{"task_spec": instantiate("x"), "write_dataset": False}]


def test_task_suitability_does_not_launder_structural_pass_into_diagnosis_pass() -> None:
    matrix = json.loads(TASK_MATRIX.read_text(encoding="utf-8"))
    assert matrix["source_count"] == 1
    assert matrix["structural_evidence"]["public_semantic_projection_denominator"] == 237
    assert matrix["structural_evidence"]["public_semantic_projection_passed"] == 237
    statuses = {item["task_id"]: item["current_status"] for item in matrix["task_matrix"]}
    assert statuses["DATA_QUALITY"] == "STRUCTURAL_INPUT_CONTRACT_ELIGIBLE"
    assert all(
        not status.endswith("ELIGIBLE")
        for task, status in statuses.items()
        if task != "DATA_QUALITY"
    )
    assert matrix["decision"]["diagnosis_task_data_qualification"] == "BLOCKED"
    assert matrix["decision"]["s0_overall"] == "IN_PROGRESS"
    assert matrix["execution_receipt"]["formal_denominator_changed"] is False


def test_grouped_split_is_design_only_and_covers_every_s0_axis() -> None:
    protocol = json.loads(SPLIT_PREREGISTRATION.read_text(encoding="utf-8"))
    assert protocol["source_count"] == 1
    assert protocol["authority_boundary"]["split_materialization_authorized"] is False
    assert protocol["authority_boundary"]["model_fit_authorized"] is False
    receipt = protocol["materialization_receipt"]
    assert receipt["episode_assignment_count"] == 0
    assert receipt["split_count"] == 0
    assert receipt["development_seed"] is None
    assert receipt["calibration_seed"] is None
    assert receipt["technical_validation_seed"] is None
    axes = {item["axis_id"] for item in protocol["preregistered_holdout_axes"]}
    assert axes == {
        "LEAVE_ONE_OBJECT_INSTANCE_OUT",
        "LEAVE_ONE_CONFIGURATION_FAMILY_OUT",
        "LEAVE_ONE_SCENARIO_OR_CAMPAIGN_FAMILY_OUT",
        "OPERATING_CONDITION_OUT",
        "GROUPED_SEVERITY_ONSET_DURATION_TIME_AND_SEED",
        "LEAVE_ONE_ARCHETYPE_OUT_REJECTION_OR_ADAPTATION_ONLY",
    }
    assert "random telemetry rows" in protocol["atomic_group"]["prohibited_units"]


def _minimal_catalogs():
    scope = {
        "objects": [
            {
                "object_id": "component.a",
                "level": "component",
                "domain": "demo",
                "capability_id": "component.a.v1",
            }
        ]
    }
    taxonomy = {
        "faults": [
            {
                "fault_id": "component.a:drift",
                "object_id": "component.a",
                "scenario_type": "fault",
                "effect_id": "drift",
            }
        ]
    }
    coverage = {
        "rows": [
            {
                "object_id": "component.a",
                "capability_id": "component.a.v1",
                "scenario_type": "nominal",
                "scenario_id": "a.nominal",
                "effect_ids": [],
            },
            {
                "object_id": "component.a",
                "capability_id": "component.a.v1",
                "scenario_type": "fault",
                "scenario_id": "a.fault",
                "effect_ids": ["drift"],
            },
        ]
    }
    report = {
        "gate_status": "PASS",
        "technical_baseline_status": "PASS",
        "issues": [],
        "blockers": [],
        "objects": [
            {
                "object_id": "component.a",
                "level": "component",
                "domain": "demo",
                "primary_capability_id": "component.a.v1",
                "modes": {
                    "nominal": {
                        "scenario_ids": ["a.nominal"],
                        "effect_ids": ["nominal"],
                    },
                    "fault": {
                        "scenario_ids": ["a.fault"],
                        "effect_ids": ["drift"],
                    },
                },
            }
        ],
    }
    return scope, taxonomy, coverage, report


def test_formal_catalog_audit_passes_exact_associations_and_fails_closed() -> None:
    scope, taxonomy, coverage, report = _minimal_catalogs()
    exact = LINEAGE.audit_formal_catalog(scope, taxonomy, coverage, report)
    assert exact["status"] == "PASS"
    assert exact["denominators"]["frozen_fault_effect_occurrence_count"] == 1

    report["objects"][0]["modes"]["fault"]["effect_ids"] = ["other"]
    mismatch = LINEAGE.audit_formal_catalog(scope, taxonomy, coverage, report)
    assert mismatch["status"] == "FAIL"
    assert mismatch["exact_comparison"]["taxonomy_vs_source_missing_count"] == 1
    assert mismatch["exact_comparison"]["taxonomy_vs_source_extra_count"] == 1


def test_semantic_binding_counts_catalog_rows_but_never_materialized_episodes() -> None:
    scope, taxonomy, coverage, _ = _minimal_catalogs()
    semantic = {
        "mapping_results": [
            {
                "declared_profile_capability_id": "component.a.v1",
                "declared_profile_matched_entity_count": 2,
                "declared_profile_passed_entity_count": 2,
                "declared_profile_failed_entity_count": 0,
            }
        ]
    }
    result = LINEAGE.audit_semantic_binding(scope, taxonomy, coverage, semantic)
    assert result["status"] == "PASS"
    assert result["associated_formal_scenario_row_count"] == 2
    assert result["materialized_formal_episode_count"] == 0
    assert result["materialized_formal_episode_binding_status"] == "NOT_ESTABLISHED"
    assert result["g1_mechanism_class_mapping_status"] == "NOT_ESTABLISHED_DO_NOT_INFER"


def test_current_lineage_and_holdout_receipts_preserve_s0_boundary() -> None:
    lineage_path = (
        ROOT
        / "governance"
        / "evidence"
        / "0.7.1-extension"
        / "p018_sat_agent_formal_scope_lineage_audit.v1.json"
    )
    holdout_path = (
        ROOT
        / "governance"
        / "evidence"
        / "0.7.1-extension"
        / "p018_sat_agent_holdout_materialization_audit.v1.json"
    )
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    assert lineage["formal_catalog_join"]["status"] == "PASS"
    binding = lineage["public_semantic_binding"]
    assert binding["status"] == "PARTIAL"
    assert binding["exactly_bound_formal_object_count"] == 7
    assert binding["formal_object_denominator"] == 31
    assert binding["associated_formal_scenario_row_count"] == 21
    assert binding["associated_formal_fault_effect_occurrence_count"] == 32
    assert lineage["denominator_identity_guard"]["interchangeable_or_joinable"] is False
    assert holdout["holdout_axis_denominator"] == 6
    assert holdout["active_holdout_axis_count"] == 0
    assert holdout["decision"]["stage_s1_fit_ready"] is False
    assert holdout["execution_receipt"]["split_count"] == 0


def test_declared_output_units_handles_list_and_mapping_contracts(tmp_path: Path) -> None:
    list_contract = tmp_path / "list.yaml"
    list_contract.write_text(
        "outputs:\n  - {name: physical.a, unit: W}\n", encoding="utf-8"
    )
    mapping_contract = tmp_path / "mapping.yaml"
    mapping_contract.write_text(
        "outputs:\n  physical.b: {unit: K}\n", encoding="utf-8"
    )
    assert PROFILE_GAPS._declared_output_units(list_contract) == {"physical.a": "W"}
    assert PROFILE_GAPS._declared_output_units(mapping_contract) == {"physical.b": "K"}


def test_semantic_profile_gap_receipt_never_auto_generates_maps() -> None:
    path = (
        ROOT
        / "governance"
        / "evidence"
        / "0.7.1-extension"
        / "p018_sat_agent_semantic_profile_gap_audit.v1.json"
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["summary"]["formal_object_capability_denominator"] == 31
    assert receipt["summary"]["configured_exact_capability_profile_count"] == 7
    assert receipt["summary"]["missing_exact_capability_profile_count"] == 24
    assert receipt["summary"]["feasible_missing_profile_count"] == 21
    assert receipt["summary"]["missing_explicit_time_entity_count"] == 14
    assert receipt["summary"]["missing_time_capability_count"] == 4
    assert receipt["summary"]["profile_or_channel_map_created_count"] == 0
    assert receipt["manual_review_contract"]["automatic_profile_generation_authorized"] is False
    assert receipt["execution_receipt"]["telemetry_row_read_count"] == 0


def test_pending_public_provenance_request_cannot_authorize_model_or_backfill() -> None:
    request = json.loads(
        (
            ROOT
            / "protocols"
            / "acceptance"
            / "p018_sat_agent_public_provenance_authorization_request.json"
        ).read_text(encoding="utf-8")
    )
    schema = json.loads(
        (
            ROOT
            / "protocols"
            / "integration"
            / "sat_agent_public_episode_provenance.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert request["status"] == "PENDING_PROJECT_OWNER_APPROVAL"
    assert request["problem_statement"]["additional_missing_explicit_time_entity_count"] == 14
    source_changes = request["requested_authority"]["sat_agent_source_changes"]
    assert source_changes["existing_export_provenance_backfill_authorized"] is False
    assert request["decision"] == "PENDING_PROJECT_OWNER_APPROVAL_NO_SAT_AGENT_WRITE_PERMITTED"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["model_input_eligible"]["const"] is False
    assert schema["properties"]["split_grouping_only"]["const"] is True
    prohibited = {"label", "truth", "fault_id", "effect_id", "seed", "split"}
    assert prohibited.isdisjoint(schema["properties"])
