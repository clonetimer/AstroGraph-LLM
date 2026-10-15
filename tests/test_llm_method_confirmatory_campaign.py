from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from astrograph.evaluation.llm_method_campaign import (
    ARM_ORDER,
    confirmatory_readiness,
    evaluate_confirmatory_promotion,
    exact_mcnemar_two_sided,
    freeze_confirmatory_case_manifest,
    load_confirmatory_design,
    paired_arm_effect,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _Row:
    case_id: str
    arm_id: str
    true_fault_id: str
    predicted_fault_id: str | None
    group_id: str = "G"
    probe_execution_count: int = 0


def _runtime_admission(tag: str = "a") -> dict:
    h = (tag * 64)[:64]
    client_identity = {
        "environment_lock_sha256": h,
        "environment_packages_sha256": h,
        "cyclonedx_sbom_sha256": h,
        "source_manifest_sha256": h,
        "source_checksums_sha256": h,
        "prompt_implementation_sha256": h,
        "inference_profile_file_sha256": h,
        "inference_profile_canonical_sha256": h,
    }
    return {
        "schema_version": "astrograph-confirmatory-runtime-identity-admission.v1",
        "status": "VALID",
        "runtime_identity": {
            "client_identity_sha256": h,
            "model_runtime_identity_sha256": h,
            "inference_config_sha256": h,
        },
        "runtime_identity_sha256": h,
        "client": {"identity": client_identity},
        "model_runtime_attestation_sha256": h,
        "blockers": [],
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fault_map(name: str) -> dict[str, str]:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))["mappings"]


def test_confirmatory_design_freezes_arms_cases_statistics_and_thresholds():
    design = load_confirmatory_design(ROOT)
    assert design["qualification_status"] == "PREEXECUTION_GATES_FROZEN"
    assert design["arm_order"] == ARM_ORDER
    assert design["case_selection"]["expected_case_count"] == 38
    assert design["statistics"]["inferential_unit"] == "group_id"
    assert design["statistics"]["primary_test"] == "EXACT_TWO_SIDED_GROUP_DIRECTION_SIGN"
    assert design["statistics"]["legacy_case_level_test"] == "EXACT_TWO_SIDED_MCNEMAR_DESCRIPTIVE_ONLY"
    assert design["promotion_thresholds"]["max_induced_error_count"] == 0
    assert design["promotion_thresholds"]["min_group_mean_accuracy_delta"] == pytest.approx(0.15)
    assert design["promotion_thresholds"]["max_group_direction_exact_p"] == pytest.approx(0.05)
    assert design["probe_promotion_thresholds"]["min_probe_exposed_group_count"] == 6


def test_current_source_only_package_is_fail_closed_for_confirmatory_execution():
    readiness = confirmatory_readiness(
        ROOT,
        live_model_available=False,
        live_model_detail="unit-test:no-live-endpoint",
    )
    assert readiness["status"] == "BLOCKED"
    assert "DATASET_NOT_READY:adcs" in readiness["blockers"]
    assert "DATASET_NOT_READY:phm" in readiness["blockers"]
    assert "CASE_MANIFEST_NOT_FROZEN" in readiness["blockers"]
    assert "LIVE_MODEL_UNAVAILABLE" in readiness["blockers"]
    assert "not an acceptance result" in readiness["claim_boundary"]


def test_exact_mcnemar_and_paired_effect_are_case_paired():
    # Six one-way corrections are the first all-one-direction count that clears
    # two-sided alpha=0.05: 2*(1/2^6) == 0.03125.
    assert exact_mcnemar_two_sided(corrected=6, induced=0) == pytest.approx(0.03125)
    rows: list[_Row] = []
    for index in range(6):
        case = f"c{index}"
        group = f"g{index}"
        rows.append(_Row(case, "A0_ALGORITHM_ONLY", "F", None, group_id=group))
        rows.append(_Row(case, "A2_CONTROLLED_FULL", "F", "F", group_id=group))
    effect = paired_arm_effect(
        rows,
        baseline_arm_id="A0_ALGORITHM_ONLY",
        challenger_arm_id="A2_CONTROLLED_FULL",
    )
    assert effect["case_count"] == 6
    assert effect["corrected_case_count"] == 6
    assert effect["induced_error_count"] == 0
    assert effect["net_diagnostic_gain"] == 6
    assert effect["mcnemar_exact_two_sided_p"] == pytest.approx(0.03125)
    assert effect["independent_group_count"] == 6
    assert effect["group_direction_exact_two_sided_p"] == pytest.approx(0.03125)
    assert effect["group_mean_accuracy_delta"] == pytest.approx(1.0)


def test_freeze_confirmatory_case_manifest_uses_all_exact_mapped_test_rows(tmp_path: Path):
    adcs_map = _fault_map("configs/datasets/adcs_v5_2_2/fault_class_map.json")
    phm_map = _fault_map("configs/datasets/phm_v6_0/fault_class_map.json")
    adcs_root = tmp_path / "adcs"
    phm_root = tmp_path / "phm"

    adcs_rows: list[dict[str, object]] = []
    for class_index, dataset_class in enumerate(adcs_map):
        for sample_index in (4, 9):
            pair = f"PAIR-{class_index}-{sample_index}"
            adcs_rows.append({
                "sample_id": f"ADCS-{class_index}-{sample_index}-FAULT",
                "trajectory_pair_id": pair,
                "sample_index": sample_index,
                "scenario_id": f"SCN-{class_index}",
                "diagnostic_class": dataset_class,
                "telemetry_file": "fault.csv",
                "labels_file": "labels.json",
            })
            adcs_rows.append({
                "sample_id": f"ADCS-{class_index}-{sample_index}-NOM",
                "trajectory_pair_id": pair,
                "sample_index": sample_index,
                "scenario_id": f"SCN-{class_index}-NOM",
                "diagnostic_class": "nominal",
                "telemetry_file": "nominal.csv",
                "labels_file": "labels.json",
            })
    _write_csv(adcs_root / "sample_index.csv", adcs_rows)
    (adcs_root / "fault.csv").write_text("time_s,x\n0,1\n", encoding="utf-8")
    (adcs_root / "nominal.csv").write_text("time_s,x\n0,0\n", encoding="utf-8")
    (adcs_root / "labels.json").write_text('{"schema":"test"}\n', encoding="utf-8")

    phm_rows: list[dict[str, object]] = []
    for class_index, dataset_class in enumerate(phm_map):
        subsystem = f"SUBSYSTEM-{class_index}"
        for sample_index in (4, 9):
            common = {
                "sample_index": sample_index,
                "subsystem": subsystem,
                "activation_time_s": 10.0,
                "event_start_s": 9.0,
                "fault_manifestation_time_s": 11.0,
                "fault_injection_phase": "test",
                "telemetry_file": "telemetry.csv",
            }
            phm_rows.append({
                "sample_id": f"PHM-{class_index}-{sample_index}-FAULT",
                "condition_id": dataset_class,
                **common,
            })
            phm_rows.append({
                "sample_id": f"PHM-{class_index}-{sample_index}-NOM",
                "condition_id": "nominal",
                **common,
            })
    _write_csv(phm_root / "sample_index.csv", phm_rows)
    (phm_root / "telemetry.csv").write_text("time_s,x\n0,1\n", encoding="utf-8")

    output = tmp_path / "case_manifest.json"
    manifest = freeze_confirmatory_case_manifest(
        ROOT,
        output_path=output,
        dataset_roots={"adcs": adcs_root, "phm": phm_root},
    )
    assert output.is_file()
    assert manifest["case_count"] == 38
    assert len(manifest["cases"]) == 38
    assert {case["sample_index"] for case in manifest["cases"]} == {4, 9}
    assert len({case["case_id"] for case in manifest["cases"]}) == 38
    assert {case["dataset_class"] for case in manifest["cases"] if case["dataset"] == "adcs"} == set(adcs_map)
    assert {case["dataset_class"] for case in manifest["cases"] if case["dataset"] == "phm"} == set(phm_map)
    assert all(case["nominal_sample_id"] for case in manifest["cases"])

    readiness = confirmatory_readiness(
        ROOT,
        dataset_roots={"adcs": adcs_root, "phm": phm_root},
        live_model_available=True,
        live_model_detail="models=Qwen3.5-9B",
        requested_model_id="Qwen3.5-9B",
        observed_model_ids=["Qwen3.5-9B"],
        case_manifest_path=output,
    )
    assert readiness["status"] == "READY"
    assert readiness["case_manifest_path"] == str(output)
    assert readiness["declared_case_manifest_path"].endswith("case_manifest.json")
    assert readiness["case_manifest"]["case_count"] == 38
    assert readiness["datasets"]["adcs"]["mount_audit_status"] == "VALID"
    assert readiness["datasets"]["phm"]["mount_audit_status"] == "VALID"
    assert len(readiness["datasets"]["adcs"]["payload_identity_sha256"]) == 64
    assert len(readiness["datasets"]["phm"]["payload_identity_sha256"]) == 64


def test_readiness_rejects_mounted_dataset_with_missing_referenced_payload_files(tmp_path: Path):
    adcs_map = _fault_map("configs/datasets/adcs_v5_2_2/fault_class_map.json")
    phm_map = _fault_map("configs/datasets/phm_v6_0/fault_class_map.json")
    adcs_root = tmp_path / "adcs"
    phm_root = tmp_path / "phm"
    adcs_rows = []
    for class_index, dataset_class in enumerate(adcs_map):
        for sample_index in (4, 9):
            pair = f"PAIR-{class_index}-{sample_index}"
            adcs_rows += [
                {"sample_id": f"AF-{class_index}-{sample_index}", "trajectory_pair_id": pair, "sample_index": sample_index, "scenario_id": "S", "diagnostic_class": dataset_class, "telemetry_file": "missing-f.csv", "labels_file": "missing-l.json"},
                {"sample_id": f"AN-{class_index}-{sample_index}", "trajectory_pair_id": pair, "sample_index": sample_index, "scenario_id": "N", "diagnostic_class": "nominal", "telemetry_file": "missing-n.csv", "labels_file": "missing-l.json"},
            ]
    _write_csv(adcs_root / "sample_index.csv", adcs_rows)
    phm_rows = []
    for class_index, dataset_class in enumerate(phm_map):
        for sample_index in (4, 9):
            common = {"sample_index": sample_index, "subsystem": f"SS-{class_index}", "activation_time_s": 1.0, "event_start_s": 1.0, "fault_manifestation_time_s": 1.0, "fault_injection_phase": "test", "telemetry_file": "missing-t.csv"}
            phm_rows += [
                {"sample_id": f"PF-{class_index}-{sample_index}", "condition_id": dataset_class, **common},
                {"sample_id": f"PN-{class_index}-{sample_index}", "condition_id": "nominal", **common},
            ]
    _write_csv(phm_root / "sample_index.csv", phm_rows)
    manifest_path = tmp_path / "case_manifest.json"
    freeze_confirmatory_case_manifest(ROOT, output_path=manifest_path, dataset_roots={"adcs": adcs_root, "phm": phm_root})

    readiness = confirmatory_readiness(
        ROOT, dataset_roots={"adcs": adcs_root, "phm": phm_root},
        live_model_available=True, live_model_detail="models=Qwen3.5-9B",
        requested_model_id="Qwen3.5-9B", observed_model_ids=["Qwen3.5-9B"],
        case_manifest_path=manifest_path,
    )
    assert readiness["status"] == "BLOCKED"
    assert "DATASET_MOUNT_INVALID:adcs" in readiness["blockers"]
    assert "DATASET_MOUNT_INVALID:phm" in readiness["blockers"]
    assert readiness["datasets"]["adcs"]["mount_audit_status"] == "INVALID"
    assert readiness["datasets"]["adcs"]["missing_files"]
    assert readiness["datasets"]["phm"]["missing_files"]


def test_promotion_gate_cannot_pass_without_all_frozen_primary_checks():
    design = load_confirmatory_design(ROOT)
    # Six beneficial independent groups and 32 ties preserve the old all-benefit
    # significance-clearing pattern while making group_id the inferential unit.
    deltas = [1.0] * 6 + [0.0] * 32
    mean = sum(deltas) / len(deltas)
    variance = sum((x - mean) ** 2 for x in deltas) / (len(deltas) - 1)
    half = 2.026 * (variance / len(deltas)) ** 0.5
    paired = {
        "induced_error_count": 0,
        "net_diagnostic_gain": 6,
        "independent_group_count": 38,
        "positive_group_count": 6,
        "negative_group_count": 0,
        "tie_group_count": 32,
        "probe_exposed_group_count": 6,
        "group_mean_accuracy_delta": mean,
        "group_mean_accuracy_delta_ci95_lower": mean - half,
        "group_mean_accuracy_delta_ci95_upper": mean + half,
        "group_mean_accuracy_delta_ci95_half_width": half,
        "group_direction_exact_two_sided_p": 0.03125,
    }
    report = {
        "arm_metrics": {
            "A2_CONTROLLED_FULL": {
                "true_fault_in_candidate_set_rate": 1.0,
                "evidence_citation_valid_rate": 1.0,
            }
        },
        "paired_comparisons": {
            "A0_ALGORITHM_ONLY__vs__A2_CONTROLLED_FULL": dict(paired),
            "A7_NO_ACTIVE_PROBE__vs__A2_CONTROLLED_FULL": dict(paired),
        },
    }
    decision = evaluate_confirmatory_promotion(report, design)
    assert decision["status"] == "PASS"
    assert decision["probe_promotion_review"]["status"] == "ELIGIBLE_FOR_SEPARATE_FORMAL_EVIDENCE_REVIEW"
    report["paired_comparisons"]["A0_ALGORITHM_ONLY__vs__A2_CONTROLLED_FULL"]["induced_error_count"] = 1
    decision = evaluate_confirmatory_promotion(report, design)
    assert decision["status"] == "FAIL"
    assert decision["checks"]["no_induced_errors"] is False


def test_clustered_cases_do_not_inflate_primary_significance():
    rows: list[_Row] = []
    # Six corrected cases in one shared group look significant under ordinary
    # case-level McNemar, but constitute only one independent directional unit.
    for index in range(6):
        case = f"clustered-{index}"
        rows.append(_Row(case, "A0_ALGORITHM_ONLY", "F", None, group_id="shared"))
        rows.append(_Row(case, "A2_CONTROLLED_FULL", "F", "F", group_id="shared"))
    effect = paired_arm_effect(rows, baseline_arm_id="A0_ALGORITHM_ONLY", challenger_arm_id="A2_CONTROLLED_FULL")
    assert effect["mcnemar_exact_two_sided_p"] == pytest.approx(0.03125)
    assert effect["case_level_mcnemar_inference_status"] == "DESCRIPTIVE_ONLY_CLUSTERING_IGNORED"
    assert effect["independent_group_count"] == 1
    assert effect["group_direction_exact_two_sided_p"] == pytest.approx(1.0)
    assert effect["group_mean_accuracy_delta_ci95_lower"] is None


def test_runner_rejects_case_manifest_denominator_or_campaign_drift(tmp_path: Path):
    from scripts.experiments.run_llm_method_ablation import _load_case_manifest

    payload = {
        "schema_version": "astrograph-llm-method-case-manifest.v1",
        "campaign_id": "wrong-campaign",
        "case_count": 1,
        "cases": [{
            "case_id": "c1",
            "dataset": "adcs",
            "dataset_class": "adcs_rw_jamming",
            "true_fault_id": "ADCS_RW_JAM",
            "sample_id": "sample-1",
            "group_id": "g1",
        }],
    }
    path = tmp_path / "case_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="campaign_id"):
        _load_case_manifest(path, expected_campaign_id="frozen-campaign", expected_case_count=1)
    payload["campaign_id"] = "frozen-campaign"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="denominator"):
        _load_case_manifest(path, expected_campaign_id="frozen-campaign", expected_case_count=38)


def test_case_manifest_validation_and_execution_receipt_freeze_identity(tmp_path: Path):
    from astrograph.evaluation.llm_method_campaign import (
        build_confirmatory_execution_receipt,
        validate_confirmatory_case_manifest,
    )

    adcs_map = _fault_map("configs/datasets/adcs_v5_2_2/fault_class_map.json")
    phm_map = _fault_map("configs/datasets/phm_v6_0/fault_class_map.json")
    adcs_root = tmp_path / "adcs"
    phm_root = tmp_path / "phm"
    adcs_rows = []
    for class_index, dataset_class in enumerate(adcs_map):
        for sample_index in (4, 9):
            pair = f"PAIR-{class_index}-{sample_index}"
            adcs_rows += [
                {"sample_id": f"AF-{class_index}-{sample_index}", "trajectory_pair_id": pair, "sample_index": sample_index, "scenario_id": "S", "diagnostic_class": dataset_class, "telemetry_file": "f.csv", "labels_file": "l.json"},
                {"sample_id": f"AN-{class_index}-{sample_index}", "trajectory_pair_id": pair, "sample_index": sample_index, "scenario_id": "N", "diagnostic_class": "nominal", "telemetry_file": "n.csv", "labels_file": "l.json"},
            ]
    _write_csv(adcs_root / "sample_index.csv", adcs_rows)
    (adcs_root / "f.csv").write_text("time_s,x\n0,1\n", encoding="utf-8")
    (adcs_root / "n.csv").write_text("time_s,x\n0,0\n", encoding="utf-8")
    (adcs_root / "l.json").write_text('{"schema":"test"}\n', encoding="utf-8")
    phm_rows = []
    for class_index, dataset_class in enumerate(phm_map):
        for sample_index in (4, 9):
            common = {"sample_index": sample_index, "subsystem": f"SS-{class_index}", "activation_time_s": 1.0, "event_start_s": 1.0, "fault_manifestation_time_s": 1.0, "fault_injection_phase": "test", "telemetry_file": "t.csv"}
            phm_rows += [
                {"sample_id": f"PF-{class_index}-{sample_index}", "condition_id": dataset_class, **common},
                {"sample_id": f"PN-{class_index}-{sample_index}", "condition_id": "nominal", **common},
            ]
    _write_csv(phm_root / "sample_index.csv", phm_rows)
    (phm_root / "t.csv").write_text("time_s,x\n0,1\n", encoding="utf-8")

    manifest_path = tmp_path / "case_manifest.json"
    manifest = freeze_confirmatory_case_manifest(
        ROOT, output_path=manifest_path, dataset_roots={"adcs": adcs_root, "phm": phm_root}
    )
    design = load_confirmatory_design(ROOT)
    canonical_path = tmp_path / "root" / design["case_manifest_path"]
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.write_text(json.dumps(manifest), encoding="utf-8")
    # Validation can target an explicit frozen path without mutating the source tree.
    status = validate_confirmatory_case_manifest(ROOT, case_manifest_path=manifest_path)
    assert status["status"] == "VALID"
    assert status["case_count"] == 38
    assert len(status["sha256"]) == 64
    assert len(status["case_identity_sha256"]) == 64

    readiness = {
        "status": "READY",
        "design": {"sha256": "d" * 64},
        "case_manifest": status,
        "datasets": {
            "adcs": {"root": str(adcs_root), "sample_index_sha256": "a" * 64, "payload_identity_sha256": "b" * 64, "payload_file_count": 84},
            "phm": {"root": str(phm_root), "sample_index_sha256": "p" * 64, "payload_identity_sha256": "q" * 64, "payload_file_count": 20},
        },
        "observed_model_ids": ["Qwen3.5-9B"],
    }
    runtime_admission = _runtime_admission()
    receipt = build_confirmatory_execution_receipt(
        ROOT, readiness=readiness, vllm_url="http://127.0.0.1:8000/v1", vllm_model="Qwen3.5-9B",
        runtime_identity_admission=runtime_admission,
    )
    assert receipt["case_count"] == 38
    assert receipt["vllm_model"] == "Qwen3.5-9B"
    assert receipt["dataset_fingerprints"]["adcs"]["sample_index_sha256"] == "a" * 64
    assert receipt["dataset_fingerprints"]["adcs"]["payload_identity_sha256"] == "b" * 64
    assert receipt["dataset_fingerprints"]["adcs"]["payload_file_count"] == 84
    assert len(receipt["receipt_sha256"]) == 64


def test_readiness_blocks_when_requested_model_is_not_advertised():
    readiness = confirmatory_readiness(
        ROOT,
        live_model_available=True,
        live_model_detail="models=OtherModel",
        requested_model_id="Qwen3.5-9B",
        observed_model_ids=["OtherModel"],
    )
    assert readiness["status"] == "BLOCKED"
    assert "REQUESTED_MODEL_NOT_ADVERTISED" in readiness["blockers"]
    assert readiness["requested_model_id"] == "Qwen3.5-9B"


def test_execution_receipt_rejects_blocked_admission():
    from astrograph.evaluation.llm_method_campaign import build_confirmatory_execution_receipt
    with pytest.raises(ValueError, match="READY admission"):
        build_confirmatory_execution_receipt(
            ROOT,
            readiness={"status": "BLOCKED"},
            vllm_url="http://127.0.0.1:8000/v1",
            vllm_model="Qwen3.5-9B",
        )


def test_post_execution_identity_check_detects_dataset_and_model_drift():
    from astrograph.evaluation.llm_method_campaign import verify_confirmatory_execution_identity

    receipt = {
        "campaign_id": "campaign",
        "design_sha256": "d" * 64,
        "case_manifest_sha256": "m" * 64,
        "case_identity_sha256": "i" * 64,
        "case_count": 38,
        "dataset_fingerprints": {
            "adcs": {"sample_index_sha256": "a" * 64, "payload_identity_sha256": "b" * 64, "payload_file_count": 84},
            "phm": {"sample_index_sha256": "p" * 64, "payload_identity_sha256": "q" * 64, "payload_file_count": 20},
        },
        "vllm_model": "Qwen3.5-9B",
        "runtime_identity_sha256": "a" * 64,
        "client_runtime_identity_sha256": "a" * 64,
        "model_runtime_identity_sha256": "a" * 64,
        "inference_config_sha256": "a" * 64,
    }
    from astrograph.evaluation import llm_method_campaign as campaign_module
    receipt["receipt_sha256"] = campaign_module._canonical_sha256(receipt)
    stable = {
        "status": "READY",
        "campaign_id": "campaign",
        "blockers": [],
        "design": {"sha256": "d" * 64},
        "case_manifest": {"sha256": "m" * 64, "case_identity_sha256": "i" * 64, "case_count": 38},
        "datasets": {
            "adcs": {"sample_index_sha256": "a" * 64, "payload_identity_sha256": "b" * 64, "payload_file_count": 84},
            "phm": {"sample_index_sha256": "p" * 64, "payload_identity_sha256": "q" * 64, "payload_file_count": 20},
        },
        "observed_model_ids": ["Qwen3.5-9B"],
    }
    runtime_admission = _runtime_admission()
    result = verify_confirmatory_execution_identity(
        execution_receipt=receipt, current_readiness=stable, current_runtime_identity_admission=runtime_admission
    )
    assert result["status"] == "STABLE"
    assert result["drift_codes"] == []

    drifted = json.loads(json.dumps(stable))
    drifted["datasets"]["phm"]["sample_index_sha256"] = "x" * 64
    drifted["datasets"]["adcs"]["payload_identity_sha256"] = "y" * 64
    drifted["observed_model_ids"] = ["OtherModel"]
    result = verify_confirmatory_execution_identity(
        execution_receipt=receipt, current_readiness=drifted, current_runtime_identity_admission=runtime_admission
    )
    assert result["status"] == "DRIFTED"
    assert "DATASET_SAMPLE_INDEX_DRIFT:phm" in result["drift_codes"]
    assert "DATASET_PAYLOAD_IDENTITY_DRIFT:adcs" in result["drift_codes"]
    assert "MODEL_ENDPOINT_ADVERTISEMENT_DRIFT" in result["drift_codes"]


def test_post_execution_identity_check_rejects_tampered_execution_receipt():
    from astrograph.evaluation import llm_method_campaign as campaign_module

    runtime_admission = _runtime_admission()
    receipt = {
        "campaign_id": "campaign",
        "design_sha256": "d" * 64,
        "case_manifest_sha256": "m" * 64,
        "case_identity_sha256": "i" * 64,
        "case_count": 38,
        "dataset_fingerprints": {},
        "vllm_model": "Qwen3.5-9B",
        "runtime_identity_sha256": "a" * 64,
        "client_runtime_identity_sha256": "a" * 64,
        "model_runtime_identity_sha256": "a" * 64,
        "inference_config_sha256": "a" * 64,
    }
    receipt["receipt_sha256"] = campaign_module._canonical_sha256(receipt)
    receipt["case_count"] = 37
    current = {
        "status": "READY", "campaign_id": "campaign", "blockers": [],
        "design": {"sha256": "d" * 64},
        "case_manifest": {"sha256": "m" * 64, "case_identity_sha256": "i" * 64, "case_count": 38},
        "datasets": {},
        "observed_model_ids": ["Qwen3.5-9B"],
    }
    result = campaign_module.verify_confirmatory_execution_identity(
        execution_receipt=receipt, current_readiness=current, current_runtime_identity_admission=runtime_admission
    )
    assert result["status"] == "DRIFTED"
    assert "EXECUTION_RECEIPT_SHA256_INVALID" in result["drift_codes"]


def test_post_execution_identity_check_blocks_if_readiness_degrades():
    from astrograph.evaluation.llm_method_campaign import verify_confirmatory_execution_identity

    result = verify_confirmatory_execution_identity(
        execution_receipt={"campaign_id": "c", "vllm_model": "m", "dataset_fingerprints": {}},
        current_readiness={"status": "BLOCKED", "campaign_id": "c", "blockers": ["LIVE_MODEL_UNAVAILABLE"]},
    )
    assert result["status"] == "DRIFTED"
    assert "POST_EXECUTION_READINESS_BLOCKED" in result["drift_codes"]
    assert "POST_BLOCKER:LIVE_MODEL_UNAVAILABLE" in result["drift_codes"]


def test_case_manifest_admission_requires_preregistered_independent_group_count(tmp_path: Path):
    design = load_confirmatory_design(ROOT)
    cases = []
    for index in range(design["case_selection"]["expected_case_count"]):
        cases.append({
            "case_id": f"C-{index}",
            "dataset": "adcs" if index < 28 else "phm",
            "dataset_id": "D",
            "dataset_class": "CLASS",
            "true_fault_id": "FAULT",
            "sample_id": f"S-{index}",
            "nominal_sample_id": f"N-{index}",
            "group_id": "ONE-SHARED-GROUP",
        })
    path = tmp_path / "case_manifest.json"
    path.write_text(json.dumps({
        "schema_version": "astrograph-llm-method-case-manifest.v1",
        "campaign_id": design["campaign_id"],
        "case_count": len(cases),
        "cases": cases,
    }), encoding="utf-8")
    from astrograph.evaluation.llm_method_campaign import validate_confirmatory_case_manifest
    status = validate_confirmatory_case_manifest(ROOT, case_manifest_path=path)
    assert status["status"] == "INVALID"
    assert status["independent_group_count"] == 1
    assert "CASE_MANIFEST_INSUFFICIENT_INDEPENDENT_GROUPS" in status["errors"]
