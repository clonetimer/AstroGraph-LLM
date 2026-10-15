from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from astrograph.data.trusted import (
    AcceptanceLevel,
    audit_round6_dataset,
    classify_leak_columns,
    scan_public_csv,
    sha256_file,
    write_trusted_audit_outputs,
)


def _write_csv(path: Path, header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow([0 for _ in header])


def _write_run(root: Path, split: str, case_id: str, variant: str, *, full_pass: bool, leak: bool) -> None:
    case = root / "basilisk_campaigns" / split / "raw" / case_id
    run = case / variant
    run.mkdir(parents=True, exist_ok=True)
    status = "PASS" if full_pass else "FAIL"
    summary = {
        "status": status,
        "execution_status": status,
        "physics_status": status,
        "data_conservation_status": status,
        "energy_conservation_status": "PASS",
        "numerical_status": "PASS",
        "mission_status": "PASS",
        "effect_execution_status": "PASS",
        "runtime_fault_triggered_count": 0 if variant == "nominal" else 1,
        "degradation_effect_count": 0,
        "basilisk_simbase_used": True,
        "execute_simulation_used": True,
        "backend": "basilisk",
    }
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    manifest = {"platform": {"platform_version": "test", "bsk_version": "2.11.0", "python_version": "3.13"}}
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_csv(run / "trace.csv", ["time_s", "battery_soc"])
    header = ["time_s", "battery_soc"] + (["label.fault_active", "effect.active_ids"] if leak else [])
    _write_csv(case / f"{variant}_trace.csv", header)


def _synthetic_package(tmp_path: Path, *, full_pass: bool, leak: bool) -> tuple[Path, Path]:
    dataset = tmp_path / "dataset"
    sat = tmp_path / "sat"
    for split, prefix in (("development", "d"), ("holdout", "h"), ("final", "f")):
        campaign = dataset / "basilisk_campaigns" / split
        result = {"case_id": f"{prefix}-001", "scenario_id": "rw_jam", "difficulty": "simple", "category": "single", "status": "PASS"}
        campaign.mkdir(parents=True, exist_ok=True)
        (campaign / "campaign_summary.json").write_text(json.dumps({"results": [result]}), encoding="utf-8")
        for variant in ("nominal", "effect"):
            _write_run(dataset, split, f"{prefix}-001", variant, full_pass=full_pass, leak=leak)
    (dataset / "reproduction").mkdir(parents=True)
    generator = dataset / "reproduction" / "generate_coupled_fault_campaign.py"
    generator.write_text("print('test')\n", encoding="utf-8")
    (dataset / "evidence").mkdir()
    (dataset / "evidence" / "round6_real_coupling_freeze_manifest.json").write_text(json.dumps({
        "files": [{"path": "/x/generate_coupled_fault_campaign.py", "sha256": sha256_file(generator)}]
    }), encoding="utf-8")
    (sat / "scripts").mkdir(parents=True)
    (sat / "scripts" / "run_basilisk_fault_matrix.py").write_text("print('test')\n", encoding="utf-8")
    return dataset, sat


def test_direct_and_proxy_columns_are_separated() -> None:
    direct, proxies = classify_leak_columns(["time_s", "label.fault_active", "effect.active_ids"])
    assert direct == ["label.fault_active"]
    assert proxies == ["effect.active_ids"]


def test_adversarial_answer_proxy_names_fail_closed() -> None:
    direct, proxies = classify_leak_columns(
        [
            "time_s",
            "fault_type",
            "Fault Type",
            "condition_id",
            "diagnostic_class",
            "event_start_s",
            "event.active",
            "injected_fault",
            "case_id",
            "ground_truth_fault",
        ]
    )
    assert direct == ["Fault Type", "diagnostic_class", "fault_type", "ground_truth_fault"]
    assert proxies == [
        "case_id",
        "condition_id",
        "event.active",
        "event_start_s",
        "injected_fault",
    ]


def test_observable_fault_named_channel_is_not_rejected_by_generic_substring() -> None:
    direct, proxies = classify_leak_columns(["time_s", "sensor.fault_current_a"])
    assert direct == []
    assert proxies == []


def test_bom_header_is_normalised(tmp_path: Path) -> None:
    trace = tmp_path / "trace.csv"
    _write_csv(trace, ["label.health_state", "battery_soc"])
    result = scan_public_csv(trace)
    assert result["leakage"] is True
    assert "label.health_state" in result["direct_label_columns"]


def test_missing_public_trace_fails_closed(tmp_path: Path) -> None:
    result = scan_public_csv(tmp_path / "missing.csv")
    assert result["leakage"] is True
    assert result["error"] == "PUBLIC_TRACE_MISSING"


def test_clean_csv_has_no_leakage(tmp_path: Path) -> None:
    trace = tmp_path / "trace.csv"
    _write_csv(trace, ["time_s", "battery_soc"])
    assert scan_public_csv(trace)["leakage"] is False


def test_full_gate_failure_cannot_be_acceptance_valid(tmp_path: Path) -> None:
    dataset, sat = _synthetic_package(tmp_path, full_pass=False, leak=False)
    audit, rows, _ = audit_round6_dataset(dataset, sat, dataset_archive_sha256="a" * 64, sat_agent_archive_sha256="b" * 64)
    assert audit.acceptance_valid_count == 0
    assert {row.acceptance_level for row in rows} == {AcceptanceLevel.EFFECT_VALID_RESEARCH_ONLY.value}
    assert all(row.full_gate_status == "FAIL" for row in rows)


def test_label_leakage_makes_runs_invalid(tmp_path: Path) -> None:
    dataset, sat = _synthetic_package(tmp_path, full_pass=True, leak=True)
    audit, rows, _ = audit_round6_dataset(dataset, sat, dataset_archive_sha256="a" * 64, sat_agent_archive_sha256="b" * 64)
    assert audit.public_trace_leakage_count == 6
    assert audit.invalid_count == 6
    assert all(row.allowed_uses == ["forensic_review"] for row in rows)


def test_clean_full_pass_runs_are_acceptance_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, sat = _synthetic_package(tmp_path, full_pass=True, leak=False)
    monkeypatch.setattr("astrograph.data.trusted.importlib.util.find_spec", lambda name: object())
    audit, rows, _ = audit_round6_dataset(dataset, sat, dataset_archive_sha256="a" * 64, sat_agent_archive_sha256="b" * 64)
    assert audit.acceptance_valid_count == 6
    assert audit.acceptance_valid_pair_count == 3
    assert all("acceptance" in row.allowed_uses for row in rows)
    assert audit.reproduction_ready is True


def test_research_only_has_no_formal_uses(tmp_path: Path) -> None:
    dataset, sat = _synthetic_package(tmp_path, full_pass=False, leak=False)
    _, rows, _ = audit_round6_dataset(dataset, sat)
    formal = {"training", "calibration", "validation", "acceptance"}
    assert all(not formal.intersection(row.allowed_uses) for row in rows)


def test_campaign_pass_is_not_complete_gate_pass(tmp_path: Path) -> None:
    dataset, sat = _synthetic_package(tmp_path, full_pass=False, leak=False)
    audit, rows, _ = audit_round6_dataset(dataset, sat)
    assert audit.campaign_claim_pass_count == 6
    assert all(row.full_gate_status == "FAIL" for row in rows)


def test_exact_generation_script_is_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, sat = _synthetic_package(tmp_path, full_pass=True, leak=False)
    monkeypatch.setattr("astrograph.data.trusted.importlib.util.find_spec", lambda name: object())
    (dataset / "reproduction" / "generate_coupled_fault_campaign.py").write_text("changed\n", encoding="utf-8")
    audit, _, _ = audit_round6_dataset(dataset, sat, dataset_archive_sha256="a", sat_agent_archive_sha256="b")
    assert audit.reproduction_ready is False
    assert audit.reproduction_checks["dataset_generation_script_exact_match"] is False


def test_missing_run_matrix_blocks_reproduction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, sat = _synthetic_package(tmp_path, full_pass=True, leak=False)
    monkeypatch.setattr("astrograph.data.trusted.importlib.util.find_spec", lambda name: object())
    (sat / "scripts" / "run_basilisk_fault_matrix.py").unlink()
    audit, _, _ = audit_round6_dataset(dataset, sat, dataset_archive_sha256="a", sat_agent_archive_sha256="b")
    assert audit.reproduction_ready is False


def test_missing_basilisk_blocks_reproduction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, sat = _synthetic_package(tmp_path, full_pass=True, leak=False)
    monkeypatch.setattr("astrograph.data.trusted.importlib.util.find_spec", lambda name: None)
    audit, _, _ = audit_round6_dataset(dataset, sat, dataset_archive_sha256="a", sat_agent_archive_sha256="b")
    assert audit.reproduction_ready is False
    assert audit.reproduction_checks["basilisk_runtime_available"] is False


def test_output_writer_emits_all_artifacts(tmp_path: Path) -> None:
    dataset, sat = _synthetic_package(tmp_path, full_pass=False, leak=False)
    audit, rows, leakage = audit_round6_dataset(dataset, sat)
    out = tmp_path / "out"
    write_trusted_audit_outputs(
        audit, rows, leakage,
        audit_path=out / "audit.json",
        manifest_jsonl_path=out / "manifest.jsonl",
        matrix_csv_path=out / "matrix.csv",
        leakage_path=out / "leakage.json",
    )
    assert all((out / name).is_file() for name in ("audit.json", "manifest.jsonl", "matrix.csv", "leakage.json"))
    assert len((out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()) == 6


def test_source_hashes_are_recorded(tmp_path: Path) -> None:
    dataset, sat = _synthetic_package(tmp_path, full_pass=False, leak=False)
    audit, rows, _ = audit_round6_dataset(dataset, sat)
    assert len(audit.dataset_tree_sha256) == 64
    assert len(audit.sat_agent_tree_sha256) == 64
    assert all(len(row.source_summary_sha256) == 64 for row in rows)


def test_audit_counts_sum_to_total(tmp_path: Path) -> None:
    dataset, sat = _synthetic_package(tmp_path, full_pass=False, leak=True)
    audit, _, _ = audit_round6_dataset(dataset, sat)
    assert audit.acceptance_valid_count + audit.effect_valid_research_only_count + audit.invalid_count == audit.total_run_count


def test_public_root_trace_is_scanned_not_only_internal_trace(tmp_path: Path) -> None:
    dataset, sat = _synthetic_package(tmp_path, full_pass=True, leak=True)
    # Internal traces are deliberately clean; delivered root traces leak labels.
    assert scan_public_csv(next(dataset.rglob("nominal/trace.csv")))["leakage"] is False
    audit, _, _ = audit_round6_dataset(dataset, sat)
    assert audit.public_trace_leakage_count == 6


def test_truth_proxy_is_not_downgraded_to_warning(tmp_path: Path) -> None:
    dataset, sat = _synthetic_package(tmp_path, full_pass=True, leak=True)
    audit, _, leakage = audit_round6_dataset(dataset, sat)
    assert audit.status == "BLOCKED"
    assert any("effect.active_ids" in record["truth_proxy_columns"] for record in leakage)


def test_delivery_leakage_does_not_erase_source_research_evidence(tmp_path: Path) -> None:
    dataset, sat = _synthetic_package(tmp_path, full_pass=False, leak=True)
    audit, rows, _ = audit_round6_dataset(dataset, sat)
    assert audit.source_effect_valid_research_only_count == 6
    assert all(row.source_evidence_level == "EFFECT_VALID_RESEARCH_ONLY" for row in rows)
    assert all(row.acceptance_level == "INVALID" for row in rows)


def _synthetic_trusted_campaign(tmp_path: Path, *, leak: bool = False, count_pairs: int = 6) -> tuple[Path, Path]:
    campaign = tmp_path / "trusted_campaign"
    sat = tmp_path / "sat"
    for name in (
        "run_basilisk_fault_matrix.py",
        "generate_coupled_fault_campaign.py",
        "install_basilisk_runtime.py",
        "export_trusted_run_evidence.py",
    ):
        path = sat / "scripts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("print('test')\n", encoding="utf-8")
    (campaign / "runtime_identity.json").parent.mkdir(parents=True, exist_ok=True)
    (campaign / "runtime_identity.json").write_text(json.dumps({
        "status": "PASS", "version": "2.11.0", "python_version": "3.13"
    }), encoding="utf-8")
    results = []
    for index in range(count_pairs):
        case_id = f"case-{index:02d}"
        pair = campaign / "cases" / case_id
        pair.mkdir(parents=True, exist_ok=True)
        pair_validation = pair / "pair_validation.json"
        pair_validation.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
        results.append({"case_id": case_id, "status": "PASS"})
        for variant in ("nominal", "effect"):
            run = pair / variant
            run.mkdir()
            source = {
                "status": "PASS", "execution_status": "PASS",
                "physics_status": "PASS_SINGLE_RUN_LIMITED",
                "data_conservation_status": "PASS", "energy_conservation_status": "PASS",
                "runtime_injection_status": "PASS", "runtime_fault_triggered_count": 0 if variant == "nominal" else 1,
                "runtime_unsupported_count": 0, "basilisk_simbase_used": True,
                "execute_simulation_used": True, "backend": "basilisk",
            }
            source_path = run / "source_summary.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            trace = run / "trace.csv"
            header = "time_s,battery_soc" + (",label.fault_active" if leak else "")
            trace.write_text(header + "\n0,0.6" + (",0" if leak else "") + "\n", encoding="utf-8")
            summary = {
                "case_id": case_id, "scenario_id": "test_fault", "split": "development", "variant": variant,
                "status": "PASS", "execution_status": "PASS", "physics_status": "PASS",
                "data_conservation_status": "PASS", "energy_conservation_status": "PASS",
                "effect_execution_status": "NOT_REQUIRED" if variant == "nominal" else "PASS",
                "acceptance_level": "ACCEPTANCE_VALID", "basilisk_version": "2.11.0",
                "source_summary_sha256": sha256_file(source_path),
                "pair_validation_sha256": sha256_file(pair_validation),
                "public_trace_sha256": sha256_file(trace),
            }
            (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (run / "manifest.json").write_text(json.dumps({
                "source_version": "test", "platform": {"version": "2.11.0", "python_version": "3.13"}
            }), encoding="utf-8")
    (campaign / "campaign_manifest.json").write_text(json.dumps({
        "status": "PASS", "total_run_count": count_pairs * 2,
        "acceptance_valid_count": count_pairs * 2,
        "acceptance_valid_pair_count": count_pairs,
        "case_results": results,
    }), encoding="utf-8")
    return campaign, sat


def test_regenerated_campaign_passes_minimum_with_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from astrograph.data.trusted import audit_trusted_campaign
    campaign, sat = _synthetic_trusted_campaign(tmp_path)
    monkeypatch.setattr("astrograph.data.trusted.importlib.util.find_spec", lambda name: object())
    audit, rows, leakage = audit_trusted_campaign(
        campaign, sat, dataset_archive_sha256="a" * 64, sat_agent_archive_sha256="b" * 64
    )
    assert audit.status == "PASS"
    assert audit.acceptance_valid_count == 12
    assert audit.acceptance_valid_pair_count == 6
    assert audit.public_trace_leakage_count == 0
    assert audit.reproduction_ready is True
    assert len(rows) == 12 and len(leakage) == 12


def test_regenerated_campaign_requires_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from astrograph.data.trusted import audit_trusted_campaign
    campaign, sat = _synthetic_trusted_campaign(tmp_path)
    monkeypatch.setattr("astrograph.data.trusted.importlib.util.find_spec", lambda name: None)
    audit, _, _ = audit_trusted_campaign(
        campaign, sat, dataset_archive_sha256="a", sat_agent_archive_sha256="b"
    )
    assert audit.status == "BLOCKED"
    assert audit.reproduction_ready is False


def test_regenerated_campaign_label_leak_is_invalid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from astrograph.data.trusted import audit_trusted_campaign
    campaign, sat = _synthetic_trusted_campaign(tmp_path, leak=True)
    monkeypatch.setattr("astrograph.data.trusted.importlib.util.find_spec", lambda name: object())
    audit, rows, _ = audit_trusted_campaign(
        campaign, sat, dataset_archive_sha256="a", sat_agent_archive_sha256="b"
    )
    assert audit.status == "BLOCKED"
    assert audit.public_trace_leakage_count == 12
    assert audit.acceptance_valid_count == 0
    assert all(row.acceptance_level == "INVALID" for row in rows)


def test_regenerated_campaign_requires_twelve_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from astrograph.data.trusted import audit_trusted_campaign
    campaign, sat = _synthetic_trusted_campaign(tmp_path, count_pairs=3)
    monkeypatch.setattr("astrograph.data.trusted.importlib.util.find_spec", lambda name: object())
    audit, _, _ = audit_trusted_campaign(
        campaign, sat, dataset_archive_sha256="a", sat_agent_archive_sha256="b"
    )
    assert audit.status == "BLOCKED"
    assert any("BELOW_MINIMUM" in blocker for blocker in audit.blockers)
