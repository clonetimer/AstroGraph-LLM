from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from astrograph.evaluation import llm_method_campaign as campaign
from astrograph.evaluation.llm_method_campaign import (
    build_confirmatory_dataset_snapshot_receipt,
    confirmatory_readiness,
    freeze_confirmatory_case_manifest,
    materialize_confirmatory_dataset_snapshot,
    verify_confirmatory_dataset_snapshot_receipt,
    write_confirmatory_snapshot_export_manifest,
    verify_confirmatory_snapshot_export_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def _runtime_admission() -> dict:
    h = "a" * 64
    return {
        "schema_version": "astrograph-confirmatory-runtime-identity-admission.v1",
        "status": "VALID",
        "runtime_identity": {
            "client_identity_sha256": h,
            "model_runtime_identity_sha256": h,
            "inference_config_sha256": h,
        },
        "runtime_identity_sha256": h,
        "client": {"identity": {
            "environment_lock_sha256": h, "environment_packages_sha256": h,
            "cyclonedx_sbom_sha256": h, "source_manifest_sha256": h,
            "source_checksums_sha256": h, "prompt_implementation_sha256": h,
            "inference_profile_file_sha256": h, "inference_profile_canonical_sha256": h,
        }},
        "model_runtime_attestation_sha256": h,
        "blockers": [],
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fault_map(path: str) -> dict[str, str]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))["mappings"]


def _make_roots(base: Path) -> tuple[Path, Path]:
    adcs_map = _fault_map("configs/datasets/adcs_v5_2_2/fault_class_map.json")
    phm_map = _fault_map("configs/datasets/phm_v6_0/fault_class_map.json")
    adcs_root = base / "adcs"
    phm_root = base / "phm"
    adcs_rows: list[dict[str, object]] = []
    for class_index, dataset_class in enumerate(adcs_map):
        for sample_index in (4, 9):
            pair = f"PAIR-{class_index}-{sample_index}"
            adcs_rows += [
                {
                    "sample_id": f"AF-{class_index}-{sample_index}", "trajectory_pair_id": pair,
                    "sample_index": sample_index, "scenario_id": "S", "diagnostic_class": dataset_class,
                    "telemetry_file": "fault.csv", "labels_file": "labels.json",
                },
                {
                    "sample_id": f"AN-{class_index}-{sample_index}", "trajectory_pair_id": pair,
                    "sample_index": sample_index, "scenario_id": "N", "diagnostic_class": "nominal",
                    "telemetry_file": "nominal.csv", "labels_file": "labels.json",
                },
            ]
    _write_csv(adcs_root / "sample_index.csv", adcs_rows)
    (adcs_root / "fault.csv").write_text("time_s,x\n0,1\n", encoding="utf-8")
    (adcs_root / "nominal.csv").write_text("time_s,x\n0,0\n", encoding="utf-8")
    (adcs_root / "labels.json").write_text('{"schema":"snapshot-test"}\n', encoding="utf-8")

    phm_rows: list[dict[str, object]] = []
    for class_index, dataset_class in enumerate(phm_map):
        for sample_index in (4, 9):
            common = {
                "sample_index": sample_index, "subsystem": f"SS-{class_index}",
                "activation_time_s": 1.0, "event_start_s": 1.0, "fault_manifestation_time_s": 1.0,
                "fault_injection_phase": "test", "telemetry_file": "telemetry.csv",
            }
            phm_rows += [
                {"sample_id": f"PF-{class_index}-{sample_index}", "condition_id": dataset_class, **common},
                {"sample_id": f"PN-{class_index}-{sample_index}", "condition_id": "nominal", **common},
            ]
    _write_csv(phm_root / "sample_index.csv", phm_rows)
    (phm_root / "telemetry.csv").write_text("time_s,x\n0,1\n", encoding="utf-8")
    return adcs_root, phm_root


def _ready(adcs_root: Path, phm_root: Path, manifest: Path, *, readonly: bool = False) -> dict:
    return confirmatory_readiness(
        ROOT,
        dataset_roots={"adcs": adcs_root, "phm": phm_root},
        live_model_available=True,
        live_model_detail="models=Qwen3.5-9B",
        requested_model_id="Qwen3.5-9B",
        observed_model_ids=["Qwen3.5-9B"],
        case_manifest_path=manifest,
        require_readonly_datasets=readonly,
    )


def test_content_addressed_snapshot_is_mount_path_independent_and_detects_payload_drift(tmp_path: Path):
    adcs_root, phm_root = _make_roots(tmp_path / "mount-a")
    manifest = tmp_path / "case_manifest.json"
    freeze_confirmatory_case_manifest(
        ROOT, output_path=manifest, dataset_roots={"adcs": adcs_root, "phm": phm_root}
    )
    readiness = _ready(adcs_root, phm_root, manifest)
    assert readiness["status"] == "READY"
    receipt = build_confirmatory_dataset_snapshot_receipt(readiness=readiness)
    assert receipt["status"] == "VALID"
    assert len(receipt["snapshot_sha256"]) == 64
    assert len(receipt["receipt_sha256"]) == 64
    assert receipt["datasets"]["adcs"]["payload_entries"]

    clone = tmp_path / "mount-b"
    shutil.copytree(tmp_path / "mount-a", clone)
    cloned_readiness = _ready(clone / "adcs", clone / "phm", manifest)
    cloned_receipt = build_confirmatory_dataset_snapshot_receipt(readiness=cloned_readiness)
    assert cloned_receipt["snapshot_sha256"] == receipt["snapshot_sha256"]
    stable = verify_confirmatory_dataset_snapshot_receipt(
        snapshot_receipt=receipt, current_readiness=cloned_readiness
    )
    assert stable["status"] == "STABLE"

    (clone / "adcs" / "fault.csv").write_text("time_s,x\n0,999\n", encoding="utf-8")
    drifted_readiness = _ready(clone / "adcs", clone / "phm", manifest)
    drifted = verify_confirmatory_dataset_snapshot_receipt(
        snapshot_receipt=receipt, current_readiness=drifted_readiness
    )
    assert drifted["status"] == "DRIFTED"
    assert "DATASET_SNAPSHOT_SHA256_DRIFT" in drifted["drift_codes"]
    assert "DATASET_CONTENT_SNAPSHOT_DRIFT:adcs" in drifted["drift_codes"]


def test_readonly_dataset_policy_is_explicit_and_fail_closed_when_requested(tmp_path: Path, monkeypatch):
    adcs_root, phm_root = _make_roots(tmp_path / "mount")
    manifest = tmp_path / "case_manifest.json"
    freeze_confirmatory_case_manifest(
        ROOT, output_path=manifest, dataset_roots={"adcs": adcs_root, "phm": phm_root}
    )
    monkeypatch.setattr(
        campaign, "_mount_access_status",
        lambda _path: {"status": "READ_WRITE", "mount_point": "/fixture", "options": ["rw"], "detail": None},
    )
    readiness = _ready(adcs_root, phm_root, manifest, readonly=True)
    assert readiness["status"] == "BLOCKED"
    assert readiness["readonly_dataset_policy"]["required"] is True
    assert readiness["readonly_dataset_policy"]["status"] == "UNSATISFIED"
    assert "DATASET_NOT_READ_ONLY:adcs" in readiness["blockers"]
    assert "DATASET_NOT_READ_ONLY:phm" in readiness["blockers"]
    receipt = build_confirmatory_dataset_snapshot_receipt(
        readiness=readiness, require_readonly_datasets=True
    )
    assert receipt["status"] == "INVALID"
    assert "DATASET_NOT_READ_ONLY:adcs" in receipt["errors"]


def test_execution_receipt_binds_snapshot_receipt_and_rejects_wrong_snapshot(tmp_path: Path):
    from astrograph.evaluation.llm_method_campaign import build_confirmatory_execution_receipt

    adcs_root, phm_root = _make_roots(tmp_path / "mount")
    manifest = tmp_path / "case_manifest.json"
    freeze_confirmatory_case_manifest(
        ROOT, output_path=manifest, dataset_roots={"adcs": adcs_root, "phm": phm_root}
    )
    readiness = _ready(adcs_root, phm_root, manifest)
    snapshot = build_confirmatory_dataset_snapshot_receipt(readiness=readiness)
    receipt = build_confirmatory_execution_receipt(
        ROOT,
        readiness=readiness,
        vllm_url="http://127.0.0.1:8000/v1",
        vllm_model="Qwen3.5-9B",
        dataset_snapshot_receipt=snapshot,
        runtime_identity_admission=_runtime_admission(),
    )
    assert receipt["dataset_snapshot_sha256"] == snapshot["snapshot_sha256"]
    assert receipt["dataset_snapshot_receipt_sha256"] == snapshot["receipt_sha256"]

    bad = json.loads(json.dumps(snapshot))
    bad["snapshot_sha256"] = "0" * 64
    try:
        build_confirmatory_execution_receipt(
            ROOT,
            readiness=readiness,
            vllm_url="http://127.0.0.1:8000/v1",
            vllm_model="Qwen3.5-9B",
            dataset_snapshot_receipt=bad,
            runtime_identity_admission=_runtime_admission(),
        )
    except ValueError as exc:
        assert "snapshot receipt" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("tampered snapshot receipt was admitted")


def test_materialized_snapshot_is_self_contained_content_identical_and_permission_hardened(tmp_path: Path):
    adcs_root, phm_root = _make_roots(tmp_path / "source")
    manifest = tmp_path / "case_manifest.json"
    freeze_confirmatory_case_manifest(
        ROOT, output_path=manifest, dataset_roots={"adcs": adcs_root, "phm": phm_root}
    )
    readiness = _ready(adcs_root, phm_root, manifest)
    frozen = build_confirmatory_dataset_snapshot_receipt(readiness=readiness)
    output = tmp_path / "materialized"
    receipt = materialize_confirmatory_dataset_snapshot(
        ROOT, snapshot_receipt=frozen, case_manifest_path=manifest, output_root=output
    )
    assert receipt["status"] == "VALID"
    assert receipt["source_snapshot_sha256"] == frozen["snapshot_sha256"]
    assert receipt["materialized_snapshot_sha256"] == frozen["snapshot_sha256"]
    assert receipt["permission_hardening"]["status"] == "READ_ONLY_MODE_BITS"
    assert (output / "case_manifest.json").is_file()
    assert (output / "datasets/adcs/sample_index.csv").is_file()
    assert (output / "datasets/phm/sample_index.csv").is_file()
    assert (output / "snapshot_metadata/materialized_dataset_snapshot_receipt.json").is_file()

    copied_readiness = confirmatory_readiness(
        ROOT,
        dataset_roots={"adcs": output / "datasets/adcs", "phm": output / "datasets/phm"},
        live_model_available=True, live_model_detail="models=Qwen3.5-9B",
        requested_model_id="Qwen3.5-9B", observed_model_ids=["Qwen3.5-9B"],
        case_manifest_path=output / "case_manifest.json",
    )
    assert copied_readiness["status"] == "READY"
    copied = build_confirmatory_dataset_snapshot_receipt(readiness=copied_readiness)
    assert copied["snapshot_sha256"] == frozen["snapshot_sha256"]


def test_materializer_fails_closed_if_source_bytes_drift_after_freeze(tmp_path: Path):
    adcs_root, phm_root = _make_roots(tmp_path / "source")
    manifest = tmp_path / "case_manifest.json"
    freeze_confirmatory_case_manifest(
        ROOT, output_path=manifest, dataset_roots={"adcs": adcs_root, "phm": phm_root}
    )
    frozen = build_confirmatory_dataset_snapshot_receipt(readiness=_ready(adcs_root, phm_root, manifest))
    (adcs_root / "fault.csv").write_text("time_s,x\n0,999\n", encoding="utf-8")
    output = tmp_path / "should-not-exist"
    import pytest
    with pytest.raises(ValueError, match="source dataset bytes"):
        materialize_confirmatory_dataset_snapshot(
            ROOT, snapshot_receipt=frozen, case_manifest_path=manifest, output_root=output
        )
    assert not output.exists()


def test_materializer_refuses_to_write_snapshot_inside_source_tree(tmp_path: Path):
    adcs_root, phm_root = _make_roots(tmp_path / "source")
    manifest = tmp_path / "case_manifest.json"
    freeze_confirmatory_case_manifest(
        ROOT, output_path=manifest, dataset_roots={"adcs": adcs_root, "phm": phm_root}
    )
    frozen = build_confirmatory_dataset_snapshot_receipt(readiness=_ready(adcs_root, phm_root, manifest))
    import pytest
    with pytest.raises(ValueError, match="outside the source tree"):
        materialize_confirmatory_dataset_snapshot(
            ROOT, snapshot_receipt=frozen, case_manifest_path=manifest, output_root=ROOT / ".forbidden-r9-snapshot"
        )
    assert not (ROOT / ".forbidden-r9-snapshot").exists()


def test_materializer_accepts_remounted_identical_source_roots_via_explicit_overrides(tmp_path: Path):
    adcs_root, phm_root = _make_roots(tmp_path / "mount-a")
    manifest = tmp_path / "case_manifest.json"
    freeze_confirmatory_case_manifest(
        ROOT, output_path=manifest, dataset_roots={"adcs": adcs_root, "phm": phm_root}
    )
    frozen = build_confirmatory_dataset_snapshot_receipt(readiness=_ready(adcs_root, phm_root, manifest))
    clone = tmp_path / "mount-b"
    shutil.copytree(tmp_path / "mount-a", clone)
    output = tmp_path / "materialized-from-remount"
    receipt = materialize_confirmatory_dataset_snapshot(
        ROOT, snapshot_receipt=frozen, case_manifest_path=manifest, output_root=output,
        dataset_roots={"adcs": clone / "adcs", "phm": clone / "phm"},
    )
    assert receipt["status"] == "VALID"
    assert receipt["materialized_snapshot_sha256"] == frozen["snapshot_sha256"]



def test_r10_export_manifest_is_path_independent_and_detects_copy_drift(tmp_path: Path):
    adcs_root, phm_root = _make_roots(tmp_path / "source")
    manifest = tmp_path / "case_manifest.json"
    freeze_confirmatory_case_manifest(
        ROOT, output_path=manifest, dataset_roots={"adcs": adcs_root, "phm": phm_root}
    )
    frozen = build_confirmatory_dataset_snapshot_receipt(readiness=_ready(adcs_root, phm_root, manifest))
    output = tmp_path / "materialized"
    materialize_confirmatory_dataset_snapshot(
        ROOT, snapshot_receipt=frozen, case_manifest_path=manifest, output_root=output
    )
    export_manifest = write_confirmatory_snapshot_export_manifest(output)
    assert export_manifest["status"] == "VALID"
    assert export_manifest["file_count"] > 0
    assert len(export_manifest["manifest_sha256"]) == 64
    assert len(export_manifest["content_manifest_sha256"]) == 64
    stable = verify_confirmatory_snapshot_export_manifest(output)
    assert stable["status"] == "STABLE"

    clone = tmp_path / "archive-copy"
    shutil.copytree(output, clone)
    cloned = verify_confirmatory_snapshot_export_manifest(clone)
    assert cloned["status"] == "STABLE"
    assert cloned["manifest_sha256"] == stable["manifest_sha256"]

    drift_target = clone / "datasets/adcs/fault.csv"
    drift_target.chmod(0o644)
    drift_target.write_text("time_s,x\n0,12345\n", encoding="utf-8")
    drifted = verify_confirmatory_snapshot_export_manifest(clone)
    assert drifted["status"] == "DRIFTED"
    assert "EXPORT_FILE_CONTENT_DRIFT" in drifted["drift_codes"]
    assert "datasets/adcs/fault.csv" in drifted["changed_files"]


def test_r10_export_manifest_detects_unexpected_files_and_self_tampering(tmp_path: Path):
    adcs_root, phm_root = _make_roots(tmp_path / "source")
    manifest = tmp_path / "case_manifest.json"
    freeze_confirmatory_case_manifest(
        ROOT, output_path=manifest, dataset_roots={"adcs": adcs_root, "phm": phm_root}
    )
    frozen = build_confirmatory_dataset_snapshot_receipt(readiness=_ready(adcs_root, phm_root, manifest))
    output = tmp_path / "materialized"
    materialize_confirmatory_dataset_snapshot(
        ROOT, snapshot_receipt=frozen, case_manifest_path=manifest, output_root=output
    )
    write_confirmatory_snapshot_export_manifest(output)
    # Make metadata writable for the test fixture; a real archive copy may not preserve mode bits either.
    output.chmod(0o755)
    (output / "datasets").chmod(0o755)
    (output / "datasets/adcs").chmod(0o755)
    (output / "datasets/adcs/extra.txt").write_text("unexpected", encoding="utf-8")
    unexpected = verify_confirmatory_snapshot_export_manifest(output)
    assert unexpected["status"] == "DRIFTED"
    assert "EXPORT_UNEXPECTED_FILE" in unexpected["drift_codes"]
    assert "datasets/adcs/extra.txt" in unexpected["unexpected_files"]

    (output / "datasets/adcs/extra.txt").unlink()
    manifest_path = output / "snapshot_metadata/export_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["file_count"] += 1
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tampered = verify_confirmatory_snapshot_export_manifest(output)
    assert tampered["status"] == "DRIFTED"
    assert "EXPORT_MANIFEST_SELF_HASH_MISMATCH" in tampered["drift_codes"]


def test_r10_export_manifest_rejects_symlink_in_snapshot_tree(tmp_path: Path):
    import pytest
    adcs_root, phm_root = _make_roots(tmp_path / "source")
    manifest = tmp_path / "case_manifest.json"
    freeze_confirmatory_case_manifest(
        ROOT, output_path=manifest, dataset_roots={"adcs": adcs_root, "phm": phm_root}
    )
    frozen = build_confirmatory_dataset_snapshot_receipt(readiness=_ready(adcs_root, phm_root, manifest))
    output_root = tmp_path / "materialized"
    materialize_confirmatory_dataset_snapshot(
        ROOT, snapshot_receipt=frozen, case_manifest_path=manifest, output_root=output_root
    )
    write_confirmatory_snapshot_export_manifest(output_root)
    # Reopen the directory only so the test can inject an adversarial symlink.
    output_root.chmod(0o755)
    target = output_root / "case_manifest.json"
    link = output_root / "case_manifest-link.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable in this environment")
    result = verify_confirmatory_snapshot_export_manifest(output_root)
    assert result["status"] == "DRIFTED"
    assert "EXPORT_TREE_CONTAINS_SYMLINK" in result["drift_codes"]
