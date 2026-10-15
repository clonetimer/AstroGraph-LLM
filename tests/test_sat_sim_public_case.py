from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from astrograph.integrations.sat_sim_public_case import diagnose_public_campaign, diagnose_public_case


def _case(tmp_path: Path, *, field: str = "eps.battery.soc") -> Path:
    root = tmp_path / "public" / "case-001"
    root.mkdir(parents=True)
    telemetry = root / "telemetry.csv"
    with telemetry.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_s", field])
        writer.writeheader(); writer.writerows([{"time_s": 0, field: 0.8}, {"time_s": 1, field: 0.7}])
    manifest = {"schema_version": "sat-sim.astrograph-public-case.v1", "case_id": "case-001", "telemetry": {"file": "telemetry.csv", "sha256": hashlib.sha256(telemetry.read_bytes()).hexdigest(), "rows": 2, "fields": ["time_s", field]}}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_public_case_diagnosis_is_truth_blind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.read_text
    def guarded(self: Path, *args, **kwargs):
        if "truth" in str(self).lower():
            raise AssertionError("truth path accessed")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded)
    closure = diagnose_public_case(_case(tmp_path), output_root=tmp_path / "diagnosis")
    assert closure["diagnosis_provenance"]["labels_used"] is False
    assert closure["diagnosis_provenance"]["truth_used"] is False
    assert closure["formal_diagnosis_publishable"] is False


def test_public_case_rejects_answer_proxy_and_hash_tamper(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="answer-proxy"):
        diagnose_public_case(_case(tmp_path, field="object.eps.soc"), output_root=tmp_path / "bad")
    root = _case(tmp_path / "second")
    (root / "telemetry.csv").write_text("time_s,eps.battery.soc\n0,0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        diagnose_public_case(root, output_root=tmp_path / "tampered")


def test_public_case_rejects_modifier_fields_and_proxy_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="answer-proxy fields"):
        diagnose_public_case(_case(tmp_path / "modifier", field="modifier.active_count"), output_root=tmp_path / "bad-modifier")
    root = _case(tmp_path / "value", field="status")
    telemetry = root / "telemetry.csv"
    telemetry.write_text("time_s,status\n0,event.capacity_loss_pct\n1,nominal\n", encoding="utf-8")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["telemetry"]["sha256"] = hashlib.sha256(telemetry.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="answer-proxy values"):
        diagnose_public_case(root, output_root=tmp_path / "bad-value")


def test_public_campaign_requires_exact_count_and_truth_blind_closure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = _case(tmp_path / "first")
    second = _case(tmp_path / "second")
    public = tmp_path / "campaign"
    public.mkdir()
    first.rename(public / "case-001")
    second.rename(public / "case-002")
    second_manifest = public / "case-002" / "manifest.json"
    payload = json.loads(second_manifest.read_text(encoding="utf-8"))
    payload["case_id"] = "case-002"
    second_manifest.write_text(json.dumps(payload), encoding="utf-8")

    original = Path.read_text
    def guarded(self: Path, *args, **kwargs):
        if "truth" in str(self).lower():
            raise AssertionError("truth path accessed")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded)

    report = diagnose_public_campaign(public, output_root=tmp_path / "out", expected_case_count=2)
    assert report["status"] == "PASS"
    assert report["completed_case_count"] == 2
    assert report["truth_root_accessed"] is False
    assert report["public_answer_proxy_rejection_count"] == 0
    with pytest.raises(ValueError, match="expected 3"):
        diagnose_public_campaign(public, output_root=tmp_path / "bad-count", expected_case_count=3)


def test_public_case_records_explicit_model_manifest_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = tmp_path / "catalog.json"
    open_set = tmp_path / "open-set.json"
    catalog.write_text("{}", encoding="utf-8")
    open_set.write_text("{}", encoding="utf-8")
    original = diagnose_public_case.__globals__["UnifiedDiagnosisService"].diagnose_paths
    captured: dict[str, Path | None] = {}

    def wrapped(self, **kwargs):
        captured["catalog"] = kwargs.get("catalog_model_path")
        captured["open_set"] = kwargs.get("open_set_model_path")
        kwargs["catalog_model_path"] = None
        kwargs["open_set_model_path"] = None
        return original(self, **kwargs)

    monkeypatch.setattr(diagnose_public_case.__globals__["UnifiedDiagnosisService"], "diagnose_paths", wrapped)
    closure = diagnose_public_case(
        _case(tmp_path),
        output_root=tmp_path / "diagnosis",
        catalog_model_path=catalog,
        open_set_model_path=open_set,
    )
    assert captured == {"catalog": catalog.resolve(), "open_set": open_set.resolve()}
    assert closure["diagnosis_provenance"]["catalog_model_manifest_sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert closure["diagnosis_provenance"]["open_set_model_manifest_sha256"] == hashlib.sha256(b"{}").hexdigest()


def test_simulation_handoff_v2_fails_closed_on_bad_provenance_or_gate(tmp_path: Path) -> None:
    root = _case(tmp_path)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update({
        "schema_version": "simulation_handoff.v2",
        "provenance": {
            name: "0" * 64 for name in (
                "source_result_sha256", "source_nominal_sha256", "source_telemetry_sha256",
                "config_sha256", "seed_sha256",
            )
        },
        "qualification": {"source_validation": "PASS", "physics_gate": "FAIL"},
    })
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="qualification"):
        diagnose_public_case(root, output_root=tmp_path / "bad-gate")

    manifest["qualification"]["physics_gate"] = "PASS"
    manifest["provenance"]["seed_sha256"] = "not-a-hash"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance hashes"):
        diagnose_public_case(root, output_root=tmp_path / "bad-hash")
