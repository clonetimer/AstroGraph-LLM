"""Material-first customer delivery acceptance and safety regression tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from astrograph.api import create_app
from astrograph.integrations.platform_cli import main as cli_main
from astrograph.integrations.result_log_review import (
    analyze_result_log_review,
    load_manifest,
    read_lines,
    validate_review_inputs,
)
from astrograph.integrations.upload_report import UploadReportJobStore


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "result_log_review"


def _files() -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in EXAMPLE.iterdir() if p.is_file()}


def _stage(store: UploadReportJobStore, content: dict[str, bytes]) -> str:
    job_id = store.create("依据外部诊断、日志和手册分析原因、建议措施")["job_id"]
    for name, blob in content.items():
        store.put_file(job_id, name, blob)
    return job_id


def test_material_review_store_preflight_report_and_provenance(tmp_path: Path) -> None:
    store = UploadReportJobStore(tmp_path / "jobs")
    job_id = _stage(store, _files())
    preflight = store.preflight(job_id)
    assert preflight["can_start"] is True
    assert preflight["artifact_kind"] == "RESULT_LOG_REVIEW"
    report = store.run(job_id)
    assert report["status"] == "SUCCEEDED"
    assert report["artifact_kind"] == "RESULT_LOG_REVIEW"
    review = report["analysis"]["result_log_review"]
    assert review["top_fault_id"] == "ADCS_RW_FRICTION_INCREASE"
    assert review["status"] == "REVIEW_REQUIRED"
    assert review["cause_hypotheses"]
    assert review["handling_measures"]
    assert review["verification_checks"]
    assert all(x["control_authorized"] is False for x in review["handling_measures"])
    assert all(x["basis_ids"] for x in review["handling_measures"])
    assert {x["role"] for x in review["evidence"]} == {"diagnosis_results", "log_reports", "references"}
    assert {x["role"] for x in review["source_manifest"]} == {"diagnosis_results", "log_reports", "references"}
    for item in review["source_manifest"]:
        assert item["sha256"] == hashlib.sha256(_files()[item["filename"]]).hexdigest()
    assert report["formal_diagnosis_publishable"] is False
    assert report["analysis"]["effect_guard"]["raw_telemetry_detection_executed"] is False
    assert report["visualization"]["status"] == "NOT_APPLICABLE_DOCUMENT_ONLY"
    assert store.report(job_id) == report
    assert "处置措施" in store.report_markdown_path(job_id).read_text(encoding="utf-8")
    maintenance = store.maintenance(job_id)
    assert maintenance["diagnosis_disposition"] == "REVIEW_REQUIRED"
    assert maintenance["material_review"] == review


def test_json_citation_locator_points_to_original_uploaded_line(tmp_path: Path) -> None:
    evidence = tmp_path / "result.json"
    evidence.write_text(' { "fault_id" : "ADCS_RW_JAM", "note" : "first" }\n', encoding="utf-8")
    lines = read_lines(evidence, "diagnosis_results")
    assert lines[0].locator == "line:1"
    assert lines[0].text.startswith('{ "fault_id"')


def test_manifest_missing_role_traversal_and_unlisted_file_rejected(tmp_path: Path) -> None:
    for name, data in _files().items():
        (tmp_path / name).write_bytes(data)
    manifest_path = tmp_path / "analysis_input.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["references"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(tmp_path, manifest_path)

    manifest["references"] = ["../external.md"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(tmp_path, manifest_path)

    manifest["references"] = ["reference.md"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "unexpected.txt").write_text("not authorized as a role", encoding="utf-8")
    with pytest.raises(ValueError, match="unlisted"):
        validate_review_inputs(tmp_path, manifest_path)


def test_missing_reference_fails_preflight(tmp_path: Path) -> None:
    store = UploadReportJobStore(tmp_path / "jobs")
    content = _files()
    del content["reference.md"]
    job_id = _stage(store, content)
    preflight = store.preflight(job_id)
    assert preflight["can_start"] is False
    assert preflight["status"] == "BLOCKED"


def test_unknown_upstream_fault_kept_without_made_up_mechanism(tmp_path: Path) -> None:
    content = _files()
    content["diagnosis_result.json"] = b'{"fault_id": "ADCS_RW_JAM", "review_status": "UNCONFIRMED"}'
    content["operations.log"] = b'2026-10-08T01:00Z ERROR ADCS_RW_JAM reported by monitor\n'
    content["reference.md"] = b'ADCS_RW_JAM: operator review required; this is just a sample.'
    for name, data in content.items():
        (tmp_path / name).write_bytes(data)
    review = analyze_result_log_review(tmp_path, tmp_path / "analysis_input.json")
    assert review["top_fault_id"] == "ADCS_RW_JAM"
    assert review["candidates"][0]["status"] == "PROVISIONAL_UNCORROBORATED"
    assert "具体物理根因尚无" in review["candidates"][0]["cause_hypotheses"][0]["text"]
    assert any("不在已审规则目录" in x for x in review["evidence_gaps"])
    assert len(review["evidence"]) == 3


def test_only_log_contains_fault_cannot_invent_upstream_candidate(tmp_path: Path) -> None:
    content = _files()
    content["diagnosis_result.json"] = b'{"status": "normal", "summary": "no diagnosed fault"}'
    content["operations.log"] = b'ADCS_RW_FRICTION_INCREASE motor_current_a elevated\n'
    for name, data in content.items():
        (tmp_path / name).write_bytes(data)
    review = analyze_result_log_review(tmp_path, tmp_path / "analysis_input.json")
    assert review["top_fault_id"] is None
    assert review["status"] == "INSUFFICIENT_EVIDENCE"
    assert not review["cause_hypotheses"]
    assert review["handling_measures"]


def test_upstream_only_label_without_known_code_remains_visible(tmp_path: Path) -> None:
    content = _files()
    content["diagnosis_result.json"] = json.dumps({"fault_name": "反作用轮卡滞", "note": "未经验证的诊断意见"}, ensure_ascii=False).encode("utf-8")
    for name, data in content.items():
        (tmp_path / name).write_bytes(data)
    review = analyze_result_log_review(tmp_path, tmp_path / "analysis_input.json")
    assert review["candidates"][0]["fault_name"] == "反作用轮卡滞"
    assert review["top_fault_id"].startswith("UPSTREAM_UNMAPPED_")


def test_api_upload_preflight_run_and_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASTROGRAPH_PLATFORM_ROOT", str(tmp_path / "api-jobs"))
    client = TestClient(create_app())
    capabilities = client.get("/v1/platform/capabilities").json()
    assert "RESULT_LOG_REVIEW" in {x["kind"] for x in capabilities["supported_artifact_sets"]}
    created = client.post("/v1/platform/jobs", json={"task": "复核诊断材料"})
    assert created.status_code == 200
    job_id = created.json()["job_id"]
    for name, data in _files().items():
        assert client.put(f"/v1/platform/jobs/{job_id}/files/{name}", content=data).status_code == 200
    preflight = client.post(f"/v1/platform/jobs/{job_id}/preflight")
    assert preflight.status_code == 200
    assert preflight.json()["can_start"] is True
    run = client.post(f"/v1/platform/jobs/{job_id}/run")
    assert run.status_code == 200
    assert run.json()["status"] == "SUCCEEDED"
    report = client.get(f"/v1/platform/jobs/{job_id}/report")
    assert report.json()["analysis"]["result_log_review"]["top_fault_id"] == "ADCS_RW_FRICTION_INCREASE"
    md = client.get(f"/v1/platform/jobs/{job_id}/report.md")
    assert md.status_code == 200
    assert "故障原因" in md.text


def test_cli_review_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["--root", str(tmp_path / "cli-jobs"), "review",
            "--result", str(EXAMPLE / "diagnosis_result.json"),
            "--log", str(EXAMPLE / "operations.log"),
            "--reference", str(EXAMPLE / "reference.md")]
    assert cli_main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "SUCCEEDED"
    assert result["artifact_kind"] == "RESULT_LOG_REVIEW"
    assert result["candidate"] == "ADCS_RW_FRICTION_INCREASE"
