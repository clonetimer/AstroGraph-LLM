"""R11-B3: OAG post-result explanation is outside the native hash-chained harness."""
from __future__ import annotations

import json
from pathlib import Path

from astrograph.integrations.platform_knowledge import (
    PlatformKnowledgeConfig, PlatformKnowledgeEnrichmentService, PlatformKnowledgeExplanation,
)
from astrograph.integrations.upload_report import UploadReportJobStore
from astrograph.oag.cli import demo_input
from astrograph.oag.contracts import OAGExplanation
from astrograph.oag.harness import verify_run


ROOT = Path(__file__).resolve().parents[1]


class DualAgentBackend:
    """Deterministic local protocol double; this is NOT a real LLM run."""
    def __init__(self):
        self.calls = []
        self.last_call_metadata = {}

    def generate_structured(self, *, agent_id, output_model, payload):
        self.calls.append((agent_id, payload))
        if agent_id == "oag-diagnosis-explanation":
            fault = payload["summary"]["suspected_fault_id"]
            return OAGExplanation(
                candidate_fault_id=fault, abstained=fault is None,
                assessment=payload["required_assessment"],
            )
        if agent_id == "platform-knowledge-explanation":
            fault = payload["diagnosis"]["candidate_fault_id"]
            return PlatformKnowledgeExplanation(
                candidate_fault_id=fault, abstained=fault is None,
                summary="A frozen OAG candidate requires independent verification.",
                claims=[], limitations=["This is reference-only context."],
            )
        raise AssertionError("Unexpected model capability: " + agent_id)


def _job(tmp_path, backend=None):
    service = PlatformKnowledgeEnrichmentService(
        config=PlatformKnowledgeConfig(a2o_enabled=False, material_tool_calls_enabled=True),
        llm_backend=backend,
    )
    store = UploadReportJobStore(tmp_path / "jobs", knowledge_service=service)
    job = store.create("Explain frozen OAG evidence and alternatives")
    store.put_file(job["job_id"], "input.json", demo_input(ROOT).model_dump_json().encode())
    store.put_file(job["job_id"], "reference.md",
                   b"Reaction wheel speed and friction require additional evidence.\n")
    return store, job["job_id"]


def test_oag_post_explanation_uses_separate_model_call_without_mutating_harness(tmp_path):
    backend = DualAgentBackend()
    store, job_id = _job(tmp_path, backend)
    report = store.run(job_id)
    assert report["status"] == "SUCCEEDED", report.get("error")
    analysis = report["analysis"]
    original = analysis["oag_result"]
    assert verify_run(Path(original["harness"]["run_directory"]))["status"] == "PASS"
    assert [name for name, _ in backend.calls] == [
        "oag-diagnosis-explanation", "platform-knowledge-explanation"
    ]
    k = analysis["knowledge_enrichment"]
    assert k["post_diagnosis_binding"]["status"] == "NOT_AVAILABLE"
    assert k["post_diagnosis_binding"]["source_kind"] == "OAG_DIAGNOSTIC"
    assert k["diagnostic_candidate_can_be_overridden"] is False
    assert k["generation"]["status"] == "SUCCEEDED"
    payload = backend.calls[-1][1]
    assert payload["diagnosis"]["candidate_fault_id"] == original["output"]["summary"]["suspected_fault_id"]
    material = payload.get("material_reasoning") or {}
    assert material.get("reviewed_causal_chains") is None
    assert (material.get("tool_investigation") or {}).get("status") != "SUCCEEDED"
    assert "独立后诊断 RAG/LLM" in store.report_markdown_path(job_id).read_text(encoding="utf-8")
    stored = json.loads((store._job_root(job_id) / "output" / "oag_post_diagnosis.json").read_text())
    assert stored["generation"]["status"] == "SUCCEEDED"


def test_oag_post_explanation_failure_cannot_erase_completed_diagnosis(tmp_path, monkeypatch):
    import astrograph.integrations.upload_report as module
    store, job_id = _job(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("internal details must not leak")

    monkeypatch.setattr(module, "run_frozen_post_diagnosis_explanation", fail)
    report = store.run(job_id)
    assert report["status"] == "SUCCEEDED", report.get("error")
    knowledge = report["analysis"]["knowledge_enrichment"]
    assert knowledge["status"] == "FAILED_CLOSED"
    assert knowledge["error"] == {"type": "RuntimeError"}
    assert verify_run(Path(report["analysis"]["oag_result"]["harness"]["run_directory"]))["status"] == "PASS"
    assert "internal details must not leak" not in json.dumps(report)


def test_oag_without_llm_preserves_native_harness(tmp_path):
    store, job_id = _job(tmp_path)
    report = store.run(job_id)
    assert report["status"] == "SUCCEEDED", report.get("error")
    assert report["analysis"]["knowledge_enrichment"]["generation"]["status"] == "NOT_CONFIGURED"
    assert verify_run(Path(report["analysis"]["oag_result"]["harness"]["run_directory"]))["status"] == "PASS"
