from __future__ import annotations

from copy import deepcopy

from astrograph.integrations.claim_governance import build_claim_governance
from astrograph.integrations.external_diagnosis_analysis import ExternalDiagnosisAnalysis


def fixture():
    sources = [
        {"source_id": "source-a", "role": "diagnosis_results", "filename": "diagnosis.txt",
         "locator": "line:1", "sha256": "a" * 64, "text": "报告记录反作用轮轮速出现波动。"},
        {"source_id": "source-b", "role": "log_reports", "filename": "events.log",
         "locator": "line:7", "sha256": "b" * 64, "text": "同一时段遥测链路存在延迟。"},
    ]
    analysis = ExternalDiagnosisAnalysis.model_validate({
        "summary": "当前原因仍需核查。",
        "findings": [{"text": "原报告记录轮速出现波动。", "epistemic_status": "REPORTED",
                      "citation_ids": ["source-a"]}],
        "hypotheses": [{"title": "遥测链路延迟", "analysis": "可能影响观测时序。",
                        "citation_ids": ["source-b"], "opposing_citation_ids": ["source-a"]}],
        "information_gaps": ["缺少独立轮速测量。"],
        "recommended_review_actions": ["对照独立测量记录。"],
    })
    return sources, analysis


class Backend:
    def __init__(self, model_id, build=None):
        self.model_id, self.build, self.calls = model_id, build, []

    def generate_structured(self, *, agent_id, output_model, payload):
        self.calls.append((agent_id, deepcopy(payload)))
        if self.build:
            return self.build(output_model, payload)
        return output_model.model_validate({"assessments": [{
            "claim_id": claim["claim_id"], "support_status": "SUPPORTED",
            "rationale": "所给来源明确包含与该主张一致的文字。",
            "supporting_source_ids": ["source-a"], "counter_evidence_source_ids": [],
            "information_gaps": [],
        } for claim in payload["claims"]]})


def run(sources, analysis, primary, verifier):
    return build_claim_governance(
        task_id="JOB-CLAIM-1", input_sha256="c" * 64, analysis=analysis,
        sources=sources, analysis_backend=primary, verification_backend=verifier,
    )


def test_claim_registry_binds_source_receipts_and_keeps_expert_state_pending():
    sources, analysis = fixture()
    result = run(sources, analysis, Backend("primary"), None)
    assert result["schema_version"] == "astrograph-claim-governance.v1"
    assert len(result["claims"]) == 2
    reported = next(item for item in result["claims"] if item["claim_type"] == "FINDING")
    hypothesis = next(item for item in result["claims"] if item["claim_type"] == "HYPOTHESIS")
    assert reported["epistemic_status"] == "REPORTED"
    assert reported["evidence_refs"][0]["sha256"] == "a" * 64
    assert hypothesis["epistemic_status"] == "INFERRED"
    assert hypothesis["opposing_evidence_refs"][0]["source_id"] == "source-a"
    assert all(item["expert_review_status"] == "PENDING" for item in result["claims"])
    assert result["verification"]["status"] == "NOT_CONFIGURED"
    assert result["verification"]["fact_truth_certified"] is False


def test_independent_verifier_sees_claims_and_original_sources_but_not_generator_reasoning():
    sources, analysis = fixture()
    primary, verifier = Backend("analysis-model"), Backend("review-model")
    result = run(sources, analysis, primary, verifier)
    assert len(verifier.calls) == 1
    agent_id, payload = verifier.calls[0]
    assert agent_id == "external-diagnosis-evidence-verifier"
    assert "summary" not in payload and "hypotheses" not in payload
    assert all("citation_ids" not in claim for claim in payload["claims"])
    assert payload["sources"][0]["text"] == sources[0]["text"]
    assert result["verification"]["status"] == "COMPLETED_REQUIRES_EXPERT_REVIEW"
    assert result["verification"]["verifier_model"] == "review-model"
    assert result["verification"]["fact_truth_certified"] is False
    assert all(item["expert_review_status"] == "PENDING" for item in result["claims"])
    assert all(item["verification_status"] == "SUPPORTED" for item in result["claims"])


def test_same_model_identity_is_not_accepted_as_independent_verification():
    sources, analysis = fixture()
    primary, verifier = Backend("same-model"), Backend("same-model")
    result = run(sources, analysis, primary, verifier)
    assert verifier.calls == []
    assert result["verification"]["status"] == "NOT_CONFIGURED"
    assert result["verification"]["reason_code"] == "VERIFIER_NOT_INDEPENDENT"
    assert all(item["verification_status"] == "NOT_EVALUATED" for item in result["claims"])


def test_verifier_cannot_forge_sources_or_omit_claim_assessments():
    sources, analysis = fixture()
    def forged(output_model, payload):
        first = payload["claims"][0]
        return output_model.model_validate({"assessments": [{
            "claim_id": first["claim_id"], "support_status": "SUPPORTED",
            "rationale": "伪造引用。", "supporting_source_ids": ["foreign-source"],
            "counter_evidence_source_ids": [], "information_gaps": [],
        }]})
    result = run(sources, analysis, Backend("primary"), Backend("reviewer", forged))
    assert result["verification"]["status"] == "FAILED"
    assert result["verification"]["reason_code"] == "VERIFIER_CITATION_OR_CLAIM_BINDING_INVALID"
    assert all(item["verification_status"] == "NOT_EVALUATED" for item in result["claims"])


def test_verifier_status_requires_matching_evidence():
    sources, analysis = fixture()
    def unsupported(output_model, payload):
        return output_model.model_validate({"assessments": [{
            "claim_id": claim["claim_id"], "support_status": "CONTRADICTED",
            "rationale": "没有给出反向来源。", "supporting_source_ids": [],
            "counter_evidence_source_ids": [], "information_gaps": [],
        } for claim in payload["claims"]]})
    result = run(sources, analysis, Backend("primary"), Backend("reviewer", unsupported))
    assert result["verification"]["status"] == "FAILED"
    assert result["verification"]["reason_code"] == "VERIFIER_ASSESSMENT_EVIDENCE_MISSING"
