from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from astrograph.investigation import HypothesisSet
from astrograph.llm import OpenAICompatibleConfig, OpenAICompatibleStructuredBackend
from astrograph.llm.openai_compatible import OutputBudgetError, StructuredBackendError


def _reference(count: int = 16) -> dict:
    hypotheses = []
    for index in range(count):
        fault_id = f"FAULT_{index:02d}"
        supported = index == 3
        hypotheses.append({
            "hypothesis_id": f"HYP-{index:02d}",
            "hypothesis_type": "REGISTERED_FAULT",
            "target_fault_id": fault_id,
            "description": fault_id,
            "status": "SUPPORTED" if supported else "CONTRADICTED",
            "supporting_evidence_fact_ids": ["EF-SUP"] if supported else [],
            "contradicting_evidence_fact_ids": [] if supported else [f"EF-CON-{index:02d}"],
            "missing_evidence_definition_ids": [],
            "not_applicable_evidence_definition_ids": [],
            "reasoning_summary": "reference",
            "claim_type": "INFERRED",
            "member_fault_ids": [],
            "interaction_hypothesis": "",
            "verification_requirements": [],
            "research_metadata": {},
        })
    hypotheses.append({
        "hypothesis_id": "HYP-NONE",
        "hypothesis_type": "NO_REGISTERED_FAULT",
        "target_fault_id": None,
        "description": "none",
        "status": "CONTRADICTED",
        "supporting_evidence_fact_ids": [],
        "contradicting_evidence_fact_ids": ["EF-SUP"],
        "missing_evidence_definition_ids": [],
        "not_applicable_evidence_definition_ids": [],
        "reasoning_summary": "fault supported",
        "claim_type": "INFERRED",
        "member_fault_ids": [],
        "interaction_hypothesis": "",
        "verification_requirements": [],
        "research_metadata": {},
    })
    return {
        "hypothesis_set_id": "HS-REF",
        "episode_id": "EP-1",
        "generated_by_agent_id": "diagnostic-reasoner-agent",
        "hypotheses": hypotheses,
    }


class CompactHandler(BaseHTTPRequestHandler):
    calls = 0
    truncate_first = False
    request_schemas = []
    malformed_shadow = False
    omit_last_candidate = False

    def do_POST(self):
        type(self).calls += 1
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        user = json.loads(body["messages"][1]["content"])
        type(self).request_schemas.append(body.get("response_format", {}).get("json_schema", {}).get("name", "plain_json"))
        selected = user["payload"].get("selected_candidate_fault_ids", [])
        proposal = {
            "hypotheses": [
                {
                    "hypothesis_type": "REGISTERED_FAULT",
                    "target_fault_id": fault_id,
                    "status": "SUPPORTED" if fault_id == "FAULT_03" else "CONTRADICTED",
                    "supporting_evidence_fact_ids": ["EF-SUP"] if fault_id == "FAULT_03" else [],
                    "contradicting_evidence_fact_ids": [] if fault_id == "FAULT_03" else [f"EF-CON-{int(fault_id[-2:]):02d}"],
                    "missing_evidence_definition_ids": [],
                    "reasoning_summary": "compact proposal",
                }
                for fault_id in (selected[:-1] if type(self).omit_last_candidate else selected)
            ]
        }
        if user["payload"].get("capture_shadow_candidate_revision"):
            proposal["shadow_candidate_revision"] = "malformed" if type(self).malformed_shadow else {
                "candidate_fault_ids": selected[:1],
                "abstain_recommended": False,
                "cited_observation_ids": [],
            }
        content = json.dumps(proposal)
        finish_reason = "stop"
        if type(self).truncate_first and type(self).calls == 1:
            content = content[: max(1, len(content) // 2)]
            finish_reason = "length"
        payload = json.dumps({
            "choices": [{"finish_reason": finish_reason, "message": {"content": content}}]
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return


def _payload() -> dict:
    reference = _reference()
    definitions = []
    facts = []
    for index in range(16):
        definition_id = f"ED-{index:02d}"
        fault_id = f"FAULT_{index:02d}"
        definitions.append({
            "evidence_definition_id": definition_id,
            "target_fault_id": fault_id,
        })
        facts.append({
            "evidence_fact_id": "EF-SUP" if index == 3 else f"EF-CON-{index:02d}",
            "evidence_definition_id": definition_id,
        })
    return {
        "reference_hypothesis_set": reference,
        "registered_fault_ids": [f"FAULT_{i:02d}" for i in range(16)],
        "evidence_definitions": definitions,
        "evidence_facts": facts,
        "diagnostic_protocol": {"faults": {f"FAULT_{i:02d}": {} for i in range(16)}},
    }


def test_compact_protocol_compiles_complete_hypothesis_set():
    CompactHandler.calls = 0
    CompactHandler.truncate_first = False
    server = HTTPServer(("127.0.0.1", 0), CompactHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend = OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="fake",
            prompt_profile="evidence_protocol_v059_compact",
            response_format_strategy="plain_json",
            include_reference_hypothesis_set=True,
            candidate_top_k=4,
            max_tokens=2400,
            repair_attempts=0,
        ))
        result = backend.generate_structured(
            agent_id="diagnostic-reasoner-agent",
            output_model=HypothesisSet,
            payload=_payload(),
        )
        assert len(result.hypotheses) == 17
        assert result.validation_metadata["proposal_protocol"] == "compact-v1"
        assert result.validation_metadata["llm_proposal_count"] == 4
        assert result.validation_metadata["deterministically_completed_hypothesis_count"] == 13
        assert backend.last_call_metadata["compact_protocol"] is True
        assert len(backend.last_call_metadata["selected_candidate_fault_ids"]) == 4
        assert backend.last_call_metadata["response_chars"] < 1800
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_a2_a8_shadow_schema_reuses_one_compact_reasoning_call_each():
    CompactHandler.calls = 0
    CompactHandler.truncate_first = False
    CompactHandler.request_schemas = []
    CompactHandler.malformed_shadow = False
    server = HTTPServer(("127.0.0.1", 0), CompactHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend = OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(
            base_url=f"http://127.0.0.1:{server.server_port}/v1", model="fake",
            prompt_profile="evidence_protocol_v059_compact",
            response_format_strategy="json_schema", include_reference_hypothesis_set=True,
            candidate_top_k=4, max_tokens=2400, repair_attempts=0,
        ))
        for arm in ("A2", "A8"):
            payload = _payload()
            payload["capture_shadow_candidate_revision"] = True
            payload["llm_diagnostic_deliberation"] = {"arm": arm}
            result = backend.generate_structured(
                agent_id="diagnostic-reasoner-agent", output_model=HypothesisSet, payload=payload,
            )
            assert len(result.hypotheses) == 17
            assert backend.last_call_metadata["shadow_candidate_revision_proposal"]["candidate_fault_ids"]
            assert backend.last_call_metadata["compact_protocol"] is True
        assert CompactHandler.calls == 2
        assert CompactHandler.request_schemas == ["_CompactReasonerProposalSetWithShadow"] * 2
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_malformed_shadow_wire_does_not_invalidate_formal_compact_hypotheses():
    CompactHandler.calls = 0
    CompactHandler.truncate_first = False
    CompactHandler.malformed_shadow = True
    server = HTTPServer(("127.0.0.1", 0), CompactHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend = OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(
            base_url=f"http://127.0.0.1:{server.server_port}/v1", model="fake",
            prompt_profile="evidence_protocol_v059_compact", response_format_strategy="plain_json",
            include_reference_hypothesis_set=True, candidate_top_k=4,
            max_tokens=2400, repair_attempts=0,
        ))
        payload = _payload()
        payload["capture_shadow_candidate_revision"] = True
        result = backend.generate_structured(
            agent_id="diagnostic-reasoner-agent", output_model=HypothesisSet, payload=payload,
        )
        assert len(result.hypotheses) == 17
        assert backend.last_call_metadata["shadow_candidate_revision_proposal"] == "malformed"
        assert CompactHandler.calls == 1
    finally:
        CompactHandler.malformed_shadow = False
        server.shutdown()
        thread.join(timeout=2)


@pytest.mark.parametrize("exploratory", [False, True])
def test_missing_compact_candidate_only_degrades_exploratory_shadow(exploratory):
    CompactHandler.calls = 0
    CompactHandler.truncate_first = False
    CompactHandler.omit_last_candidate = True
    server = HTTPServer(("127.0.0.1", 0), CompactHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend = OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(
            base_url=f"http://127.0.0.1:{server.server_port}/v1", model="fake",
            prompt_profile="evidence_protocol_v059_compact", response_format_strategy="plain_json",
            include_reference_hypothesis_set=True, candidate_top_k=4,
            max_tokens=2400, repair_attempts=0,
        ))
        payload = _payload()
        if exploratory:
            payload["capture_shadow_candidate_revision"] = True
            result = backend.generate_structured(
                agent_id="diagnostic-reasoner-agent", output_model=HypothesisSet, payload=payload,
            )
            assert len(result.hypotheses) == 17
            assert result.validation_metadata["exploratory_compact_proposal_degraded"] is True
            assert len(result.validation_metadata["exploratory_missing_compact_candidate_fault_ids"]) == 1
        else:
            with pytest.raises(StructuredBackendError, match="compact model must return exactly selected candidates"):
                backend.generate_structured(
                    agent_id="diagnostic-reasoner-agent", output_model=HypothesisSet, payload=payload,
                )
    finally:
        CompactHandler.omit_last_candidate = False
        server.shutdown()
        thread.join(timeout=2)


def test_full_protocol_budget_preflight_rejects_short_output_budget():
    backend = OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(
        model="fake",
        prompt_profile="evidence_protocol_v052",
        response_format_strategy="plain_json",
        max_tokens=2400,
        budget_preflight=True,
        budget_policy="fail",
    ))
    with pytest.raises(OutputBudgetError):
        backend.generate_structured(
            agent_id="diagnostic-reasoner-agent",
            output_model=HypothesisSet,
            payload=_payload(),
        )
    assert backend.last_call_metadata["status"] == "BUDGET_REJECTED"


def test_compact_fusion_drops_window_sized_provider_lineage_after_selection():
    backend = OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(
        model='fake', prompt_profile='evidence_protocol_v059_compact', candidate_top_k=1,
    ))
    marker = 'WINDOW-LINEAGE-' * 20000
    ranking = {
        'schema_version': 'v', 'ranking_id': 'R', 'episode_id': 'EP-1',
        'source': 'DATA_DRIVEN', 'score_semantics': 'RANKING_SCORE',
        'producer_id': 'P', 'producer_version': '1', 'metadata': {'raw': marker},
        'kg_fault_coverage': {'raw': marker}, 'candidates': [{
            'fault_id': 'FAULT_03', 'rank': 1, 'score': 0.9, 'source': 'DATA_DRIVEN',
            'evidence_fact_ids': ['EF-SUP'], 'source_evidence_ids': ['SOURCE-1'],
            'correlation_group_ids': [marker], 'metadata': {'raw': marker},
            'lineage_complete': True, 'contributes_to_fusion': True,
        }],
    }
    compact = backend._compact_candidate_fusion({
        'data_ranking': ranking, 'forward_ranking': ranking, 'backward_ranking': ranking,
        'birrf_result': {'candidates': [{'fault_id': 'FAULT_03'}]},
        'registered_fault_ids': ['FAULT_03'],
    }, {'FAULT_03'})
    rendered = json.dumps(compact)
    assert 'WINDOW-LINEAGE' not in rendered
    assert compact['data_ranking']['candidates'][0]['fault_id'] == 'FAULT_03'


def test_compact_retrieval_keeps_selected_and_experience_context_only():
    compact = OpenAICompatibleStructuredBackend._compact_pre_reasoning_retrieval({
        'package_id': 'R-1', 'status': 'SUFFICIENT',
        'local_hits': [
            {'chunk_id': 'selected', 'source_id': 'S-1', 'lane': 'LOCAL_DOCUMENT',
             'matched_concept_ids': ['fault:F-1'], 'snippet': 'x' * 500},
            {'chunk_id': 'other', 'source_id': 'S-2', 'lane': 'LOCAL_DOCUMENT',
             'matched_concept_ids': ['fault:F-2'], 'snippet': 'other'},
            {'chunk_id': 'experience', 'source_id': 'experience:E-1', 'lane': 'EXPERIENCE',
             'matched_concept_ids': ['fault:F-1'], 'snippet': 'lesson'},
            {'chunk_id': 'unrelated-experience', 'source_id': 'experience:E-2',
             'lane': 'EXPERIENCE', 'matched_concept_ids': ['fault:F-2'],
             'snippet': 'unrelated lesson'},
        ],
        'ontology_paths': [], 'experience_hit_count': 1,
        'knowledge_update_candidates': [{'active_graph_modified': False}],
        'active_graph_modified': False,
    }, {'F-1'})
    assert [item['chunk_id'] for item in compact['local_hits']] == ['selected', 'experience']
    assert len(compact['local_hits'][0]['snippet']) == 320
    assert compact['knowledge_update_candidates'][0]['active_graph_modified'] is False


def test_compact_retrieval_preserves_explicit_reviewed_tbox_seed_paths():
    path = {
        'seed_concept_id': 'mechanism:thermal-exchange-degradation',
        'target_concept_id': 'effect:thermal-trend-deviation',
        'hops': [{'relation_id': 'edge:approved', 'source_concept_id':
                  'mechanism:thermal-exchange-degradation', 'relation_type': 'MANIFESTS_AS',
                  'target_concept_id': 'effect:thermal-trend-deviation'}],
    }
    audit = {
        'retrieval_only': True, 'diagnostic_evidence_allowed': False,
        'runtime_vote_allowed': False, 'active_graph_modified': False,
    }
    compact = OpenAICompatibleStructuredBackend._compact_pre_reasoning_retrieval({
        'package_id': 'R-TBOX', 'status': 'SUFFICIENT', 'local_hits': [],
        'ontology_seed_concept_ids': ['fault:F-1', 'mechanism:thermal-exchange-degradation'],
        'ontology_asset_audits': [audit], 'ontology_paths': [path],
        'active_graph_modified': False,
    }, {'F-1'})
    assert compact['ontology_paths'] == [path]
    assert compact['ontology_asset_audits'] == [audit]


def test_finish_reason_length_is_detected_and_auto_expanded():
    CompactHandler.calls = 0
    CompactHandler.truncate_first = True
    server = HTTPServer(("127.0.0.1", 0), CompactHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend = OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="fake",
            prompt_profile="evidence_protocol_v059_compact",
            response_format_strategy="plain_json",
            include_reference_hypothesis_set=True,
            candidate_top_k=4,
            max_tokens=1200,
            max_tokens_ceiling=4096,
            budget_policy="auto_expand",
            repair_attempts=1,
        ))
        result = backend.generate_structured(
            agent_id="diagnostic-reasoner-agent",
            output_model=HypothesisSet,
            payload=_payload(),
        )
        assert len(result.hypotheses) == 17
        assert CompactHandler.calls == 2
        assert backend.last_call_metadata["finish_reasons"] == ["length", "stop"]
        assert backend.last_call_metadata["attempts"] == 2
    finally:
        server.shutdown()
        thread.join(timeout=2)
