from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from astrograph.llm import OpenAICompatibleConfig, OpenAICompatibleStructuredBackend


class Probe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    value: int


class Handler(BaseHTTPRequestHandler):
    request_ids: list[str | None] = []

    def do_POST(self):
        self.__class__.request_ids.append(self.headers.get("X-Request-Id"))
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        content = json.dumps({"status": "OK", "value": 7})
        payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return


def test_openai_compatible_backend_parses_and_validates():
    Handler.request_ids = []
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = OpenAICompatibleConfig(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="fake-model",
            response_format_strategy="plain_json",
            repair_attempts=0,
        )
        backend = OpenAICompatibleStructuredBackend(config)
        result = backend.generate_structured(agent_id="test", output_model=Probe, payload={"x": 1})
        assert result.status == "OK"
        assert result.value == 7
        assert backend.last_call_metadata["status"] == "SUCCEEDED"
        assert Handler.request_ids == [backend.last_call_metadata["request_id"]]
        assert Handler.request_ids[0].startswith("astrograph-")
        assert backend.last_call_metadata["request_ids"] == Handler.request_ids
    finally:
        server.shutdown()
        thread.join(timeout=2)

from astrograph.investigation import HypothesisSet


class ProposalHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        content = json.dumps({
            "hypotheses": [
                {
                    "hypothesis_type": "REGISTERED_FAULT",
                    "target_fault_id": "FAULT_A",
                    "status": "SUPPORTED",
                    "supporting_evidence_fact_ids": ["EF-1"],
                    "contradicting_evidence_fact_ids": [],
                    "missing_evidence_definition_ids": [],
                    "reasoning_summary": "evidence supports fault A"
                },
                {
                    "hypothesis_type": "NO_REGISTERED_FAULT",
                    "target_fault_id": None,
                    "status": "CONTRADICTED",
                    "supporting_evidence_fact_ids": [],
                    "contradicting_evidence_fact_ids": ["EF-1"],
                    "missing_evidence_definition_ids": [],
                    "reasoning_summary": "registered fault is supported"
                }
            ]
        })
        payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return


def test_reasoner_proposal_is_compiled_into_full_hypothesis_set():
    reference = {
        "hypothesis_set_id": "HS-REF",
        "episode_id": "EP-1",
        "generated_by_agent_id": "diagnostic-reasoner-agent",
        "hypotheses": [
            {
                "hypothesis_id": "HYP-A",
                "hypothesis_type": "REGISTERED_FAULT",
                "target_fault_id": "FAULT_A",
                "description": "candidate A",
                "status": "PROPOSED",
                "supporting_evidence_fact_ids": [],
                "contradicting_evidence_fact_ids": [],
                "missing_evidence_definition_ids": [],
                "not_applicable_evidence_definition_ids": [],
                "reasoning_summary": "",
                "claim_type": "INFERRED"
            },
            {
                "hypothesis_id": "HYP-NONE",
                "hypothesis_type": "NO_REGISTERED_FAULT",
                "target_fault_id": None,
                "description": "none",
                "status": "PROPOSED",
                "supporting_evidence_fact_ids": [],
                "contradicting_evidence_fact_ids": [],
                "missing_evidence_definition_ids": [],
                "not_applicable_evidence_definition_ids": [],
                "reasoning_summary": "",
                "claim_type": "INFERRED"
            }
        ]
    }
    server = HTTPServer(("127.0.0.1", 0), ProposalHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend = OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="fake-reasoner",
            response_format_strategy="plain_json",
            repair_attempts=0,
            include_reference_hypothesis_set=False,
        ))
        result = backend.generate_structured(
            agent_id="diagnostic-reasoner-agent",
            output_model=HypothesisSet,
            payload={"reference_hypothesis_set": reference},
        )
        assert result.hypotheses[0].hypothesis_id == "HYP-A"
        assert result.hypotheses[0].status.value == "SUPPORTED"
        assert backend.last_call_metadata["compiled_from_proposal"] is True
        assert backend.last_call_metadata["reference_included"] is False
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_v052_prompt_profile_is_recorded():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = OpenAICompatibleConfig(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="fake-model",
            response_format_strategy="plain_json",
            repair_attempts=0,
            prompt_profile="evidence_protocol_v052",
        )
        backend = OpenAICompatibleStructuredBackend(config)
        backend.generate_structured(agent_id="test", output_model=Probe, payload={"x": 1})
        assert backend.last_call_metadata["prompt_profile"] == "evidence_protocol_v052"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_lmstudio_auto_structured_strategy_prefers_host_validated_json_object():
    backend = OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(
        provider="lmstudio",
        model="qwen/qwen3.5-9b",
        response_format_strategy="auto",
        repair_attempts=0,
    ))
    assert backend._strategies() == ["json_object", "plain_json", "json_schema"]
