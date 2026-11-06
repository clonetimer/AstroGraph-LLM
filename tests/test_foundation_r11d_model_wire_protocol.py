"""R11-D loopback HTTP protocol test; scripted responses are NOT live LLM inference."""
from __future__ import annotations
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

from astrograph.integrations.platform_knowledge import PlatformKnowledgeConfig, PlatformKnowledgeEnrichmentService
from astrograph.llm import OpenAICompatibleConfig, OpenAICompatibleStructuredBackend


def material(group="BENCH-RUN"):
    chain = {
        "path_id": "BENCH-PATH", "hop_count": 2, "terminal_symptom_id": "TEMP_RISE",
        "joint_applicability_status": "REVIEWED_SEQUENTIAL_CONSTRAINTS_COMPATIBLE_FLIGHT_UNVERIFIED",
        "reviewed_context_consistency": {"status": "CONFLICT", "conflict_reasons": ["OPERATING_MODE_DISJOINT"]},
        "reviewed_temporal_consistency": {"status": "REVIEWED_SEQUENCE_COMPATIBLE_NOT_FLIGHT_VERIFIED",
                                          "cumulative_lag_seconds": {"min": 8, "max": 16}, "conflict_reasons": []},
        "hops": [{"edge_id": "BENCH-EDGE", "direction": "FORWARD", "source_name": "wear",
                  "target_name": "heat", "source_sha256": "a" * 64,
                  "source_locators": ["synthetic:1"], "evidence_quote": "Only a synthetic bench illustration."}],
    }
    return {
        "source_group_id": group, "competing_hypothesis_review": {"status": "COMPARE_ONLY"},
        "candidates": [
            {"fault_id": "PUMP_A", "multi_hop": {"causal_chain_review": {
                "chains": [chain], "blocked": [{"edge_id": "BENCH-EDGE-2",
                "reasons": ["TEMPORAL_STATE_TRANSITION_INCOMPATIBLE"]}]}}},
            {"fault_id": "PUMP_B", "multi_hop": {"causal_chain_review": {"chains": [], "blocked": []}}},
        ],
    }


def snapshot():
    return {"candidate_fault_id": "PUMP_A", "candidate_fault_ids": ["PUMP_A", "PUMP_B"],
            "material_source_group_id": "BENCH-RUN", "evidence": []}


@contextmanager
def wire_server(*, selection="valid", generated="valid"):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append({"body": request, "auth": self.headers.get("Authorization"),
                          "rid": self.headers.get("X-Request-Id")})
            user = json.loads(request["messages"][-1]["content"])["payload"]
            if "frozen_candidate_ids" in user and "available_tools" in user:
                if selection == "empty":
                    value = {"actions": []}
                elif selection == "unknown":
                    value = {"actions": [{"tool_id": "material.graph.check_context",
                                          "fault_id": "FOREIGN", "path_id": "BENCH-PATH"}]}
                elif selection == "invalid_schema":
                    value = {"actions": [{"tool_id": "material.graph.drop_database"}]}
                else:
                    value = {"actions": [
                        {"tool_id": "material.graph.search_causal_paths", "fault_id": "PUMP_A"},
                        {"tool_id": "material.graph.check_context", "fault_id": "PUMP_A", "path_id": "BENCH-PATH"},
                        {"tool_id": "material.graph.check_temporal", "fault_id": "PUMP_A", "path_id": "BENCH-PATH"},
                    ]}
            else:
                value = {
                    "candidate_fault_id": "FOREIGN" if generated == "override" else "PUMP_A",
                    "abstained": False,
                    "summary": "The frozen pump diagnosis requires independent confirmation.",
                    "claims": [], "limitations": ["Synthetic bench reference; not current-flight proof."],
                    "recommended_review_actions": [],
                }
            encoded = json.dumps({"choices": [{"message": {"content": json.dumps(value)},
                                                "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 20, "completion_tokens": 25}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=srv.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{srv.server_port}/v1", calls
    finally:
        srv.shutdown()
        srv.server_close()
        worker.join(timeout=3)


def invoke(tmp_path, url, review=None):
    backend = OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(
        provider="lmstudio", base_url=url, model="scripted-wire-protocol-not-a-model",
        response_format_strategy="json_object", repair_attempts=0,
        max_http_attempts=2, timeout_s=10, budget_preflight=False,
    ))
    service = PlatformKnowledgeEnrichmentService(
        PlatformKnowledgeConfig(a2o_enabled=False, material_tool_calls_enabled=True),
        llm_backend=backend,
    )
    report = service.enrich(task="Compare synthetic lab observations without claiming root-cause proof.",
                            diagnosis_snapshot=snapshot(), document_paths=[],
                            output_root=tmp_path / "knowledge",
                            material_review_context=material() if review is None else review)
    return report, backend


def test_real_http_two_pass_tool_receipts_and_temporal_reasoning(tmp_path):
    with wire_server() as (url, calls):
        report, backend = invoke(tmp_path, url)
    assert len(calls) == 2
    assert all(x["auth"] == "Bearer lm-studio" and x["rid"].startswith("astrograph-") for x in calls)
    assert all(x["body"]["response_format"] == {"type": "json_object"} for x in calls)
    assert report["material_binding"]["status"] == "BOUND_FROZEN_DIAGNOSIS"
    assert report["generation"]["status"] == "SUCCEEDED"
    investigation = report["generation"]["material_tool_investigation"]
    assert [x["status"] for x in investigation["calls"]] == ["SUCCEEDED"] * 3
    audit = investigation["review_coverage_audit"]
    assert audit["status"] == "REVIEW_COVERAGE_INCOMPLETE"
    assert audit["checks_matching_opportunities"] == 3
    assert audit["review_gaps"][0]["tool_id"] == "material.rag.compare_hypotheses"
    assert investigation["calls"][1]["output"]["observed_flight_conditions_verified"] is False
    assert investigation["calls"][2]["output"]["actual_telemetry_event_alignment_verified"] is False
    final = json.loads(calls[-1]["body"]["messages"][-1]["content"])["payload"]
    assert final["material_reasoning"]["tool_investigation"]["calls"][2]["output"]["reviewed_temporal_consistency"]["cumulative_lag_seconds"]["min"] == 8
    assert backend.last_call_metadata["status"] == "SUCCEEDED"
    assert report["diagnostic_candidate_can_be_overridden"] is False


def test_wire_foreign_fault_is_denied_but_safe_explanation_completes(tmp_path):
    with wire_server(selection="unknown") as (url, calls):
        report, _ = invoke(tmp_path, url)
    assert len(calls) == 2
    event = report["generation"]["material_tool_investigation"]["calls"][0]
    assert event["status"] == "DENIED"
    assert event["reason"] == "ARGUMENT_NOT_IN_FROZEN_REVIEW"
    assert report["generation"]["status"] == "SUCCEEDED"


def test_wire_forged_tool_id_fails_closed_and_cannot_modify_graph(tmp_path):
    with wire_server(selection="invalid_schema") as (url, calls):
        report, _ = invoke(tmp_path, url)
    assert len(calls) == 2
    assert report["generation"]["material_tool_investigation"]["status"] == "FAILED_CLOSED"
    assert report["generation"]["material_tool_investigation"]["diagnostic_vote_allowed"] is False
    assert report["generation"]["status"] == "SUCCEEDED"


def test_wire_model_generated_foreign_candidate_is_rejected(tmp_path):
    with wire_server(generated="override") as (url, calls):
        report, _ = invoke(tmp_path, url)
    assert len(calls) == 2
    assert report["generation"]["status"] == "FAILED"
    assert report["generation"]["explanation"] is None
    assert report["diagnostic_candidate_can_be_overridden"] is False


def test_wire_cross_task_review_cannot_reach_tool_selector(tmp_path):
    with wire_server() as (url, calls):
        report, _ = invoke(tmp_path, url, material("FOREIGN-RUN"))
    assert len(calls) == 1
    assert report["material_binding"]["status"] == "DENIED_TASK_BINDING"
    assert report["generation"]["status"] == "SUCCEEDED"
    user = json.loads(calls[0]["body"]["messages"][-1]["content"])["payload"]
    assert user["material_reasoning"]["tool_investigation"]["status"] == "DENIED_TASK_BINDING"
    assert user["material_reasoning"].get("reviewed_causal_chains") is None


def test_wire_model_no_tool_selection_has_explicit_missed_review_receipt(tmp_path):
    with wire_server(selection="empty") as (url, calls):
        report, _ = invoke(tmp_path, url)
    assert len(calls) == 2
    investigation = report["generation"]["material_tool_investigation"]
    audit = investigation["review_coverage_audit"]
    assert audit["status"] == "MODEL_DECLINED_AVAILABLE_REVIEW"
    assert audit["checks_missing_in_visible_list"] >= 3
    assert not investigation["current_root_cause_proven"]
    final = json.loads(calls[-1]["body"]["messages"][-1]["content"])["payload"]
    assert final["material_reasoning"]["tool_investigation"]["review_coverage_audit"] == audit
