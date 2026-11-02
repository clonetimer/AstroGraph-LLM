"""Regression coverage for the selectively imported LLM transport fixes."""
from collections import UserDict
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread

import pytest
from pydantic import BaseModel, ConfigDict

from astrograph.integrations.bounded_http import request_json
from astrograph.integrations.platform_llm_probe import _structured
from astrograph.llm.openai_compatible import (
    OpenAICompatibleConfig, OpenAICompatibleStructuredBackend, StructuredBackendError,
)
from astrograph.llm.token_budget import count_chat_tokens


class _Reply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str


def _completion(*, refusal=None, finish_reason="stop"):
    message = {"role": "assistant", "content": '{"status":"OK"}'}
    if refusal is not None:
        message["refusal"] = refusal
    return {"id": "fixture", "model": "fixture", "choices": [
        {"index": 0, "message": message, "finish_reason": finish_reason},
    ]}


@contextmanager
def _provider(*documents, redirect=None):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append((self.path, dict(self.headers), json.loads(body)))
            if redirect is not None:
                self.send_response(307)
                self.send_header("Location", redirect)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            raw = json.dumps(documents[min(len(requests) - 1, len(documents) - 1)],
                             ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_worker_returns_chinese_utf8_even_with_gbk_stdio(monkeypatch):
    monkeypatch.setenv("PYTHONIOENCODING", "gbk")
    with _provider({"analysis": "温度升高，原因尚待核查。"}) as (url, _requests):
        result = request_json(url, payload={"task": "分析中文报告"}, timeout_s=5)
    assert result == {"analysis": "温度升高，原因尚待核查。"}


@pytest.mark.parametrize("wrapped", [False, True], ids=["dict-tensor", "mapping-tensor"])
def test_tokenizer_counts_input_ids_wrapped_tensor(wrapped):
    class Tensor:
        def tolist(self):
            return [[10, 11, 12, 13]]

    class Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            result = {"input_ids": Tensor()}
            return UserDict(result) if wrapped else result

    assert count_chat_tokens(Tokenizer(), [{"role": "user", "content": "报告"}]) == 4


@pytest.mark.parametrize("rejection", [
    {"refusal": "private provider rejection"}, {"finish_reason": "content_filter"},
], ids=["refusal", "content-filter"])
def test_generation_refusal_stops_repairs_and_format_fallback(rejection):
    with _provider(_completion(**rejection), _completion()) as (url, requests):
        backend = OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(
            base_url=url, model="fixture", tokenizer_mode="disabled", timeout_s=5,
            response_format_strategy="auto", repair_attempts=2, budget_preflight=False,
        ))
        with pytest.raises(StructuredBackendError, match="MODEL_REFUSAL") as caught:
            backend.generate_structured(agent_id="external-diagnosis-analysis",
                                        output_model=_Reply, payload={})
    assert type(caught.value).__name__ == "ModelRefusalError"
    assert len(requests) == 1
    assert backend.last_call_metadata["status"] == "REFUSED"
    assert "private provider rejection" not in str(caught.value)


@pytest.mark.parametrize("rejection", [
    {"refusal": "private provider rejection"}, {"finish_reason": "content_filter"},
], ids=["refusal", "content-filter"])
def test_probe_refusal_stops_format_fallback(rejection):
    with _provider(_completion(**rejection), _completion()) as (url, requests):
        result = _structured(url, "fixture", "secret", 5, "vllm", "auto")
    assert result["status"] == "FAILED"
    assert len(requests) == 1
    assert len(result["attempts"]) == 1
    assert result["error"]["type"] == "ModelRefusalError"
    assert "private provider rejection" not in json.dumps(result)


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens"])
def test_probe_rejects_complete_json_when_provider_marks_truncated(finish_reason):
    with _provider(_completion(finish_reason=finish_reason)) as (url, _requests):
        result = _structured(url, "fixture", "secret", 5, "vllm", "json_schema")
    assert result["status"] == "FAILED"
    assert result["error"]["message"] == "STRUCTURED_PROBE_OUTPUT_TRUNCATED"


def test_probe_still_accepts_normal_completion():
    with _provider(_completion()) as (url, requests):
        result = _structured(url, "fixture", "secret", 5, "vllm", "auto")
    assert result["status"] == "SUCCEEDED"
    assert len(requests) == 1


def test_worker_does_not_forward_credentials_to_redirect_target():
    with _provider({"status": "unexpected"}) as (target, target_requests):
        with _provider(redirect=target) as (source, source_requests):
            with pytest.raises(RuntimeError, match="HTTP_REDIRECT_NOT_ALLOWED"):
                request_json(source, payload={}, headers={"Authorization": "Bearer private"}, timeout_s=5)
    assert len(source_requests) == 1
    assert target_requests == []
