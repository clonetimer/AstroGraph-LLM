from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from pydantic import BaseModel, ConfigDict

from astrograph.llm import OpenAICompatibleConfig, OpenAICompatibleStructuredBackend
from astrograph.llm.openai_compatible import OutputBudgetError
from astrograph.llm.token_budget import TokenizerLoadError


class TinyOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str


class FixedChatTokenizer:
    def __init__(self, count: int) -> None:
        self.count = count
        self.last_messages: list[dict[str, str]] | None = None

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        self.last_messages = conversation
        return list(range(self.count))


class UsageHandler(BaseHTTPRequestHandler):
    prompt_tokens = 500

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        assert body["messages"][0]["role"] == "system"
        content = json.dumps({"value": "ok"}).encode()
        payload = json.dumps({
            "choices": [{"finish_reason": "stop", "message": {"content": content.decode()}}],
            "usage": {
                "prompt_tokens": type(self).prompt_tokens,
                "completion_tokens": 7,
                "total_tokens": type(self).prompt_tokens + 7,
            },
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return


def _config(base_url: str = "http://127.0.0.1:1/v1", **overrides):
    data = {
        "base_url": base_url,
        "model": "fake",
        "response_format_strategy": "plain_json",
        "max_tokens": 600,
        "repair_attempts": 0,
        "tokenizer_mode": "required",
        "prompt_token_safety_margin": 1.0,
        "prompt_token_reserve": 32,
    }
    data.update(overrides)
    return OpenAICompatibleConfig(**data)


def test_tokenizer_budget_exact_fit_and_server_calibration():
    tokenizer = FixedChatTokenizer(500)
    UsageHandler.prompt_tokens = 505
    server = HTTPServer(("127.0.0.1", 0), UsageHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend = OpenAICompatibleStructuredBackend(
            _config(
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                context_length=1132,
            ),
            tokenizer=tokenizer,
        )
        result = backend.generate_structured(
            agent_id="budget-test-agent",
            output_model=TinyOutput,
            payload={"message": "包含最终JSON Schema的真实messages应由chat template计数"},
        )
        assert result.value == "ok"
        measurement = backend.last_call_metadata["output_budget"]["prompt_token_measurement"]
        assert measurement["method"] == "tokenizer_chat_template"
        assert measurement["tokenizer_prompt_tokens"] == 500
        assert measurement["guarded_prompt_tokens"] == 532
        assert measurement["server_prompt_tokens"] == 505
        assert measurement["tokenizer_relative_error_vs_server"] == pytest.approx(-5 / 505, abs=1e-6)
        assert tokenizer.last_messages and "output_json_schema" in tokenizer.last_messages[1]["content"]
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_tokenizer_budget_rejects_one_token_over_context():
    backend = OpenAICompatibleStructuredBackend(
        _config(context_length=1131),
        tokenizer=FixedChatTokenizer(500),
    )
    with pytest.raises(OutputBudgetError):
        backend.generate_structured(
            agent_id="budget-test-agent",
            output_model=TinyOutput,
            payload={"message": "x"},
        )
    budget = backend.last_call_metadata["output_budget"]
    assert budget["guarded_prompt_tokens"] == 532
    assert budget["effective_max_tokens"] == 600


def test_tokenizer_budget_auto_expands_to_required_output():
    backend = OpenAICompatibleStructuredBackend(
        _config(
            max_tokens=128,
            budget_policy="auto_expand",
            max_tokens_ceiling=600,
            context_length=1092,
        ),
        tokenizer=FixedChatTokenizer(500),
    )
    messages = backend._messages(output_model=TinyOutput, payload={"message": "x"})
    budget = backend._budget(payload={"message": "x"}, messages=messages, request_hypothesis_count=1)
    assert budget["required_output_tokens"] == 560
    assert budget["effective_max_tokens"] == 560
    assert budget["auto_expanded"] is True
    assert budget["guarded_prompt_tokens"] + budget["effective_max_tokens"] == 1092


def test_required_tokenizer_unavailable_fails_closed():
    backend = OpenAICompatibleStructuredBackend(_config(tokenizer_path=None))
    with pytest.raises(TokenizerLoadError):
        backend.generate_structured(
            agent_id="budget-test-agent",
            output_model=TinyOutput,
            payload={"message": "x"},
        )
    assert backend.last_call_metadata["status"] == "BUDGET_REJECTED"


def test_auto_mode_falls_back_but_records_tokenizer_error():
    backend = OpenAICompatibleStructuredBackend(_config(
        tokenizer_mode="auto",
        context_length=None,
        tokenizer_path=None,
    ))
    messages = backend._messages(output_model=TinyOutput, payload={"message": "x"})
    budget = backend._budget(payload={"message": "x"}, messages=messages, request_hypothesis_count=1)
    measurement = budget["prompt_token_measurement"]
    assert measurement["method"] == "rough_messages_chars_per_3_5"
    assert measurement["tokenizer_prompt_tokens"] is None
    assert "tokenizer_path" in measurement["tokenizer_error"]
