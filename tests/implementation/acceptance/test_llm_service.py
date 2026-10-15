"""Local HTTP fixtures test the setup helper, NOT real model capability."""
from __future__ import annotations
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import pytest

from scripts.acceptance import llm_service as helper


@pytest.fixture
def local_root(tmp_path, monkeypatch):
    root = tmp_path / "source"
    (root / "configs/agent").mkdir(parents=True)
    (root / "configs/agent/mission.default.json").write_text(json.dumps({
        "planner": "deterministic", "max_checks": 3, "max_optional_actions": 2,
        "allow_external_literature": False, "approved_model_sha256": ["a" * 64],
    }))
    monkeypatch.setattr(helper, "ROOT", root)
    return root


def configure(root, *extra):
    return helper.main(["configure", "--model", "HTTP_PROTOCOL_FIXTURE", *extra])


def test_local_config_retains_permissions_and_no_service_claim(local_root, capsys):
    assert configure(local_root) == 0
    llm = helper.read_json(local_root / "configs/agent/llm.local.json")
    host = helper.read_json(local_root / "configs/agent/mission.llm.local.json")
    assert llm["api_key"] == "local-placeholder"
    assert host["planner"] == "llm_assisted"
    assert host["approved_model_sha256"] == ["a" * 64]
    assert host["allow_external_literature"] is False
    assert "CONFIGURED_NOT_LIVE_VERIFIED" in capsys.readouterr().out


def test_config_does_not_overwrite(local_root):
    assert configure(local_root) == 0
    p = local_root / "configs/agent/llm.local.json"
    old = p.read_bytes()
    assert configure(local_root) == 2
    assert p.read_bytes() == old


def test_key_resolved_explicitly_and_never_printed(local_root, monkeypatch, capsys):
    secret = "test-only-not-a-real-key-abc123"
    monkeypatch.setenv("ASTROGRAPH_TEST_KEY", secret)
    assert configure(local_root, "--api-key-env", "ASTROGRAPH_TEST_KEY") == 0
    assert helper.read_json(local_root / "configs/agent/llm.local.json")["api_key"] == secret
    assert secret not in capsys.readouterr().out


def test_missing_env_stops_before_writing(local_root, monkeypatch):
    monkeypatch.delenv("NOT_SET_TEST_KEY", raising=False)
    assert configure(local_root, "--api-key-env", "NOT_SET_TEST_KEY") == 2
    assert not (local_root / "configs/agent/llm.local.json").exists()


@pytest.mark.parametrize("url", ["http://untrusted.test/v1", "http://127.0.0.1:1234/api/v1",
                                    "http://user:secret@127.0.0.1/v1", "http://127.0.0.1/v1?token=x"])
def test_bad_endpoints_rejected(url):
    with pytest.raises(ValueError):
        helper.validate_endpoint(url)


def test_remote_requires_named_key(local_root):
    assert configure(local_root, "--base-url", "https://unused.invalid/v1") == 2
    assert not (local_root / "configs/agent/llm.local.json").exists()


@contextmanager
def fixture_server(answer, *, finish_reason="stop", catalog_ok=True):
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, None, self.headers.get("Authorization")))
            if not catalog_ok:
                self.send_error(404)
                return
            self.respond({"data": [{"id": "HTTP_PROTOCOL_FIXTURE"}]})
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body, self.headers.get("Authorization")))
            self.respond({"choices": [{"message": {"content": json.dumps(answer)}, "finish_reason": finish_reason}]})
        def respond(self, payload):
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_real_transport_contract_not_mocked_backend(local_root, tmp_path):
    with fixture_server({"task": "research", "actions": [], "rationale": "Protocol fixture only."}) as (url, seen):
        assert configure(local_root, "--base-url", url) == 0
        receipt = tmp_path / "preflight.json"
        assert helper.main(["check", "--settings", str(local_root / "configs/agent/mission.llm.local.json"),
                            "--output", str(receipt)]) == 0
        r = helper.read_json(receipt)
        assert r["status"] == "PASSED"
        assert r["model"] == "HTTP_PROTOCOL_FIXTURE"
        assert r["diagnostic_accuracy_evaluated"] is False
        assert r["sends_telemetry"] is False
        assert any(x[0] == "/v1/chat/completions" for x in seen)
        assert any(x[1] and x[1]["response_format"]["type"] == "json_schema" for x in seen)
        assert helper.main(["check", "--settings", str(local_root / "configs/agent/mission.llm.local.json"),
                            "--output", str(receipt)]) == 2


@pytest.mark.parametrize("answer,finish", [({"invented": "not-a-plan"}, "stop"),
                                          ({"task": "research", "actions": [], "rationale": "Truncated"}, "length")])
def test_schema_or_truncation_fallback_is_not_success(local_root, tmp_path, answer, finish):
    with fixture_server(answer, finish_reason=finish) as (url, _):
        assert configure(local_root, "--base-url", url) == 0
        receipt = tmp_path / "failure.json"
        assert helper.main(["check", "--settings", str(local_root / "configs/agent/mission.llm.local.json"),
                            "--output", str(receipt)]) == 2
        r = helper.read_json(receipt)
        assert r["status"] == "FAILED"
        assert any(x["outcome"] == "DETERMINISTIC_FALLBACK" for x in r["records"])


def test_optional_model_catalog_does_not_replace_chat_validation(local_root, tmp_path):
    with fixture_server({"task": "research", "actions": [], "rationale": "Fixture."}, catalog_ok=False) as (url, _):
        assert configure(local_root, "--base-url", url) == 0
        receipt = tmp_path / "no-catalog.json"
        assert helper.main(["check", "--settings", str(local_root / "configs/agent/mission.llm.local.json"),
                            "--output", str(receipt)]) == 0
        r = helper.read_json(receipt)
        assert r["model_catalog"]["status"] == "UNAVAILABLE"
        assert r["llm_calls"] == 1


def test_manual_env_placeholder_is_rejected(local_root):
    assert configure(local_root) == 0
    llm_path = local_root / "configs/agent/llm.local.json"
    llm = helper.read_json(llm_path)
    llm["api_key"] = "${ASTROGRAPH_API_KEY}"
    llm_path.write_text(json.dumps(llm))
    with pytest.raises(ValueError, match="does not expand"):
        helper.probe(local_root / "configs/agent/mission.llm.local.json")
