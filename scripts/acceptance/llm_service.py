#!/usr/bin/env python3
"""Configure and probe a user-operated LLM; never starts/downloads a model service.

Run from the AstroGraph repository root. Generated *.local.json files can contain
API credentials; they must not be committed or included in shared result bundles.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import sys
from typing import Any
import urllib.error
import urllib.request
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a JSON object")
    return value


def validate_endpoint(base_url: str) -> bool:
    """Return whether the endpoint is loopback; do not accept secrets in URLs."""
    u = urlsplit(base_url)
    if u.scheme not in {"http", "https"} or not u.hostname or u.query or u.fragment or u.username or u.password:
        raise ValueError("Use an absolute http(s) base URL without credentials, query or fragment")
    try:
        loopback = ipaddress.ip_address(u.hostname).is_loopback
    except ValueError:
        loopback = u.hostname.casefold() == "localhost"
    if u.scheme == "http" and not loopback:
        raise ValueError("Use HTTPS for a non-loopback endpoint, or access it through a local secure tunnel")
    if not u.path.rstrip("/").endswith("/v1"):
        raise ValueError("This helper expects the compatible API base ending in /v1, not /api/v1 or a chat page")
    if u.path.rstrip("/").endswith("/api/v1"):
        raise ValueError("Use the OpenAI-compatible /v1 base, not LM Studio native /api/v1")
    return loopback


def new_json(path: Path, value: dict[str, Any], *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600 if private else 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def scrub(value: Any, secret: str) -> Any:
    if isinstance(value, dict):
        return {k: ("<redacted>" if k.lower() in {"api_key", "authorization"} else scrub(v, secret)) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v, secret) for v in value]
    return value.replace(secret, "<redacted>") if isinstance(value, str) and secret else value


def configure(args: argparse.Namespace) -> int:
    from astrograph.agents.mission.models import AgentSettings
    from astrograph.llm import OpenAICompatibleConfig

    loopback = validate_endpoint(args.base_url)
    if not args.model.strip() or "REPLACE" in args.model.upper():
        raise ValueError("Provide the real server model ID, not a template placeholder")
    if args.api_key_env:
        key = os.environ.get(args.api_key_env, "")
        if not key or "${" in key or "\n" in key or "\r" in key:
            raise ValueError("The named API key environment variable is missing or invalid")
    elif loopback:
        key = "local-placeholder"
    else:
        raise ValueError("For a remote service, provide --api-key-env with an existing environment variable")

    out = Path(args.output_dir).expanduser().resolve()
    llm_file, host_file = out / "llm.local.json", out / "mission.llm.local.json"
    if llm_file.exists() or host_file.exists():
        raise FileExistsError("Local configs already exist; edit them deliberately or choose a new --output-dir")
    llm = {
        "provider": "user-openai-compatible", "base_url": args.base_url.rstrip("/"),
        "model": args.model, "api_key": key, "temperature": 0.0,
        "max_tokens": args.max_tokens, "timeout_s": args.timeout,
        "context_length": args.context_length, "max_response_bytes": 4_000_000, "max_http_attempts": 2,
        "response_format_strategy": args.strategy, "repair_attempts": 0,
        "seed": None, "tokenizer_mode": "required" if getattr(args, "tokenizer_path", None) else "disabled",
        "tokenizer_path": str(Path(args.tokenizer_path).expanduser().resolve()) if getattr(args, "tokenizer_path", None) else None,
        "tokenizer_local_files_only": True, "tokenizer_trust_remote_code": False,
    }
    if llm["tokenizer_path"] and not Path(llm["tokenizer_path"]).is_dir():
        raise ValueError("Tokenizer path must be an existing local model/tokenizer directory")
    OpenAICompatibleConfig.model_validate(llm)
    host = read_json(ROOT / "configs/agent/mission.default.json")
    # Retain baseline models, grants and budgets. Enabling LLM does not authorize new tools.
    host.update(planner="llm_assisted", llm_config=str(llm_file), project_root=str(ROOT))
    host["read_roots"] = [str(ROOT)] if out.is_relative_to(ROOT) else [str(ROOT), str(out)]
    AgentSettings.model_validate(host)
    new_json(llm_file, llm, private=True)
    try:
        new_json(host_file, host, private=True)
    except BaseException:
        llm_file.unlink(missing_ok=True)
        raise
    print(json.dumps({"status": "CONFIGURED_NOT_LIVE_VERIFIED", "settings": str(host_file),
                      "llm_config": str(llm_file), "key_source": args.api_key_env or "LOCAL_NO_AUTH_PLACEHOLDER",
                      "warning": "Local LLM config contains the resolved credential; do not share it. No model was downloaded or started."}, ensure_ascii=False, indent=2))
    return 0


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def probe(settings_path: Path) -> dict[str, Any]:
    from astrograph.agents.mission.models import AgentSettings, AgentRequest
    from astrograph.agents.mission.planner import Planner
    from astrograph.agents.mission.service import MissionAgent

    settings = AgentSettings.model_validate(read_json(settings_path))
    if settings.planner != "llm_assisted" or not settings.llm_config:
        raise ValueError("Enable planner=llm_assisted and provide llm_config first")
    # Match the real CLI's configuration/path behavior and backend. No stand-in backend.
    agent = MissionAgent(settings)
    backend = agent.backend
    if backend is None:
        raise ValueError("No LLM backend was constructed")
    cfg = backend.config
    validate_endpoint(cfg.base_url)
    secret = cfg.api_key.get_secret_value()
    if "${" in secret or (secret.startswith("$") and len(secret) > 1):
        raise ValueError("The core JSON loader does not expand environment variables; regenerate the local config")
    report: dict[str, Any] = {
        "schema_version": "astrograph-llm-service-preflight.v1", "created_at": utc_now(),
        "status": "FAILED", "scope": "ENDPOINT_AND_PLANNING_SCHEMA_ONLY_NOT_DOMAIN_QUALITY",
        "base_url": cfg.base_url, "model": cfg.model,
        "llm_config_sha256": hashlib.sha256(Path(settings.llm_config).read_bytes()).hexdigest(),
        "sends_telemetry": False, "sends_documents": False,
        "weights_downloaded": False, "model_service_started": False,
        "diagnostic_accuracy_evaluated": False,
    }
    req = urllib.request.Request(cfg.base_url.rstrip("/") + "/models",
                                 headers={"Authorization": "Bearer " + secret})
    try:
        opener = urllib.request.build_opener(NoRedirect())
        with opener.open(req, timeout=min(cfg.timeout_s, 15.0)) as response:
            raw = response.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError("Model catalog exceeds preflight limit")
        catalog = json.loads(raw)
        ids = [x["id"] for x in catalog.get("data", []) if isinstance(x, dict) and isinstance(x.get("id"), str)]
        report["model_catalog"] = {"status": "AVAILABLE", "ids": ids[:100], "configured_model_listed": cfg.model in ids}
    except Exception as exc:
        # Some compatible services do not expose /models. The POST is the definitive probe.
        report["model_catalog"] = {"status": "UNAVAILABLE", "reason": str(exc)[:600]}

    planner = Planner(backend, max_calls=1, mode="llm_assisted", work_mode="thinking")
    task = AgentRequest(task_id="llm-preflight", task="research", work_mode="thinking",
                        goal="Connectivity and schema check only. Return task=research, actions=[] and a short rationale. No tools are offered. Do not diagnose anything.")
    plan = planner.prepare(task, [], max_actions=0)
    accepted = any(r.get("phase") == "TASK_PLAN" and r.get("outcome") == "ACCEPTED" for r in planner.records)
    accepted = accepted and plan.task == "research" and not plan.actions and len(backend.call_history) > 0
    report.update(status="PASSED" if accepted else "FAILED", llm_calls=planner.calls,
                  records=planner.records, planning_output=plan.model_dump(mode="json"),
                  transport_history=backend.call_history,
                  next_step="Run stage 39-llm-planning; this preflight does not replace mode or domain acceptance.")
    return scrub(report, secret)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("configure", help="Generate private configs, without downloading or contacting a service")
    c.add_argument("--model", required=True)
    c.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    c.add_argument("--api-key-env", help="Resolve this environment variable NOW; raw core JSON does not expand it")
    c.add_argument("--output-dir", default=str(ROOT / "configs/agent"))
    c.add_argument("--context-length", type=int, default=16384, help="Actual loaded-model context length; verify server setting")
    c.add_argument("--max-tokens", type=int, default=4096)
    c.add_argument("--tokenizer-path", help="Local tokenizer from the SAME served model revision and chat template; enables required exact local counting")
    c.add_argument("--timeout", type=float, default=120)
    c.add_argument("--strategy", choices=["json_schema", "json_object", "plain_json"], default="json_schema")
    q = sub.add_parser("check", help="Actually call the configured endpoint with non-sensitive schema-test content")
    q.add_argument("--settings", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        if args.command == "configure":
            return configure(args)
        if args.output.exists():
            raise FileExistsError("Preflight receipt exists; choose a new --output to preserve prior results")
        report = probe(args.settings.resolve())
        new_json(args.output.resolve(), report)
        print(json.dumps({"status": report["status"], "receipt": str(args.output.resolve()),
                          "scope": report["scope"]}, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "PASSED" else 2
    except Exception as exc:
        # Avoid echoing arbitrary provider bodies or Pydantic input values in setup errors.
        print(json.dumps({"status": "BLOCKED", "error_type": type(exc).__name__,
                          "message": "Configuration/preflight could not complete. Check paths, endpoint, credentials and existing files against LLM_SERVICE_SETUP.md."}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
