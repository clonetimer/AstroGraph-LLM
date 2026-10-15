"""Connectivity probe for administrator-configured OpenAI-compatible services.

The probe is deliberately non-diagnostic. It checks the model catalogue, exercises
AstroGraph's bounded HTTP worker, then validates one *tiny* structured response.
For LM Studio auto mode the compatibility path prefers ``json_object`` because local
reasoning models may spend an entire request budget inside strict grammar-constrained
``json_schema`` decoding. The host still validates the returned object with Pydantic.
Secrets are never returned.
"""
from __future__ import annotations

import json
import time
import urllib.request
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from astrograph.integrations.bounded_http import request_json
from astrograph.llm.openai_compatible import OpenAICompatibleStructuredBackend


_MAX_MODELS_BYTES = 1_000_000


def _models_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return base + "/models"


def _chat_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


def _catalogue(base_url: str, api_key: str, timeout_s: float) -> dict[str, Any]:
    started = time.monotonic()
    request = urllib.request.Request(
        _models_url(base_url),
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=min(timeout_s, 20.0)) as response:
            length = response.headers.get("Content-Length")
            if length and int(length) > _MAX_MODELS_BYTES:
                raise ValueError("MODEL_CATALOGUE_RESPONSE_TOO_LARGE")
            raw = response.read(_MAX_MODELS_BYTES + 1)
            status_code = int(getattr(response, "status", 200))
        if len(raw) > _MAX_MODELS_BYTES:
            raise ValueError("MODEL_CATALOGUE_RESPONSE_TOO_LARGE")
        document = json.loads(raw.decode("utf-8"))
        rows = document.get("data") if isinstance(document, dict) else None
        model_ids = [str(item.get("id")) for item in (rows or []) if isinstance(item, dict) and item.get("id")]
        return {
            "status": "SUCCEEDED",
            "http_status": status_code,
            "model_ids": model_ids[:100],
            "elapsed_ms": round((time.monotonic() - started) * 1000.0, 2),
        }
    except Exception as exc:
        return {
            "status": "FAILED",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "model_ids": [],
            "elapsed_ms": round((time.monotonic() - started) * 1000.0, 2),
        }


def _chat(base_url: str, model: str, api_key: str, timeout_s: float) -> dict[str, Any]:
    started = time.monotonic()
    try:
        document = request_json(
            _chat_url(base_url),
            payload={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a connectivity probe."},
                    {"role": "user", "content": "Reply with exactly: OK"},
                ],
                "temperature": 0,
                "max_tokens": 8,
                "stream": False,
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout_s=min(timeout_s, 30.0),
            max_response_bytes=1_000_000,
        )
        content, _ = OpenAICompatibleStructuredBackend._message_content(document)
        if content is None:
            raise ValueError("OPENAI_COMPATIBLE_RESPONSE_MISSING_CHOICES")
        return {
            "status": "SUCCEEDED",
            "response_shape_valid": True,
            "elapsed_ms": round((time.monotonic() - started) * 1000.0, 2),
        }
    except Exception as exc:
        return {
            "status": "FAILED",
            "response_shape_valid": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "elapsed_ms": round((time.monotonic() - started) * 1000.0, 2),
        }


class _StructuredProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["OK"]


def _strategy_order(provider: str, configured: str) -> list[str]:
    if configured != "auto":
        return [configured]
    if provider.strip().casefold() in {"lmstudio", "lm-studio"}:
        return ["json_object", "plain_json", "json_schema"]
    return ["json_schema", "json_object", "plain_json"]


def _structured_body(model: str, strategy: str) -> dict[str, Any]:
    schema = _StructuredProbeResult.model_json_schema()
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return one JSON object immediately. Do not provide analysis or hidden reasoning. "
                    "The object must be exactly compatible with the requested schema. /no_think"
                ),
            },
            {"role": "user", "content": 'Return {"status":"OK"} now.'},
        ],
        "temperature": 0,
        "max_tokens": 48,
        "stream": False,
    }
    if strategy == "json_schema":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "AstroGraphProbe", "strict": True, "schema": schema},
        }
    elif strategy == "json_object":
        body["response_format"] = {"type": "json_object"}
    return body


def _structured(
    base_url: str,
    model: str,
    api_key: str,
    timeout_s: float,
    provider: str,
    response_format_strategy: str,
) -> dict[str, Any]:
    """Validate at least one host-validated structured-output path.

    A timeout in one constrained-decoding strategy is a compatibility failure for
    that strategy, not proof that the model service is unusable. Auto mode therefore
    tries the next bounded strategy instead of spending the whole service deadline on
    the first grammar.
    """
    started = time.monotonic()
    attempts: list[dict[str, Any]] = []
    for strategy in _strategy_order(provider, response_format_strategy):
        attempt_started = time.monotonic()
        try:
            document = request_json(
                _chat_url(base_url),
                payload=_structured_body(model, strategy),
                headers={"Authorization": f"Bearer {api_key}"},
                timeout_s=min(timeout_s, 25.0),
                max_response_bytes=1_000_000,
            )
            content, finish_reason = OpenAICompatibleStructuredBackend._message_content(document)
            parsed = OpenAICompatibleStructuredBackend._json_from_text(content)
            result = _StructuredProbeResult.model_validate(parsed)
            if result.status != "OK":
                raise ValueError("STRUCTURED_PROBE_UNEXPECTED_VALUE")
            attempts.append({
                "strategy": strategy,
                "status": "SUCCEEDED",
                "finish_reason": finish_reason,
                "elapsed_ms": round((time.monotonic() - attempt_started) * 1000.0, 2),
            })
            return {
                "status": "SUCCEEDED",
                "schema_valid": True,
                "effective_strategy": strategy,
                "configured_strategy": response_format_strategy,
                "attempts": attempts,
                "elapsed_ms": round((time.monotonic() - started) * 1000.0, 2),
            }
        except Exception as exc:
            attempts.append({
                "strategy": strategy,
                "status": "FAILED",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "elapsed_ms": round((time.monotonic() - attempt_started) * 1000.0, 2),
            })
            if response_format_strategy != "auto":
                break
    last = attempts[-1]["error"] if attempts else {"type": "RuntimeError", "message": "NO_STRUCTURED_STRATEGY"}
    return {
        "status": "FAILED",
        "schema_valid": False,
        "configured_strategy": response_format_strategy,
        "effective_strategy": None,
        "attempts": attempts,
        "error": last,
        "elapsed_ms": round((time.monotonic() - started) * 1000.0, 2),
    }


def probe_openai_compatible_service(
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout_s: float,
    provider: str,
    response_format_strategy: str = "auto",
) -> dict[str, Any]:
    catalogue = _catalogue(base_url, api_key, timeout_s)
    chat = _chat(base_url, model, api_key, timeout_s)
    structured = _structured(
        base_url, model, api_key, timeout_s, provider, response_format_strategy
    )
    model_ids = catalogue.get("model_ids") or []
    ready = all(item.get("status") == "SUCCEEDED" for item in (catalogue, chat, structured))
    qwen_reasoning_hint = None
    if provider.strip().casefold() in {"lmstudio", "lm-studio"} and "qwen3.5" in model.casefold():
        qwen_reasoning_hint = (
            "Qwen3.5-9B in LM Studio is a reasoning model and its model card defaults Enable Thinking to true. "
            "For short schema-bound platform tasks, disable Enable Thinking in LM Studio if latency remains high."
        )
    return {
        "schema_version": "astrograph-platform-llm-probe.v3",
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "catalogue": catalogue,
        "chat_runtime": chat,
        "structured_runtime": structured,
        "model_listed": model in model_ids if model_ids else None,
        "status": "READY" if ready else "FAILED",
        "operator_hint": qwen_reasoning_hint,
        "secret_returned": False,
        "diagnostic_vote_allowed": False,
        "applies_to": "POST_DIAGNOSIS_STRUCTURED_EXPLANATION_AND_AGENT_INVESTIGATION",
    }


__all__ = ["probe_openai_compatible_service"]
