"""Bounded HTTP JSON in a killable stdlib-only worker.

Credentials travel through stdin, never argv, logs or on-disk requests. Worker
caps success AND error bodies. Parent enforces an absolute elapsed deadline.
"""
from __future__ import annotations
import json
import base64
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit
from .execution_budget import remaining_time, ExecutionDeadlineExceeded
from .process_control import run_managed_subprocess


def request_json(url: str, *, payload: dict, headers: dict | None = None,
                 timeout_s: float = 120, max_response_bytes: int = 4_000_000,
                 response_observer=None) -> dict:
    if urlsplit(url).scheme not in {"http", "https"}:
        raise ValueError("HTTP_ENDPOINT_REQUIRED")
    if max_response_bytes < 1:
        raise ValueError("positive response byte cap required")
    limit = remaining_time(timeout_s)
    request = {"url":url,"payload":payload,"headers":headers or {},
               "timeout_s":limit,"max_response_bytes":max_response_bytes,
               "capture_response":response_observer is not None}
    # Executing a file keeps the worker stdlib-only and avoids importing torch.
    worker = str(Path(__file__).with_name("http_worker.py"))
    try:
        result = run_managed_subprocess([sys.executable, worker], input=json.dumps(request).encode(),
            capture_output=True, timeout=limit, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ExecutionDeadlineExceeded("HTTP_TOTAL_DEADLINE_EXCEEDED") from exc
    if len(result.stdout or b"") > max_response_bytes * 6 + 4096:
        raise ValueError("HTTP response size limit exceeded (HTTP_RESPONSE_TOO_LARGE)")
    if result.returncode:
        raise RuntimeError("HTTP_WORKER_FAILED")
    envelope = json.loads(result.stdout)
    if response_observer is not None and "raw_body_b64" in envelope:
        response_observer(base64.b64decode(envelope["raw_body_b64"],validate=True),
                          http_status=envelope.get("http_status"))
    if not envelope.get("ok"):
        error = str(envelope.get("error") or "HTTP_REQUEST_FAILED")
        # The killable worker intentionally serializes exception classes rather than
        # raw remote bodies. Preserve caller-visible parse/size contract semantics
        # instead of collapsing malformed JSON into an unrelated RuntimeError.
        if error in {"JSONDecodeError", "UnicodeDecodeError"}:
            raise ValueError("HTTP_RESPONSE_INVALID_JSON")
        if error == "HTTP_RESPONSE_TOO_LARGE":
            raise ValueError("HTTP response size limit exceeded (HTTP_RESPONSE_TOO_LARGE)")
        raise RuntimeError(error)
    document = envelope["document"]
    if not isinstance(document,dict):
        raise ValueError("HTTP_RESPONSE_NOT_OBJECT")
    return document
