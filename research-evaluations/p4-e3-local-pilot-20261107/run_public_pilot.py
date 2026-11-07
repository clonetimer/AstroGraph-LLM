"""Run two public, cutoff-bound cases against the user-provided local LM Studio.

This is a model-interface pilot, not an end-to-end AstroGraph diagnosis test.
Only cases/*/input.md is sent. Reviewer-only notes and source HTML are excluded.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CASES = ROOT / "p4-e1-public-candidates-20261107" / "cases"
URL = "http://127.0.0.1:1234/api/v1/chat"
MODELS = ("qwen/qwen3.5-9b", "qwen3.5-0.8b")
CASE_IDS = ("NH2015", "GAL2017")


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        for case_id in CASE_IDS:
            source = CASES / case_id / "input.md"
            input_bytes = source.read_bytes()
            request = {
                "model": model,
                "input": input_bytes.decode("utf-8"),
                "reasoning": "off",
                "temperature": 0,
                "max_output_tokens": 650,
                "store": False,
            }
            slug = model.replace("/", "_")
            target = HERE / f"{case_id}_{slug}.json"
            if target.exists():
                print(f"SKIP {case_id} {model}: output exists", flush=True)
                continue
            started = time.monotonic()
            record = {
                "case_id": case_id,
                "model_requested": model,
                "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
                "request_settings": {k: v for k, v in request.items() if k != "input"},
                "endpoint_kind": "LM Studio native /api/v1/chat",
                "started_utc": datetime.now(timezone.utc).isoformat(),
            }
            try:
                payload = json.dumps(request, ensure_ascii=False).encode("utf-8")
                http_request = urllib.request.Request(
                    URL, data=payload, headers={"Content-Type": "application/json; charset=utf-8"}, method="POST"
                )
                with urllib.request.urlopen(http_request, timeout=240) as response:
                    raw = response.read()
                    record["http_status"] = response.status
                result = json.loads(raw)
                record["output"] = result.get("output")
                record["stats"] = result.get("stats")
                record["error"] = result.get("error")
                record["response_sha256"] = hashlib.sha256(raw).hexdigest()
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["elapsed_s"] = round(time.monotonic() - started, 3)
            target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            content = " ".join(str(x.get("content") or "") for x in record.get("output") or [] if isinstance(x, dict))
            print(f"{case_id} {model}: status={record.get('http_status')} chars={len(content)} elapsed={record['elapsed_s']}s error={record.get('error')}", flush=True)


if __name__ == "__main__":
    main()
