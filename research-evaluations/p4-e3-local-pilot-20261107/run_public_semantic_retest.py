"""Development-only public case retest; not an AstroGraph end-to-end diagnosis."""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
CASES = HERE.parent / "p4-e1-public-candidates-20261107" / "cases"
OUTPUT = HERE / "public_semantic_retest"
ENDPOINT = "http://127.0.0.1:1234/api/v1/chat"
GUIDANCE = (
    "\n\n报告编写步骤：先在“来源量表”中逐项列出材料里的时间量及其含义，"
    "或逐项列出设备类型、卫星构型与失效台数的对应关系；每项标注来源编号。"
    "然后写故障现象、至少两种仍待区分的原因、支持或削弱原因的待核查记录。"
    "不要把通信延迟写成故障持续时间，也不要把一种设备或构型的数字套给另一种。"
    "无法从材料确认的时间、设备状态和因果关系写‘未知’。"
    "全文控制在约 700 个汉字，保留不确定性，不给未经授权的设备操作指令。"
)


def main() -> None:
    models = tuple(sys.argv[1:]) or ("qwen3.5-0.8b",)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for model in models:
        for case_id in ("NH2015", "GAL2017"):
            source = CASES / case_id / "input.md"
            source_bytes = source.read_bytes()
            request = {
                "model": model,
                "input": source_bytes.decode("utf-8") + GUIDANCE,
                "reasoning": "off",
                "temperature": 0,
                "max_output_tokens": 850,
                "store": False,
            }
            target = OUTPUT / f"{case_id}_{model.replace('/', '_')}.json"
            if target.exists():
                print(f"SKIP {target.name}", flush=True)
                continue
            record = {
                "case_id": case_id,
                "model_requested": model,
                "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "guidance_sha256": hashlib.sha256(GUIDANCE.encode("utf-8")).hexdigest(),
                "request_sha256": hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
                "settings": {key: value for key, value in request.items() if key != "input"},
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "endpoint_kind": "LM Studio native /api/v1/chat",
            }
            started = time.monotonic()
            try:
                raw = json.dumps(request, ensure_ascii=False).encode("utf-8")
                http_request = urllib.request.Request(
                    ENDPOINT, data=raw,
                    headers={"Content-Type": "application/json; charset=utf-8"}, method="POST",
                )
                with urllib.request.urlopen(http_request, timeout=300) as response:
                    record["http_status"] = response.status
                    response_bytes = response.read()
                response = json.loads(response_bytes)
                record["output"] = response.get("output")
                record["stats"] = response.get("stats")
                record["error"] = response.get("error")
                record["response_sha256"] = hashlib.sha256(response_bytes).hexdigest()
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["elapsed_s"] = round(time.monotonic() - started, 3)
            target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(case_id, model, record.get("http_status"), record["elapsed_s"], record.get("error"), flush=True)


if __name__ == "__main__":
    main()
