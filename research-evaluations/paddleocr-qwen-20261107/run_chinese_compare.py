"""Compare direct vision transcription on a known-truth synthetic Chinese table."""

from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path

import requests


HERE = Path(__file__).resolve().parent
IMAGE = HERE / "synthetic_rw_table.png"
PROMPT = (
    "请逐字识别这张文档图片中的全部可见文字。表格请保持列标题与每一行的对应关系，"
    "保留数值、单位和时间。不要推断图中没有的内容。只输出识别结果。"
)
MODELS = {
    "qwen3_vl_4b": (
        "http://127.0.0.1:1234/v1/chat/completions",
        "qwen/qwen3-vl-4b",
    ),
    "paddleocr_vl_1_6": (
        "http://127.0.0.1:1235/v1/chat/completions",
        r"E:\lmstudio\PaddlePaddle\PaddleOCR-VL-1.6-GGUF\PaddleOCR-VL-1.6-GGUF.gguf",
    ),
}


def main() -> None:
    image = IMAGE.read_bytes()
    url_data = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
    session = requests.Session()
    session.trust_env = False
    summary = {
        "image": IMAGE.name,
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "prompt": PROMPT,
        "temperature": 0,
        "max_tokens": 2600,
        "results": {},
    }
    for key, (url, model) in MODELS.items():
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": url_data}},
            ]}],
            "temperature": 0,
            "max_tokens": 2600,
            "stream": False,
        }
        started = time.monotonic()
        try:
            response = session.post(url, json=payload, timeout=240)
            record = {
                "http_status": response.status_code,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            try:
                record["response"] = response.json()
            except ValueError:
                record["response_text"] = response.text[:3000]
        except requests.RequestException as exc:
            record = {
                "error": type(exc).__name__,
                "message": str(exc)[:500],
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        (HERE / f"synthetic_{key}_raw.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        result = {"endpoint": url, "model": model,
                  **{k: v for k, v in record.items() if k != "response"}}
        choices = (record.get("response") or {}).get("choices") or []
        if choices:
            content = (choices[0].get("message") or {}).get("content") or ""
            if isinstance(content, str):
                (HERE / f"synthetic_{key}_content.txt").write_text(
                    content, encoding="utf-8"
                )
                result["content_chars"] = len(content)
            result["finish_reason"] = choices[0].get("finish_reason")
        summary["results"][key] = result
        (HERE / "synthetic_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(key, json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
