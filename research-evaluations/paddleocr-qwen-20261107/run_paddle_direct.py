"""Probe locally paired PaddleOCR-VL GGUF/mmproj on saved public pages."""

from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path

import requests


HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent / "mineru-local-20261107"
MODEL = r"E:\lmstudio\PaddlePaddle\PaddleOCR-VL-1.6-GGUF\PaddleOCR-VL-1.6-GGUF.gguf"
URL = "http://127.0.0.1:1235/v1/chat/completions"
PROMPTS = {
    "generic": (
        "Transcribe every visible word on this document page, including all labels "
        "and bullet points inside the figure. Preserve reading order as far as possible. "
        "Do not infer words that are not visible. Output only the transcription."
    ),
    "native_ocr": "ocr",
}
CASES = [("mco_page_11", "generic"), ("mco_page_07", "generic"),
         ("mco_page_11", "native_ocr")]


def main() -> None:
    session = requests.Session()
    session.trust_env = False
    summary = {"endpoint": URL, "model": MODEL, "results": {}}
    for stem, prompt_name in CASES:
        image = (INPUTS / f"{stem}.png").read_bytes()
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPTS[prompt_name]},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                  base64.b64encode(image).decode("ascii")}},
            ]}],
            "temperature": 0,
            "max_tokens": 2600,
            "stream": False,
        }
        key = stem + "_" + prompt_name
        started = time.monotonic()
        try:
            response = session.post(URL, json=payload, timeout=240)
            record = {"http_status": response.status_code,
                      "elapsed_seconds": round(time.monotonic() - started, 3)}
            try:
                record["response"] = response.json()
            except ValueError:
                record["response_text"] = response.text[:3000]
        except requests.RequestException as exc:
            record = {"error": type(exc).__name__, "message": str(exc)[:500],
                      "elapsed_seconds": round(time.monotonic() - started, 3)}
        (HERE / f"{key}_raw.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        view = {"source_image": f"{stem}.png", "image_sha256": hashlib.sha256(image).hexdigest(),
                "prompt": PROMPTS[prompt_name],
                **{k: v for k, v in record.items() if k != "response"}}
        choices = (record.get("response") or {}).get("choices") or []
        if choices:
            content = (choices[0].get("message") or {}).get("content") or ""
            if isinstance(content, str):
                (HERE / f"{key}_content.txt").write_text(content, encoding="utf-8")
                view["content_chars"] = len(content)
            view["finish_reason"] = choices[0].get("finish_reason")
        summary["results"][key] = view
        (HERE / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
