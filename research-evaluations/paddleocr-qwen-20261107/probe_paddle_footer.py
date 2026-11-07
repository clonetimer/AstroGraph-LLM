"""Check whether PaddleOCR-VL can recover text omitted below a detected table."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import requests
from PIL import Image


HERE = Path(__file__).resolve().parent
MODEL = r"E:\lmstudio\PaddlePaddle\PaddleOCR-VL-1.6-GGUF\PaddleOCR-VL-1.6-GGUF.gguf"
image = Image.open(HERE / "synthetic_rw_table.png").crop((0, 680, 1600, 1100))
image.save(HERE / "synthetic_rw_footer_crop.png")
buffer = io.BytesIO()
image.save(buffer, format="PNG")
url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "请逐字识别图片中的全部可见文字，不要添加解释。"},
        {"type": "image_url", "image_url": {"url": url}},
    ]}],
    "temperature": 0,
    "max_tokens": 800,
    "stream": False,
}
session = requests.Session()
session.trust_env = False
response = session.post("http://127.0.0.1:1235/v1/chat/completions", json=payload, timeout=120)
record = {"http_status": response.status_code, "response": response.json()}
(HERE / "synthetic_paddle_footer_raw.json").write_text(
    json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
content = record["response"].get("choices", [{}])[0].get("message", {}).get("content", "")
(HERE / "synthetic_paddle_footer_content.txt").write_text(content, encoding="utf-8")
print(response.status_code, content)
