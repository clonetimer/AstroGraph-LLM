"""Create a labeled, known-truth Chinese engineering OCR test page."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
FONT = r"C:\Windows\Fonts\msyh.ttc"
BOLD = r"C:\Windows\Fonts\msyhbd.ttc"
TRUTH = {
    "title": "合成样例：反作用轮运行记录",
    "notice": "仅用于 OCR 对比，非真实卫星遥测",
    "columns": ["阶段", "时刻", "轮速 rpm", "电流 A", "摩擦力矩 N·m", "温度 °C"],
    "rows": [
        ["基线", "10:00:00", "600", "0.20", "0.003", "31.0"],
        ["负载注入", "10:01:00", "600", "0.34", "0.009", "38.6"],
        ["负载解除", "10:02:00", "600", "0.21", "0.003", "31.5"],
    ],
    "note": "核查提示：三阶段控制模式均为恒速；电流与力矩变化仍需排除指令变化。",
    "equipment": "设备编号：RW-A；数据类型：合成测试表",
}


def main() -> None:
    image = Image.new("RGB", (1600, 1100), "white")
    draw = ImageDraw.Draw(image)
    title = ImageFont.truetype(BOLD, 49)
    body = ImageFont.truetype(FONT, 30)
    head = ImageFont.truetype(BOLD, 27)
    small = ImageFont.truetype(FONT, 27)
    draw.text((90, 65), TRUTH["title"], font=title, fill="#14212a")
    draw.text((92, 145), TRUTH["notice"], font=small, fill="#8a3a28")
    draw.text((92, 205), TRUTH["equipment"], font=body, fill="#253a44")
    lefts = [95, 310, 535, 765, 950, 1270]
    y_lines = [285, 363, 465, 567, 669]
    for y in y_lines:
        draw.line((85, y, 1515, y), fill="#6b848e", width=3)
    for x in [85, 300, 525, 755, 940, 1260, 1515]:
        draw.line((x, 285, x, 669), fill="#9aabb1", width=2)
    for x, text in zip(lefts, TRUTH["columns"]):
        draw.text((x, 307), text, font=head, fill="#14212a")
    for row, y in zip(TRUTH["rows"], [392, 494, 596]):
        for x, text in zip(lefts, row):
            draw.text((x, y), text, font=body, fill="#14212a")
    draw.text((92, 735), TRUTH["note"], font=body, fill="#14212a")
    draw.text((92, 830), "结论边界：以上数值不代表故障已确诊。", font=body, fill="#8a3a28")
    image.save(HERE / "synthetic_rw_table.png")
    (HERE / "synthetic_rw_table_truth.json").write_text(
        json.dumps(TRUTH, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
