"""Exercise AstroGraph's default PDF parsing on a text PDF and image-only PDF."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from astrograph.retrieval.structured_document import SectionFirstDocumentIngestor  # noqa: E402


def main() -> None:
    public_pdf = Path(
        r"D:\Project\AstoGraph-LLM\knowledge-sources\batch2-20261024"
        r"\public\originals\PUB2-MCO-PHASE1-1999.pdf"
    )
    image_pdf = HERE / "synthetic_rw_table_image_only.pdf"
    Image.open(HERE / "synthetic_rw_table.png").convert("RGB").save(image_pdf)
    ingestor = SectionFirstDocumentIngestor()
    results = {}
    for name, path in (("public_pdf", public_pdf), ("image_only_pdf", image_pdf)):
        try:
            elements, warnings, audit = ingestor.parse_pdf_layout(path)
            results[name] = {
                "status": "parsed",
                "parser_id": audit.parser_id,
                "page_count": audit.page_count,
                "text_page_count": audit.text_page_count,
                "element_count": len(elements),
                "warnings": warnings[:30],
            }
        except Exception as exc:
            results[name] = {
                "status": "error",
                "exception": type(exc).__name__,
                "message": str(exc),
            }
    (HERE / "default_parser_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
