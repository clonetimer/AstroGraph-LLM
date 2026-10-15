from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astrograph.integrations.platform_knowledge import PlatformKnowledgeEnrichmentService


def main() -> int:
    service = PlatformKnowledgeEnrichmentService.from_environment()
    payload = service.capabilities()
    paperseek_probe = None
    probe = getattr(service.paperseek_client, "probe", None)
    if callable(probe):
        try:
            paperseek_probe = probe()
        except Exception as exc:
            paperseek_probe = {
                "status": "PROBE_FAILED",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
    a2o_probe = None
    if service.config.a2o_enabled:
        probe = getattr(service.a2o_adapter, "probe", None)
        if callable(probe):
            try:
                a2o_probe = probe()
            except Exception as exc:
                a2o_probe = {
                    "status": "PROBE_FAILED",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
    payload["paperseek_probe"] = paperseek_probe
    payload["anything2ontology_probe"] = a2o_probe
    payload["llm_api_key_present"] = bool(os.environ.get("ASTROGRAPH_LLM_API_KEY"))
    payload["claim_boundary"] = (
        "This command checks configuration/import readiness only. It does not prove live PaperSeek retrieval "
        "quality, Anything2Ontology extraction quality, or model diagnosis quality."
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
