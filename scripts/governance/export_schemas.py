#!/usr/bin/env python3
"""Export deterministic JSON Schemas for governance artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from astrograph.governance import (  # noqa: E402
    ChangeRequest,
    FailureMemoryEntry,
    GateEvidence,
    GateQueryReceipt,
    ImprovementContract,
    MaterialPassport,
    ReleaseDecision,
    RollbackManifest,
    TaskBoard,
)

MODELS = {
    "improvement_contract.schema.json": ImprovementContract,
    "change_request.schema.json": ChangeRequest,
    "task_board.schema.json": TaskBoard,
    "gate_evidence.schema.json": GateEvidence,
    "gate_query_receipt.schema.json": GateQueryReceipt,
    "release_decision.schema.json": ReleaseDecision,
    "material_passport.schema.json": MaterialPassport,
    "failure_memory.schema.json": FailureMemoryEntry,
    "rollback_manifest.schema.json": RollbackManifest,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "governance" / "schemas")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, model in MODELS.items():
        path = args.output_dir / filename
        path.write_text(json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "schema_count": len(MODELS), "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
